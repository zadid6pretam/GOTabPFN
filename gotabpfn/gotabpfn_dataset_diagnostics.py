# ============================================================
# gotabpfn_dataset_diagnostics.py
#
# Standalone dataset-level diagnostics for numeric CSV files:
#   IDF / FOE / P_success + locality gains + LES + AUC
#
# Requires:
#   uses GraphFeatureOrdering from gotabpfn.py
#
# Single CSV:
#   python gotabpfn_dataset_diagnostics.py \
#       --csv coloncancer_encoded.csv \
#       --target label \
#       --dataset Colon \
#       --out colon_ordering_metrics.csv
#
# Feature-only CSV:
#   python gotabpfn_dataset_diagnostics.py \
#       --csv my_features.csv \
#       --dataset MyDataset \
#       --out my_dataset_ordering_metrics.csv
#
# Multiple datasets:
#   python gotabpfn_dataset_diagnostics.py \
#       --dataset-list datasets.csv \
#       --out all_ordering_metrics.csv
#
# datasets.csv format:
#   csv_path,target_col,dataset_name
#   coloncancer_encoded.csv,label,Colon
#   arcene_combined_encoded.csv,Label,Arcene
#
# Output columns:
#   FOE_rank, dataset, category, n, m, rho, IDF_final, FOE,
#   P_success, Delta_AdjCoh, Delta_HitRate, Delta_Cut, LES, AUC
# ============================================================

from __future__ import annotations

import argparse
import math
import os
import random
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from gotabpfn import GraphFeatureOrdering


# ============================================================
# Config
# ============================================================

GLOBAL_SEED = 42

VAR_THRESHOLDS = [0.90, 0.95, 0.99, 0.9975]

SENSITIVITY_EXPONENT = 2

IDF_MAX_ROWS = 20000

MAX_SUB_N_DELTAS = 20

HIT_K = 10
HIT_H = 16
HIT_FEATURE_SUBSAMPLE = 2048
HIT_ROW_SUBSAMPLE = 5000

M_FIXED = 32
LMIN_FIXED = 8
CUT_SEG_RULE = "equal_mass"  # "uniform" | "largest_jump" | "equal_mass"

RANDOM_SEEDS = (0, 1, 2, 3, 4)

GO_NUM_CLUSTERS = 7
GO_METRIC = "correlation"
GO_REFINE = True
GO_REFINE_PASSES = 1
GO_DIRECTION_SELECT = True
GO_DETERMINISTIC = True
GO_USE_CPU_KMEANS = True

OUTPUT_COLS = [
    "FOE_rank",
    "dataset",
    "category",
    "n",
    "m",
    "rho",
    "IDF_final",
    "FOE",
    "P_success",
    "Delta_AdjCoh",
    "Delta_HitRate",
    "Delta_Cut",
    "LES",
    "AUC",
]


# ============================================================
# Dataset regime rule
# ============================================================

def categorize_dataset_regime(n: int, m: int) -> tuple[str, float]:
    """
    Empirical dataset-regime categorization rule.

    rho = m / n

    HDLSS:
        m > 1000, n < 1000, rho > 2

    HDHSS:
        m > 1000, n > 10000, 0.05 < rho <= 2

    LDHSS:
        m <= 100, n > 10000, rho <= 0.01

    LDLSS:
        m <= 100, n <= 1000, rho <= 0.05

    MixedRegime:
        otherwise
    """
    n = int(n)
    m = int(m)

    if n <= 0:
        return "MixedRegime", float("nan")

    rho = float(m) / float(n)

    if (m > 1000) and (n < 1000) and (rho > 2.0):
        category = "HDLSS"

    elif (m > 1000) and (n > 10000) and (0.05 < rho <= 2.0):
        category = "HDHSS"

    elif (m <= 100) and (n > 10000) and (rho <= 0.01):
        category = "LDHSS"

    elif (m <= 100) and (n <= 1000) and (rho <= 0.05):
        category = "LDLSS"

    else:
        category = "MixedRegime"

    return category, rho


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ============================================================
# Data loading
# ============================================================

def _none_if_empty(x):
    if x is None:
        return None

    if isinstance(x, float) and np.isnan(x):
        return None

    x = str(x).strip()

    if x == "" or x.lower() in {"none", "null", "nan"}:
        return None

    return x


def load_numeric_csv(
    csv_path: str,
    target_col: Optional[str] = None,
    standardize: bool = True,
) -> np.ndarray:
    """
    Loads a numeric CSV. If target_col is given, drops it before computing metrics.

    Requirements:
      - All remaining feature columns must be numeric.
      - Missing/inf values are replaced with 0 before optional standardization.
    """
    df = pd.read_csv(csv_path)

    target_col = _none_if_empty(target_col)

    if target_col is not None:
        if target_col not in df.columns:
            raise ValueError(
                f"Target column '{target_col}' not found in {csv_path}.\n"
                f"Available columns: {list(df.columns)}"
            )
        df = df.drop(columns=[target_col])

    non_numeric = [
        c for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c])
    ]

    if non_numeric:
        raise ValueError(
            "This script expects numeric feature columns only after dropping target.\n"
            f"Non-numeric feature columns found: {non_numeric}"
        )

    X = df.to_numpy(dtype=np.float32)

    if X.ndim != 2 or X.shape[1] < 2:
        raise ValueError(
            f"Expected at least two numeric feature columns. Got shape={X.shape}"
        )

    X = np.nan_to_num(
        X,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    if standardize:
        X = StandardScaler().fit_transform(X).astype(np.float32)

    return X


# ============================================================
# PCA-IDF / CumVar / AUC / FOE
# ============================================================

def subsample_rows(
    X: np.ndarray,
    max_rows: int,
    seed: int = 42,
) -> np.ndarray:
    n = X.shape[0]

    if n <= max_rows:
        return X

    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=max_rows, replace=False)

    return X[idx]


def pca_eigvals(X: np.ndarray) -> np.ndarray:
    """
    HDLSS-safe PCA eigenvalue computation.

    If n <= m, use Gram matrix X X^T.
    Else, use covariance X^T X.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    n, m = Xc.shape

    if n <= m:
        G = (Xc @ Xc.T) / max(1, n - 1)
        vals = np.linalg.eigvalsh(G)
    else:
        C = (Xc.T @ Xc) / max(1, n - 1)
        vals = np.linalg.eigvalsh(C)

    vals = np.clip(vals, 0.0, None)
    vals = np.sort(vals)[::-1]

    if vals.sum() <= 1e-12:
        vals = np.ones(1, dtype=np.float64)

    return vals


def idf_cumvar_curve(
    X: np.ndarray,
    thresholds: list[float],
    max_rows: int = IDF_MAX_ROWS,
    seed: int = GLOBAL_SEED,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """
    Computes IDF-CumVar points.

    For each variance threshold tau:
      d_hat(tau) = number of PCs needed to explain tau variance
      IDF(tau)   = d_hat(tau) / m
      CumVar     = achieved cumulative explained variance
    """
    _, m = X.shape

    X_idf = subsample_rows(
        X,
        max_rows=max_rows,
        seed=seed + 101,
    )

    vals = pca_eigvals(X_idf)
    cum = np.cumsum(vals) / (vals.sum() + 1e-12)

    idfs = []
    cumvars = []
    d_hats = []

    for tau in thresholds:
        d_hat = int(np.searchsorted(cum, tau) + 1)
        d_hat = max(1, min(d_hat, len(vals)))

        d_hats.append(d_hat)
        cumvars.append(float(cum[d_hat - 1]))
        idfs.append(float(d_hat / m))

    return (
        np.asarray(idfs, dtype=np.float64),
        np.asarray(cumvars, dtype=np.float64),
        int(d_hats[-1]),
        float(cumvars[-1]),
    )


def auc_trapezoid(x: np.ndarray, y: np.ndarray) -> float:
    """
    AUC under the IDF-CumVar curve using trapezoidal integration.

    x = IDF values
    y = cumulative explained variance values
    """
    idx = np.argsort(x)

    xs = np.asarray(x, dtype=np.float64)[idx]
    ys = np.asarray(y, dtype=np.float64)[idx]

    if len(xs) < 2:
        return float(ys[0]) if len(ys) else float("nan")

    return float(np.trapezoid(ys, xs))


# ============================================================
# Locality diagnostics
# ============================================================

def make_random_perm(m: int, seed: int) -> list[int]:
    rng = np.random.RandomState(seed)
    return rng.permutation(m).tolist()


def compute_deltas_for_ordering(
    X: np.ndarray,
    Pi: list[int],
    max_sub_n: int = MAX_SUB_N_DELTAS,
) -> np.ndarray:
    """
    Computes adjacent transition costs under an ordering:

      delta_t = 1 - |corr(x_{pi_t}, x_{pi_{t+1}})|

    Smaller AdjCoh means adjacent features are more correlated/coherent.
    """
    sub_n = min(max_sub_n, X.shape[0])

    X_sub = X[:sub_n].astype(np.float32)
    Xp = X_sub[:, Pi]

    Xp = Xp - Xp.mean(axis=0, keepdims=True)
    Xp = Xp / (Xp.std(axis=0, keepdims=True) + 1e-6)

    corr_adj = (Xp[:, :-1] * Xp[:, 1:]).mean(axis=0)
    deltas = 1.0 - np.abs(corr_adj.astype(np.float32))

    return deltas


def adjacency_coherence(deltas: np.ndarray) -> float:
    if deltas is None or len(deltas) == 0:
        return float("nan")

    return float(np.mean(deltas))


def segment_uniform(m: int, M: int) -> list[tuple[int, int]]:
    step = int(math.ceil(m / M))
    segs = []

    for t in range(M):
        a = t * step
        b = min((t + 1) * step, m)

        if a < b:
            segs.append((a, b))

    return segs


def segment_equal_mass(
    deltas: np.ndarray,
    m: int,
    M: int,
    l_min: int = LMIN_FIXED,
    eps: float = 1e-12,
) -> list[tuple[int, int]]:
    deltas = np.clip(np.asarray(deltas, dtype=np.float32), 0.0, None)

    c = np.zeros(m, dtype=np.float32)
    c[1:] = np.cumsum(deltas)

    total = float(c[-1])

    if total <= eps:
        return segment_uniform(m, M)

    targets = [(k / M) * total for k in range(1, M)]

    cuts = []
    last = 0

    for tgt in targets:
        t = int(np.searchsorted(c, tgt))
        t = max(1, min(m - 1, t))
        t = max(last + l_min, t)

        if (m - t) < l_min:
            break

        cuts.append(t)
        last = t

    cuts = sorted(set(cuts))

    segs = []
    prev = 0

    for t in cuts:
        segs.append((prev, t))
        prev = t

    segs.append((prev, m))

    return segs


def segment_largest_jump(
    deltas: np.ndarray,
    m: int,
    M: int,
    l_min: int = LMIN_FIXED,
) -> list[tuple[int, int]]:
    deltas = np.asarray(deltas, dtype=np.float32)
    idx = np.argsort(-deltas)

    cuts = []

    for t0 in idx.tolist():
        t = t0 + 1

        if t < l_min or (m - t) < l_min:
            continue

        if all(abs(t - c) >= l_min for c in cuts):
            cuts.append(t)

        if len(cuts) >= M - 1:
            break

    cuts.sort()

    if len(cuts) == 0:
        return segment_uniform(m, M)

    segs = []
    prev = 0

    for t in cuts:
        segs.append((prev, t))
        prev = t

    segs.append((prev, m))

    return segs if len(segs) >= 2 else segment_uniform(m, M)


def get_segments(
    rule: str,
    deltas: np.ndarray,
    m: int,
    M: int,
    l_min: int = LMIN_FIXED,
) -> list[tuple[int, int]]:
    rule = rule.lower()

    if rule == "uniform":
        return segment_uniform(m, M)

    if rule == "equal_mass":
        return segment_equal_mass(
            deltas=deltas,
            m=m,
            M=M,
            l_min=l_min,
        )

    if rule == "largest_jump":
        return segment_largest_jump(
            deltas=deltas,
            m=m,
            M=M,
            l_min=l_min,
        )

    raise ValueError("rule must be one of: uniform | equal_mass | largest_jump")


def cut_cost_proxy(
    deltas: np.ndarray,
    segments: list[tuple[int, int]],
) -> float:
    """
    Mean transition cost at segment boundaries.
    Lower Cut is better, so Delta_Cut = Cut_random - Cut_star.
    """
    if deltas is None or segments is None or len(segments) <= 1:
        return float("nan")

    boundaries = [b for _, b in segments[:-1]]
    vals = []

    for b in boundaries:
        if 1 <= b <= len(deltas):
            vals.append(float(deltas[b - 1]))

    return float(np.mean(vals)) if vals else float("nan")


def hitrate_kh(
    X: np.ndarray,
    Pi: list[int],
    k: int = HIT_K,
    h: int = HIT_H,
    feat_subsample: int = HIT_FEATURE_SUBSAMPLE,
    row_subsample: int = HIT_ROW_SUBSAMPLE,
    seed: int = GLOBAL_SEED,
) -> float:
    """
    HitRate_{k,h}:
      For a feature i, find top-k nearest features under correlation distance.
      Count how many of those top-k neighbors fall within +/- h positions
      in the proposed ordering.
    """
    n, m = X.shape
    rng = np.random.RandomState(seed)

    if n > row_subsample:
        ridx = rng.choice(n, size=row_subsample, replace=False)
        Xr = X[ridx].astype(np.float32)
    else:
        Xr = X.astype(np.float32)

    S = int(min(feat_subsample, m))

    if S <= k + 1 or S < 10:
        return float("nan")

    feat_idx = rng.choice(m, size=S, replace=False)

    pos = np.empty(m, dtype=np.int32)
    for t, f in enumerate(Pi):
        pos[f] = t

    Xs = Xr[:, feat_idx]
    Xs = Xs - Xs.mean(axis=0, keepdims=True)
    Xs = Xs / (Xs.std(axis=0, keepdims=True) + 1e-6)

    corr = (Xs.T @ Xs) / max(1, Xs.shape[0] - 1)
    corr = np.clip(corr, -1.0, 1.0)

    dist = 1.0 - np.abs(corr)

    hits = []

    for i in range(S):
        d = dist[i].copy()
        d[i] = 1e9

        nn = np.argpartition(d, kth=k)[:k]

        fi = feat_idx[i]
        pi = int(pos[fi])

        left = max(0, pi - h)
        right = min(m - 1, pi + h)

        cnt = 0

        for j in nn:
            fj = feat_idx[j]
            pj = int(pos[fj])

            if left <= pj <= right:
                cnt += 1

        hits.append(cnt / float(k))

    return float(np.mean(hits))


def safe_nanmean(arr, default=np.nan) -> float:
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return float(default)

    return float(np.mean(arr))


def zscore_series(x: pd.Series) -> pd.Series:
    arr = x.to_numpy(dtype=np.float64)
    mask = np.isfinite(arr)

    out = np.full_like(arr, fill_value=np.nan, dtype=np.float64)

    if mask.sum() == 0:
        return pd.Series(out, index=x.index)

    if mask.sum() == 1:
        out[mask] = 0.0
        return pd.Series(out, index=x.index)

    mu = np.nanmean(arr[mask])
    sd = np.nanstd(arr[mask]) + 1e-12
    out[mask] = (arr[mask] - mu) / sd

    return pd.Series(out, index=x.index)


# ============================================================
# Main analyzer
# ============================================================

def analyze_array_ordering_metrics(
    X: np.ndarray,
    dataset_name: str = "Dataset",
    verbose: bool = True,
) -> dict:
    seed_everything(GLOBAL_SEED)

    n, m = X.shape
    category, rho = categorize_dataset_regime(n=n, m=m)

    if verbose:
        print(
            f"[INFO] Dataset={dataset_name} | "
            f"category={category} | n={n} | m={m} | rho={rho:.6g}"
        )

    # --------------------------------------------------------
    # IDF curve + AUC under IDF-CumVar curve
    # --------------------------------------------------------
    idf_curve, cumvar_curve, d_hat_final, cumvar_final = idf_cumvar_curve(
        X,
        thresholds=VAR_THRESHOLDS,
        max_rows=IDF_MAX_ROWS,
        seed=GLOBAL_SEED,
    )

    IDF_final = float(idf_curve[-1])
    AUC = auc_trapezoid(idf_curve, cumvar_curve)

    s = SENSITIVITY_EXPONENT

    psi_star = float(AUC ** s)
    FOE = float(psi_star / ((AUC * IDF_final + 1e-12) ** s))
    P_success = float(1.0 - IDF_final)

    # --------------------------------------------------------
    # GO-LR ordering Pi_star
    # --------------------------------------------------------
    k_clusters = min(GO_NUM_CLUSTERS, max(1, n))

    go = GraphFeatureOrdering(
        num_clusters=k_clusters,
        metric=GO_METRIC,
        refine=GO_REFINE,
        refine_passes=GO_REFINE_PASSES,
        direction_select=GO_DIRECTION_SELECT,
    )

    Pi_star, *_ = go.fit(
        X,
        seed=GLOBAL_SEED,
        deterministic=GO_DETERMINISTIC,
        use_cpu_kmeans=GO_USE_CPU_KMEANS,
    )
    Pi_star = list(Pi_star)

    # --------------------------------------------------------
    # Small-m guardrails
    # --------------------------------------------------------
    hit_valid = m > (2 * HIT_H + 1)
    cut_valid = m > M_FIXED

    # --------------------------------------------------------
    # Pi_star diagnostics
    # --------------------------------------------------------
    deltas_star = compute_deltas_for_ordering(
        X,
        Pi_star,
        max_sub_n=MAX_SUB_N_DELTAS,
    )

    AdjCoh_star = adjacency_coherence(deltas_star)

    if cut_valid:
        segs_star = get_segments(
            CUT_SEG_RULE,
            deltas_star,
            m=m,
            M=M_FIXED,
            l_min=LMIN_FIXED,
        )
        Cut_star = cut_cost_proxy(deltas_star, segs_star)
    else:
        Cut_star = float("nan")

    if hit_valid:
        Hit_star = hitrate_kh(
            X,
            Pi_star,
            k=HIT_K,
            h=HIT_H,
            feat_subsample=HIT_FEATURE_SUBSAMPLE,
            row_subsample=HIT_ROW_SUBSAMPLE,
            seed=GLOBAL_SEED + 999,
        )
    else:
        Hit_star = float("nan")

    # --------------------------------------------------------
    # Random expectation
    # --------------------------------------------------------
    AdjCoh_r = []
    Cut_r = []
    Hit_r = []

    for rs in RANDOM_SEEDS:
        Pi_r = make_random_perm(m, seed=rs)

        deltas_r = compute_deltas_for_ordering(
            X,
            Pi_r,
            max_sub_n=MAX_SUB_N_DELTAS,
        )

        AdjCoh_r.append(adjacency_coherence(deltas_r))

        if cut_valid:
            segs_r = get_segments(
                CUT_SEG_RULE,
                deltas_r,
                m=m,
                M=M_FIXED,
                l_min=LMIN_FIXED,
            )
            Cut_r.append(cut_cost_proxy(deltas_r, segs_r))
        else:
            Cut_r.append(float("nan"))

        if hit_valid:
            Hit_r.append(
                hitrate_kh(
                    X,
                    Pi_r,
                    k=HIT_K,
                    h=HIT_H,
                    feat_subsample=HIT_FEATURE_SUBSAMPLE,
                    row_subsample=HIT_ROW_SUBSAMPLE,
                    seed=GLOBAL_SEED + rs + 123,
                )
            )
        else:
            Hit_r.append(float("nan"))

    AdjCoh_rand_mean = safe_nanmean(AdjCoh_r, default=np.nan)
    Cut_rand_mean = safe_nanmean(Cut_r, default=np.nan)
    Hit_rand_mean = safe_nanmean(Hit_r, default=np.nan)

    Delta_AdjCoh = (
        float(AdjCoh_rand_mean - AdjCoh_star)
        if np.isfinite(AdjCoh_rand_mean) and np.isfinite(AdjCoh_star)
        else float("nan")
    )

    Delta_HitRate = (
        float(Hit_star - Hit_rand_mean)
        if np.isfinite(Hit_star) and np.isfinite(Hit_rand_mean)
        else float("nan")
    )

    Delta_Cut = (
        float(Cut_rand_mean - Cut_star)
        if np.isfinite(Cut_star) and np.isfinite(Cut_rand_mean)
        else float("nan")
    )

    return {
        "dataset": dataset_name,
        "category": category,
        "n": int(n),
        "m": int(m),
        "rho": float(rho),

        "d_hat_final": int(d_hat_final),
        "CumVar_final": float(cumvar_final),
        "IDF_final": float(IDF_final),
        "AUC": float(AUC),
        "psi_star": float(psi_star),

        "FOE": float(FOE),
        "P_success": float(P_success),

        "AdjCoh_star": float(AdjCoh_star),
        "HitRate_star": float(Hit_star),
        "Cut_star": float(Cut_star),

        "AdjCoh_rand_mean": float(AdjCoh_rand_mean),
        "HitRate_rand_mean": float(Hit_rand_mean),
        "Cut_rand_mean": float(Cut_rand_mean),

        "Delta_AdjCoh": float(Delta_AdjCoh),
        "Delta_HitRate": float(Delta_HitRate),
        "Delta_Cut": float(Delta_Cut),
    }


def finalize_metrics_table(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLS)

    # ========================================================
    # Special case: single-dataset evaluation
    # ========================================================
    # With one dataset, z-scoring each gain against itself gives 0.
    # Therefore, for a single row, LES is a raw dataset-level
    # average of the finite locality gains:
    #
    #   LES_single = mean(Delta_AdjCoh, Delta_HitRate, Delta_Cut)
    #
    # For multiple datasets, we keep the original benchmark-relative
    # LES: mean of z-scored locality gains across the evaluated set.
    # ========================================================
    if len(df) == 1:
        les_vals = []

        for _, r in df.iterrows():
            vals = [
                r.get("Delta_AdjCoh", np.nan),
                r.get("Delta_HitRate", np.nan),
                r.get("Delta_Cut", np.nan),
            ]
            vals = [float(v) for v in vals if np.isfinite(v)]

            les_vals.append(float(np.mean(vals)) if vals else float("nan"))

        df["LES"] = les_vals

    else:
        df["z_Delta_AdjCoh"] = zscore_series(df["Delta_AdjCoh"])
        df["z_Delta_HitRate"] = zscore_series(df["Delta_HitRate"])
        df["z_Delta_Cut"] = zscore_series(df["Delta_Cut"])

        les_vals = []

        for _, r in df.iterrows():
            vals = [
                r["z_Delta_AdjCoh"],
                r["z_Delta_HitRate"],
                r["z_Delta_Cut"],
            ]
            vals = [float(v) for v in vals if np.isfinite(v)]

            les_vals.append(float(np.mean(vals)) if vals else float("nan"))

        df["LES"] = les_vals

    df = df.sort_values(
        ["FOE", "dataset"],
        ascending=[False, True],
    ).reset_index(drop=True)

    df["FOE_rank"] = np.arange(1, len(df) + 1)

    return df[OUTPUT_COLS]


def analyze_csv_ordering_metrics(
    csv_path: str,
    target_col: Optional[str] = None,
    dataset_name: Optional[str] = None,
    out_csv: Optional[str] = None,
    standardize: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    if dataset_name is None:
        dataset_name = os.path.splitext(os.path.basename(csv_path))[0]

    X = load_numeric_csv(
        csv_path=csv_path,
        target_col=target_col,
        standardize=standardize,
    )

    row = analyze_array_ordering_metrics(
        X,
        dataset_name=dataset_name,
        verbose=verbose,
    )

    df = finalize_metrics_table([row])

    if out_csv is not None:
        df.to_csv(out_csv, index=False)

        if verbose:
            print(f"[SAVED] {out_csv}")

    return df


def analyze_many_csvs(
    datasets: list[dict],
    out_csv: str = "ordering_metrics_all.csv",
    standardize: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    rows = []

    for item in datasets:
        csv_path = item["csv_path"]
        target_col = _none_if_empty(item.get("target_col", None))

        dataset_name = _none_if_empty(item.get("dataset_name", None))
        if dataset_name is None:
            dataset_name = os.path.splitext(os.path.basename(csv_path))[0]

        if verbose:
            print("=" * 80)

        X = load_numeric_csv(
            csv_path=csv_path,
            target_col=target_col,
            standardize=standardize,
        )

        row = analyze_array_ordering_metrics(
            X,
            dataset_name=dataset_name,
            verbose=verbose,
        )

        rows.append(row)

    df = finalize_metrics_table(rows)

    if out_csv is not None:
        df.to_csv(out_csv, index=False)

        if verbose:
            print("=" * 80)
            print(f"[SAVED] {out_csv}")

    return df


def load_dataset_list_csv(path: str) -> list[dict]:
    """
    Expected columns:
      csv_path,target_col,dataset_name

    target_col can be empty for feature-only CSVs.
    dataset_name is optional.

    Category is NOT needed because this script computes category
    automatically from n, m, and rho.
    """
    df = pd.read_csv(path)

    if "csv_path" not in df.columns:
        raise ValueError(
            "Dataset-list CSV must contain at least a 'csv_path' column."
        )

    datasets = []

    for _, r in df.iterrows():
        item = {
            "csv_path": str(r["csv_path"]),
            "target_col": _none_if_empty(r.get("target_col", None)),
            "dataset_name": _none_if_empty(r.get("dataset_name", None)),
        }

        datasets.append(item)

    return datasets


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute dataset-level GOTabPFN ordering diagnostics: "
            "IDF, FOE, P_success, locality gains, LES, AUC, and automatic category."
        )
    )

    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to one numeric CSV file.",
    )

    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Optional target column to drop.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset display name.",
    )

    parser.add_argument(
        "--dataset-list",
        type=str,
        default=None,
        help=(
            "Optional CSV listing multiple datasets. "
            "Columns: csv_path,target_col,dataset_name"
        ),
    )

    parser.add_argument(
        "--out",
        type=str,
        default="ordering_metrics.csv",
        help="Output CSV path.",
    )

    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="Disable StandardScaler before computing diagnostics.",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose logs.",
    )

    args = parser.parse_args()

    verbose = not args.quiet
    standardize = not args.no_standardize

    if args.dataset_list is None and args.csv is None:
        raise ValueError("Provide either --csv or --dataset-list.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        if args.dataset_list is not None:
            datasets = load_dataset_list_csv(args.dataset_list)

            df = analyze_many_csvs(
                datasets=datasets,
                out_csv=args.out,
                standardize=standardize,
                verbose=verbose,
            )

        else:
            df = analyze_csv_ordering_metrics(
                csv_path=args.csv,
                target_col=args.target,
                dataset_name=args.dataset,
                out_csv=args.out,
                standardize=standardize,
                verbose=verbose,
            )

    if verbose:
        print("\n[Preview]")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()