# ============================================================
# NSC-pSP.py
#
# NSC-pSP: PCA-IDF + Segment-wise Principal Subspace Projection
#
# Core class:
#   pidf_segpca
#
# Added utilities:
#   - Load numeric CSV
#   - Drop target column if given
#   - Drop non-numeric columns if needed
#   - Load an ordering from CSV or use identity ordering
#   - Compute adjacent-correlation deltas for transition-aware segmentation
#   - Compress dataset
#   - Save compressed CSV, segments CSV, and metrics JSON
#
# CLI example:
#   python NSC-pSP.py \
#       --csv coloncancer_encoded.csv \
#       --target label \
#       --ordering-csv colon_golr_ordering.csv \
#       --dataset Colon \
#       --segmentation equal_mass \
#       --m-rule idf \
#       --tau 0.99 \
#       --gamma 1.7570143129240916 \
#       --beta 0.2244046472232107 \
#       --M-min 64 \
#       --M-max 384 \
#       --l-min 16 \
#       --out-prefix colon_nsc_psp
#
# Python/package example:
#   from gotabpfn import run_nsc_psp_csv
#
#   result = run_nsc_psp_csv(
#       csv_path="coloncancer_encoded.csv",
#       target_col="label",
#       ordering_csv="colon_golr_ordering.csv",
#       dataset_name="Colon",
#       segmentation="equal_mass",
#       m_rule="idf",
#       tau=0.99,
#   )
#
#   print(result["metrics"])
#   result["compressed_df"].to_csv("colon_nsc_psp.csv", index=False)
# ============================================================

from __future__ import annotations

import os
import json
import time
import argparse
from typing import Optional

import numpy as np
import pandas as pd

import math
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler


# ============================================================
# Core NSC-pSP class
# ============================================================

class pidf_segpca(nn.Module):
    """
    Variant v4: PCA-IDF + Segment-then-PCA tokens (no statistical descriptors).

    Steps:
      1) reorder features by a feature ordering algorithm (Pi*)
      2) segment along Pi* into M segments
      3) fit PC1 per segment on X_train
      4) token z_t(x) = (x_seg - mu_seg) @ v1_seg   (scalar)
      5) output (N, M) for mode="flatten"

    PCA-IDF:
      - if d_hat not provided and M not provided, estimate d_hat_pca via explained variance >= tau
      - IDF = d_hat_pca / m
      - choose M with m_rule in {"gamma","default","idf"}

    Determinism:
      - uses CPU SVD for segment PCA (HDLSS safe + stable)
      - deterministic sign fix for each PC1
    """

    def __init__(
        self,
        gamma: float = 2.0,
        M_min: int = 32,
        M_max: int = 512,
        bypass_if_m_leq: int = 400,

        segmentation: str = "uniform",   # "uniform" | "largest_jump" | "equal_mass"
        l_min: int = 8,

        # M rule
        m_rule: str = "default",         # "gamma" | "default" | "idf"
        beta: float = 0.5,

        # PCA-ID
        tau: float = 0.99,
        assume_standardized: bool = True,
        standardize_eps: float = 1e-12,

        # segment PCA
        center: bool = True,

        eps: float = 1e-12,
        device: str | None = None,
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.M_min = int(M_min)
        self.M_max = int(M_max)
        self.bypass_if_m_leq = int(bypass_if_m_leq)

        self.segmentation = segmentation.lower()
        if self.segmentation not in ("uniform", "largest_jump", "equal_mass"):
            raise ValueError("segmentation must be: 'uniform' | 'largest_jump' | 'equal_mass'")
        self.l_min = int(l_min)

        self.m_rule = m_rule.lower()
        if self.m_rule not in ("gamma", "default", "idf"):
            raise ValueError("m_rule must be 'gamma' | 'default' | 'idf'")
        self.beta = float(beta)

        self.tau = float(tau)
        if not (0.0 < self.tau < 1.0):
            raise ValueError("tau must be in (0,1).")

        self.assume_standardized = bool(assume_standardized)
        self.standardize_eps = float(standardize_eps)
        self.center = bool(center)

        self.eps = float(eps)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # configured
        self.Pi_star: list[int] | None = None
        self.M: int | None = None
        self.segments: list[tuple[int, int]] | None = None
        self.idf_: float | None = None
        self.d_hat_pca_: float | None = None

        # buffers for segment params will be registered dynamically:
        self._n_segments = 0

    def clear_cache(self):
        if torch.cuda.is_available() and self.device == "cuda":
            torch.cuda.empty_cache()

    # ------------------------- standardize / PCA-ID -------------------------
    @torch.no_grad()
    def _standardize_if_needed(self, X: torch.Tensor) -> torch.Tensor:
        X = X.float()
        if self.assume_standardized:
            return X
        mu = X.mean(dim=0, keepdim=True)
        Xc = X - mu
        std = Xc.std(dim=0, unbiased=False, keepdim=True).clamp_min(self.standardize_eps)
        return Xc / std

    @torch.no_grad()
    def estimate_d_hat_pca(self, X_train: torch.Tensor, tau: float | None = None) -> float:
        """
        HDLSS-safe PCA-ID via Gram eigenvalues:
          G = (1/(n-1)) X X^T  (n x n)
        """
        if tau is None:
            tau = self.tau
        tau = float(tau)

        X = torch.as_tensor(X_train).float()
        X = self._standardize_if_needed(X)
        n, m = X.shape
        if n <= 1 or m <= 0:
            return 1.0

        Xc = (X - X.mean(dim=0, keepdim=True)).cpu()
        G = (Xc @ Xc.T) / max(1, n - 1)  # (n,n)

        evals = torch.linalg.eigvalsh(G).clamp_min(0.0)  # ascending
        total = float(evals.sum().item())
        if total <= self.eps:
            return 1.0

        evals_desc = torch.flip(evals, dims=[0])
        evr = evals_desc / (total + self.eps)
        cum = torch.cumsum(evr, dim=0)

        k = int(torch.searchsorted(cum, torch.tensor(tau)).item()) + 1
        k = max(1, min(k, int(cum.numel())))
        return float(k)

    # ------------------------- choose M -------------------------
    def choose_M(self, m: int, d_hat: float | None, M: int | None):
        m = int(m)
        if M is not None:
            M_final = max(1, min(int(M), m))
            idf = float(d_hat) / float(m) if d_hat is not None else None
            return M_final, idf

        if d_hat is None:
            raise ValueError("Provide either M or d_hat to choose M.")

        d_hat = float(d_hat)
        idf = d_hat / float(m) if m > 0 else 1.0

        if m <= self.bypass_if_m_leq:
            return m, idf

        if self.m_rule == "gamma":
            M_raw = math.ceil(self.gamma * d_hat)
        elif self.m_rule == "default":
            M_raw = math.ceil(2.0 * d_hat)
        else:  # idf
            scale = 1.0 + self.beta * (1.0 - idf)
            M_raw = math.ceil(scale * d_hat)

        M_clip = max(self.M_min, min(self.M_max, min(m, M_raw)))
        return int(M_clip), idf

    # ------------------------- segmentation -------------------------
    def _segment_uniform(self, m: int, M: int) -> list[tuple[int, int]]:
        s = int(math.ceil(m / M))
        segs = []
        for t in range(M):
            a = t * s
            b = min((t + 1) * s, m)
            if a < b:
                segs.append((a, b))
        return segs

    @torch.no_grad()
    def _segment_equal_mass(self, deltas: torch.Tensor, m: int, M: int) -> list[tuple[int, int]]:
        deltas = deltas.clamp_min(0.0).float()
        c = torch.zeros(m, device=deltas.device, dtype=torch.float32)
        c[1:] = torch.cumsum(deltas, dim=0)
        total = float(c[-1].item())
        if total <= self.eps:
            return self._segment_uniform(m, M)

        targets = [(k / M) * total for k in range(1, M)]
        cuts, last = [], 0
        for tgt in targets:
            t = int(torch.searchsorted(c, torch.tensor(tgt, device=c.device)).item())
            t = max(1, min(m - 1, t))
            t = max(last + self.l_min, t)
            if (m - t) < self.l_min:
                break
            cuts.append(t)
            last = t

        cuts = sorted(set(cuts))
        segs, prev = [], 0
        for t in cuts:
            segs.append((prev, t))
            prev = t
        segs.append((prev, m))
        return segs

    @torch.no_grad()
    def _segment_largest_jump(self, deltas: torch.Tensor, m: int, M: int) -> list[tuple[int, int]]:
        deltas = deltas.float()
        if deltas.numel() == 0:
            return [(0, m)]
        _, idx = torch.sort(deltas, descending=True)

        cuts = []
        for t0 in idx.tolist():
            t = t0 + 1
            if t < self.l_min or (m - t) < self.l_min:
                continue
            if all(abs(t - c) >= self.l_min for c in cuts):
                cuts.append(t)
            if len(cuts) >= (M - 1):
                break

        cuts.sort()
        segs, prev = [], 0
        for t in cuts:
            segs.append((prev, t))
            prev = t
        segs.append((prev, m))
        return segs if len(segs) >= 2 else self._segment_uniform(m, M)

    # ------------------------- segment PCA fit -------------------------
    @torch.no_grad()
    def _fit_pc1(self, Xseg_cpu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Xseg_cpu: (N, d) on CPU float32.
        Returns mu(d,), v1(d,)
        """
        Xseg_cpu = Xseg_cpu.float()
        if self.center:
            mu = Xseg_cpu.mean(dim=0)
            Xc = Xseg_cpu - mu
        else:
            mu = torch.zeros(Xseg_cpu.shape[1], dtype=Xseg_cpu.dtype)
            Xc = Xseg_cpu

        d = Xc.shape[1]
        if d == 1:
            v1 = torch.ones(1, dtype=Xc.dtype)
            return mu, v1

        try:
            # Xc = U S Vt; PC1 direction is Vt[0]
            _, _, Vt = torch.linalg.svd(Xc, full_matrices=False)
            v1 = Vt[0]
        except RuntimeError:
            C = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)
            _, evecs = torch.linalg.eigh(C)
            v1 = evecs[:, -1]

        # deterministic sign fix: correlate PC scores with per-sample segment mean
        scores = Xc @ v1
        ref = Xc.mean(dim=1)
        if (scores * ref).mean() < 0:
            v1 = -v1

        v1 = v1 / (v1.norm() + self.eps)
        return mu, v1

    def _clear_segment_buffers(self):
        # remove old mu_/v1_ buffers if re-configured
        for i in range(self._n_segments):
            if f"mu_{i}" in self._buffers:
                del self._buffers[f"mu_{i}"]
            if f"v1_{i}" in self._buffers:
                del self._buffers[f"v1_{i}"]
        self._n_segments = 0

    # ------------------------- configure/transform -------------------------
    @torch.no_grad()
    def configure(
        self,
        Pi_star: list[int],
        m: int | None = None,
        d_hat: float | None = None,
        M: int | None = None,
        deltas: torch.Tensor | None = None,
        X_train: torch.Tensor | None = None,
        tau: float | None = None,
    ):
        if X_train is None:
            raise ValueError("pidf_segpca requires X_train to fit per-segment PCA directions.")

        self.Pi_star = list(Pi_star)
        if m is None:
            m = len(Pi_star)
        m = int(m)

        # PCA-inspired intrinsic dimension if needed
        if (d_hat is None) and (M is None):
            d_hat = self.estimate_d_hat_pca(X_train, tau=tau)
            self.d_hat_pca_ = float(d_hat)
        else:
            self.d_hat_pca_ = float(d_hat) if d_hat is not None else None

        # choose M with PCA-IDF (if m_rule="idf")
        self.M, self.idf_ = self.choose_M(m=m, d_hat=d_hat, M=M)

        # segmentation
        if self.segmentation == "uniform":
            self.segments = self._segment_uniform(m, self.M)
        else:
            if deltas is None:
                raise ValueError("For transition-aware segmentation, provide deltas (shape m-1).")
            deltas = torch.as_tensor(deltas, device="cpu").float()
            if deltas.numel() != (m - 1):
                raise ValueError(f"deltas must have shape (m-1,) = ({m-1},), got {tuple(deltas.shape)}")
            if self.segmentation == "equal_mass":
                self.segments = self._segment_equal_mass(deltas, m, self.M)
            else:
                self.segments = self._segment_largest_jump(deltas, m, self.M)

        # fit per-segment PC1 on CPU
        Xtr = torch.as_tensor(X_train).float()
        Xtr = self._standardize_if_needed(Xtr)
        Xtr = Xtr[:, torch.tensor(self.Pi_star, dtype=torch.long)]  # reorder by Pi*

        self._clear_segment_buffers()
        for i, (a, b) in enumerate(self.segments):
            mu, v1 = self._fit_pc1(Xtr[:, a:b].cpu())
            self.register_buffer(f"mu_{i}", mu)   # cpu buffer
            self.register_buffer(f"v1_{i}", v1)   # cpu buffer

        self._n_segments = len(self.segments)
        return self

    @torch.no_grad()
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        if self.Pi_star is None or self.segments is None:
            raise RuntimeError("Call configure(...) first.")

        X = torch.as_tensor(X).float().to(self.device)
        X = self._standardize_if_needed(X)
        perm = torch.tensor(self.Pi_star, device=self.device, dtype=torch.long)
        Xp = X[:, perm]

        tokens = []
        for i, (a, b) in enumerate(self.segments):
            mu = getattr(self, f"mu_{i}").to(self.device)
            v1 = getattr(self, f"v1_{i}").to(self.device)
            Xseg = Xp[:, a:b]
            if self.center:
                Xseg = Xseg - mu[None, :]
            z = (Xseg @ v1).unsqueeze(1)  # (N,1)
            tokens.append(z)

        return torch.cat(tokens, dim=1)  # (N,M)

    @torch.no_grad()
    def compress(self, X: torch.Tensor, mode: str = "flatten") -> torch.Tensor:
        mode = mode.lower()
        Z = self.transform(X)
        if mode == "flatten":
            return Z
        if mode == "mean":
            return Z.mean(dim=1, keepdim=True)
        if mode == "max":
            return Z.max(dim=1, keepdim=True).values
        if mode == "meanmax":
            return torch.cat([Z.mean(dim=1, keepdim=True), Z.max(dim=1, keepdim=True).values], dim=1)
        raise ValueError("mode must be one of: flatten | mean | max | meanmax")

    @torch.no_grad()
    def compress_numpy(self, X: torch.Tensor, mode: str = "flatten", dtype="float32"):
        import numpy as np
        out = self.compress(X, mode=mode).detach().cpu().numpy()
        out = out.astype(np.float32 if dtype == "float32" else np.float64, copy=False)
        return out

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.transform(X)


# Friendly alias for package export
NSC_pSP = pidf_segpca


# ============================================================
# Input/output wrapper
# ============================================================

def _nsc_load_csv(
    csv_path: str,
    target_col: Optional[str] = None,
    standardize: bool = True,
    drop_non_numeric: bool = True,
):
    df = pd.read_csv(csv_path)

    y = None
    if target_col is not None:
        if target_col not in df.columns:
            raise ValueError(
                f"Target column '{target_col}' not found. "
                f"Available columns: {list(df.columns)}"
            )
        y = df[target_col].to_numpy()
        df = df.drop(columns=[target_col])

    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]

    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

    if non_numeric_cols:
        if drop_non_numeric:
            print(f"[INFO] Dropping non-numeric columns: {non_numeric_cols}")
            df = df[numeric_cols]
        else:
            raise ValueError(f"Non-numeric feature columns found: {non_numeric_cols}")

    feature_names = list(df.columns)

    if len(feature_names) < 2:
        raise ValueError(
            f"Expected at least two numeric feature columns. Got {len(feature_names)}."
        )

    X = df.to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if standardize:
        X = StandardScaler().fit_transform(X).astype(np.float32)

    return X, y, feature_names


def _load_ordering(ordering_csv: Optional[str], m: int) -> list[int]:
    """
    Load ordering from a CSV with a column named feature_index.
    If no ordering_csv is given, returns identity ordering.
    """
    if ordering_csv is None:
        return list(range(m))

    odf = pd.read_csv(ordering_csv)

    if "feature_index" not in odf.columns:
        raise ValueError(
            f"Ordering CSV must contain column 'feature_index'. "
            f"Available columns: {list(odf.columns)}"
        )

    Pi_star = odf["feature_index"].astype(int).tolist()

    if len(Pi_star) != m or sorted(Pi_star) != list(range(m)):
        raise ValueError(
            f"Invalid ordering. Expected a permutation of 0..{m-1}, "
            f"got length={len(Pi_star)}."
        )

    return Pi_star


def compute_adjacent_corr_deltas(
    X: np.ndarray,
    Pi_star: list[int],
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    delta[t] = 1 - |corr(feature_{Pi[t]}, feature_{Pi[t+1]})|

    Used for transition-aware segmentation:
      - largest_jump
      - equal_mass
    """
    X_t = torch.as_tensor(X, dtype=torch.float32)
    perm = torch.tensor(Pi_star, dtype=torch.long)

    Xp = X_t[:, perm]
    Xc = Xp - Xp.mean(dim=0, keepdim=True)
    std = Xc.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    Z = Xc / std

    corr_adj = (Z[:, :-1] * Z[:, 1:]).mean(dim=0)
    deltas = 1.0 - corr_adj.abs()

    return deltas.cpu()


def _segments_to_dataframe(segments: list[tuple[int, int]]) -> pd.DataFrame:
    rows = []

    for i, (a, b) in enumerate(segments):
        rows.append({
            "segment_id": i,
            "start_rank": int(a),
            "end_rank_exclusive": int(b),
            "segment_size": int(b - a),
        })

    return pd.DataFrame(rows)


def _compressed_to_dataframe(Z: np.ndarray, prefix: str = "z") -> pd.DataFrame:
    cols = [f"{prefix}_{i+1:03d}" for i in range(Z.shape[1])]
    return pd.DataFrame(Z, columns=cols)


def run_nsc_psp_csv(
    csv_path: str,
    target_col: Optional[str] = None,
    ordering_csv: Optional[str] = None,
    dataset_name: Optional[str] = None,

    gamma: float = 2.0,
    M_min: int = 32,
    M_max: int = 512,
    bypass_if_m_leq: int = 400,
    segmentation: str = "uniform",
    l_min: int = 8,
    m_rule: str = "default",
    beta: float = 0.5,
    tau: float = 0.99,
    center: bool = True,

    standardize_input: bool = True,
    drop_non_numeric: bool = True,
    mode: str = "flatten",
    device: Optional[str] = None,

    save_outputs: bool = True,
    out_prefix: Optional[str] = None,
):
    """
    Run NSC-pSP compression on a CSV dataset.

    Returns:
        dict with:
          - compressed
          - compressed_df
          - segments_df
          - ordering
          - metrics
          - model
    """
    if dataset_name is None:
        dataset_name = os.path.splitext(os.path.basename(csv_path))[0]

    if out_prefix is None:
        safe_name = dataset_name.replace(" ", "_").replace("/", "_")
        out_prefix = f"{safe_name}_nsc_psp"

    X, y, feature_names = _nsc_load_csv(
        csv_path=csv_path,
        target_col=target_col,
        standardize=standardize_input,
        drop_non_numeric=drop_non_numeric,
    )

    n, m = X.shape
    Pi_star = _load_ordering(ordering_csv=ordering_csv, m=m)

    segmentation = segmentation.lower()

    deltas = None
    if segmentation in {"largest_jump", "equal_mass"}:
        deltas = compute_adjacent_corr_deltas(X, Pi_star)

    model = pidf_segpca(
        gamma=gamma,
        M_min=M_min,
        M_max=M_max,
        bypass_if_m_leq=bypass_if_m_leq,
        segmentation=segmentation,
        l_min=l_min,
        m_rule=m_rule,
        beta=beta,
        tau=tau,
        assume_standardized=True,
        center=center,
        device=device,
    )

    print(f"[NSC-pSP] Dataset: {dataset_name}")
    print(f"[NSC-pSP] X shape: {X.shape}")
    print(f"[NSC-pSP] ordering: {'identity' if ordering_csv is None else ordering_csv}")
    print(f"[NSC-pSP] segmentation={segmentation}, m_rule={m_rule}, tau={tau}")

    t0 = time.perf_counter()

    X_t = torch.as_tensor(X, dtype=torch.float32)

    model.configure(
        Pi_star=Pi_star,
        X_train=X_t,
        tau=tau,
        deltas=deltas,
    )

    Z = model.compress_numpy(X_t, mode=mode, dtype="float32")

    runtime_sec = float(time.perf_counter() - t0)

    compressed_df = _compressed_to_dataframe(Z)
    segments_df = _segments_to_dataframe(model.segments)

    metrics = {
        "dataset": dataset_name,
        "n": int(n),
        "m_original": int(m),
        "m_compressed": int(Z.shape[1]),
        "compression_ratio": float(m / max(1, Z.shape[1])),
        "ordering_source": "identity" if ordering_csv is None else ordering_csv,
        "segmentation": segmentation,
        "m_rule": m_rule,
        "tau": float(tau),
        "gamma": float(gamma),
        "beta": float(beta),
        "M_min": int(M_min),
        "M_max": int(M_max),
        "l_min": int(l_min),
        "M_selected": int(model.M),
        "idf": None if model.idf_ is None else float(model.idf_),
        "d_hat_pca": None if model.d_hat_pca_ is None else float(model.d_hat_pca_),
        "runtime_sec": runtime_sec,
    }

    result = {
        "compressed": Z,
        "compressed_df": compressed_df,
        "segments_df": segments_df,
        "ordering": Pi_star,
        "feature_names": feature_names,
        "metrics": metrics,
        "model": model,
    }

    print("\n================ NSC-pSP SUMMARY ================")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    if save_outputs:
        compressed_path = f"{out_prefix}_compressed.csv"
        segments_path = f"{out_prefix}_segments.csv"
        metrics_path = f"{out_prefix}_metrics.json"

        compressed_df.to_csv(compressed_path, index=False)
        segments_df.to_csv(segments_path, index=False)

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print("\n[SAVED]")
        print(f"  - {compressed_path}")
        print(f"  - {segments_path}")
        print(f"  - {metrics_path}")

    return result


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run NSC-pSP compression on a numeric CSV dataset."
    )

    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--ordering-csv", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)

    parser.add_argument("--segmentation", type=str, default="uniform",
                        choices=["uniform", "largest_jump", "equal_mass"])
    parser.add_argument("--m-rule", type=str, default="default",
                        choices=["gamma", "default", "idf"])

    parser.add_argument("--tau", type=float, default=0.99)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--M-min", type=int, default=32)
    parser.add_argument("--M-max", type=int, default=512)
    parser.add_argument("--l-min", type=int, default=8)
    parser.add_argument("--bypass-if-m-leq", type=int, default=400)

    parser.add_argument("--mode", type=str, default="flatten",
                        choices=["flatten", "mean", "max", "meanmax"])
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--no-center", action="store_true")
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--keep-non-numeric", action="store_true")

    parser.add_argument("--out-prefix", type=str, default=None)

    args = parser.parse_args()

    run_nsc_psp_csv(
        csv_path=args.csv,
        target_col=args.target,
        ordering_csv=args.ordering_csv,
        dataset_name=args.dataset,
        gamma=args.gamma,
        M_min=args.M_min,
        M_max=args.M_max,
        bypass_if_m_leq=args.bypass_if_m_leq,
        segmentation=args.segmentation,
        l_min=args.l_min,
        m_rule=args.m_rule,
        beta=args.beta,
        tau=args.tau,
        center=not args.no_center,
        standardize_input=not args.no_standardize,
        drop_non_numeric=not args.keep_non_numeric,
        mode=args.mode,
        device=args.device,
        save_outputs=True,
        out_prefix=args.out_prefix,
    )


if __name__ == "__main__":
    main()