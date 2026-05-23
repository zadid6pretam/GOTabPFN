# ============================================================
# NSC-P.py
#
# NSC-P: PCA-IDF-aware descriptor/statistics-based NSC
#
# Core class:
#   NSC_PCA
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
#   python NSC-P.py \
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
#       --descriptor basic \
#       --pooling learn_free \
#       --out-prefix colon_nsc_p
#
# Python/package example:
#   from gotabpfn import run_nsc_p_csv
#
#   result = run_nsc_p_csv(
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
#   result["compressed_df"].to_csv("colon_nsc_p.csv", index=False)
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
# Core NSC-P class
# ============================================================

class NSC_PCA(nn.Module):
    """
    Neuro-Inspired Subunit Compression (NSC) with intrinsic-dim-aware M.

    - Output tokens: Z(X) in R^{N x M x 1}
    - Compressed latent via compress(...): e.g., (N, M) for mode="flatten".

    M selection:
      - m_rule="gamma"   : M ≈ γ * d_hat
      - m_rule="default" : M ≈ 2 * d_hat  (Eq. (defaultM))
      - m_rule="idf"     : M ≈ (1 + β (1-IDF)) d_hat (Eq. (idf_budget)), IDF=d_hat/m

    Intrinsic dimension (PCA-inspired):
      - If d_hat is not provided to configure(...), estimate d_hat from X_train via
        cumulative explained variance >= tau (default tau=0.99).
      - HDLSS-friendly: compute eigenvalues via Gram matrix G = (1/(n-1)) X X^T (n x n),
        whose eigenvalues equal the nonzero eigenvalues of the covariance.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        M_min: int = 32,
        M_max: int = 512,
        bypass_if_m_leq: int = 400,

        segmentation: str = "uniform",   # "uniform" | "largest_jump" | "equal_mass"
        l_min: int = 8,

        descriptor: str = "basic",       # "basic" | "robust" | "moments" | "all"
        pooling: str = "learn_free",     # "learn_free" | "linear" | "mlp"
        mlp_hidden: int = 64,
        mlp_depth: int = 2,
        dropout: float = 0.0,

        # How to choose M
        m_rule: str = "default",         # "gamma" | "default" | "idf"
        beta: float = 0.5,               # only used if m_rule="idf"

        # PCA-ID params
        tau: float = 0.99,               # explained-variance threshold for d_hat_PCA(tau)
        assume_standardized: bool = True,# if False, will standardize X_train internally
        standardize_eps: float = 1e-12,  # for stable std

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

        self.descriptor = descriptor.lower()
        if self.descriptor not in ("basic", "robust", "moments", "all"):
            raise ValueError("descriptor must be: 'basic' | 'robust' | 'moments' | 'all'")

        self.pooling = pooling.lower()
        if self.pooling not in ("learn_free", "linear", "mlp"):
            raise ValueError("pooling must be: 'learn_free' | 'linear' | 'mlp'")
        self.mlp_hidden = int(mlp_hidden)
        self.mlp_depth = int(mlp_depth)
        self.dropout = float(dropout)

        # M rule / IDF
        self.m_rule = m_rule.lower()
        if self.m_rule not in ("gamma", "default", "idf"):
            raise ValueError("m_rule must be 'gamma' | 'default' | 'idf'")
        self.beta = float(beta)

        # PCA-ID params
        self.tau = float(tau)
        if not (0.0 < self.tau < 1.0):
            raise ValueError("tau must be in (0, 1).")
        self.assume_standardized = bool(assume_standardized)
        self.standardize_eps = float(standardize_eps)

        self.eps = float(eps)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # configured state
        self.Pi_star: list[int] | None = None
        self.M: int | None = None
        self.segments: list[tuple[int, int]] | None = None

        # store IDF once configured
        self.idf_: float | None = None

        # store PCA-ID once configured (optional)
        self.d_hat_pca_: float | None = None

        # pooling module g_theta
        self.q: int | None = None
        self.g_theta: nn.Module | None = None

        # NOTE: do NOT pre-create self._w here; it will be registered as a buffer in _build_pooler

        # caches
        self.last_tokens_shape_: tuple[int, ...] | None = None
        self.last_compressed_: torch.Tensor | None = None
        self.last_compressed_np_ = None
        self.last_compressed_df_ = None

    def clear_cache(self):
        self.last_tokens_shape_ = None
        self.last_compressed_ = None
        self.last_compressed_np_ = None
        self.last_compressed_df_ = None
        if torch.cuda.is_available() and self.device == "cuda":
            torch.cuda.empty_cache()

    # ------------------------- PCA-ID -------------------------
    @torch.no_grad()
    def _standardize_if_needed(self, X: torch.Tensor) -> torch.Tensor:
        """
        If assume_standardized=False, standardize per feature: zero mean, unit variance.
        """
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
        PCA-inspired intrinsic dimension via cumulative explained variance >= tau.

        HDLSS-friendly:
          - compute eigenvalues of Gram: G=(1/(n-1)) X X^T (n x n)
          - eigenvalues(G) equal the nonzero eigenvalues of covariance

        Returns
        -------
        d_hat_pca : float  (returned as float for consistency with existing d_hat usage)
        """
        if tau is None:
            tau = self.tau
        tau = float(tau)
        if not (0.0 < tau < 1.0):
            raise ValueError("tau must be in (0, 1).")

        X = torch.as_tensor(X_train).float()
        X = self._standardize_if_needed(X)

        n, m = X.shape
        if n <= 1 or m <= 0:
            return 1.0

        # Gram matrix on CPU is often more stable in HDLSS (n is small).
        Xc = X.detach().cpu()
        G = (Xc @ Xc.T) / max(n - 1, 1)

        evals = torch.linalg.eigvalsh(G).real  # ascending
        evals = evals.clamp_min(0.0)
        total = float(evals.sum().item())
        if total <= self.eps:
            return 1.0

        evals_desc = torch.flip(evals, dims=[0])
        evr = evals_desc / (evals_desc.sum() + self.eps)
        cum = torch.cumsum(evr, dim=0)

        k = int(torch.searchsorted(cum, torch.tensor(tau, device=cum.device)).item()) + 1
        k = max(1, min(k, int(cum.numel())))
        return float(k)

    # ------------------------- M rule -------------------------
    def choose_M(self, m: int, d_hat: float | None, M: int | None):
        """
        Returns
        -------
        M_final : int
        idf     : float | None
            Intrinsic Dimensionality Factor (d_hat / m) if d_hat is given.
        """
        m = int(m)

        # Explicit override wins; we still compute IDF if d_hat is known.
        if M is not None:
            M_final = max(1, min(int(M), m))
            idf = float(d_hat) / float(m) if d_hat is not None else None
            return M_final, idf

        if d_hat is None:
            raise ValueError("Provide either M or d_hat to choose M.")

        d_hat = float(d_hat)
        idf = d_hat / float(m) if m > 0 else 1.0

        # Bypass compression for low-d tabular
        if m <= self.bypass_if_m_leq:
            return m, idf

        # Raw proposal based on chosen rule
        if self.m_rule == "gamma":
            M_raw = math.ceil(self.gamma * d_hat)
        elif self.m_rule == "default":
            M_raw = math.ceil(2.0 * d_hat)
        else:  # "idf"
            scale = 1.0 + self.beta * (1.0 - idf)
            M_raw = math.ceil(scale * d_hat)

        # Clip to [M_min, min(M_max, m)]
        M_clip = max(self.M_min, min(self.M_max, min(m, M_raw)))
        return int(M_clip), idf

    # ---------------------- segmentation ----------------------
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
    def _adjacent_deltas_from_Wbar(self, Wbar: torch.Tensor, Pi_star: list[int]) -> torch.Tensor:
        perm = torch.tensor(Pi_star, device=Wbar.device, dtype=torch.long)
        return Wbar[perm[:-1], perm[1:]].float()

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

    # ---------------------- descriptor ψ ----------------------
    def _descriptor_dim(self) -> int:
        return 8 if self.descriptor == "all" else 4

    @torch.no_grad()
    def _psi(self, U: torch.Tensor) -> torch.Tensor:
        U = U.float()
        mean = U.mean(dim=1)
        std = U.std(dim=1, unbiased=False).clamp_min(self.eps)

        if self.descriptor == "basic":
            mn = U.min(dim=1).values
            mx = U.max(dim=1).values
            return torch.stack([mean, std, mn, mx], dim=1)

        if self.descriptor == "robust":
            med = torch.quantile(U, 0.5, dim=1)
            q25 = torch.quantile(U, 0.25, dim=1)
            q75 = torch.quantile(U, 0.75, dim=1)
            iqr = q75 - q25
            return torch.stack([mean, std, med, iqr], dim=1)

        z = (U - mean[:, None]) / std[:, None]
        skew = (z ** 3).mean(dim=1)
        kurt = (z ** 4).mean(dim=1) - 3.0

        if self.descriptor == "moments":
            return torch.stack([mean, std, skew, kurt], dim=1)

        mn = U.min(dim=1).values
        mx = U.max(dim=1).values
        med = torch.quantile(U, 0.5, dim=1)
        q25 = torch.quantile(U, 0.25, dim=1)
        q75 = torch.quantile(U, 0.75, dim=1)
        iqr = q75 - q25
        return torch.stack([mean, std, mn, mx, med, iqr, skew, kurt], dim=1)

    # ---------------------- pooling gθ (q -> 1) ------------------------
    def _build_pooler(self, q: int):
        # always output 1D tokens
        if self.pooling == "learn_free":
            w = torch.ones(q, dtype=torch.float32)
            if q >= 2:
                w[0] = 1.0  # mean
                w[1] = 0.5  # std
            if q >= 4:
                w[2] = 0.25 # min
                w[3] = 0.25 # max
            w = w / (w.abs().sum() + self.eps)
            w = w.to(self.device)

            # safe for reconfigure: update buffer if it exists
            if "_w" in self._buffers:
                self._buffers["_w"] = w
            else:
                self.register_buffer("_w", w)

            self.g_theta = None
            return

        if self.pooling == "linear":
            self.g_theta = nn.Linear(q, 1).to(self.device)
            return

        layers = []
        d = q
        depth = max(1, self.mlp_depth)
        for _ in range(depth - 1):
            layers += [nn.Linear(d, self.mlp_hidden), nn.ReLU(inplace=True)]
            if self.dropout > 0:
                layers += [nn.Dropout(self.dropout)]
            d = self.mlp_hidden
        layers += [nn.Linear(d, 1)]
        self.g_theta = nn.Sequential(*layers).to(self.device)

    # -------------------- main configuration -------------------
    @torch.no_grad()
    def configure(
        self,
        Pi_star: list[int],
        m: int | None = None,
        d_hat: float | None = None,
        M: int | None = None,
        Wbar: torch.Tensor | None = None,
        deltas: torch.Tensor | None = None,

        # PCA-ID inputs (optional)
        X_train: torch.Tensor | None = None,
        tau: float | None = None,
    ):
        self.Pi_star = list(Pi_star)
        if m is None:
            m = len(Pi_star)
        m = int(m)

        # If neither d_hat nor M is given, estimate intrinsic dim via PCA-inspired rule.
        if (d_hat is None) and (M is None):
            if X_train is None:
                raise ValueError("Provide d_hat or M, or provide X_train to estimate d_hat via PCA.")
            d_hat = self.estimate_d_hat_pca(X_train, tau=tau)
            self.d_hat_pca_ = float(d_hat)
        else:
            self.d_hat_pca_ = float(d_hat) if d_hat is not None else None

        # choose M + IDF
        self.M, self.idf_ = self.choose_M(m=m, d_hat=d_hat, M=M)

        # segmentation
        if self.segmentation == "uniform":
            self.segments = self._segment_uniform(m, self.M)
        else:
            if deltas is None:
                if Wbar is None:
                    raise ValueError("For transition-aware segmentation, provide deltas or Wbar.")
                Wbar = torch.as_tensor(Wbar, device=self.device)
                deltas = self._adjacent_deltas_from_Wbar(Wbar, self.Pi_star)
            else:
                deltas = torch.as_tensor(deltas, device=self.device)

            if deltas.numel() != (m - 1):
                raise ValueError(f"deltas must have shape (m-1,) = ({m-1},), got {tuple(deltas.shape)}")

            if self.segmentation == "equal_mass":
                self.segments = self._segment_equal_mass(deltas, m, self.M)
            else:
                self.segments = self._segment_largest_jump(deltas, m, self.M)

        self.q = self._descriptor_dim()
        self._build_pooler(self.q)
        return self

    # -------------------- tokenization ------------------------
    @torch.no_grad()
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (N, m)
        returns: (N, M, 1)
        """
        if self.Pi_star is None or self.segments is None:
            raise RuntimeError("Call configure(...) first.")

        X = torch.as_tensor(X, device=self.device).float()
        perm = torch.tensor(self.Pi_star, device=X.device, dtype=torch.long)
        Xp = X[:, perm]  # x^Pi

        tokens = []
        for (a, b) in self.segments:
            psi = self._psi(Xp[:, a:b])  # (N, q)
            if self.pooling == "learn_free":
                z = (psi * self._w[None, :]).sum(dim=1, keepdim=True)  # (N,1)
            else:
                z = self.g_theta(psi)  # (N,1)
            tokens.append(z)

        Z = torch.stack(tokens, dim=1)  # (N, M, 1)
        self.last_tokens_shape_ = tuple(Z.shape)
        return Z

    # -------------------- compression for serialization/SFS ------------------------
    @torch.no_grad()
    def compress(self, X: torch.Tensor, mode: str = "flatten") -> torch.Tensor:
        Z = self.transform(X).float()  # (N, M, 1)
        mode = mode.lower()

        if mode == "flatten":
            X2 = Z.squeeze(-1)  # (N, M)
        elif mode == "mean":
            X2 = Z.mean(dim=1)  # (N, 1)
        elif mode == "max":
            X2 = Z.max(dim=1).values  # (N, 1)
        elif mode == "meanmax":
            X2 = torch.cat([Z.mean(dim=1), Z.max(dim=1).values], dim=1)  # (N, 2)
        else:
            raise ValueError("mode must be one of: flatten | mean | max | meanmax")

        self.last_compressed_ = X2.detach()
        return X2

    @torch.no_grad()
    def compress_numpy(self, X: torch.Tensor, mode: str = "flatten", dtype="float32"):
        import numpy as np
        X2 = self.compress(X, mode=mode).detach().cpu().numpy()
        X2 = X2.astype(np.float32 if dtype == "float32" else np.float64, copy=False)
        self.last_compressed_np_ = X2
        return X2

    @torch.no_grad()
    def compress_dataframe(
        self,
        X: torch.Tensor,
        mode: str = "flatten",
        y=None,
        feature_prefix: str = "nsc",
        target_name: str = "target",
        dtype="float32",
    ):
        import numpy as np
        import pandas as pd

        X_np = self.compress_numpy(X, mode=mode, dtype=dtype)
        cols = [f"{feature_prefix}_{i}" for i in range(X_np.shape[1])]
        df = pd.DataFrame(X_np, columns=cols)

        if y is not None:
            y_np = np.asarray(y).reshape(-1)
            if y_np.shape[0] != X_np.shape[0]:
                raise ValueError(f"y length {y_np.shape[0]} != N {X_np.shape[0]}")
            df[target_name] = y_np

        self.last_compressed_df_ = df
        return df

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.transform(X)


# Friendly alias for package export
NSC_P = NSC_PCA


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


def run_nsc_p_csv(
    csv_path: str,
    target_col: Optional[str] = None,
    ordering_csv: Optional[str] = None,
    dataset_name: Optional[str] = None,

    d_hat: Optional[float] = None,
    M: Optional[int] = None,

    gamma: float = 2.0,
    M_min: int = 32,
    M_max: int = 512,
    bypass_if_m_leq: int = 400,
    segmentation: str = "uniform",
    l_min: int = 8,
    descriptor: str = "basic",
    pooling: str = "learn_free",
    mlp_hidden: int = 64,
    mlp_depth: int = 2,
    dropout: float = 0.0,
    m_rule: str = "default",
    beta: float = 0.5,
    tau: float = 0.99,

    standardize_input: bool = True,
    drop_non_numeric: bool = True,
    mode: str = "flatten",
    device: Optional[str] = None,

    save_outputs: bool = True,
    out_prefix: Optional[str] = None,
):
    """
    Run NSC-P compression on a CSV dataset.

    NSC-P can estimate d_hat internally from X_train if neither d_hat nor M is given.

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
        out_prefix = f"{safe_name}_nsc_p"

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

    model = NSC_PCA(
        gamma=gamma,
        M_min=M_min,
        M_max=M_max,
        bypass_if_m_leq=bypass_if_m_leq,
        segmentation=segmentation,
        l_min=l_min,
        descriptor=descriptor,
        pooling=pooling,
        mlp_hidden=mlp_hidden,
        mlp_depth=mlp_depth,
        dropout=dropout,
        m_rule=m_rule,
        beta=beta,
        tau=tau,
        assume_standardized=True,
        device=device,
    )

    print(f"[NSC-P] Dataset: {dataset_name}")
    print(f"[NSC-P] X shape: {X.shape}")
    print(f"[NSC-P] ordering: {'identity' if ordering_csv is None else ordering_csv}")
    print(
        f"[NSC-P] segmentation={segmentation}, m_rule={m_rule}, "
        f"descriptor={descriptor}, pooling={pooling}, tau={tau}"
    )

    t0 = time.perf_counter()

    X_t = torch.as_tensor(X, dtype=torch.float32)

    model.configure(
        Pi_star=Pi_star,
        X_train=X_t,
        d_hat=d_hat,
        M=M,
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
        "descriptor": descriptor,
        "pooling": pooling,
        "m_rule": m_rule,
        "tau": float(tau),
        "d_hat_given": None if d_hat is None else float(d_hat),
        "M_given": None if M is None else int(M),
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

    print("\n================ NSC-P SUMMARY ================")
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
        description="Run NSC-P compression on a numeric CSV dataset."
    )

    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--ordering-csv", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)

    parser.add_argument("--d-hat", type=float, default=None)
    parser.add_argument("--M", type=int, default=None)

    parser.add_argument("--segmentation", type=str, default="uniform",
                        choices=["uniform", "largest_jump", "equal_mass"])
    parser.add_argument("--m-rule", type=str, default="default",
                        choices=["gamma", "default", "idf"])

    parser.add_argument("--descriptor", type=str, default="basic",
                        choices=["basic", "robust", "moments", "all"])
    parser.add_argument("--pooling", type=str, default="learn_free",
                        choices=["learn_free", "linear", "mlp"])

    parser.add_argument("--tau", type=float, default=0.99)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--M-min", type=int, default=32)
    parser.add_argument("--M-max", type=int, default=512)
    parser.add_argument("--l-min", type=int, default=8)
    parser.add_argument("--bypass-if-m-leq", type=int, default=400)

    parser.add_argument("--mlp-hidden", type=int, default=64)
    parser.add_argument("--mlp-depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--mode", type=str, default="flatten",
                        choices=["flatten", "mean", "max", "meanmax"])
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--keep-non-numeric", action="store_true")

    parser.add_argument("--out-prefix", type=str, default=None)

    args = parser.parse_args()

    run_nsc_p_csv(
        csv_path=args.csv,
        target_col=args.target,
        ordering_csv=args.ordering_csv,
        dataset_name=args.dataset,
        d_hat=args.d_hat,
        M=args.M,
        gamma=args.gamma,
        M_min=args.M_min,
        M_max=args.M_max,
        bypass_if_m_leq=args.bypass_if_m_leq,
        segmentation=args.segmentation,
        l_min=args.l_min,
        descriptor=args.descriptor,
        pooling=args.pooling,
        mlp_hidden=args.mlp_hidden,
        mlp_depth=args.mlp_depth,
        dropout=args.dropout,
        m_rule=args.m_rule,
        beta=args.beta,
        tau=args.tau,
        standardize_input=not args.no_standardize,
        drop_non_numeric=not args.keep_non_numeric,
        mode=args.mode,
        device=args.device,
        save_outputs=True,
        out_prefix=args.out_prefix,
    )


if __name__ == "__main__":
    main()