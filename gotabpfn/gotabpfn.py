import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data import DataLoader, TensorDataset
from kmeans_gpu import KMeans as KMeansGPU
from sklearn.cluster import KMeans as KMeansCPU

class GraphFeatureOrdering:
    def __init__(
        self,
        num_clusters=7,
        metric='kl_divergence',
        order='ascending',
        bins=32,
        refine=True,
        direction_select=True,
        refine_passes=1,
    ):
        self.num_clusters = num_clusters
        self.metric = metric
        self.order = order
        self.bins = bins

        # GPU only for KMeans; graphs/refinement will be on CPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Local refinement controls
        self.refine = refine
        self.direction_select = direction_select
        self.refine_passes = refine_passes

    # ---------- Global Seeding ----------
    def _set_seed(self, seed=42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)

    # ---------- Utility ----------
    def _to_1d(self, x, dtype=None):
        t = torch.as_tensor(x)
        if dtype:
            t = t.to(dtype)
        return t.flatten().to(self.device)

    def _discretize(self, x, bins=32):
        xmin, xmax = x.min(), x.max()
        if (xmax - xmin) < 1e-12:
            return torch.zeros_like(x, dtype=torch.long)
        edges = torch.linspace(xmin, xmax, bins + 1, device=x.device)
        return (torch.bucketize(x, edges) - 1).clamp(0, bins - 1).long()

    # ---------- Distance Metrics (CPU) ----------
    def _pairwise_euclidean(self, X_cpu):
        Z = X_cpu.t().contiguous()            # (D, n_cluster_samples)
        return torch.cdist(Z, Z, p=2)         # (D, D) on CPU

    def _pairwise_manhattan(self, X_cpu):
        Z = X_cpu.t().contiguous()
        return torch.cdist(Z, Z, p=1)         # (D, D) on CPU

    def _pairwise_cosine(self, X_cpu):
        Z = F.normalize(X_cpu.t().contiguous(), dim=1)
        sim = Z @ Z.t()
        return 1.0 - sim.clamp(-1, 1)

    def _pairwise_correlation(self, X_cpu):
        N, D = X_cpu.shape
        Xm = X_cpu - X_cpu.mean(dim=0, keepdim=True)
        std = Xm.std(dim=0, unbiased=False).clamp_min(1e-12)
        Z = Xm / std
        C = (Z.T @ Z) / float(N)
        Dcorr = 1.0 - C.abs()
        Dcorr.fill_diagonal_(0.0)
        return Dcorr

    # ---------- Histograms + Divergences (CPU) ----------
    @torch.no_grad()
    def _histograms_shared_bins(self, X_cpu):
        xmin = X_cpu.min()
        xmax = X_cpu.max()
        if (xmax - xmin) < 1e-12:
            P = torch.zeros((X_cpu.shape[1], self.bins), device=X_cpu.device)
            P[:, 0] = 1.0
            return P
        edges = torch.linspace(xmin, xmax, self.bins + 1, device=X_cpu.device)
        idx = torch.bucketize(X_cpu, edges) - 1
        idx = idx.clamp(0, self.bins - 1)
        N, D = X_cpu.shape
        counts = torch.zeros((D, self.bins), device=X_cpu.device)
        ones = torch.ones_like(idx, dtype=torch.float32)
        counts.scatter_add_(1, idx.T, ones.T)
        return counts / counts.sum(dim=1, keepdim=True).clamp_min(1e-12)

    @torch.no_grad()
    def _kl_matrix(self, P_cpu):
        logP = torch.log(P_cpu.clamp_min(1e-12))
        Hself = (P_cpu * logP).sum(dim=1)
        XEnt = P_cpu @ logP.T
        K = Hself[:, None] - XEnt
        K.fill_diagonal_(0.0)
        return K

    # ---------- Graph Construction (always on CPU) ----------
    def _construct_graph(self, X_cluster):
        """
        Build G on CPU to avoid CUDA OOM. X_cluster may be on GPU; we move it.
        """
        X_cpu = X_cluster.detach().to('cpu')
        metric = self.metric.lower()

        if metric in ('euclidean', 'l2'):
            return self._pairwise_euclidean(X_cpu)
        elif metric in ('l1', 'manhattan', 'cityblock'):
            return self._pairwise_manhattan(X_cpu)
        elif metric in ('cosine', 'cos'):
            return self._pairwise_cosine(X_cpu)
        elif metric in ('correlation', 'corr', 'pearson'):
            return self._pairwise_correlation(X_cpu)
        elif metric in ('kl', 'kl_divergence'):
            P = self._histograms_shared_bins(X_cpu)
            return self._kl_matrix(P)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    # ---------- MinLA Dispersion (streaming, CPU) ----------
    @torch.no_grad()
    def _dispersion_cost(self, G: torch.Tensor, order: list[int]) -> torch.Tensor:
        """
        D_G(π) = sum_{i<j} w_ij * |pos_i - pos_j|
        Implemented on CPU with O(D^2) compute, O(D) extra memory.
        """
        D = len(order)
        device = G.device  # CPU
        order_t = torch.tensor(order, device=device, dtype=torch.long)

        pos = torch.empty(D, device=device, dtype=torch.float32)
        pos[order_t] = torch.arange(D, device=device, dtype=torch.float32)

        Gf = G.to(torch.float32)
        total = torch.tensor(0.0, device=device, dtype=torch.float32)

        for i in range(D - 1):
            diff = (pos[i] - pos[i + 1:]).abs()
            w_row = Gf[i, i + 1:]
            total += (w_row * diff).sum()

        return total

    # ---------- NNPath (TSP-style init, CPU) ----------
    def _minimize_dispersion(self, G):
        D = G.shape[0]
        device = G.device  # CPU
        row_sums = G.sum(1)
        start = int(torch.argmin(row_sums).item())

        visited = torch.zeros(D, dtype=torch.bool, device=device)
        ordering = [start]
        visited[start] = True

        for _ in range(D - 1):
            d = G[ordering[-1]].clone()
            d[visited] = float("inf")
            nxt = int(torch.argmin(d).item())
            ordering.append(nxt)
            visited[nxt] = True
        return ordering

    # ---------- Adjacent swap delta (CPU, no big tensors) ----------
    @torch.no_grad()
    def _adjacent_swap_delta(self, G: torch.Tensor, order_t: torch.Tensor, t: int) -> torch.Tensor:
        u = order_t[t]
        v = order_t[t + 1]

        if t > 0:
            left = order_t[:t]
            left_term = (G[u, left].to(torch.float32) -
                         G[v, left].to(torch.float32)).sum()
        else:
            left_term = torch.zeros((), device=G.device, dtype=torch.float32)

        if t + 2 < order_t.numel():
            right = order_t[t + 2:]
            right_term = (G[u, right].to(torch.float32) -
                          G[v, right].to(torch.float32)).sum()
        else:
            right_term = torch.zeros((), device=G.device, dtype=torch.float32)

        return left_term - right_term

    # ---------- Local refinement (CPU) ----------
    @torch.no_grad()
    def _refine_order(self, G: torch.Tensor, order: list[int], passes: int) -> list[int]:
        if (not self.refine) or passes <= 0:
            return order

        # Direction selection
        if self.direction_select:
            rev = list(reversed(order))
            if self._dispersion_cost(G, rev) < self._dispersion_cost(G, order):
                order = rev

        device = G.device  # CPU
        D = len(order)
        order_t = torch.tensor(order, device=device, dtype=torch.long)

        for _ in range(int(passes)):
            improved = False
            for t in range(D - 1):
                delta = self._adjacent_swap_delta(G, order_t, t)
                if delta < 0:
                    tmp = order_t[t].clone()
                    order_t[t] = order_t[t + 1]
                    order_t[t + 1] = tmp
                    improved = True
            if not improved:
                break

        return order_t.tolist()

    # ---------- Cluster distances + integration (tiny, can stay on GPU) ----------
    def _calculate_inter_cluster_distances(self, centroids):
        C = torch.stack(centroids, dim=0)  # centroids on self.device
        return torch.cdist(C, C, p=2), C

    def _integrate_orderings(self, local_orderings, cluster_distances):
        C = cluster_distances.shape[0]
        D = len(local_orderings[0])
        device = cluster_distances.device

        md = cluster_distances + torch.eye(C, device=device)
        w = 1.0 / md.mean(dim=1)
        w = w / (w.sum() + 1e-12)

        ranks = torch.empty((C, D), device=device)
        for c, order in enumerate(local_orderings):
            pos = torch.empty(D, device=device)
            pos[order] = torch.arange(D, device=device, dtype=torch.float32)
            ranks[c] = pos

        avg_rank = (w[:, None] * ranks).sum(0)
        return torch.argsort(avg_rank).tolist()

    # ---------- Main API ----------
    @torch.no_grad()
    def fit(self, X_train, seed=42, deterministic=True, use_cpu_kmeans=False):
        if deterministic:
            self._set_seed(seed)

        # X for clustering on self.device (GPU if available)
        X = torch.as_tensor(X_train, dtype=torch.float32, device=self.device)
        N, D = X.shape

        # -----------------------
        # Step 1: Cluster Samples
        # -----------------------
        if use_cpu_kmeans or (self.device == 'cpu'):
            labels = KMeansCPU(
                n_clusters=self.num_clusters,
                random_state=seed
            ).fit_predict(X.cpu().numpy())
            cluster_labels = torch.tensor(labels, device=self.device)
            centroids = torch.stack(
                [X[cluster_labels == i].mean(dim=0)
                 for i in range(self.num_clusters)],
                dim=0
            )
        else:
            points = X.unsqueeze(0).contiguous()
            features = X.t().unsqueeze(0).contiguous()
            kmeans = KMeansGPU(
                n_clusters=self.num_clusters,
                max_iter=100,
                tolerance=1e-4,
                distance='euclidean',
                sub_sampling=None,
                max_neighbors=15,
            )
            centroids_b, _ = kmeans(points, features)
            centroids = centroids_b[0]                     # (k, D) on GPU
            dists = torch.cdist(X, centroids, p=2)
            cluster_labels = torch.argmin(dists, dim=1)

        # -----------------------
        # Step 2: Graphs & Orders (CPU)
        # -----------------------
        graphs = []
        local_orderings = []
        centroids_list = []

        unique_labels = torch.unique(cluster_labels)
        for i in unique_labels.tolist():
            cluster_data = X[cluster_labels == i]          # on self.device
            if cluster_data.numel() == 0:
                continue
            centroids_list.append(cluster_data.mean(dim=0))  # keep on self.device

            G = self._construct_graph(cluster_data)       # G on CPU
            graphs.append(G)

            init_order = self._minimize_dispersion(G)     # CPU
            ordered_features = self._refine_order(
                G, init_order, passes=self.refine_passes
            )
            local_orderings.append(ordered_features)

        cluster_distances, _ = self._calculate_inter_cluster_distances(centroids_list)
        global_ordering = self._integrate_orderings(local_orderings, cluster_distances)
        centroids_out = [c for c in centroids_list]

        return global_ordering, local_orderings, graphs, centroids_out

import math
import torch
import torch.nn as nn

class pidf_segpca(nn.Module):
    """
    Variant v4: PCA-IDF + Segment-then-PCA tokens (no statistical descriptors).

    Steps:
      1) reorder features by Pi*
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

# ================================
# TabPFN-2.5 head (non-differentiable)
# - call .fit(Z_train, y_train)
# - forward(Z) returns logits for classification
# - forward(Z) returns prediction values for regression
# ================================

from dataclasses import dataclass
from typing import Optional, Literal
import numpy as np
import torch
import torch.nn as nn

TaskType = Literal["binary", "multiclass", "regression"]

# ---- TabPFN imports ----
from tabpfn import TabPFNClassifier

try:
    from tabpfn import TabPFNRegressor
    _HAVE_TABPFN_REGRESSOR = True
except Exception:
    TabPFNRegressor = None
    _HAVE_TABPFN_REGRESSOR = False

try:
    from tabpfn.constants import ModelVersion
    _HAVE_MODELVERSION = True
except Exception:
    _HAVE_MODELVERSION = False


def _make_tabpfn_25(device: str = "cuda"):
    """
    Create TabPFN v2.5 classifier if supported by your installed tabpfn.
    """
    if _HAVE_MODELVERSION:
        try:
            clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2_5)
            try:
                clf.set_params(device=device)
            except Exception:
                pass
            return clf
        except Exception:
            pass

    # fallback
    try:
        return TabPFNClassifier(device=device)
    except TypeError:
        return TabPFNClassifier()


def _make_tabpfn_25_regressor(device: str = "cuda"):
    """
    Create TabPFN v2.5 regressor if supported by your installed tabpfn.
    """
    if not _HAVE_TABPFN_REGRESSOR:
        raise ImportError(
            "TabPFNRegressor is not available in your installed tabpfn package. "
            "Please install a tabpfn version that supports TabPFNRegressor."
        )

    if _HAVE_MODELVERSION:
        try:
            reg = TabPFNRegressor.create_default_for_version(ModelVersion.V2_5)
            try:
                reg.set_params(device=device)
            except Exception:
                pass
            return reg
        except Exception:
            pass

    # fallback
    try:
        return TabPFNRegressor(device=device)
    except TypeError:
        return TabPFNRegressor()


def _to_numpy_2d(Z: torch.Tensor) -> np.ndarray:
    """
    Accepts Z as:
      (B,M) or (B,M,1) or (B,M,d_in)
    Returns float32 numpy (B, D) with flatten if needed.
    """
    if isinstance(Z, np.ndarray):
        Znp = Z
    else:
        if not torch.is_tensor(Z):
            Z = torch.as_tensor(Z)
        Z = Z.detach()

        # (B,M,1) -> (B,M)
        if Z.dim() == 3 and Z.shape[-1] == 1:
            Z = Z.squeeze(-1)

        # if still 3D, flatten last dims into features
        if Z.dim() == 3:
            B, M, D = Z.shape
            Z = Z.reshape(B, M * D)

        if Z.dim() != 2:
            raise ValueError(f"Z must be 2D (B,D) or 3D (B,M,d). Got {tuple(Z.shape)}")

        Znp = Z.cpu().numpy()

    Znp = np.asarray(Znp, dtype=np.float32)
    if Znp.ndim != 2:
        raise ValueError(f"Z must be 2D after conversion. Got {Znp.shape}")
    return Znp


def _ensure_labels(y, task_type: str):
    y = np.asarray(y).reshape(-1)
    task_type = task_type.lower()

    if task_type in ("binary", "multiclass"):
        # allow {-1,+1} -> {0,1}
        u = set(np.unique(y).tolist())
        if task_type == "binary" and u.issubset({-1, 1}):
            y = (y == 1).astype(np.int64)
        else:
            y = y.astype(np.int64, copy=False)
        return y

    if task_type == "regression":
        return y.astype(np.float32, copy=False)

    raise ValueError(f"Unknown task_type: {task_type}")


def _proba_to_logits_binary(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _proba_to_logits_multiclass(P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # logits can be log-probs; CrossEntropyLoss accepts that fine
    P = np.clip(P, eps, 1.0)
    return np.log(P)


@dataclass
class TabPFN25Config:
    task_type: TaskType = "binary"
    num_classes: int = 2
    device: Optional[str] = None
    random_state: int = 42


class TabPFN25Head(nn.Module):
    """
    A sklearn-style head wrapped as nn.Module:
      - NOT differentiable
      - Use per-fold: model.fit(Z_tr, y_tr)
      - classification: model(Z_va) -> logits
      - regression: model(Z_va) -> continuous predictions
    """

    def __init__(self, cfg: TabPFN25Config):
        super().__init__()
        self.cfg = cfg
        self.task_type = cfg.task_type.lower()
        self.num_classes = int(cfg.num_classes)

        if self.task_type == "multiclass" and self.num_classes < 2:
            raise ValueError("num_classes must be >=2 for multiclass.")

        if self.task_type not in ("binary", "multiclass", "regression"):
            raise ValueError(
                "task_type must be one of: 'binary', 'multiclass', 'regression'."
            )

        self.device_str = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")

        # classifier or regressor created at fit-time
        self.clf = None
        self._is_fitted = False

    def fit(self, Z_train, y_train):
        """
        Fit TabPFN on training data for this split.
        """
        Ztr = _to_numpy_2d(Z_train)
        ytr = _ensure_labels(y_train, self.task_type)

        if self.task_type == "regression":
            self.clf = _make_tabpfn_25_regressor(device=self.device_str)

            try:
                self.clf.set_params(random_state=int(self.cfg.random_state))
            except Exception:
                pass

            self.clf.fit(Ztr, ytr)
            self._is_fitted = True
            return self

        # Basic sanity for binary classification
        if self.task_type == "binary":
            u = set(np.unique(ytr).tolist())
            if not u.issubset({0, 1}):
                raise ValueError(f"Binary labels must be in {{0,1}} or {{-1,1}}. Got {u}")

        self.clf = _make_tabpfn_25(device=self.device_str)

        try:
            self.clf.set_params(random_state=int(self.cfg.random_state))
        except Exception:
            pass

        self.clf.fit(Ztr, ytr)
        self._is_fitted = True
        return self

    @torch.no_grad()
    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Returns:
          - binary:     (B,1) logits
          - multiclass: (B,C) logits
          - regression: (B,1) continuous predictions
        """
        if not self._is_fitted or self.clf is None:
            raise RuntimeError("TabPFN25Head is not fitted. Call .fit(Z_train, y_train) first.")

        Znp = _to_numpy_2d(Z)

        if self.task_type == "binary":
            prob = self.clf.predict_proba(Znp)[:, 1]
            logits = _proba_to_logits_binary(prob)[:, None]
            return torch.from_numpy(logits).float()

        if self.task_type == "multiclass":
            P = self.clf.predict_proba(Znp)
            logits = _proba_to_logits_multiclass(P)
            return torch.from_numpy(logits).float()

        if self.task_type == "regression":
            pred = self.clf.predict(Znp)
            pred = np.asarray(pred, dtype=np.float32).reshape(-1, 1)
            return torch.from_numpy(pred).float()

        raise ValueError(f"Unknown task_type: {self.task_type}")

    @torch.no_grad()
    def predict(self, Z):
        """
        Returns class labels for classification and continuous predictions for regression.
        """
        if not self._is_fitted or self.clf is None:
            raise RuntimeError("Call .fit(...) first.")

        Znp = _to_numpy_2d(Z)
        return self.clf.predict(Znp)

    @torch.no_grad()
    def predict_proba(self, Z):
        """
        Classification only.
        """
        if not self._is_fitted or self.clf is None:
            raise RuntimeError("Call .fit(...) first.")

        if self.task_type == "regression":
            raise RuntimeError("predict_proba is not defined for regression. Use predict(...) instead.")

        Znp = _to_numpy_2d(Z)
        return self.clf.predict_proba(Znp)

# ============================================================
# CSV-level GOTabPFN convenience wrapper
# ============================================================

import gc
import random
import time
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    RepeatedKFold,
    StratifiedKFold,
    KFold,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)


def _seed_everything_for_wrapper(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cleanup_cuda_for_wrapper():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _compute_deltas_adjacent_corr_for_wrapper(X_tr, Pi_star, eps=1e-12):
    """
    Compute adjacent transition scores along the GO-LR order:
        delta_t = 1 - |corr(feature_t, feature_{t+1})|.

    Required for transition-aware NSC segmentation rules:
        - equal_mass
        - largest_jump
    """
    X_t = torch.from_numpy(np.asarray(X_tr, dtype=np.float32)).float()
    perm = torch.tensor(Pi_star, dtype=torch.long)

    Xp = X_t[:, perm]
    Xc = Xp - Xp.mean(dim=0, keepdim=True)
    std = Xc.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    Z = Xc / std

    corr_adj = (Z[:, :-1] * Z[:, 1:]).mean(dim=0)
    deltas = 1.0 - corr_adj.abs()

    return deltas.cpu()


def _safe_multiclass_macro_ovr_auc(y_true, proba, num_classes):
    try:
        y_bin = label_binarize(y_true, classes=np.arange(num_classes))
        return float(
            roc_auc_score(
                y_bin,
                proba,
                average="macro",
                multi_class="ovr",
            )
        )
    except Exception:
        return np.nan


def _preprocess_csv_for_gotabpfn(
    csv_path,
    target_col,
    task_type="classification",
    non_numeric="drop",
    standardize=True,
):
    """
    Preprocess CSV exactly following the manual examples.

    Classification:
      - y_raw = target as string, fill missing as "missing_target"
      - LabelEncoder
      - numeric features only by default
      - median imputation + 0 fallback
      - StandardScaler

    Regression:
      - target converted to numeric
      - missing target filled with target median, matching manual DrivFace script
      - numeric features only by default
      - median imputation + 0 fallback
      - StandardScaler
    """
    df = pd.read_csv(csv_path)

    if target_col not in df.columns:
        raise ValueError(f"target_col='{target_col}' was not found in the CSV file.")

    y_raw = df[target_col]
    X_df = df.drop(columns=[target_col])

    non_numeric = str(non_numeric).lower()
    if non_numeric == "drop":
        X_df = X_df.select_dtypes(include=[np.number])
    elif non_numeric == "encode":
        X_df = pd.get_dummies(X_df, dummy_na=True)
    else:
        raise ValueError("non_numeric must be one of: 'drop' or 'encode'.")

    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    X_df = X_df.fillna(X_df.median(numeric_only=True)).fillna(0.0)

    if X_df.shape[1] == 0:
        raise ValueError("No usable numeric feature columns found after preprocessing.")

    task_type = str(task_type).lower()

    if task_type in {"binary", "multiclass", "classification"}:
        y_raw = y_raw.astype(str).fillna("missing_target")

        le = LabelEncoder()
        y = le.fit_transform(y_raw).astype(np.int64)
        num_classes = len(le.classes_)

        if task_type == "classification":
            task_type_final = "binary" if num_classes == 2 else "multiclass"
        else:
            task_type_final = task_type

        if task_type_final == "binary" and num_classes != 2:
            raise ValueError(
                f"Binary classification expected 2 classes, found {num_classes}."
            )

        if task_type_final == "multiclass" and num_classes < 2:
            raise ValueError(
                f"Multiclass classification requires >=2 classes, found {num_classes}."
            )

        label_encoder = le

    elif task_type == "regression":
        y_raw = pd.to_numeric(y_raw, errors="coerce")
        y_raw = y_raw.fillna(y_raw.median())
        y = y_raw.to_numpy(dtype=np.float32)

        num_classes = None
        task_type_final = "regression"
        label_encoder = None

        if len(y) < 3:
            raise ValueError("Regression requires at least 3 target values.")

    else:
        raise ValueError(
            "task_type must be one of: 'classification', 'binary', 'multiclass', 'regression'."
        )

    X = X_df.to_numpy(dtype=np.float32)

    scaler = None
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)

    return X, y, X_df.columns.tolist(), label_encoder, scaler, task_type_final, num_classes


def run_gotabpfn_csv(
    csv_path,
    target_col,
    task_type="classification",
    non_numeric="drop",

    # Evaluation
    cv="5x5",
    test_size=0.30,  # kept for API compatibility; not used in manual-style CV
    seed=42,

    # GO-LR
    go_metric="euclidean",
    go_num_clusters=7,
    go_refine=True,
    go_direction_select=True,
    go_refine_passes=1,
    go_use_cpu_kmeans=False,
    go_fallback_cpu_kmeans=True,
    go_feat_subsample=None,

    # NSC-pSP
    nsc_segmentation="uniform",
    nsc_m_rule="default",
    nsc_tau=0.99,
    nsc_gamma=2.0,
    nsc_beta=0.5,
    nsc_M_min=32,
    nsc_M_max=512,
    nsc_l_min=8,
    assume_standardized=True,

    # TabPFN
    tabpfn_seed=42,
    device=None,

    # Output
    return_predictions=True,
    verbose=True,
):
    """
    Run the full GOTabPFN CSV pipeline:

        CSV -> preprocessing -> GO-LR ordering -> NSC-pSP compression -> TabPFN-2.5 head

    This wrapper is designed to match the manual implementation style:
      - full-feature GO-LR ordering once before CV
      - GPU KMeans attempt with CPU fallback
      - fold-wise NSC-pSP compression
      - fold-wise TabPFN-2.5 head
      - classification uses predict_proba(...) then argmax
      - regression cv="5x5" uses RepeatedKFold(5, 5)

    Returns
    -------
    dict with:
        - summary
        - metrics_df
        - predictions_df
        - ordering
        - ordered_feature_names
        - task_type
        - num_features
        - num_samples
        - elapsed_time_sec
    """
    _seed_everything_for_wrapper(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.perf_counter()

    X, y, feature_names, label_encoder, scaler, task_type_final, num_classes = (
        _preprocess_csv_for_gotabpfn(
            csv_path=csv_path,
            target_col=target_col,
            task_type=task_type,
            non_numeric=non_numeric,
            standardize=True,
        )
    )

    n, m = X.shape

    if verbose:
        if task_type_final in {"binary", "multiclass"}:
            print(f"X shape: {X.shape}, classes: {num_classes}")
        else:
            print(f"X shape: {X.shape}, y shape: {y.shape}")
        print(f"Using device: {device}")

    # -----------------------
# Learn GO-LR feature ordering once
# -----------------------
m_full = X.shape[1]

rng = np.random.default_rng(seed + 999)

if go_feat_subsample is not None and int(go_feat_subsample) < m_full:
    feat_idx = rng.choice(m_full, size=int(go_feat_subsample), replace=False)
    feat_idx.sort()
else:
    feat_idx = np.arange(m_full)

X_go = X[:, feat_idx]

go = GraphFeatureOrdering(
    num_clusters=go_num_clusters,
    metric=go_metric,
    refine=go_refine,
    direction_select=go_direction_select,
    refine_passes=go_refine_passes,
)

if go_fallback_cpu_kmeans and not go_use_cpu_kmeans:
    try:
        Pi_sub, local_orderings, graphs, centroids = go.fit(
            X_go,
            seed=seed,
            deterministic=True,
            use_cpu_kmeans=False,
        )
        golr_kmeans_mode = "gpu"
    except Exception:
        _cleanup_cuda_for_wrapper()
        Pi_sub, local_orderings, graphs, centroids = go.fit(
            X_go,
            seed=seed,
            deterministic=True,
            use_cpu_kmeans=True,
        )
        golr_kmeans_mode = "cpu_fallback"
else:
    Pi_sub, local_orderings, graphs, centroids = go.fit(
        X_go,
        seed=seed,
        deterministic=True,
        use_cpu_kmeans=go_use_cpu_kmeans,
    )
    golr_kmeans_mode = "cpu" if go_use_cpu_kmeans else "gpu"

ordered_subset = feat_idx[np.array(Pi_sub, dtype=np.int64)].tolist()

if len(feat_idx) < m_full:
    remaining = np.setdiff1d(np.arange(m_full), feat_idx, assume_unique=False)
    Pi_star = ordered_subset + remaining.tolist()
else:
    Pi_star = ordered_subset

Pi_star = list(map(int, Pi_star))
ordered_feature_names = [feature_names[i] for i in Pi_star]

    # -----------------------
    # CV setup
    # -----------------------
    cv = str(cv).lower()

    if task_type_final in {"binary", "multiclass"}:
        if cv == "5x5":
            splitter = RepeatedStratifiedKFold(
                n_splits=5,
                n_repeats=5,
                random_state=seed,
            )
            split_iter = splitter.split(X, y)
            eval_name = "5x5 repeated stratified CV"
        elif cv in {"3fold", "3-fold", "3"}:
            splitter = StratifiedKFold(
                n_splits=3,
                shuffle=True,
                random_state=seed,
            )
            split_iter = splitter.split(X, y)
            eval_name = "3-fold stratified CV"
        else:
            splitter = StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=seed,
            )
            split_iter = splitter.split(X, y)
            eval_name = "5-fold stratified CV"

        head_cfg = TabPFN25Config(
            task_type=task_type_final,
            num_classes=int(num_classes),
            device=device,
            random_state=int(tabpfn_seed),
        )

    else:
        if cv == "5x5":
            splitter = RepeatedKFold(
                n_splits=5,
                n_repeats=5,
                random_state=seed,
            )
            split_iter = splitter.split(X)
            eval_name = "5x5 repeated CV"
        elif cv in {"3fold", "3-fold", "3"}:
            splitter = KFold(
                n_splits=3,
                shuffle=True,
                random_state=seed,
            )
            split_iter = splitter.split(X)
            eval_name = "3-fold CV"
        else:
            splitter = KFold(
                n_splits=5,
                shuffle=True,
                random_state=seed,
            )
            split_iter = splitter.split(X)
            eval_name = "5-fold CV"

        head_cfg = TabPFN25Config(
            task_type="regression",
            num_classes=1,
            device=device,
            random_state=int(tabpfn_seed),
        )

    # -----------------------
    # Fold-wise NSC + TabPFN
    # -----------------------
    rows = []
    pred_rows = []

    for fold_id, (tr_idx, va_idx) in enumerate(split_iter, start=1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        nsc = pidf_segpca(
            segmentation=nsc_segmentation,
            l_min=int(nsc_l_min),
            m_rule=nsc_m_rule,
            gamma=float(nsc_gamma),
            beta=float(nsc_beta),
            tau=float(nsc_tau),
            M_min=int(nsc_M_min),
            M_max=int(nsc_M_max),
            assume_standardized=bool(assume_standardized),
            device=device,
        )

        deltas = None
        if str(nsc_segmentation).lower() in {"equal_mass", "largest_jump"}:
            deltas = _compute_deltas_adjacent_corr_for_wrapper(X_tr, Pi_star)

        X_tr_t = torch.from_numpy(X_tr)

        nsc.configure(
            Pi_star=Pi_star,
            X_train=X_tr_t,
            tau=float(nsc_tau),
            deltas=deltas,
        )

        Z_tr = nsc.compress(X_tr_t, mode="flatten").cpu().numpy()
        Z_va = nsc.compress(torch.from_numpy(X_va), mode="flatten").cpu().numpy()

        head = TabPFN25Head(head_cfg)
        head.fit(Z_tr, y_tr)

        if task_type_final in {"binary", "multiclass"}:
            P = head.predict_proba(Z_va)
            pred = np.argmax(P, axis=1)

            acc = float(accuracy_score(y_va, pred))
            f1m = float(f1_score(y_va, pred, average="macro", zero_division=0))

            row = {
                "fold": fold_id,
                "accuracy": acc,
                "macro_f1": f1m,
                "nsc_tokens": Z_tr.shape[1],
            }

            if task_type_final == "binary":
                try:
                    row["auc"] = float(roc_auc_score(y_va, P[:, 1]))
                except Exception:
                    row["auc"] = np.nan

                if verbose:
                    print(
                        f"Fold {fold_id:02d}: "
                        f"ACC={acc:.4f}, F1={f1m:.4f}, AUC={row['auc']:.4f}"
                    )

            else:
                aucm = _safe_multiclass_macro_ovr_auc(y_va, P, int(num_classes))
                row["macro_ovr_auc"] = aucm

                if verbose:
                    print(
                        f"Fold {fold_id:02d}: "
                        f"ACC={acc:.4f}, Macro-F1={f1m:.4f}, Macro-OvR-AUC={aucm:.4f}"
                    )

            rows.append(row)

            if return_predictions:
                true_labels = (
                    label_encoder.inverse_transform(y_va)
                    if label_encoder is not None else y_va
                )
                pred_labels = (
                    label_encoder.inverse_transform(pred)
                    if label_encoder is not None else pred
                )

                for idx, yt, yp in zip(va_idx, true_labels, pred_labels):
                    pred_rows.append({
                        "row_index": int(idx),
                        "fold": int(fold_id),
                        "true_label": yt,
                        "predicted_label": yp,
                    })

        else:
            pred = np.asarray(head.predict(Z_va), dtype=np.float32).reshape(-1)

            r2 = float(r2_score(y_va, pred))
            rmse = float(np.sqrt(mean_squared_error(y_va, pred)))
            mae = float(mean_absolute_error(y_va, pred))

            row = {
                "fold": fold_id,
                "r2": r2,
                "rmse": rmse,
                "mae": mae,
                "nsc_tokens": Z_tr.shape[1],
            }
            rows.append(row)

            if verbose:
                print(
                    f"Fold {fold_id:02d}: "
                    f"R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}"
                )

            if return_predictions:
                for idx, yt, yp in zip(va_idx, y_va, pred):
                    pred_rows.append({
                        "row_index": int(idx),
                        "fold": int(fold_id),
                        "true_value": float(yt),
                        "predicted_value": float(yp),
                    })

        _cleanup_cuda_for_wrapper()

    metrics_df = pd.DataFrame(rows)
    predictions_df = pd.DataFrame(pred_rows)
    elapsed_time_sec = time.perf_counter() - t0

    # -----------------------
    # Summary
    # -----------------------
    summary_lines = [
        "GOTabPFN completed successfully.",
        "",
        f"Task type: {task_type_final}",
        f"Evaluation: {eval_name}",
        f"Samples: {n}",
        f"Original numeric features: {m}",
        f"GO-LR metric: {go_metric}",
        f"GO-LR KMeans mode: {golr_kmeans_mode}",
        f"GO-LR feature subsample: {go_feat_subsample if go_feat_subsample is not None else 'full'}",
        f"NSC segmentation: {nsc_segmentation}",
        f"NSC M rule: {nsc_m_rule}",
        f"Device: {device}",
        f"Elapsed time: {elapsed_time_sec:.2f} seconds",
        "",
        "Metric summary:",
    ]

    for col in metrics_df.columns:
        if col == "fold":
            continue
        if pd.api.types.is_numeric_dtype(metrics_df[col]):
            values = metrics_df[col].dropna().to_numpy()
            if len(values) == 0:
                continue
            summary_lines.append(
                f"{col}: {np.mean(values):.4f} ± {np.std(values, ddof=1):.4f}"
            )

    if verbose:
        print("\nFinal CV results")
        for line in summary_lines:
            print(line)

    return {
        "summary": "\n".join(summary_lines),
        "metrics_df": metrics_df,
        "predictions_df": predictions_df,
        "ordering": Pi_star,
        "ordered_feature_names": ordered_feature_names,
        "task_type": task_type_final,
        "num_features": m,
        "num_samples": n,
        "elapsed_time_sec": elapsed_time_sec,
        "golr_kmeans_mode": golr_kmeans_mode,
    }
