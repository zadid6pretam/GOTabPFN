# ============================================================
# GO-LR.py
#
# Standalone Graph-guided Ordering with Local Refinement (GO-LR)
# utility for GOTabPFN.
#
# This file keeps the original GO-LR algorithm unchanged:
#   sample clustering -> local feature graphs -> NNPath init
#   -> direction selection -> adjacent-swap refinement
#   -> global rank integration
#
# Added utilities:
#   - Load CSV input
#   - Drop target column if given
#   - Drop non-numeric columns if needed
#   - Return/save reordered feature table
#   - Report ordering runtime
#   - Report TSP path cost
#   - Report MinLA / dispersion cost
#
# CLI example:
#   python GO-LR.py \
#       --csv coloncancer_encoded.csv \
#       --target label \
#       --dataset Colon \
#       --metric correlation \
#       --num-clusters 7 \
#       --refine-passes 1 \
#       --out-prefix colon_golr
#
# Python/package example:
#   from gotabpfn import run_golr_csv
#
#   result = run_golr_csv(
#       csv_path="coloncancer_encoded.csv",
#       target_col="label",
#       dataset_name="Colon",
#       metric="correlation",
#       num_clusters=7,
#   )
#
#   print(result["metrics"])
#   result["reordered_df"].to_csv("colon_reordered.csv", index=False)
# ============================================================

from __future__ import annotations

import os
import gc
import json
import time
import random
import argparse
from typing import Optional

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from sklearn.cluster import KMeans as KMeansCPU
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------
# Optional GPU KMeans import
# ------------------------------------------------------------
try:
    from kmeans_gpu import KMeans as KMeansGPU
    _HAS_KMEANS_GPU = True
except Exception:
    KMeansGPU = None
    _HAS_KMEANS_GPU = False


# ============================================================
# Original GO-LR class
# ============================================================

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
        if torch.cuda.is_available():
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
        Z = X_cpu.t().contiguous()
        return torch.cdist(Z, Z, p=2)

    def _pairwise_manhattan(self, X_cpu):
        Z = X_cpu.t().contiguous()
        return torch.cdist(Z, Z, p=1)

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
        D_G(pi) = sum_{i<j} w_ij * |pos_i - pos_j|.
        Implemented on CPU with O(D^2) compute, O(D) extra memory.
        """
        D = len(order)
        device = G.device
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
        device = G.device

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
            left_term = (
                G[u, left].to(torch.float32)
                - G[v, left].to(torch.float32)
            ).sum()
        else:
            left_term = torch.zeros((), device=G.device, dtype=torch.float32)

        if t + 2 < order_t.numel():
            right = order_t[t + 2:]
            right_term = (
                G[u, right].to(torch.float32)
                - G[v, right].to(torch.float32)
            ).sum()
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

        device = G.device
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

    # ---------- Cluster distances + integration ----------
    def _calculate_inter_cluster_distances(self, centroids):
        C = torch.stack(centroids, dim=0)
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

        X = torch.as_tensor(X_train, dtype=torch.float32, device=self.device)
        N, D = X.shape

        k = min(int(self.num_clusters), int(N))
        if k < 1:
            raise ValueError("num_clusters must be at least 1.")
        self.num_clusters = k

        # -----------------------
        # Step 1: Cluster Samples
        # -----------------------
        if use_cpu_kmeans or (self.device == 'cpu') or (not _HAS_KMEANS_GPU):
            labels = KMeansCPU(
                n_clusters=self.num_clusters,
                random_state=seed,
                n_init=10,
            ).fit_predict(X.cpu().numpy())

            cluster_labels = torch.tensor(labels, device=self.device)

            centroids = torch.stack(
                [
                    X[cluster_labels == i].mean(dim=0)
                    for i in range(self.num_clusters)
                    if torch.any(cluster_labels == i)
                ],
                dim=0,
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
            centroids = centroids_b[0]

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
            cluster_data = X[cluster_labels == i]

            if cluster_data.numel() == 0:
                continue

            centroids_list.append(cluster_data.mean(dim=0))

            G = self._construct_graph(cluster_data)
            graphs.append(G)

            init_order = self._minimize_dispersion(G)
            ordered_features = self._refine_order(
                G,
                init_order,
                passes=self.refine_passes,
            )

            local_orderings.append(ordered_features)

        if len(local_orderings) == 0:
            raise RuntimeError("GO-LR produced no local orderings.")

        cluster_distances, _ = self._calculate_inter_cluster_distances(centroids_list)
        global_ordering = self._integrate_orderings(local_orderings, cluster_distances)
        centroids_out = [c for c in centroids_list]

        return global_ordering, local_orderings, graphs, centroids_out


# ============================================================
# Minimal CSV / reporting wrapper
# ============================================================

def _golr_load_csv(
    csv_path,
    target_col=None,
    standardize=True,
    drop_non_numeric=True,
):
    df = pd.read_csv(csv_path)

    if target_col is not None:
        if target_col not in df.columns:
            raise ValueError(
                f"Target column '{target_col}' not found. "
                f"Available columns: {list(df.columns)}"
            )
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
            raise ValueError(f"Non-numeric columns found: {non_numeric_cols}")

    feature_names = list(df.columns)

    if len(feature_names) < 2:
        raise ValueError(
            f"Expected at least two numeric feature columns. Got {len(feature_names)}."
        )

    X = df.to_numpy(dtype=np.float32)
    X = np.nan_to_num(
        X,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    if standardize:
        X = StandardScaler().fit_transform(X).astype(np.float32)

    return X, feature_names


def _golr_tsp_path_cost(G, order):
    """
    TSP-style Hamiltonian path cost:
        sum_t G[order[t], order[t+1]]
    """
    total = 0.0

    for i in range(len(order) - 1):
        total += float(G[order[i], order[i + 1]].item())

    return float(total)


def _golr_make_ordering_dataframe(ordering, feature_names):
    ordered_feature_names = [feature_names[i] for i in ordering]

    return pd.DataFrame({
        "rank": np.arange(1, len(ordering) + 1),
        "feature_index": ordering,
        "feature_name": ordered_feature_names,
    })


def _golr_cleanup_cuda():
    gc.collect()

    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()


def run_golr_csv(
    csv_path,
    target_col=None,
    dataset_name=None,
    metric="correlation",
    num_clusters=7,
    refine=True,
    direction_select=True,
    refine_passes=1,
    bins=32,
    seed=42,
    standardize=True,
    drop_non_numeric=True,
    use_cpu_kmeans=True,
    save_outputs=True,
    out_prefix=None,
):
    """
    Run GO-LR on a CSV file.

    Returns a dict with:
        - ordering
        - ordered_feature_names
        - reordered_df
        - ordering_df
        - runtime_sec
        - tsp_path_cost
        - minla_cost
        - local_orderings
        - metrics
    """
    if dataset_name is None:
        dataset_name = os.path.splitext(os.path.basename(csv_path))[0]

    if out_prefix is None:
        safe_name = dataset_name.replace(" ", "_").replace("/", "_")
        out_prefix = f"{safe_name}_golr"

    X, feature_names = _golr_load_csv(
        csv_path=csv_path,
        target_col=target_col,
        standardize=standardize,
        drop_non_numeric=drop_non_numeric,
    )

    print(f"[GO-LR] Dataset: {dataset_name}")
    print(f"[GO-LR] X shape: {X.shape}")
    print(
        f"[GO-LR] metric={metric}, num_clusters={num_clusters}, "
        f"refine={refine}, direction_select={direction_select}, "
        f"refine_passes={refine_passes}"
    )

    model = GraphFeatureOrdering(
        num_clusters=num_clusters,
        metric=metric,
        bins=bins,
        refine=refine,
        direction_select=direction_select,
        refine_passes=refine_passes,
    )

    t0 = time.perf_counter()

    ordering, local_orderings, graphs, centroids = model.fit(
        X,
        seed=seed,
        deterministic=True,
        use_cpu_kmeans=use_cpu_kmeans,
    )

    runtime_sec = float(time.perf_counter() - t0)

    # Global graph is used only for reporting TSP and MinLA costs.
    # This does not change the learned ordering.
    X_tensor = torch.as_tensor(X, dtype=torch.float32, device=model.device)
    G_global = model._construct_graph(X_tensor)

    tsp_cost = _golr_tsp_path_cost(G_global, ordering)
    minla = float(model._dispersion_cost(G_global, ordering).item())

    ordered_feature_names = [feature_names[i] for i in ordering]
    reordered_df = pd.DataFrame(X[:, ordering], columns=ordered_feature_names)
    ordering_df = _golr_make_ordering_dataframe(ordering, feature_names)

    metrics = {
        "dataset": dataset_name,
        "n": int(X.shape[0]),
        "m": int(X.shape[1]),
        "metric": metric,
        "num_clusters": int(model.num_clusters),
        "refine": bool(refine),
        "direction_select": bool(direction_select),
        "refine_passes": int(refine_passes),
        "runtime_sec": runtime_sec,
        "tsp_path_cost": tsp_cost,
        "minla_cost": minla,
    }

    result = {
        "ordering": ordering,
        "ordered_feature_names": ordered_feature_names,
        "reordered_df": reordered_df,
        "ordering_df": ordering_df,
        "runtime_sec": runtime_sec,
        "tsp_path_cost": tsp_cost,
        "minla_cost": minla,
        "local_orderings": local_orderings,
        "graphs": graphs,
        "centroids": centroids,
        "metrics": metrics,
    }

    print("\n================ GO-LR SUMMARY ================")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    if save_outputs:
        reordered_path = f"{out_prefix}_reordered.csv"
        ordering_path = f"{out_prefix}_ordering.csv"
        metrics_path = f"{out_prefix}_metrics.json"

        reordered_df.to_csv(reordered_path, index=False)
        ordering_df.to_csv(ordering_path, index=False)

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print("\n[SAVED]")
        print(f"  - {reordered_path}")
        print(f"  - {ordering_path}")
        print(f"  - {metrics_path}")

    _golr_cleanup_cuda()

    return result


# Backward-friendly alias for package use
GOLRFeatureOrdering = GraphFeatureOrdering


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run GO-LR feature ordering on a CSV dataset and report "
            "runtime, TSP path cost, and MinLA cost."
        )
    )

    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)

    parser.add_argument("--metric", type=str, default="correlation")
    parser.add_argument("--num-clusters", type=int, default=7)
    parser.add_argument("--refine-passes", type=int, default=1)
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--out-prefix", type=str, default=None)

    parser.add_argument("--no-refine", action="store_true")
    parser.add_argument("--no-direction-select", action="store_true")
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--keep-non-numeric", action="store_true")
    parser.add_argument("--gpu-kmeans", action="store_true")

    args = parser.parse_args()

    run_golr_csv(
        csv_path=args.csv,
        target_col=args.target,
        dataset_name=args.dataset,
        metric=args.metric,
        num_clusters=args.num_clusters,
        refine=not args.no_refine,
        direction_select=not args.no_direction_select,
        refine_passes=args.refine_passes,
        bins=args.bins,
        seed=args.seed,
        standardize=not args.no_standardize,
        drop_non_numeric=not args.keep_non_numeric,
        use_cpu_kmeans=not args.gpu_kmeans,
        save_outputs=True,
        out_prefix=args.out_prefix,
    )


if __name__ == "__main__":
    main()