## This repository is under construction now

# GOTabPFN: From Feature Ordering to Compact Tokenization for Tabular Foundation Models on High-Dimensional Data

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Task](https://img.shields.io/badge/Task-High--Dimensional%20Tabular%20Learning-orange)
![Model](https://img.shields.io/badge/Model-GOTabPFN-blueviolet)
![Method](https://img.shields.io/badge/Method-Graph--guided%20Feature%20Ordering-informational)
![Algorithm](https://img.shields.io/badge/Algorithm-GO--LR-purple)
![Compression](https://img.shields.io/badge/Compression-NSC--pSP-critical)
![Backbone](https://img.shields.io/badge/Backbone-TabPFN--2.5-teal)
![Optimization](https://img.shields.io/badge/Formulation-Column%20Permutation%20%2B%20Compression-9cf)
![Domain](https://img.shields.io/badge/Domain-Tabular%20Data-lightgrey)
[![Conference](https://img.shields.io/badge/Conference-ICML%202026%20Regular-blue)](https://icml.cc/virtual/2026/poster/62523)
![Status](https://img.shields.io/badge/Status-Accepted-brightgreen)
![PyPI](https://img.shields.io/badge/PyPI-gotabpfn-blue)

<p align="center">
  <img src="GOTabPFN_Architecture.png" alt="GOTabPFN Architecture" width="580">
</p>

<p align="center">
  <em>Overview of GOTabPFN: graph-guided feature ordering, NSC meta-feature construction, and frozen TabPFN-2.5 inference.</em>
</p>

GOTabPFN is a theory-grounded representation interface for making small **TabPFN-style tabular foundation models** effective in **High-Dimensional, Low-Sample Size (HDLSS)** regimes, where the number of features is much larger than the number of samples. The method combines **Graph-guided Ordering with Local Refinement (GO-LR)** and **Neuro-Inspired Subunit Compression (NSC)**. GO-LR builds cluster-wise feature graphs `G_c` from local sample contexts, treats features as graph nodes, and learns a single global feature order `Pi*` using a MinLA-grounded objective with TSP-path-style initialization and local refinement. In the architecture diagram, the **Feature Clustering** block is a high-level shorthand for discovering local feature-dependence groups through these cluster-wise feature graphs; it is not a separate prediction module. NSC then uses the learned order to segment adjacent features into contiguous neighborhoods and compress each segment into a scalar meta-feature, producing a compact token vector `Z(x) = (z_1, ..., z_M)`, where `M << m` and the token budget is tied to intrinsic dimensionality estimates rather than raw feature count. These tokens are passed to a **frozen TabPFN-2.5 head**, allowing GOTabPFN to improve high-dimensional compatibility without retraining or modifying the TabPFN backbone. This ordering-to-tokenization design is motivated by the observation that unordered HDLSS feature spaces often contain local redundancy and dependence structure that standard global compression or direct foundation-model inference may fail to exploit. Across 8 biomedical HDLSS benchmarks, GOTabPFN achieves the best accuracy on every dataset and an average rank of `1.00 ± 0.00` against 50+ baselines, while additional cross-domain experiments show the best average rank on 8 more high-dimensional datasets spanning text, face-image, image-feature, sensor, and RNA-seq domains. Overall, GOTabPFN provides a practical route to scalable in-context tabular prediction under tight feature budgets by turning very high-dimensional raw tables into stable, locality-preserving meta-feature representations for frozen TabPFN-style inference.
