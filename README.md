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

## Overview

**GOTabPFN** is designed for high-dimensional tabular datasets where standard TabPFN-style predictors become difficult to use directly due to large feature counts. It takes a raw feature matrix and labels, learns an ordered and compressed representation, and then performs prediction using a **frozen TabPFN-2.5 head**.

The pipeline has three main stages:

1. **GO-LR** learns a global feature order `Pi*` from cluster-wise feature graphs.
2. **NSC** segments the ordered feature axis and compresses each segment into a compact meta-feature.
3. **TabPFN-2.5** performs prediction on the compressed token vector `Z(x)`.

GOTabPFN is not limited to biomedical datasets. It can be applied to any high-dimensional tabular dataset with a compatible feature matrix and target labels, including biomedical, transcriptomic, text, sensor, and image-feature datasets.

In the architecture diagram, the 'Feature Clustering' block is a high-level visual shorthand for discovering local feature-dependence groups through cluster-wise feature graphs. The final output of GO-LR is a single global feature order, which NSC uses to produce compact tokens for frozen TabPFN-style inference.

## Citation

Al Zadid Sultan Bin Habib, Md Younus Ahamed, Prashnna Gyawali, Gianfranco Doretto, and Donald A. Adjeroh. **“GOTabPFN: From Feature Ordering to Compact Tokenization for Tabular Foundation Models on High-Dimensional Data.”** In *Proceedings of the 43rd International Conference on Machine Learning (ICML)*, 2026.

BibTeX:
```bibtex
@inproceedings{habib2026gotabpfn,
  title     = {GOTabPFN: From Feature Ordering to Compact Tokenization for Tabular Foundation Models on High-Dimensional Data},
  author    = {Habib, Al Zadid Sultan Bin and Ahamed, Md Younus and Gyawali, Prashnna and Doretto, Gianfranco and Adjeroh, Donald A.},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026}
}
```

