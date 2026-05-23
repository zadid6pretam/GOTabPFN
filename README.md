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
  title     = {{GOTabPFN: From Feature Ordering to Compact Tokenization for Tabular Foundation Models on High-Dimensional Data}},
  author    = {Habib, Al Zadid Sultan Bin and Ahamed, Md Younus and Gyawali, Prashnna and Doretto, Gianfranco and Adjeroh, Donald A.},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026}
}
```

- Find it on ICML portal: https://icml.cc/virtual/2026/poster/62523


## Files and Repository Structure

### Python package: `gotabpfn/`

This folder contains the core GOTabPFN implementation and standalone utility modules:

- `__init__.py` - Package initializer and high-level API exports.
- `gotabpfn.py` - Main GOTabPFN implementation, including:
  - `GraphFeatureOrdering` for graph-guided feature ordering.
  - `pidf_segpca` / NSC-pSP for PCA-IDF-aware segment-wise compression.
  - `TabPFN25Head` and `TabPFN25Config` for using a frozen TabPFN-2.5 classifier/regressor head.
  - End-to-end components for feature ordering, compact tokenization, and TabPFN-based prediction.
- `GO-LR.py` - Standalone Graph-guided Ordering with Local Refinement (GO-LR) module. It can be used independently as a feature-ordering/metaheuristic algorithm and reports ordering runtime, TSP path cost, MinLA cost, learned ordering, and reordered feature tables.
- `NSC-pSP.py` - Standalone NSC-pSP compression module: PCA-IDF-aware segment-wise principal subspace projection.
- `NSC-SP.py` - Standalone NSC-SP compression module: segment-wise principal subspace projection with user-provided `M` or `d_hat`.
- `NSC-P.py` - Standalone NSC-P compression module: PCA-IDF-aware descriptor/statistics-based compression.
- `NSC.py` - Standalone original NSC descriptor/statistics-based compression module.
- `gotabpfn_dataset_diagnostics.py` - Dataset-level diagnostics for IDF/FOE/`P_success`, locality gains, LES, and AUC under the cumulative explained variance-IDF curve.

### Experiment notebooks: `GOTabPFN Experiments/`

This folder contains experiment notebooks used during the initial submission and rebuttal/ablation period. Some notebooks may reflect earlier package/module names or earlier experimental scripts, but they are retained for reproducibility and transparency. Some notebooks contain full Optuna tuning scripts, while others provide fixed-run scripts using the best GO-LR and NSC hyperparameters found after Optuna search.

Representative notebooks include:

- **`GOLR_NSC_TabICL_Colon.ipynb`** and **`GOLR_NSC_TabICL_Lung.ipynb`**  
  Experiments combining GO-LR ordering and NSC compression with TabICL-style evaluation baselines.
- **`GOTabPFN_Colon_exp.ipynb`**, **`GOTabPFN_Lung.ipynb`**, **`GOTabPFN_ALLAML.ipynb`**, **`GOTabPFN_Arcene.ipynb`**, **`GOTabPFN_SMK.ipynb`**, **`GOTabPFN_TOX.ipynb`**  
  Main HDLSS dataset experiments for GOTabPFN.
- **`GOTabPFN_BASEHOCK.ipynb`**, **`GOTabPFN_RELATHE.ipynb`**, **`GOTabPFN_Cell_Cycle.ipynb`**  
  Cross-domain tabular experiments.
- **`GOTabPFN_DrivFace_Classification.ipynb`**  
  DrivFace classification experiment.
- **`GOTabPFN_Colon_AUC_F1.ipynb`**  
  Additional AUC/F1 evaluation for Colon.
- **`GOTabPFN_ClusterSizeAblation.ipynb`**  
  Cluster-size sensitivity/ablation experiments.
- **`GOTabPFN_Seed_Sensitivity.ipynb`**  
  Seed-sensitivity analysis.

### Package test notebook

- **`GOTabPFN_Package_Test.ipynb`**  
  Tests the local package setup. This notebook checks package imports, GO-LR as a standalone metaheuristic ordering module, the four NSC compression variants, and small binary/multiclass/regression runs on a separate local machine.


- **`GOTabPFN_PIP_Install_Check.ipynb`**  
  Minimal notebook for checking the installed `gotabpfn` package after `pip install`. It will verify imports, initialize core modules, and run a toy workflow.

### Main dependencies

The repository uses the following main dependencies:

```txt
numpy>=1.23
pandas>=1.5
scipy>=1.11
scikit-learn>=1.2
tqdm>=4.64
optuna>=3.5
torch>=2.1
tabpfn==6.3.1
kmeans-gpu==0.0.5
matplotlib>=3.7
```

### Other top-level files

- **`requirements.txt`** - Python dependencies required to run the GOTabPFN package and notebooks.
- **`GOTabPFN_Architecture.png`** - High-level architecture diagram of the GOTabPFN framework.
- **`GOTabPFN_Package_Test.ipynb`** - Local package test notebook covering imports, GO-LR, NSC variants, and small binary/multiclass/regression runs.
- **`LICENSE`** - MIT license for this repository.
- **`README.md`** - Project overview, installation, usage instructions, repository structure, and citation information.
- **`.gitignore`** - Standard Git ignore rules for Python, Jupyter, cache files, checkpoints, and experiment outputs.
- **`pyproject.toml`** - Modern Python build-system and package metadata file for installation and PyPI upload.
- **`setup.cfg`** - Optional setuptools configuration file for package metadata and installation settings, if used alongside `pyproject.toml`.


### Tested Environment

The package has been tested primarily with:

- Python 3.10+
- numpy 1.23+
- pandas 1.5+
- scipy 1.11+
- scikit-learn 1.2+
- tqdm 4.64+
- optuna 3.5+
- torch 2.1+
- tabpfn 6.3.1
- kmeans-gpu 0.0.5
- matplotlib 3.7+
- jupyterlab 4.0+

Additional diagnostics, package tests, and fixed-parameter runs were executed on a separate local machine with an 8× NVIDIA RTX A6000 GPU cluster. Small numerical/runtime differences from the main paper results may therefore be observed depending on hardware configurations. On the first run, TabPFN may download the required TabPFN-2.5 checkpoint from Hugging Face; the checkpoint is cached afterward.

