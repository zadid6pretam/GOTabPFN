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
  Portion of the main HDLSS dataset experiments for GOTabPFN.
- **`GOTabPFN_BASEHOCK.ipynb`**, **`GOTabPFN_RELATHE.ipynb`**, **`GOTabPFN_Cell_Cycle.ipynb`**, **`GOTabPFN_DrivFace_Classification.ipynb`** 
  Portion of the cross-domain tabular experiments.
- **`GOTabPFN_Colon_AUC_F1.ipynb`**  
  Additional AUC/F1 evaluation for Colon.
- **`GOTabPFN_ClusterSizeAblation.ipynb`**  
  Cluster-size sensitivity/ablation experiments.
- **`GOTabPFN_Seed_Sensitivity.ipynb`**  
  TabPFN seed sensitivity analysis.

### Package test notebook

- **`GOTabPFN_Package_Test.ipynb`**  
  Tests the local package setup. This notebook checks package imports, GO-LR as a standalone metaheuristic ordering module, the four NSC compression variants, and binary (Colon)/multiclass (orlaws10P)/regression (DrivFace) runs on a separate local machine.


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

The main experiments were conducted on the TITAN cluster (`x86_64`, 188 GB RAM, 8 × NVIDIA TITAN RTX GPUs, 24 GB VRAM per GPU). Additional diagnostics, package tests, and fixed-parameter runs were executed on Vulcan, an 8-GPU NVIDIA RTX A6000 machine with 8 × 49 GB VRAM, 2 × Intel Xeon Gold 5320 CPUs, and 503 GB RAM. Therefore, small numerical/runtime differences from the main paper results may be observed depending on hardware configuration. The PyPI-installed package was also checked and tested on Google Colab. On the first run, TabPFN may download the required TabPFN-2.5 checkpoint from Hugging Face; the checkpoint is cached afterward.

## Installation

You can install **GOTabPFN** in several ways depending on your workflow.

### Option 1: Clone the Repository (Recommended for Development)

```bash
git clone https://github.com/zadid6pretam/GOTabPFN.git
cd GOTabPFN
pip install -r requirements.txt
pip install -e .
```


### Option 2: Install Directly from GitHub

```bash
pip install "git+https://github.com/zadid6pretam/GOTabPFN.git"
```


### Option 3: Use a Virtual Environment

```bash
python -m venv gotabpfn-env
source gotabpfn-env/bin/activate  # On Windows: gotabpfn-env\Scripts\activate

git clone https://github.com/zadid6pretam/GOTabPFN.git
cd GOTabPFN
pip install -r requirements.txt
pip install -e .
```


### Option 4: Local Install Without Editable Mode

```bash
git clone https://github.com/zadid6pretam/GOTabPFN.git
cd GOTabPFN
pip install -r requirements.txt
pip install .
```


### Option 5: Install from PyPI

```bash
pip install gotabpfn
```

## Dataset Compatibility and Preprocessing Guidelines

GOTabPFN is designed for tabular datasets, with particular focus on high-dimensional low-sample size tabular data where the number of features can be much larger than the number of samples. Typical examples include gene expression datasets, biomedical tabular datasets, document-term/tabular representations, extracted image feature embeddings, sensor derived data, and other numeric high-dimensional datasets.

### Supported Task Types

GOTabPFN supports:

- **Binary classification**
- **Multiclass classification**
- **Regression**

The task type is controlled through the TabPFN head configuration:

```python
TabPFN25Config(task_type="binary", ...)
TabPFN25Config(task_type="multiclass", ...)
TabPFN25Config(task_type="regression", ...)
```
- For classification, labels should be encoded as class labels. The example notebooks usually apply LabelEncoder or convert labels into contiguous integer classes before training. For regression, the target column should contain continuous numeric values.


### Expected input format

The recommended input format is a **CSV file** where:

- Rows correspond to samples.
- Columns correspond to features.
- One column is used as the target column.
- Feature columns should be numeric or convertible to numeric values.

- Example for classification:

```text
feature_1,feature_2,feature_3,...,label
0.12,1.48,-0.33,...,1
0.08,1.21,-0.52,...,0
...
```
- Example for regression:

```text
feature_1,feature_2,feature_3,...,target
0.12,1.48,-0.33,...,35.7
0.08,1.21,-0.52,...,42.1
...
```


### Numeric features

GOTabPFN’s GO-LR ordering and NSC compression modules operate on numeric feature matrices. Therefore, the safest setup is to provide a CSV where all feature columns are numeric after removing the target column.

If non-numeric columns are present, the provided notebook scripts and wrappers can drop them automatically. For example, columns containing sample IDs, filenames, text IDs, or categorical strings can be removed before fitting:

```python
num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
X_df = X_df[num_cols]
```
- This is useful for datasets that include metadata columns such as:

```text
sample_id
patient_id
cell
filename
image_path
group_name
```
- These columns should not be used directly as numeric features unless they have been properly encoded.


### Categorical features

The current GOTabPFN release is primarily intended for numeric tabular features. If your dataset contains categorical columns, recommended options are:

1. Drop non-numeric categorical columns if they are identifiers or metadata.
2. Encode meaningful categorical variables before using GOTabPFN.
3. Avoid using arbitrary ID columns as categorical features, because they can introduce spurious ordering or leakage.

Simple label encoding may be acceptable for ordinal categories, but for nominal categories, one-hot encoding or another appropriate categorical encoding should be considered before running GOTabPFN.

### Missing Values

GOTabPFN expects a numeric matrix without `NaN` or infinite values. The example scripts typically handle missing values by replacing invalid values with zero:

```python
X = np.nan_to_num(
    X,
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
).astype(np.float32)
```
- For more careful preprocessing, especially in applied datasets, users may prefer median imputation:

```python
X_num = X_num.fillna(X_num.median(numeric_only=True))
X_num = X_num.fillna(0.0)
```

- The same preprocessing rule used for training data should also be applied to validation/test data. In cross-validation experiments, imputation and scaling should ideally be fit on the training fold only and then applied to the validation fold.


### Feature scaling

Feature scaling is recommended. In most experiments, GOTabPFN uses standardization:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).astype(np.float32)
```
- For cross-validation, the leakage-safe version is:

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
X_valid = scaler.transform(X_valid_raw).astype(np.float32)
```
- Some released experiment scripts use global standardization to match the original experimental protocol. For new experiments or real applications, fold-wise standardization is usually preferred.


### Target preprocessing

For classification, the target should be encoded into integer class labels:

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y_raw).astype(np.int64)
```
- For binary classification, labels should become:
```text
0, 1
```
- For multiclass classification, labels should become:
```text
0, 1, 2, ..., C-1
```
- For regression, the target should be numeric:
```text
y = pd.to_numeric(df[target_col], errors="coerce")
y = y.fillna(y.median())
y = y.to_numpy(dtype=np.float32)
```


### Dataset size and dimensionality

GOTabPFN is especially useful for high-dimensional regimes, including:

- **HDLSS**: high-dimensional, low-sample-size datasets.
- Datasets where feature ordering may expose local structure.
- Datasets where compact tokenization can reduce the feature space before passing data to TabPFN-2.5 interface.

The method can also run on lower-dimensional datasets, but the benefits of feature ordering and NSC compression are expected to be stronger when the feature space contains redundancy, correlated feature groups, or structured feature neighborhoods.

### TabPFN Constraints

GOTabPFN uses a frozen TabPFN-2.5 head through `tabpfn==6.3.1`. Therefore, it inherits the practical constraints of the installed TabPFN version.

In general:

- Classification tasks should stay within the class-count limit supported by TabPFN.
- Very large sample sizes may require subsampling, batching strategies, or another downstream model.
- The first run may download a TabPFN-2.5 checkpoint from Hugging Face. The checkpoint is cached afterward.

For best reproducibility, use:

```txt
pip install tabpfn==6.3.1
```


### GO-LR feature ordering input

The GO-LR module expects a numeric matrix:

```python
X.shape == (n_samples, n_features)
```
- GO-LR learns a feature ordering:
```python
Pi_star = [feature_index_1, feature_index_2, ..., feature_index_m]
```

The standalone GO-LR.py wrapper can take a CSV file, drop the target column, keep numeric features, run ordering, and save:
- reordered feature table,
- learned feature ordering,
- ordering runtime,
- TSP path cost,
- MinLA cost.

Example:
```python
from gotabpfn import run_golr_csv

result = run_golr_csv(
    csv_path="coloncancer_encoded.csv",
    target_col="label",
    dataset_name="Colon",
    metric="euclidean",
    num_clusters=10,
    refine_passes=3,
    direction_select=True,
    out_prefix="colon_golr",
)
```


### NSC compression input

The NSC modules expect:

- a numeric feature matrix,
- a learned or identity feature ordering,
- optional hyperparameters controlling segmentation and compression.

The main GOTabPFN variant uses **NSC-pSP**, which combines PCA-IDF-aware budget selection with segment-wise principal subspace projection.

The package also includes standalone variants:

- `NSC-pSP.py`: PCA-IDF-aware segment-wise projection.
- `NSC-SP.py`: segment-wise projection with fixed/provided compression budget.
- `NSC-P.py`: PCA-IDF-aware descriptor/statistics pooling.
- `NSC.py`: original descriptor/statistics pooling.

### Recommended Minimal Preprocessing Pipeline

For most users, the recommended preprocessing workflow is:

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv("dataset.csv")

target_col = "label"
y_raw = df[target_col]
X_df = df.drop(columns=[target_col])

# Keep numeric features only
X_df = X_df.select_dtypes(include=[np.number])

# Handle missing values
X_df = X_df.apply(pd.to_numeric, errors="coerce")
X_df = X_df.fillna(X_df.median(numeric_only=True))
X_df = X_df.fillna(0.0)

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X_df.values).astype(np.float32)

# Encode labels for classification
le = LabelEncoder()
y = le.fit_transform(y_raw).astype(np.int64)
```
For regression, replace the target preprocessing with:
```python
y = pd.to_numeric(y_raw, errors="coerce")
y = y.fillna(y.median())
y = y.to_numpy(dtype=np.float32)
```


### What users do not need to do

Users do not need to manually construct a feature graph, manually define feature neighborhoods, or manually create TabPFN tokens. GOTabPFN handles:

- graph-guided feature ordering,
- local refinement of the ordering,
- feature segmentation,
- NSC compression/tokenization,
- TabPFN-2.5 prediction head fitting.

Users mainly need to provide a clean numeric feature matrix and a target column.

### Practical Notes

- Remove sample IDs, filenames, patient IDs, and other non-feature metadata before training.
- Standardize features before GO-LR and NSC.
- Use fold-wise preprocessing for strict cross-validation.
- Use `tabpfn==6.3.1` for TabPFN-2.5 compatibility.
- The first TabPFN run may download the required checkpoint from Hugging Face.
- GPU is recommended for faster experiments, but some components can fall back to CPU.
- Runtime and numerical results may vary slightly across hardware configurations.
