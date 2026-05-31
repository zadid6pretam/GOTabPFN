# GOTabPFN: From Feature Ordering to Compact Tokenization for Tabular Foundation Models on High-Dimensional Data (ICML 2026)

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
- Project Webpage: https://www.zadidhabib.com/gotabpfn.html

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
  Minimal notebook for checking the installed `gotabpfn` package after `pip install` in Google Colab. It will verify imports, initialize core modules, and run a workflow.
- **`Single_Wrapper_Test.md`**
    Minimal notebook for checking the installed `gotabpfn` package after `pip install` in Google Colab using the single wrapper examples.

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


## Example Usage

Below is a minimal example showing how to train **GOTabPFN**:

### Example 1: Binary Classification with Fixed GOTabPFN Hyperparameters

This example runs GOTabPFN on a binary-classification CSV dataset using fixed GO-LR and NSC-pSP hyperparameters. The dataset should contain numeric feature columns and one target column.

The hyperparameters below correspond to the Colon configuration reported in the paper. For other datasets, users can tune these values or replace them with the dataset-specific settings reported in the appendix.

```python
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from gotabpfn import GraphFeatureOrdering, pidf_segpca, TabPFN25Head, TabPFN25Config


# -----------------------
# User settings
# -----------------------
DATA_FILE = "coloncancer_encoded.csv"  # change your dataset file name
TARGET_COL = "label"                   # change your dataset target column
SEED = 42

# Fixed GOTabPFN hyperparameters
GO_METRIC = "euclidean"
GO_NUM_CLUSTERS = 10
GO_REFINE_PASSES = 3
GO_DIRECTION_SELECT = True

NSC_SEGMENTATION = "equal_mass"
NSC_M_RULE = "idf"
NSC_TAU = 0.99
NSC_GAMMA = 1.7570143129240916
NSC_BETA = 0.2244046472232107
NSC_MMIN = 64
NSC_MMAX = 384
NSC_LMIN = 16
ASSUME_STANDARDIZED = False

TABPFN_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------
# Utility
# -----------------------
def compute_deltas_adjacent_corr(X_tr, Pi_star, eps=1e-12):
    """
    Compute adjacent transition scores along the GO-LR order:
        delta_t = 1 - |corr(feature_t, feature_{t+1})|.

    Required for transition-aware NSC segmentation rules:
        - equal_mass
        - largest_jump
    """
    X_t = torch.from_numpy(X_tr).float()
    perm = torch.tensor(Pi_star, dtype=torch.long)

    Xp = X_t[:, perm]
    Xc = Xp - Xp.mean(dim=0, keepdim=True)
    std = Xc.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    Z = Xc / std

    corr_adj = (Z[:, :-1] * Z[:, 1:]).mean(dim=0)
    deltas = 1.0 - corr_adj.abs()

    return deltas.cpu()


# -----------------------
# Load and preprocess data
# -----------------------
df = pd.read_csv(DATA_FILE)

if TARGET_COL not in df.columns:
    raise ValueError(f"TARGET_COL='{TARGET_COL}' not found in the CSV file.")

y_raw = df[TARGET_COL].astype(str).fillna("missing_target")
X_df = df.drop(columns=[TARGET_COL])

# Keep numeric features only
X_df = X_df.select_dtypes(include=[np.number])
X_df = X_df.apply(pd.to_numeric, errors="coerce")
X_df = X_df.fillna(X_df.median(numeric_only=True)).fillna(0.0)

if X_df.shape[1] == 0:
    raise ValueError("No numeric feature columns found after preprocessing.")

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y_raw).astype(np.int64)

num_classes = len(le.classes_)
if num_classes != 2:
    raise ValueError(
        f"This example expects binary classification, but found {num_classes} classes."
    )

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X_df.values).astype(np.float32)

print(f"X shape: {X.shape}")
print(f"Classes: {list(le.classes_)}")
print(f"Using device: {DEVICE}")


# -----------------------
# Learn GO-LR feature ordering once
# -----------------------
go = GraphFeatureOrdering(
    num_clusters=GO_NUM_CLUSTERS,
    metric=GO_METRIC,
    refine=True,
    direction_select=GO_DIRECTION_SELECT,
    refine_passes=GO_REFINE_PASSES,
)

try:
    Pi_star, _, _, _ = go.fit(
        X,
        seed=SEED,
        deterministic=True,
        use_cpu_kmeans=False,
    )
except Exception:
    Pi_star, _, _, _ = go.fit(
        X,
        seed=SEED,
        deterministic=True,
        use_cpu_kmeans=True,
    )

Pi_star = list(map(int, Pi_star))

print(f"Learned GO-LR order length: {len(Pi_star)}")


# -----------------------
# 5x5 cross-validation
# -----------------------
rkf = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=5,
    random_state=SEED,
)

head_cfg = TabPFN25Config(
    task_type="binary",
    num_classes=2,
    device=DEVICE,
    random_state=TABPFN_SEED,
)

accs, f1s, aucs = [], [], []

for fold_id, (tr_idx, va_idx) in enumerate(rkf.split(X, y), start=1):
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    nsc = pidf_segpca(
        segmentation=NSC_SEGMENTATION,
        l_min=NSC_LMIN,
        m_rule=NSC_M_RULE,
        gamma=NSC_GAMMA,
        beta=NSC_BETA,
        tau=NSC_TAU,
        M_min=NSC_MMIN,
        M_max=NSC_MMAX,
        assume_standardized=ASSUME_STANDARDIZED,
        device=DEVICE,
    )

    X_tr_t = torch.from_numpy(X_tr)

    # equal_mass and largest_jump require transition scores.
    deltas = None
    if NSC_SEGMENTATION in {"equal_mass", "largest_jump"}:
        deltas = compute_deltas_adjacent_corr(X_tr, Pi_star)

    nsc.configure(
        Pi_star=Pi_star,
        X_train=X_tr_t,
        tau=NSC_TAU,
        deltas=deltas,
    )

    Z_tr = nsc.compress(X_tr_t, mode="flatten").cpu().numpy()
    Z_va = nsc.compress(torch.from_numpy(X_va), mode="flatten").cpu().numpy()

    head = TabPFN25Head(head_cfg)
    head.fit(Z_tr, y_tr)

    P = head.predict_proba(Z_va)
    pred = np.argmax(P, axis=1)

    acc = accuracy_score(y_va, pred)
    f1 = f1_score(y_va, pred, average="macro")
    auc = roc_auc_score(y_va, P[:, 1])

    accs.append(acc)
    f1s.append(f1)
    aucs.append(auc)

    print(f"Fold {fold_id:02d}: ACC={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")


print("\nFinal 5x5 CV results")
print(f"Accuracy : {np.mean(accs):.4f} ± {np.std(accs, ddof=1):.4f}")
print(f"Macro-F1 : {np.mean(f1s):.4f} ± {np.std(f1s, ddof=1):.4f}")
print(f"AUC      : {np.mean(aucs):.4f} ± {np.std(aucs, ddof=1):.4f}")
```
> **NB.** In our experiments, GO-LR is used as an unsupervised dataset-level feature ordering step. For each dataset and hyperparameter setting, the global feature order is learned from the full unlabeled feature matrix `X`, without using the target labels `y`. The learned order is then kept fixed during repeated cross-validation, where NSC is configured on each training split and the TabPFN head is fit only on `(Z_train, y_train)`. Therefore, the evaluation does not leak validation labels into the model or the ordering procedure. This protocol should be interpreted as unsupervised transductive feature ordering: validation feature values may contribute to the global feature order, but validation labels are never used.

### Example 2: Binary Classification with Optuna Hyperparameter Tuning

This example tunes GOTabPFN hyperparameters using Optuna. For each trial, GO-LR learns one feature ordering on the full preprocessed matrix for simplicity, then NSC-pSP and TabPFN-2.5 are evaluated using repeated stratified cross-validation. For a strictly leakage-free benchmark evaluation, preprocessing and GO-LR should be fit separately inside each training fold.

```python
import gc
import random
import numpy as np
import pandas as pd
import torch
import optuna

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

from gotabpfn import GraphFeatureOrdering, pidf_segpca, TabPFN25Head, TabPFN25Config


# -----------------------
# User settings
# -----------------------
DATA_FILE = "coloncancer_encoded.csv"  # change your dataset file name
TARGET_COL = "label"                   # change your dataset target column
SEED = 42
N_TRIALS = 50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------
# Utilities
# -----------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compute_deltas_adjacent_corr(X_tr, Pi_star, eps=1e-12):
    """
    Compute adjacent transition scores along the GO-LR order:
        delta_t = 1 - |corr(feature_t, feature_{t+1})|.

    Required for transition-aware NSC segmentation rules:
        - equal_mass
        - largest_jump
    """
    X_t = torch.from_numpy(X_tr).float()
    perm = torch.tensor(Pi_star, dtype=torch.long)

    Xp = X_t[:, perm]
    Xc = Xp - Xp.mean(dim=0, keepdim=True)
    std = Xc.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    Z = Xc / std

    corr_adj = (Z[:, :-1] * Z[:, 1:]).mean(dim=0)
    deltas = 1.0 - corr_adj.abs()

    return deltas.cpu()


# -----------------------
# Load and preprocess data
# -----------------------
seed_everything(SEED)

df = pd.read_csv(DATA_FILE)

if TARGET_COL not in df.columns:
    raise ValueError(f"TARGET_COL='{TARGET_COL}' not found in the CSV file.")

y_raw = df[TARGET_COL].astype(str).fillna("missing_target")
X_df = df.drop(columns=[TARGET_COL])

# Keep numeric features only
X_df = X_df.select_dtypes(include=[np.number])
X_df = X_df.apply(pd.to_numeric, errors="coerce")
X_df = X_df.fillna(X_df.median(numeric_only=True)).fillna(0.0)

if X_df.shape[1] == 0:
    raise ValueError("No numeric feature columns found after preprocessing.")

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y_raw).astype(np.int64)

num_classes = len(le.classes_)
if num_classes != 2:
    raise ValueError(
        f"This example expects binary classification, but found {num_classes} classes."
    )

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X_df.values).astype(np.float32)

print(f"X shape: {X.shape}")
print(f"Classes: {list(le.classes_)}")
print(f"Using device: {DEVICE}")


# -----------------------
# Cross-validation setup
# -----------------------
rkf = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=5,
    random_state=SEED,
)


# -----------------------
# Optuna objective
# -----------------------
def objective(trial):
    try:
        seed_everything(SEED)

        # GO-LR hyperparameters
        go_metric = trial.suggest_categorical(
            "go_metric",
            ["correlation", "cosine", "manhattan", "euclidean", "kl_divergence"],
        )
        go_num_clusters = trial.suggest_int("go_num_clusters", 4, 12)
        go_refine_passes = trial.suggest_int("go_refine_passes", 1, 3)
        go_direction_select = trial.suggest_categorical(
            "go_direction_select",
            [True, False],
        )

        # NSC-pSP hyperparameters
        nsc_segmentation = trial.suggest_categorical(
            "nsc_segmentation",
            ["uniform", "largest_jump", "equal_mass"],
        )
        nsc_m_rule = trial.suggest_categorical(
            "nsc_m_rule",
            ["default", "idf", "gamma"],
        )
        nsc_tau = trial.suggest_categorical("nsc_tau", [0.95, 0.99, 0.9975])
        nsc_gamma = trial.suggest_float("nsc_gamma", 1.0, 3.0)
        nsc_beta = trial.suggest_float("nsc_beta", 0.0, 0.9)
        nsc_Mmin = trial.suggest_categorical("nsc_Mmin", [16, 32, 48, 64])
        nsc_Mmax = trial.suggest_categorical("nsc_Mmax", [128, 256, 384, 512, 640])
        nsc_lmin = trial.suggest_categorical("nsc_lmin", [8, 12, 16])
        assume_standardized = trial.suggest_categorical(
            "assume_standardized",
            [True, False],
        )

        tabpfn_seed = trial.suggest_categorical(
            "tabpfn_seed",
            [0, 1, 2, 3, 4, 42],
        )

        # -----------------------
        # Learn GO-LR ordering once per trial
        # -----------------------
        go = GraphFeatureOrdering(
            num_clusters=go_num_clusters,
            metric=go_metric,
            refine=True,
            direction_select=go_direction_select,
            refine_passes=go_refine_passes,
        )

        try:
            Pi_star, _, _, _ = go.fit(
                X,
                seed=SEED,
                deterministic=True,
                use_cpu_kmeans=False,
            )
        except Exception:
            cleanup_cuda()
            Pi_star, _, _, _ = go.fit(
                X,
                seed=SEED,
                deterministic=True,
                use_cpu_kmeans=True,
            )

        Pi_star = list(map(int, Pi_star))

        head_cfg = TabPFN25Config(
            task_type="binary",
            num_classes=2,
            device=DEVICE,
            random_state=tabpfn_seed,
        )

        accs = []

        # -----------------------
        # Repeated CV evaluation
        # -----------------------
        for fold_id, (tr_idx, va_idx) in enumerate(rkf.split(X, y), start=1):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            nsc = pidf_segpca(
                segmentation=nsc_segmentation,
                l_min=nsc_lmin,
                m_rule=nsc_m_rule,
                gamma=nsc_gamma,
                beta=nsc_beta,
                tau=nsc_tau,
                M_min=nsc_Mmin,
                M_max=nsc_Mmax,
                assume_standardized=assume_standardized,
                device=DEVICE,
            )

            # equal_mass and largest_jump require transition scores.
            deltas = None
            if nsc_segmentation in {"largest_jump", "equal_mass"}:
                deltas = compute_deltas_adjacent_corr(X_tr, Pi_star)

            X_tr_t = torch.from_numpy(X_tr)

            nsc.configure(
                Pi_star=Pi_star,
                X_train=X_tr_t,
                tau=nsc_tau,
                deltas=deltas,
            )

            Z_tr = nsc.compress(X_tr_t, mode="flatten").cpu().numpy()
            Z_va = nsc.compress(torch.from_numpy(X_va), mode="flatten").cpu().numpy()

            head = TabPFN25Head(head_cfg)
            head.fit(Z_tr, y_tr)

            P = head.predict_proba(Z_va)
            pred = np.argmax(P, axis=1)

            acc = accuracy_score(y_va, pred)
            accs.append(acc)

            trial.report(float(np.mean(accs)), step=fold_id)

            if trial.should_prune():
                cleanup_cuda()
                raise optuna.TrialPruned()

            cleanup_cuda()

        return float(np.mean(accs))

    except optuna.TrialPruned:
        raise

    except Exception as e:
        cleanup_cuda()
        trial.set_user_attr("failed_reason", repr(e))
        return 0.0


# -----------------------
# Run Optuna
# -----------------------
sampler = optuna.samplers.TPESampler(
    seed=SEED,
    multivariate=True,
    group=True,
)

pruner = optuna.pruners.MedianPruner(
    n_warmup_steps=10,
)

study = optuna.create_study(
    direction="maximize",
    sampler=sampler,
    pruner=pruner,
)

study.optimize(
    objective,
    n_trials=N_TRIALS,
    show_progress_bar=True,
    gc_after_trial=True,
    n_jobs=1,
)

print("\nBest trial")
print(f"Best mean accuracy: {study.best_value:.6f}")

print("\nBest hyperparameters:")
for key, value in study.best_params.items():
    print(f"{key}: {value}")

print("\nFailed trials, if any:")
for t in study.trials:
    reason = t.user_attrs.get("failed_reason", None)
    if reason is not None:
        print(f"Trial {t.number}: {reason}")
```
### Example 3: Binary Classification with the single wrapper

The easiest way to run the full GOTabPFN pipeline on a CSV dataset is to use the high-level `run_gotabpfn_csv` wrapper. This example runs GOTabPFN on the Colon dataset for binary classification using GO-LR feature ordering, NSC-pSP compression, and the TabPFN-2.5 prediction head.

```python
import numpy as np
import torch

from gotabpfn import run_gotabpfn_csv


# -----------------------
# User settings
# -----------------------
DATA_FILE = "coloncancer_encoded.csv"  # change this to your dataset file name
TARGET_COL = "label"                   # change this to your target column
SEED = 42

# Fixed GO-LR hyperparameters
GO_METRIC = "euclidean"
GO_NUM_CLUSTERS = 10
GO_REFINE_PASSES = 3
GO_DIRECTION_SELECT = True

# Fixed NSC-pSP hyperparameters
NSC_SEGMENTATION = "equal_mass"
NSC_M_RULE = "idf"
NSC_TAU = 0.99
NSC_GAMMA = 1.7570143129240916
NSC_BETA = 0.2244046472232107
NSC_MMIN = 64
NSC_MMAX = 384
NSC_LMIN = 16
ASSUME_STANDARDIZED = False

TABPFN_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------
# Run GOTabPFN
# -----------------------
results = run_gotabpfn_csv(
    csv_path=DATA_FILE,
    target_col=TARGET_COL,
    task_type="binary",
    non_numeric="drop",

    # 5x5 repeated stratified cross-validation
    cv="5x5",
    seed=SEED,

    # GO-LR settings
    go_metric=GO_METRIC,
    go_num_clusters=GO_NUM_CLUSTERS,
    go_refine=True,
    go_direction_select=GO_DIRECTION_SELECT,
    go_refine_passes=GO_REFINE_PASSES,

    # Try GPU KMeans first, then fall back to CPU KMeans if needed
    go_use_cpu_kmeans=False,
    go_fallback_cpu_kmeans=True,

    # NSC-pSP settings
    nsc_segmentation=NSC_SEGMENTATION,
    nsc_m_rule=NSC_M_RULE,
    nsc_tau=NSC_TAU,
    nsc_gamma=NSC_GAMMA,
    nsc_beta=NSC_BETA,
    nsc_M_min=NSC_MMIN,
    nsc_M_max=NSC_MMAX,
    nsc_l_min=NSC_LMIN,
    assume_standardized=ASSUME_STANDARDIZED,

    # TabPFN-2.5 head
    tabpfn_seed=TABPFN_SEED,
    device=DEVICE,

    return_predictions=True,
    verbose=False,
)


# -----------------------
# Access outputs
# -----------------------
print(results["summary"])

metrics_df = results["metrics_df"]
predictions_df = results["predictions_df"]

print("\nFold-wise metrics")
print(metrics_df)

print("\nPrediction preview")
print(predictions_df.head())

print("\nGO-LR ordering")
print(f"Learned GO-LR order length: {len(results['ordering'])}")
print("First 10 ordered features:")
print(results["ordered_feature_names"][:10])


# -----------------------
# Final 5x5 CV summary
# -----------------------
accs = metrics_df["accuracy"].to_numpy()
f1s = metrics_df["macro_f1"].to_numpy()
aucs = metrics_df["auc"].dropna().to_numpy()

print("\nFinal 5x5 CV results")
print(f"Accuracy : {np.mean(accs):.4f} ± {np.std(accs, ddof=1):.4f}")
print(f"Macro-F1 : {np.mean(f1s):.4f} ± {np.std(f1s, ddof=1):.4f}")
print(f"AUC      : {np.mean(aucs):.4f} ± {np.std(aucs, ddof=1):.4f}")
```
- The returned dictionary contains:
```python
results["summary"]                # text summary of the run
results["metrics_df"]             # fold-wise accuracy, macro-F1, AUC, and token count
results["predictions_df"]         # per-fold predictions
results["ordering"]               # learned GO-LR feature ordering
results["ordered_feature_names"]   # ordered feature names
results["num_features"]           # number of numeric input features
results["num_samples"]            # number of samples
```
- For this example, NSC-pSP compresses the original feature vector into a smaller number of structured meta-features before applying the TabPFN-2.5 head. For instance, if nsc_tokens = 64, then the original feature vector is compressed into 64 NSC meta-features/tokens for each sample.


### Example 4: Multiclass Classification with Fixed GOTabPFN Hyperparameters

This example runs GOTabPFN on a multiclass CSV dataset using fixed GO-LR and NSC-pSP hyperparameters.

```python
import gc
import time
import random
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from gotabpfn import GraphFeatureOrdering, pidf_segpca, TabPFN25Head, TabPFN25Config

# -----------------------
# User settings
# -----------------------
DATA_FILE = "orlraws10P.csv" # change this to your dataset file name
TARGET_COL = "label" # change this to your dataset target column
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Fixed GOTabPFN hyperparameters
FIXED_PARAMS = {
    "go_metric": "cosine",
    "go_num_clusters": 5,
    "go_refine_passes": 1,
    "go_direction_select": False,
    "go_feat_subsample": 3000,

    "nsc_segmentation": "uniform",
    "nsc_m_rule": "default",
    "nsc_tau": 0.99,
    "nsc_gamma": 2.049512863264476,
    "nsc_beta": 0.3887505167779042,
    "nsc_Mmin": 32,
    "nsc_Mmax": 384,
    "nsc_lmin": 12,
    "assume_standardized": False,

    "tabpfn_seed": 42,
}

# -----------------------
# Utilities
# -----------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def safe_multiclass_macro_ovr_auc(y_true, proba, num_classes):
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


# -----------------------
# Load and preprocess data
# -----------------------
seed_everything(SEED)

df = pd.read_csv(DATA_FILE)

y_raw = df[TARGET_COL].astype(str).fillna("missing_target")
X_df = df.drop(columns=[TARGET_COL])

# Keep numeric features only
X_df = X_df.select_dtypes(include=[np.number])
X_df = X_df.apply(pd.to_numeric, errors="coerce")
X_df = X_df.fillna(X_df.median(numeric_only=True)).fillna(0.0)

# Encode multiclass labels
le = LabelEncoder()
y = le.fit_transform(y_raw).astype(np.int64)
num_classes = len(le.classes_)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X_df.values).astype(np.float32)

print(f"X shape: {X.shape}, classes: {num_classes}")

# -----------------------
# Learn GO-LR ordering once
# -----------------------
m_full = X.shape[1]
feat_subsample = FIXED_PARAMS["go_feat_subsample"]

rng = np.random.default_rng(SEED + 999)

if feat_subsample is not None and feat_subsample < m_full:
    feat_idx = rng.choice(m_full, size=feat_subsample, replace=False)
    feat_idx.sort()
else:
    feat_idx = np.arange(m_full)

X_go = X[:, feat_idx]

go = GraphFeatureOrdering(
    num_clusters=FIXED_PARAMS["go_num_clusters"],
    metric=FIXED_PARAMS["go_metric"],
    refine=True,
    direction_select=FIXED_PARAMS["go_direction_select"],
    refine_passes=FIXED_PARAMS["go_refine_passes"],
)

try:
    Pi_sub, _, _, _ = go.fit(
        X_go,
        seed=SEED,
        deterministic=True,
        use_cpu_kmeans=False,
    )
except Exception:
    cleanup_cuda()
    Pi_sub, _, _, _ = go.fit(
        X_go,
        seed=SEED,
        deterministic=True,
        use_cpu_kmeans=True,
    )

ordered_subset = feat_idx[np.array(Pi_sub, dtype=np.int64)].tolist()

if len(feat_idx) < m_full:
    remaining = np.setdiff1d(np.arange(m_full), feat_idx, assume_unique=False)
    Pi_star = ordered_subset + remaining.tolist()
else:
    Pi_star = ordered_subset

Pi_star = list(map(int, Pi_star))

# -----------------------
# 5x5 cross-validation
# -----------------------
rkf = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=5,
    random_state=SEED,
)

head_cfg = TabPFN25Config(
    task_type="multiclass",
    num_classes=int(num_classes),
    device=DEVICE,
    random_state=int(FIXED_PARAMS["tabpfn_seed"]),
)

accs, f1s, aucs = [], [], []
t0 = time.perf_counter()

for fold_id, (tr_idx, va_idx) in enumerate(rkf.split(X, y), start=1):
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    nsc = pidf_segpca(
        segmentation=FIXED_PARAMS["nsc_segmentation"],
        l_min=int(FIXED_PARAMS["nsc_lmin"]),
        m_rule=FIXED_PARAMS["nsc_m_rule"],
        gamma=float(FIXED_PARAMS["nsc_gamma"]),
        beta=float(FIXED_PARAMS["nsc_beta"]),
        tau=float(FIXED_PARAMS["nsc_tau"]),
        M_min=int(FIXED_PARAMS["nsc_Mmin"]),
        M_max=int(FIXED_PARAMS["nsc_Mmax"]),
        assume_standardized=bool(FIXED_PARAMS["assume_standardized"]),
        device=DEVICE,
    )

    X_tr_t = torch.from_numpy(X_tr)

    nsc.configure(
        Pi_star=Pi_star,
        X_train=X_tr_t,
        tau=float(FIXED_PARAMS["nsc_tau"]),
        deltas=None,
    )

    Z_tr = nsc.compress(X_tr_t, mode="flatten").cpu().numpy()
    Z_va = nsc.compress(torch.from_numpy(X_va), mode="flatten").cpu().numpy()

    head = TabPFN25Head(head_cfg)
    head.fit(Z_tr, y_tr)

    P = head.predict_proba(Z_va)
    pred = np.argmax(P, axis=1)

    acc = float(accuracy_score(y_va, pred))
    f1m = float(f1_score(y_va, pred, average="macro"))
    aucm = safe_multiclass_macro_ovr_auc(y_va, P, num_classes)

    accs.append(acc)
    f1s.append(f1m)
    aucs.append(aucm)

    print(
        f"Fold {fold_id:02d}: "
        f"ACC={acc:.4f}, Macro-F1={f1m:.4f}, Macro-OvR-AUC={aucm:.4f}"
    )

    cleanup_cuda()

print("\nFinal 5x5 CV results")
print(f"Accuracy      : {np.mean(accs):.4f} ± {np.std(accs, ddof=1):.4f}")
print(f"Macro-F1      : {np.mean(f1s):.4f} ± {np.std(f1s, ddof=1):.4f}")
print(f"Macro-OvR-AUC : {np.nanmean(aucs):.4f} ± {np.nanstd(aucs, ddof=1):.4f}")
print(f"Elapsed time  : {time.perf_counter() - t0:.2f} seconds")
```


### Example 5: Multiclass Classification with Optuna Hyperparameter Tuning

This example tunes GOTabPFN hyperparameters for a multiclass classification dataset. For each trial, GO-LR learns one feature ordering, then NSC-pSP and the frozen TabPFN-2.5 head are evaluated using repeated stratified cross-validation.

```python
import gc
import random
import numpy as np
import pandas as pd
import torch
import optuna

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

from gotabpfn import GraphFeatureOrdering, pidf_segpca, TabPFN25Head, TabPFN25Config

# -----------------------
# User settings
# -----------------------
DATA_FILE = "orlraws10P.csv" #change this to your dataset file name
TARGET_COL = "label" # change this to your dataset target column
SEED = 42
N_TRIALS = 50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# Utilities
# -----------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compute_deltas_adjacent_corr(X_tr, Pi_star, eps=1e-12):
    X_t = torch.from_numpy(X_tr).float()
    perm = torch.tensor(Pi_star, dtype=torch.long)

    Xp = X_t[:, perm]
    Xc = Xp - Xp.mean(dim=0, keepdim=True)
    std = Xc.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    Z = Xc / std

    corr = (Z[:, :-1] * Z[:, 1:]).mean(dim=0)
    return (1.0 - corr.abs()).cpu()


# -----------------------
# Load and preprocess data
# -----------------------
seed_everything(SEED)

df = pd.read_csv(DATA_FILE)

y_raw = df[TARGET_COL].astype(str).fillna("missing_target")
X_df = df.drop(columns=[TARGET_COL])

# Keep numeric features only
X_df = X_df.select_dtypes(include=[np.number])
X_df = X_df.apply(pd.to_numeric, errors="coerce")
X_df = X_df.fillna(X_df.median(numeric_only=True)).fillna(0.0)

le = LabelEncoder()
y = le.fit_transform(y_raw).astype(np.int64)
num_classes = len(le.classes_)

scaler = StandardScaler()
X = scaler.fit_transform(X_df.values).astype(np.float32)

rkf = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=5,
    random_state=SEED,
)

m_full = X.shape[1]

# -----------------------
# Optuna objective
# -----------------------
def objective(trial):
    seed_everything(SEED)

    # GO-LR hyperparameters
    go_metric = trial.suggest_categorical(
        "go_metric",
        ["correlation", "cosine", "manhattan", "euclidean", "kl_divergence"],
    )
    go_num_clusters = trial.suggest_int("go_num_clusters", 4, 12)
    go_refine_passes = trial.suggest_int("go_refine_passes", 1, 3)
    go_direction_select = trial.suggest_categorical(
        "go_direction_select",
        [True, False],
    )

    # Optional feature subsampling for very high-dimensional datasets
    go_feat_subsample = trial.suggest_categorical(
        "go_feat_subsample",
        [None, 1000, 2000, 3000],
    )

    # NSC-pSP hyperparameters
    nsc_segmentation = trial.suggest_categorical(
        "nsc_segmentation",
        ["uniform", "largest_jump", "equal_mass"],
    )
    nsc_m_rule = trial.suggest_categorical(
        "nsc_m_rule",
        ["default", "idf", "gamma"],
    )
    nsc_tau = trial.suggest_categorical("nsc_tau", [0.95, 0.99, 0.9975])
    nsc_gamma = trial.suggest_float("nsc_gamma", 1.0, 3.0)
    nsc_beta = trial.suggest_float("nsc_beta", 0.0, 0.9)
    nsc_Mmin = trial.suggest_categorical("nsc_Mmin", [16, 32, 48, 64])
    nsc_Mmax = trial.suggest_categorical("nsc_Mmax", [128, 256, 384, 512, 640])
    nsc_lmin = trial.suggest_categorical("nsc_lmin", [8, 12, 16])
    assume_standardized = trial.suggest_categorical(
        "assume_standardized",
        [True, False],
    )

    tabpfn_seed = trial.suggest_categorical(
        "tabpfn_seed",
        [0, 1, 2, 3, 4, 42],
    )

    # Feature subsampling before GO-LR
    if go_feat_subsample is not None and int(go_feat_subsample) < m_full:
        rng = np.random.default_rng(SEED + 999)
        feat_idx = rng.choice(m_full, size=int(go_feat_subsample), replace=False)
        feat_idx.sort()
    else:
        feat_idx = np.arange(m_full)

    X_go = X[:, feat_idx]

    # Learn GO-LR ordering once per trial
    go = GraphFeatureOrdering(
        num_clusters=go_num_clusters,
        metric=go_metric,
        refine=True,
        direction_select=go_direction_select,
        refine_passes=go_refine_passes,
    )

    try:
        Pi_sub, _, _, _ = go.fit(
            X_go,
            seed=SEED,
            deterministic=True,
            use_cpu_kmeans=False,
        )
    except Exception:
        cleanup_cuda()
        Pi_sub, _, _, _ = go.fit(
            X_go,
            seed=SEED,
            deterministic=True,
            use_cpu_kmeans=True,
        )

    ordered_subset = feat_idx[np.array(Pi_sub, dtype=np.int64)].tolist()

    if len(feat_idx) < m_full:
        remaining = np.setdiff1d(np.arange(m_full), feat_idx, assume_unique=False)
        Pi_star = ordered_subset + remaining.tolist()
    else:
        Pi_star = ordered_subset

    Pi_star = list(map(int, Pi_star))

    head_cfg = TabPFN25Config(
        task_type="multiclass",
        num_classes=int(num_classes),
        device=DEVICE,
        random_state=int(tabpfn_seed),
    )

    accs = []

    for fold_id, (tr_idx, va_idx) in enumerate(rkf.split(X, y), start=1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        nsc = pidf_segpca(
            segmentation=nsc_segmentation,
            l_min=nsc_lmin,
            m_rule=nsc_m_rule,
            gamma=nsc_gamma,
            beta=nsc_beta,
            tau=nsc_tau,
            M_min=nsc_Mmin,
            M_max=nsc_Mmax,
            assume_standardized=assume_standardized,
            device=DEVICE,
        )

        deltas = None
        if nsc_segmentation != "uniform":
            deltas = compute_deltas_adjacent_corr(X_tr, Pi_star)

        X_tr_t = torch.from_numpy(X_tr)

        nsc.configure(
            Pi_star=Pi_star,
            X_train=X_tr_t,
            tau=nsc_tau,
            deltas=deltas,
        )

        Z_tr = nsc.compress(X_tr_t, mode="flatten").cpu().numpy()
        Z_va = nsc.compress(torch.from_numpy(X_va), mode="flatten").cpu().numpy()

        head = TabPFN25Head(head_cfg)
        head.fit(Z_tr, y_tr)

        P = head.predict_proba(Z_va)
        pred = np.argmax(P, axis=1)

        acc = accuracy_score(y_va, pred)
        accs.append(acc)

        trial.report(float(np.mean(accs)), step=fold_id)

        if trial.should_prune():
            cleanup_cuda()
            raise optuna.TrialPruned()

        cleanup_cuda()

    return float(np.mean(accs))


# -----------------------
# Run Optuna
# -----------------------
sampler = optuna.samplers.TPESampler(
    seed=SEED,
    multivariate=True,
    group=True,
)

pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)

study = optuna.create_study(
    direction="maximize",
    sampler=sampler,
    pruner=pruner,
)

study.optimize(
    objective,
    n_trials=N_TRIALS,
    show_progress_bar=True,
    gc_after_trial=True,
    n_jobs=1,
)

print("\nBest trial")
print(f"Best mean accuracy: {study.best_value:.6f}")

print("\nBest hyperparameters:")
for key, value in study.best_params.items():
    print(f"{key}: {value}")
```

### Example 6: Multiclass Classification with the Single Wrapper

This example runs GOTabPFN on the ORL face dataset (`orlraws10P`) for multiclass classification using the high-level `run_gotabpfn_csv` wrapper. The pipeline performs GO-LR feature ordering, NSC-pSP compression, and TabPFN-2.5 prediction under 5×5 repeated stratified cross-validation.

```python
import numpy as np
import torch

from gotabpfn import run_gotabpfn_csv


# -----------------------
# User settings
# -----------------------
DATA_FILE = "orlraws10P.csv"  # change this to your dataset file name
TARGET_COL = "label"          # change this to your target column
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------
# Fixed GO-LR hyperparameters
# -----------------------
GO_METRIC = "cosine"
GO_NUM_CLUSTERS = 5
GO_REFINE_PASSES = 1
GO_DIRECTION_SELECT = False
GO_FEAT_SUBSAMPLE = 3000


# -----------------------
# Fixed NSC-pSP hyperparameters
# -----------------------
NSC_SEGMENTATION = "uniform"
NSC_M_RULE = "default"
NSC_TAU = 0.99
NSC_GAMMA = 2.049512863264476
NSC_BETA = 0.3887505167779042
NSC_MMIN = 32
NSC_MMAX = 384
NSC_LMIN = 12
ASSUME_STANDARDIZED = False

TABPFN_SEED = 42


# -----------------------
# Run GOTabPFN
# -----------------------
results = run_gotabpfn_csv(
    csv_path=DATA_FILE,
    target_col=TARGET_COL,
    task_type="multiclass",
    non_numeric="drop",

    # 5x5 repeated stratified cross-validation
    cv="5x5",
    seed=SEED,

    # GO-LR settings
    go_metric=GO_METRIC,
    go_num_clusters=GO_NUM_CLUSTERS,
    go_refine=True,
    go_direction_select=GO_DIRECTION_SELECT,
    go_refine_passes=GO_REFINE_PASSES,
    go_feat_subsample=GO_FEAT_SUBSAMPLE,

    # Try GPU KMeans first, then fall back to CPU KMeans if needed
    go_use_cpu_kmeans=False,
    go_fallback_cpu_kmeans=True,

    # NSC-pSP settings
    nsc_segmentation=NSC_SEGMENTATION,
    nsc_m_rule=NSC_M_RULE,
    nsc_tau=NSC_TAU,
    nsc_gamma=NSC_GAMMA,
    nsc_beta=NSC_BETA,
    nsc_M_min=NSC_MMIN,
    nsc_M_max=NSC_MMAX,
    nsc_l_min=NSC_LMIN,
    assume_standardized=ASSUME_STANDARDIZED,

    # TabPFN-2.5 head
    tabpfn_seed=TABPFN_SEED,
    device=DEVICE,

    return_predictions=True,
    verbose=False,
)


# -----------------------
# Access outputs
# -----------------------
print(results["summary"])

metrics_df = results["metrics_df"]
predictions_df = results["predictions_df"]

print("\nFold-wise metrics")
print(metrics_df)

print("\nPrediction preview")
print(predictions_df.head())

print("\nGO-LR ordering")
print(f"Learned GO-LR order length: {len(results['ordering'])}")
print("First 10 ordered features:")
print(results["ordered_feature_names"][:10])


# -----------------------
# Final 5x5 CV summary
# -----------------------
accs = metrics_df["accuracy"].to_numpy()
f1s = metrics_df["macro_f1"].to_numpy()

print("\nFinal 5x5 CV results")
print(f"Accuracy      : {np.mean(accs):.4f} ± {np.std(accs, ddof=1):.4f}")
print(f"Macro-F1      : {np.mean(f1s):.4f} ± {np.std(f1s, ddof=1):.4f}")

if "macro_ovr_auc" in metrics_df.columns:
    aucs = metrics_df["macro_ovr_auc"].dropna().to_numpy()
    if len(aucs) > 0:
        print(f"Macro-OvR-AUC : {np.mean(aucs):.4f} ± {np.std(aucs, ddof=1):.4f}")
```
- The returned dictionary contains:
```python
results["summary"]                # text summary of the run
results["metrics_df"]             # fold-wise accuracy, macro-F1, macro-OvR-AUC, and token count
results["predictions_df"]         # per-fold predictions
results["ordering"]               # learned GO-LR feature ordering
results["ordered_feature_names"]   # ordered feature names
results["num_features"]           # number of numeric input features
results["num_samples"]            # number of samples
```


### Example 7: Regression with Fixed GOTabPFN Hyperparameters

This example runs GOTabPFN on a regression CSV dataset using fixed GO-LR and NSC-pSP hyperparameters. The target column should contain continuous numeric values.

```python
import gc
import time
import random
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from gotabpfn import GraphFeatureOrdering, pidf_segpca, TabPFN25Head, TabPFN25Config

# -----------------------
# User settings
# -----------------------
DATA_FILE = "drivface.csv" # change this to your dataset file name
TARGET_COL = "angle" # change this to your dataset file name
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Fixed GOTabPFN hyperparameters
FIXED_PARAMS = {
    "go_metric": "manhattan",
    "go_num_clusters": 5,
    "go_refine_passes": 1,
    "go_direction_select": False,
    "go_feat_subsample": 2000,

    "nsc_segmentation": "largest_jump",
    "nsc_m_rule": "idf",
    "nsc_tau": 0.99,
    "nsc_gamma": 2.654390393837633,
    "nsc_beta": 0.043192175152615336,
    "nsc_Mmin": 16,
    "nsc_Mmax": 256,
    "nsc_lmin": 12,
    "assume_standardized": True,

    "tabpfn_seed": 3,
}

# -----------------------
# Utilities
# -----------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compute_deltas_adjacent_corr(X_tr, Pi_star, eps=1e-12):
    X_t = torch.from_numpy(X_tr).float()
    perm = torch.tensor(Pi_star, dtype=torch.long)

    Xp = X_t[:, perm]
    Xc = Xp - Xp.mean(dim=0, keepdim=True)
    std = Xc.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    Z = Xc / std

    corr = (Z[:, :-1] * Z[:, 1:]).mean(dim=0)
    return (1.0 - corr.abs()).cpu()


# -----------------------
# Load and preprocess data
# -----------------------
seed_everything(SEED)

df = pd.read_csv(DATA_FILE)

# Regression target
y_raw = pd.to_numeric(df[TARGET_COL], errors="coerce")
y_raw = y_raw.fillna(y_raw.median())
y = y_raw.to_numpy(dtype=np.float32)

X_df = df.drop(columns=[TARGET_COL])

# Keep numeric features only
X_df = X_df.select_dtypes(include=[np.number])
X_df = X_df.apply(pd.to_numeric, errors="coerce")
X_df = X_df.fillna(X_df.median(numeric_only=True)).fillna(0.0)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X_df.values).astype(np.float32)

print(f"X shape: {X.shape}, y shape: {y.shape}")

# -----------------------
# Learn GO-LR ordering once
# -----------------------
m_full = X.shape[1]
feat_subsample = FIXED_PARAMS["go_feat_subsample"]

rng = np.random.default_rng(SEED + 999)

if feat_subsample is not None and feat_subsample < m_full:
    feat_idx = rng.choice(m_full, size=feat_subsample, replace=False)
    feat_idx.sort()
else:
    feat_idx = np.arange(m_full)

X_go = X[:, feat_idx]

go = GraphFeatureOrdering(
    num_clusters=FIXED_PARAMS["go_num_clusters"],
    metric=FIXED_PARAMS["go_metric"],
    refine=True,
    direction_select=FIXED_PARAMS["go_direction_select"],
    refine_passes=FIXED_PARAMS["go_refine_passes"],
)

try:
    Pi_sub, _, _, _ = go.fit(
        X_go,
        seed=SEED,
        deterministic=True,
        use_cpu_kmeans=False,
    )
except Exception:
    cleanup_cuda()
    Pi_sub, _, _, _ = go.fit(
        X_go,
        seed=SEED,
        deterministic=True,
        use_cpu_kmeans=True,
    )

ordered_subset = feat_idx[np.array(Pi_sub, dtype=np.int64)].tolist()

if len(feat_idx) < m_full:
    remaining = np.setdiff1d(np.arange(m_full), feat_idx, assume_unique=False)
    Pi_star = ordered_subset + remaining.tolist()
else:
    Pi_star = ordered_subset

Pi_star = list(map(int, Pi_star))

# -----------------------
# 5x5 cross-validation
# -----------------------
rkf = RepeatedKFold(
    n_splits=5,
    n_repeats=5,
    random_state=SEED,
)

head_cfg = TabPFN25Config(
    task_type="regression",
    num_classes=1,
    device=DEVICE,
    random_state=int(FIXED_PARAMS["tabpfn_seed"]),
)

r2s, rmses, maes = [], [], []
t0 = time.perf_counter()

for fold_id, (tr_idx, va_idx) in enumerate(rkf.split(X), start=1):
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    nsc = pidf_segpca(
        segmentation=FIXED_PARAMS["nsc_segmentation"],
        l_min=int(FIXED_PARAMS["nsc_lmin"]),
        m_rule=FIXED_PARAMS["nsc_m_rule"],
        gamma=float(FIXED_PARAMS["nsc_gamma"]),
        beta=float(FIXED_PARAMS["nsc_beta"]),
        tau=float(FIXED_PARAMS["nsc_tau"]),
        M_min=int(FIXED_PARAMS["nsc_Mmin"]),
        M_max=int(FIXED_PARAMS["nsc_Mmax"]),
        assume_standardized=bool(FIXED_PARAMS["assume_standardized"]),
        device=DEVICE,
    )

    deltas = None
    if FIXED_PARAMS["nsc_segmentation"] != "uniform":
        deltas = compute_deltas_adjacent_corr(X_tr, Pi_star)

    X_tr_t = torch.from_numpy(X_tr)

    nsc.configure(
        Pi_star=Pi_star,
        X_train=X_tr_t,
        tau=float(FIXED_PARAMS["nsc_tau"]),
        deltas=deltas,
    )

    Z_tr = nsc.compress(X_tr_t, mode="flatten").cpu().numpy()
    Z_va = nsc.compress(torch.from_numpy(X_va), mode="flatten").cpu().numpy()

    head = TabPFN25Head(head_cfg)
    head.fit(Z_tr, y_tr)

    pred = np.asarray(head.predict(Z_va), dtype=np.float32).reshape(-1)

    r2 = float(r2_score(y_va, pred))
    rmse = float(np.sqrt(mean_squared_error(y_va, pred)))
    mae = float(mean_absolute_error(y_va, pred))

    r2s.append(r2)
    rmses.append(rmse)
    maes.append(mae)

    print(f"Fold {fold_id:02d}: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

    cleanup_cuda()

print("\nFinal 5x5 CV results")
print(f"R2   : {np.mean(r2s):.4f} ± {np.std(r2s, ddof=1):.4f}")
print(f"RMSE : {np.mean(rmses):.4f} ± {np.std(rmses, ddof=1):.4f}")
print(f"MAE  : {np.mean(maes):.4f} ± {np.std(maes, ddof=1):.4f}")
print(f"Elapsed time: {time.perf_counter() - t0:.2f} seconds")
```

### Example 8: Regression with Optuna Hyperparameter Tuning

This example tunes GOTabPFN hyperparameters for a regression dataset. For each trial, GO-LR learns one feature ordering, then NSC-pSP and the frozen TabPFN-2.5 regression head are evaluated using repeated cross-validation.

```python
import gc
import random
import numpy as np
import pandas as pd
import torch
import optuna

from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from gotabpfn import GraphFeatureOrdering, pidf_segpca, TabPFN25Head, TabPFN25Config

# -----------------------
# User settings
# -----------------------
DATA_FILE = "drivface.csv" # change this to your dataset file name
TARGET_COL = "angle" # change this to your dataset target column name
SEED = 42
N_TRIALS = 50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# Utilities
# -----------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compute_deltas_adjacent_corr(X_tr, Pi_star, eps=1e-12):
    X_t = torch.from_numpy(X_tr).float()
    perm = torch.tensor(Pi_star, dtype=torch.long)

    Xp = X_t[:, perm]
    Xc = Xp - Xp.mean(dim=0, keepdim=True)
    std = Xc.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    Z = Xc / std

    corr = (Z[:, :-1] * Z[:, 1:]).mean(dim=0)
    return (1.0 - corr.abs()).cpu()


# -----------------------
# Load and preprocess data
# -----------------------
seed_everything(SEED)

df = pd.read_csv(DATA_FILE)

# Regression target
y_raw = pd.to_numeric(df[TARGET_COL], errors="coerce")
y_raw = y_raw.fillna(y_raw.median())
y = y_raw.to_numpy(dtype=np.float32)

X_df = df.drop(columns=[TARGET_COL])

# Keep numeric features only
X_df = X_df.select_dtypes(include=[np.number])
X_df = X_df.apply(pd.to_numeric, errors="coerce")
X_df = X_df.fillna(X_df.median(numeric_only=True)).fillna(0.0)

scaler = StandardScaler()
X = scaler.fit_transform(X_df.values).astype(np.float32)

rkf = RepeatedKFold(
    n_splits=5,
    n_repeats=5,
    random_state=SEED,
)

m_full = X.shape[1]

# -----------------------
# Optuna objective
# -----------------------
def objective(trial):
    seed_everything(SEED)

    # GO-LR hyperparameters
    go_metric = trial.suggest_categorical(
        "go_metric",
        ["correlation", "cosine", "manhattan", "euclidean", "kl_divergence"],
    )
    go_num_clusters = trial.suggest_int("go_num_clusters", 4, 12)
    go_refine_passes = trial.suggest_int("go_refine_passes", 1, 3)
    go_direction_select = trial.suggest_categorical(
        "go_direction_select",
        [True, False],
    )

    # Optional feature subsampling for high-dimensional datasets
    go_feat_subsample = trial.suggest_categorical(
        "go_feat_subsample",
        [None, 1000, 2000, 3000],
    )

    # NSC-pSP hyperparameters
    nsc_segmentation = trial.suggest_categorical(
        "nsc_segmentation",
        ["uniform", "largest_jump", "equal_mass"],
    )
    nsc_m_rule = trial.suggest_categorical(
        "nsc_m_rule",
        ["default", "idf", "gamma"],
    )
    nsc_tau = trial.suggest_categorical("nsc_tau", [0.95, 0.99, 0.9975])
    nsc_gamma = trial.suggest_float("nsc_gamma", 1.0, 3.0)
    nsc_beta = trial.suggest_float("nsc_beta", 0.0, 0.9)
    nsc_Mmin = trial.suggest_categorical("nsc_Mmin", [16, 32, 48, 64])
    nsc_Mmax = trial.suggest_categorical("nsc_Mmax", [128, 256, 384, 512, 640])
    nsc_lmin = trial.suggest_categorical("nsc_lmin", [8, 12, 16])
    assume_standardized = trial.suggest_categorical(
        "assume_standardized",
        [True, False],
    )

    tabpfn_seed = trial.suggest_categorical(
        "tabpfn_seed",
        [0, 1, 2, 3, 4, 42],
    )

    # Feature subsampling before GO-LR
    if go_feat_subsample is not None and int(go_feat_subsample) < m_full:
        rng = np.random.default_rng(SEED + 999)
        feat_idx = rng.choice(m_full, size=int(go_feat_subsample), replace=False)
        feat_idx.sort()
    else:
        feat_idx = np.arange(m_full)

    X_go = X[:, feat_idx]

    # Learn GO-LR ordering once per trial
    go = GraphFeatureOrdering(
        num_clusters=go_num_clusters,
        metric=go_metric,
        refine=True,
        direction_select=go_direction_select,
        refine_passes=go_refine_passes,
    )

    try:
        Pi_sub, _, _, _ = go.fit(
            X_go,
            seed=SEED,
            deterministic=True,
            use_cpu_kmeans=False,
        )
    except Exception:
        cleanup_cuda()
        Pi_sub, _, _, _ = go.fit(
            X_go,
            seed=SEED,
            deterministic=True,
            use_cpu_kmeans=True,
        )

    ordered_subset = feat_idx[np.array(Pi_sub, dtype=np.int64)].tolist()

    if len(feat_idx) < m_full:
        remaining = np.setdiff1d(np.arange(m_full), feat_idx, assume_unique=False)
        Pi_star = ordered_subset + remaining.tolist()
    else:
        Pi_star = ordered_subset

    Pi_star = list(map(int, Pi_star))

    head_cfg = TabPFN25Config(
        task_type="regression",
        num_classes=1,
        device=DEVICE,
        random_state=int(tabpfn_seed),
    )

    r2s = []

    for fold_id, (tr_idx, va_idx) in enumerate(rkf.split(X), start=1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        nsc = pidf_segpca(
            segmentation=nsc_segmentation,
            l_min=nsc_lmin,
            m_rule=nsc_m_rule,
            gamma=nsc_gamma,
            beta=nsc_beta,
            tau=nsc_tau,
            M_min=nsc_Mmin,
            M_max=nsc_Mmax,
            assume_standardized=assume_standardized,
            device=DEVICE,
        )

        deltas = None
        if nsc_segmentation != "uniform":
            deltas = compute_deltas_adjacent_corr(X_tr, Pi_star)

        X_tr_t = torch.from_numpy(X_tr)

        nsc.configure(
            Pi_star=Pi_star,
            X_train=X_tr_t,
            tau=nsc_tau,
            deltas=deltas,
        )

        Z_tr = nsc.compress(X_tr_t, mode="flatten").cpu().numpy()
        Z_va = nsc.compress(torch.from_numpy(X_va), mode="flatten").cpu().numpy()

        head = TabPFN25Head(head_cfg)
        head.fit(Z_tr, y_tr)

        pred = np.asarray(head.predict(Z_va), dtype=np.float32).reshape(-1)
        r2 = r2_score(y_va, pred)

        r2s.append(float(r2))

        trial.report(float(np.mean(r2s)), step=fold_id)

        if trial.should_prune():
            cleanup_cuda()
            raise optuna.TrialPruned()

        cleanup_cuda()

    return float(np.mean(r2s))


# -----------------------
# Run Optuna
# -----------------------
sampler = optuna.samplers.TPESampler(
    seed=SEED,
    multivariate=True,
    group=True,
)

pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)

study = optuna.create_study(
    direction="maximize",
    sampler=sampler,
    pruner=pruner,
)

study.optimize(
    objective,
    n_trials=N_TRIALS,
    show_progress_bar=True,
    gc_after_trial=True,
    n_jobs=1,
)

print("\nBest trial")
print(f"Best mean R2: {study.best_value:.6f}")

print("\nBest hyperparameters:")
for key, value in study.best_params.items():
    print(f"{key}: {value}")
```

### Example 9: Regression with the CSV Wrapper

This example runs GOTabPFN on the DrivFace regression dataset using the high-level `run_gotabpfn_csv` wrapper. The pipeline performs GO-LR feature ordering, NSC-pSP compression, and TabPFN-2.5 regression under 5×5 repeated cross-validation.

```python
import numpy as np
import torch

from gotabpfn import run_gotabpfn_csv


# -----------------------
# User settings
# -----------------------
DATA_FILE = "drivface.csv"  # change this to your dataset file name
TARGET_COL = "angle"        # change this to your regression target column
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------
# Fixed GO-LR hyperparameters
# -----------------------
GO_METRIC = "manhattan"
GO_NUM_CLUSTERS = 5
GO_REFINE_PASSES = 1
GO_DIRECTION_SELECT = False
GO_FEAT_SUBSAMPLE = 2000


# -----------------------
# Fixed NSC-pSP hyperparameters
# -----------------------
NSC_SEGMENTATION = "largest_jump"
NSC_M_RULE = "idf"
NSC_TAU = 0.99
NSC_GAMMA = 2.654390393837633
NSC_BETA = 0.043192175152615336
NSC_MMIN = 16
NSC_MMAX = 256
NSC_LMIN = 12
ASSUME_STANDARDIZED = True

TABPFN_SEED = 3


# -----------------------
# Run GOTabPFN
# -----------------------
results = run_gotabpfn_csv(
    csv_path=DATA_FILE,
    target_col=TARGET_COL,
    task_type="regression",
    non_numeric="drop",

    # 5x5 repeated cross-validation
    cv="5x5",
    seed=SEED,

    # GO-LR settings
    go_metric=GO_METRIC,
    go_num_clusters=GO_NUM_CLUSTERS,
    go_refine=True,
    go_direction_select=GO_DIRECTION_SELECT,
    go_refine_passes=GO_REFINE_PASSES,
    go_feat_subsample=GO_FEAT_SUBSAMPLE,

    # Try GPU KMeans first, then fall back to CPU KMeans if needed
    go_use_cpu_kmeans=False,
    go_fallback_cpu_kmeans=True,

    # NSC-pSP settings
    nsc_segmentation=NSC_SEGMENTATION,
    nsc_m_rule=NSC_M_RULE,
    nsc_tau=NSC_TAU,
    nsc_gamma=NSC_GAMMA,
    nsc_beta=NSC_BETA,
    nsc_M_min=NSC_MMIN,
    nsc_M_max=NSC_MMAX,
    nsc_l_min=NSC_LMIN,
    assume_standardized=ASSUME_STANDARDIZED,

    # TabPFN-2.5 head
    tabpfn_seed=TABPFN_SEED,
    device=DEVICE,

    return_predictions=True,
    verbose=False,
)


# -----------------------
# Access outputs
# -----------------------
print(results["summary"])

metrics_df = results["metrics_df"]
predictions_df = results["predictions_df"]

print("\nFold-wise metrics")
print(metrics_df)

print("\nPrediction preview")
print(predictions_df.head())

print("\nGO-LR ordering")
print(f"Learned GO-LR order length: {len(results['ordering'])}")
print("First 10 ordered features:")
print(results["ordered_feature_names"][:10])


# -----------------------
# Final 5x5 CV summary
# -----------------------
r2s = metrics_df["r2"].to_numpy()
rmses = metrics_df["rmse"].to_numpy()
maes = metrics_df["mae"].to_numpy()

print("\nFinal 5x5 CV results")
print(f"R2   : {np.mean(r2s):.4f} ± {np.std(r2s, ddof=1):.4f}")
print(f"RMSE : {np.mean(rmses):.4f} ± {np.std(rmses, ddof=1):.4f}")
print(f"MAE  : {np.mean(maes):.4f} ± {np.std(maes, ddof=1):.4f}")
```
- The returned dictionary contains:
```python
results["summary"]                # text summary of the run
results["metrics_df"]             # fold-wise R2, RMSE, MAE, and token count
results["predictions_df"]         # per-fold regression predictions
results["ordering"]               # learned GO-LR feature ordering
results["ordered_feature_names"]   # ordered feature names
results["num_features"]           # number of numeric input features
results["num_samples"]            # number of samples
```

### Example 10: GO-LR as an Ordering Metaheuristic

This example uses **GO-LR alone** as a feature ordering metaheuristic. Instead of running the full GOTabPFN pipeline, it tests the ordering module directly and reports ordering runtime, TSP-path cost, MinLA-style dispersion cost, the learned feature order, and the reordered feature table.

This is useful when you want to inspect the ordering quality, compare GO-LR against other ordering/metaheuristic methods, or export a reordered version of the dataset for downstream analysis.

```python
# ============================================================
# GO-LR standalone ordering test through the gotabpfn package
# Tests ordering runtime, TSP path cost, MinLA cost, and reordered features.
# Runtime may vary across machines/GPUs. 
# ============================================================

import os
import sys
import warnings
import importlib
import pandas as pd

warnings.filterwarnings(
    "ignore",
    message=".*pynvml package is deprecated.*",
    category=FutureWarning,
)

warnings.filterwarnings(
    "ignore",
    message=".*cumsum_cuda_kernel does not have a deterministic implementation.*",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message=".*Deterministic behavior was enabled.*CuBLAS.*",
    category=UserWarning,
)

# ------------------------------------------------------------
# Make current folder importable, useful for local notebooks
# ------------------------------------------------------------
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

# ------------------------------------------------------------
# Import package
# ------------------------------------------------------------
import gotabpfn
importlib.reload(gotabpfn)

print("[OK] Imported gotabpfn package.")

# ------------------------------------------------------------
# Config: GO-LR settings from the Colon ordering ablation
# ------------------------------------------------------------
SEED = 42
DATA_FILE = "coloncancer_encoded.csv"  # change your dataset file name
TARGET_COL = "label"                   # change your target column
DATASET_NAME = "Colon"

BEST_GO = {
    "metric": "euclidean",
    "num_clusters": 10,
    "refine_passes": 3,
    "direction_select": True,
}

OUT_PREFIX = "colon_golr_test"

# ------------------------------------------------------------
# Check files / package exports
# ------------------------------------------------------------
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

if getattr(gotabpfn, "run_golr_csv", None) is None:
    raise ImportError(
        "gotabpfn.run_golr_csv is not available. "
        "Check your gotabpfn installation and package exports."
    )

print(f"[OK] Found dataset: {DATA_FILE}")
print("[OK] Found gotabpfn.run_golr_csv")

# ------------------------------------------------------------
# Run GO-LR
# ------------------------------------------------------------
result = gotabpfn.run_golr_csv(
    csv_path=DATA_FILE,
    target_col=TARGET_COL,
    dataset_name=DATASET_NAME,
    metric=BEST_GO["metric"],
    num_clusters=BEST_GO["num_clusters"],
    refine=True,
    direction_select=BEST_GO["direction_select"],
    refine_passes=BEST_GO["refine_passes"],
    bins=32,
    seed=SEED,
    standardize=True,
    drop_non_numeric=True,
    use_cpu_kmeans=True,   # safer for notebook testing
    save_outputs=True,
    out_prefix=OUT_PREFIX,
)

# ------------------------------------------------------------
# Display metrics
# ------------------------------------------------------------
metrics = result["metrics"]
metrics_df = pd.DataFrame([metrics])

print("\n[GO-LR metrics]")
display(metrics_df)

# ------------------------------------------------------------
# Display learned ordering preview
# ------------------------------------------------------------
ordering_df = result["ordering_df"]

print("\n[Ordering preview]")
display(ordering_df.head(20))

# ------------------------------------------------------------
# Display reordered feature table preview
# ------------------------------------------------------------
reordered_df = result["reordered_df"]

print("\n[Reordered feature table preview]")
display(reordered_df.head())

# ------------------------------------------------------------
# Access important values directly
# ------------------------------------------------------------
Pi_star = result["ordering"]
runtime_sec = result["runtime_sec"]
tsp_cost = result["tsp_path_cost"]
minla_cost = result["minla_cost"]

print("\n[Direct values]")
print(f"Number of ordered features: {len(Pi_star)}")
print(f"Runtime seconds: {runtime_sec:.6f}")
print(f"TSP path cost: {tsp_cost:.6f}")
print(f"MinLA cost: {minla_cost:.6f}")

print("\n[SAVED]")
print(f"  - {OUT_PREFIX}_reordered.csv")
print(f"  - {OUT_PREFIX}_ordering.csv")
print(f"  - {OUT_PREFIX}_metrics.json")
```

Expected saved outputs:

- `colon_golr_test_reordered.csv`: the dataset with features reordered by GO-LR.
- `colon_golr_test_ordering.csv`: the learned feature order.
- `colon_golr_test_metrics.json`: runtime, TSP-path cost, MinLA cost, and related ordering diagnostics.

In this example, GO-LR is used as a standalone ordering metaheuristic. It constructs a graph over features, initializes an ordering using a TSP path-style heuristic, and then refines the order under a MinLA-style dispersion objective. Lower TSP-path and MinLA costs indicate stronger ordering quality under the corresponding surrogate criteria.

### Example 11: Checking NSC Compression Variants

This example tests the four NSC compression variants implemented in GOTabPFN:

- **NSC-pSP**: PCA-based intrinsic-dimensionality rule for selecting `M` + SegPCA pooling.
- **NSC-SP**: fixed `M` + SegPCA pooling.
- **NSC-P**: PCA-based intrinsic-dimensionality rule for selecting `M` + descriptor pooling.
- **NSC**: fixed `M` + descriptor pooling.

The script first checks whether a GO-LR ordering file already exists. If not, it runs GO-LR on the dataset and saves the ordering. Then it applies all four NSC variants using the same ordered feature axis and reports compression statistics such as the original feature count, compressed feature count, selected `M`, compression ratio, intrinsic dimensionality estimate, and runtime.

```python
# ============================================================
# Tests all four NSC variants:
#   - NSC-pSP
#   - NSC-SP
#   - NSC-P
#   - NSC
# ============================================================

import os
import sys
import gc
import warnings
import importlib

import numpy as np
import pandas as pd
import torch

# ------------------------------------------------------------
# Optional environment settings
# ------------------------------------------------------------
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

warnings.filterwarnings(
    "ignore",
    message=".*pynvml package is deprecated.*",
    category=FutureWarning,
)

warnings.filterwarnings(
    "ignore",
    message=".*cumsum_cuda_kernel does not have a deterministic implementation.*",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message=".*Deterministic behavior was enabled.*CuBLAS.*",
    category=UserWarning,
)

# ------------------------------------------------------------
# Make current folder importable, useful for local notebooks
# ------------------------------------------------------------
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

# ------------------------------------------------------------
# Import package
# ------------------------------------------------------------
import gotabpfn
importlib.reload(gotabpfn)

print("[OK] Imported gotabpfn package.")

# ------------------------------------------------------------
# User settings
# ------------------------------------------------------------
DATA_FILE = "coloncancer_encoded.csv"  # change your dataset file name
TARGET_COL = "label"                   # change your target column
DATASET_NAME = "Colon"

ORDERING_CSV = "colon_golr_ordering.csv"
SEED = 42

# Common NSC settings
NSC_COMMON = {
    "segmentation": "equal_mass",
    "m_rule": "idf",
    "gamma": 1.7570143129240916,
    "beta": 0.2244046472232107,
    "M_min": 64,
    "M_max": 384,
    "l_min": 16,
    "standardize_input": True,
    "drop_non_numeric": True,
    "mode": "flatten",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_outputs": True,
}

# Descriptor-pooling settings for NSC-P and NSC
DESC_COMMON = {
    "descriptor": "basic",
    "pooling": "learn_free",
}

TAU = 0.99

print(f"[INFO] Using device: {NSC_COMMON['device']}")

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()


def maybe_make_golr_ordering():
    """
    Use an existing GO-LR ordering file if available.
    Otherwise, run GO-LR and save a new ordering.
    If GO-LR wrapper is unavailable, return None and NSC uses identity ordering.
    """
    if os.path.exists(ORDERING_CSV):
        print(f"[OK] Found ordering file: {ORDERING_CSV}")
        return ORDERING_CSV

    if getattr(gotabpfn, "run_golr_csv", None) is None:
        print("[WARN] gotabpfn.run_golr_csv is unavailable.")
        print("[WARN] Falling back to identity ordering for all NSC variants.")
        return None

    print(f"[INFO] {ORDERING_CSV} not found. Running GO-LR...")

    gotabpfn.run_golr_csv(
        csv_path=DATA_FILE,
        target_col=TARGET_COL,
        dataset_name=DATASET_NAME,
        metric="euclidean",
        num_clusters=10,
        refine=True,
        direction_select=True,
        refine_passes=3,
        bins=32,
        seed=SEED,
        standardize=True,
        drop_non_numeric=True,
        use_cpu_kmeans=True,
        save_outputs=True,
        out_prefix="colon_golr",
    )

    if os.path.exists(ORDERING_CSV):
        print(f"[OK] Created ordering file: {ORDERING_CSV}")
        return ORDERING_CSV

    print("[WARN] GO-LR ran, but ordering file was not found.")
    print("[WARN] Falling back to identity ordering for all NSC variants.")
    return None


# ------------------------------------------------------------
# Check dataset and package exports
# ------------------------------------------------------------
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Missing dataset file: {DATA_FILE}")

required_exports = [
    "run_nsc_psp_csv",
    "run_nsc_sp_csv",
    "run_nsc_p_csv",
    "run_nsc_csv",
]

missing_exports = [x for x in required_exports if getattr(gotabpfn, x, None) is None]
if missing_exports:
    raise ImportError(
        "Missing required gotabpfn exports:\n"
        + "\n".join([f"  - {x}" for x in missing_exports])
    )

print(f"[OK] Found dataset: {DATA_FILE}")
print("[OK] Found all NSC wrapper exports.")

ordering_csv = maybe_make_golr_ordering()

# ------------------------------------------------------------
# Run all four NSC variants
# ------------------------------------------------------------
results = {}

print("\n" + "=" * 80)
print("Running NSC-pSP")
print("=" * 80)

results["NSC-pSP"] = gotabpfn.run_nsc_psp_csv(
    csv_path=DATA_FILE,
    target_col=TARGET_COL,
    ordering_csv=ordering_csv,
    dataset_name=DATASET_NAME,
    tau=TAU,
    out_prefix="colon_nsc_psp_test",
    **NSC_COMMON,
)

M_ref = int(results["NSC-pSP"]["metrics"]["M_selected"])
d_hat_ref = results["NSC-pSP"]["metrics"].get("d_hat_pca", None)

print(f"\n[REFERENCE from NSC-pSP] M_ref={M_ref}, d_hat_ref={d_hat_ref}")

cleanup()

print("\n" + "=" * 80)
print("Running NSC-SP")
print("=" * 80)

results["NSC-SP"] = gotabpfn.run_nsc_sp_csv(
    csv_path=DATA_FILE,
    target_col=TARGET_COL,
    ordering_csv=ordering_csv,
    dataset_name=DATASET_NAME,
    M=M_ref,
    out_prefix="colon_nsc_sp_test",
    **NSC_COMMON,
)

cleanup()

print("\n" + "=" * 80)
print("Running NSC-P")
print("=" * 80)

results["NSC-P"] = gotabpfn.run_nsc_p_csv(
    csv_path=DATA_FILE,
    target_col=TARGET_COL,
    ordering_csv=ordering_csv,
    dataset_name=DATASET_NAME,
    tau=TAU,
    out_prefix="colon_nsc_p_test",
    **NSC_COMMON,
    **DESC_COMMON,
)

cleanup()

print("\n" + "=" * 80)
print("Running NSC")
print("=" * 80)

results["NSC"] = gotabpfn.run_nsc_csv(
    csv_path=DATA_FILE,
    target_col=TARGET_COL,
    ordering_csv=ordering_csv,
    dataset_name=DATASET_NAME,
    M=M_ref,
    estimate_d_hat=False,
    out_prefix="colon_nsc_test",
    **NSC_COMMON,
    **DESC_COMMON,
)

cleanup()

# ------------------------------------------------------------
# Summary table
# ------------------------------------------------------------
summary_rows = []

for name, res in results.items():
    metrics = res["metrics"].copy()
    metrics["variant"] = name
    summary_rows.append(metrics)

summary_df = pd.DataFrame(summary_rows)

preferred_cols = [
    "variant",
    "dataset",
    "n",
    "m_original",
    "m_compressed",
    "compression_ratio",
    "M_selected",
    "idf",
    "d_hat_pca",
    "segmentation",
    "m_rule",
    "descriptor",
    "pooling",
    "runtime_sec",
    "ordering_source",
]

available_cols = [c for c in preferred_cols if c in summary_df.columns]
remaining_cols = [c for c in summary_df.columns if c not in available_cols]
summary_df = summary_df[available_cols + remaining_cols]

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

try:
    display(
        summary_df.style.format(
            {
                "compression_ratio": "{:.4g}",
                "idf": "{:.4g}",
                "d_hat_pca": "{:.4g}",
                "runtime_sec": "{:.4g}",
            },
            na_rep="NA",
        )
    )
except NameError:
    print(summary_df)

summary_df.to_csv("colon_all_nsc_variants_summary.csv", index=False)

# ------------------------------------------------------------
# Preview compressed outputs
# ------------------------------------------------------------
for name, res in results.items():
    print(f"\n[{name}] compressed_df preview:")
    try:
        display(res["compressed_df"].head())
    except NameError:
        print(res["compressed_df"].head())

print("\n[SAVED SUMMARY]")
print("  - colon_all_nsc_variants_summary.csv")

print("\n[SAVED COMPRESSED FILES]")
print("  - colon_nsc_psp_test_compressed.csv")
print("  - colon_nsc_sp_test_compressed.csv")
print("  - colon_nsc_p_test_compressed.csv")
print("  - colon_nsc_test_compressed.csv")

print("\n[SAVED SEGMENTS/METRICS]")
print("  - *_segments.csv")
print("  - *_metrics.json")
```

Expected saved outputs:

- `colon_all_nsc_variants_summary.csv`: summary table comparing all NSC variants.
- `colon_nsc_psp_test_compressed.csv`: compressed features from NSC-pSP.
- `colon_nsc_sp_test_compressed.csv`: compressed features from NSC-SP.
- `colon_nsc_p_test_compressed.csv`: compressed features from NSC-P.
- `colon_nsc_test_compressed.csv`: compressed features from NSC.
- `*_segments.csv`: segment boundaries used by each compression variant.
- `*_metrics.json`: compression statistics and runtime diagnostics.

This example is intended for checking the compression stage independently for final prediction. It helps verify that GO-LR ordering can be reused by different NSC variants and that high-dimensional feature matrices can be converted into compact meta-feature representations before downstream modeling.

### Example 12: Multiple-Dataset Ordering Diagnostics

This example runs GOTabPFN's dataset diagnostic utility on multiple CSV files. It computes high-dimensionality and ordering-related diagnostics such as feature-to-sample ratio, intrinsic dimensionality factor, feature ordering effectiveness score, locality/enrichment scores, and related metrics. The example first creates a few dummy high-dimensional CSV datasets so the code can be tested immediately. To use your own datasets, replace the `datasets` list with your CSV file paths, target columns, and dataset names.

```python
# This script:
#   1. Creates dummy high-dimensional CSV datasets.
#   2. Loads GOTabPFN's diagnostics module.
#   3. Runs diagnostics across multiple datasets.
#   4. Saves full and selected diagnostic tables.
#
# To use your own datasets, replace the `datasets` list below.
# ============================================================

import os
import sys
import gc
import random
import warnings
import importlib

warnings.filterwarnings("ignore")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------
# Make current folder importable, useful for local notebooks
# ------------------------------------------------------------
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())


# ------------------------------------------------------------
# Import gotabpfn and diagnostics module
# ------------------------------------------------------------
import gotabpfn
importlib.reload(gotabpfn)

diag = gotabpfn.load_dataset_diagnostics_module()

print("[OK] Imported gotabpfn package.")
print("[OK] Loaded dataset diagnostics module.")


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
SEED = 42

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)

seed_everything(SEED)


# ------------------------------------------------------------
# Create dummy high-dimensional datasets
# ------------------------------------------------------------
def create_dummy_csv(
    csv_path,
    target_col,
    n_samples,
    n_features,
    n_classes,
    n_informative,
    n_redundant,
    seed,
):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        class_sep=1.3,
        flip_y=0.02,
        random_state=seed,
    )

    feature_cols = [f"f{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_cols)
    df[target_col] = y

    # Add one non-numeric column to show that diagnostics can drop it safely.
    df["non_numeric_id"] = [f"id_{i}" for i in range(n_samples)]

    df.to_csv(csv_path, index=False)
    print(f"[CREATED] {csv_path}: n={n_samples}, m={n_features}, classes={n_classes}")


# Dummy datasets for immediate testing.
# You can delete this section when using your own CSV files.
create_dummy_csv(
    csv_path="dummy_hdlss_1.csv",
    target_col="label",
    n_samples=80,
    n_features=2000,
    n_classes=2,
    n_informative=80,
    n_redundant=40,
    seed=SEED + 1,
)

create_dummy_csv(
    csv_path="dummy_hdlss_2.csv",
    target_col="label",
    n_samples=120,
    n_features=5000,
    n_classes=3,
    n_informative=150,
    n_redundant=80,
    seed=SEED + 2,
)

create_dummy_csv(
    csv_path="dummy_text_like.csv",
    target_col="label",
    n_samples=500,
    n_features=3000,
    n_classes=2,
    n_informative=120,
    n_redundant=100,
    seed=SEED + 3,
)

create_dummy_csv(
    csv_path="dummy_image_feature_like.csv",
    target_col="label",
    n_samples=1000,
    n_features=2048,
    n_classes=5,
    n_informative=200,
    n_redundant=100,
    seed=SEED + 4,
)


# ------------------------------------------------------------
# Patch diagnostics loader:
# Drop target column, then keep numeric feature columns only.
# ------------------------------------------------------------
def load_numeric_csv_drop_non_numeric(
    csv_path,
    target_col=None,
    standardize=True,
):
    df = pd.read_csv(csv_path)

    target_col = diag._none_if_empty(target_col)

    if target_col is not None:
        if target_col not in df.columns:
            raise ValueError(
                f"Target column '{target_col}' not found in {csv_path}.\n"
                f"Available columns: {list(df.columns)}"
            )
        df = df.drop(columns=[target_col])

    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]

    dropped_cols = [
        c for c in df.columns
        if c not in numeric_cols
    ]

    if dropped_cols:
        print(f"[INFO] {csv_path}: dropped non-numeric columns: {dropped_cols}")

    if len(numeric_cols) < 2:
        raise ValueError(
            f"{csv_path}: expected at least two numeric feature columns after "
            f"dropping target/non-numeric columns. Found {len(numeric_cols)}."
        )

    X = df[numeric_cols].to_numpy(dtype=np.float32)

    X = np.nan_to_num(
        X,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    if standardize:
        X = StandardScaler().fit_transform(X).astype(np.float32)

    return X


# Replace original loader inside diagnostics module.
diag.load_numeric_csv = load_numeric_csv_drop_non_numeric


# ------------------------------------------------------------
# Dataset list
# ------------------------------------------------------------
# CHANGE THIS BLOCK FOR YOUR OWN DATASETS.
#
# Each entry should contain:
#   csv_path     : path to the CSV file
#   target_col   : target column to remove before diagnostics
#   dataset_name : display name in the output table
#
# Example for real datasets:
#
# datasets = [
#     {"csv_path": "cellcycle.csv", "target_col": "phase", "dataset_name": "Cell Cycle"},
#     {"csv_path": "BASEHOCK.csv", "target_col": "label", "dataset_name": "BASEHOCK"},
#     {"csv_path": "RELATHE.csv", "target_col": "label", "dataset_name": "RELATHE"},
#     {"csv_path": "PCMAC.csv", "target_col": "label", "dataset_name": "PCMAC"},
#     {"csv_path": "orlraws10P.csv", "target_col": "label", "dataset_name": "orlraws10P"},
# ]
datasets = [
    {
        "csv_path": "dummy_hdlss_1.csv",
        "target_col": "label",
        "dataset_name": "Dummy-HDLSS-1",
    },
    {
        "csv_path": "dummy_hdlss_2.csv",
        "target_col": "label",
        "dataset_name": "Dummy-HDLSS-2",
    },
    {
        "csv_path": "dummy_text_like.csv",
        "target_col": "label",
        "dataset_name": "Dummy-Text-Like",
    },
    {
        "csv_path": "dummy_image_feature_like.csv",
        "target_col": "label",
        "dataset_name": "Dummy-Image-Feature-Like",
    },
]


# ------------------------------------------------------------
# Check files
# ------------------------------------------------------------
missing = [d["csv_path"] for d in datasets if not os.path.exists(d["csv_path"])]

if missing:
    raise FileNotFoundError(
        "Missing dataset file(s):\n" + "\n".join([f"  - {f}" for f in missing])
    )

print("[OK] All dataset files found.")


# ------------------------------------------------------------
# Run diagnostics
# ------------------------------------------------------------
df_metrics = diag.analyze_many_csvs(
    datasets=datasets,
    out_csv="multi_dataset_ordering_metrics.csv",
    standardize=True,
    verbose=True,
)


# ------------------------------------------------------------
# Select final columns
# ------------------------------------------------------------
show_cols = [
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

available_cols = [c for c in show_cols if c in df_metrics.columns]
df_show = df_metrics[available_cols].copy()

# Save selected table
df_show.to_csv("multi_dataset_ordering_metrics_selected_columns.csv", index=False)

print("\n[SAVED]")
print("  - multi_dataset_ordering_metrics.csv")
print("  - multi_dataset_ordering_metrics_selected_columns.csv")


# ------------------------------------------------------------
# Pretty display
# ------------------------------------------------------------
format_dict = {
    "rho": "{:.4g}",
    "IDF_final": "{:.4g}",
    "FOE": "{:.4g}",
    "P_success": "{:.4g}",
    "Delta_AdjCoh": "{:.4g}",
    "Delta_HitRate": "{:.4g}",
    "Delta_Cut": "{:.4g}",
    "LES": "{:.4g}",
    "AUC": "{:.4g}",
}

try:
    display(
        df_show.style.format(
            {k: v for k, v in format_dict.items() if k in df_show.columns}
        )
    )
except NameError:
    print(df_show)
```

Expected saved outputs:

- `multi_dataset_ordering_metrics.csv`: full diagnostic table.
- `multi_dataset_ordering_metrics_selected_columns.csv`: compact selected-column summary.

To use your own datasets, only change this block:

```python
datasets = [
    {
        "csv_path": "your_dataset_1.csv",
        "target_col": "your_target_column",
        "dataset_name": "Your Dataset 1",
    },
    {
        "csv_path": "your_dataset_2.csv",
        "target_col": "your_target_column",
        "dataset_name": "Your Dataset 2",
    },
]
```

The diagnostics automatically drop the target column and any non-numeric feature columns before computing ordering-related metrics.

### Example 13: Single-Dataset Ordering Diagnostics

This example runs GOTabPFN's dataset-diagnostic utility on one CSV file. It computes high-dimensionality and ordering-related diagnostics such as feature-to-sample ratio, intrinsic dimensionality factor, feature ordering effectiveness score, locality/enrichment scores, and related metrics. The example first creates a dummy high-dimensional CSV dataset so the code can be tested immediately. To use your own dataset, change only the `DATA_FILE`, `TARGET_COL`, and `DATASET_NAME` variables.

```python
# ============================================================
# This script:
#   1. Creates one dummy high-dimensional CSV dataset.
#   2. Loads GOTabPFN's diagnostics module.
#   3. Runs ordering diagnostics for one dataset.
#   4. Saves full and selected diagnostic tables.
#
# To use your own dataset, change DATA_FILE, TARGET_COL, and DATASET_NAME.
# ============================================================

import os
import sys
import random
import warnings
import importlib

warnings.filterwarnings("ignore")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
SEED = 42

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)

seed_everything(SEED)


# ------------------------------------------------------------
# Optional: create a dummy high-dimensional dataset
# ------------------------------------------------------------
# You can delete this section when using your own CSV file.
def create_dummy_single_dataset(
    csv_path,
    target_col,
    n_samples=120,
    n_features=3000,
    n_classes=3,
    n_informative=150,
    n_redundant=80,
    seed=42,
):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        class_sep=1.3,
        flip_y=0.02,
        random_state=seed,
    )

    feature_cols = [f"f{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_cols)
    df[target_col] = y

    # Add one non-numeric column to show that diagnostics can drop it safely.
    df["non_numeric_id"] = [f"id_{i}" for i in range(n_samples)]

    df.to_csv(csv_path, index=False)
    print(f"[CREATED] {csv_path}: n={n_samples}, m={n_features}, classes={n_classes}")


# ------------------------------------------------------------
# User input: change these three lines for your own dataset
# ------------------------------------------------------------
DATA_FILE = "dummy_single_diagnostics.csv"
TARGET_COL = "label"
DATASET_NAME = "Dummy Single Dataset"

# Create dummy CSV for immediate testing.
# Comment this out when using your own existing CSV file.
create_dummy_single_dataset(
    csv_path=DATA_FILE,
    target_col=TARGET_COL,
    n_samples=120,
    n_features=3000,
    n_classes=3,
    n_informative=150,
    n_redundant=80,
    seed=SEED,
)

OUT_CSV = f"{DATASET_NAME.replace(' ', '_').replace('/', '_')}_single_ordering_metrics.csv"


# ------------------------------------------------------------
# Make current folder importable, useful for local notebooks
# ------------------------------------------------------------
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())


# ------------------------------------------------------------
# Import gotabpfn and diagnostics module
# ------------------------------------------------------------
import gotabpfn
importlib.reload(gotabpfn)

diag = gotabpfn.load_dataset_diagnostics_module()

print("[OK] Imported gotabpfn package.")
print("[OK] Loaded dataset diagnostics module.")


# ------------------------------------------------------------
# Patch diagnostics loader:
# Drop target column, then keep numeric feature columns only.
# ------------------------------------------------------------
def load_numeric_csv_drop_non_numeric(
    csv_path,
    target_col=None,
    standardize=True,
):
    df = pd.read_csv(csv_path)

    target_col = diag._none_if_empty(target_col)

    if target_col is not None:
        if target_col not in df.columns:
            raise ValueError(
                f"Target column '{target_col}' not found in {csv_path}.\n"
                f"Available columns: {list(df.columns)}"
            )
        df = df.drop(columns=[target_col])

    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]

    dropped_cols = [
        c for c in df.columns
        if c not in numeric_cols
    ]

    if dropped_cols:
        print(f"[INFO] {csv_path}: dropped non-numeric columns: {dropped_cols}")

    if len(numeric_cols) < 2:
        raise ValueError(
            f"{csv_path}: expected at least two numeric feature columns after "
            f"dropping target/non-numeric columns. Found {len(numeric_cols)}."
        )

    X = df[numeric_cols].to_numpy(dtype=np.float32)

    X = np.nan_to_num(
        X,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    if standardize:
        X = StandardScaler().fit_transform(X).astype(np.float32)

    return X


# Replace original loader inside diagnostics module.
diag.load_numeric_csv = load_numeric_csv_drop_non_numeric


# ------------------------------------------------------------
# Check file
# ------------------------------------------------------------
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Missing dataset file: {DATA_FILE}")

print(f"[OK] Dataset file found: {DATA_FILE}")


# ------------------------------------------------------------
# Run diagnostics for one dataset
# ------------------------------------------------------------
df_metrics = diag.analyze_csv_ordering_metrics(
    csv_path=DATA_FILE,
    target_col=TARGET_COL,
    dataset_name=DATASET_NAME,
    out_csv=OUT_CSV,
    standardize=True,
    verbose=True,
)


# ------------------------------------------------------------
# Select final columns
# ------------------------------------------------------------
show_cols = [
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

available_cols = [c for c in show_cols if c in df_metrics.columns]
df_show = df_metrics[available_cols].copy()

# Save selected table
selected_csv = OUT_CSV.replace(".csv", "_selected_columns.csv")
df_show.to_csv(selected_csv, index=False)

print("\n[SAVED]")
print(f"  - {OUT_CSV}")
print(f"  - {selected_csv}")


# ------------------------------------------------------------
# Pretty display
# ------------------------------------------------------------
format_dict = {
    "rho": "{:.4g}",
    "IDF_final": "{:.4g}",
    "FOE": "{:.4g}",
    "P_success": "{:.4g}",
    "Delta_AdjCoh": "{:.4g}",
    "Delta_HitRate": "{:.4g}",
    "Delta_Cut": "{:.4g}",
    "LES": "{:.4g}",
    "AUC": "{:.4g}",
}

try:
    display(
        df_show.style.format(
            {k: v for k, v in format_dict.items() if k in df_show.columns}
        )
    )
except NameError:
    print(df_show)
```

Expected saved outputs:

- `Dummy_Single_Dataset_single_ordering_metrics.csv`: full diagnostic table for the dataset.
- `Dummy_Single_Dataset_single_ordering_metrics_selected_columns.csv`: compact selected-column summary.

To use your own dataset, only change this block:

```python
DATA_FILE = "your_dataset.csv"
TARGET_COL = "your_target_column"
DATASET_NAME = "Your Dataset Name"
```

Then comment out or delete this dummy-data creation block:

```python
create_dummy_single_dataset(...)
```

The diagnostics automatically drop the target column and any non-numeric feature columns before computing ordering-related metrics.

## Acknowledgements

This work was supported in part by the U.S. National Science Foundation under Awards #1920920, #2125872, and #2223793. We thank the anonymous ICML reviewers for their valuable feedback and suggestions.

# Our Related Works Involving Tabular Data

### BSTabDiff

Our generative modeling framework for high-dimensional low-sample-size tabular data:
- **BSTabDiff: Block-Subunit Diffusion Priors for High-Dimensional Tabular Data Generation**

- GitHub: https://github.com/zadid6pretam/BSTabDiff

- OpenReview: https://openreview.net/forum?id=RKNDy0KhGT


```bibtex
@inproceedings{habib2026bstabdiff,
  title     = {BSTabDiff: Block-Subunit Diffusion Priors for High-Dimensional Tabular Data Generation},
  author    = {Habib, Al Zadid Sultan Bin and Ahamed, Md Younus and Gyawali, Prashnna Kumar and Doretto, Gianfranco and Adjeroh, Donald A.},
  booktitle = {ICLR 2026 2nd Workshop on Deep Generative Models in Machine Learning: Theory, Principle and Efficacy (DeLTa)},
  year      = {2026}
}
```
- If you are interested in high-dimensional tabular synthesis, block-subunit generation, and diffusion/flow priors for HDLSS tabular data, please also refer to the BSTabDiff repository and paper.

### iStructTab

Our structured feature sequencing framework for multimodal learning with image and tabular data. This work involves feature sequencing or ordering for multimodal image-tabular representation learning.

- **iStructTab: Structured Feature Sequencing for Multimodal Learning of Image and Tabular Data**  

- GitHub: https://github.com/zadid6pretam/iStructTab

```bibtex
@inproceedings{habib2026istructtab,
  title     = {iStructTab: Structured Feature Sequencing for Multimodal Learning of Image and Tabular Data},
  author    = {Habib, Al Zadid Sultan Bin and Ahamed, Md Younus and Gyawali, Prashnna and Doretto, Gianfranco and Adjeroh, Donald A.},
  booktitle = {Proceedings of the 28th International Conference on Pattern Recognition},
  year      = {2026},
  address   = {Lyon, France}
}
```
- If you are interested in structured feature sequencing, multimodal fusion of image and tabular data (the integration problem), and feature order-aware tabular representation learning, please also refer to the iStructTab repository and paper.

## DynaTab

One of our older works on learned feature ordering for high-dimensional tabular data:

- **DynaTab: Dynamic Feature Ordering as Neural Rewiring for High-Dimensional Tabular Data**

- GitHub: https://github.com/zadid6pretam/DynaTab
- Paper Link: https://proceedings.mlr.press/v308/habib26a.html

Bibtex:
```bash
@InProceedings{dynatab,
  title = 	 {{DynaTab: Dynamic Feature Ordering as Neural Rewiring for High-Dimensional Tabular Data}},
  author =       {Habib, Al Zadid Sultan Bin and Doretto, Gianfranco and Adjeroh, Donald A.},
  booktitle = 	 {{Proceedings of the First Workshop on NeuroAI Multimodal Intelligence @ AAAI 2026}},
  pages     = {27--57},
  year      = {2026},
  volume    = {308},
  series    = {{Proceedings of Machine Learning Research}},
  publisher = {PMLR},
  url = 	 {https://proceedings.mlr.press/v308/habib26a.html}
}
```
- If you are interested in learned feature ordering, neural rewiring for high-dimensional tabular data, and sequential backbone design for HDLSS settings, please also refer to the benchmark study in DynaTab repository and paper.


### TabSeq

Our earlier work on sequential modeling for tabular data:

- **TabSeq: A Framework for Deep Learning on Tabular Data via Sequential Ordering**  

-  GitHub: https://github.com/zadid6pretam/TabSeq  

-  Springer ICPR 2024 proceedings: https://link.springer.com/chapter/10.1007/978-3-031-78128-5_27

-  arXiv: https://arxiv.org/abs/2410.13203

```bibtex
@inproceedings{habib2024tabseq,
  title={TabSeq: A Framework for Deep Learning on Tabular Data via Sequential Ordering},
  author={Habib, Al Zadid Sultan Bin and Wang, Kesheng and Hartley, Mary-Anne and Doretto, Gianfranco and A. Adjeroh, Donald},
  booktitle={International Conference on Pattern Recognition},
  pages={418--434},
  year={2024},
  organization={Springer}
}
```
- If you are interested in sequential feature ordering for tabular data, deep sequential backbones, and early feature ordering-based tabular modeling, please also refer to the TabSeq repository and paper.


----------------------------------------------------------------------------------------------------------------------------------------------------------


### ZAYAN

This repository corresponds to our separate collaborative work on tabular remote sensing and environmental data:
- **ZAYAN: Disentangled Contrastive Transformer for Tabular Remote Sensing Data**

- GitHub: https://github.com/zadid6pretam/ZAYAN

- arXiv: https://arxiv.org/abs/2604.27606

```bibtex
@inproceedings{habib2026zayan,
  title     = {ZAYAN: Disentangled Contrastive Transformer for Tabular Remote Sensing Data},
  author    = {Habib, Al Zadid Sultan Bin and Tasnim, Tanpia and Islam, Md. Ekramul and Tabasum, Muntasir},
  booktitle = {Proceedings of the 28th International Conference on Pattern Recognition},
  year      = {2026},
  address   = {Lyon, France}
}
```
- ZAYAN focuses on feature-level contrastive learning and Transformer-based classification for tabular remote sensing and environmental datasets.
- Note: ZAYAN is not part of my PhD dissertation work on high-dimensional tabular learning and HDLSS modeling; it was developed as a separate collaborative project.

## Contact

For any questions, issues, or suggestions related to this repository, please feel free to contact us or open an issue on GitHub.
