```python
pip install --upgrade gotabpfn
```

    Requirement already satisfied: gotabpfn in /usr/local/lib/python3.12/dist-packages (0.1.10)
    Collecting gotabpfn
      Downloading gotabpfn-0.1.11-py3-none-any.whl.metadata (116 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m116.1/116.1 kB[0m [31m3.9 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: numpy>=1.23 in /usr/local/lib/python3.12/dist-packages (from gotabpfn) (2.0.2)
    Requirement already satisfied: pandas>=1.5 in /usr/local/lib/python3.12/dist-packages (from gotabpfn) (2.2.2)
    Requirement already satisfied: scipy>=1.11 in /usr/local/lib/python3.12/dist-packages (from gotabpfn) (1.16.3)
    Requirement already satisfied: scikit-learn>=1.2 in /usr/local/lib/python3.12/dist-packages (from gotabpfn) (1.6.1)
    Requirement already satisfied: tqdm>=4.64 in /usr/local/lib/python3.12/dist-packages (from gotabpfn) (4.67.3)
    Requirement already satisfied: optuna>=3.5 in /usr/local/lib/python3.12/dist-packages (from gotabpfn) (4.8.0)
    Requirement already satisfied: torch>=2.1 in /usr/local/lib/python3.12/dist-packages (from gotabpfn) (2.11.0+cu128)
    Requirement already satisfied: tabpfn==6.3.1 in /usr/local/lib/python3.12/dist-packages (from gotabpfn) (6.3.1)
    Requirement already satisfied: kmeans-gpu==0.0.5 in /usr/local/lib/python3.12/dist-packages (from gotabpfn) (0.0.5)
    Requirement already satisfied: matplotlib>=3.7 in /usr/local/lib/python3.12/dist-packages (from gotabpfn) (3.10.0)
    Requirement already satisfied: typing_extensions>=4.12.0 in /usr/local/lib/python3.12/dist-packages (from tabpfn==6.3.1->gotabpfn) (4.15.0)
    Requirement already satisfied: einops<0.9,>=0.2.0 in /usr/local/lib/python3.12/dist-packages (from tabpfn==6.3.1->gotabpfn) (0.8.2)
    Requirement already satisfied: huggingface-hub<2,>=0.19.0 in /usr/local/lib/python3.12/dist-packages (from tabpfn==6.3.1->gotabpfn) (1.16.1)
    Requirement already satisfied: pydantic>=2.8.0 in /usr/local/lib/python3.12/dist-packages (from tabpfn==6.3.1->gotabpfn) (2.12.3)
    Requirement already satisfied: pydantic-settings>=2.10.1 in /usr/local/lib/python3.12/dist-packages (from tabpfn==6.3.1->gotabpfn) (2.14.1)
    Requirement already satisfied: eval-type-backport>=0.2.2 in /usr/local/lib/python3.12/dist-packages (from tabpfn==6.3.1->gotabpfn) (0.3.1)
    Requirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.12/dist-packages (from tabpfn==6.3.1->gotabpfn) (1.5.3)
    Requirement already satisfied: tabpfn-common-utils>=0.2.13 in /usr/local/lib/python3.12/dist-packages (from tabpfn-common-utils[telemetry-interactive]>=0.2.13->tabpfn==6.3.1->gotabpfn) (0.2.21)
    Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.12/dist-packages (from matplotlib>=3.7->gotabpfn) (1.3.3)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.12/dist-packages (from matplotlib>=3.7->gotabpfn) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.12/dist-packages (from matplotlib>=3.7->gotabpfn) (4.63.0)
    Requirement already satisfied: kiwisolver>=1.3.1 in /usr/local/lib/python3.12/dist-packages (from matplotlib>=3.7->gotabpfn) (1.5.0)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.12/dist-packages (from matplotlib>=3.7->gotabpfn) (26.2)
    Requirement already satisfied: pillow>=8 in /usr/local/lib/python3.12/dist-packages (from matplotlib>=3.7->gotabpfn) (11.3.0)
    Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.12/dist-packages (from matplotlib>=3.7->gotabpfn) (3.3.2)
    Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.12/dist-packages (from matplotlib>=3.7->gotabpfn) (2.9.0.post0)
    Requirement already satisfied: alembic>=1.5.0 in /usr/local/lib/python3.12/dist-packages (from optuna>=3.5->gotabpfn) (1.18.4)
    Requirement already satisfied: colorlog in /usr/local/lib/python3.12/dist-packages (from optuna>=3.5->gotabpfn) (6.10.1)
    Requirement already satisfied: sqlalchemy>=1.4.2 in /usr/local/lib/python3.12/dist-packages (from optuna>=3.5->gotabpfn) (2.0.49)
    Requirement already satisfied: PyYAML in /usr/local/lib/python3.12/dist-packages (from optuna>=3.5->gotabpfn) (6.0.3)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.12/dist-packages (from pandas>=1.5->gotabpfn) (2025.2)
    Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.12/dist-packages (from pandas>=1.5->gotabpfn) (2026.2)
    Requirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.12/dist-packages (from scikit-learn>=1.2->gotabpfn) (3.6.0)
    Requirement already satisfied: filelock in /usr/local/lib/python3.12/dist-packages (from torch>=2.1->gotabpfn) (3.29.0)
    Requirement already satisfied: setuptools<82 in /usr/local/lib/python3.12/dist-packages (from torch>=2.1->gotabpfn) (75.2.0)
    Requirement already satisfied: sympy>=1.13.3 in /usr/local/lib/python3.12/dist-packages (from torch>=2.1->gotabpfn) (1.14.0)
    Requirement already satisfied: networkx>=2.5.1 in /usr/local/lib/python3.12/dist-packages (from torch>=2.1->gotabpfn) (3.6.1)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.12/dist-packages (from torch>=2.1->gotabpfn) (3.1.6)
    Requirement already satisfied: fsspec>=0.8.5 in /usr/local/lib/python3.12/dist-packages (from torch>=2.1->gotabpfn) (2025.3.0)
    Requirement already satisfied: cuda-toolkit==12.8.1 in /usr/local/lib/python3.12/dist-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==12.8.1; platform_system == "Linux"->torch>=2.1->gotabpfn) (12.8.1)
    Requirement already satisfied: cuda-bindings<13,>=12.9.4 in /usr/local/lib/python3.12/dist-packages (from torch>=2.1->gotabpfn) (12.9.6)
    Requirement already satisfied: nvidia-cudnn-cu12==9.19.0.56 in /usr/local/lib/python3.12/dist-packages (from torch>=2.1->gotabpfn) (9.19.0.56)
    Requirement already satisfied: nvidia-cusparselt-cu12==0.7.1 in /usr/local/lib/python3.12/dist-packages (from torch>=2.1->gotabpfn) (0.7.1)
    Requirement already satisfied: nvidia-nccl-cu12==2.28.9 in /usr/local/lib/python3.12/dist-packages (from torch>=2.1->gotabpfn) (2.28.9)
    Requirement already satisfied: nvidia-nvshmem-cu12==3.4.5 in /usr/local/lib/python3.12/dist-packages (from torch>=2.1->gotabpfn) (3.4.5)
    Requirement already satisfied: triton==3.6.0 in /usr/local/lib/python3.12/dist-packages (from torch>=2.1->gotabpfn) (3.6.0)
    Requirement already satisfied: nvidia-cublas-cu12==12.8.4.1.* in /usr/local/lib/python3.12/dist-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==12.8.1; platform_system == "Linux"->torch>=2.1->gotabpfn) (12.8.4.1)
    Requirement already satisfied: nvidia-cuda-runtime-cu12==12.8.90.* in /usr/local/lib/python3.12/dist-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==12.8.1; platform_system == "Linux"->torch>=2.1->gotabpfn) (12.8.90)
    Requirement already satisfied: nvidia-cufft-cu12==11.3.3.83.* in /usr/local/lib/python3.12/dist-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==12.8.1; platform_system == "Linux"->torch>=2.1->gotabpfn) (11.3.3.83)
    Requirement already satisfied: nvidia-cufile-cu12==1.13.1.3.* in /usr/local/lib/python3.12/dist-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==12.8.1; platform_system == "Linux"->torch>=2.1->gotabpfn) (1.13.1.3)
    Requirement already satisfied: nvidia-cuda-cupti-cu12==12.8.90.* in /usr/local/lib/python3.12/dist-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==12.8.1; platform_system == "Linux"->torch>=2.1->gotabpfn) (12.8.90)
    Requirement already satisfied: nvidia-curand-cu12==10.3.9.90.* in /usr/local/lib/python3.12/dist-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==12.8.1; platform_system == "Linux"->torch>=2.1->gotabpfn) (10.3.9.90)
    Requirement already satisfied: nvidia-cusolver-cu12==11.7.3.90.* in /usr/local/lib/python3.12/dist-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==12.8.1; platform_system == "Linux"->torch>=2.1->gotabpfn) (11.7.3.90)
    Requirement already satisfied: nvidia-cusparse-cu12==12.5.8.93.* in /usr/local/lib/python3.12/dist-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==12.8.1; platform_system == "Linux"->torch>=2.1->gotabpfn) (12.5.8.93)
    Requirement already satisfied: nvidia-nvjitlink-cu12==12.8.93.* in /usr/local/lib/python3.12/dist-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==12.8.1; platform_system == "Linux"->torch>=2.1->gotabpfn) (12.8.93)
    Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.8.93.* in /usr/local/lib/python3.12/dist-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==12.8.1; platform_system == "Linux"->torch>=2.1->gotabpfn) (12.8.93)
    Requirement already satisfied: nvidia-nvtx-cu12==12.8.90.* in /usr/local/lib/python3.12/dist-packages (from cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==12.8.1; platform_system == "Linux"->torch>=2.1->gotabpfn) (12.8.90)
    Requirement already satisfied: Mako in /usr/local/lib/python3.12/dist-packages (from alembic>=1.5.0->optuna>=3.5->gotabpfn) (1.3.12)
    Requirement already satisfied: cuda-pathfinder~=1.1 in /usr/local/lib/python3.12/dist-packages (from cuda-bindings<13,>=12.9.4->torch>=2.1->gotabpfn) (1.5.4)
    Requirement already satisfied: hf-xet<2.0.0,>=1.4.3 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub<2,>=0.19.0->tabpfn==6.3.1->gotabpfn) (1.5.0)
    Requirement already satisfied: httpx<1,>=0.23.0 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub<2,>=0.19.0->tabpfn==6.3.1->gotabpfn) (0.28.1)
    Requirement already satisfied: typer>=0.20.0 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub<2,>=0.19.0->tabpfn==6.3.1->gotabpfn) (0.25.1)
    Requirement already satisfied: annotated-types>=0.6.0 in /usr/local/lib/python3.12/dist-packages (from pydantic>=2.8.0->tabpfn==6.3.1->gotabpfn) (0.7.0)
    Requirement already satisfied: pydantic-core==2.41.4 in /usr/local/lib/python3.12/dist-packages (from pydantic>=2.8.0->tabpfn==6.3.1->gotabpfn) (2.41.4)
    Requirement already satisfied: typing-inspection>=0.4.2 in /usr/local/lib/python3.12/dist-packages (from pydantic>=2.8.0->tabpfn==6.3.1->gotabpfn) (0.4.2)
    Requirement already satisfied: python-dotenv>=0.21.0 in /usr/local/lib/python3.12/dist-packages (from pydantic-settings>=2.10.1->tabpfn==6.3.1->gotabpfn) (1.2.2)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.12/dist-packages (from python-dateutil>=2.7->matplotlib>=3.7->gotabpfn) (1.17.0)
    Requirement already satisfied: greenlet>=1 in /usr/local/lib/python3.12/dist-packages (from sqlalchemy>=1.4.2->optuna>=3.5->gotabpfn) (3.5.1)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.12/dist-packages (from sympy>=1.13.3->torch>=2.1->gotabpfn) (1.3.0)
    Requirement already satisfied: nvidia-ml-py>=13.590.48 in /usr/local/lib/python3.12/dist-packages (from tabpfn-common-utils>=0.2.13->tabpfn-common-utils[telemetry-interactive]>=0.2.13->tabpfn==6.3.1->gotabpfn) (13.595.45)
    Requirement already satisfied: platformdirs>=4 in /usr/local/lib/python3.12/dist-packages (from tabpfn-common-utils>=0.2.13->tabpfn-common-utils[telemetry-interactive]>=0.2.13->tabpfn==6.3.1->gotabpfn) (4.9.6)
    Requirement already satisfied: posthog>=6.7 in /usr/local/lib/python3.12/dist-packages (from tabpfn-common-utils>=0.2.13->tabpfn-common-utils[telemetry-interactive]>=0.2.13->tabpfn==6.3.1->gotabpfn) (7.16.2)
    Requirement already satisfied: requests>=2.32.5 in /usr/local/lib/python3.12/dist-packages (from tabpfn-common-utils>=0.2.13->tabpfn-common-utils[telemetry-interactive]>=0.2.13->tabpfn==6.3.1->gotabpfn) (2.34.2)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.12/dist-packages (from jinja2->torch>=2.1->gotabpfn) (3.0.3)
    Requirement already satisfied: anyio in /usr/local/lib/python3.12/dist-packages (from httpx<1,>=0.23.0->huggingface-hub<2,>=0.19.0->tabpfn==6.3.1->gotabpfn) (4.13.0)
    Requirement already satisfied: certifi in /usr/local/lib/python3.12/dist-packages (from httpx<1,>=0.23.0->huggingface-hub<2,>=0.19.0->tabpfn==6.3.1->gotabpfn) (2026.5.20)
    Requirement already satisfied: httpcore==1.* in /usr/local/lib/python3.12/dist-packages (from httpx<1,>=0.23.0->huggingface-hub<2,>=0.19.0->tabpfn==6.3.1->gotabpfn) (1.0.9)
    Requirement already satisfied: idna in /usr/local/lib/python3.12/dist-packages (from httpx<1,>=0.23.0->huggingface-hub<2,>=0.19.0->tabpfn==6.3.1->gotabpfn) (3.15)
    Requirement already satisfied: h11>=0.16 in /usr/local/lib/python3.12/dist-packages (from httpcore==1.*->httpx<1,>=0.23.0->huggingface-hub<2,>=0.19.0->tabpfn==6.3.1->gotabpfn) (0.16.0)
    Requirement already satisfied: backoff>=1.10.0 in /usr/local/lib/python3.12/dist-packages (from posthog>=6.7->tabpfn-common-utils>=0.2.13->tabpfn-common-utils[telemetry-interactive]>=0.2.13->tabpfn==6.3.1->gotabpfn) (2.2.1)
    Requirement already satisfied: distro>=1.5.0 in /usr/local/lib/python3.12/dist-packages (from posthog>=6.7->tabpfn-common-utils>=0.2.13->tabpfn-common-utils[telemetry-interactive]>=0.2.13->tabpfn==6.3.1->gotabpfn) (1.9.0)
    Requirement already satisfied: charset_normalizer<4,>=2 in /usr/local/lib/python3.12/dist-packages (from requests>=2.32.5->tabpfn-common-utils>=0.2.13->tabpfn-common-utils[telemetry-interactive]>=0.2.13->tabpfn==6.3.1->gotabpfn) (3.4.7)
    Requirement already satisfied: urllib3<3,>=1.26 in /usr/local/lib/python3.12/dist-packages (from requests>=2.32.5->tabpfn-common-utils>=0.2.13->tabpfn-common-utils[telemetry-interactive]>=0.2.13->tabpfn==6.3.1->gotabpfn) (2.5.0)
    Requirement already satisfied: click>=8.2.1 in /usr/local/lib/python3.12/dist-packages (from typer>=0.20.0->huggingface-hub<2,>=0.19.0->tabpfn==6.3.1->gotabpfn) (8.4.0)
    Requirement already satisfied: shellingham>=1.3.0 in /usr/local/lib/python3.12/dist-packages (from typer>=0.20.0->huggingface-hub<2,>=0.19.0->tabpfn==6.3.1->gotabpfn) (1.5.4)
    Requirement already satisfied: rich>=13.8.0 in /usr/local/lib/python3.12/dist-packages (from typer>=0.20.0->huggingface-hub<2,>=0.19.0->tabpfn==6.3.1->gotabpfn) (13.9.4)
    Requirement already satisfied: annotated-doc>=0.0.2 in /usr/local/lib/python3.12/dist-packages (from typer>=0.20.0->huggingface-hub<2,>=0.19.0->tabpfn==6.3.1->gotabpfn) (0.0.4)
    Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.12/dist-packages (from rich>=13.8.0->typer>=0.20.0->huggingface-hub<2,>=0.19.0->tabpfn==6.3.1->gotabpfn) (4.2.0)
    Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.12/dist-packages (from rich>=13.8.0->typer>=0.20.0->huggingface-hub<2,>=0.19.0->tabpfn==6.3.1->gotabpfn) (2.20.0)
    Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.12/dist-packages (from markdown-it-py>=2.2.0->rich>=13.8.0->typer>=0.20.0->huggingface-hub<2,>=0.19.0->tabpfn==6.3.1->gotabpfn) (0.1.2)
    Downloading gotabpfn-0.1.11-py3-none-any.whl (84 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m84.7/84.7 kB[0m [31m3.7 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: gotabpfn
      Attempting uninstall: gotabpfn
        Found existing installation: gotabpfn 0.1.10
        Uninstalling gotabpfn-0.1.10:
          Successfully uninstalled gotabpfn-0.1.10
    Successfully installed gotabpfn-0.1.11





```python
pip show gotabpfn
```

    Name: gotabpfn
    Version: 0.1.11
    Summary: GOTabPFN: From Feature Ordering to Compact Tokenization for Tabular Foundation Models on High-Dimensional Data
    Home-page: https://github.com/zadid6pretam/GOTabPFN
    Author: Al Zadid Sultan Bin Habib, Md Younus Ahamed, Prashnna Kumar Gyawali, Gianfranco Doretto, Donald A. Adjeroh
    Author-email: 
    License: MIT
    Location: /usr/local/lib/python3.12/dist-packages
    Requires: kmeans-gpu, matplotlib, numpy, optuna, pandas, scikit-learn, scipy, tabpfn, torch, tqdm
    Required-by: 


# DrivFace - Regression || Single Wrapper


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

    GOTabPFN completed successfully.
    
    Task type: regression
    Evaluation: 5x5 repeated CV
    Samples: 606
    Original numeric features: 6400
    GO-LR metric: manhattan
    GO-LR KMeans mode: gpu
    GO-LR feature subsample: 2000
    NSC segmentation: largest_jump
    NSC M rule: idf
    NSC bypass_if_m_leq: 0
    Device: cuda
    Elapsed time: 132.81 seconds
    
    Metric summary:
    r2: 0.6548 ± 0.0995
    rmse: 8.2695 ± 1.4722
    mae: 3.6689 ± 0.6156
    nsc_tokens: 256.0000 ± 0.0000
    
    Fold-wise metrics
        fold        r2       rmse       mae  nsc_tokens
    0      1  0.512414  10.792954  4.808915         256
    1      2  0.702385   6.701759  2.874984         256
    2      3  0.767160   7.359853  3.626536         256
    3      4  0.743784   7.608474  3.411266         256
    4      5  0.631153   7.991230  3.200625         256
    5      6  0.740289   7.367319  3.588372         256
    6      7  0.649674   9.099552  4.290849         256
    7      8  0.571329   8.833177  3.676428         256
    8      9  0.497356   9.076891  3.413999         256
    9     10  0.748292   7.312588  3.523059         256
    10    11  0.622624   9.356255  3.803633         256
    11    12  0.595724   9.949636  3.920303         256
    12    13  0.679390   7.707831  3.763957         256
    13    14  0.563509   8.462486  3.540481         256
    14    15  0.749710   6.904514  3.319380         256
    15    16  0.745994   7.227481  3.498031         256
    16    17  0.630782   8.408267  3.615018         256
    17    18  0.829010   6.256423  2.766467         256
    18    19  0.552532   9.662133  4.790599         256
    19    20  0.592430   8.572208  3.927465         256
    20    21  0.496594  10.328049  3.836538         256
    21    22  0.538694  11.446564  5.281747         256
    22    23  0.808876   5.226996  2.634352         256
    23    24  0.743347   7.090323  3.176280         256
    24    25  0.657169   7.994880  3.433472         256
    
    Prediction preview
       row_index  fold  true_value  predicted_value
    0          2     1         0.0        -2.887695
    1          6     1         0.0        -0.809608
    2         10     1         0.0        -0.381828
    3         11     1         0.0         0.447103
    4         24     1       -15.0         8.835121
    
    GO-LR ordering
    Learned GO-LR order length: 6400
    First 10 ordered features:
    ['pixel_1872', 'pixel_1871', 'pixel_1714', 'pixel_1715', 'pixel_1636', 'pixel_1634', 'pixel_2133', 'pixel_2052', 'pixel_2213', 'pixel_1793']
    
    Final 5x5 CV results
    R2   : 0.6548 ± 0.0995
    RMSE : 8.2695 ± 1.4722
    MAE  : 3.6689 ± 0.6156


# orlaws10P || Multiclass Classification || Single Wrapper


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
    verbose=True,
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

    X shape: (100, 10304), classes: 10
    Using device: cuda
    Learned GO-LR order length: 10304
    First 10 ordered features:
    ['feature_339', 'feature_340', 'feature_10195', 'feature_341', 'feature_9633', 'feature_10087', 'feature_116', 'feature_10085', 'feature_10', 'feature_9860']
    Fold 01 Z_tr shape: (80, 142)
    Fold 01: ACC=1.0000, Macro-F1=1.0000, Macro-OvR-AUC=1.0000
    Fold 02 Z_tr shape: (80, 142)
    Fold 02: ACC=1.0000, Macro-F1=1.0000, Macro-OvR-AUC=1.0000
    Fold 03 Z_tr shape: (80, 142)
    Fold 03: ACC=1.0000, Macro-F1=1.0000, Macro-OvR-AUC=1.0000
    Fold 04 Z_tr shape: (80, 140)
    Fold 04: ACC=1.0000, Macro-F1=1.0000, Macro-OvR-AUC=1.0000
    Fold 05 Z_tr shape: (80, 142)
    Fold 05: ACC=1.0000, Macro-F1=1.0000, Macro-OvR-AUC=1.0000
    Fold 06 Z_tr shape: (80, 142)
    Fold 06: ACC=1.0000, Macro-F1=1.0000, Macro-OvR-AUC=1.0000
    Fold 07 Z_tr shape: (80, 142)
    Fold 07: ACC=1.0000, Macro-F1=1.0000, Macro-OvR-AUC=1.0000
    Fold 08 Z_tr shape: (80, 142)
    Fold 08: ACC=1.0000, Macro-F1=1.0000, Macro-OvR-AUC=1.0000
    Fold 09 Z_tr shape: (80, 140)
    Fold 09: ACC=1.0000, Macro-F1=1.0000, Macro-OvR-AUC=1.0000
    Fold 10 Z_tr shape: (80, 142)
    Fold 10: ACC=1.0000, Macro-F1=1.0000, Macro-OvR-AUC=1.0000
    Fold 11 Z_tr shape: (80, 142)
    Fold 11: ACC=1.0000, Macro-F1=1.0000, Macro-OvR-AUC=1.0000
    Fold 12 Z_tr shape: (80, 142)
    Fold 12: ACC=1.0000, Macro-F1=1.0000, Macro-OvR-AUC=1.0000
    Fold 13 Z_tr shape: (80, 142)
    Fold 13: ACC=1.0000, Macro-F1=1.0000, Macro-OvR-AUC=1.0000
    Fold 14 Z_tr shape: (80, 140)
    Fold 14: ACC=1.0000, Macro-F1=1.0000, Macro-OvR-AUC=1.0000
    Fold 15 Z_tr shape: (80, 142)
    Fold 15: ACC=1.0000, Macro-F1=1.0000, Macro-OvR-AUC=1.0000
    Fold 16 Z_tr shape: (80, 142)
    Fold 16: ACC=1.0000, Macro-F1=1.0000, Macro-OvR-AUC=1.0000
    Fold 17 Z_tr shape: (80, 142)
    Fold 17: ACC=1.0000, Macro-F1=1.0000, Macro-OvR-AUC=1.0000
    Fold 18 Z_tr shape: (80, 140)
    Fold 18: ACC=1.0000, Macro-F1=1.0000, Macro-OvR-AUC=1.0000
    Fold 19 Z_tr shape: (80, 142)
    Fold 19: ACC=1.0000, Macro-F1=1.0000, Macro-OvR-AUC=1.0000
    Fold 20 Z_tr shape: (80, 142)
    Fold 20: ACC=1.0000, Macro-F1=1.0000, Macro-OvR-AUC=1.0000
    Fold 21 Z_tr shape: (80, 142)
    Fold 21: ACC=1.0000, Macro-F1=1.0000, Macro-OvR-AUC=1.0000
    Fold 22 Z_tr shape: (80, 142)
    Fold 22: ACC=1.0000, Macro-F1=1.0000, Macro-OvR-AUC=1.0000
    Fold 23 Z_tr shape: (80, 140)
    Fold 23: ACC=1.0000, Macro-F1=1.0000, Macro-OvR-AUC=1.0000
    Fold 24 Z_tr shape: (80, 142)
    Fold 24: ACC=1.0000, Macro-F1=1.0000, Macro-OvR-AUC=1.0000
    Fold 25 Z_tr shape: (80, 142)
    Fold 25: ACC=1.0000, Macro-F1=1.0000, Macro-OvR-AUC=1.0000
    
    Final CV results
    GOTabPFN completed successfully.
    
    Task type: multiclass
    Evaluation: 5x5 repeated stratified CV
    Samples: 100
    Original numeric features: 10304
    GO-LR metric: cosine
    GO-LR KMeans mode: gpu
    GO-LR feature subsample: 3000
    NSC segmentation: uniform
    NSC M rule: default
    NSC bypass_if_m_leq: 0
    Device: cuda
    Elapsed time: 47.59 seconds
    
    Metric summary:
    accuracy: 1.0000 ± 0.0000
    macro_f1: 1.0000 ± 0.0000
    macro_ovr_auc: 1.0000 ± 0.0000
    nsc_tokens: 141.6000 ± 0.8165
    GOTabPFN completed successfully.
    
    Task type: multiclass
    Evaluation: 5x5 repeated stratified CV
    Samples: 100
    Original numeric features: 10304
    GO-LR metric: cosine
    GO-LR KMeans mode: gpu
    GO-LR feature subsample: 3000
    NSC segmentation: uniform
    NSC M rule: default
    NSC bypass_if_m_leq: 0
    Device: cuda
    Elapsed time: 47.59 seconds
    
    Metric summary:
    accuracy: 1.0000 ± 0.0000
    macro_f1: 1.0000 ± 0.0000
    macro_ovr_auc: 1.0000 ± 0.0000
    nsc_tokens: 141.6000 ± 0.8165
    
    Fold-wise metrics
        fold  accuracy  macro_f1  macro_ovr_auc  nsc_tokens
    0      1       1.0       1.0            1.0         142
    1      2       1.0       1.0            1.0         142
    2      3       1.0       1.0            1.0         142
    3      4       1.0       1.0            1.0         140
    4      5       1.0       1.0            1.0         142
    5      6       1.0       1.0            1.0         142
    6      7       1.0       1.0            1.0         142
    7      8       1.0       1.0            1.0         142
    8      9       1.0       1.0            1.0         140
    9     10       1.0       1.0            1.0         142
    10    11       1.0       1.0            1.0         142
    11    12       1.0       1.0            1.0         142
    12    13       1.0       1.0            1.0         142
    13    14       1.0       1.0            1.0         140
    14    15       1.0       1.0            1.0         142
    15    16       1.0       1.0            1.0         142
    16    17       1.0       1.0            1.0         142
    17    18       1.0       1.0            1.0         140
    18    19       1.0       1.0            1.0         142
    19    20       1.0       1.0            1.0         142
    20    21       1.0       1.0            1.0         142
    21    22       1.0       1.0            1.0         142
    22    23       1.0       1.0            1.0         140
    23    24       1.0       1.0            1.0         142
    24    25       1.0       1.0            1.0         142
    
    Prediction preview
       row_index  fold true_label predicted_label
    0          1     1          0               0
    1          3     1          0               0
    2         10     1          1               1
    3         11     1          1               1
    4         22     1          2               2
    
    GO-LR ordering
    Learned GO-LR order length: 10304
    First 10 ordered features:
    ['feature_339', 'feature_340', 'feature_10195', 'feature_341', 'feature_9633', 'feature_10087', 'feature_116', 'feature_10085', 'feature_10', 'feature_9860']
    
    Final 5x5 CV results
    Accuracy      : 1.0000 ± 0.0000
    Macro-F1      : 1.0000 ± 0.0000
    Macro-OvR-AUC : 1.0000 ± 0.0000


# Colon || Binary Classification || Single Wrapper


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
GO_FEAT_SUBSAMPLE = None

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
    verbose=True,
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

    X shape: (62, 2000), classes: 2
    Using device: cuda
    Learned GO-LR order length: 2000
    First 10 ordered features:
    ['138', '1244', '1912', '1242', '381', '692', '25', '248', '697', '211']
    Fold 01 Z_tr shape: (49, 64)
    Fold 01: ACC=1.0000, F1=1.0000, AUC=1.0000
    Fold 02 Z_tr shape: (49, 64)
    Fold 02: ACC=0.8462, F1=0.8375, AUC=0.9000
    Fold 03 Z_tr shape: (50, 64)
    Fold 03: ACC=0.9167, F1=0.9111, AUC=1.0000
    Fold 04 Z_tr shape: (50, 64)
    Fold 04: ACC=0.8333, F1=0.8125, AUC=0.8438
    Fold 05 Z_tr shape: (50, 64)
    Fold 05: ACC=1.0000, F1=1.0000, AUC=1.0000
    Fold 06 Z_tr shape: (49, 64)
    Fold 06: ACC=1.0000, F1=1.0000, AUC=1.0000
    Fold 07 Z_tr shape: (49, 64)
    Fold 07: ACC=0.6154, F1=0.5752, AUC=0.6750
    Fold 08 Z_tr shape: (50, 64)
    Fold 08: ACC=1.0000, F1=1.0000, AUC=1.0000
    Fold 09 Z_tr shape: (50, 64)
    Fold 09: ACC=0.8333, F1=0.8125, AUC=0.9375
    Fold 10 Z_tr shape: (50, 64)
    Fold 10: ACC=0.9167, F1=0.8992, AUC=1.0000
    Fold 11 Z_tr shape: (49, 64)
    Fold 11: ACC=0.8462, F1=0.8375, AUC=0.9000
    Fold 12 Z_tr shape: (49, 64)
    Fold 12: ACC=0.8462, F1=0.8375, AUC=0.9250
    Fold 13 Z_tr shape: (50, 64)
    Fold 13: ACC=0.9167, F1=0.9111, AUC=0.9688
    Fold 14 Z_tr shape: (50, 64)
    Fold 14: ACC=0.8333, F1=0.8286, AUC=0.8125
    Fold 15 Z_tr shape: (50, 64)
    Fold 15: ACC=0.9167, F1=0.8992, AUC=0.8438
    Fold 16 Z_tr shape: (49, 64)
    Fold 16: ACC=0.8462, F1=0.8375, AUC=0.9000
    Fold 17 Z_tr shape: (49, 64)
    Fold 17: ACC=1.0000, F1=1.0000, AUC=1.0000
    Fold 18 Z_tr shape: (50, 64)
    Fold 18: ACC=1.0000, F1=1.0000, AUC=1.0000
    Fold 19 Z_tr shape: (50, 64)
    Fold 19: ACC=0.9167, F1=0.9111, AUC=1.0000
    Fold 20 Z_tr shape: (50, 64)
    Fold 20: ACC=0.8333, F1=0.8125, AUC=0.7500
    Fold 21 Z_tr shape: (49, 64)
    Fold 21: ACC=0.7692, F1=0.7451, AUC=0.8750
    Fold 22 Z_tr shape: (49, 64)
    Fold 22: ACC=0.6923, F1=0.6750, AUC=0.6750
    Fold 23 Z_tr shape: (50, 64)
    Fold 23: ACC=0.8333, F1=0.8286, AUC=0.8750
    Fold 24 Z_tr shape: (50, 64)
    Fold 24: ACC=0.8333, F1=0.8125, AUC=0.9688
    Fold 25 Z_tr shape: (50, 64)
    Fold 25: ACC=1.0000, F1=1.0000, AUC=1.0000
    
    Final CV results
    GOTabPFN completed successfully.
    
    Task type: binary
    Evaluation: 5x5 repeated stratified CV
    Samples: 62
    Original numeric features: 2000
    GO-LR metric: euclidean
    GO-LR KMeans mode: gpu
    GO-LR feature subsample: full
    NSC segmentation: equal_mass
    NSC M rule: idf
    NSC bypass_if_m_leq: 0
    Device: cuda
    Elapsed time: 48.30 seconds
    
    Metric summary:
    accuracy: 0.8818 ± 0.1005
    macro_f1: 0.8714 ± 0.1093
    auc: 0.9140 ± 0.1012
    nsc_tokens: 64.0000 ± 0.0000
    GOTabPFN completed successfully.
    
    Task type: binary
    Evaluation: 5x5 repeated stratified CV
    Samples: 62
    Original numeric features: 2000
    GO-LR metric: euclidean
    GO-LR KMeans mode: gpu
    GO-LR feature subsample: full
    NSC segmentation: equal_mass
    NSC M rule: idf
    NSC bypass_if_m_leq: 0
    Device: cuda
    Elapsed time: 48.30 seconds
    
    Metric summary:
    accuracy: 0.8818 ± 0.1005
    macro_f1: 0.8714 ± 0.1093
    auc: 0.9140 ± 0.1012
    nsc_tokens: 64.0000 ± 0.0000
    
    Fold-wise metrics
        fold  accuracy  macro_f1      auc  nsc_tokens
    0      1  1.000000  1.000000  1.00000          64
    1      2  0.846154  0.837500  0.90000          64
    2      3  0.916667  0.911111  1.00000          64
    3      4  0.833333  0.812500  0.84375          64
    4      5  1.000000  1.000000  1.00000          64
    5      6  1.000000  1.000000  1.00000          64
    6      7  0.615385  0.575163  0.67500          64
    7      8  1.000000  1.000000  1.00000          64
    8      9  0.833333  0.812500  0.93750          64
    9     10  0.916667  0.899160  1.00000          64
    10    11  0.846154  0.837500  0.90000          64
    11    12  0.846154  0.837500  0.92500          64
    12    13  0.916667  0.911111  0.96875          64
    13    14  0.833333  0.828571  0.81250          64
    14    15  0.916667  0.899160  0.84375          64
    15    16  0.846154  0.837500  0.90000          64
    16    17  1.000000  1.000000  1.00000          64
    17    18  1.000000  1.000000  1.00000          64
    18    19  0.916667  0.911111  1.00000          64
    19    20  0.833333  0.812500  0.75000          64
    20    21  0.769231  0.745098  0.87500          64
    21    22  0.692308  0.675000  0.67500          64
    22    23  0.833333  0.828571  0.87500          64
    23    24  0.833333  0.812500  0.96875          64
    24    25  1.000000  1.000000  1.00000          64
    
    Prediction preview
       row_index  fold true_label predicted_label
    0          8     1          0               0
    1         13     1          1               1
    2         17     1          1               1
    3         18     1          0               0
    4         30     1          0               0
    
    GO-LR ordering
    Learned GO-LR order length: 2000
    First 10 ordered features:
    ['138', '1244', '1912', '1242', '381', '692', '25', '248', '697', '211']
    
    Final 5x5 CV results
    Accuracy : 0.8818 ± 0.1005
    Macro-F1 : 0.8714 ± 0.1093
    AUC      : 0.9140 ± 0.1012


# Colon || Binary Classification || Manual Approach


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

    X shape: (62, 2000)
    Classes: ['0', '1']
    Using device: cuda
    Learned GO-LR order length: 2000
    Fold 01: ACC=1.0000, F1=1.0000, AUC=1.0000
    Fold 02: ACC=0.8462, F1=0.8375, AUC=0.9000
    Fold 03: ACC=0.9167, F1=0.9111, AUC=1.0000
    Fold 04: ACC=0.8333, F1=0.8125, AUC=0.8438
    Fold 05: ACC=1.0000, F1=1.0000, AUC=1.0000
    Fold 06: ACC=1.0000, F1=1.0000, AUC=1.0000
    Fold 07: ACC=0.6154, F1=0.5752, AUC=0.6750
    Fold 08: ACC=1.0000, F1=1.0000, AUC=1.0000
    Fold 09: ACC=0.8333, F1=0.8125, AUC=0.9375
    Fold 10: ACC=0.9167, F1=0.8992, AUC=1.0000
    Fold 11: ACC=0.8462, F1=0.8375, AUC=0.9000
    Fold 12: ACC=0.8462, F1=0.8375, AUC=0.9250
    Fold 13: ACC=0.9167, F1=0.9111, AUC=0.9688
    Fold 14: ACC=0.8333, F1=0.8286, AUC=0.8125
    Fold 15: ACC=0.9167, F1=0.8992, AUC=0.8438
    Fold 16: ACC=0.8462, F1=0.8375, AUC=0.9000
    Fold 17: ACC=1.0000, F1=1.0000, AUC=1.0000
    Fold 18: ACC=1.0000, F1=1.0000, AUC=1.0000
    Fold 19: ACC=0.9167, F1=0.9111, AUC=1.0000
    Fold 20: ACC=0.8333, F1=0.8125, AUC=0.7500
    Fold 21: ACC=0.7692, F1=0.7451, AUC=0.8750
    Fold 22: ACC=0.6923, F1=0.6750, AUC=0.6750
    Fold 23: ACC=0.8333, F1=0.8286, AUC=0.8750
    Fold 24: ACC=0.8333, F1=0.8125, AUC=0.9688
    Fold 25: ACC=1.0000, F1=1.0000, AUC=1.0000
    
    Final 5x5 CV results
    Accuracy : 0.8818 ± 0.1005
    Macro-F1 : 0.8714 ± 0.1093
    AUC      : 0.9140 ± 0.1012



```python
import gotabpfn
import tabpfn
import inspect

print("gotabpfn version:", gotabpfn.__version__)
print("gotabpfn path:", gotabpfn.__file__)
print("run_gotabpfn_csv file:", inspect.getsourcefile(gotabpfn.run_gotabpfn_csv))

print("tabpfn version:", getattr(tabpfn, "__version__", "NO __version__"))
print("tabpfn path:", tabpfn.__file__)
```

    gotabpfn version: 0.1.11
    gotabpfn path: /usr/local/lib/python3.12/dist-packages/gotabpfn/__init__.py
    run_gotabpfn_csv file: /usr/local/lib/python3.12/dist-packages/gotabpfn/gotabpfn.py
    tabpfn version: 6.3.1
    tabpfn path: /usr/local/lib/python3.12/dist-packages/tabpfn/__init__.py



```python

```
