# ============================================================
# gotabpfn/__init__.py
#
# Unified package exports for:
#   - GOTabPFN core model/utilities
#   - GO-LR feature ordering
#   - Four NSC compression variants
#   - Dataset diagnostics wrappers
#
# Expected package structure:
#
# gotabpfn/
# ├── __init__.py
# ├── gotabpfn.py
# ├── GO-LR.py
# ├── NSC.py
# ├── NSC-P.py
# ├── NSC-SP.py
# ├── NSC-pSP.py
# └── gotabpfn_dataset_diagnostics.py
#
# Files with hyphens cannot be imported normally, so we load
# them dynamically and expose clean package-level names.
# ============================================================

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys


__version__ = "0.1.9"


# ============================================================
# Internal dynamic loader
# ============================================================

_PKG_DIR = Path(__file__).resolve().parent


def _load_local_module(module_name: str, filename: str):
    """
    Dynamically load a local .py file as a package submodule.

    Needed for files such as:
      - GO-LR.py
      - NSC-P.py
      - NSC-SP.py
      - NSC-pSP.py
    because Python module names cannot contain hyphens.
    """
    path = _PKG_DIR / filename

    if not path.exists():
        raise ImportError(f"Could not find {filename} at {path}")

    full_name = f"{__name__}.{module_name}"

    spec = importlib.util.spec_from_file_location(full_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {filename}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)

    return module


# ============================================================
# Core GOTabPFN
# ============================================================

_core = _load_local_module("_core", "gotabpfn.py")

GraphFeatureOrdering = getattr(_core, "GraphFeatureOrdering", None)
pidf_segpca = getattr(_core, "pidf_segpca", None)
TabPFN25Head = getattr(_core, "TabPFN25Head", None)
TabPFN25Config = getattr(_core, "TabPFN25Config", None)
run_gotabpfn_csv = getattr(_core, "run_gotabpfn_csv", None)


# ============================================================
# GO-LR ordering utility
# ============================================================

_golr = _load_local_module("_go_lr", "GO-LR.py")

GraphFeatureOrdering_GO_LR = getattr(_golr, "GraphFeatureOrdering", None)
GOLRFeatureOrdering = getattr(
    _golr,
    "GOLRFeatureOrdering",
    GraphFeatureOrdering_GO_LR,
)
run_golr_csv = getattr(_golr, "run_golr_csv", None)


# If core gotabpfn.py does not expose GraphFeatureOrdering,
# fall back to the standalone GO-LR implementation.
if GraphFeatureOrdering is None:
    GraphFeatureOrdering = GraphFeatureOrdering_GO_LR


# ============================================================
# NSC variants
# ============================================================

# ----------------------------
# Original NSC
# ----------------------------
_nsc = _load_local_module("_nsc", "NSC.py")

NSC = getattr(_nsc, "NSC", None)
NSC_Base = getattr(_nsc, "NSC_Base", NSC)
run_nsc_csv = getattr(_nsc, "run_nsc_csv", None)
estimate_intrinsic_dim_effective_rank = getattr(
    _nsc,
    "estimate_intrinsic_dim_effective_rank",
    None,
)


# ----------------------------
# NSC-P
# ----------------------------
_nsc_p = _load_local_module("_nsc_p", "NSC-P.py")

NSC_PCA = getattr(_nsc_p, "NSC_PCA", None)
NSC_P = getattr(_nsc_p, "NSC_P", NSC_PCA)
run_nsc_p_csv = getattr(_nsc_p, "run_nsc_p_csv", None)


# ----------------------------
# NSC-SP
# ----------------------------
_nsc_sp = _load_local_module("_nsc_sp", "NSC-SP.py")

NSC_SEG_PCA = getattr(_nsc_sp, "NSC_SEG_PCA", None)
NSC_SP = getattr(_nsc_sp, "NSC_SP", NSC_SEG_PCA)
run_nsc_sp_csv = getattr(_nsc_sp, "run_nsc_sp_csv", None)


# ----------------------------
# NSC-pSP
# ----------------------------
_nsc_psp = _load_local_module("_nsc_psp", "NSC-pSP.py")

pidf_segpca_standalone = getattr(_nsc_psp, "pidf_segpca", None)
NSC_pSP = getattr(_nsc_psp, "NSC_pSP", pidf_segpca_standalone)
run_nsc_psp_csv = getattr(_nsc_psp, "run_nsc_psp_csv", None)


# If core gotabpfn.py does not expose pidf_segpca,
# fall back to standalone NSC-pSP.
if pidf_segpca is None:
    pidf_segpca = pidf_segpca_standalone


# ============================================================
# Dataset diagnostics
# ============================================================
# Lazy loading avoids circular import issues because
# gotabpfn_dataset_diagnostics.py imports:
#   from gotabpfn import GraphFeatureOrdering
# ============================================================

def load_dataset_diagnostics_module():
    """
    Lazily load gotabpfn_dataset_diagnostics.py.

    Example:
        from gotabpfn import load_dataset_diagnostics_module

        diag = load_dataset_diagnostics_module()
        df = diag.analyze_csv_ordering_metrics(...)
    """
    return _load_local_module(
        "_dataset_diagnostics",
        "gotabpfn_dataset_diagnostics.py",
    )


def analyze_csv_ordering_metrics(*args, **kwargs):
    """
    Lazy wrapper for diagnostics.analyze_csv_ordering_metrics.
    """
    diag = load_dataset_diagnostics_module()
    return diag.analyze_csv_ordering_metrics(*args, **kwargs)


def analyze_many_csvs(*args, **kwargs):
    """
    Lazy wrapper for diagnostics.analyze_many_csvs.
    """
    diag = load_dataset_diagnostics_module()
    return diag.analyze_many_csvs(*args, **kwargs)


# ============================================================
# Public API
# ============================================================

__all__ = [
    "__version__",

    # Core GOTabPFN
    "GraphFeatureOrdering",
    "pidf_segpca",
    "TabPFN25Head",
    "TabPFN25Config",
    "run_gotabpfn_csv",

    # GO-LR
    "GraphFeatureOrdering_GO_LR",
    "GOLRFeatureOrdering",
    "run_golr_csv",

    # NSC original
    "NSC",
    "NSC_Base",
    "run_nsc_csv",
    "estimate_intrinsic_dim_effective_rank",

    # NSC-P
    "NSC_PCA",
    "NSC_P",
    "run_nsc_p_csv",

    # NSC-SP
    "NSC_SEG_PCA",
    "NSC_SP",
    "run_nsc_sp_csv",

    # NSC-pSP
    "NSC_pSP",
    "pidf_segpca_standalone",
    "run_nsc_psp_csv",

    # Diagnostics
    "load_dataset_diagnostics_module",
    "analyze_csv_ordering_metrics",
    "analyze_many_csvs",
]


# Remove unavailable names from __all__.
__all__ = [
    name for name in __all__
    if name in globals() and globals()[name] is not None
]
