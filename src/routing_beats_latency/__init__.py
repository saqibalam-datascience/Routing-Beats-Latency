"""
Routing Beats Latency: Input-Adaptive Heterogeneous Inference Under Strict Single-Sample Latency Constraints
Official PyTorch Implementation.
"""
from .config import (
    NUM_CLASSES, IMG_SIZE, GATER_IMG_SIZE,
    LATENCY_SQUEEZENET, LATENCY_DEIT, LATENCY_GATER,
    SLA_BUDGET_MS, LAMBDAS, SEEDS, DEVICE,
    CACHE_DIR, CHECKPOINT_DIR, GATER_DIR, PLOTS_DIR, RESULTS_XLSX
)
from .features import extract_spatial_distortion_features
from .corruptions import apply_corruption, ALL_CORRUPTIONS, UNSEEN_CORRUPTIONS
from .models import Gater, EarlyExitDeiTS16, build_expert, load_expert
from .gater import train_gumbel_gater, load_gumbel_gater, train_all_gaters
from .benchmark import benchmark_latency
from .baselines import evaluate_all_baselines_from_cache
from .evaluation import evaluate_gumbel_router, evaluate_all_gumbel_models

__version__ = "1.0.0"
__all__ = [
    "Gater",
    "EarlyExitDeiTS16",
    "build_expert",
    "load_expert",
    "train_gumbel_gater",
    "load_gumbel_gater",
    "train_all_gaters",
    "benchmark_latency",
    "evaluate_gumbel_router",
    "evaluate_all_gumbel_models",
    "evaluate_all_baselines_from_cache",
    "extract_spatial_distortion_features",
    "apply_corruption",
]
