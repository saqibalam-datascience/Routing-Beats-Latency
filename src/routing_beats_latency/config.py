"""
Central configuration for Routing Beats Latency.

Contains hyperparameters, hardware latency constants, corruption suites,
and portable directory path resolution for cache, checkpoints, and outputs.
"""
import os
import random
import numpy as np
import torch

# -----------------------------------------------------------------------------
# 1. Directory Paths (Portable Resolution)
# -----------------------------------------------------------------------------
def get_project_root():
    """Dynamically determine the project root whether running from repo root,
    notebooks/ directory, or scripts/ directory."""
    cwd = os.path.abspath(os.getcwd())
    if os.path.exists(os.path.join(cwd, "outputs_imagenette_math")):
        return cwd
    parent = os.path.dirname(cwd)
    if os.path.exists(os.path.join(parent, "outputs_imagenette_math")):
        return parent
    # Default to current directory or parent
    return cwd

PROJECT_ROOT = get_project_root()

# Data directory with environment variable override and local fallback
DATA_DIR = os.environ.get("IMAGENETTE_DIR", os.path.join(PROJECT_ROOT, "data", "imagenette"))
if not os.path.exists(DATA_DIR) and os.path.exists(r"d:\Saqib\RoutingBeatsLatency\routing_beats_latency_revision2\data\imagenette"):
    DATA_DIR = r"d:\Saqib\RoutingBeatsLatency\routing_beats_latency_revision2\data\imagenette"

TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

# Artifacts directories
OUTPUT_DIR = os.path.normpath(os.path.join(PROJECT_ROOT, "outputs_imagenette_math"))
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
GATER_DIR = os.path.join(OUTPUT_DIR, "gaters")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
RESULTS_XLSX = os.path.join(OUTPUT_DIR, "results_imagenette_math.xlsx")

for d in [CHECKPOINT_DIR, CACHE_DIR, GATER_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# -----------------------------------------------------------------------------
# 2. Hardware Latency Constants (Single-Sample B=1 on NVIDIA RTX 4000 Ada)
# -----------------------------------------------------------------------------
# Measured via hardware CUDA Events with explicit synchronization
LATENCY_SQUEEZENET = 1.19        # ms (SqueezeNet 1.0 CNN)
LATENCY_DEIT = 4.98              # ms (DeiT-S-16 Vision Transformer)
LATENCY_GATER = 0.0023           # ms (2.3 microseconds micro-probe gater)
LATENCY_MAX = LATENCY_DEIT       # Normalizing constant for latency loss

SLA_BUDGET_MS = 5.0              # Strict real-time SLA budget (ms)

# -----------------------------------------------------------------------------
# 3. Model & Dataset Dimensions
# -----------------------------------------------------------------------------
NUM_CLASSES = 10
IMG_SIZE = 384                   # Input resolution seen by expert classifiers
GATER_IMG_SIZE = 64              # Input resolution seen by micro-probe gater (downsampled)
EXPERT_CACHE_SIZE = 224          # Preprocessing crop size for training pool

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# -----------------------------------------------------------------------------
# 4. Expert Definitions
# -----------------------------------------------------------------------------
EXPERT_DEFINITIONS = {
    "SqueezeNet": {"source": "torchvision", "name": "squeezenet1_0"},
    "DeiT-S-16":  {"source": "timm", "name": "deit_small_patch16_224"},
}
EXPERT_NAMES = list(EXPERT_DEFINITIONS.keys())

EXPERT_BATCH_SIZES = {"SqueezeNet": 128, "DeiT-S-16": 64}
EXPERT_GRAD_ACCUM  = {"SqueezeNet": 4, "DeiT-S-16": 8}

# -----------------------------------------------------------------------------
# 5. Optimization & Sweep Hyperparameters
# -----------------------------------------------------------------------------
LAMBDAS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.92, 0.95, 0.97, 1.00]
SEEDS = [0, 1, 2]

GUMBEL_TAU_START = 5.0
GUMBEL_TAU_END = 0.1
EPOCHS_GATER = 100
LR_GATER = 1e-3

# -----------------------------------------------------------------------------
# 6. Corruption Suites (ImageNet-C Benchmark)
# -----------------------------------------------------------------------------
ALL_CORRUPTIONS = ["blur", "noise", "jpeg", "contrast", "speckle"]
SEVERITY_LEVELS = ["clean", "moderate", "severe"]
UNSEEN_CORRUPTIONS = ["pixelate", "impulse_noise"]

# -----------------------------------------------------------------------------
# 7. Device Setup & Seeding
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
