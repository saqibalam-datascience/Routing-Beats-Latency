# Routing Beats Latency: Input-Adaptive Heterogeneous Inference Under Strict Single-Sample Latency Constraints

[![Paper](https://img.shields.io/badge/TMLR-Accepted%20(Paper%209611)-b31b1b.svg)](https://openreview.net/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hardware](https://img.shields.io/badge/Benchmarked-NVIDIA%20RTX%204000%20Ada-76b900.svg)](https://www.nvidia.com/)

Official PyTorch implementation of **"Routing Beats Latency: Input-Adaptive Heterogeneous Inference Under Strict Single-Sample Latency Constraints"**, accepted for publication in the *Transactions on Machine Learning Research (TMLR)*.

---

## 📌 Executive Summary

Deploying deep vision models under strict real-time Service Level Agreements (SLAs) requires balancing high classification accuracy against severe inference latency limits. Conventional adaptive inference methods rely on:
1. **Sequential Confidence Cascades:** Suffer from an **Additive Latency Catastrophe** ($L_{\text{total}} = L_{\text{cheap}} + L_{\text{heavy}} = 1.19\,\text{ms} + 4.98\,\text{ms} = 6.17\,\text{ms}$) under distribution shift when confidence collapses, exceeding the latency of running the heavy model alone.
2. **Homogeneous Early-Exit ViTs:** Physically bound by an **irreducible latency floor** ($1.53$\,ms) due to mandatory tokenization and early attention blocks.

### Our Solution: First-Touch Heterogeneous Routing
We propose a **First-Touch Heterogeneous Mixture-of-Experts (MoE)** framework:
- **Instantaneous Micro-Probe Gater ($2.3\,\mu$s latency, 624 parameters):** Operates upfront on $10$ domain-agnostic spatial distortion statistics ($\sigma^2_{\text{global}}$, discrete Laplacian edge variance, anisotropic Total Variation, and 8×8 Blockiness).
- **Heterogeneous Architectural Pool:** Dispatches queries directly between a fast CNN (**SqueezeNet 1.0**, $1.19$\,ms) and a resilient Vision Transformer (**DeiT-S-16**, $4.98$\,ms).
- **100% Hard SLA Compliance:** Strictly bounds tail latency to $L_{\max} = 4.982$\,ms, achieving **100.0% SLA budget compliance** under a $5.0$\,ms threshold (where cascades suffer a $47.9\%$ violation rate).
- **Superior Zero-Shot Out-of-Distribution Robustness:** Attains **$+5.23\%$ higher accuracy** on completely unseen corruptions ($55.28\%$ vs. $50.05\%$) over early-exit transformers by avoiding internal entropy confusion.
- **The Two-Model Pareto Dial:** Train two anchor models once, and dynamically tune the continuous Pareto frontier simply by swapping a $<5$\,KB scalar-regularized router checkpoint.

---

## 🏆 Master Empirical Comparison (Table 1 in Manuscript)

Evaluated on the Imagenette benchmark under standard seen ImageNet-C corruptions (`test_mixed.pt`) and completely held-out zero-shot corruptions (`test_unseen_mixed.pt`). All latencies are measured via hardware CUDA Events ($B=1$, single-sample) on an NVIDIA RTX 4000 Ada Generation GPU.

| Inference Paradigm | Configuration | Exiting Distribution | Mean Latency (ms) | Latency Saved (%) | Seen Acc. (%) | Unseen Acc. (%) | VRAM (MB) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Static Anchors** | SqueezeNet 1.0 (CNN) | 100% SqueezeNet | **1.19** | 76.2% | 64.36 | 35.59 | **5.0** |
| | DeiT-S-16 (Transformer) | 100% DeiT | 4.98 | 0.0% (Anchor) | 89.76 | 65.21 | 87.2 |
| **Sequential Cascades** | Confidence Cascade ($t=0.50$) | 87.2% Sq / 12.8% DeiT | 1.83 | 63.3% | 71.01 | 37.80 | 92.2 |
| | Confidence Cascade ($t=0.85$) | 52.1% Sq / 47.9% DeiT | 3.57 | 28.3% | 84.11 | 53.43 | 92.2 |
| **Homogeneous Early-Exit ViT** | Early-Exit DeiT-S-16 ($t=0.40$) | 82.6% / 16.5% / 0.8% | 1.86 | 62.7% | **82.47** | 35.43 | 87.4 |
| | Early-Exit DeiT-S-16 ($t=0.60$) | 55.2% / 39.4% / 5.4% | 2.40 | 51.8% | **91.29** | 44.61 | 87.4 |
| | Early-Exit DeiT-S-16 ($t=0.80$) | 32.7% / 54.8% / 12.5% | 2.89 | 42.0% | **95.46** | 56.49 | 87.4 |
| | Early-Exit DeiT-S-16 ($t=0.90$) | 18.8% / 63.1% / 18.1% | 3.22 | 35.3% | **96.41** | 62.09 | 87.4 |
| **Proposed First-Touch Gumbel Router** | First-Touch Gumbel ($\lambda=0.70$) | 84.8% Sq / 15.2% DeiT | **1.77** | **64.5%** | 74.15 | **55.28** | 92.2 |
| | First-Touch Gumbel ($\lambda=0.80$) | 66.4% Sq / 33.6% DeiT | 2.47 | 50.4% | 81.52 | 55.56 | 92.2 |
| | First-Touch Gumbel ($\lambda=0.90$) | 55.3% Sq / 44.7% DeiT | 2.89 | 42.0% | 85.43 | **61.79** | 92.2 |
| | First-Touch Gumbel ($\lambda=0.92$) | 51.0% Sq / 49.0% DeiT | 3.06 | 38.6% | 85.83 | **62.28** | 92.2 |
| **Theoretical Limit** | First-Touch Oracle | Dynamic Optimal | 3.63 | 27.1% | 97.41 | 66.84 | 92.2 |

---

## 📁 Repository Structure

```
RoutingBeatsLatency/
├── src/routing_beats_latency/     # Core modular Python package
│   ├── __init__.py                # Package exports
│   ├── config.py                  # Paths, constants, SLA limits, hardware configs
│   ├── corruptions.py             # Math corruptions (blur, noise, jpeg, contrast, speckle, pixelate, impulse)
│   ├── features.py                # 10-D micro-probe spatial distortion feature extractor
│   ├── models.py                  # SqueezeNet 1.0, DeiT-S-16, Gater, EarlyExitDeiTS16 architectures
│   ├── gater.py                   # Straight-Through Gumbel-Softmax router & training
│   ├── baselines.py               # 7 Reviewer baselines (cascades, random, oracle) & early-exit ViT
│   ├── benchmark.py               # Hardware CUDA Events single-sample latency profiling
│   ├── evaluation.py              # Seen/unseen evaluation & SLA compliance metrics
│   └── visualization.py           # Publication-quality plotting tools
│
├── notebooks/                     # Interactive Jupyter Notebooks with all outputs preserved & visible
│   ├── routing_beats_latency_without_gumbel_v8.ipynb # Main experimental notebook (all plots & tables visible)
│   └── Early_exit_vit_baseline.ipynb                 # Official Early-Exit ViT baseline notebook
│
├── scripts/                       # Ready-to-run CLI scripts
│   ├── benchmark_latency.py       # Benchmark single-sample GPU latency
│   ├── evaluate.py                # Evaluation on seen/unseen test caches
│   ├── train_gater.py             # Train gaters for arbitrary lambda/seed or full matrix
│   └── download_imagenette.py     # Download Imagenette if retraining backbones from scratch
│
├── requirements.txt               # Python dependencies
├── setup.py                       # Installable package setup (pip install -e .)
├── pyproject.toml                 # Modern packaging configuration
└── LICENSE                        # MIT License
```

---

## 🚀 Quickstart & Installation

### 1. Environment Setup
```bash
git clone https://github.com/your-username/RoutingBeatsLatency.git
cd RoutingBeatsLatency

# Create and activate conda environment
conda create -n routingbeatslatency python=3.10 -y
conda activate routingbeatslatency

# Install dependencies and editable package
pip install -r requirements.txt
pip install -e .
```

### 2. Interactive Notebooks with Visible Results
The complete empirical trajectory, including training traces, ablation sweeps, Pareto frontiers, error-bar charts, and baseline comparisons, is pre-rendered and visible directly in GitHub's web interface:
- [`notebooks/routing_beats_latency_without_gumbel_v8.ipynb`](notebooks/routing_beats_latency_without_gumbel_v8.ipynb): Main manuscript experiments, Gumbel Pareto sweep across 9 $\lambda$ values $\times$ 3 seeds, unseen corruptions, and 7 reviewer baselines.
- [`notebooks/Early_exit_vit_baseline.ipynb`](notebooks/Early_exit_vit_baseline.ipynb): Multi-exit ViT baseline (BranchyNet / LGViT standard), threshold sweeping, and head-to-head comparison.

To inspect or run locally:
```bash
jupyter lab notebooks/
```

### 3. Profile Hardware Latency (CUDA Events)
Profile single-sample ($B=1$) GPU latency on your hardware:
```bash
python scripts/benchmark_latency.py
```

---

## 🛠️ Reproduction from Scratch

To re-run the entire pipeline from raw image data:

### Step 1: Download Benchmark Dataset
```bash
python scripts/download_imagenette.py
```

### Step 2: Train Experts & Build Cache
Run sections 3, 4, and 5 of `notebooks/routing_beats_latency_without_gumbel_v8.ipynb` to fine-tune SqueezeNet and DeiT-S-16 and cache expert outputs.

### Step 3: Train Gumbel Routers
```bash
# Train an individual gater (e.g., lambda=0.70, seed=0)
python scripts/train_gater.py --lambda 0.70 --seed 0

# Or train all 27 gaters across the full lambda and seed grid
python scripts/train_gater.py --all
```

### Step 4: Evaluate
```bash
python scripts/evaluate.py --split test
python scripts/evaluate.py --split unseen
```

---

## 📖 Citation

```bibtex
@article{routingbeatslatency2026,
  title={Routing Beats Latency: Input-Adaptive Heterogeneous Inference Under Strict Single-Sample Latency Constraints},
  author={Anonymous Authors},
  journal={Transactions on Machine Learning Research (TMLR)},
  year={2026},
  note={Submission ID: Paper 9611}
}
```

---

## 📄 License
This repository is released under the [MIT License](LICENSE).
