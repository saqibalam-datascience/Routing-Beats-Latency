"""
Master Evaluation Suite for First-Touch Routing and Dynamic Inference.

Evaluates:
- Gumbel Router Pareto Sweeps (9 lambdas x 3 seeds)
- In-Distribution Seen Mathematical Corruptions (test_mixed.pt)
- Zero-Shot Out-of-Distribution Generalization (test_unseen_mixed.pt)
- Service Level Agreement (SLA) Hard Budget Compliance (5.0 ms threshold)
"""
import os
import glob
import re
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

from .config import (
    CACHE_DIR, GATER_DIR, DEVICE,
    LAMBDAS, SEEDS, SLA_BUDGET_MS,
    LATENCY_SQUEEZENET, LATENCY_DEIT, LATENCY_GATER
)
from .models import Gater
from .gater import get_gater_mean_std

@torch.inference_mode()
def evaluate_gumbel_router(gater: Gater, cache_data: dict, device = DEVICE) -> dict:
    """
    Evaluate a single Gumbel Router on a cache dictionary.
    
    Args:
        gater: Loaded Gater model.
        cache_data: Dictionary containing 'logits', 'latencies_ms', 'labels', and 'gater_inputs'.
        device: Device to run evaluation on.
        
    Returns:
        Dictionary with accuracy, latency, routing distribution, and SLA compliance.
    """
    gater.eval()
    logits = cache_data["logits"].to(device)
    lats = cache_data["latencies_ms"].to(device)
    labels = cache_data["labels"].to(device)
    
    gater_mean, gater_std = get_gater_mean_std()
    gx = (cache_data["gater_inputs"].to(device).float() / 255.0 - gater_mean) / gater_std
    n_total = labels.size(0)

    # Route decision: Argmax over output logits
    scores = gater(gx)
    route_to_deit = (scores.argmax(dim=-1) == 1)
    
    # Selected logits & latencies
    selected_logits = torch.where(route_to_deit.unsqueeze(1).unsqueeze(2), logits[:, 1:2], logits[:, 0:1]).squeeze(1)
    selected_lats = torch.where(route_to_deit, lats[:, 1], lats[:, 0]) + LATENCY_GATER
    
    acc = (selected_logits.argmax(dim=-1) == labels).float().mean().item()
    mean_lat = selected_lats.mean().item()
    p50_lat = np.percentile(selected_lats.cpu().numpy(), 50)
    p95_lat = np.percentile(selected_lats.cpu().numpy(), 95)
    p99_lat = np.percentile(selected_lats.cpu().numpy(), 99)
    max_lat = selected_lats.max().item()
    
    n_deit = int(route_to_deit.sum().item())
    pct_deit = 100.0 * n_deit / n_total
    pct_sq = 100.0 - pct_deit
    
    # SLA Compliance: fraction of samples <= SLA_BUDGET_MS
    sla_compliant = (selected_lats <= SLA_BUDGET_MS).float().mean().item() * 100.0

    return {
        "accuracy": acc,
        "latency_mean_ms": mean_lat,
        "latency_p50_ms": p50_lat,
        "latency_p95_ms": p95_lat,
        "latency_p99_ms": p99_lat,
        "latency_max_ms": max_lat,
        "%_to_DeiT": pct_deit,
        "%_to_SqueezeNet": pct_sq,
        "sla_compliance_%": sla_compliant
    }

@torch.inference_mode()
def evaluate_all_gumbel_models(
    cache_path: str = None,
    gater_dir: str = GATER_DIR,
    device = DEVICE
) -> tuple:
    """
    Evaluate every trained Gumbel gater checkpoint in gater_dir.
    
    Returns:
        (df_raw, df_summary): Per-seed metrics and mean aggregated metrics across lambdas.
    """
    if cache_path is None:
        cache_path = os.path.join(CACHE_DIR, "test_mixed.pt")
        
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache file not found at: {cache_path}")

    c = torch.load(cache_path, map_location="cpu", weights_only=False)
    
    records = []
    pattern = re.compile(r"gater_gumbel_lambda([0-9]*\.?[0-9]+)_seed([0-9]+)\.pt")

    for fpath in sorted(glob.glob(os.path.join(gater_dir, "gater_gumbel_lambda*_seed*.pt"))):
        fname = os.path.basename(fpath)
        m = pattern.search(fname)
        if not m:
            continue
        lam = round(float(m.group(1)), 3)
        seed = int(m.group(2))

        model = Gater().to(device)
        model.load_state_dict(torch.load(fpath, map_location=device, weights_only=False))
        model.eval()

        res = evaluate_gumbel_router(model, c, device=device)
        res["Lambda"] = lam
        res["Seed"] = seed
        records.append(res)

    df_raw = pd.DataFrame(records)
    if len(df_raw) == 0:
        return df_raw, pd.DataFrame()

    df_summary = df_raw.groupby("Lambda", as_index=False).agg({
        "accuracy": ["mean", "std"],
        "latency_mean_ms": ["mean", "std"],
        "%_to_DeiT": "mean",
        "%_to_SqueezeNet": "mean",
        "sla_compliance_%": "mean"
    })
    
    # Flatten multi-level columns
    df_summary.columns = [
        "Lambda", "Accuracy_Mean", "Accuracy_Std",
        "Latency_Mean_ms", "Latency_Std_ms",
        "%_to_DeiT", "%_to_SqueezeNet", "SLA_Compliance_%"
    ]

    return df_raw, df_summary
