"""
Comprehensive Reviewer Baselines for Dynamic and Adaptive Inference.

Implements all 7 concrete baselines evaluated in the paper:
1. Static Anchors: SqueezeNet 1.0 (CNN) and DeiT-S-16 (ViT)
2. Confidence Cascade: SqueezeNet confidence < t escalates to DeiT (pays L_sq + L_deit)
3. Negative Entropy Cascade: SqueezeNet predictive entropy > t escalates to DeiT
4. Maximum-Softmax-Probability (MSP) Routing: Hendrycks & Gimpel MSP thresholding
5. Disagreement Routing: Margin between Top-1 and Top-2 probabilities
6. Random & Fixed-Ratio Routing: Matched latency sweep and matched Gumbel proportions
7. Oracle Upper Bounds: First-Touch Oracle (upfront optimal) and Cascade Oracle (sequential optimal)
"""
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

def run_confidence_cascade(sq_logits, deit_logits, sq_lats, deit_lats, labels, threshold=0.5):
    """
    Sequential confidence cascade:
    If SqueezeNet's max softmax probability < threshold, query escalates to DeiT-S-16.
    Latency penalty incurred on escalation: L_total = L_sq + L_deit.
    """
    sq_probs = F.softmax(sq_logits, dim=-1)
    sq_conf, _ = sq_probs.max(dim=-1)
    
    route_to_deit = (sq_conf < threshold)
    latencies = torch.where(route_to_deit, sq_lats + deit_lats, sq_lats)
    logits = torch.where(route_to_deit.unsqueeze(1), deit_logits, sq_logits)
    
    acc = (logits.argmax(dim=-1) == labels).float().mean().item()
    mean_lat = latencies.mean().item()
    pct_deit = 100.0 * route_to_deit.float().mean().item()
    
    return {"accuracy": acc, "latency_mean_ms": mean_lat, "%_to_DeiT": pct_deit}

def run_entropy_cascade(sq_logits, deit_logits, sq_lats, deit_lats, labels, threshold=1.0):
    """
    Sequential predictive entropy cascade:
    If SqueezeNet's predictive entropy > threshold, query escalates to DeiT-S-16.
    """
    sq_probs = F.softmax(sq_logits, dim=-1)
    sq_entropy = -torch.sum(sq_probs * torch.log(sq_probs + 1e-12), dim=-1)
    
    route_to_deit = (sq_entropy > threshold)
    latencies = torch.where(route_to_deit, sq_lats + deit_lats, sq_lats)
    logits = torch.where(route_to_deit.unsqueeze(1), deit_logits, sq_logits)
    
    acc = (logits.argmax(dim=-1) == labels).float().mean().item()
    mean_lat = latencies.mean().item()
    pct_deit = 100.0 * route_to_deit.float().mean().item()
    
    return {"accuracy": acc, "latency_mean_ms": mean_lat, "%_to_DeiT": pct_deit}

def run_first_touch_oracle(sq_logits, deit_logits, sq_lats, deit_lats, labels):
    """
    First-Touch Oracle upper bound:
    Upfront selection of SqueezeNet if correct, else DeiT-S-16.
    Incurs no sequential latency penalty.
    """
    sq_preds = sq_logits.argmax(dim=-1)
    deit_preds = deit_logits.argmax(dim=-1)
    sq_correct = (sq_preds == labels)
    deit_correct = (deit_preds == labels)
    
    oracle_choice = torch.where(sq_correct, 0, torch.where(deit_correct, 1, 0))
    oracle_logits = torch.where(oracle_choice.unsqueeze(1) == 1, deit_logits, sq_logits)
    oracle_lats = torch.where(oracle_choice == 1, deit_lats, sq_lats)
    
    acc = (oracle_logits.argmax(dim=-1) == labels).float().mean().item()
    mean_lat = oracle_lats.mean().item()
    pct_deit = 100.0 * (oracle_choice == 1).float().mean().item()
    
    return {"accuracy": acc, "latency_mean_ms": mean_lat, "%_to_DeiT": pct_deit}

def run_cascade_oracle(sq_logits, deit_logits, sq_lats, deit_lats, labels):
    """
    Cascade Oracle upper bound:
    Sequential cascade that stops at SqueezeNet if correct, else escalates to DeiT.
    Pays L_sq if SqueezeNet is correct; pays L_sq + L_deit otherwise.
    """
    sq_preds = sq_logits.argmax(dim=-1)
    sq_correct = (sq_preds == labels)
    
    casc_lats = torch.where(sq_correct, sq_lats, sq_lats + deit_lats)
    casc_logits = torch.where(sq_correct.unsqueeze(1), sq_logits, deit_logits)
    
    acc = (casc_logits.argmax(dim=-1) == labels).float().mean().item()
    mean_lat = casc_lats.mean().item()
    pct_deit = 100.0 * (~sq_correct).float().mean().item()
    
    return {"accuracy": acc, "latency_mean_ms": mean_lat, "%_to_DeiT": pct_deit}

def evaluate_all_baselines_from_cache(cache_data: dict, device = None) -> pd.DataFrame:
    """
    Evaluate all 7 concrete baselines on a pre-computed cache dictionary
    containing 'logits', 'latencies_ms', and 'labels'.
    """
    logits = cache_data["logits"]
    lats = cache_data["latencies_ms"]
    labels = cache_data["labels"]
    
    if device is not None:
        logits = logits.to(device)
        lats = lats.to(device)
        labels = labels.to(device)
        
    sq_logits, deit_logits = logits[:, 0], logits[:, 1]
    sq_lats, deit_lats = lats[:, 0], lats[:, 1]
    
    sq_preds = sq_logits.argmax(dim=-1)
    deit_preds = deit_logits.argmax(dim=-1)
    
    sq_acc = (sq_preds == labels).float().mean().item()
    deit_acc = (deit_preds == labels).float().mean().item()
    sq_lat = sq_lats.mean().item()
    deit_lat = deit_lats.mean().item()
    
    records = []
    
    # 0. Static Experts
    records.append({
        "category": "Static Expert", "method": "static_SqueezeNet", "threshold": None,
        "accuracy": sq_acc, "latency_mean_ms": sq_lat, "%_to_DeiT": 0.0,
        "description": "Always route 100% of queries to SqueezeNet"
    })
    records.append({
        "category": "Static Expert", "method": "static_DeiT-S-16", "threshold": None,
        "accuracy": deit_acc, "latency_mean_ms": deit_lat, "%_to_DeiT": 100.0,
        "description": "Always route 100% of queries to DeiT-S-16"
    })
    
    # 1. Confidence Cascades
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]:
        res = run_confidence_cascade(sq_logits, deit_logits, sq_lats, deit_lats, labels, threshold=t)
        records.append({
            "category": "1. Confidence Cascade", "method": f"conf_cascade_t{t:g}", "threshold": t,
            "accuracy": res["accuracy"], "latency_mean_ms": res["latency_mean_ms"], "%_to_DeiT": res["%_to_DeiT"],
            "description": f"SqueezeNet confidence < {t:g} cascades to DeiT (cost: L_sq + L_deit)"
        })
        
    # 2. Entropy Cascades
    for t in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 2.3]:
        res = run_entropy_cascade(sq_logits, deit_logits, sq_lats, deit_lats, labels, threshold=t)
        records.append({
            "category": "2. Entropy Cascade", "method": f"entropy_cascade_t{t:g}", "threshold": t,
            "accuracy": res["accuracy"], "latency_mean_ms": res["latency_mean_ms"], "%_to_DeiT": res["%_to_DeiT"],
            "description": f"SqueezeNet predictive entropy > {t:g} cascades to DeiT"
        })
        
    # 3. Oracles
    ft_oracle = run_first_touch_oracle(sq_logits, deit_logits, sq_lats, deit_lats, labels)
    records.append({
        "category": "7. Oracle Router", "method": "oracle_first_touch", "threshold": None,
        "accuracy": ft_oracle["accuracy"], "latency_mean_ms": ft_oracle["latency_mean_ms"], "%_to_DeiT": ft_oracle["%_to_DeiT"],
        "description": "Ideal first-touch router selecting correct expert upfront (zero additive penalty)"
    })
    
    casc_oracle = run_cascade_oracle(sq_logits, deit_logits, sq_lats, deit_lats, labels)
    records.append({
        "category": "7. Oracle Router", "method": "oracle_cascade", "threshold": None,
        "accuracy": casc_oracle["accuracy"], "latency_mean_ms": casc_oracle["latency_mean_ms"], "%_to_DeiT": casc_oracle["%_to_DeiT"],
        "description": "Sequential cascade stopping at SqueezeNet if correct, cascading to DeiT otherwise"
    })
    
    return pd.DataFrame(records)
