"""
Gumbel-Softmax Router Training, Loading, and Inference.

Implements:
- Straight-Through Gumbel-Softmax discrete routing
- Dynamic temperature annealing
- Hardware-grounded latency regularized loss function
- Multi-seed and multi-lambda training loops
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .config import (
    GATER_DIR, CACHE_DIR, DEVICE,
    LAMBDAS, SEEDS, GUMBEL_TAU_START, GUMBEL_TAU_END, EPOCHS_GATER, LR_GATER
)
from .models import Gater

def get_gater_mean_std(cache_path: str = None):
    """Compute or load global normalization statistics for gater inputs."""
    if cache_path is None:
        cache_path = os.path.join(CACHE_DIR, "val_mixed.pt")
        if not os.path.exists(cache_path):
            cache_path = os.path.join(CACHE_DIR, "test_mixed.pt")
            
    if os.path.exists(cache_path):
        c = torch.load(cache_path, map_location="cpu", weights_only=False)
        _g = c["gater_inputs"].float() / 255.0
        return _g.mean().item(), _g.std().item()
    return 0.449, 0.226

def load_gumbel_gater(lambda_val: float = 0.5, seed: int = 0, gater_dir: str = GATER_DIR, device = DEVICE) -> Gater:
    """Load a trained Gumbel gater checkpoint."""
    save_path = os.path.join(gater_dir, f"gater_gumbel_lambda{lambda_val:.2f}_seed{seed}.pt")
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Gater checkpoint not found: {save_path}")
    model = Gater().to(device)
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=False))
    model.eval()
    return model

def train_gumbel_gater(
    lambda_val: float = 0.5,
    epochs: int = EPOCHS_GATER,
    lr: float = LR_GATER,
    tau_start: float = GUMBEL_TAU_START,
    tau_end: float = GUMBEL_TAU_END,
    seed: int = 0,
    val_cache: dict = None,
    cache_dir: str = CACHE_DIR,
    gater_dir: str = GATER_DIR,
    device = DEVICE,
    batch_size: int = 512,
    verbose: bool = True
) -> str:
    """
    Train one Gumbel-Softmax gater for a given (lambda, seed).
    
    Args:
        lambda_val: Regularization parameter balancing task accuracy and hardware latency.
        epochs: Training epochs (default: 100).
        lr: Learning rate (default: 1e-3).
        tau_start: Initial temperature (default: 5.0).
        tau_end: Final annealed temperature (default: 0.1).
        seed: Random seed for initialization.
        val_cache: Pre-loaded validation cache dictionary.
        cache_dir: Directory containing cache files.
        gater_dir: Directory to save gater checkpoints.
        device: Device to train on.
        batch_size: Training batch size.
        verbose: Whether to log training progress.
        
    Returns:
        save_path: Filepath to the saved checkpoint.
    """
    os.makedirs(gater_dir, exist_ok=True)
    save_path = os.path.join(gater_dir, f"gater_gumbel_lambda{lambda_val:.2f}_seed{seed}.pt")
    if os.path.exists(save_path):
        if verbose:
            print(f"Gater checkpoint exists at {save_path}, skipping training.")
        return save_path

    torch.manual_seed(seed)
    model = Gater().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if val_cache is None:
        val_path = os.path.join(cache_dir, "val_mixed.pt")
        c = torch.load(val_path, map_location=device, weights_only=False)
    else:
        c = val_cache

    logits = c["logits"].to(device)
    latencies = c["latencies_ms"].to(device)
    labels = c["labels"].to(device)

    gater_mean, gater_std = get_gater_mean_std()
    gx = (c["gater_inputs"].to(device).float() / 255.0 - gater_mean) / gater_std

    min_lat = 1.0
    max_lat = 4.7
    norm_lats = torch.clamp((latencies - min_lat) / (max_lat - min_lat), 0.0, 1.0)

    probs = F.softmax(logits, dim=-1)
    p_true = probs[torch.arange(logits.size(0)), :, labels]
    norm_acc_loss = 1.0 - p_true

    n_samples = gx.size(0)
    indices = torch.arange(n_samples)

    for epoch in range(epochs):
        model.train()
        indices = indices[torch.randperm(n_samples)]
        tau = tau_start * (tau_end / tau_start) ** (epoch / max(1, epochs - 1))

        total_loss = 0.0
        for i in range(0, n_samples, batch_size):
            idx = indices[i:i + batch_size]
            optimizer.zero_grad()
            scores = model(gx[idx])
            route = F.gumbel_softmax(scores, tau=tau, hard=True)
            batch_acc_loss = (route * norm_acc_loss[idx]).sum(dim=1)
            batch_lat_loss = (route * norm_lats[idx]).sum(dim=1)

            loss = (lambda_val * batch_acc_loss + (1.0 - lambda_val) * batch_lat_loss).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)

        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Tau: {tau:.3f} | Loss: {total_loss/n_samples:.4f}")

    torch.save(model.state_dict(), save_path)
    return save_path

def train_all_gaters(lambdas = LAMBDAS, seeds = SEEDS, cache_dir: str = CACHE_DIR, gater_dir: str = GATER_DIR):
    """Train the full matrix of gaters across lambdas and seeds."""
    val_path = os.path.join(cache_dir, "val_mixed.pt")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation cache {val_path} missing. Cannot train gaters without cache.")
    
    val_cache_data = torch.load(val_path, map_location=DEVICE, weights_only=False)
    for l in lambdas:
        for s in seeds:
            print(f"\n--- Training Gumbel Gater (Lambda = {l:.2f}, Seed = {s}) ---")
            train_gumbel_gater(lambda_val=l, seed=s, val_cache=val_cache_data, gater_dir=gater_dir)
