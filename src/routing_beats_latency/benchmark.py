"""
Hardware-level Single-Sample Latency Profiling.

Uses GPU CUDA Events with explicit stream synchronization to measure real
single-sample execution latency (B=1, 20 warmup iterations, 200 timed iterations).
Includes fallback to high-resolution CPU performance counters when CUDA is unavailable.
"""
import time
import numpy as np
import torch
import torch.nn as nn

from .config import DEVICE, IMG_SIZE

def _forward_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Helper to extract raw logits across different model return types."""
    out = model(x)
    return out.logits if hasattr(out, "logits") else out

@torch.inference_mode()
def benchmark_latency(
    model: nn.Module,
    input_size: tuple = (1, 3, IMG_SIZE, IMG_SIZE),
    n_warmup: int = 20,
    n_runs: int = 200,
    device = DEVICE
) -> float:
    """
    Benchmark single-sample (B=1) inference latency in milliseconds.
    
    Args:
        model: Neural network module.
        input_size: Shape of dummy input tensor (default: (1, 3, 384, 384)).
        n_warmup: Warmup iterations to reach steady-state GPU clocks.
        n_runs: Timed evaluation iterations.
        device: Device to profile on.
        
    Returns:
        mean_ms: Mean latency per single sample in milliseconds.
    """
    dev = device or DEVICE
    model = model.to(dev)
    model.eval()
    x = torch.randn(*input_size, device=dev)

    # Warmup runs
    for _ in range(n_warmup):
        _forward_logits(model, x)

    if dev.type == "cuda":
        torch.cuda.synchronize()
        ms = []
        for _ in range(n_runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _forward_logits(model, x)
            end.record()
            torch.cuda.synchronize()
            ms.append(start.elapsed_time(end))
    else:
        ms = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _forward_logits(model, x)
            t1 = time.perf_counter()
            ms.append((t1 - t0) * 1000.0)

    return float(np.mean(ms)) / input_size[0]
