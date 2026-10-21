"""
Command-Line Interface to Benchmark Real Single-Sample GPU Inference Latency.
"""
import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from routing_beats_latency import (
    build_expert, load_expert, Gater, benchmark_latency,
    CHECKPOINT_DIR, DEVICE, IMG_SIZE, GATER_IMG_SIZE
)

def main():
    parser = argparse.ArgumentParser(description="Benchmark Single-Sample Hardware Latency (B=1)")
    parser.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_DIR, help="Path to expert checkpoints")
    parser.add_argument("--warmup", type=int, default=20, help="Number of warmup iterations")
    parser.add_argument("--runs", type=int, default=200, help="Number of timed iterations")
    args = parser.parse_args()

    print("=" * 80)
    print("HARDWARE LATENCY PROFILING (CUDA Events, Batch Size = 1)")
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 80)

    # 1. Benchmark Micro-Probe Gater
    print("\n1. Micro-Probe Gater (10-D Spatial Feature Extraction + 2-Layer MLP):")
    gater = Gater().to(DEVICE)
    gater_input = torch.randn(1, 3, GATER_IMG_SIZE, GATER_IMG_SIZE, device=DEVICE)
    # Warmup and time gater end-to-end
    for _ in range(50):
        _ = gater(gater_input)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(500):
            _ = gater(gater_input)
        end.record()
        torch.cuda.synchronize()
        gater_ms = start.elapsed_time(end) / 500.0
    else:
        import time
        t0 = time.perf_counter()
        for _ in range(500):
            _ = gater(gater_input)
        t1 = time.perf_counter()
        gater_ms = (t1 - t0) * 1000.0 / 500.0
    print(f"   Mean Gater Latency : {gater_ms:.4f} ms ({gater_ms * 1000:.1f} microseconds)")

    # 2. Benchmark SqueezeNet 1.0
    print("\n2. SqueezeNet 1.0 (Lightweight CNN):")
    try:
        sq = load_expert("SqueezeNet", checkpoint_dir=args.checkpoint_dir)
        sq_ms = benchmark_latency(sq, input_size=(1, 3, IMG_SIZE, IMG_SIZE), n_warmup=args.warmup, n_runs=args.runs)
        print(f"   Mean Latency       : {sq_ms:.2f} ms")
    except Exception as e:
        print(f"   Could not load fine-tuned checkpoint ({e}), profiling uninitialized model...")
        sq = build_expert("SqueezeNet")
        sq_ms = benchmark_latency(sq, input_size=(1, 3, IMG_SIZE, IMG_SIZE), n_warmup=args.warmup, n_runs=args.runs)
        print(f"   Mean Latency       : {sq_ms:.2f} ms")

    # 3. Benchmark DeiT-S-16
    print("\n3. DeiT-S-16 (Vision Transformer):")
    try:
        deit = load_expert("DeiT-S-16", checkpoint_dir=args.checkpoint_dir)
        deit_ms = benchmark_latency(deit, input_size=(1, 3, IMG_SIZE, IMG_SIZE), n_warmup=args.warmup, n_runs=args.runs)
        print(f"   Mean Latency       : {deit_ms:.2f} ms")
    except Exception as e:
        print(f"   Could not load fine-tuned checkpoint ({e}), profiling uninitialized model...")
        deit = build_expert("DeiT-S-16")
        deit_ms = benchmark_latency(deit, input_size=(1, 3, IMG_SIZE, IMG_SIZE), n_warmup=args.warmup, n_runs=args.runs)
        print(f"   Mean Latency       : {deit_ms:.2f} ms")

    ratio = deit_ms / max(1e-4, sq_ms)
    print("\n" + "=" * 80)
    print(f"LATENCY RATIO: DeiT-S-16 is {ratio:.2f}x slower than SqueezeNet 1.0 per image.")
    print("=" * 80)

if __name__ == "__main__":
    main()
