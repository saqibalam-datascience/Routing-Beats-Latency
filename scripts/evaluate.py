"""
Command-Line Interface to Evaluate First-Touch Gumbel Routers and Baselines.
"""
import os
import sys
import argparse

# Add src to pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from routing_beats_latency import (
    evaluate_all_gumbel_models,
    evaluate_all_baselines_from_cache,
    CACHE_DIR, GATER_DIR
)
import torch

def main():
    parser = argparse.ArgumentParser(description="Evaluate Gumbel Routers on Cached Test Sets")
    parser.add_argument("--cache-dir", type=str, default=CACHE_DIR, help="Path to cache directory")
    parser.add_argument("--gater-dir", type=str, default=GATER_DIR, help="Path to gater checkpoints")
    parser.add_argument("--split", type=str, default="test", choices=["test", "unseen"], help="Which cache to evaluate")
    args = parser.parse_args()

    fname = "test_mixed.pt" if args.split == "test" else "test_unseen_mixed.pt"
    cache_path = os.path.join(args.cache_dir, fname)

    print("=" * 80)
    print(f"EVALUATING ROUTING BEATS LATENCY ON: {fname}")
    print("=" * 80)

    if not os.path.exists(cache_path):
        print(f"Error: Cache file not found at {cache_path}")
        sys.exit(1)

    # 1. Evaluate Gumbel Models
    print("\n--- Gumbel Router Pareto Sweep Summary (Averaged over 3 seeds) ---")
    df_raw, df_summary = evaluate_all_gumbel_models(cache_path=cache_path, gater_dir=args.gater_dir)
    if len(df_summary):
        print(df_summary.to_string(index=False))
    else:
        print("No trained gater checkpoints found in", args.gater_dir)

    # 2. Evaluate Baselines if on test_mixed
    if args.split == "test":
        print("\n--- Reviewer Adaptive Inference Baselines ---")
        c = torch.load(cache_path, map_location="cpu", weights_only=False)
        base_df = evaluate_all_baselines_from_cache(c)
        summary_rows = base_df[base_df["method"].isin([
            "static_SqueezeNet", "static_DeiT-S-16",
            "conf_cascade_t0.5", "conf_cascade_t0.85",
            "entropy_cascade_t0.5", "entropy_cascade_t1",
            "oracle_first_touch", "oracle_cascade"
        ])]
        print(summary_rows[["category", "method", "accuracy", "latency_mean_ms", "%_to_DeiT"]].to_string(index=False))

    print("\nEvaluation completed successfully.")

if __name__ == "__main__":
    main()
