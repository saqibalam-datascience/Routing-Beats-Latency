"""
Command-Line Interface to Train Gumbel-Softmax Gater(s).
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from routing_beats_latency import (
    train_gumbel_gater, train_all_gaters,
    CACHE_DIR, GATER_DIR, LAMBDAS, SEEDS
)

def main():
    parser = argparse.ArgumentParser(description="Train Gumbel-Softmax Gater")
    parser.add_argument("--lambda", dest="lambda_val", type=float, default=0.70, help="Pareto lambda penalty parameter")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--all", action="store_true", help="Train all 27 gaters across full lambda and seed grid")
    parser.add_argument("--cache-dir", type=str, default=CACHE_DIR, help="Path to cache directory")
    parser.add_argument("--gater-dir", type=str, default=GATER_DIR, help="Path to save gaters")
    args = parser.parse_args()

    if args.all:
        print(f"Training all {len(LAMBDAS) * len(SEEDS)} gaters across grid...")
        train_all_gaters(cache_dir=args.cache_dir, gater_dir=args.gater_dir)
    else:
        print(f"Training Gumbel Gater (Lambda = {args.lambda_val:.2f}, Seed = {args.seed})...")
        save_path = train_gumbel_gater(
            lambda_val=args.lambda_val,
            epochs=args.epochs,
            lr=args.lr,
            seed=args.seed,
            cache_dir=args.cache_dir,
            gater_dir=args.gater_dir
        )
        print(f"Gater ready at: {save_path}")

if __name__ == "__main__":
    main()
