"""
Publication-Quality Visualization Tools for Routing Beats Latency.

Produces all primary figures used in the TMLR manuscript:
1. Pareto Frontier Comparison (First-Touch Router vs. Baselines)
2. Accuracy & Latency vs. Lambda Trade-Off Curves
3. Dynamic Routing Distribution Across Operating Points
4. Multi-Seed Stability and Variance Error Bars
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import PLOTS_DIR

def set_publication_style():
    """Apply consistent publication-grade matplotlib style."""
    plt.rcParams.update({
        "font.sans-serif": "Arial",
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "axes.grid": True,
        "grid.alpha": 0.3
    })

def plot_pareto_frontier(gumbel_df: pd.DataFrame, baselines_df: pd.DataFrame = None, save_path: str = None):
    """Plot the continuous Pareto frontier of Gumbel Router against baseline curves."""
    set_publication_style()
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

    # Plot Gumbel Router Pareto Curve
    g_sorted = gumbel_df.sort_values("Latency_Mean_ms")
    ax.plot(
        g_sorted["Latency_Mean_ms"], g_sorted["Accuracy_Mean"] * 100,
        "o-", color="#1f77b4", linewidth=2.5, markersize=8, label="First-Touch Gumbel Router (Ours)"
    )

    for _, row in g_sorted.iterrows():
        ax.annotate(
            f"λ={row['Lambda']:.2f}",
            (row["Latency_Mean_ms"], row["Accuracy_Mean"] * 100),
            textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color="#1f77b4"
        )

    # Plot Baselines if provided
    if baselines_df is not None:
        # Confidence Cascade
        casc = baselines_df[baselines_df["category"] == "1. Confidence Cascade"].sort_values("latency_mean_ms")
        if len(casc):
            ax.plot(casc["latency_mean_ms"], casc["accuracy"] * 100, "s--", color="#ff7f0e", linewidth=1.8, label="Confidence Cascade")

        # Static Experts
        sq = baselines_df[baselines_df["method"] == "static_SqueezeNet"]
        deit = baselines_df[baselines_df["method"] == "static_DeiT-S-16"]
        if len(sq):
            ax.scatter(sq["latency_mean_ms"], sq["accuracy"] * 100, color="#2ca02c", s=120, zorder=5, marker="^", label="SqueezeNet 1.0 (Anchor)")
        if len(deit):
            ax.scatter(deit["latency_mean_ms"], deit["accuracy"] * 100, color="#d62728", s=120, zorder=5, marker="*", label="DeiT-S-16 (Anchor)")

        # Oracle
        orc = baselines_df[baselines_df["method"] == "oracle_first_touch"]
        if len(orc):
            ax.scatter(orc["latency_mean_ms"], orc["accuracy"] * 100, color="#9467bd", s=120, zorder=5, marker="D", label="First-Touch Oracle")

    ax.set_xlabel("Mean Inference Latency (ms, Single-Sample B=1)")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("Empirical Latency-Accuracy Pareto Frontier")
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()

    if save_path is None:
        save_path = os.path.join(PLOTS_DIR, "gumbel_pareto_frontier.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path

def plot_routing_distribution(gumbel_summary_df: pd.DataFrame, save_path: str = None):
    """Plot the stacked bar chart of routing percentage to SqueezeNet vs. DeiT across lambdas."""
    set_publication_style()
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)

    g_sorted = gumbel_summary_df.sort_values("Lambda")
    x = np.arange(len(g_sorted))
    width = 0.55

    sq_pct = g_sorted["%_to_SqueezeNet"]
    deit_pct = g_sorted["%_to_DeiT"]

    ax.bar(x, sq_pct, width, label="SqueezeNet 1.0 (Fast CNN)", color="#2ca02c", alpha=0.85)
    ax.bar(x, deit_pct, width, bottom=sq_pct, label="DeiT-S-16 (Robust ViT)", color="#1f77b4", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{l:.2f}" for l in g_sorted["Lambda"]])
    ax.set_xlabel("Pareto Parameter (λ)")
    ax.set_ylabel("Routing Allocation (%)")
    ax.set_title("First-Touch Dynamic Routing Distribution Across λ")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()

    if save_path is None:
        save_path = os.path.join(PLOTS_DIR, "gumbel_routing_distribution.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path
