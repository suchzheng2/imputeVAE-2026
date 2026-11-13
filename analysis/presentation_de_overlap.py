"""Slide figure: DE overlap comparison across models and baselines.

Shows how imputation recovers differentially expressed genes compared to raw nuclei.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "notes", "qualify-exam", "pre-qual", "figures")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Data from comprehensive eval results
    models = ["Raw nuclei", "ConfigB\ncond", "ConfigB\ncond+xr", "PerCT5\ncond", "PerCT5\ncond+xr"]
    de_overlap = [0.450, 0.463, 0.521, 0.495, 0.547]
    gene_corr = [None, 0.941, 0.979, 0.909, 0.963]
    classif = [0.827, 0.940, 0.958, 0.888, 0.942]

    colors = ["#999999", "#7bafd4", "#1f77b4", "#f4b183", "#e8912d"]

    # --- Figure 1: DE overlap bar chart ---
    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(range(len(models)), [v * 100 for v in de_overlap],
                  color=colors, edgecolor="white", width=0.7)

    # Add value labels on bars
    for bar, val in zip(bars, de_overlap):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1%}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("DE overlap with real cells (%)", fontsize=12)
    ax.set_title("Differential expression recovery", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 65)
    ax.axhline(45, color="#999999", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(4.35, 45.5, "raw nuclei baseline", fontsize=9, color="#999999", ha="right")

    fig.tight_layout()
    out1 = os.path.join(OUTPUT_DIR, "slide_de_overlap.png")
    fig.savefig(out1, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out1}")

    # --- Figure 2: Multi-metric comparison ---
    fig2, axes = plt.subplots(1, 3, figsize=(14, 5))

    # DE overlap
    ax = axes[0]
    bars = ax.bar(range(len(models)), [v * 100 for v in de_overlap],
                  color=colors, edgecolor="white", width=0.7)
    for bar, val in zip(bars, de_overlap):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1%}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=8)
    ax.set_ylabel("%", fontsize=11)
    ax.set_title("DE overlap", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 65)

    # Gene correlation
    ax = axes[1]
    gc_models = [m for m, g in zip(models, gene_corr) if g is not None]
    gc_vals = [g for g in gene_corr if g is not None]
    gc_colors = [c for c, g in zip(colors, gene_corr) if g is not None]
    bars = ax.bar(range(len(gc_models)), gc_vals, color=gc_colors, edgecolor="white", width=0.7)
    for bar, val in zip(bars, gc_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(gc_models)))
    ax.set_xticklabels(gc_models, fontsize=8)
    ax.set_ylabel("Pearson r", fontsize=11)
    ax.set_title("Gene correlation → real cells", fontsize=12, fontweight="bold")
    ax.set_ylim(0.85, 1.0)

    # Classification accuracy
    ax = axes[2]
    bars = ax.bar(range(len(models)), [v * 100 for v in classif],
                  color=colors, edgecolor="white", width=0.7)
    for bar, val in zip(bars, classif):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1%}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=8)
    ax.set_ylabel("%", fontsize=11)
    ax.set_title("Cell type classification", fontsize=12, fontweight="bold")
    ax.set_ylim(70, 100)

    fig2.suptitle("Imputation quality metrics", fontsize=14, fontweight="bold", y=1.02)
    fig2.tight_layout()
    out2 = os.path.join(OUTPUT_DIR, "slide_metrics_comparison.png")
    fig2.savefig(out2, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
