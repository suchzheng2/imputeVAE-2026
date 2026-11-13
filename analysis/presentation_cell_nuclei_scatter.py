"""Presentation-ready scatter plot: cell vs nuclei mean expression.

Clean version for slides: no gene labels, 3-tier log2FC coloring,
Pearson r + p-value annotation.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV = os.path.join(PROJECT_ROOT, "analysis", "filtering_cutoffs_v3",
                         "cell_vs_nuclei_gene_diff_v3.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "notes", "presentation_figures")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_CSV, index_col=0)
    print(f"Loaded {len(df)} genes from {INPUT_CSV}")

    cm = df["cell_mean"].values + 1e-6
    nm = df["nuclei_mean"].values + 1e-6

    # Pearson r on log-transformed means (matches the log-log axes)
    r, p = stats.pearsonr(np.log10(cm), np.log10(nm))
    print(f"Pearson r (log10): {r:.4f}, p = {p:.2e}")

    # 3-tier coloring by |log2FC|
    log2fc = df["log_mean_ratio"].values
    tier = np.where(
        np.abs(log2fc) > 3, 2,       # >8-fold
        np.where(np.abs(log2fc) > 1, 1, 0)  # >2-fold
    )
    color_map = {0: "#4682B4", 1: "#E8912D", 2: "#DC3545"}  # blue, orange, red
    colors = [color_map[t] for t in tier]

    n_similar = (tier == 0).sum()
    n_moderate = (tier == 1).sum()
    n_extreme = (tier == 2).sum()

    # Plot
    fig, ax = plt.subplots(figsize=(7, 7))

    # Draw in order: blue first, then orange, then red on top
    for t_val, label in [(0, f"|log$_2$FC| < 1 (n={n_similar})"),
                          (1, f"1 < |log$_2$FC| < 3 (n={n_moderate})"),
                          (2, f"|log$_2$FC| > 3 (n={n_extreme})")]:
        mask = tier == t_val
        ax.scatter(cm[mask], nm[mask], c=color_map[t_val], s=5, alpha=0.5,
                   edgecolors="none", label=label, rasterized=True)

    # Identity line
    lims = [1e-4, max(cm.max(), nm.max()) * 2]
    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Cell mean expression (raw counts)", fontsize=13)
    ax.set_ylabel("Nuclei mean expression (raw counts)", fontsize=13)
    ax.set_title("Gene mean expression: Cell vs Nuclei", fontsize=14, fontweight="bold")

    # r and p annotation
    if p < 1e-300:
        p_str = "p < 1e-300"
    else:
        p_str = f"p = {p:.1e}"
    ax.text(0.05, 0.95, f"Pearson r = {r:.3f}\n{p_str}\nn = {len(df)} genes",
            transform=ax.transAxes, fontsize=12, va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray",
                      alpha=0.9))

    ax.legend(fontsize=10, loc="lower right", framealpha=0.9)
    ax.tick_params(labelsize=11)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "cell_vs_nuclei_mean_expr_scatter_presentation.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
