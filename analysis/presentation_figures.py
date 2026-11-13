"""Generate presentation figures for lab meeting talk."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT = "notes/presentation_figures"
os.makedirs(OUT, exist_ok=True)

# Style defaults
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
})

# Palette
BLUE = "#4878CF"
LIGHT_BLUE = "#77BEDB"
DARK_BLUE = "#2A5DAA"
GRAY = "#999999"
GREEN = "#6ACC65"
RED = "#D65F5F"
PURPLE = "#B47CC7"

DPI = 200


# ── Figure 1: Model comparison (per-gene correlation) ────────────────────────

def fig1():
    models = ["Shared", "Cond", "Cond+XR"]
    corr = [0.788, 0.941, 0.979]
    baseline = 0.821  # raw nuclei vs raw cell

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(models))
    w = 0.5

    colors = ["#a8c4e0", "#6197d0", DARK_BLUE]
    bars = ax.bar(x, corr, w, color=colors)

    # Baseline line
    ax.axhline(baseline, ls="--", color=GRAY, lw=1.2, zorder=0)
    ax.text(len(models) - 0.5, baseline + 0.005, f"raw nuclei baseline ({baseline})",
            fontsize=10, color="#666666", ha="right", va="bottom")

    # Value labels
    for bar, val in zip(bars, corr):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Per-Gene Pearson Correlation")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.65, 1.05)
    ax.set_title("Gene Expression Correlation with Raw Cell\n(cell-type means, per gene across 17 cell types)",
                 fontsize=14)
    plt.tight_layout()
    fig.savefig(f"{OUT}/model_comparison_corr.png", dpi=DPI)
    plt.close(fig)
    print("  Saved model_comparison_corr.png")


# ── Figure 2: Classification accuracy ────────────────────────────────────────

def fig2():
    labels = ["Raw nuclei", "Shared", "Cond", "Cond+XR"]
    acc = [0.851, 0.738, 0.940, 0.958]
    colors = [GRAY, "#a8c4e0", "#6197d0", DARK_BLUE]

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(labels))
    bars = ax.bar(x, acc, 0.55, color=colors)

    # Baseline line
    ax.axhline(0.851, ls="--", color=GRAY, lw=1.2, zorder=0)
    ax.text(len(labels) - 0.5, 0.854, "raw nuclei baseline", fontsize=10,
            color="#666666", ha="right", va="bottom")

    # Value labels
    for bar, val in zip(bars, acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val*100:.1f}%", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Classification Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.65, 1.0)
    ax.set_title("Cell-Type Classification Accuracy", fontsize=15)
    plt.tight_layout()
    fig.savefig(f"{OUT}/classification_accuracy.png", dpi=DPI)
    plt.close(fig)
    print("  Saved classification_accuracy.png")


# ── Figure 3: Per-cell-type DE overlap ────────────────────────────────────────

def fig3():
    data = [
        ("Lamp5",     0.87, 0.57),
        ("L6 CT",     0.85, 0.57),
        ("L6 IT",     0.83, 0.48),
        ("Vip",       0.82, 0.42),
        ("L5/6 NP",   0.82, 0.59),
        ("L2/3 IT",   0.80, 0.50),
        ("Sst",       0.80, 0.61),
        ("L5 IT",     0.73, 0.54),
        ("L5 ET",     0.61, 0.35),
        ("L6b",       0.61, 0.40),
        ("Pvalb",     0.56, 0.38),
        ("Sncg",      0.31, 0.52),
        ("Macrophage", 0.07, 0.50),
        ("Oligo",     0.06, 0.54),
        ("OPC",       0.06, 0.18),
        ("Astro",     0.04, 0.36),
        ("Endo",      0.02, 0.49),
    ]
    # Already sorted by imputed descending
    ct   = [d[0] for d in data]
    imp  = [d[1] for d in data]
    raw  = [d[2] for d in data]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(ct))
    w = 0.35

    ax.bar(x - w / 2, imp, w, label="Imputed (Cond+XR)", color=BLUE)
    ax.bar(x + w / 2, raw, w, label="Raw nuclei", color=GRAY)

    # Recon ceiling line
    ax.axhline(0.55, ls="--", color=RED, lw=1.2, zorder=0)
    ax.text(0.3, 0.56, "recon ceiling", fontsize=10, color=RED, va="bottom")

    # Bracket for non-neuronal types (last 5: indices 12–16)
    bracket_start = 12 - 0.5
    bracket_end = 16 + 0.5
    y_br = -0.08
    ax.annotate("", xy=(bracket_start, y_br), xytext=(bracket_end, y_br),
                arrowprops=dict(arrowstyle="-", lw=1.5, color=RED),
                annotation_clip=False)
    # Vertical ticks on bracket
    for bx in [bracket_start, bracket_end]:
        ax.annotate("", xy=(bx, y_br), xytext=(bx, y_br + 0.03),
                     arrowprops=dict(arrowstyle="-", lw=1.5, color=RED),
                     annotation_clip=False)
    ax.text((bracket_start + bracket_end) / 2, y_br - 0.03,
            "gene filtering", ha="center", va="top", fontsize=10,
            color=RED, clip_on=False)

    ax.set_ylabel("DE Gene Overlap")
    ax.set_xticks(x)
    ax.set_xticklabels(ct, rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Per-Cell-Type DE Overlap with Raw Cell", fontsize=15)
    plt.tight_layout()
    # Extra bottom margin for bracket
    fig.subplots_adjust(bottom=0.25)
    fig.savefig(f"{OUT}/per_ct_de_overlap.png", dpi=DPI)
    plt.close(fig)
    print("  Saved per_ct_de_overlap.png")


# ── Figure 4: Recon ceiling comparison ────────────────────────────────────────

def fig4():
    metrics = ["Classification", "DE overlap", "Coexpression", "Per-cell corr"]
    recon   = [0.926, 0.534, 0.977, 0.842]
    xdec    = [0.958, 0.521, 0.915, 0.842]
    raw_nuc = [0.851, 0.471, 0.818, 0.843]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))
    w = 0.24

    bars1 = ax.bar(x - w, recon, w, label="Recon ceiling", color=LIGHT_BLUE)
    bars2 = ax.bar(x, xdec, w, label="Cross-decode (Cond+XR)", color=DARK_BLUE)
    bars3 = ax.bar(x + w, raw_nuc, w, label="Raw nuclei", color=GRAY)

    # Value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.12)
    ax.legend(frameon=False, fontsize=11)
    ax.set_title("Reconstruction Ceiling vs Cross-Decode vs Raw Nuclei", fontsize=15)
    plt.tight_layout()
    fig.savefig(f"{OUT}/recon_ceiling_comparison.png", dpi=DPI)
    plt.close(fig)
    print("  Saved recon_ceiling_comparison.png")


if __name__ == "__main__":
    print("Generating presentation figures...")
    fig1()
    fig2()
    fig3()
    fig4()
    print("Done.")
