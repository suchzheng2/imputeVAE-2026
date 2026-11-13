"""
Generate ISMB extended abstract Figure 1 (multi-panel).

Panels:
  (A) Architecture schematic (programmatic)
  (B) UMAP: raw nuclei vs cross-decoded nuclei vs raw cells
  (C) Ablation bar chart: classification + DE overlap
  (D) Zero/nonzero: nonzero Pearson by sparsity bucket (model vs CT mean baseline)

Usage:
  conda activate impute2
  python analysis/ismb_figures.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os
import pandas as pd

OUT = "notes/presentation/ismb_figures"
os.makedirs(OUT, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.fontsize": 8,
    "legend.frameon": False,
})

# Palette
BLUE    = "#3274A1"
TEAL    = "#45A27D"
ORANGE  = "#E8803A"
GRAY    = "#999999"
LTGRAY  = "#CCCCCC"
DKBLUE  = "#1B4F72"
RED     = "#C0392B"
PURPLE  = "#8E44AD"

DPI = 300


# ── Panel A: Architecture schematic ──────────────────────────────────────────

def draw_architecture(ax):
    """Programmatic architecture diagram."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("(A) Architecture", fontsize=11, fontweight="bold", loc="left")

    def box(x, y, w, h, label, color, fontsize=7, alpha=0.85):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor="black", lw=0.8, alpha=alpha)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="white" if color in [DKBLUE, BLUE, PURPLE, RED] else "black")

    def arrow(x1, y1, x2, y2, color="black", style="-|>", lw=1.2, ls="-"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw, ls=ls))

    # --- Input data ---
    box(0.1, 4.2, 1.5, 0.8, "Cells\n(log1p)", ORANGE, fontsize=7)
    box(0.1, 1.0, 1.5, 0.8, "Nuclei\n(log1p)", TEAL, fontsize=7)

    # --- Encoder ---
    box(2.5, 2.4, 1.3, 1.2, "Shared\nEncoder", BLUE, fontsize=7)
    ax.text(3.15, 2.15, "1024→512", ha="center", fontsize=5.5, color="#555")

    # Arrows: inputs → encoder
    arrow(1.6, 4.6, 2.5, 3.4, color=ORANGE)
    arrow(1.6, 1.4, 2.5, 2.6, color=TEAL)

    # --- Latent ---
    box(4.4, 2.6, 1.0, 0.8, "z\n(dim 32)", PURPLE, fontsize=7)
    arrow(3.8, 3.0, 4.4, 3.0)

    # --- Modality embeddings ---
    box(5.8, 4.5, 1.3, 0.6, "Cell emb\n(dim 8)", ORANGE, fontsize=6, alpha=0.7)
    box(5.8, 0.9, 1.3, 0.6, "Nuc emb\n(dim 8)", TEAL, fontsize=6, alpha=0.7)

    # --- Concat node ---
    ax.plot(6.0, 3.0, "o", color="black", markersize=5, zorder=5)
    ax.text(6.0, 3.3, "concat", ha="center", fontsize=5.5, color="#555")
    arrow(5.4, 3.0, 5.9, 3.0)

    # Embedding arrows to concat
    arrow(6.45, 4.5, 6.1, 3.15, color=ORANGE, lw=0.9)
    arrow(6.45, 1.5, 6.1, 2.85, color=TEAL, lw=0.9)

    # --- Decoder ---
    box(6.5, 2.4, 1.3, 1.2, "Shared\nDecoder", BLUE, fontsize=7)
    ax.text(7.15, 2.15, "512→1024", ha="center", fontsize=5.5, color="#555")
    arrow(6.1, 3.0, 6.5, 3.0)

    # --- Outputs ---
    box(8.5, 4.2, 1.4, 0.8, "Cell\noutput", ORANGE, fontsize=7, alpha=0.7)
    box(8.5, 1.0, 1.4, 0.8, "Nuclei\noutput", TEAL, fontsize=7, alpha=0.7)

    arrow(7.8, 3.4, 8.5, 4.6, color=ORANGE)
    arrow(7.8, 2.6, 8.5, 1.4, color=TEAL)

    # --- Cross-reconstruction path (dashed) ---
    # Nuclei input → encoder → z → cell embedding → decoder → cell output
    # Highlight with a dashed red arrow from nuclei side to cell output
    ax.annotate("cross-\nreconstruction",
                xy=(8.5, 4.3), xytext=(4.9, 1.0),
                fontsize=6.5, color=RED, fontweight="bold",
                ha="center",
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.5,
                                ls="--", connectionstyle="arc3,rad=-0.3"))

    # --- Loss labels ---
    ax.text(9.2, 3.6, "MSE", fontsize=6.5, color="#555", ha="center", style="italic")
    ax.text(9.2, 2.4, "MSE", fontsize=6.5, color="#555", ha="center", style="italic")
    ax.text(4.9, 4.8, "JSD", fontsize=6.5, color=PURPLE, ha="center", style="italic")


# ── Panel B: UMAP (use existing image) ──────────────────────────────────────

def draw_umap_panel(ax):
    """Load existing UMAP (downsampled) and display in panel."""
    from PIL import Image
    umap_path = "run/run_2026-04-07-02-12-05_3551465/umap/valid/cell_vs_cross/e599.png"
    if os.path.exists(umap_path):
        pil_img = Image.open(umap_path)
        # Downsample to prevent OOM on login nodes
        pil_img.thumbnail((2000, 800), Image.LANCZOS)
        img = np.asarray(pil_img)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title("(B) UMAP: Real cells vs cross-decoded nuclei (epoch 599)",
                      fontsize=11, fontweight="bold", loc="left")
    else:
        ax.text(0.5, 0.5, f"UMAP not found:\n{umap_path}",
                ha="center", va="center", transform=ax.transAxes, fontsize=8)
        ax.axis("off")


# ── Panel C: Ablation bar chart ──────────────────────────────────────────────

def draw_ablation(ax_clf, ax_de):
    """Classification + DE overlap ablation, grouped bars."""
    variants = ["Raw\nnuclei", "Linear\nregression", "VAE\ncond", "VAE\ncond+xr"]
    clf  = [83.5, 93.0, 88.7, 94.1]
    de   = [45.0, 46.0, 49.5, 54.6]
    colors = [GRAY, LTGRAY, BLUE, DKBLUE]

    x = np.arange(len(variants))
    w = 0.6

    # Classification
    bars = ax_clf.bar(x, clf, w, color=colors, edgecolor="white", lw=0.5)
    for bar, val in zip(bars, clf):
        ax_clf.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax_clf.set_ylabel("Classification Accuracy (%)")
    ax_clf.set_xticks(x)
    ax_clf.set_xticklabels(variants, fontsize=7)
    ax_clf.set_ylim(70, 100)
    ax_clf.axhline(83.5, ls=":", color=GRAY, lw=0.8, zorder=0)
    ax_clf.set_title("Cell-type classification", fontsize=9, fontweight="bold")

    # DE overlap
    bars = ax_de.bar(x, de, w, color=colors, edgecolor="white", lw=0.5)
    for bar, val in zip(bars, de):
        ax_de.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax_de.set_ylabel("DE Overlap (%)")
    ax_de.set_xticks(x)
    ax_de.set_xticklabels(variants, fontsize=7)
    ax_de.set_ylim(30, 62)
    ax_de.axhline(45.0, ls=":", color=GRAY, lw=0.8, zorder=0)
    ax_de.set_title("DE gene overlap (top 100)", fontsize=9, fontweight="bold")


# ── Panel D: Zero/nonzero by sparsity ───────────────────────────────────────

def draw_zero_nonzero(ax):
    """Nonzero Pearson: model vs CT mean baseline by sparsity bucket."""
    buckets = ["Dense\n(<30%\nzero)", "Moderate\n(30-70%)", "Sparse\n(70-90%)", "Very sparse\n(>90%)"]
    model_pearson  = [0.862, 0.556, 0.542, 0.525]
    ctmean_pearson = [0.770, 0.451, 0.493, 0.484]

    x = np.arange(len(buckets))
    w = 0.32

    bars1 = ax.bar(x - w/2, model_pearson, w, label="VAE (nonzero)", color=BLUE, edgecolor="white", lw=0.5)
    bars2 = ax.bar(x + w/2, ctmean_pearson, w, label="CT mean baseline", color=LTGRAY, edgecolor="white", lw=0.5)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=6)

    ax.set_ylabel("Nonzero Pearson Correlation")
    ax.set_xticks(x)
    ax.set_xticklabels(buckets, fontsize=7)
    ax.set_ylim(0.3, 1.0)
    ax.legend(loc="upper right", fontsize=7)

    # Annotation: model adds value
    ax.annotate("+9.2pp", xy=(0 - w/2, 0.862), xytext=(-0.5, 0.92),
                fontsize=7, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.8))

    ax.set_title("Nonzero entries: model vs cell-type mean baseline",
                 fontsize=9, fontweight="bold")


# ── Composite figure ─────────────────────────────────────────────────────────

def make_composite():
    """Generate the full 4-panel ISMB figure."""
    fig = plt.figure(figsize=(14, 10), constrained_layout=False)

    # Top row: A (architecture) and B (UMAP)
    gs_top = fig.add_gridspec(1, 2, left=0.04, right=0.98, top=0.97, bottom=0.54,
                               wspace=0.05, width_ratios=[1, 1.5])
    ax_a = fig.add_subplot(gs_top[0, 0])
    ax_b = fig.add_subplot(gs_top[0, 1])

    # Bottom row: C (ablation — two subplots) and D (zero/nonzero)
    gs_bot = fig.add_gridspec(1, 3, left=0.06, right=0.98, top=0.48, bottom=0.06,
                               wspace=0.35)
    ax_c1 = fig.add_subplot(gs_bot[0, 0])
    ax_c2 = fig.add_subplot(gs_bot[0, 1])
    ax_d  = fig.add_subplot(gs_bot[0, 2])

    # Draw panels
    draw_architecture(ax_a)
    draw_umap_panel(ax_b)

    # Panel C label
    ax_c1.text(-0.15, 1.15, "(C)", transform=ax_c1.transAxes,
               fontsize=11, fontweight="bold")
    draw_ablation(ax_c1, ax_c2)

    ax_d.set_title("(D) " + ax_d.get_title() if ax_d.get_title() else "(D)",
                   fontsize=9, fontweight="bold")
    draw_zero_nonzero(ax_d)
    # Prefix title with panel label
    ax_d.set_title("(D) Nonzero entries: model vs cell-type mean baseline",
                   fontsize=9, fontweight="bold")

    fig.savefig(f"{OUT}/ismb_figure1_composite.png", dpi=200, facecolor="white")
    fig.savefig(f"{OUT}/ismb_figure1_composite.pdf", facecolor="white")
    plt.close(fig)
    print(f"  Saved {OUT}/ismb_figure1_composite.png")
    print(f"  Saved {OUT}/ismb_figure1_composite.pdf")


# ── Individual panels (for flexibility) ──────────────────────────────────────

def make_individual_panels():
    """Generate each panel as a standalone file."""

    # Panel A
    fig, ax = plt.subplots(figsize=(6, 3.5))
    draw_architecture(ax)
    fig.savefig(f"{OUT}/panel_A_architecture.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{OUT}/panel_A_architecture.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved panel_A_architecture")

    # Panel C
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))
    draw_ablation(ax1, ax2)
    fig.suptitle("(C) Architecture ablation", fontsize=11, fontweight="bold", x=0.02, ha="left")
    plt.tight_layout()
    fig.savefig(f"{OUT}/panel_C_ablation.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{OUT}/panel_C_ablation.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved panel_C_ablation")

    # Panel D
    fig, ax = plt.subplots(figsize=(5, 3.5))
    draw_zero_nonzero(ax)
    ax.set_title("(D) Nonzero entries: model vs cell-type mean baseline",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{OUT}/panel_D_zero_nonzero.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{OUT}/panel_D_zero_nonzero.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved panel_D_zero_nonzero")


if __name__ == "__main__":
    print("Generating ISMB figures...")
    make_composite()
    make_individual_panels()
    print("Done.")
