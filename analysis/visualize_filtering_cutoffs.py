"""Visualize gene/cell filtering cutoffs to guide stricter filtering.

Generates plots showing:
1. Per-gene: cells expressing each gene (detection rate) — histogram + cumulative
2. Per-gene: mean expression distribution — histogram + cumulative
3. Per-cell: total counts (library size) — histogram + cumulative
4. Per-cell: number of expressed genes — histogram + cumulative
5. Joint: detection rate vs mean expression (scatter, colored by density)
6. Summary table: how many genes/cells survive at various cutoff combinations

Usage:
    python analysis/visualize_filtering_cutoffs.py [--output_dir OUTPUT_DIR]
"""

import argparse
import os
import sys

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp


DATA_ROOT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi"
DATASETS = {
    "cell": "10X_cells_v2_v3_AIBS",
    "nuclei": "10X_nuclei_v2_v3_AIBS",
}


def load_data(dataset_name):
    path = os.path.join(DATA_ROOT, dataset_name, "data.h5ad")
    print(f"Loading {path}...")
    adata = ad.read_h5ad(path)
    X = adata.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    print(f"  Shape: {X.shape}, sparse: {sp.issparse(X)}")
    return X, adata


def compute_stats(X):
    """Compute per-gene and per-cell stats from sparse matrix."""
    n_cells, n_genes = X.shape

    # Per-cell (sparse-friendly)
    lib_sizes = np.asarray(X.sum(axis=1)).ravel()
    genes_per_cell = np.asarray((X > 0).sum(axis=1)).ravel()

    # Per-gene (sparse-friendly)
    cells_per_gene = np.asarray((X > 0).sum(axis=0)).ravel()
    gene_means = np.asarray(X.mean(axis=0)).ravel()
    detection_rate = cells_per_gene / n_cells

    return {
        "lib_sizes": lib_sizes,
        "genes_per_cell": genes_per_cell,
        "cells_per_gene": cells_per_gene,
        "gene_means": gene_means,
        "detection_rate": detection_rate,
        "n_cells": n_cells,
        "n_genes": n_genes,
    }


def plot_gene_detection_rate(stats, modality, output_dir):
    """Histogram + cumulative of cells per gene (detection rate)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    det_rate = stats["detection_rate"] * 100  # percentage

    # Histogram
    ax = axes[0]
    ax.hist(det_rate, bins=100, color="steelblue", edgecolor="none", alpha=0.8)
    ax.set_xlabel("Detection rate (%)")
    ax.set_ylabel("Number of genes")
    ax.set_title(f"{modality}: Gene detection rate distribution")
    ax.set_yscale("log")
    # Add cutoff lines
    for cutoff, color in [(1, "orange"), (5, "red"), (10, "darkred")]:
        n_survive = (det_rate >= cutoff).sum()
        ax.axvline(cutoff, color=color, linestyle="--", linewidth=1.5,
                   label=f"{cutoff}% → {n_survive} genes ({n_survive/len(det_rate)*100:.0f}%)")
    ax.legend(fontsize=8)

    # Cumulative (genes surviving at each threshold)
    ax = axes[1]
    thresholds = np.linspace(0, 50, 500)
    surviving = [(det_rate >= t).sum() for t in thresholds]
    ax.plot(thresholds, surviving, color="steelblue", linewidth=2)
    ax.set_xlabel("Min detection rate (%)")
    ax.set_ylabel("Genes surviving")
    ax.set_title(f"{modality}: Genes surviving at each detection rate cutoff")
    for cutoff, color in [(1, "orange"), (5, "red"), (10, "darkred")]:
        n = (det_rate >= cutoff).sum()
        ax.axvline(cutoff, color=color, linestyle="--", linewidth=1.5,
                   label=f"{cutoff}% → {n} genes")
        ax.plot(cutoff, n, 'o', color=color, markersize=8)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 50)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{modality}_gene_detection_rate.png"), dpi=150)
    plt.close()


def plot_gene_mean_expression(stats, modality, output_dir):
    """Histogram + cumulative of mean expression per gene."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    gm = stats["gene_means"]
    gm_nz = gm[gm > 0]
    log_gm = np.log10(gm_nz)

    # Histogram in log space
    ax = axes[0]
    ax.hist(log_gm, bins=100, color="forestgreen", edgecolor="none", alpha=0.8)
    ax.set_xlabel("log10(mean expression)")
    ax.set_ylabel("Number of genes")
    ax.set_title(f"{modality}: Gene mean expression distribution")
    for cutoff, color, label in [(-3, "orange", "0.001"), (-2, "red", "0.01"), (-1, "darkred", "0.1")]:
        n_survive = (log_gm >= cutoff).sum()
        ax.axvline(cutoff, color=color, linestyle="--", linewidth=1.5,
                   label=f">{label} → {n_survive} genes")
    ax.legend(fontsize=8)

    # Cumulative
    ax = axes[1]
    thresholds = np.linspace(-4, 2, 500)
    surviving = [(log_gm >= t).sum() for t in thresholds]
    ax.plot(thresholds, surviving, color="forestgreen", linewidth=2)
    ax.set_xlabel("log10(min mean expression)")
    ax.set_ylabel("Genes surviving")
    ax.set_title(f"{modality}: Genes surviving at each mean expression cutoff")
    for cutoff, color, label in [(-3, "orange", "0.001"), (-2, "red", "0.01"), (-1, "darkred", "0.1")]:
        n = (log_gm >= cutoff).sum()
        ax.axvline(cutoff, color=color, linestyle="--", linewidth=1.5,
                   label=f">{label} → {n} genes")
        ax.plot(cutoff, n, 'o', color=color, markersize=8)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{modality}_gene_mean_expression.png"), dpi=150)
    plt.close()


def plot_cell_library_size(stats, modality, output_dir):
    """Histogram + cumulative of library sizes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    lib = stats["lib_sizes"]
    log_lib = np.log10(lib + 1)

    # Pick data-appropriate cutoffs based on percentiles
    p1, p5, p10 = np.percentile(lib, [1, 5, 10])
    # Round to nice numbers
    def _nice_round(v):
        if v < 100: return int(round(v / 10) * 10)
        if v < 1000: return int(round(v / 50) * 50)
        if v < 10000: return int(round(v / 500) * 500)
        return int(round(v / 1000) * 1000)
    cutoffs_raw = [_nice_round(p1), _nice_round(p5), _nice_round(p10)]
    # Deduplicate — if rounding collapses, use raw percentile values
    if len(set(cutoffs_raw)) < 3:
        cutoffs_raw = [int(p1), int(p5), int(p10)]
    pct_labels = ["1st %ile", "5th %ile", "10th %ile"]
    colors = ["orange", "red", "darkred"]

    # Histogram
    ax = axes[0]
    ax.hist(log_lib, bins=100, color="coral", edgecolor="none", alpha=0.8)
    ax.set_xlabel("log10(total counts + 1)")
    ax.set_ylabel("Number of cells")
    ax.set_title(f"{modality}: Library size distribution")
    for cutoff_raw, color, plabel in zip(cutoffs_raw, colors, pct_labels):
        n_survive = (lib >= cutoff_raw).sum()
        ax.axvline(np.log10(cutoff_raw), color=color, linestyle="--", linewidth=1.5,
                   label=f">{cutoff_raw} ({plabel}) → {n_survive} cells ({n_survive/len(lib)*100:.1f}%)")
    ax.legend(fontsize=7)

    # Cumulative
    ax = axes[1]
    thresholds = np.logspace(np.log10(max(10, lib.min() * 0.5)), np.log10(lib.max() * 1.1), 500)
    surviving = [(lib >= t).sum() for t in thresholds]
    ax.plot(thresholds, surviving, color="coral", linewidth=2)
    ax.set_xlabel("Min total counts")
    ax.set_ylabel("Cells surviving")
    ax.set_title(f"{modality}: Cells surviving at each library size cutoff")
    ax.set_xscale("log")
    for cutoff_raw, color, plabel in zip(cutoffs_raw, colors, pct_labels):
        n = (lib >= cutoff_raw).sum()
        ax.axvline(cutoff_raw, color=color, linestyle="--", linewidth=1.5,
                   label=f">{cutoff_raw} ({plabel}) → {n} cells")
        ax.plot(cutoff_raw, n, 'o', color=color, markersize=8)
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{modality}_cell_library_size.png"), dpi=150)
    plt.close()


def plot_cell_genes_expressed(stats, modality, output_dir):
    """Histogram + cumulative of genes expressed per cell."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    gpc = stats["genes_per_cell"]

    # Pick data-appropriate cutoffs based on percentiles
    p1, p5, p10 = np.percentile(gpc, [1, 5, 10])
    def _nice_round_genes(v):
        if v < 100: return int(round(v / 10) * 10)
        if v < 1000: return int(round(v / 50) * 50)
        if v < 5000: return int(round(v / 100) * 100)
        return int(round(v / 500) * 500)
    cutoffs = [_nice_round_genes(p1), _nice_round_genes(p5), _nice_round_genes(p10)]
    if len(set(cutoffs)) < 3:
        cutoffs = [int(p1), int(p5), int(p10)]
    pct_labels = ["1st %ile", "5th %ile", "10th %ile"]
    colors = ["orange", "red", "darkred"]

    # Histogram
    ax = axes[0]
    ax.hist(gpc, bins=100, color="mediumpurple", edgecolor="none", alpha=0.8)
    ax.set_xlabel("Number of expressed genes")
    ax.set_ylabel("Number of cells")
    ax.set_title(f"{modality}: Genes expressed per cell")
    for cutoff, color, plabel in zip(cutoffs, colors, pct_labels):
        n_survive = (gpc >= cutoff).sum()
        ax.axvline(cutoff, color=color, linestyle="--", linewidth=1.5,
                   label=f">{cutoff} ({plabel}) → {n_survive} cells ({n_survive/len(gpc)*100:.1f}%)")
    ax.legend(fontsize=7)

    # Cumulative
    ax = axes[1]
    thresholds = np.linspace(0, gpc.max() * 1.05, 500)
    surviving = [(gpc >= t).sum() for t in thresholds]
    ax.plot(thresholds, surviving, color="mediumpurple", linewidth=2)
    ax.set_xlabel("Min expressed genes")
    ax.set_ylabel("Cells surviving")
    ax.set_title(f"{modality}: Cells surviving at each min-genes cutoff")
    for cutoff, color, plabel in zip(cutoffs, colors, pct_labels):
        n = (gpc >= cutoff).sum()
        ax.axvline(cutoff, color=color, linestyle="--", linewidth=1.5,
                   label=f">{cutoff} ({plabel}) → {n} cells")
        ax.plot(cutoff, n, 'o', color=color, markersize=8)
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{modality}_cell_genes_expressed.png"), dpi=150)
    plt.close()


def plot_joint_detection_vs_expression(stats, modality, output_dir):
    """Scatter: detection rate vs mean expression, colored by density."""
    fig, ax = plt.subplots(figsize=(8, 6))
    det = stats["detection_rate"] * 100
    gm = stats["gene_means"]
    mask = gm > 0
    log_gm = np.log10(gm[mask])
    det_m = det[mask]

    # Use hexbin for density
    hb = ax.hexbin(log_gm, det_m, gridsize=80, cmap="viridis", mincnt=1,
                   xscale="linear", bins="log")
    plt.colorbar(hb, ax=ax, label="Gene count (log)")
    ax.set_xlabel("log10(mean expression)")
    ax.set_ylabel("Detection rate (%)")
    ax.set_title(f"{modality}: Detection rate vs mean expression")

    # Add cutoff lines
    ax.axhline(1, color="orange", linestyle="--", alpha=0.7, label="1% detection")
    ax.axhline(5, color="red", linestyle="--", alpha=0.7, label="5% detection")
    ax.axvline(-2, color="orange", linestyle=":", alpha=0.7, label="mean=0.01")
    ax.axvline(-1, color="red", linestyle=":", alpha=0.7, label="mean=0.1")
    ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{modality}_joint_detection_expression.png"), dpi=150)
    plt.close()


def plot_softmax_proportion(stats, modality, output_dir):
    """Show how expression mass is distributed — cumulative proportion curve."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    gm = stats["gene_means"]
    total = gm.sum()
    props = np.sort(gm / total)[::-1]
    cum_props = np.cumsum(props)

    # Cumulative proportion
    ax = axes[0]
    ax.plot(range(1, len(cum_props) + 1), cum_props * 100, color="teal", linewidth=2)
    ax.set_xlabel("Number of genes (ranked by expression)")
    ax.set_ylabel("Cumulative % of total expression")
    ax.set_title(f"{modality}: Expression mass concentration")
    ax.set_xscale("log")
    for n_genes, color in [(10, "orange"), (100, "red"), (1000, "darkred"), (5000, "purple")]:
        if n_genes < len(cum_props):
            pct = cum_props[n_genes - 1] * 100
            ax.axvline(n_genes, color=color, linestyle="--", alpha=0.7,
                       label=f"Top {n_genes}: {pct:.1f}%")
            ax.plot(n_genes, pct, 'o', color=color, markersize=6)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 100)

    # Per-gene proportion (log scale)
    ax = axes[1]
    ax.plot(range(1, len(props) + 1), props * 100, color="teal", linewidth=1)
    ax.set_xlabel("Gene rank")
    ax.set_ylabel("% of total expression")
    ax.set_title(f"{modality}: Per-gene expression proportion")
    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{modality}_softmax_proportion.png"), dpi=150)
    plt.close()


def print_filtering_summary(cell_stats, nuclei_stats, output_dir):
    """Print and save a summary table of gene/cell counts at various cutoff combos."""
    # Gene filtering: intersection of cell and nuclei filtering
    lines = []
    lines.append("=" * 80)
    lines.append("FILTERING CUTOFF SUMMARY")
    lines.append("=" * 80)

    for modality, stats in [("Cell", cell_stats), ("Nuclei", nuclei_stats)]:
        lines.append(f"\n--- {modality} data ---")
        lines.append(f"Original: {stats['n_cells']} cells, {stats['n_genes']} genes")

        lines.append(f"\nGene filtering (detection rate):")
        for pct in [0, 0.5, 1, 2, 5, 10]:
            n = (stats["detection_rate"] * 100 >= pct).sum()
            lines.append(f"  >{pct}%: {n} genes ({n/stats['n_genes']*100:.1f}%)")

        lines.append(f"\nGene filtering (mean expression):")
        for thresh in [0, 0.001, 0.01, 0.1, 1.0]:
            n = (stats["gene_means"] >= thresh).sum()
            lines.append(f"  >{thresh}: {n} genes ({n/stats['n_genes']*100:.1f}%)")

        lines.append(f"\nCell filtering (library size):")
        lib = stats["lib_sizes"]
        lib_pcts = np.percentile(lib, [1, 5, 10, 25, 50])
        lines.append(f"  Percentiles: 1%={lib_pcts[0]:.0f}, 5%={lib_pcts[1]:.0f}, 10%={lib_pcts[2]:.0f}, 25%={lib_pcts[3]:.0f}, median={lib_pcts[4]:.0f}")
        for thresh in [0, 1000, 2000, 5000, 10000, 20000]:
            n = (lib >= thresh).sum()
            lines.append(f"  >{thresh}: {n} cells ({n/stats['n_cells']*100:.1f}%)")

        lines.append(f"\nCell filtering (genes expressed):")
        gpc = stats["genes_per_cell"]
        gpc_pcts = np.percentile(gpc, [1, 5, 10, 25, 50])
        lines.append(f"  Percentiles: 1%={gpc_pcts[0]:.0f}, 5%={gpc_pcts[1]:.0f}, 10%={gpc_pcts[2]:.0f}, 25%={gpc_pcts[3]:.0f}, median={gpc_pcts[4]:.0f}")
        for thresh in [500, 1000, 1500, 2000, 3000, 5000]:
            n = (gpc >= thresh).sum()
            lines.append(f"  >{thresh}: {n} cells ({n/stats['n_cells']*100:.1f}%)")

    # Combined recommendations
    lines.append(f"\n{'='*80}")
    lines.append("RECOMMENDED CUTOFF COMBINATIONS")
    lines.append("=" * 80)

    for det_pct, mean_thresh, lib_min, genes_min in [
        (1, 0.01, 1000, 1000),
        (5, 0.01, 1000, 1000),
        (5, 0.1, 2000, 1500),
        (10, 0.1, 2000, 1500),
        (10, 0.1, 5000, 2000),
    ]:
        lines.append(f"\n  Cutoff: det>{det_pct}%, mean>{mean_thresh}, lib>{lib_min}, genes>{genes_min}")
        for modality, stats in [("Cell", cell_stats), ("Nuclei", nuclei_stats)]:
            gene_mask = (stats["detection_rate"] * 100 >= det_pct) & (stats["gene_means"] >= mean_thresh)
            cell_mask = (stats["lib_sizes"] >= lib_min) & (stats["genes_per_cell"] >= genes_min)
            lines.append(f"    {modality}: {gene_mask.sum()} genes, {cell_mask.sum()} cells")

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(output_dir, "filtering_summary.txt"), "w") as f:
        f.write(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="analysis/filtering_cutoffs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_stats = {}
    for modality, dataset in DATASETS.items():
        X, adata = load_data(dataset)
        stats = compute_stats(X)
        all_stats[modality] = stats

        print(f"\nGenerating plots for {modality}...")
        plot_gene_detection_rate(stats, modality, args.output_dir)
        plot_gene_mean_expression(stats, modality, args.output_dir)
        plot_cell_library_size(stats, modality, args.output_dir)
        plot_cell_genes_expressed(stats, modality, args.output_dir)
        plot_joint_detection_vs_expression(stats, modality, args.output_dir)
        plot_softmax_proportion(stats, modality, args.output_dir)

        del X, adata
        import gc; gc.collect()

    print_filtering_summary(all_stats["cell"], all_stats["nuclei"], args.output_dir)
    print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
