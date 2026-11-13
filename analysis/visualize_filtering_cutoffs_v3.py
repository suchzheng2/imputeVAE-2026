"""Visualize gene/cell filtering cutoffs for v3 chemistry only.

Same analysis as visualize_filtering_cutoffs.py but restricted to v3 data,
since we train models on v3 only (--protocol v3).
"""

import os
import re
import sys

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp


DATA_ROOT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi"
DATASETS = {
    "cell": ("10X_cells_v2_v3_AIBS", r"c3A"),
    "nuclei": ("10X_nuclei_v2_v3_AIBS", r"n3A"),
}


def load_data_v3(dataset_name, v3_suffix):
    """Load h5ad, filter to v3 chemistry, compute stats from sparse matrix."""
    path = os.path.join(DATA_ROOT, dataset_name, "data.h5ad")
    print(f"Loading {path}...")
    adata = ad.read_h5ad(path, backed="r")

    # Derive meta_batch_id (strip replicate suffix after '.')
    meta_batch = adata.obs["batch_id"].apply(lambda x: x.split(".")[0])
    v3_mask = meta_batch.str.endswith(v3_suffix).values
    v3_indices = np.where(v3_mask)[0]
    print(f"  Total: {adata.shape[0]} cells, v3 ({v3_suffix}): {len(v3_indices)} cells")

    # Read v3 data in chunks (sparse-friendly)
    n_v3 = len(v3_indices)
    n_genes = adata.shape[1]

    lib_sizes = np.zeros(n_v3, dtype=np.float64)
    genes_per_cell = np.zeros(n_v3, dtype=np.int64)
    # For gene stats, accumulate across chunks
    gene_sums = np.zeros(n_genes, dtype=np.float64)
    gene_nonzero = np.zeros(n_genes, dtype=np.int64)

    chunk_size = 5000
    for ci, start in enumerate(range(0, n_v3, chunk_size)):
        end = min(start + chunk_size, n_v3)
        idx = v3_indices[start:end]
        X_chunk = adata.X[idx]
        if sp.issparse(X_chunk):
            X_chunk_dense = X_chunk.toarray()
        else:
            X_chunk_dense = np.asarray(X_chunk)

        lib_sizes[start:end] = X_chunk_dense.sum(axis=1)
        genes_per_cell[start:end] = (X_chunk_dense > 0).sum(axis=1)
        gene_sums += X_chunk_dense.sum(axis=0)
        gene_nonzero += (X_chunk_dense > 0).sum(axis=0)

        if ci % 10 == 0:
            print(f"  chunk {start}-{end} / {n_v3}")

    gene_means = gene_sums / n_v3
    detection_rate = gene_nonzero / n_v3

    adata.file.close()

    gene_names = None  # not needed for plots

    stats = {
        "lib_sizes": lib_sizes,
        "genes_per_cell": genes_per_cell,
        "gene_means": gene_means,
        "detection_rate": detection_rate,
        "gene_nonzero": gene_nonzero,
        "n_cells": n_v3,
        "n_genes": n_genes,
    }
    return stats


def plot_gene_detection_rate(stats, modality, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    det_rate = stats["detection_rate"] * 100

    ax = axes[0]
    ax.hist(det_rate, bins=100, color="steelblue", edgecolor="none", alpha=0.8)
    ax.set_xlabel("Detection rate (%)")
    ax.set_ylabel("Number of genes")
    ax.set_title(f"{modality} (v3 only): Gene detection rate distribution")
    ax.set_yscale("log")
    for cutoff, color in [(1, "orange"), (5, "red"), (10, "darkred")]:
        n_survive = (det_rate >= cutoff).sum()
        ax.axvline(cutoff, color=color, linestyle="--", linewidth=1.5,
                   label=f"{cutoff}% → {n_survive} genes ({n_survive/len(det_rate)*100:.0f}%)")
    ax.legend(fontsize=8)

    ax = axes[1]
    thresholds = np.linspace(0, 50, 500)
    surviving = [(det_rate >= t).sum() for t in thresholds]
    ax.plot(thresholds, surviving, color="steelblue", linewidth=2)
    ax.set_xlabel("Min detection rate (%)")
    ax.set_ylabel("Genes surviving")
    ax.set_title(f"{modality} (v3 only): Genes surviving at each detection rate cutoff")
    for cutoff, color in [(1, "orange"), (5, "red"), (10, "darkred")]:
        n = (det_rate >= cutoff).sum()
        ax.axvline(cutoff, color=color, linestyle="--", linewidth=1.5,
                   label=f"{cutoff}% → {n} genes")
        ax.plot(cutoff, n, 'o', color=color, markersize=8)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 50)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{modality}_v3_gene_detection_rate.png"), dpi=150)
    plt.close()


def plot_gene_mean_expression(stats, modality, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    gm = stats["gene_means"]
    gm_nz = gm[gm > 0]
    log_gm = np.log10(gm_nz)

    ax = axes[0]
    ax.hist(log_gm, bins=100, color="forestgreen", edgecolor="none", alpha=0.8)
    ax.set_xlabel("log10(mean expression)")
    ax.set_ylabel("Number of genes")
    ax.set_title(f"{modality} (v3 only): Gene mean expression distribution")
    for cutoff, color, label in [(-3, "orange", "0.001"), (-2, "red", "0.01"), (-1, "darkred", "0.1")]:
        n_survive = (log_gm >= cutoff).sum()
        ax.axvline(cutoff, color=color, linestyle="--", linewidth=1.5,
                   label=f">{label} → {n_survive} genes")
    ax.legend(fontsize=8)

    ax = axes[1]
    thresholds = np.linspace(-4, 2, 500)
    surviving = [(log_gm >= t).sum() for t in thresholds]
    ax.plot(thresholds, surviving, color="forestgreen", linewidth=2)
    ax.set_xlabel("log10(min mean expression)")
    ax.set_ylabel("Genes surviving")
    ax.set_title(f"{modality} (v3 only): Genes surviving at each mean expression cutoff")
    for cutoff, color, label in [(-3, "orange", "0.001"), (-2, "red", "0.01"), (-1, "darkred", "0.1")]:
        n = (log_gm >= cutoff).sum()
        ax.axvline(cutoff, color=color, linestyle="--", linewidth=1.5,
                   label=f">{label} → {n} genes")
        ax.plot(cutoff, n, 'o', color=color, markersize=8)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{modality}_v3_gene_mean_expression.png"), dpi=150)
    plt.close()


def plot_cell_library_size(stats, modality, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    lib = stats["lib_sizes"]
    log_lib = np.log10(lib + 1)

    p1, p5, p10 = np.percentile(lib, [1, 5, 10])
    def _nice(v):
        if v < 100: return int(round(v / 10) * 10)
        if v < 1000: return int(round(v / 50) * 50)
        if v < 10000: return int(round(v / 500) * 500)
        return int(round(v / 1000) * 1000)
    cutoffs = [_nice(p1), _nice(p5), _nice(p10)]
    if len(set(cutoffs)) < 3:
        cutoffs = [int(p1), int(p5), int(p10)]
    pct_labels = ["1st %ile", "5th %ile", "10th %ile"]
    colors = ["orange", "red", "darkred"]

    ax = axes[0]
    ax.hist(log_lib, bins=100, color="coral", edgecolor="none", alpha=0.8)
    ax.set_xlabel("log10(total counts + 1)")
    ax.set_ylabel("Number of cells")
    ax.set_title(f"{modality} (v3 only): Library size distribution")
    for c, col, pl in zip(cutoffs, colors, pct_labels):
        n = (lib >= c).sum()
        ax.axvline(np.log10(c), color=col, linestyle="--", linewidth=1.5,
                   label=f">{c} ({pl}) → {n} cells ({n/len(lib)*100:.1f}%)")
    ax.legend(fontsize=7)

    ax = axes[1]
    thresholds = np.logspace(np.log10(max(10, lib.min() * 0.5)), np.log10(lib.max() * 1.1), 500)
    surviving = [(lib >= t).sum() for t in thresholds]
    ax.plot(thresholds, surviving, color="coral", linewidth=2)
    ax.set_xlabel("Min total counts")
    ax.set_ylabel("Cells surviving")
    ax.set_title(f"{modality} (v3 only): Cells surviving at each library size cutoff")
    ax.set_xscale("log")
    for c, col, pl in zip(cutoffs, colors, pct_labels):
        n = (lib >= c).sum()
        ax.axvline(c, color=col, linestyle="--", linewidth=1.5,
                   label=f">{c} ({pl}) → {n} cells")
        ax.plot(c, n, 'o', color=col, markersize=8)
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{modality}_v3_cell_library_size.png"), dpi=150)
    plt.close()


def plot_cell_genes_expressed(stats, modality, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    gpc = stats["genes_per_cell"]

    p1, p5, p10 = np.percentile(gpc, [1, 5, 10])
    def _nice(v):
        if v < 100: return int(round(v / 10) * 10)
        if v < 1000: return int(round(v / 50) * 50)
        if v < 5000: return int(round(v / 100) * 100)
        return int(round(v / 500) * 500)
    cutoffs = [_nice(p1), _nice(p5), _nice(p10)]
    if len(set(cutoffs)) < 3:
        cutoffs = [int(p1), int(p5), int(p10)]
    pct_labels = ["1st %ile", "5th %ile", "10th %ile"]
    colors = ["orange", "red", "darkred"]

    ax = axes[0]
    ax.hist(gpc, bins=100, color="mediumpurple", edgecolor="none", alpha=0.8)
    ax.set_xlabel("Number of expressed genes")
    ax.set_ylabel("Number of cells")
    ax.set_title(f"{modality} (v3 only): Genes expressed per cell")
    for c, col, pl in zip(cutoffs, colors, pct_labels):
        n = (gpc >= c).sum()
        ax.axvline(c, color=col, linestyle="--", linewidth=1.5,
                   label=f">{c} ({pl}) → {n} cells ({n/len(gpc)*100:.1f}%)")
    ax.legend(fontsize=7)

    ax = axes[1]
    thresholds = np.linspace(0, gpc.max() * 1.05, 500)
    surviving = [(gpc >= t).sum() for t in thresholds]
    ax.plot(thresholds, surviving, color="mediumpurple", linewidth=2)
    ax.set_xlabel("Min expressed genes")
    ax.set_ylabel("Cells surviving")
    ax.set_title(f"{modality} (v3 only): Cells surviving at each min-genes cutoff")
    for c, col, pl in zip(cutoffs, colors, pct_labels):
        n = (gpc >= c).sum()
        ax.axvline(c, color=col, linestyle="--", linewidth=1.5,
                   label=f">{c} ({pl}) → {n} cells")
        ax.plot(c, n, 'o', color=col, markersize=8)
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{modality}_v3_cell_genes_expressed.png"), dpi=150)
    plt.close()


def plot_joint_detection_vs_expression(stats, modality, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    det = stats["detection_rate"] * 100
    gm = stats["gene_means"]
    mask = gm > 0
    log_gm = np.log10(gm[mask])
    det_m = det[mask]

    hb = ax.hexbin(log_gm, det_m, gridsize=80, cmap="viridis", mincnt=1, bins="log")
    plt.colorbar(hb, ax=ax, label="Gene count (log)")
    ax.set_xlabel("log10(mean expression)")
    ax.set_ylabel("Detection rate (%)")
    ax.set_title(f"{modality} (v3 only): Detection rate vs mean expression")

    ax.axhline(1, color="orange", linestyle="--", alpha=0.7, label="1% detection")
    ax.axhline(5, color="red", linestyle="--", alpha=0.7, label="5% detection")
    ax.axvline(-2, color="orange", linestyle=":", alpha=0.7, label="mean=0.01")
    ax.axvline(-1, color="red", linestyle=":", alpha=0.7, label="mean=0.1")
    ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{modality}_v3_joint_detection_expression.png"), dpi=150)
    plt.close()


def plot_softmax_proportion(stats, modality, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    gm = stats["gene_means"]
    total = gm.sum()
    props = np.sort(gm / total)[::-1]
    cum_props = np.cumsum(props)

    ax = axes[0]
    ax.plot(range(1, len(cum_props) + 1), cum_props * 100, color="teal", linewidth=2)
    ax.set_xlabel("Number of genes (ranked by expression)")
    ax.set_ylabel("Cumulative % of total expression")
    ax.set_title(f"{modality} (v3 only): Expression mass concentration")
    ax.set_xscale("log")
    for n_genes, color in [(10, "orange"), (100, "red"), (1000, "darkred"), (5000, "purple")]:
        if n_genes < len(cum_props):
            pct = cum_props[n_genes - 1] * 100
            ax.axvline(n_genes, color=color, linestyle="--", alpha=0.7,
                       label=f"Top {n_genes}: {pct:.1f}%")
            ax.plot(n_genes, pct, 'o', color=color, markersize=6)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 100)

    ax = axes[1]
    ax.plot(range(1, len(props) + 1), props * 100, color="teal", linewidth=1)
    ax.set_xlabel("Gene rank")
    ax.set_ylabel("% of total expression")
    ax.set_title(f"{modality} (v3 only): Per-gene expression proportion")
    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{modality}_v3_softmax_proportion.png"), dpi=150)
    plt.close()


def print_filtering_summary(cell_stats, nuclei_stats, output_dir):
    lines = []
    lines.append("=" * 80)
    lines.append("FILTERING CUTOFF SUMMARY — V3 ONLY")
    lines.append("=" * 80)

    for modality, stats in [("Cell (v3)", cell_stats), ("Nuclei (v3)", nuclei_stats)]:
        lines.append(f"\n--- {modality} ---")
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
    lines.append("RECOMMENDED CUTOFF COMBINATIONS (v3 only)")
    lines.append("=" * 80)

    for det_pct, mean_thresh, lib_min, genes_min in [
        (1, 0.01, 1000, 1000),
        (5, 0.01, 2000, 1500),
        (5, 0.1, 5000, 2000),
        (10, 0.1, 5000, 2000),
    ]:
        lines.append(f"\n  Cutoff: det>{det_pct}%, mean>{mean_thresh}, lib>{lib_min}, genes>{genes_min}")
        for modality, stats in [("Cell (v3)", cell_stats), ("Nuclei (v3)", nuclei_stats)]:
            gene_mask = (stats["detection_rate"] * 100 >= det_pct) & (stats["gene_means"] >= mean_thresh)
            cell_mask = (stats["lib_sizes"] >= lib_min) & (stats["genes_per_cell"] >= genes_min)
            lines.append(f"    {modality}: {gene_mask.sum()} genes, {cell_mask.sum()} cells")

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(output_dir, "filtering_summary_v3.txt"), "w") as f:
        f.write(text)


def main():
    output_dir = "analysis/filtering_cutoffs_v3"
    os.makedirs(output_dir, exist_ok=True)

    all_stats = {}
    for modality, (dataset, v3_suffix) in DATASETS.items():
        stats = load_data_v3(dataset, v3_suffix)
        all_stats[modality] = stats

        print(f"\nGenerating plots for {modality} (v3 only)...")
        plot_gene_detection_rate(stats, modality, output_dir)
        plot_gene_mean_expression(stats, modality, output_dir)
        plot_cell_library_size(stats, modality, output_dir)
        plot_cell_genes_expressed(stats, modality, output_dir)
        plot_joint_detection_vs_expression(stats, modality, output_dir)
        plot_softmax_proportion(stats, modality, output_dir)

    print_filtering_summary(all_stats["cell"], all_stats["nuclei"], output_dir)
    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
