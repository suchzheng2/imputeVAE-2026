"""Delta heatmaps: compact top-gene view + per-CT summary bar chart.

Usage:
    python analysis/delta_heatmap_full.py <delta_analysis_dir> [--top_n 20]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import pdist


def load_delta_matrix(tables_dir):
    """Load per-CT delta CSVs into a genes x CTs DataFrame."""
    ct_files = sorted([f for f in os.listdir(tables_dir) if f.endswith("_delta.csv")])
    if not ct_files:
        print(f"No delta CSV files found in {tables_dir}")
        sys.exit(1)
    dfs = {}
    for f in ct_files:
        ct_name = f.replace("_delta.csv", "")
        df = pd.read_csv(os.path.join(tables_dir, f))
        dfs[ct_name] = df.set_index("gene")["median_delta"]
    matrix = pd.DataFrame(dfs).fillna(0)
    return matrix


def select_top_genes(matrix, n=20, direction="positive"):
    """Select top N genes by mean delta across CTs, positive or negative."""
    global_mean = matrix.mean(axis=1)
    if direction == "positive":
        return global_mean.sort_values(ascending=False).head(n).index
    else:
        return global_mean.sort_values(ascending=True).head(n).index


def cluster_order(mat, axis=0):
    """Return reordered index after hierarchical clustering."""
    if axis == 0:
        data = mat.values
        labels = mat.index
    else:
        data = mat.values.T
        labels = mat.columns
    if len(labels) < 3:
        return labels
    dist = pdist(data, metric="correlation")
    # Handle NaN distances (constant rows)
    dist = np.nan_to_num(dist, nan=0.0)
    Z = linkage(dist, method="ward")
    order = leaves_list(Z)
    return labels[order], Z


def plot_clustered_heatmap(matrix, output_path, n=20, direction="positive", ct_order=None):
    """Compact heatmap: top N genes in one direction, both axes clustered."""
    genes = select_top_genes(matrix, n, direction)
    sub = matrix.loc[genes]  # already in rank order from select_top_genes

    # Use shared CT order if provided
    if ct_order is None:
        ct_order, _ = cluster_order(sub, axis=1)
    sub = sub[ct_order]

    fig, ax = plt.subplots(figsize=(10, 7))
    vmax = np.percentile(np.abs(sub.values), 95)
    cmap = "Reds" if direction == "positive" else "Blues_r"
    if direction == "positive":
        im = ax.imshow(sub.values, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
    else:
        im = ax.imshow(sub.values, aspect="auto", cmap=cmap, vmin=-vmax, vmax=0)

    ax.set_xticks(range(len(sub.columns)))
    ax.set_xticklabels(sub.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(sub.index)))
    ax.set_yticklabels(sub.index, fontsize=8)

    # Annotate values
    for i in range(len(sub.index)):
        for j in range(len(sub.columns)):
            val = sub.values[i, j]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=6, color=color)

    fig.colorbar(im, ax=ax, shrink=0.6, label="Median delta (CPM)")

    label = "increased" if direction == "positive" else "decreased"
    ax.set_title(f"Top {n} {label} genes by imputation (ranked by mean delta)", fontsize=12)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_ct_summary(matrix, output_path, ct_order=None):
    """Bar chart: mean positive and negative delta per cell type."""
    mean_pos = matrix.clip(lower=0).mean(axis=0)
    mean_neg = matrix.clip(upper=0).mean(axis=0)
    n_pos = (matrix > 0).sum(axis=0)
    n_neg = (matrix < 0).sum(axis=0)
    n_total = len(matrix)

    if ct_order is None:
        _, ct_Z = cluster_order(matrix, axis=1)
        ct_order = matrix.columns[leaves_list(ct_Z)]
    cts = ct_order

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [2, 1]})

    # Top panel: mean delta magnitude
    ax = axes[0]
    x = np.arange(len(cts))
    ax.bar(x, mean_pos[cts], color="firebrick", alpha=0.7, label="Mean positive delta")
    ax.bar(x, mean_neg[cts], color="steelblue", alpha=0.7, label="Mean negative delta")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(cts, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean delta (CPM)")
    ax.set_title("Imputation effect per cell type")
    ax.legend(fontsize=8)

    # Bottom panel: gene counts
    ax = axes[1]
    ax.bar(x, n_pos[cts], color="firebrick", alpha=0.5, label=f"Increased (of {n_total})")
    ax.bar(x, -n_neg[cts], color="steelblue", alpha=0.5, label=f"Decreased (of {n_total})")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(cts, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Number of genes")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved CT summary: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("delta_dir", help="Path to delta_analysis/ folder with tables/")
    parser.add_argument("--top_n", type=int, default=20, help="Number of top genes per direction")
    args = parser.parse_args()

    tables_dir = os.path.join(args.delta_dir, "tables")
    matrix = load_delta_matrix(tables_dir)
    print(f"Full matrix: {matrix.shape[0]} genes x {matrix.shape[1]} cell types")

    # Compute CT order once from full matrix, shared across all plots
    ct_order, _ = cluster_order(matrix, axis=1)
    print(f"CT order: {list(ct_order)}")

    plot_clustered_heatmap(matrix, os.path.join(args.delta_dir, "top_increased_heatmap.png"),
                           n=args.top_n, direction="positive", ct_order=ct_order)
    plot_clustered_heatmap(matrix, os.path.join(args.delta_dir, "top_decreased_heatmap.png"),
                           n=args.top_n, direction="negative", ct_order=ct_order)
    plot_ct_summary(matrix, os.path.join(args.delta_dir, "ct_summary.png"), ct_order=ct_order)

    # Also save the full matrix CSV
    csv_out = os.path.join(args.delta_dir, "full_delta_matrix.csv")
    matrix.to_csv(csv_out)
    print(f"Saved matrix: {csv_out}")


if __name__ == "__main__":
    main()
