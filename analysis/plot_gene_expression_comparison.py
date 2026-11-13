#!/usr/bin/env python3
"""Gene expression comparison plots: original nuclei vs imputed cell expression."""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
from types import SimpleNamespace

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imputeVAE.data import load_raw_data, filter_by_protocol, balanced_split_by_cell_type
from analysis.delta_analysis import load_model, cross_decode_nuclei

def plot_gene_expression_comparison(run_folder, output_dir=None, n_genes=20, cell_types=None):
    """Create gene expression comparison plots."""

    if output_dir is None:
        output_dir = Path("notes/experiments/gene_expression_comparison")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")

    # Load model and config
    print(f"Loading model from {run_folder}...")
    model, run_args = load_model(run_folder, device)
    norm = run_args.get("raw_data_normalization", "log1p")

    # Load nuclei data
    print("Loading nuclei data...")
    args_ns = SimpleNamespace(**run_args)
    adata_nuc = load_raw_data("10X_nuclei_v2_v3_AIBS/", args_ns)
    adata_nuc = filter_by_protocol(adata_nuc, args_ns)

    # Apply normalization
    raw_mode = run_args.get("use_raw_data", False) and norm == "raw"
    if not raw_mode:
        for layer_name in [norm, "log1p", "cpm", "raw"]:
            if layer_name in adata_nuc.layers:
                adata_nuc.X = adata_nuc.layers[layer_name]
                print(f"  Using layer: {layer_name}")
                break

    # Split data
    print("Splitting data...")
    (X_train, obs_train), (X_valid, obs_valid), (X_test, obs_test), var = \
        balanced_split_by_cell_type(adata_nuc, args_ns)

    # Use validation set for comparison
    X_nuc = np.array(X_valid, dtype=np.float32)
    cell_types_array = np.array(obs_valid["cell_type"].values)
    gene_names = np.array(var["gene_names"].values) if "gene_names" in var.columns else np.array(var.index)

    # Cross-decode
    print("Cross-decoding nuclei -> imputed cell expression...")
    X_imputed = cross_decode_nuclei(model, X_nuc, run_args, device)

    # Load delta analysis results to get top genes
    delta_dir = Path("notes/experiments/delta_analysis/tables")
    if not delta_dir.exists():
        print(f"Warning: Delta analysis results not found at {delta_dir}")
        return

    # Select cell types to plot
    if cell_types is None:
        # Get cell types with good sample sizes
        unique_cts, counts = np.unique(cell_types_array, return_counts=True)
        cell_types = [ct for ct, count in zip(unique_cts, counts) if count >= 30][:6]

    print(f"Creating plots for cell types: {cell_types}")

    # 1. Delta-driven gene expression comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, ct in enumerate(cell_types):
        if i >= len(axes):
            break

        ax = axes[i]

        # Load delta results for this cell type
        ct_safe = ct.replace("/", "_")
        delta_file = delta_dir / f"{ct_safe}_delta.csv"

        if not delta_file.exists():
            ax.text(0.5, 0.5, f"No delta data\nfor {ct}", ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(ct)
            continue

        delta_df = pd.read_csv(delta_file)

        # Get top increased and decreased genes
        top_inc = delta_df.head(n_genes//2)["gene"].values
        top_dec = delta_df.tail(n_genes//2)["gene"].values
        selected_genes = np.concatenate([top_inc, top_dec])

        # Get indices for this cell type and selected genes
        ct_mask = cell_types_array == ct
        gene_indices = [np.where(gene_names == g)[0][0] for g in selected_genes if g in gene_names]

        if len(gene_indices) == 0 or ct_mask.sum() == 0:
            ax.text(0.5, 0.5, f"No data\nfor {ct}", ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(ct)
            continue

        # Get expression data
        nuc_expr = X_nuc[ct_mask][:, gene_indices].mean(axis=0)
        imp_expr = X_imputed[ct_mask][:, gene_indices].mean(axis=0)

        # Plot scatter
        ax.scatter(nuc_expr, imp_expr, alpha=0.7, s=50)

        # Add diagonal line
        min_val = min(nuc_expr.min(), imp_expr.min())
        max_val = max(nuc_expr.max(), imp_expr.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)

        # Calculate correlation
        corr = np.corrcoef(nuc_expr, imp_expr)[0, 1]
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.set_xlabel('Original Nuclei Expression', fontsize=10)
        ax.set_ylabel('Imputed Cell Expression', fontsize=10)
        ax.set_title(f'{ct} (n={ct_mask.sum()})', fontsize=12)
        ax.grid(True, alpha=0.3)

    # Remove empty subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(output_dir / "gene_expression_scatter_by_celltype.png",
                dpi=300, bbox_inches='tight')
    plt.show()

    # 2. Top delta genes heatmap
    print("Creating delta genes heatmap...")

    # Collect top delta genes across all cell types
    all_top_genes = set()
    for ct in cell_types:
        ct_safe = ct.replace("/", "_")
        delta_file = delta_dir / f"{ct_safe}_delta.csv"
        if delta_file.exists():
            delta_df = pd.read_csv(delta_file)
            top_genes = delta_df.head(10)["gene"].values
            all_top_genes.update(top_genes)

    all_top_genes = list(all_top_genes)[:30]  # Limit to top 30 for visibility

    # Create heatmap data
    heatmap_data = []
    heatmap_labels = []

    for ct in cell_types:
        ct_mask = cell_types_array == ct
        if ct_mask.sum() == 0:
            continue

        gene_indices = [np.where(gene_names == g)[0][0] for g in all_top_genes if g in gene_names]

        if len(gene_indices) > 0:
            nuc_mean = X_nuc[ct_mask][:, gene_indices].mean(axis=0)
            imp_mean = X_imputed[ct_mask][:, gene_indices].mean(axis=0)
            delta_mean = imp_mean - nuc_mean

            heatmap_data.append(delta_mean)
            heatmap_labels.append(ct)

    if heatmap_data:
        heatmap_array = np.array(heatmap_data)
        gene_subset = [g for g in all_top_genes if g in gene_names][:heatmap_array.shape[1]]

        fig, ax = plt.subplots(figsize=(15, 8))
        sns.heatmap(heatmap_array,
                   xticklabels=gene_subset,
                   yticklabels=heatmap_labels,
                   cmap='RdBu_r', center=0,
                   cbar_kws={'label': 'Expression Delta (Imputed - Original)'},
                   ax=ax)

        ax.set_title('Gene Expression Deltas Across Cell Types\n(Top Delta Genes)',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Genes', fontsize=12)
        ax.set_ylabel('Cell Types', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / "delta_genes_heatmap.png",
                    dpi=300, bbox_inches='tight')
        plt.show()

    # 3. Individual gene expression distributions
    print("Creating gene distribution plots...")

    # Select a few interesting genes
    interesting_genes = ["Malat1", "Rbp1", "Rpl13", "Actb", "Gapdh"]  # Mix of nuclear lncRNA and cytoplasmic
    available_genes = [g for g in interesting_genes if g in gene_names][:6]

    if available_genes:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, gene in enumerate(available_genes):
            if i >= len(axes):
                break

            ax = axes[i]
            gene_idx = np.where(gene_names == gene)[0][0]

            # Collect data across all cell types
            nuc_values = []
            imp_values = []
            ct_labels = []

            for ct in cell_types:
                ct_mask = cell_types_array == ct
                if ct_mask.sum() >= 10:  # Minimum sample size
                    nuc_vals = X_nuc[ct_mask, gene_idx]
                    imp_vals = X_imputed[ct_mask, gene_idx]

                    nuc_values.extend(nuc_vals)
                    imp_values.extend(imp_vals)
                    ct_labels.extend([f"{ct}_nuc"] * len(nuc_vals))
                    ct_labels.extend([f"{ct}_imp"] * len(imp_vals))

            # Create violin plot
            plot_data = pd.DataFrame({
                'expression': nuc_values + imp_values,
                'condition': ct_labels
            })

            # Separate original and imputed
            plot_data['modality'] = plot_data['condition'].str.extract(r'_(\w+)$')[0]
            plot_data['cell_type'] = plot_data['condition'].str.replace(r'_(nuc|imp)$', '', regex=True)

            sns.violinplot(data=plot_data, x='cell_type', y='expression', hue='modality',
                          ax=ax, palette=['lightcoral', 'skyblue'])

            ax.set_title(f'{gene} Expression Distribution', fontsize=12, fontweight='bold')
            ax.set_xlabel('Cell Type', fontsize=10)
            ax.set_ylabel('Expression Level', fontsize=10)
            ax.tick_params(axis='x', rotation=45)

            # Add legend
            ax.legend(title='Modality', labels=['Imputed', 'Original'])

        # Remove empty subplots
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.savefig(output_dir / "gene_expression_distributions.png",
                    dpi=300, bbox_inches='tight')
        plt.show()

    print(f"\n✓ Gene expression comparison plots saved to: {output_dir}")
    print(f"✓ Created plots:")
    print(f"  - gene_expression_scatter_by_celltype.png")
    print(f"  - delta_genes_heatmap.png")
    print(f"  - gene_expression_distributions.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("run_folder", help="Path to model run folder")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    parser.add_argument("--n_genes", type=int, default=20, help="Number of top genes to plot")
    args = parser.parse_args()

    plot_gene_expression_comparison(args.run_folder, args.output_dir, args.n_genes)
