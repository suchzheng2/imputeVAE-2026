"""Delta analysis: what does the model change during imputation?

For each cell type, computes per-gene delta = imputed_nucleus - raw_nucleus,
ranks genes by median delta, and generates tables + plots.

Usage:
    python analysis/delta_analysis.py <run_folder> [--output_dir <dir>]

Outputs (per cell type):
    - CSV table: gene, median_delta, mean_delta, frac_positive, n_cells
    - Top 20 positive / negative delta genes
    - Histogram of delta distributions for top genes
    - Cross-cell-type heatmap of top shifted genes
"""

import argparse
import gc
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imputeVAE.data import load_raw_data, filter_by_protocol
from types import SimpleNamespace


def load_model(run_folder, device):
    """Load model from run folder, return (model, args_dict)."""
    with open(f"{run_folder}/arguments.json") as f:
        args = json.load(f)

    is_cond = args.get("fine_tune_cond_stage3", False)
    scalar_cond = args.get("scalar_cond", False)
    cond_encoder = args.get("cond_encoder", False)
    embed_dim = args.get("embed_dim", 8)
    num_features = args["num_feature"]
    num_internal = args.get("num_internal", [1024, 512])
    num_latent = args.get("num_latent", 32)

    if is_cond:
        from imputeVAE.models_cond import VAE_JSD_Cond
        kwargs = dict(
            input_size=num_features,
            num_internal=num_internal,
            num_latent=num_latent,
            embed_dim=1 if scalar_cond else embed_dim,
            scalar_cond=scalar_cond,
        )
        if cond_encoder:
            kwargs["cond_encoder"] = True
        model = VAE_JSD_Cond(**kwargs)
    else:
        from imputeVAE.models import VAE_JSD
        model = VAE_JSD(
            input_size=num_features,
            num_internal=num_internal,
            num_latent=num_latent,
        )

    model.load_state_dict(torch.load(f"{run_folder}/models/model.pt", map_location=device))
    model.to(device)
    model.eval()
    return model, args


def cross_decode_nuclei(model, X_nuc, args, device, batch_size=256):
    """Cross-decode nuclei through the model to get imputed values."""
    is_cond = args.get("fine_tune_cond_stage3", False)
    CELL_ID, NUCLEI_ID = 0, 1

    imputed = []
    with torch.no_grad():
        for i in range(0, len(X_nuc), batch_size):
            batch = torch.tensor(X_nuc[i:i + batch_size], dtype=torch.float32, device=device)
            if is_cond:
                xd = model.decode_cross(batch, source_modality_id=NUCLEI_ID, target_modality_id=CELL_ID)
            else:
                encoded = model.encoder(batch)
                mu = model.mu(encoded)
                z_cond = mu  # deterministic
                decoded = model.decoder(z_cond)
                xd = decoded
            imputed.append(xd.cpu().numpy())

    return np.concatenate(imputed, axis=0)


def compute_delta_stats(X_nuc, X_imputed, cell_types, gene_names):
    """Compute per-gene delta stats for each cell type.

    Returns:
        dict[cell_type] -> DataFrame with columns:
            gene, median_delta, mean_delta, std_delta, frac_positive, n_cells
    """
    unique_cts = sorted(set(cell_types))
    results = {}

    for ct in unique_cts:
        mask = cell_types == ct
        n = mask.sum()
        if n < 5:
            continue

        delta = X_imputed[mask] - X_nuc[mask]  # [n_cells, n_genes]

        median_delta = np.median(delta, axis=0)
        mean_delta = np.mean(delta, axis=0)
        std_delta = np.std(delta, axis=0)
        frac_positive = (delta > 0).mean(axis=0)

        df = pd.DataFrame({
            "gene": gene_names,
            "median_delta": median_delta,
            "mean_delta": mean_delta,
            "std_delta": std_delta,
            "frac_positive": frac_positive,
            "n_cells": n,
        })
        df = df.sort_values("median_delta", ascending=False).reset_index(drop=True)
        results[ct] = df

    return results


def save_tables(results, output_dir):
    """Save per-CT delta tables and top gene summaries."""
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    for ct, df in results.items():
        ct_safe = ct.replace("/", "_")
        df.to_csv(os.path.join(tables_dir, f"{ct_safe}_delta.csv"), index=False)

        # Top 20 positive and negative
        top_pos = df.head(20)[["gene", "median_delta", "mean_delta", "frac_positive"]]
        top_neg = df.tail(20).iloc[::-1][["gene", "median_delta", "mean_delta", "frac_positive"]]

        with open(os.path.join(tables_dir, f"{ct_safe}_top_genes.txt"), "w") as f:
            f.write(f"=== {ct} (n={df['n_cells'].iloc[0]}) ===\n\n")
            f.write("Top 20 INCREASED by imputation (positive delta):\n")
            f.write(top_pos.to_string(index=False))
            f.write("\n\nTop 20 DECREASED by imputation (negative delta):\n")
            f.write(top_neg.to_string(index=False))
            f.write("\n")

    print(f"  Tables saved to {tables_dir}")


def plot_delta_histograms(results, output_dir, n_top=10):
    """Plot delta distribution histograms for top shifted genes per CT."""
    hist_dir = os.path.join(output_dir, "histograms")
    os.makedirs(hist_dir, exist_ok=True)

    # We need the raw delta matrix for histograms — store during compute
    # For now, plot from summary stats (median/mean)
    for ct, df in results.items():
        ct_safe = ct.replace("/", "_")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Top positive
        top_pos = df.head(n_top)
        axes[0].barh(range(n_top), top_pos["median_delta"].values, color="firebrick", alpha=0.7)
        axes[0].set_yticks(range(n_top))
        axes[0].set_yticklabels(top_pos["gene"].values, fontsize=8)
        axes[0].set_xlabel("Median delta (imputed - raw)")
        axes[0].set_title(f"{ct}: Top {n_top} increased")
        axes[0].invert_yaxis()

        # Top negative
        top_neg = df.tail(n_top).iloc[::-1]
        axes[1].barh(range(n_top), top_neg["median_delta"].values, color="steelblue", alpha=0.7)
        axes[1].set_yticks(range(n_top))
        axes[1].set_yticklabels(top_neg["gene"].values, fontsize=8)
        axes[1].set_xlabel("Median delta (imputed - raw)")
        axes[1].set_title(f"{ct}: Top {n_top} decreased")
        axes[1].invert_yaxis()

        plt.tight_layout()
        plt.savefig(os.path.join(hist_dir, f"{ct_safe}_top_genes.png"), dpi=150)
        plt.close()

    print(f"  Histograms saved to {hist_dir}")


def plot_global_delta_distribution(X_nuc, X_imputed, cell_types, output_dir):
    """Plot overall delta distribution across all cells."""
    delta = X_imputed - X_nuc
    median_per_gene = np.median(delta, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Per-gene median delta distribution
    axes[0].hist(median_per_gene, bins=100, color="gray", alpha=0.7, edgecolor="black", linewidth=0.3)
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Median delta per gene (all cells)")
    axes[0].set_ylabel("Number of genes")
    axes[0].set_title(f"Distribution of per-gene median delta\n"
                      f"(mean={median_per_gene.mean():.4f}, "
                      f"pos={np.sum(median_per_gene > 0)}, "
                      f"neg={np.sum(median_per_gene < 0)})")

    # Per-cell mean absolute delta
    mean_abs_delta_per_cell = np.mean(np.abs(delta), axis=1)
    axes[1].hist(mean_abs_delta_per_cell, bins=100, color="gray", alpha=0.7, edgecolor="black", linewidth=0.3)
    axes[1].set_xlabel("Mean |delta| per cell")
    axes[1].set_ylabel("Number of cells")
    axes[1].set_title("Per-cell magnitude of change")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "global_delta_distribution.png"), dpi=150)
    plt.close()
    print(f"  Global distribution plot saved")


def plot_cross_ct_heatmap(results, output_dir, n_top=15):
    """Heatmap of top shifted genes across all cell types."""
    # Collect top positive + negative genes across all CTs
    all_top_genes = set()
    for ct, df in results.items():
        all_top_genes.update(df.head(n_top)["gene"].values)
        all_top_genes.update(df.tail(n_top)["gene"].values)

    all_top_genes = sorted(all_top_genes)
    cts = sorted(results.keys())

    # Build matrix
    matrix = np.zeros((len(all_top_genes), len(cts)))
    for j, ct in enumerate(cts):
        df = results[ct].set_index("gene")
        for i, gene in enumerate(all_top_genes):
            if gene in df.index:
                matrix[i, j] = df.loc[gene, "median_delta"]

    fig, ax = plt.subplots(figsize=(max(10, len(cts) * 0.8), max(8, len(all_top_genes) * 0.25)))
    vmax = np.percentile(np.abs(matrix), 95)
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(cts)))
    ax.set_xticklabels(cts, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(all_top_genes)))
    ax.set_yticklabels(all_top_genes, fontsize=6)
    ax.set_title(f"Median delta (imputed - raw) for top shifted genes\n(top/bottom {n_top} per CT)")
    plt.colorbar(im, ax=ax, label="Median delta", shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cross_ct_delta_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Cross-CT heatmap saved")


def plot_consistency_summary(results, output_dir):
    """Plot fraction of genes with consistent direction (frac_positive > 0.8 or < 0.2)."""
    cts = sorted(results.keys())
    n_consistent_pos = []
    n_consistent_neg = []
    n_total = []

    for ct in cts:
        df = results[ct]
        n_total.append(len(df))
        n_consistent_pos.append((df["frac_positive"] > 0.8).sum())
        n_consistent_neg.append((df["frac_positive"] < 0.2).sum())

    fig, ax = plt.subplots(figsize=(max(8, len(cts) * 0.6), 5))
    x = range(len(cts))
    ax.bar(x, n_consistent_pos, label="Consistently increased (>80% cells)", color="firebrick", alpha=0.7)
    ax.bar(x, n_consistent_neg, bottom=n_consistent_pos, label="Consistently decreased (<20% cells)", color="steelblue", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(cts, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Number of genes")
    ax.set_title("Genes with consistent delta direction across cells within each CT")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "consistency_summary.png"), dpi=150)
    plt.close()
    print(f"  Consistency summary saved")


def main():
    parser = argparse.ArgumentParser(description="Delta analysis: what does the model change?")
    parser.add_argument("run_folder", type=str, help="Path to model run folder")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: <run_folder>/delta_analysis)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    run_folder = args.run_folder.rstrip("/")
    output_dir = args.output_dir or os.path.join(run_folder, "delta_analysis")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(args.device)

    # Load model and config
    print(f"Loading model from {run_folder}...")
    model, run_args = load_model(run_folder, device)
    norm = run_args.get("raw_data_normalization", "log1p")

    # Load nuclei data
    print("Loading nuclei data...")
    args_ns = SimpleNamespace(**run_args)
    adata_nuc = load_raw_data("10X_nuclei_v2_v3_AIBS/", args_ns)

    # Filter by protocol
    adata_nuc = filter_by_protocol(adata_nuc, args_ns)

    # Get expression matrix
    layer = norm if run_args.get("use_raw_data", False) and norm != "raw" else None
    if layer and layer in adata_nuc.layers:
        X_nuc = np.array(adata_nuc.layers[layer], dtype=np.float32)
    else:
        X_nuc = adata_nuc.X
        if hasattr(X_nuc, "toarray"):
            X_nuc = X_nuc.toarray()
        X_nuc = np.array(X_nuc, dtype=np.float32)

    cell_types = np.array(adata_nuc.obs["cell_type"].values)
    gene_names = np.array(adata_nuc.var["gene_names"].values)
    print(f"  Nuclei: {X_nuc.shape[0]} cells, {X_nuc.shape[1]} genes, {len(set(cell_types))} cell types")

    # Cross-decode
    print("Cross-decoding nuclei -> imputed cell expression...")
    X_imputed = cross_decode_nuclei(model, X_nuc, run_args, device)
    print(f"  Imputed shape: {X_imputed.shape}")

    del model
    gc.collect()

    # Compute delta stats
    print("Computing delta statistics per cell type...")
    results = compute_delta_stats(X_nuc, X_imputed, cell_types, gene_names)
    print(f"  Computed for {len(results)} cell types")

    # Save tables
    save_tables(results, output_dir)

    # Plots
    print("Generating plots...")
    plot_delta_histograms(results, output_dir)
    plot_global_delta_distribution(X_nuc, X_imputed, cell_types, output_dir)
    plot_cross_ct_heatmap(results, output_dir)
    plot_consistency_summary(results, output_dir)

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for ct in sorted(results.keys()):
        df = results[ct]
        n = df["n_cells"].iloc[0]
        top_gene = df.iloc[0]["gene"]
        top_delta = df.iloc[0]["median_delta"]
        bot_gene = df.iloc[-1]["gene"]
        bot_delta = df.iloc[-1]["median_delta"]
        n_consistent = ((df["frac_positive"] > 0.8) | (df["frac_positive"] < 0.2)).sum()
        print(f"  {ct:<15} n={n:>6}  top+: {top_gene:<12} ({top_delta:+.4f})  "
              f"top-: {bot_gene:<12} ({bot_delta:+.4f})  "
              f"consistent: {n_consistent}/{len(df)}")

    print(f"\nOutput: {output_dir}")


if __name__ == "__main__":
    main()
