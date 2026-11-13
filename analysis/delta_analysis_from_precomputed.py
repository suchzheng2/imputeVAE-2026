"""Delta analysis using pre-computed imputed matrix (no model inference needed).

Reuses imputed_cell_from_nucleus.npy + nucleus_obs.csv from export_cvae_imputed.py,
loads the raw nuclei data for the input side, then runs the same delta analysis
as delta_analysis.py.

Usage:
    python analysis/delta_analysis_from_precomputed.py \
        --imputed run/cvae_full_eval/imputed_cell_from_nucleus.npy \
        --nucleus_obs run/cvae_full_eval/nucleus_obs.csv \
        --run_folder run/run_2026-07-18-19-40-51_570984 \
        --output_dir notes/experiments/delta_analysis_full
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imputeVAE.data import load_raw_data, filter_by_protocol
from types import SimpleNamespace

# Reuse plotting/table functions from delta_analysis.py
from analysis.delta_analysis import (
    compute_delta_stats,
    save_tables,
    plot_delta_histograms,
    plot_global_delta_distribution,
    plot_cross_ct_heatmap,
    plot_consistency_summary,
)


def save_all_genes_delta(X_nuc, X_imputed, gene_names, output_dir):
    """Save global all-genes delta table (same format as nuclear_lncrna_analysis/all_genes_delta.csv)."""
    cell_mean = np.mean(X_imputed, axis=0)
    nuc_mean = np.mean(X_nuc, axis=0)
    delta = cell_mean - nuc_mean

    df = pd.DataFrame({
        "gene": gene_names,
        "cell_mean": np.round(cell_mean, 5),
        "nuc_mean": np.round(nuc_mean, 5),
        "delta": np.round(delta, 5),
        "abs_delta": np.round(np.abs(delta), 5),
    })
    df = df.sort_values("delta", ascending=True).reset_index(drop=True)
    path = os.path.join(output_dir, "all_genes_delta.csv")
    df.to_csv(path, index=False)
    print(f"  All-genes delta saved to {path}")
    return df


def save_top_gene_lists(df, output_dir, n=100):
    """Save top increased/decreased gene lists."""
    top_inc = df.sort_values("delta", ascending=False).head(n)
    top_dec = df.sort_values("delta", ascending=True).head(n)

    with open(os.path.join(output_dir, f"top{n}_increased_genes.txt"), "w") as f:
        for g in top_inc["gene"]:
            f.write(g + "\n")
    with open(os.path.join(output_dir, f"top{n}_decreased_genes.txt"), "w") as f:
        for g in top_dec["gene"]:
            f.write(g + "\n")
    print(f"  Top-{n} gene lists saved")


def main():
    parser = argparse.ArgumentParser(description="Delta analysis from pre-computed imputed matrix")
    parser.add_argument("--imputed", required=True, help="Path to imputed_cell_from_nucleus.npy")
    parser.add_argument("--nucleus_obs", required=True, help="Path to nucleus_obs.csv")
    parser.add_argument("--run_folder", required=True, help="Path to model run folder (for arguments.json)")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load run args for data loading config
    with open(f"{args.run_folder}/arguments.json") as f:
        run_args = json.load(f)

    # Load imputed matrix
    print("Loading pre-computed imputed matrix...")
    X_imputed = np.load(args.imputed).astype(np.float32)
    print(f"  Imputed shape: {X_imputed.shape}")

    # Load nucleus obs
    nucleus_obs = pd.read_csv(args.nucleus_obs, index_col=0)
    cell_types = np.array(nucleus_obs["cell_type"].values)
    print(f"  Cell types: {len(set(cell_types))} unique")

    # Load raw nuclei data (same pipeline as training)
    print("Loading raw nuclei data...")
    args_ns = SimpleNamespace(**run_args)
    adata_nuc = load_raw_data("10X_nuclei_v2_v3_AIBS/", args_ns)
    adata_nuc = filter_by_protocol(adata_nuc, args_ns)

    norm = run_args.get("raw_data_normalization", "log1p")
    layer = norm if run_args.get("use_raw_data", False) and norm != "raw" else None
    if layer and layer in adata_nuc.layers:
        X_nuc = np.array(adata_nuc.layers[layer], dtype=np.float32)
    else:
        X_nuc = adata_nuc.X
        if hasattr(X_nuc, "toarray"):
            X_nuc = X_nuc.toarray()
        X_nuc = np.array(X_nuc, dtype=np.float32)

    gene_names = np.array(adata_nuc.var["gene_names"].values)
    print(f"  Raw nuclei shape: {X_nuc.shape}")

    # Verify shapes match
    assert X_nuc.shape == X_imputed.shape, f"Shape mismatch: nuc {X_nuc.shape} vs imputed {X_imputed.shape}"
    assert len(cell_types) == X_nuc.shape[0], f"Cell type count mismatch: {len(cell_types)} vs {X_nuc.shape[0]}"

    # Compute per-CT delta stats
    print("Computing delta statistics per cell type...")
    results = compute_delta_stats(X_nuc, X_imputed, cell_types, gene_names)
    print(f"  Computed for {len(results)} cell types")

    # Save tables
    save_tables(results, args.output_dir)

    # Save all-genes delta
    delta_df = save_all_genes_delta(X_nuc, X_imputed, gene_names, args.output_dir)
    save_top_gene_lists(delta_df, args.output_dir)

    # Plots
    print("Generating plots...")
    plot_delta_histograms(results, args.output_dir)
    plot_global_delta_distribution(X_nuc, X_imputed, cell_types, args.output_dir)
    plot_cross_ct_heatmap(results, args.output_dir)
    plot_consistency_summary(results, args.output_dir)

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

    print(f"\nOutput: {args.output_dir}")


if __name__ == "__main__":
    main()
