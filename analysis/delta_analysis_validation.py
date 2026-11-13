"""Delta analysis on validation set only.

Reuses core functions from delta_analysis.py but loads only the validation
split of nuclei data (using the same random_state=42 split as training).

Usage:
    python analysis/delta_analysis_validation.py <run_folder> [--output_dir <dir>] [--device cpu]
"""

import argparse
import gc
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imputeVAE.data import load_raw_data, filter_by_protocol, balanced_split_by_cell_type
from types import SimpleNamespace
from analysis.delta_analysis import (
    load_model, cross_decode_nuclei, compute_delta_stats,
    save_tables, plot_delta_histograms, plot_global_delta_distribution,
    plot_cross_ct_heatmap, plot_consistency_summary,
)


def main():
    parser = argparse.ArgumentParser(description="Delta analysis on validation set")
    parser.add_argument("run_folder", type=str)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--split", type=str, default="valid", choices=["train", "valid", "test"])
    args = parser.parse_args()

    run_folder = args.run_folder.rstrip("/")
    output_dir = args.output_dir or os.path.join(
        "notes/experiments/delta_analysis",
        f"{args.split}_set"
    )
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Load model and config
    print(f"Loading model from {run_folder}...")
    model, run_args = load_model(run_folder, device)
    norm = run_args.get("raw_data_normalization", "log1p")

    # Load nuclei data and apply same split as training
    print("Loading nuclei data...")
    args_ns = SimpleNamespace(**run_args)
    adata_nuc = load_raw_data("10X_nuclei_v2_v3_AIBS/", args_ns)
    adata_nuc = filter_by_protocol(adata_nuc, args_ns)

    # Swap adata.X with the normalized layer (same as create_dataloaders_by_cell_type)
    raw_mode = run_args.get("use_raw_data", False) and norm == "raw"
    if not raw_mode:
        for layer_name in [norm, "log1p", "cpm", "raw"]:
            if layer_name in adata_nuc.layers:
                adata_nuc.X = adata_nuc.layers[layer_name]
                print(f"  Using layer: {layer_name}")
                break

    print(f"Splitting data (split={args.split})...")
    (X_train, obs_train), (X_valid, obs_valid), (X_test, obs_test), var = \
        balanced_split_by_cell_type(adata_nuc, args_ns)

    split_map = {"train": (X_train, obs_train), "valid": (X_valid, obs_valid), "test": (X_test, obs_test)}
    X_nuc, obs_split = split_map[args.split]
    X_nuc = np.array(X_nuc, dtype=np.float32)

    cell_types = np.array(obs_split["cell_type"].values)
    gene_names = np.array(var["gene_names"].values) if "gene_names" in var.columns else np.array(var.index)

    print(f"  {args.split} set: {X_nuc.shape[0]} cells, {X_nuc.shape[1]} genes, "
          f"{len(set(cell_types))} cell types")

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

    # Compare with training set results if available
    train_dir = "notes/experiments/delta_analysis/tables"
    if os.path.exists(train_dir) and args.split != "train":
        print("\n--- Comparison with full-data delta analysis ---")
        comparison_rows = []
        for ct, df_val in results.items():
            ct_safe = ct.replace("/", "_")
            train_file = os.path.join(train_dir, f"{ct_safe}_delta.csv")
            if os.path.exists(train_file):
                df_train = pd.read_csv(train_file)
                # Compare top 20 genes
                top20_train = set(df_train.head(20)["gene"].values)
                top20_val = set(df_val.head(20)["gene"].values)
                overlap = len(top20_train & top20_val)
                # Correlation of median deltas
                merged = pd.merge(
                    df_train[["gene", "median_delta"]].rename(columns={"median_delta": "train_delta"}),
                    df_val[["gene", "median_delta"]].rename(columns={"median_delta": "valid_delta"}),
                    on="gene"
                )
                corr = merged["train_delta"].corr(merged["valid_delta"])
                comparison_rows.append({
                    "cell_type": ct,
                    "n_valid": df_val["n_cells"].iloc[0],
                    "top20_overlap": overlap,
                    "delta_correlation": corr,
                })

        if comparison_rows:
            comp_df = pd.DataFrame(comparison_rows)
            print(comp_df.to_string(index=False))
            comp_df.to_csv(os.path.join(output_dir, "train_vs_valid_comparison.csv"), index=False)
            print(f"\nMean top-20 overlap: {comp_df['top20_overlap'].mean():.1f}/20")
            print(f"Mean delta correlation: {comp_df['delta_correlation'].mean():.4f}")

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"SUMMARY ({args.split} set)")
    print(f"{'=' * 60}")
    for ct in sorted(results.keys()):
        df = results[ct]
        n = df["n_cells"].iloc[0]
        top_gene = df.iloc[0]["gene"]
        top_delta = df.iloc[0]["median_delta"]
        bot_gene = df.iloc[-1]["gene"]
        bot_delta = df.iloc[-1]["median_delta"]
        print(f"  {ct:<15} n={n:>5}  top+: {top_gene:<12} ({top_delta:+.4f})  "
              f"top-: {bot_gene:<12} ({bot_delta:+.4f})")

    print(f"\nOutput: {output_dir}")


if __name__ == "__main__":
    main()
