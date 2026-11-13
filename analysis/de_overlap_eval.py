"""DE gene overlap evaluation — standalone, normalization-aware.

Computes one-vs-rest Wilcoxon rank-sum DE on raw cells, raw nuclei, and
cross-decoded (imputed) nuclei, then reports top-K overlap per cell type.

Usage:
    python analysis/de_overlap_eval.py <run_folder> [--output_dir <dir>] [--device cuda] [--top_k 100]
"""

import argparse
import gc
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import ranksums

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.delta_analysis import load_model, cross_decode_nuclei
from imputeVAE.data import load_raw_data, filter_by_protocol
from types import SimpleNamespace


def run_de_one_vs_rest(X, cell_types, top_k=100):
    """One-vs-rest Wilcoxon rank-sum DE. Returns dict: ct -> array of top-K gene indices."""
    de_genes = {}
    for ct in sorted(set(cell_types)):
        mask = cell_types == ct
        if mask.sum() < 5 or (~mask).sum() < 5:
            continue
        pvals = []
        fold_changes = []
        for g in range(X.shape[1]):
            in_group = X[mask, g]
            out_group = X[~mask, g]
            _, p = ranksums(in_group, out_group)
            fc = in_group.mean() - out_group.mean()
            pvals.append(p)
            fold_changes.append(fc)
        pvals = np.array(pvals)
        fold_changes = np.array(fold_changes)
        order = np.lexsort((-np.abs(fold_changes), pvals))
        de_genes[ct] = order[:top_k]
    return de_genes


def main():
    parser = argparse.ArgumentParser(description="DE overlap evaluation")
    parser.add_argument("run_folder", type=str)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--top_k", type=int, default=100)
    args = parser.parse_args()

    run_folder = args.run_folder.rstrip("/")
    output_dir = args.output_dir or os.path.join(run_folder, "de_overlap")
    os.makedirs(output_dir, exist_ok=True)
    TOP_K = args.top_k

    import torch
    device = torch.device(args.device)

    # Load model config
    with open(f"{run_folder}/arguments.json") as f:
        run_args = json.load(f)

    norm = run_args.get("raw_data_normalization", "log1p")
    use_raw = run_args.get("use_raw_data", False)
    args_ns = SimpleNamespace(**run_args)

    # Load data — cell batch is c3A (not n3A which is nuclei)
    print("Loading cell data...")
    cell_args_ns = SimpleNamespace(**{**run_args, "batch": "c3A"})
    adata_cell = load_raw_data("10X_cells_v2_v3_AIBS/", cell_args_ns)
    adata_cell = filter_by_protocol(adata_cell, cell_args_ns)
    print(f"  Cells: {adata_cell.shape}")

    print("Loading nuclei data...")
    adata_nuc = load_raw_data("10X_nuclei_v2_v3_AIBS/", args_ns)
    adata_nuc = filter_by_protocol(adata_nuc, args_ns)
    print(f"  Nuclei: {adata_nuc.shape}")

    # Get expression matrices
    layer = norm if use_raw and norm and norm != "raw" else None
    if layer and layer in adata_cell.layers:
        X_cell = np.array(adata_cell.layers[layer], dtype=np.float32)
        X_nuc = np.array(adata_nuc.layers[layer], dtype=np.float32)
    else:
        X_cell = adata_cell.X
        X_nuc = adata_nuc.X
        if hasattr(X_cell, "toarray"):
            X_cell = X_cell.toarray()
        if hasattr(X_nuc, "toarray"):
            X_nuc = X_nuc.toarray()
        X_cell = np.array(X_cell, dtype=np.float32)
        X_nuc = np.array(X_nuc, dtype=np.float32)

    ct_cell = np.array(adata_cell.obs["cell_type"].values)
    ct_nuc = np.array(adata_nuc.obs["cell_type"].values)
    gene_names = np.array(adata_nuc.var["gene_names"].values)
    common_cts = sorted(set(ct_cell) & set(ct_nuc))

    del adata_cell, adata_nuc
    gc.collect()

    # Cross-decode nuclei
    print("Loading model and cross-decoding...")
    model, _ = load_model(run_folder, device)
    X_imputed = cross_decode_nuclei(model, X_nuc, run_args, device)
    del model
    gc.collect()
    print(f"  Imputed: {X_imputed.shape}")

    # Run DE
    print(f"\nRunning one-vs-rest DE (top-{TOP_K})...")
    print("  DE on raw cells...")
    de_raw_cell = run_de_one_vs_rest(X_cell, ct_cell, top_k=TOP_K)
    print("  DE on raw nuclei...")
    de_raw_nuc = run_de_one_vs_rest(X_nuc, ct_nuc, top_k=TOP_K)
    print("  DE on imputed nuclei...")
    de_imputed = run_de_one_vs_rest(X_imputed, ct_nuc, top_k=TOP_K)

    # Compute overlaps
    rows = []
    for ct in common_cts:
        imp_ov = 0
        raw_ov = 0
        if ct in de_raw_cell and ct in de_imputed:
            imp_ov = len(set(de_raw_cell[ct]) & set(de_imputed[ct])) / TOP_K
        if ct in de_raw_cell and ct in de_raw_nuc:
            raw_ov = len(set(de_raw_cell[ct]) & set(de_raw_nuc[ct])) / TOP_K
        rows.append({
            "cell_type": ct,
            "de_overlap_imputed": imp_ov,
            "de_overlap_raw_nuc": raw_ov,
            "improvement": imp_ov - raw_ov,
        })

    df = pd.DataFrame(rows)
    mean_imp = df["de_overlap_imputed"].mean()
    mean_raw = df["de_overlap_raw_nuc"].mean()

    # Print results
    print(f"\n{'='*65}")
    print(f"Top-{TOP_K} DE overlap with raw cell DE")
    print(f"{'='*65}")
    print(f"{'Cell Type':<20} {'Imputed':>10} {'Raw Nuc':>10} {'Delta':>10}")
    print("-" * 55)
    for _, row in df.iterrows():
        print(f"{row['cell_type']:<20} {row['de_overlap_imputed']:>10.3f} "
              f"{row['de_overlap_raw_nuc']:>10.3f} {row['improvement']:>+10.3f}")
    print("-" * 55)
    print(f"{'MEAN':<20} {mean_imp:>10.3f} {mean_raw:>10.3f} {mean_imp - mean_raw:>+10.3f}")

    # Save
    df.to_csv(os.path.join(output_dir, "de_overlap_per_ct.csv"), index=False)

    summary = {
        "run_folder": run_folder,
        "normalization": norm,
        "top_k": TOP_K,
        "mean_de_overlap_imputed": float(mean_imp),
        "mean_de_overlap_raw_nuc": float(mean_raw),
        "improvement": float(mean_imp - mean_raw),
    }
    with open(os.path.join(output_dir, "de_overlap_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Save top DE genes per CT for imputed (for inspection)
    de_dir = os.path.join(output_dir, "de_genes")
    os.makedirs(de_dir, exist_ok=True)
    for ct in common_cts:
        if ct in de_imputed and ct in de_raw_cell:
            ct_safe = ct.replace("/", "_")
            imp_genes = gene_names[de_imputed[ct]]
            cell_genes = gene_names[de_raw_cell[ct]]
            shared = set(imp_genes) & set(cell_genes)
            with open(os.path.join(de_dir, f"{ct_safe}.txt"), "w") as f:
                f.write(f"# {ct}: top-{TOP_K} DE genes\n")
                f.write(f"# Overlap: {len(shared)}/{TOP_K}\n\n")
                f.write("Shared DE genes:\n")
                for g in sorted(shared):
                    f.write(f"  {g}\n")

    print(f"\nOutput saved to: {output_dir}")


if __name__ == "__main__":
    main()
