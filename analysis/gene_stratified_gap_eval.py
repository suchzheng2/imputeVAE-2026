"""
Task 6: Gene-stratified evaluation by nucleus-cell gap magnitude.

Tests whether the model helps most on genes with the largest cell-nucleus gap
and does not hurt genes with small gaps. Bins genes into quartiles by
|mean(real_cell) - mean(real_nuclei)| and computes per-bin improvement of
imputed vs raw-nuclei against real cell.

Usage:
    python analysis/gene_stratified_gap_eval.py \
        --imputed_npy run/cvae_full_eval/imputed_cell_from_nucleus.npy \
        --obs_csv run/cvae_full_eval/nucleus_obs.csv
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from scipy import sparse, stats
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
from analysis.evaluate_baseline import load_raw_data, DEFAULT_GENE_LIST

OUT_DIR = os.path.join(PROJECT, "notes/experiments/gene_stratified_eval")


def per_gene_mse(pred, target):
    """Per-gene MSE across all cells."""
    return np.mean((pred - target) ** 2, axis=0)


def per_gene_corr(X, Y):
    """Per-gene Pearson correlation (across cell-type means)."""
    corrs = np.zeros(X.shape[1])
    for g in range(X.shape[1]):
        if np.std(X[:, g]) > 1e-10 and np.std(Y[:, g]) > 1e-10:
            corrs[g], _ = stats.pearsonr(X[:, g], Y[:, g])
        else:
            corrs[g] = np.nan
    return corrs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imputed_npy", default="run/cvae_full_eval/imputed_cell_from_nucleus.npy")
    parser.add_argument("--obs_csv", default="run/cvae_full_eval/nucleus_obs.csv")
    parser.add_argument("--gene_list", default=DEFAULT_GENE_LIST)
    parser.add_argument("--n_bins", type=int, default=4, help="Number of gap bins (default: 4 = quartiles)")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # Load data
    print("Loading imputed matrix...")
    imputed = np.load(os.path.join(PROJECT, args.imputed_npy)).astype(np.float32)
    obs = pd.read_csv(os.path.join(PROJECT, args.obs_csv), index_col=0)
    ct_imputed = obs["cell_type"].values.astype(str)

    print("Loading raw data...")
    adata_cell, adata_nuc = load_raw_data(args.gene_list)
    X_cell = adata_cell.X if not sparse.issparse(adata_cell.X) else adata_cell.X.toarray()
    X_nuc = adata_nuc.X if not sparse.issparse(adata_nuc.X) else adata_nuc.X.toarray()
    ct_cell = adata_cell.obs["cell_type"].values.astype(str)
    ct_nuc = adata_nuc.obs["cell_type"].values.astype(str)
    gene_names = adata_cell.var["gene_names"].values
    n_genes = X_cell.shape[1]

    common_cts = sorted(set(ct_cell) & set(ct_nuc) & set(ct_imputed))
    print(f"Common CTs: {len(common_cts)}, Genes: {n_genes}")

    # =====================================================================
    # 1. Compute ground-truth per-gene gap (global, across all cells)
    # =====================================================================
    cell_global_mean = X_cell.mean(axis=0)
    nuc_global_mean = X_nuc.mean(axis=0)
    gt_gap = np.abs(cell_global_mean - nuc_global_mean)
    gt_delta = cell_global_mean - nuc_global_mean  # signed

    # =====================================================================
    # 2. Bin genes into quartiles by gap
    # =====================================================================
    n_bins = args.n_bins
    bin_edges = np.quantile(gt_gap, np.linspace(0, 1, n_bins + 1))
    bin_edges[-1] += 1e-6  # include max
    gene_bins = np.digitize(gt_gap, bin_edges) - 1
    gene_bins = np.clip(gene_bins, 0, n_bins - 1)

    bin_labels = []
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        bin_labels.append(f"Q{b+1} [{lo:.3f}, {hi:.3f})")

    print(f"\nGap quartile edges: {[f'{e:.3f}' for e in bin_edges]}")
    for b in range(n_bins):
        n_in = (gene_bins == b).sum()
        print(f"  {bin_labels[b]}: {n_in} genes")

    # =====================================================================
    # 3. Compute per-gene metrics: MSE and CT-mean correlation
    # =====================================================================

    # --- 3a. Per-gene MSE (all cells) ---
    # For imputed vs real cell: need to match against cell means per CT
    # since imputed and cell have different cells.
    # Use CT-mean-level comparison instead.

    # CT means
    cell_ct_means = {}
    nuc_ct_means = {}
    imp_ct_means = {}
    for ct in common_cts:
        cell_ct_means[ct] = X_cell[ct_cell == ct].mean(axis=0)
        nuc_ct_means[ct] = X_nuc[ct_nuc == ct].mean(axis=0)
        imp_ct_means[ct] = imputed[ct_imputed == ct].mean(axis=0)

    cell_mean_mat = np.array([cell_ct_means[ct] for ct in common_cts])
    nuc_mean_mat = np.array([nuc_ct_means[ct] for ct in common_cts])
    imp_mean_mat = np.array([imp_ct_means[ct] for ct in common_cts])

    # Per-gene MSE at CT-mean level
    mse_imp = np.mean((imp_mean_mat - cell_mean_mat) ** 2, axis=0)
    mse_nuc = np.mean((nuc_mean_mat - cell_mean_mat) ** 2, axis=0)
    mse_improvement = mse_nuc - mse_imp  # positive = imputed is better

    # Per-gene correlation (across CT means)
    corr_imp = per_gene_corr(imp_mean_mat, cell_mean_mat)
    corr_nuc = per_gene_corr(nuc_mean_mat, cell_mean_mat)
    corr_improvement = corr_imp - corr_nuc

    # Per-gene absolute error of delta
    pred_delta = imputed.mean(axis=0) - X_nuc.mean(axis=0)
    delta_error = np.abs(pred_delta - gt_delta)

    # =====================================================================
    # 4. Aggregate by bin
    # =====================================================================
    bin_results = []
    for b in range(n_bins):
        mask = gene_bins == b
        n_genes_bin = mask.sum()
        result = {
            "bin": bin_labels[b],
            "n_genes": int(n_genes_bin),
            "mean_gap": float(gt_gap[mask].mean()),
            "median_gap": float(np.median(gt_gap[mask])),
            # MSE
            "mse_imputed": float(mse_imp[mask].mean()),
            "mse_raw_nuc": float(mse_nuc[mask].mean()),
            "mse_improvement": float(mse_improvement[mask].mean()),
            "mse_improvement_pct": float(
                100 * mse_improvement[mask].mean() / (mse_nuc[mask].mean() + 1e-12)
            ),
            # Correlation
            "corr_imputed_median": float(np.nanmedian(corr_imp[mask])),
            "corr_raw_nuc_median": float(np.nanmedian(corr_nuc[mask])),
            "corr_improvement_median": float(np.nanmedian(corr_improvement[mask])),
            # Delta accuracy
            "delta_error_mean": float(delta_error[mask].mean()),
            "delta_error_median": float(np.median(delta_error[mask])),
        }
        bin_results.append(result)
        print(f"\n{bin_labels[b]} ({n_genes_bin} genes):")
        print(f"  Mean gap: {result['mean_gap']:.4f}")
        print(f"  MSE imputed: {result['mse_imputed']:.6f}  raw_nuc: {result['mse_raw_nuc']:.6f}  "
              f"improvement: {result['mse_improvement_pct']:+.1f}%")
        print(f"  Corr imputed: {result['corr_imputed_median']:.4f}  raw_nuc: {result['corr_raw_nuc_median']:.4f}  "
              f"improvement: {result['corr_improvement_median']:+.4f}")

    # =====================================================================
    # 5. Save results
    # =====================================================================
    results_df = pd.DataFrame(bin_results)
    results_df.to_csv(os.path.join(OUT_DIR, "gene_stratified_results.csv"), index=False)

    # Per-gene detail
    gene_df = pd.DataFrame({
        "gene": gene_names,
        "gt_gap": gt_gap,
        "gt_delta": gt_delta,
        "pred_delta": pred_delta,
        "bin": [bin_labels[b] for b in gene_bins],
        "mse_imputed": mse_imp,
        "mse_raw_nuc": mse_nuc,
        "mse_improvement": mse_improvement,
        "corr_imputed": corr_imp,
        "corr_raw_nuc": corr_nuc,
        "corr_improvement": corr_improvement,
    })
    gene_df.to_csv(os.path.join(OUT_DIR, "per_gene_detail.csv"), index=False)

    # Stats JSON
    stats_out = {"n_bins": n_bins, "bin_edges": bin_edges.tolist(), "bins": bin_results}
    with open(os.path.join(OUT_DIR, "stratified_stats.json"), "w") as f:
        json.dump(stats_out, f, indent=2)
    print(f"\nSaved stats to {OUT_DIR}/stratified_stats.json")

    # =====================================================================
    # 6. Figures
    # =====================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Gene-Stratified Evaluation by Cell-Nucleus Gap", fontsize=14)
    x = range(n_bins)

    # 6a: MSE improvement by bin
    ax = axes[0, 0]
    pcts = [r["mse_improvement_pct"] for r in bin_results]
    colors = ["#2ecc71" if p > 0 else "#e74c3c" for p in pcts]
    ax.bar(x, pcts, color=colors, edgecolor="black")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{i+1}" for i in x])
    ax.set_ylabel("MSE improvement (%)")
    ax.set_title("MSE improvement (imputed vs raw nuclei)")
    ax.set_xlabel("Gap quartile (Q1=smallest, Q4=largest)")

    # 6b: Correlation by bin
    ax = axes[0, 1]
    ax.bar([i - 0.15 for i in x], [r["corr_raw_nuc_median"] for r in bin_results],
           width=0.3, label="Raw nuclei", color="#3498db", edgecolor="black")
    ax.bar([i + 0.15 for i in x], [r["corr_imputed_median"] for r in bin_results],
           width=0.3, label="Imputed", color="#e67e22", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{i+1}" for i in x])
    ax.set_ylabel("Median gene correlation")
    ax.set_title("Gene correlation (vs real cell CT means)")
    ax.set_xlabel("Gap quartile")
    ax.legend()

    # 6c: MSE raw vs imputed by bin
    ax = axes[1, 0]
    ax.bar([i - 0.15 for i in x], [r["mse_raw_nuc"] for r in bin_results],
           width=0.3, label="Raw nuclei", color="#3498db", edgecolor="black")
    ax.bar([i + 0.15 for i in x], [r["mse_imputed"] for r in bin_results],
           width=0.3, label="Imputed", color="#e67e22", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{i+1}" for i in x])
    ax.set_ylabel("Mean MSE (vs real cell)")
    ax.set_title("MSE by gap quartile")
    ax.set_xlabel("Gap quartile")
    ax.legend()

    # 6d: Gap distribution
    ax = axes[1, 1]
    for b in range(n_bins):
        mask = gene_bins == b
        ax.hist(gt_gap[mask], bins=30, alpha=0.5, label=f"Q{b+1}")
    ax.set_xlabel("|mean(cell) - mean(nuclei)| (log1p)")
    ax.set_ylabel("# genes")
    ax.set_title("Gap distribution by quartile")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "gene_stratified_eval.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {OUT_DIR}/gene_stratified_eval.png")

    # =====================================================================
    # 7. Summary
    # =====================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Bin':<25} {'N':>6} {'Gap':>8} {'MSE impr%':>10} {'Corr imp':>10} {'Corr nuc':>10}")
    for r in bin_results:
        print(f"{r['bin']:<25} {r['n_genes']:>6} {r['mean_gap']:>8.4f} "
              f"{r['mse_improvement_pct']:>+10.1f} {r['corr_imputed_median']:>10.4f} "
              f"{r['corr_raw_nuc_median']:>10.4f}")


if __name__ == "__main__":
    main()
