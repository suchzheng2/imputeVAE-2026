"""
Task 5: Diagnose L5 ET anomaly — delta R² = -0.560 despite r = 0.950.

High correlation + negative R² implies a systematic scale/offset mismatch:
the predicted delta is correlated with ground truth but shifted or scaled
such that residuals exceed total variance.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats, sparse
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT, "notes/experiments/l5et_diagnosis")


def load_and_filter(data_path, batch, gene_list_path):
    raw_dp = os.path.join(
        "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi",
        data_path, "data.h5ad",
    )
    adata = ad.read_h5ad(raw_dp)
    if "meta_batch_id" not in adata.obs:
        adata.obs["meta_batch_id"] = adata.obs["batch_id"].apply(lambda x: x.split(".")[0])
    adata = adata[adata.obs["meta_batch_id"].str.endswith(batch)].copy()

    common_genes = pd.read_csv(os.path.join(PROJECT, "data/common_genes.csv"), header=0)["common_genes"].tolist()
    adata = adata[:, adata.var["gene_names"].isin(common_genes)].copy()
    common_ct = pd.read_csv(os.path.join(PROJECT, "data/common_cell_types.csv"), header=0)["common_cell_types"].tolist()
    adata = adata[adata.obs["cell_type"].isin(common_ct)].copy()

    gene_list = pd.read_csv(gene_list_path, header=None)[0].tolist()
    adata = adata[:, adata.var["gene_names"].isin(gene_list)].copy()
    adata = adata[:, adata.var["gene_names"].argsort()].copy()

    X = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
    adata.X = np.log1p(X).astype(np.float32)
    return adata


def compute_ct_mean(X, ct_labels, target_ct):
    mask = ct_labels == target_ct
    return X[mask].mean(axis=0) if mask.sum() > 0 else None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    gene_list = os.path.join(PROJECT, "data/training_gene_list_v3.csv")

    print("Loading data...")
    cell_adata = load_and_filter("10X_cells_v2_v3_AIBS/", "c3A", gene_list)
    nuc_adata = load_and_filter("10X_nuclei_v2_v3_AIBS/", "n3A", gene_list)
    gene_names = cell_adata.var["gene_names"].values

    X_cell = cell_adata.X if isinstance(cell_adata.X, np.ndarray) else cell_adata.X.toarray()
    X_nuc = nuc_adata.X if isinstance(nuc_adata.X, np.ndarray) else nuc_adata.X.toarray()
    ct_cell = cell_adata.obs["cell_type"].values
    ct_nuc = nuc_adata.obs["cell_type"].values

    # Load imputed
    imputed = np.load(os.path.join(PROJECT, "run/cvae_full_eval/imputed_cell_from_nucleus.npy"))
    obs = pd.read_csv(os.path.join(PROJECT, "run/cvae_full_eval/nucleus_obs.csv"), index_col=0)
    ct_imp = obs["cell_type"].values

    # Analyze L5 ET + a healthy control (L2/3 IT)
    targets = ["L5 ET", "L2/3 IT", "Endo", "OPC"]

    results = {}
    for ct in targets:
        print(f"\n{'='*60}")
        print(f"Cell type: {ct}")
        print(f"{'='*60}")

        n_cell = (ct_cell == ct).sum()
        n_nuc = (ct_nuc == ct).sum()
        n_imp = (ct_imp == ct).sum()
        print(f"  n_cell={n_cell}, n_nuc={n_nuc}, n_imputed={n_imp}")

        # Ground truth delta = cell_mean - nuc_mean
        cell_mean = compute_ct_mean(X_cell, ct_cell, ct)
        nuc_mean = compute_ct_mean(X_nuc, ct_nuc, ct)
        imp_mean = compute_ct_mean(imputed, ct_imp, ct)

        if cell_mean is None or nuc_mean is None or imp_mean is None:
            print(f"  Skipping — missing data")
            continue

        gt_delta = cell_mean - nuc_mean
        pred_delta = imp_mean - nuc_mean

        # Correlation and R²
        r, p = stats.pearsonr(gt_delta, pred_delta)
        ss_res = np.sum((pred_delta - gt_delta) ** 2)
        ss_tot = np.sum((gt_delta - gt_delta.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Slope/intercept of pred vs gt
        slope, intercept, _, _, _ = stats.linregress(gt_delta, pred_delta)

        # Mean absolute error
        mae = np.mean(np.abs(pred_delta - gt_delta))

        # Scale ratio
        pred_std = np.std(pred_delta)
        gt_std = np.std(gt_delta)
        scale_ratio = pred_std / gt_std if gt_std > 0 else float('inf')

        # Mean offset
        mean_offset = np.mean(pred_delta) - np.mean(gt_delta)

        print(f"  Correlation r = {r:.4f}")
        print(f"  R² = {r2:.4f}")
        print(f"  Slope (pred vs gt) = {slope:.4f}")
        print(f"  Intercept = {intercept:.4f}")
        print(f"  MAE = {mae:.4f}")
        print(f"  Scale ratio (pred_std/gt_std) = {scale_ratio:.4f}")
        print(f"  Mean offset (pred - gt) = {mean_offset:.4f}")
        print(f"  GT delta: mean={gt_delta.mean():.4f}, std={gt_std:.4f}")
        print(f"  Pred delta: mean={pred_delta.mean():.4f}, std={pred_std:.4f}")

        # Check for high-leverage genes
        residuals = pred_delta - gt_delta
        top_resid_idx = np.argsort(np.abs(residuals))[-10:][::-1]
        print(f"\n  Top 10 high-residual genes:")
        print(f"  {'Gene':<15} {'GT delta':>10} {'Pred delta':>12} {'Residual':>10}")
        for idx in top_resid_idx:
            print(f"  {gene_names[idx]:<15} {gt_delta[idx]:>10.4f} {pred_delta[idx]:>12.4f} {residuals[idx]:>10.4f}")

        results[ct] = {
            "n_cell": int(n_cell),
            "n_nuc": int(n_nuc),
            "r": float(r),
            "r2": float(r2),
            "slope": float(slope),
            "intercept": float(intercept),
            "mae": float(mae),
            "scale_ratio": float(scale_ratio),
            "mean_offset": float(mean_offset),
            "gt_delta_mean": float(gt_delta.mean()),
            "gt_delta_std": float(gt_std),
            "pred_delta_mean": float(pred_delta.mean()),
            "pred_delta_std": float(pred_std),
        }

    # Save results
    with open(os.path.join(OUT_DIR, "diagnosis_stats.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ── Diagnostic plots ───────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for i, ct in enumerate(targets):
        ax = axes[i // 2, i % 2]

        cell_mean = compute_ct_mean(X_cell, ct_cell, ct)
        nuc_mean = compute_ct_mean(X_nuc, ct_nuc, ct)
        imp_mean = compute_ct_mean(imputed, ct_imp, ct)

        if cell_mean is None or nuc_mean is None or imp_mean is None:
            ax.set_title(f"{ct} — no data")
            continue

        gt_delta = cell_mean - nuc_mean
        pred_delta = imp_mean - nuc_mean
        r = results[ct]["r"]
        r2 = results[ct]["r2"]
        n_cell = results[ct]["n_cell"]
        n_nuc = results[ct]["n_nuc"]

        ax.scatter(gt_delta, pred_delta, s=1, alpha=0.3, color="steelblue")

        # Identity line
        lims = [min(gt_delta.min(), pred_delta.min()), max(gt_delta.max(), pred_delta.max())]
        ax.plot(lims, lims, "r--", linewidth=1, label="identity")

        # Regression line
        slope = results[ct]["slope"]
        intercept = results[ct]["intercept"]
        x_fit = np.linspace(lims[0], lims[1], 100)
        ax.plot(x_fit, slope * x_fit + intercept, "g-", linewidth=1,
                label=f"fit: y={slope:.2f}x+{intercept:.2f}")

        ax.set_xlabel("Ground truth delta (cell - nuc)")
        ax.set_ylabel("Predicted delta (imputed - nuc)")
        ax.set_title(f"{ct}  (n_cell={n_cell}, n_nuc={n_nuc})\nr={r:.3f}, R²={r2:.3f}")
        ax.legend(fontsize=8)
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "delta_scatter_comparison.png"), dpi=150)
    plt.close()
    print(f"\nPlot saved to {OUT_DIR}/delta_scatter_comparison.png")

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'CT':<12} {'n_cell':>7} {'n_nuc':>7} {'r':>7} {'R²':>7} {'slope':>7} {'offset':>8} {'scale':>7}")
    for ct in targets:
        if ct in results:
            r = results[ct]
            print(f"{ct:<12} {r['n_cell']:>7} {r['n_nuc']:>7} {r['r']:>7.3f} {r['r2']:>7.3f} "
                  f"{r['slope']:>7.3f} {r['mean_offset']:>8.3f} {r['scale_ratio']:>7.3f}")


if __name__ == "__main__":
    main()
