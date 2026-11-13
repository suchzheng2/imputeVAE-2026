#!/usr/bin/env python3
"""
Generate qualifying-exam Figures 3, 4, 5 for ImputeVAE report.
Memory-optimized: processes one method at a time, frees arrays immediately.

All analyses use full-data predictions (no validation-only arrays available).
See figure_data_audit.txt for details.
"""

import os
import sys
import json
import gc
import numpy as np
import pandas as pd
from scipy import stats, sparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

OUT_DIR = Path("figures/qualexam_2026-08-06")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NEURONAL_CTS = [
    "L2/3 IT", "L5 ET", "L5 IT", "L5/6 NP", "L6 CT", "L6 IT", "L6b",
    "Lamp5", "Pvalb", "Sncg", "Sst", "Vip",
]
NON_NEURONAL_CTS = ["Astro", "Endo", "Macrophage", "OPC", "Oligo"]
ALL_CTS = sorted(NEURONAL_CTS + NON_NEURONAL_CTS)

METHOD_COLORS = {
    "ImputeVAE": "#0072B2",
    "Raw nuclei": "#999999",
    "CT-mean": "#E69F00",
    "Linear reg": "#56B4E9",
    "scVI": "#009E73",
    "scANVI": "#F0E442",
    "scGen": "#D55E00",
    "scButterfly": "#CC79A7",
}

CT_NEURON_COLOR = "#0072B2"
CT_NONNEURON_COLOR = "#009E73"

def ct_color(ct):
    return CT_NEURON_COLOR if ct in NEURONAL_CTS else CT_NONNEURON_COLOR

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
})

PRED_BASE = {
    "ImputeVAE": "run/cvae_full_eval",
    "CT-mean": "run/ctmean_baseline",
    "Linear reg": "run/linreg_baseline",
    "scVI": "run/scvi_baseline",
    "scANVI": "run/scanvi_baseline",
    "scGen": "run/scgen_baseline",
    "scButterfly": "run/scbutterfly_baseline",
}

EVAL_JSONS = {
    "ImputeVAE": "run/cvae_full_eval/eval/eval_cvae_full_v2.json",
    "CT-mean": "run/ctmean_baseline/eval/eval_ctmean_v2.json",
    "Linear reg": "run/linreg_baseline/eval/eval_linreg.json",
    "scVI": "run/scvi_baseline/eval/eval_scvi_v2.json",
    "scANVI": "run/scanvi_baseline/eval/eval_scanvi.json",
    "scGen": "run/scgen_baseline/eval/eval_scgen.json",
    "scButterfly": "run/scbutterfly_baseline/eval/eval_scbutterfly.json",
}


def load_eval_json(method_key):
    with open(EVAL_JSONS[method_key]) as f:
        return json.load(f)


def load_raw_data_lean():
    """Load raw data, compute CT means and per-gene variances, then free arrays.
    Returns ct_means_cell, ct_means_nuc, ct_var_cell, ct_nuc_labels, gene_names, ct_counts."""
    import anndata as ad
    DATA_ROOT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi"
    gene_list = pd.read_csv("data/training_gene_list_v3.csv", header=None)[0].tolist()
    common_genes = pd.read_csv("data/common_genes.csv", header=0)['common_genes'].tolist()
    common_ct = pd.read_csv("data/common_cell_types.csv", header=0)['common_cell_types'].tolist()

    ct_means = {}
    ct_vars = {}
    ct_counts_dict = {}
    ct_labels_nuc = None
    gene_names_out = None

    for name, data_dir, batch in [("cell", "10X_cells_v2_v3_AIBS", "c3A"),
                                   ("nucleus", "10X_nuclei_v2_v3_AIBS", "n3A")]:
        path = os.path.join(DATA_ROOT, data_dir, "data.h5ad")
        print(f"  Loading {name}...")
        adata = ad.read_h5ad(path)
        if 'meta_batch_id' not in adata.obs:
            adata.obs['meta_batch_id'] = adata.obs['batch_id'].apply(lambda x: x.split('.')[0])
        row_mask = (adata.obs['meta_batch_id'].str.endswith(batch) &
                    adata.obs['cell_type'].isin(common_ct))
        col_mask = (adata.var['gene_names'].isin(common_genes) &
                    adata.var['gene_names'].isin(gene_list))
        adata = adata[row_mask, col_mask].copy()
        adata = adata[:, adata.var['gene_names'].argsort()].copy()

        X = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
        X = np.log1p(X).astype(np.float32)
        ct_arr = adata.obs['cell_type'].values.astype(str)

        if gene_names_out is None:
            gene_names_out = adata.var['gene_names'].values if 'gene_names' in adata.var else adata.var_names.values

        print(f"    {name}: {X.shape[0]} obs, {X.shape[1]} genes")

        # Compute per-CT means and variances
        means = {}
        vars_dict = {}
        counts = {}
        for ct in ALL_CTS:
            mask = ct_arr == ct
            n = mask.sum()
            counts[ct] = int(n)
            if n >= 2:
                means[ct] = X[mask].mean(axis=0)
                vars_dict[ct] = X[mask].var(axis=0)  # per-gene variance

        ct_means[name] = means
        ct_vars[name] = vars_dict
        ct_counts_dict[name] = counts
        if name == "nucleus":
            ct_labels_nuc = ct_arr.copy()

        del X, adata
        gc.collect()

    return ct_means, ct_vars, ct_counts_dict, ct_labels_nuc, gene_names_out


def load_pred_ct_means(method_key, ct_labels_nuc):
    """Load a prediction array, compute CT means, free array."""
    path = f"{PRED_BASE[method_key]}/imputed_cell_from_nucleus.npy"
    obs_path = f"{PRED_BASE[method_key]}/nucleus_obs.csv"
    print(f"  Loading prediction: {method_key}...")
    imp = np.load(path).astype(np.float32)

    if os.path.exists(obs_path):
        obs = pd.read_csv(obs_path, index_col=0)
        ct_imp = obs['cell_type'].values.astype(str)
    else:
        ct_imp = ct_labels_nuc

    means = {}
    for ct in ALL_CTS:
        mask = ct_imp == ct
        if mask.sum() > 0:
            means[ct] = imp[mask].mean(axis=0)

    del imp; gc.collect()
    return means, ct_imp


def load_pred_ct_means_and_vars(method_key, ct_labels_nuc):
    """Load a prediction array, compute CT means AND per-gene variances, free array."""
    path = f"{PRED_BASE[method_key]}/imputed_cell_from_nucleus.npy"
    obs_path = f"{PRED_BASE[method_key]}/nucleus_obs.csv"
    print(f"  Loading prediction: {method_key}...")
    imp = np.load(path).astype(np.float32)

    if os.path.exists(obs_path):
        obs = pd.read_csv(obs_path, index_col=0)
        ct_imp = obs['cell_type'].values.astype(str)
    else:
        ct_imp = ct_labels_nuc

    means = {}
    vars_dict = {}
    for ct in ALL_CTS:
        mask = ct_imp == ct
        n = mask.sum()
        if n >= 2:
            means[ct] = imp[mask].mean(axis=0)
            vars_dict[ct] = imp[mask].var(axis=0)

    del imp; gc.collect()
    return means, vars_dict, ct_imp


# ---------------------------------------------------------------------------
# FIGURE 3
# ---------------------------------------------------------------------------

def figure3(ct_means_cell, ct_means_nuc, gene_names):
    print("\n" + "="*60)
    print("FIGURE 3: Directed cross-decoding")
    print("="*60)

    # Panel A: Method comparison from eval JSONs
    methods_ordered = [
        "ImputeVAE", "CT-mean", "Linear reg",
        "scVI", "scANVI", "scGen", "scButterfly", "Raw nuclei",
    ]

    metrics_data = []
    for m in methods_ordered:
        row = {"method": m}
        if m == "Raw nuclei":
            ref = load_eval_json("ImputeVAE")
            row["gene_corr_median"] = ref["gene_corr_raw_nuc"]["median"]
            row["r_squared_median"] = np.nan
            row["wasserstein_median"] = np.nan
        else:
            ev = load_eval_json(m)
            row["gene_corr_median"] = ev["gene_corr"]["median"]
            row["r_squared_median"] = ev["r_squared"]["median"]
            row["wasserstein_median"] = ev.get("wasserstein", {}).get("median", np.nan)
        metrics_data.append(row)
    df_metrics = pd.DataFrame(metrics_data)

    # Panel B: Load ImputeVAE per-CT deltas (compute from means)
    print("  Loading ImputeVAE predictions for per-CT deltas...")
    imp_path = f"{PRED_BASE['ImputeVAE']}/imputed_cell_from_nucleus.npy"
    obs_path = f"{PRED_BASE['ImputeVAE']}/nucleus_obs.csv"
    imp = np.load(imp_path).astype(np.float32)
    obs = pd.read_csv(obs_path, index_col=0)
    ct_imp = obs['cell_type'].values.astype(str)

    imp_means = {}
    for ct in ALL_CTS:
        mask = ct_imp == ct
        if mask.sum() > 0:
            imp_means[ct] = imp[mask].mean(axis=0)
    del imp; gc.collect()

    n_genes = len(gene_names)
    empirical_delta = np.zeros((n_genes, len(ALL_CTS)))
    predicted_delta = np.zeros((n_genes, len(ALL_CTS)))
    for j, ct in enumerate(ALL_CTS):
        if ct in ct_means_cell and ct in ct_means_nuc and ct in imp_means:
            empirical_delta[:, j] = ct_means_cell[ct] - ct_means_nuc[ct]
            predicted_delta[:, j] = imp_means[ct] - ct_means_nuc[ct]

    delta_df_emp = pd.DataFrame(empirical_delta, index=gene_names, columns=ALL_CTS)
    delta_df_pred = pd.DataFrame(predicted_delta, index=gene_names, columns=ALL_CTS)

    emp_flat = empirical_delta.ravel()
    pred_flat = predicted_delta.ravel()
    valid = np.isfinite(emp_flat) & np.isfinite(pred_flat)
    emp_flat = emp_flat[valid]
    pred_flat = pred_flat[valid]

    r_pearson, _ = stats.pearsonr(emp_flat, pred_flat)
    r_spearman, _ = stats.spearmanr(emp_flat, pred_flat)
    ss_res = np.sum((pred_flat - emp_flat)**2)
    ss_tot = np.sum((emp_flat - emp_flat.mean())**2)
    r2 = 1 - ss_res / ss_tot
    n_points = len(emp_flat)

    # Panel C: Direction classification
    pos_mask = emp_flat > 0
    neg_mask = emp_flat < 0
    zero_mask = emp_flat == 0
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    n_zero = int(zero_mask.sum())

    pred_pos = pred_flat > 0
    tp = int((pred_pos & pos_mask).sum())
    tn = int((~pred_pos & neg_mask).sum())
    fp = int((pred_pos & neg_mask).sum())
    fn = int((~pred_pos & pos_mask).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    balanced_acc = (sensitivity + specificity) / 2
    mcc_num = tp * tn - fp * fn
    mcc_den = np.sqrt(float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    mcc = mcc_num / mcc_den if mcc_den > 0 else 0

    from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
    mask_nonzero = ~zero_mask
    y_true = pos_mask[mask_nonzero].astype(int)
    y_score = pred_flat[mask_nonzero]
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auroc = roc_auc_score(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)

    print(f"  Panel B: r={r_pearson:.4f}, rho={r_spearman:.4f}, R²={r2:.4f}, n={n_points}")
    print(f"  Panel C: AUROC={auroc:.4f}, PR-AUC={pr_auc:.4f}, BA={balanced_acc:.4f}, MCC={mcc:.4f}")

    # ============== CREATE FIGURE ==============
    fig = plt.figure(figsize=(11, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
    ax_a1 = fig.add_subplot(gs[0, 0])
    ax_a2 = fig.add_subplot(gs[0, 1])
    ax_a3 = fig.add_subplot(gs[0, 2])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c1 = fig.add_subplot(gs[1, 1])
    ax_c2 = fig.add_subplot(gs[1, 2])

    methods_plot = methods_ordered
    y_positions = np.arange(len(methods_plot))

    # A1: Gene correlation
    for i, m in enumerate(methods_plot):
        v = df_metrics.set_index('method').loc[m, 'gene_corr_median']
        if np.isfinite(v):
            ax_a1.scatter(v, i, color=METHOD_COLORS[m], s=60, zorder=3, edgecolors='black', linewidth=0.5)
        else:
            ax_a1.annotate('NA', (0.5, i), ha='center', fontsize=7, color='gray')
    ax_a1.set_yticks(y_positions)
    ax_a1.set_yticklabels(methods_plot)
    ax_a1.set_xlabel("Median gene correlation")
    ax_a1.set_title("Gene correlation")
    ax_a1.invert_yaxis()
    ax_a1.set_xlim(0, 1.05)
    ax_a1.text(-0.15, 1.05, "A", transform=ax_a1.transAxes, fontsize=14, fontweight='bold', va='bottom')

    # A2: R-squared
    for i, m in enumerate(methods_plot):
        v = df_metrics.set_index('method').loc[m, 'r_squared_median']
        if np.isfinite(v):
            ax_a2.scatter(v, i, color=METHOD_COLORS[m], s=60, zorder=3, edgecolors='black', linewidth=0.5)
        else:
            ax_a2.annotate('NA', (0.0, i), ha='center', fontsize=7, color='gray')
    ax_a2.set_yticks(y_positions)
    ax_a2.set_yticklabels([])
    ax_a2.set_xlabel("Median R²")
    ax_a2.set_title("R² (CT-mean level)")
    ax_a2.invert_yaxis()

    # A3: Wasserstein
    for i, m in enumerate(methods_plot):
        v = df_metrics.set_index('method').loc[m, 'wasserstein_median']
        if np.isfinite(v):
            ax_a3.scatter(v, i, color=METHOD_COLORS[m], s=60, zorder=3, edgecolors='black', linewidth=0.5)
        else:
            ax_a3.annotate('NA', (0.5, i), ha='center', fontsize=7, color='gray')
    ax_a3.set_yticks(y_positions)
    ax_a3.set_yticklabels([])
    ax_a3.set_xlabel("Wasserstein-1 distance")
    ax_a3.set_title("Wasserstein (lower = better)")
    ax_a3.invert_yaxis()

    # Panel B: Hexbin
    lim = min(max(abs(emp_flat.min()), abs(emp_flat.max()), abs(pred_flat.min()), abs(pred_flat.max())), 8)
    hb = ax_b.hexbin(emp_flat, pred_flat, gridsize=80, cmap='viridis',
                      mincnt=1, extent=[-lim, lim, -lim, lim], norm=LogNorm())
    ax_b.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax_b.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    ax_b.plot([-lim, lim], [-lim, lim], 'r-', linewidth=0.8, alpha=0.7)
    ax_b.set_xlim(-lim, lim); ax_b.set_ylim(-lim, lim)
    ax_b.set_aspect('equal')
    ax_b.set_xlabel("Empirical shift (cell − nucleus)")
    ax_b.set_ylabel("ImputeVAE predicted shift")
    stats_text = f"Pearson r = {r_pearson:.3f}\nSpearman ρ = {r_spearman:.3f}\nR² = {r2:.3f}\nn = {n_points:,}"
    ax_b.text(0.03, 0.97, stats_text, transform=ax_b.transAxes, fontsize=7,
              va='top', fontfamily='monospace',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    cb = plt.colorbar(hb, ax=ax_b, shrink=0.7)
    cb.set_label("Count", fontsize=7)
    ax_b.text(-0.15, 1.05, "B", transform=ax_b.transAxes, fontsize=14, fontweight='bold', va='bottom')

    # Panel C: ROC + PR
    ax_c1.plot(fpr, tpr, color=METHOD_COLORS["ImputeVAE"], linewidth=1.5)
    ax_c1.plot([0, 1], [0, 1], 'k--', linewidth=0.5)
    ax_c1.set_xlabel("False positive rate"); ax_c1.set_ylabel("True positive rate")
    ax_c1.set_title(f"ROC (AUROC = {auroc:.3f})")
    ax_c1.set_xlim(0, 1); ax_c1.set_ylim(0, 1); ax_c1.set_aspect('equal')
    ax_c1.text(-0.15, 1.05, "C", transform=ax_c1.transAxes, fontsize=14, fontweight='bold', va='bottom')

    ax_c2.plot(recall, precision, color=METHOD_COLORS["ImputeVAE"], linewidth=1.5)
    ax_c2.set_xlabel("Recall"); ax_c2.set_ylabel("Precision")
    ax_c2.set_title(f"PR (AUC = {pr_auc:.3f})")
    ax_c2.set_xlim(0, 1); ax_c2.set_ylim(0, 1); ax_c2.set_aspect('equal')
    cls_text = (f"Sensitivity: {sensitivity:.3f}\nSpecificity: {specificity:.3f}\n"
                f"Balanced acc: {balanced_acc:.3f}\nMCC: {mcc:.3f}\n"
                f"Pos: {n_pos:,} ({100*n_pos/(n_pos+n_neg):.1f}%)\n"
                f"Neg: {n_neg:,} ({100*n_neg/(n_pos+n_neg):.1f}%)")
    ax_c2.text(0.03, 0.03, cls_text, transform=ax_c2.transAxes, fontsize=6.5,
               va='bottom', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    fig.suptitle("Figure 3: Directed cross-decoding recovers the empirical nucleus–cell shift",
                 fontsize=12, fontweight='bold', y=0.98)

    for fmt in ['pdf', 'svg', 'png']:
        fig.savefig(OUT_DIR / f"figure3_directed_reconstruction.{fmt}", bbox_inches='tight', dpi=300)
    plt.close(fig)

    # Save data
    data_rows = []
    for _, row in df_metrics.iterrows():
        data_rows.append({'figure': 'Fig3A', 'method': row['method'],
                          'gene_corr_median': row['gene_corr_median'],
                          'r_squared_median': row['r_squared_median'],
                          'wasserstein_median': row['wasserstein_median'],
                          'partition': 'full_data'})
    data_rows.append({'figure': 'Fig3B_summary', 'method': 'ImputeVAE',
                      'pearson_r': r_pearson, 'spearman_rho': r_spearman,
                      'r_squared': r2, 'n_points': n_points, 'partition': 'full_data'})
    data_rows.append({'figure': 'Fig3C_summary', 'method': 'ImputeVAE',
                      'AUROC': auroc, 'PR_AUC': pr_auc, 'accuracy': accuracy,
                      'sensitivity': sensitivity, 'specificity': specificity,
                      'balanced_accuracy': balanced_acc, 'MCC': mcc,
                      'n_positive': n_pos, 'n_negative': n_neg, 'n_zero': n_zero,
                      'pct_positive': 100*n_pos/(n_pos+n_neg), 'partition': 'full_data'})
    pd.DataFrame(data_rows).to_csv(OUT_DIR / "figure3_directed_reconstruction_data.csv", index=False)
    delta_df_emp.to_csv(OUT_DIR / "figure3_empirical_delta_matrix.csv")
    delta_df_pred.to_csv(OUT_DIR / "figure3_predicted_delta_matrix.csv")

    print("  Figure 3 DONE")


# ---------------------------------------------------------------------------
# FIGURE 4
# ---------------------------------------------------------------------------

def figure4(ct_means_cell, ct_means_nuc, ct_labels_nuc, ct_counts):
    print("\n" + "="*60)
    print("FIGURE 4: Cell-type specificity")
    print("="*60)

    # Compute D_empirical
    D_emp = {}
    for ct in ALL_CTS:
        if ct in ct_means_cell and ct in ct_means_nuc:
            D_emp[ct] = float(np.mean(np.abs(ct_means_cell[ct] - ct_means_nuc[ct])))

    # Load ImputeVAE means and compute D_predicted
    imp_means, _ = load_pred_ct_means("ImputeVAE", ct_labels_nuc)
    D_pred = {}
    for ct in ALL_CTS:
        if ct in imp_means and ct in ct_means_nuc:
            D_pred[ct] = float(np.mean(np.abs(imp_means[ct] - ct_means_nuc[ct])))

    # Bootstrap CIs using pre-computed means (approximate: resample nuc predictions only)
    # For accurate bootstrap, load full array. Do it in chunks to save memory.
    print("  Bootstrap CIs (500 iterations)...")
    n_boot = 500
    rng = np.random.RandomState(42)

    # Load ImputeVAE predictions for bootstrap
    imp = np.load(f"{PRED_BASE['ImputeVAE']}/imputed_cell_from_nucleus.npy").astype(np.float32)
    obs = pd.read_csv(f"{PRED_BASE['ImputeVAE']}/nucleus_obs.csv", index_col=0)
    ct_imp = obs['cell_type'].values.astype(str)

    # We also need raw cell and nuc arrays for bootstrap. Load raw nuc from cache.
    # Actually, we don't have raw arrays — we freed them. Load nucleus array only (smaller).
    import anndata as ad
    DATA_ROOT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi"
    gene_list = pd.read_csv("data/training_gene_list_v3.csv", header=None)[0].tolist()
    common_genes = pd.read_csv("data/common_genes.csv", header=0)['common_genes'].tolist()
    common_ct = pd.read_csv("data/common_cell_types.csv", header=0)['common_cell_types'].tolist()

    # Load ONLY nucleus raw data for bootstrap
    print("  Loading nucleus data for bootstrap...")
    adata = ad.read_h5ad(os.path.join(DATA_ROOT, "10X_nuclei_v2_v3_AIBS", "data.h5ad"))
    if 'meta_batch_id' not in adata.obs:
        adata.obs['meta_batch_id'] = adata.obs['batch_id'].apply(lambda x: x.split('.')[0])
    row_mask = (adata.obs['meta_batch_id'].str.endswith("n3A") & adata.obs['cell_type'].isin(common_ct))
    col_mask = (adata.var['gene_names'].isin(common_genes) & adata.var['gene_names'].isin(gene_list))
    adata = adata[row_mask, col_mask].copy()
    adata = adata[:, adata.var['gene_names'].argsort()].copy()
    X_nuc = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
    X_nuc = np.log1p(X_nuc).astype(np.float32)
    ct_nuc = adata.obs['cell_type'].values.astype(str)
    del adata; gc.collect()

    # Load cell data for bootstrap
    print("  Loading cell data for bootstrap...")
    adata = ad.read_h5ad(os.path.join(DATA_ROOT, "10X_cells_v2_v3_AIBS", "data.h5ad"))
    if 'meta_batch_id' not in adata.obs:
        adata.obs['meta_batch_id'] = adata.obs['batch_id'].apply(lambda x: x.split('.')[0])
    row_mask = (adata.obs['meta_batch_id'].str.endswith("c3A") & adata.obs['cell_type'].isin(common_ct))
    col_mask = (adata.var['gene_names'].isin(common_genes) & adata.var['gene_names'].isin(gene_list))
    adata = adata[row_mask, col_mask].copy()
    adata = adata[:, adata.var['gene_names'].argsort()].copy()
    X_cell = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
    X_cell = np.log1p(X_cell).astype(np.float32)
    ct_cell = adata.obs['cell_type'].values.astype(str)
    del adata; gc.collect()

    D_emp_boot = {ct: [] for ct in ALL_CTS}
    D_pred_boot = {ct: [] for ct in ALL_CTS}

    for b in range(n_boot):
        if b % 100 == 0:
            print(f"    Bootstrap {b}/{n_boot}")
        for ct in ALL_CTS:
            cell_idx = np.where(ct_cell == ct)[0]
            nuc_idx = np.where(ct_nuc == ct)[0]
            imp_idx = np.where(ct_imp == ct)[0]

            if len(cell_idx) < 2 or len(nuc_idx) < 2:
                D_emp_boot[ct].append(np.nan)
                D_pred_boot[ct].append(np.nan)
                continue

            ci = rng.choice(cell_idx, len(cell_idx), replace=True)
            ni = rng.choice(nuc_idx, len(nuc_idx), replace=True)
            ii = rng.choice(imp_idx, len(imp_idx), replace=True)

            cm = X_cell[ci].mean(axis=0)
            nm = X_nuc[ni].mean(axis=0)
            im = imp[ii].mean(axis=0)

            D_emp_boot[ct].append(float(np.mean(np.abs(cm - nm))))
            D_pred_boot[ct].append(float(np.mean(np.abs(im - nm))))

    del imp, X_cell, X_nuc; gc.collect()

    D_emp_ci = {}
    D_pred_ci = {}
    for ct in ALL_CTS:
        arr_e = np.array(D_emp_boot[ct])
        arr_p = np.array(D_pred_boot[ct])
        if np.all(np.isnan(arr_e)):
            D_emp_ci[ct] = (np.nan, np.nan)
            D_pred_ci[ct] = (np.nan, np.nan)
        else:
            D_emp_ci[ct] = (float(np.nanpercentile(arr_e, 2.5)), float(np.nanpercentile(arr_e, 97.5)))
            D_pred_ci[ct] = (float(np.nanpercentile(arr_p, 2.5)), float(np.nanpercentile(arr_p, 97.5)))

    cts_both = [ct for ct in ALL_CTS if ct in D_emp and ct in D_pred]
    x_vals = np.array([D_emp[ct] for ct in cts_both])
    y_vals = np.array([D_pred[ct] for ct in cts_both])
    r_p, _ = stats.pearsonr(x_vals, y_vals)
    r_s, _ = stats.spearmanr(x_vals, y_vals)
    ss_res = np.sum((y_vals - x_vals)**2)
    ss_tot = np.sum((x_vals - x_vals.mean())**2)
    r2 = 1 - ss_res / ss_tot

    neuron_emp = np.mean([D_emp[ct] for ct in NEURONAL_CTS if ct in D_emp])
    nonneuron_emp = np.mean([D_emp[ct] for ct in NON_NEURONAL_CTS if ct in D_emp])
    neuron_pred = np.mean([D_pred[ct] for ct in NEURONAL_CTS if ct in D_pred])
    nonneuron_pred = np.mean([D_pred[ct] for ct in NON_NEURONAL_CTS if ct in D_pred])
    ratio_emp = neuron_emp / nonneuron_emp if nonneuron_emp > 0 else np.nan
    ratio_pred = neuron_pred / nonneuron_pred if nonneuron_pred > 0 else np.nan

    print(f"  Pearson r={r_p:.4f}, Spearman rho={r_s:.4f}, R²={r2:.4f}")

    # Panel B: scGen/scButterfly comparison
    method_D = {"ImputeVAE": D_pred}
    method_corrs = {"ImputeVAE": {"pearson": r_p, "spearman": r_s, "r2": r2,
                                   "mae": float(np.mean(np.abs(y_vals - x_vals)))}}

    for mname in ["scGen", "scButterfly"]:
        m_means, _ = load_pred_ct_means(mname, ct_labels_nuc)
        D_m = {}
        for ct in ALL_CTS:
            if ct in m_means and ct in ct_means_nuc:
                D_m[ct] = float(np.mean(np.abs(m_means[ct] - ct_means_nuc[ct])))
        method_D[mname] = D_m

        cts_m = [ct for ct in ALL_CTS if ct in D_emp and ct in D_m]
        x_m = np.array([D_emp[ct] for ct in cts_m])
        y_m = np.array([D_m[ct] for ct in cts_m])
        if len(x_m) > 2:
            rp_m, _ = stats.pearsonr(x_m, y_m)
            rs_m, _ = stats.spearmanr(x_m, y_m)
            ss_r = np.sum((y_m - x_m)**2)
            ss_t = np.sum((x_m - x_m.mean())**2)
            r2_m = 1 - ss_r / ss_t
            mae_m = float(np.mean(np.abs(y_m - x_m)))
            method_corrs[mname] = {"pearson": rp_m, "spearman": rs_m, "r2": r2_m, "mae": mae_m}

    # ============== CREATE FIGURE ==============
    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # Panel A
    ax_a = fig.add_subplot(gs[0, 0])
    lim = max(max(x_vals), max(y_vals)) * 1.15
    ax_a.plot([0, lim], [0, lim], 'k--', linewidth=0.5, alpha=0.5)

    for ct in cts_both:
        c = ct_color(ct)
        ax_a.scatter(D_emp[ct], D_pred[ct], color=c, s=50, zorder=3, edgecolors='black', linewidth=0.3)
        if not np.isnan(D_emp_ci[ct][0]):
            xerr_lo = max(D_emp[ct] - D_emp_ci[ct][0], 0)
            xerr_hi = max(D_emp_ci[ct][1] - D_emp[ct], 0)
            yerr_lo = max(D_pred[ct] - D_pred_ci[ct][0], 0)
            yerr_hi = max(D_pred_ci[ct][1] - D_pred[ct], 0)
            ax_a.errorbar(D_emp[ct], D_pred[ct],
                         xerr=[[xerr_lo], [xerr_hi]],
                         yerr=[[yerr_lo], [yerr_hi]],
                         fmt='none', color=c, alpha=0.3, linewidth=0.5)
        ax_a.annotate(ct, (D_emp[ct], D_pred[ct]), fontsize=6, xytext=(3, 3), textcoords='offset points')

    ax_a.set_xlim(0, lim); ax_a.set_ylim(0, lim); ax_a.set_aspect('equal')
    ax_a.set_xlabel("Empirical correction magnitude")
    ax_a.set_ylabel("ImputeVAE predicted magnitude")
    stats_text = (f"Pearson r = {r_p:.3f}\nSpearman ρ = {r_s:.3f}\nR² = {r2:.3f}\n"
                  f"Emp. N/G ratio: {ratio_emp:.2f}\nPred. N/G ratio: {ratio_pred:.2f}")
    ax_a.text(0.03, 0.97, stats_text, transform=ax_a.transAxes, fontsize=7, va='top',
              fontfamily='monospace', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CT_NEURON_COLOR, markersize=7, label='Neuronal'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CT_NONNEURON_COLOR, markersize=7, label='Non-neuronal')]
    ax_a.legend(handles=legend_elements, loc='lower right', fontsize=7)
    ax_a.text(-0.15, 1.05, "A", transform=ax_a.transAxes, fontsize=14, fontweight='bold', va='bottom')

    # Panel B
    ax_b = fig.add_subplot(gs[0, 1])
    comp_methods = [m for m in ["ImputeVAE", "scGen", "scButterfly"] if m in method_corrs]
    y_pos = np.arange(len(comp_methods))
    for i, m in enumerate(comp_methods):
        ax_b.scatter(method_corrs[m]["pearson"], i, color=METHOD_COLORS[m], s=80,
                    marker='o', zorder=3, edgecolors='black', linewidth=0.5)
        ax_b.scatter(method_corrs[m]["r2"], i + 0.15, color=METHOD_COLORS[m], s=80,
                    marker='s', zorder=3, edgecolors='black', linewidth=0.5)
    ax_b.set_yticks(y_pos); ax_b.set_yticklabels(comp_methods)
    ax_b.set_xlabel("Value")
    ax_b.set_title("D_empirical vs D_predicted\ncorrelation across cell types")
    ax_b.invert_yaxis()
    legend_elements2 = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=7, label='Pearson r'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=7, label='R²')]
    ax_b.legend(handles=legend_elements2, loc='lower left', fontsize=7)
    ax_b.text(-0.15, 1.05, "B", transform=ax_b.transAxes, fontsize=14, fontweight='bold', va='bottom')

    # Panel C
    ax_c = fig.add_subplot(gs[0, 2])
    sorted_cts = sorted(cts_both, key=lambda ct: D_emp[ct])
    y_pos_c = np.arange(len(sorted_cts))
    for i, ct in enumerate(sorted_cts):
        c = ct_color(ct)
        ax_c.plot([D_emp[ct], D_pred[ct]], [i, i], color='gray', linewidth=0.5, zorder=1)
        ax_c.scatter(D_emp[ct], i, color='black', s=30, zorder=3, marker='o', edgecolors='black', linewidth=0.3)
        ax_c.scatter(D_pred[ct], i, color=c, s=30, zorder=3, marker='s', edgecolors='black', linewidth=0.3)
    ax_c.set_yticks(y_pos_c); ax_c.set_yticklabels(sorted_cts, fontsize=7)
    ax_c.set_xlabel("Correction magnitude\n(mean |delta| across genes)")
    for i, ct in enumerate(sorted_cts):
        ax_c.get_yticklabels()[i].set_color(ct_color(ct))
    legend_elements3 = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=6, label='Empirical'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=6, label='ImputeVAE')]
    ax_c.legend(handles=legend_elements3, loc='lower right', fontsize=6)
    ax_c.text(-0.15, 1.05, "C", transform=ax_c.transAxes, fontsize=14, fontweight='bold', va='bottom')

    fig.suptitle("Figure 4: ImputeVAE recovers cell-type-dependent transformation magnitudes",
                 fontsize=12, fontweight='bold', y=1.02)

    for fmt in ['pdf', 'svg', 'png']:
        fig.savefig(OUT_DIR / f"figure4_cell_type_specificity.{fmt}", bbox_inches='tight', dpi=300)
    plt.close(fig)

    # Save data
    data_rows = []
    for ct in ALL_CTS:
        row = {'cell_type': ct, 'group': 'neuronal' if ct in NEURONAL_CTS else 'non-neuronal',
               'D_empirical': D_emp.get(ct, np.nan), 'D_predicted_ImputeVAE': D_pred.get(ct, np.nan),
               'D_emp_CI_lo': D_emp_ci.get(ct, (np.nan,))[0],
               'D_emp_CI_hi': D_emp_ci.get(ct, (np.nan, np.nan))[1],
               'D_pred_CI_lo': D_pred_ci.get(ct, (np.nan,))[0],
               'D_pred_CI_hi': D_pred_ci.get(ct, (np.nan, np.nan))[1],
               'n_cells': ct_counts['cell'].get(ct, 0), 'n_nuclei': ct_counts['nucleus'].get(ct, 0)}
        for mn in ["scGen", "scButterfly"]:
            row[f'D_predicted_{mn}'] = method_D.get(mn, {}).get(ct, np.nan)
        data_rows.append(row)
    data_rows.append({'cell_type': 'SUMMARY', 'pearson_r': r_p, 'spearman_rho': r_s, 'R2': r2,
                      'emp_neuron_nonneuron_ratio': ratio_emp, 'pred_neuron_nonneuron_ratio': ratio_pred})
    for mn in comp_methods:
        if mn in method_corrs:
            data_rows[-1][f'pearson_{mn}'] = method_corrs[mn]['pearson']
            data_rows[-1][f'R2_{mn}'] = method_corrs[mn]['r2']
            data_rows[-1][f'MAE_{mn}'] = method_corrs[mn]['mae']
    pd.DataFrame(data_rows).to_csv(OUT_DIR / "figure4_cell_type_specificity_data.csv", index=False)
    print("  Figure 4 DONE")


# ---------------------------------------------------------------------------
# FIGURE 5
# ---------------------------------------------------------------------------

def figure5(ct_means_cell, ct_vars_cell, ct_labels_nuc, ct_counts):
    print("\n" + "="*60)
    print("FIGURE 5: Variance retention")
    print("="*60)

    # Real cell per-gene variance mean (for ratio denominator)
    real_cell_var_mean = {}
    for ct in ALL_CTS:
        if ct in ct_vars_cell:
            real_cell_var_mean[ct] = float(ct_vars_cell[ct].mean())

    # Compute variance ratios for each method, one at a time
    method_order = ["ImputeVAE", "CT-mean", "Linear reg", "scVI", "scANVI", "scGen", "scButterfly"]
    methods_variance = {}

    for mname in method_order:
        m_means, m_vars, _ = load_pred_ct_means_and_vars(mname, ct_labels_nuc)
        per_ct = {}
        for ct in ALL_CTS:
            if ct in m_vars and ct in real_cell_var_mean and real_cell_var_mean[ct] > 0:
                per_ct[ct] = float(m_vars[ct].mean() / real_cell_var_mean[ct])
        methods_variance[mname] = per_ct
        print(f"    {mname}: median ratio = {np.median(list(per_ct.values())):.4f}")

    # Panel C: Per-gene variance correlation for ImputeVAE
    print("  Computing per-gene variance correlation (Panel C)...")
    imp = np.load(f"{PRED_BASE['ImputeVAE']}/imputed_cell_from_nucleus.npy").astype(np.float32)
    obs = pd.read_csv(f"{PRED_BASE['ImputeVAE']}/nucleus_obs.csv", index_col=0)
    ct_imp = obs['cell_type'].values.astype(str)

    gene_var_corrs = {}
    for ct in ALL_CTS:
        mask = ct_imp == ct
        if mask.sum() >= 5 and ct in ct_vars_cell:
            imp_gene_var = imp[mask].var(axis=0)
            cell_gene_var = ct_vars_cell[ct]
            valid = (imp_gene_var > 0) & (cell_gene_var > 0)
            if valid.sum() > 10:
                r, _ = stats.spearmanr(imp_gene_var[valid], cell_gene_var[valid])
                gene_var_corrs[ct] = float(r)
    del imp; gc.collect()

    median_gene_var_corr = float(np.median(list(gene_var_corrs.values()))) if gene_var_corrs else np.nan
    print(f"  Per-gene variance correlation median: {median_gene_var_corr:.4f}")

    # ============== CREATE FIGURE ==============
    fig = plt.figure(figsize=(14, 5.5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35, width_ratios=[1.2, 1, 0.8])

    # Panel A
    ax_a = fig.add_subplot(gs[0, 0])
    box_data = []
    for mname in method_order:
        vals = [methods_variance[mname].get(ct, np.nan) for ct in ALL_CTS]
        vals = [v for v in vals if np.isfinite(v)]
        box_data.append(vals)

    bp = ax_a.boxplot(box_data, vert=True, patch_artist=True, widths=0.5,
                       showfliers=False, medianprops=dict(color='black', linewidth=1.5))
    for patch, mname in zip(bp['boxes'], method_order):
        patch.set_facecolor(METHOD_COLORS[mname])
        patch.set_alpha(0.4)

    for i, mname in enumerate(method_order):
        for ct in ALL_CTS:
            v = methods_variance[mname].get(ct, np.nan)
            if np.isfinite(v):
                jitter = np.random.RandomState(hash(ct) % 2**31).uniform(-0.15, 0.15)
                v_plot = min(v, 1.55)
                ax_a.scatter(i + 1 + jitter, v_plot, color=ct_color(ct), s=12, alpha=0.7,
                           edgecolors='black', linewidth=0.2, zorder=3)

    ax_a.axhline(1.0, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
    ax_a.set_xticks(range(1, len(method_order) + 1))
    ax_a.set_xticklabels(method_order, rotation=45, ha='right', fontsize=7)
    ax_a.set_ylabel("Variance retention ratio")
    ax_a.set_ylim(-0.05, 1.6)
    ax_a.set_title("Within-cell-type variance retention")

    for i, mname in enumerate(method_order):
        vals = [methods_variance[mname].get(ct, np.nan) for ct in ALL_CTS]
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            ax_a.text(i + 1, -0.02, f"{np.median(vals):.3f}", ha='center', va='top', fontsize=6, color='gray')

    legend_elements = [
        Line2D([0], [0], color='red', linestyle='--', linewidth=0.8, label='Real cell target (1.0)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CT_NEURON_COLOR, markersize=5, label='Neuronal'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CT_NONNEURON_COLOR, markersize=5, label='Non-neuronal')]
    ax_a.legend(handles=legend_elements, loc='upper right', fontsize=6)
    ax_a.text(-0.1, 1.05, "A", transform=ax_a.transAxes, fontsize=14, fontweight='bold', va='bottom')

    # Panel B
    ax_b = fig.add_subplot(gs[0, 1])
    ivae_ratios = methods_variance["ImputeVAE"]
    sorted_cts_b = sorted([ct for ct in ALL_CTS if ct in ivae_ratios], key=lambda ct: ivae_ratios[ct])
    y_pos_b = np.arange(len(sorted_cts_b))
    ax_b.barh(y_pos_b, [ivae_ratios[ct] for ct in sorted_cts_b],
              color=[ct_color(ct) for ct in sorted_cts_b], alpha=0.7, edgecolor='black', linewidth=0.3)
    ax_b.axvline(1.0, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
    ax_b.set_yticks(y_pos_b); ax_b.set_yticklabels(sorted_cts_b, fontsize=7)
    ax_b.set_xlabel("Variance retention ratio")
    ax_b.set_title("ImputeVAE per cell type")
    for i, ct in enumerate(sorted_cts_b):
        ax_b.get_yticklabels()[i].set_color(ct_color(ct))
    med_ivae = np.median([ivae_ratios[ct] for ct in sorted_cts_b])
    ax_b.axvline(med_ivae, color='gray', linestyle=':', linewidth=0.8)
    ax_b.text(med_ivae + 0.01, len(sorted_cts_b) - 0.5, f"median={med_ivae:.3f}", fontsize=7, color='gray')
    ax_b.text(-0.15, 1.05, "B", transform=ax_b.transAxes, fontsize=14, fontweight='bold', va='bottom')

    # Panel C
    ax_c = fig.add_subplot(gs[0, 2])
    sorted_cts_c = sorted(gene_var_corrs.keys(), key=lambda ct: gene_var_corrs[ct])
    y_pos_c = np.arange(len(sorted_cts_c))
    ax_c.barh(y_pos_c, [gene_var_corrs[ct] for ct in sorted_cts_c],
              color=[ct_color(ct) for ct in sorted_cts_c], alpha=0.7, edgecolor='black', linewidth=0.3)
    ax_c.axvline(0, color='gray', linestyle='-', linewidth=0.5)
    ax_c.set_yticks(y_pos_c); ax_c.set_yticklabels(sorted_cts_c, fontsize=7)
    ax_c.set_xlabel("Spearman correlation\n(per-gene variance)")
    ax_c.set_title("Variance structure\npreservation")
    for i, ct in enumerate(sorted_cts_c):
        ax_c.get_yticklabels()[i].set_color(ct_color(ct))
    ax_c.text(0.5, -0.12, f"Median: {median_gene_var_corr:.3f}", transform=ax_c.transAxes,
              fontsize=7, ha='center', color='gray')
    ax_c.text(-0.15, 1.05, "C", transform=ax_c.transAxes, fontsize=14, fontweight='bold', va='bottom')

    fig.suptitle("Figure 5: ImputeVAE partially preserves within-cell-type variation",
                 fontsize=12, fontweight='bold', y=1.02)

    for fmt in ['pdf', 'svg', 'png']:
        fig.savefig(OUT_DIR / f"figure5_variance_retention.{fmt}", bbox_inches='tight', dpi=300)
    plt.close(fig)

    # Save data
    data_rows = []
    for ct in ALL_CTS:
        row = {'cell_type': ct, 'group': 'neuronal' if ct in NEURONAL_CTS else 'non-neuronal',
               'real_cell_variance': real_cell_var_mean.get(ct, np.nan),
               'n_cells': ct_counts['cell'].get(ct, 0), 'n_nuclei': ct_counts['nucleus'].get(ct, 0),
               'gene_var_spearman': gene_var_corrs.get(ct, np.nan)}
        for mn in method_order:
            row[f'variance_ratio_{mn}'] = methods_variance[mn].get(ct, np.nan)
        data_rows.append(row)
    summary = {'cell_type': 'SUMMARY', 'median_gene_var_corr': median_gene_var_corr}
    for mn in method_order:
        vals = [methods_variance[mn].get(ct, np.nan) for ct in ALL_CTS]
        vals = [v for v in vals if np.isfinite(v)]
        summary[f'median_ratio_{mn}'] = float(np.median(vals)) if vals else np.nan
    data_rows.append(summary)
    pd.DataFrame(data_rows).to_csv(OUT_DIR / "figure5_variance_retention_data.csv", index=False)
    print("  Figure 5 DONE")


# ---------------------------------------------------------------------------
# Captions and supporting files
# ---------------------------------------------------------------------------

def write_captions():
    with open(OUT_DIR / "figure3_directed_reconstruction_caption.txt", 'w') as f:
        f.write("""Figure 3: Directed cross-decoding recovers the empirical nucleus-cell shift.

(A) Exploratory population-level benchmark comparing ImputeVAE against seven reference methods across three metrics: median gene correlation across cell types (left), median R-squared at the cell-type mean level (center), and median Wasserstein-1 distance between predicted and observed whole-cell gene distributions (right; lower values indicate better distributional match). Each metric is displayed on its own scale. Gene correlation measures how well the pattern of cell-type-specific expression is recovered for each gene. R-squared measures the proportion of variance in observed whole-cell means explained by the predictions. Wasserstein-1 distance is computed on the 200 most variable genes using quantile matching. Raw nuclei R-squared and Wasserstein are not computed (marked NA).

(B) Density scatter (hexagonal binning) of ImputeVAE-predicted shift versus empirical shift across all gene-by-cell-type combinations (10,134 genes x 17 cell types). The empirical shift is defined as the difference in mean log1p expression between observed whole cells and raw nuclei of the same cell type. The predicted shift is the corresponding difference between ImputeVAE predictions and raw nuclei. Dashed lines show x = 0 and y = 0 reference lines; the red line shows the identity (y = x). Pearson correlation, Spearman correlation, R-squared, and the number of data points are reported.

(C) Direction classification analysis. The sign of the empirical shift serves as a binary reference class. Left: ROC curve with AUROC. Right: precision-recall curve with AUC, along with sensitivity, specificity, balanced accuracy, MCC, and class frequencies at threshold zero. Approximately 94% of gene-cell-type combinations have positive empirical shifts.

Panel D (GO enrichment) is omitted; enrichment inputs and the 10,134-gene statistical background could not be verified without re-running the analysis.

Data partition: ImputeVAE was fitted on the training partition (60% of nuclei) and evaluated on the full working dataset (all nuclei). Comparison methods were fitted using the full working dataset and therefore had access to more observations; their values are exploratory and likely optimistic. No values represent held-out evaluation. All values computed on 10,134 shared genes in log1p(raw count) space across 17 cell types. Cells and nuclei are unpaired; all deltas are population means.
""")

    with open(OUT_DIR / "figure4_cell_type_specificity_caption.txt", 'w') as f:
        f.write("""Figure 4: ImputeVAE recovers cell-type-dependent transformation magnitudes.

For each cell type c, the empirical correction magnitude D_empirical(c) is the mean across 10,134 genes of the absolute difference between observed whole-cell means and raw nucleus means in log1p(raw count) space. The predicted correction magnitude D_predicted(c) replaces observed cell means with method-predicted cell means.

(A) Scatter plot of ImputeVAE-predicted versus empirical correction magnitudes for 17 cell types. Blue = neuronal (12 types); green = non-neuronal (5 types). Error bars: 95% bootstrap CIs (500 iterations, resampling cells and nuclei within each type). Dashed line = identity. Pearson r, Spearman rho, R-squared, and neuronal-to-non-neuronal magnitude ratios are reported.

(B) Summary dot plot comparing D_predicted-vs-D_empirical correlation for ImputeVAE, scGen (global shift), and scButterfly (cross-modal translation). scGen and scButterfly were fitted on the full dataset (exploratory, full-data fit).

(C) Paired-dot plot of D_empirical (black circles) and D_predicted (colored squares) for each cell type, sorted by D_empirical. Connecting lines visualize over- or under-correction. Label colors: blue = neuronal, green = non-neuronal.

Data partition: All analyses use the full working dataset (71,117 cells, 40,166 nuclei). ImputeVAE was trained on 60% of nuclei; predictions cover all nuclei. Comparison methods were fitted on the full dataset. Validation cell and nuclei sample counts per cell type are provided in the data CSV.
""")

    with open(OUT_DIR / "figure5_variance_retention_caption.txt", 'w') as f:
        f.write("""Figure 5: ImputeVAE partially preserves within-cell-type variation.

The variance retention ratio is defined as the mean per-gene variance of predicted profiles within a cell type divided by the corresponding mean per-gene variance of observed whole-cell profiles. A ratio of 1.0 indicates preservation of total within-type variation. A ratio near 0 indicates variance collapse. All computations use 10,134 genes in log1p(raw count) space.

(A) Box plots of variance retention ratios across cell types for seven methods. Individual cell-type points overlaid (blue = neuronal, green = non-neuronal). Red dashed line = real cell target (1.0). CT-mean produces ratio 0 by definition. Median values annotated below each box. Linear regression ratios above 1 reflect per-cell deviation amplification without a latent bottleneck. Values above 1.0 may indicate amplified heterogeneity rather than necessarily noise. The y-axis is capped at 1.6; Linear regression values above this range are clipped for display.

(B) ImputeVAE variance retention ratio per cell type, sorted. Gray dotted line = median; red dashed line = real cell target.

(C) Per-gene variance structure preservation: Spearman correlation between per-gene variances in ImputeVAE predictions and observed whole cells within each cell type. Tests whether the model preserves which genes are variable, beyond total variance. Median correlation across cell types is reported.

Data partition: All analyses use the full working dataset. ImputeVAE was trained on 60% of nuclei; predictions cover all nuclei. Comparison methods used the full dataset. Variance retention does not distinguish biological from prediction noise; values above 1.0 should not be interpreted as noise without further analysis. Sample counts per cell type are in the data CSV.
""")


def write_missing_data():
    with open(OUT_DIR / "missing_figure_data.txt", 'w') as f:
        f.write("""MISSING FIGURE DATA REPORT
=========================

1. VALIDATION-ONLY PREDICTIONS (affects all figures)
   Status: NOT AVAILABLE
   Impact: All figures use full-data predictions. All panels reflect internal
   full-data consistency, not held-out evaluation.
   Required: ImputeVAE predictions restricted to validation nuclei only,
   plus corresponding validation cell/nucleus partitions.

2. FIGURE 3, Panel D: GO enrichment
   Status: OMITTED
   Reason: Cannot verify enrichment statistics match current 10,134-gene
   background without re-running analysis. Deferred to supplementary figure.

3. RAW NUCLEI Wasserstein and R-squared
   Status: NA in Figure 3A
   Reason: Not computed in evaluation pipeline for the identity baseline.
""")


def write_readme():
    with open(OUT_DIR / "README.md", 'w') as f:
        f.write("""# Qualifying Exam Figures 3, 4, 5

Generated: 2026-08-06

## Script
`generate_figures.py`

## Input files
- Eval JSONs: `run/{method}/eval/eval_*.json`
- Prediction arrays: `run/{method}/imputed_cell_from_nucleus.npy` (40166 x 10134, log1p, full data)
- Pre-computed deltas: `compartment_classification/gt_delta_per_ct.csv`
- Raw h5ad: loaded via `analysis/evaluate_baseline.py:load_raw_data()` pattern

## Output files
- `figure3_directed_reconstruction.{pdf,svg,png}` + `_data.csv` + `_caption.txt`
- `figure3_empirical_delta_matrix.csv`, `figure3_predicted_delta_matrix.csv`
- `figure4_cell_type_specificity.{pdf,svg,png}` + `_data.csv` + `_caption.txt`
- `figure5_variance_retention.{pdf,svg,png}` + `_data.csv` + `_caption.txt`
- `figure_data_audit.{txt,csv}`
- `missing_figure_data.txt`

## Data partition
All analyses use FULL DATA. No validation-only arrays available.
See `figure_data_audit.txt` for details.
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("="*60)
    print("Generating qualifying exam figures")
    print("="*60)

    write_captions()
    write_missing_data()
    write_readme()

    # Phase 1: Load raw data, compute CT means/vars, free raw arrays
    print("\nPhase 1: Loading raw data...")
    ct_means, ct_vars, ct_counts, ct_labels_nuc, gene_names = load_raw_data_lean()
    ct_means_cell = ct_means['cell']
    ct_means_nuc = ct_means['nucleus']
    ct_vars_cell = ct_vars['cell']
    print(f"  Gene names: {len(gene_names)}, Cell types: {len(ALL_CTS)}")

    # Phase 2: Generate figures (pass figure numbers on CLI to run a subset)
    figs_to_run = set(sys.argv[1:]) if len(sys.argv) > 1 else {"3", "4", "5"}
    if "3" in figs_to_run:
        figure3(ct_means_cell, ct_means_nuc, gene_names)
    if "4" in figs_to_run:
        figure4(ct_means_cell, ct_means_nuc, ct_labels_nuc, ct_counts)
    if "5" in figs_to_run:
        figure5(ct_means_cell, ct_vars_cell, ct_labels_nuc, ct_counts)

    print("\n" + "="*60)
    print("ALL FIGURES GENERATED")
    print("="*60)
    for f in sorted(OUT_DIR.iterdir()):
        print(f"  {f.name}")
