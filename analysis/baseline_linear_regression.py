#!/usr/bin/env python3
"""
Baseline: Per-gene linear regression (nucleus → cell).

For each gene, fits cell_expr = a * nuc_expr + b using cell-type means as
anchor points (19 cell types). Then applies the learned mapping to individual
nuclei, preserving per-cell variation through the linear transform.

This is the simplest possible baseline and establishes a floor for the VAE.
Key question: does a linear per-gene transform also collapse to means, or
does it preserve more per-cell variation than the VAE?

Metrics match comprehensive_eval.py and hallucination_test.py for direct
comparison against VAE models.

Usage:
    python analysis/baseline_linear_regression.py [--gene_list PATH]
"""

import argparse
import gc
import os
import sys
import warnings

import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats, sparse
from scipy.stats import spearmanr, ranksums
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# =====================================================================
# Config
# =====================================================================
CELL_PATH = "10X_cells_v2_v3_AIBS/"
NUC_PATH = "10X_nuclei_v2_v3_AIBS/"
DEFAULT_GENE_LIST = "data/gene_list_perct_5pct.csv"

MAX_PER_TYPE = 2000
RANDOM_SEED = 42
TOP_K_DE = 100
K_COEXP_GENES = 500


def _resolve_raw_path(data_path):
    return os.path.join(
        "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi",
        data_path, "data.h5ad",
    )


def load_and_filter_data(data_path, batch, gene_list_path):
    """Load raw data, filter to v3 + gene list, return adata with log1p in .X"""
    adata = ad.read_h5ad(_resolve_raw_path(data_path))
    if 'meta_batch_id' not in adata.obs:
        adata.obs['meta_batch_id'] = adata.obs['batch_id'].apply(lambda x: x.split('.')[0])
    adata = adata[adata.obs['meta_batch_id'].str.endswith(batch)].copy()

    common_genes = pd.read_csv('data/common_genes.csv', header=0)['common_genes'].tolist()
    adata = adata[:, adata.var['gene_names'].isin(common_genes)].copy()

    common_ct = pd.read_csv('data/common_cell_types.csv', header=0)['common_cell_types'].tolist()
    adata = adata[adata.obs['cell_type'].isin(common_ct)].copy()

    gene_list = pd.read_csv(gene_list_path, header=None)[0].tolist()
    adata = adata[:, adata.var['gene_names'].isin(gene_list)].copy()

    adata = adata[:, adata.var['gene_names'].argsort()].copy()

    X = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
    adata.X = np.log1p(X).astype(np.float32)

    print(f"  Loaded {data_path} batch={batch}: {adata.shape}")
    return adata


def stratified_subsample(adata, max_per_type, seed=42):
    np.random.seed(seed)
    idx = []
    for ct in adata.obs['cell_type'].unique():
        ct_idx = np.where(adata.obs['cell_type'].values == ct)[0]
        if len(ct_idx) > max_per_type:
            ct_idx = np.random.choice(ct_idx, max_per_type, replace=False)
        idx.extend(ct_idx)
    return adata[sorted(idx)].copy()


def compute_celltype_means(X, cell_types):
    means = {}
    for ct in sorted(set(cell_types)):
        means[ct] = X[cell_types == ct].mean(axis=0)
    return means


def run_de_one_vs_rest(X, cell_types, top_k=100):
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


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gene_list', default=DEFAULT_GENE_LIST,
                        help='Gene list CSV path')
    parser.add_argument('--output_dir', default='notes/experiments/SUC-8_baseline/results',
                        help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("BASELINE: PER-GENE LINEAR REGRESSION (nucleus → cell)")
    print("=" * 80)
    print(f"Gene list: {args.gene_list}")
    print(f"Output: {args.output_dir}")

    # =================================================================
    # 1. Load data
    # =================================================================
    print("\n--- Loading data ---")
    cell_adata = load_and_filter_data(CELL_PATH, 'c3A', args.gene_list)
    nuc_adata = load_and_filter_data(NUC_PATH, 'n3A', args.gene_list)

    np.random.seed(RANDOM_SEED)
    cell_adata = stratified_subsample(cell_adata, MAX_PER_TYPE)
    nuc_adata = stratified_subsample(nuc_adata, MAX_PER_TYPE)
    print(f"After subsampling: Cell={cell_adata.shape}, Nuclei={nuc_adata.shape}")

    X_cell = cell_adata.X if isinstance(cell_adata.X, np.ndarray) else cell_adata.X.toarray()
    X_nuc = nuc_adata.X if isinstance(nuc_adata.X, np.ndarray) else nuc_adata.X.toarray()
    ct_cell = cell_adata.obs['cell_type'].values
    ct_nuc = nuc_adata.obs['cell_type'].values
    gene_names = cell_adata.var['gene_names'].values
    n_genes = X_cell.shape[1]

    common_cts = sorted(set(ct_cell) & set(ct_nuc))
    print(f"Common cell types: {len(common_cts)}")

    # =================================================================
    # 2. Compute cell-type means (anchor points for regression)
    # =================================================================
    print("\n--- Computing cell-type means ---")
    cell_means = compute_celltype_means(X_cell, ct_cell)
    nuc_means = compute_celltype_means(X_nuc, ct_nuc)

    # Build mean matrices: [n_ct, n_genes]
    cell_mean_matrix = np.array([cell_means[ct] for ct in common_cts])
    nuc_mean_matrix = np.array([nuc_means[ct] for ct in common_cts])

    # =================================================================
    # 3. Fit per-gene linear regression on cell-type means
    # =================================================================
    print("\n--- Fitting per-gene linear regression ---")
    # For each gene: cell_mean[g] = a[g] * nuc_mean[g] + b[g]
    # Fit across the 19 cell types
    slopes = np.zeros(n_genes)
    intercepts = np.zeros(n_genes)
    r_values = np.zeros(n_genes)

    for g in range(n_genes):
        x = nuc_mean_matrix[:, g]  # nuclei CT means
        y = cell_mean_matrix[:, g]  # cell CT means

        if np.std(x) > 1e-10:
            slope, intercept, r_value, _, _ = stats.linregress(x, y)
            slopes[g] = slope
            intercepts[g] = intercept
            r_values[g] = r_value
        else:
            # Gene has no variance across CT means in nuclei — use cell mean
            slopes[g] = 0.0
            intercepts[g] = y.mean()
            r_values[g] = 0.0

    print(f"  Regression fit quality (R across CT means):")
    print(f"    Median R: {np.median(r_values):.4f}")
    print(f"    Mean R:   {np.mean(r_values):.4f}")
    print(f"    Genes with R > 0.9: {(r_values > 0.9).sum()}/{n_genes}")

    # Save regression coefficients
    coeff_df = pd.DataFrame({
        'gene': gene_names,
        'slope': slopes,
        'intercept': intercepts,
        'r_value': r_values,
    })
    coeff_df.to_csv(f'{args.output_dir}/regression_coefficients.csv', index=False)

    # =================================================================
    # 4. Apply to individual nuclei → imputed cell profiles
    # =================================================================
    print("\n--- Applying regression to nuclei ---")
    # imputed[i, g] = slopes[g] * X_nuc[i, g] + intercepts[g]
    imputed = X_nuc * slopes[np.newaxis, :] + intercepts[np.newaxis, :]
    imputed = imputed.astype(np.float32)
    # Clip negative values (log1p space should be >= 0)
    imputed = np.clip(imputed, 0, None)
    print(f"  Imputed shape: {imputed.shape}")

    # Also compute cell-type mean baseline (for reference)
    mean_imputed = np.zeros_like(X_nuc)
    for i, ct in enumerate(ct_nuc):
        if ct in cell_means:
            mean_imputed[i] = cell_means[ct]

    # =================================================================
    # 5. Evaluation metrics
    # =================================================================
    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)

    results = {}

    # --- 5a. Cell-type mean correlation ---
    print("\n  [Cell-type mean correlation]")
    imputed_means = compute_celltype_means(imputed, ct_nuc)
    imp_mean_matrix = np.array([imputed_means.get(ct, np.zeros(n_genes)) for ct in common_cts])

    imp_gene_corr = []
    imp_gene_r2 = []
    for g in range(n_genes):
        if np.std(imp_mean_matrix[:, g]) > 0 and np.std(cell_mean_matrix[:, g]) > 0:
            r, _ = stats.pearsonr(imp_mean_matrix[:, g], cell_mean_matrix[:, g])
            ss_res = np.sum((imp_mean_matrix[:, g] - cell_mean_matrix[:, g]) ** 2)
            ss_tot = np.sum((cell_mean_matrix[:, g] - cell_mean_matrix[:, g].mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        else:
            r, r2 = 0, 0
        imp_gene_corr.append(r)
        imp_gene_r2.append(r2)

    # Raw nuclei means vs cell means (for comparison)
    nuc_gene_corr = []
    for g in range(n_genes):
        if np.std(nuc_mean_matrix[:, g]) > 0 and np.std(cell_mean_matrix[:, g]) > 0:
            r, _ = stats.pearsonr(nuc_mean_matrix[:, g], cell_mean_matrix[:, g])
        else:
            r = 0
        nuc_gene_corr.append(r)

    results['gene_corr_linreg'] = np.nanmedian(imp_gene_corr)
    results['gene_r2_linreg'] = np.nanmedian(imp_gene_r2)
    results['gene_corr_raw_nuc'] = np.nanmedian(nuc_gene_corr)
    print(f"  Linear reg imputed→raw cell (per-gene median corr):  {results['gene_corr_linreg']:.4f}")
    print(f"  Linear reg imputed→raw cell (per-gene median R²):    {results['gene_r2_linreg']:.4f}")
    print(f"  Raw nuclei→raw cell (per-gene median corr):          {results['gene_corr_raw_nuc']:.4f}")

    # --- 5b. Per-cell correlation ---
    print("\n  [Per-cell correlation]")
    percell_corr_to_cellmean = []
    for i, ct in enumerate(ct_nuc):
        if ct in cell_means and np.std(imputed[i]) > 0:
            r, _ = stats.pearsonr(imputed[i], cell_means[ct])
            percell_corr_to_cellmean.append(r)

    percell_corr_to_raw_nuc = []
    for i in range(len(ct_nuc)):
        if np.std(imputed[i]) > 0 and np.std(X_nuc[i]) > 0:
            r, _ = stats.pearsonr(imputed[i], X_nuc[i])
            percell_corr_to_raw_nuc.append(r)

    # Ceiling: real cell vs own CT mean
    cell_percell_to_mean = []
    for i, ct in enumerate(ct_cell):
        if ct in cell_means and np.std(X_cell[i]) > 0:
            r, _ = stats.pearsonr(X_cell[i], cell_means[ct])
            cell_percell_to_mean.append(r)

    results['percell_imp_vs_cellmean'] = np.nanmedian(percell_corr_to_cellmean)
    results['percell_imp_vs_rawnuc'] = np.nanmedian(percell_corr_to_raw_nuc)
    results['percell_realcell_ceiling'] = np.nanmedian(cell_percell_to_mean)
    print(f"  Imputed vs raw cell-type mean (median):  {results['percell_imp_vs_cellmean']:.4f}")
    print(f"  Imputed vs raw nucleus (median):         {results['percell_imp_vs_rawnuc']:.4f}")
    print(f"  [Ceiling] Real cell vs own CT mean:      {results['percell_realcell_ceiling']:.4f}")

    # --- 5c. Classification ---
    print("\n  [Cell-type classification]")
    scaler = StandardScaler()
    X_cell_scaled = scaler.fit_transform(X_cell)

    cell_ct_mask = np.isin(ct_cell, common_cts)
    nuc_ct_mask = np.isin(ct_nuc, common_cts)

    clf = LogisticRegression(
        multi_class='multinomial', solver='lbfgs',
        max_iter=1000, random_state=42, n_jobs=-1, C=1.0,
    )
    clf.fit(X_cell_scaled[cell_ct_mask], ct_cell[cell_ct_mask])

    # Raw nuclei
    X_nuc_scaled = scaler.transform(X_nuc)
    raw_nuc_pred = clf.predict(X_nuc_scaled[nuc_ct_mask])
    raw_nuc_acc = accuracy_score(ct_nuc[nuc_ct_mask], raw_nuc_pred)

    # Imputed nuclei (linear regression)
    imputed_scaled = scaler.transform(imputed)
    imp_pred = clf.predict(imputed_scaled[nuc_ct_mask])
    imp_acc = accuracy_score(ct_nuc[nuc_ct_mask], imp_pred)
    imp_f1 = f1_score(ct_nuc[nuc_ct_mask], imp_pred, average='macro')

    # CT mean imputed (pure mean baseline)
    mean_imp_scaled = scaler.transform(mean_imputed)
    mean_pred = clf.predict(mean_imp_scaled[nuc_ct_mask])
    mean_acc = accuracy_score(ct_nuc[nuc_ct_mask], mean_pred)

    results['clf_raw_nuc_acc'] = raw_nuc_acc
    results['clf_linreg_acc'] = imp_acc
    results['clf_linreg_f1'] = imp_f1
    results['clf_ctmean_acc'] = mean_acc
    print(f"  Train on raw cell, predict on:")
    print(f"    Raw nuclei:                accuracy={raw_nuc_acc:.4f}")
    print(f"    Linear regression imputed:  accuracy={imp_acc:.4f}, macro-F1={imp_f1:.4f}")
    print(f"    Cell-type mean imputed:     accuracy={mean_acc:.4f}")
    print(f"    LinReg improvement vs raw:  {(imp_acc - raw_nuc_acc)*100:+.1f}%")

    report = classification_report(ct_nuc[nuc_ct_mask], imp_pred, output_dict=True)
    pd.DataFrame(report).T.to_csv(f'{args.output_dir}/classification_report_linreg.csv')

    # --- 5d. DE preservation ---
    print("\n  [DE gene preservation]")
    de_raw_cell = run_de_one_vs_rest(X_cell, ct_cell, top_k=TOP_K_DE)
    de_imputed = run_de_one_vs_rest(imputed, ct_nuc, top_k=TOP_K_DE)
    de_raw_nuc = run_de_one_vs_rest(X_nuc, ct_nuc, top_k=TOP_K_DE)

    de_overlaps_imp = {}
    de_overlaps_raw_nuc = {}
    for ct in common_cts:
        if ct in de_raw_cell and ct in de_imputed:
            de_overlaps_imp[ct] = len(set(de_raw_cell[ct]) & set(de_imputed[ct])) / TOP_K_DE
        if ct in de_raw_cell and ct in de_raw_nuc:
            de_overlaps_raw_nuc[ct] = len(set(de_raw_cell[ct]) & set(de_raw_nuc[ct])) / TOP_K_DE

    mean_de_imp = np.mean(list(de_overlaps_imp.values())) if de_overlaps_imp else 0
    mean_de_raw = np.mean(list(de_overlaps_raw_nuc.values())) if de_overlaps_raw_nuc else 0

    results['de_overlap_linreg'] = mean_de_imp
    results['de_overlap_raw_nuc'] = mean_de_raw
    print(f"  Top-{TOP_K_DE} DE overlap with raw cell DE:")
    print(f"    Linear regression:  {mean_de_imp:.3f}")
    print(f"    Raw nuclei:         {mean_de_raw:.3f}")
    print(f"    Improvement:        {(mean_de_imp - mean_de_raw)*100:+.1f}%")

    # Per-CT DE detail
    de_rows = []
    print(f"\n  Per-cell-type DE overlap:")
    for ct in common_cts:
        imp_ov = de_overlaps_imp.get(ct, 0)
        raw_ov = de_overlaps_raw_nuc.get(ct, 0)
        print(f"    {ct:<20} linreg={imp_ov:.2f}  raw_nuc={raw_ov:.2f}  delta={imp_ov-raw_ov:+.2f}")
        de_rows.append({'cell_type': ct, 'de_overlap_linreg': imp_ov, 'de_overlap_raw_nuc': raw_ov})
    pd.DataFrame(de_rows).to_csv(f'{args.output_dir}/de_overlap_per_ct.csv', index=False)

    # --- 5e. Coexpression ---
    print("\n  [Gene-gene coexpression preservation]")
    gene_var = X_cell.var(axis=0)
    top_var_idx = np.argsort(gene_var)[-K_COEXP_GENES:]

    cell_coexp = np.corrcoef(X_cell[:, top_var_idx].T)
    imp_coexp = np.corrcoef(imputed[:, top_var_idx].T)
    nuc_coexp = np.corrcoef(X_nuc[:, top_var_idx].T)

    triu_idx = np.triu_indices(K_COEXP_GENES, k=1)
    cell_flat = cell_coexp[triu_idx]
    imp_flat = imp_coexp[triu_idx]
    nuc_flat = nuc_coexp[triu_idx]

    valid = ~(np.isnan(cell_flat) | np.isnan(imp_flat) | np.isnan(nuc_flat))
    coexp_imp, _ = stats.spearmanr(cell_flat[valid], imp_flat[valid])
    coexp_nuc, _ = stats.spearmanr(cell_flat[valid], nuc_flat[valid])

    results['coexp_linreg'] = coexp_imp
    results['coexp_raw_nuc'] = coexp_nuc
    print(f"  Coexpression (Spearman, top {K_COEXP_GENES} variable genes):")
    print(f"    Linear regression vs raw cell:  {coexp_imp:.4f}")
    print(f"    Raw nuclei vs raw cell:         {coexp_nuc:.4f}")

    # =================================================================
    # 6. Hallucination tests (variance, ranking)
    # =================================================================
    print("\n" + "=" * 80)
    print("HALLUCINATION TESTS")
    print("=" * 80)

    # --- A1: Per-cell variance ---
    print("\n  [A1] Per-cell variance (imputed vs real cell)")
    variance_rows = []
    print(f"  {'Cell Type':<20} {'Var(linreg)':>14} {'Var(real cell)':>14} {'Ratio':>10}")
    for ct in common_cts:
        nuc_mask = ct_nuc == ct
        cell_mask = ct_cell == ct
        imp_var = imputed[nuc_mask].var(axis=0).mean()
        cell_var = X_cell[cell_mask].var(axis=0).mean()
        ratio = imp_var / (cell_var + 1e-12)
        print(f"  {ct:<20} {imp_var:>14.6f} {cell_var:>14.6f} {ratio:>10.4f}")
        variance_rows.append({
            'cell_type': ct, 'n_nuc': nuc_mask.sum(), 'n_cell': cell_mask.sum(),
            'var_linreg': imp_var, 'var_real_cell': cell_var, 'ratio': ratio,
        })

    var_df = pd.DataFrame(variance_rows)
    var_df.to_csv(f'{args.output_dir}/A1_variance_linreg.csv', index=False)
    overall_var_ratio = var_df['var_linreg'].mean() / var_df['var_real_cell'].mean()
    results['variance_ratio'] = overall_var_ratio
    print(f"  Overall variance ratio (linreg/real): {overall_var_ratio:.4f}")

    # --- A2: Within-CT ranking preservation ---
    print("\n  [A2] Within-cell-type ranking preservation (Spearman)")
    ranking_rows = []
    print(f"  {'Cell Type':<20} {'Median Spearman':>16} {'Frac > 0':>10}")
    for ct in common_cts:
        nuc_mask = ct_nuc == ct
        raw_nuc_ct = X_nuc[nuc_mask]
        imp_ct = imputed[nuc_mask]

        if nuc_mask.sum() < 10:
            continue

        gene_spearmans = []
        for g in range(n_genes):
            if raw_nuc_ct[:, g].std() > 0 and imp_ct[:, g].std() > 0:
                rho, _ = spearmanr(raw_nuc_ct[:, g], imp_ct[:, g])
                if not np.isnan(rho):
                    gene_spearmans.append(rho)

        gene_spearmans = np.array(gene_spearmans)
        med = np.median(gene_spearmans) if len(gene_spearmans) > 0 else 0
        frac_pos = (gene_spearmans > 0).mean() if len(gene_spearmans) > 0 else 0

        print(f"  {ct:<20} {med:>16.4f} {frac_pos:>10.3f}")
        ranking_rows.append({
            'cell_type': ct, 'n_nuc': nuc_mask.sum(),
            'median_spearman': med, 'frac_positive': frac_pos,
        })

    rank_df = pd.DataFrame(ranking_rows)
    rank_df.to_csv(f'{args.output_dir}/A2_ranking_linreg.csv', index=False)
    results['within_ct_spearman'] = rank_df['median_spearman'].mean()
    results['within_ct_frac_pos'] = rank_df['frac_positive'].mean()
    print(f"  Overall: median Spearman={results['within_ct_spearman']:.4f}, "
          f"frac>0={results['within_ct_frac_pos']:.3f}")

    # =================================================================
    # 7. Summary
    # =================================================================
    print("\n" + "=" * 80)
    print("SUMMARY: LINEAR REGRESSION BASELINE")
    print("=" * 80)

    for k, v in sorted(results.items()):
        print(f"  {k:<35} {v:.4f}")

    # Save summary
    pd.DataFrame([results]).to_csv(f'{args.output_dir}/summary.csv', index=False)

    # =================================================================
    # 8. Summary figure
    # =================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Linear Regression Baseline: Evaluation', fontsize=14)

    # A1: Variance ratio
    ax = axes[0, 0]
    x_pos = range(len(common_cts))
    ratios = [var_df.loc[var_df['cell_type'] == ct, 'ratio'].values[0] for ct in common_cts]
    ax.bar(x_pos, ratios, color='steelblue', edgecolor='black')
    ax.axhline(1.0, color='red', linestyle='--', label='Ratio=1.0')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(common_cts, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Var(linreg) / Var(real cell)')
    ax.set_title('A1: Per-CT variance ratio')
    ax.legend(fontsize=8)

    # A2: Within-CT Spearman
    ax = axes[0, 1]
    ct_labels = rank_df['cell_type'].values
    spearmans = rank_df['median_spearman'].values
    ax.bar(range(len(ct_labels)), spearmans, color='coral', edgecolor='black')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(range(len(ct_labels)))
    ax.set_xticklabels(ct_labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Median Spearman (within CT)')
    ax.set_title('A2: Within-CT ranking preservation')

    # Regression fit quality
    ax = axes[1, 0]
    ax.hist(r_values, bins=50, edgecolor='black', alpha=0.7, color='mediumpurple')
    ax.axvline(np.median(r_values), color='red', linestyle='--',
               label=f'Median R={np.median(r_values):.3f}')
    ax.set_xlabel('Pearson R (across CT means)')
    ax.set_ylabel('# genes')
    ax.set_title('Regression fit quality per gene')
    ax.legend(fontsize=8)

    # Slope distribution
    ax = axes[1, 1]
    ax.hist(slopes, bins=50, edgecolor='black', alpha=0.7, color='forestgreen')
    ax.axvline(1.0, color='red', linestyle='--', label='slope=1 (identity)')
    ax.axvline(np.median(slopes), color='blue', linestyle='--',
               label=f'Median={np.median(slopes):.3f}')
    ax.set_xlabel('Slope (per-gene)')
    ax.set_ylabel('# genes')
    ax.set_title('Per-gene regression slopes')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{args.output_dir}/summary_linreg.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nAll results saved to: {args.output_dir}/")
    print("Done!")


if __name__ == '__main__':
    main()
