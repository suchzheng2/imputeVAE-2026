"""
Gene-stratified evaluation: per-quintile breakdown of CT-level AND sample-level metrics.

Stratification schemes:
  1. Expression quintiles (by mean cell expression)
  2. Cell-nucleus gap quintiles (by absolute difference in CT means)
  3. Biological categories (IEG, ribosomal, neuronal, glial, TFs)
"""

import numpy as np
import pandas as pd
from scipy import stats


# --- Biological category gene lists ---
IEG_GENES = {
    'Fos', 'Fosb', 'Jun', 'Junb', 'Egr1', 'Egr2', 'Egr3', 'Egr4',
    'Arc', 'Nr4a1', 'Nr4a2', 'Nr4a3', 'Npas4', 'Homer1', 'Btg2',
}
RIBOSOMAL_PREFIXES = ('Rpl', 'Rps')
NEURONAL_MARKERS = {
    'Slc17a7', 'Slc17a6', 'Gad1', 'Gad2', 'Snap25', 'Syt1',
    'Rbfox3', 'Map2', 'Tubb3', 'Dcx',
}
GLIAL_MARKERS = {
    'Gfap', 'Aqp4', 'Aldh1l1', 'Sox9',  # astrocytes
    'Mbp', 'Plp1', 'Mog', 'Mag',          # oligos
    'Cx3cr1', 'P2ry12', 'Tmem119',         # microglia
}
TF_MARKERS = {
    'Sox2', 'Pax6', 'Neurod1', 'Neurod2', 'Neurod6', 'Tbr1',
    'Dlx1', 'Dlx2', 'Dlx5', 'Lhx6', 'Nkx2-1',
}


def assign_strata(gene_names, cell_means_array, nuc_means_array):
    """Assign genes to expression quintiles, gap quintiles, and biological categories.

    Parameters
    ----------
    gene_names : [G] str array
    cell_means_array : [G] mean expression across all cell types (raw cell)
    nuc_means_array : [G] mean expression across all cell types (raw nuclei)

    Returns
    -------
    pd.DataFrame with columns:
        gene_name, mean_expr_cell, mean_expr_nuc, cell_nuc_diff, abs_diff,
        expr_quintile (1-5), diff_quintile (1-5), bio_category
    """
    df = pd.DataFrame({
        'gene_name': gene_names,
        'mean_expr_cell': cell_means_array,
        'mean_expr_nuc': nuc_means_array,
        'cell_nuc_diff': cell_means_array - nuc_means_array,
        'abs_diff': np.abs(cell_means_array - nuc_means_array),
    })

    # Expression quintiles (by cell expression)
    df['expr_quintile'] = pd.qcut(
        df['mean_expr_cell'], q=5, labels=[1, 2, 3, 4, 5], duplicates='drop'
    ).astype(int)

    # Cell-nucleus gap quintiles (by absolute difference)
    df['diff_quintile'] = pd.qcut(
        df['abs_diff'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]
    ).astype(int)

    # Biological categories
    def _categorize(name):
        if name in IEG_GENES:
            return 'IEG'
        if any(name.startswith(p) for p in RIBOSOMAL_PREFIXES):
            return 'Ribosomal'
        if name in NEURONAL_MARKERS:
            return 'Neuronal'
        if name in GLIAL_MARKERS:
            return 'Glial'
        if name in TF_MARKERS:
            return 'TF'
        return 'Other'

    df['bio_category'] = df['gene_name'].apply(_categorize)

    return df


def compute_stratum_metrics(stratum_df, stratum_col, per_gene_corr,
                            per_gene_r2=None, per_gene_var_ratio=None,
                            per_gene_spearman=None):
    """Compute summary metrics per stratum.

    Parameters
    ----------
    stratum_df : DataFrame with gene assignments (from assign_strata)
    stratum_col : str, column to group by (e.g., 'expr_quintile', 'diff_quintile')
    per_gene_corr : [G] per-gene correlation (CT-level)
    per_gene_r2 : [G] per-gene R^2 (optional)
    per_gene_var_ratio : [G] per-gene variance ratio (optional)
    per_gene_spearman : [G] per-gene within-CT Spearman (optional)

    Returns
    -------
    pd.DataFrame with one row per stratum
    """
    stratum_df = stratum_df.copy()
    stratum_df['corr'] = per_gene_corr
    if per_gene_r2 is not None:
        stratum_df['r2'] = per_gene_r2
    if per_gene_var_ratio is not None:
        stratum_df['var_ratio'] = per_gene_var_ratio
    if per_gene_spearman is not None:
        stratum_df['spearman'] = per_gene_spearman

    rows = []
    for name, group in stratum_df.groupby(stratum_col):
        row = {
            'stratum': name,
            'n_genes': len(group),
            'mean_expr_cell': float(group['mean_expr_cell'].mean()),
            'mean_expr_nuc': float(group['mean_expr_nuc'].mean()),
            'var_expr_cell': float(group['mean_expr_cell'].var()),
            'var_expr_nuc': float(group['mean_expr_nuc'].var()),
            'mean_abs_diff': float(group['abs_diff'].mean()),
            # CT-level metrics
            'median_corr': float(group['corr'].median()),
            'mean_corr': float(group['corr'].mean()),
            'std_corr': float(group['corr'].std()),
            'q25_corr': float(group['corr'].quantile(0.25)),
            'q75_corr': float(group['corr'].quantile(0.75)),
            'pct_above_0.8': float((group['corr'] > 0.8).mean() * 100),
            'pct_above_0.5': float((group['corr'] > 0.5).mean() * 100),
        }

        if per_gene_r2 is not None:
            row['median_r2'] = float(group['r2'].median())
            row['mean_r2'] = float(group['r2'].mean())

        if per_gene_var_ratio is not None:
            row['median_var_ratio'] = float(group['var_ratio'].median())
            row['mean_var_ratio'] = float(group['var_ratio'].mean())

        if per_gene_spearman is not None:
            row['median_spearman'] = float(group['spearman'].median())
            row['mean_spearman'] = float(group['spearman'].mean())

        rows.append(row)

    return pd.DataFrame(rows)


def _per_gene_variance_ratio(imputed, ct_imputed, X_cell, ct_cell, common_cts, n_genes):
    """Compute per-gene variance ratio (averaged across CTs)."""
    ratios = np.zeros(n_genes)
    counts = np.zeros(n_genes)

    for ct in common_cts:
        imp_mask = ct_imputed == ct
        cell_mask = ct_cell == ct
        if imp_mask.sum() < 5 or cell_mask.sum() < 5:
            continue

        imp_var = imputed[imp_mask].var(axis=0)
        cell_var = X_cell[cell_mask].var(axis=0)

        valid = cell_var > 0
        ratios[valid] += imp_var[valid] / cell_var[valid]
        counts[valid] += 1

    counts[counts == 0] = 1
    return ratios / counts


def _per_gene_within_ct_spearman(imputed, ct_imputed, X_nuc, ct_nuc, common_cts, n_genes):
    """Compute per-gene within-CT Spearman (averaged across CTs)."""
    corrs = np.zeros(n_genes)
    counts = np.zeros(n_genes)

    for ct in common_cts:
        imp_mask = ct_imputed == ct
        nuc_mask = ct_nuc == ct
        n = min(imp_mask.sum(), nuc_mask.sum())
        if n < 5:
            continue

        for g in range(n_genes):
            imp_vals = imputed[imp_mask, g][:n]
            nuc_vals = X_nuc[nuc_mask, g][:n]
            if np.std(imp_vals) > 0 and np.std(nuc_vals) > 0:
                r, _ = stats.spearmanr(imp_vals, nuc_vals)
                corrs[g] += r
                counts[g] += 1

    counts[counts == 0] = 1
    return corrs / counts


def run_gene_stratified(sources, eval_data, model_config):
    """Run full gene-stratified evaluation.

    Returns
    -------
    dict with:
        strata_df: gene-level DataFrame with assignments
        by_expr_quintile: summary by expression quintile
        by_diff_quintile: summary by cell-nucleus gap quintile
        by_bio_category: summary by biological category
    """
    from ..inference import get_imputed
    from .ct_level import compute_celltype_means

    imputed = get_imputed(sources, model_config)
    X_cell = sources['raw_cell']
    X_nuc = sources['raw_nuclei']
    ct_cell = eval_data.ct_cell
    ct_nuc = eval_data.ct_nuc
    common_cts = eval_data.common_cts
    gene_names = eval_data.gene_names
    n_genes = eval_data.n_genes

    # CT means
    cell_means = compute_celltype_means(X_cell, ct_cell, common_cts)
    imputed_means = compute_celltype_means(imputed, ct_nuc, common_cts)

    # Grand means across CTs
    cell_grand = np.mean([cell_means[ct] for ct in common_cts], axis=0)
    nuc_means_dict = compute_celltype_means(X_nuc, ct_nuc, common_cts)
    nuc_grand = np.mean([nuc_means_dict[ct] for ct in common_cts], axis=0)

    # Per-gene CT-level correlation
    imp_matrix = np.array([imputed_means[ct] for ct in common_cts])
    cell_matrix = np.array([cell_means[ct] for ct in common_cts])
    per_gene_corr = np.zeros(n_genes)
    per_gene_r2 = np.zeros(n_genes)
    for g in range(n_genes):
        if np.std(imp_matrix[:, g]) > 0 and np.std(cell_matrix[:, g]) > 0:
            r, _ = stats.pearsonr(imp_matrix[:, g], cell_matrix[:, g])
            per_gene_corr[g] = r
            ss_res = np.sum((imp_matrix[:, g] - cell_matrix[:, g])**2)
            ss_tot = np.sum((cell_matrix[:, g] - cell_matrix[:, g].mean())**2)
            per_gene_r2[g] = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Per-gene sample-level metrics
    print("    Computing per-gene variance ratio...")
    per_gene_var_ratio = _per_gene_variance_ratio(
        imputed, ct_nuc, X_cell, ct_cell, common_cts, n_genes
    )

    print("    Computing per-gene within-CT Spearman...")
    per_gene_spearman = _per_gene_within_ct_spearman(
        imputed, ct_nuc, X_nuc, ct_nuc, common_cts, n_genes
    )

    # Assign strata
    strata_df = assign_strata(gene_names, cell_grand, nuc_grand)

    # Compute per-stratum summaries
    results = {'strata_df': strata_df}

    for col, label in [
        ('expr_quintile', 'by_expr_quintile'),
        ('diff_quintile', 'by_diff_quintile'),
        ('bio_category', 'by_bio_category'),
    ]:
        results[label] = compute_stratum_metrics(
            strata_df, col, per_gene_corr,
            per_gene_r2=per_gene_r2,
            per_gene_var_ratio=per_gene_var_ratio,
            per_gene_spearman=per_gene_spearman,
        )

    # Also store per-gene arrays for detailed analysis
    results['per_gene'] = pd.DataFrame({
        'gene_name': gene_names,
        'corr': per_gene_corr,
        'r2': per_gene_r2,
        'var_ratio': per_gene_var_ratio,
        'spearman': per_gene_spearman,
        'mean_expr_cell': cell_grand,
        'mean_expr_nuc': nuc_grand,
        'abs_diff': np.abs(cell_grand - nuc_grand),
    })

    return results
