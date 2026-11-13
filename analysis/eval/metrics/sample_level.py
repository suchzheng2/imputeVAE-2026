"""
Sample-level (per-cell) metrics for hallucination detection.

Metrics:
  - variance_ratio: imputed within-CT variance / raw cell within-CT variance
  - within_ct_spearman: per-gene rank preservation within each cell type
  - per_cell_correlation: each imputed cell vs its CT mean in raw cell data
"""

import numpy as np
from scipy import stats


def variance_ratio(imputed, ct_imputed, X_cell, ct_cell, common_cts):
    """Per-cell-type variance ratio: var(imputed) / var(raw_cell).

    A ratio << 1 means variance collapse (hallucination).
    A ratio ~1 means per-cell variation is preserved.

    Parameters
    ----------
    imputed : [N_nuc, G] imputed expression
    ct_imputed : [N_nuc] cell type labels
    X_cell : [N_cell, G] raw cell expression
    ct_cell : [N_cell] cell type labels

    Returns
    -------
    per_ct : dict, ct -> variance ratio
    summary : dict with median_ratio, mean_ratio
    """
    per_ct = {}
    for ct in common_cts:
        imp_mask = ct_imputed == ct
        cell_mask = ct_cell == ct
        if imp_mask.sum() < 5 or cell_mask.sum() < 5:
            continue

        imp_var = imputed[imp_mask].var(axis=0).mean()
        cell_var = X_cell[cell_mask].var(axis=0).mean()

        if cell_var > 0:
            per_ct[ct] = float(imp_var / cell_var)

    ratios = list(per_ct.values())
    summary = {
        'median_ratio': float(np.median(ratios)) if ratios else 0.0,
        'mean_ratio': float(np.mean(ratios)) if ratios else 0.0,
    }
    return per_ct, summary


def within_ct_spearman(imputed, ct_imputed, X_nuc, ct_nuc, common_cts, n_genes_sample=None):
    """Within-CT Spearman: for each gene, correlate cell rankings in raw nuclei vs imputed.

    High Spearman means the model preserves which cells are high/low within a type.
    Low Spearman means the model is ignoring per-cell variation (hallucinating CT means).

    Parameters
    ----------
    imputed : [N_nuc, G]
    ct_imputed : [N_nuc]
    X_nuc : [N_nuc, G] raw nuclei (same cells as imputed)
    ct_nuc : [N_nuc]
    n_genes_sample : int or None
        If set, sample this many genes for speed.

    Returns
    -------
    per_ct : dict, ct -> median Spearman across genes
    summary : dict with overall_median
    """
    n_genes = imputed.shape[1]
    gene_idx = np.arange(n_genes)
    if n_genes_sample and n_genes_sample < n_genes:
        gene_idx = np.random.choice(n_genes, n_genes_sample, replace=False)

    per_ct = {}
    all_corrs = []

    for ct in common_cts:
        imp_mask = ct_imputed == ct
        nuc_mask = ct_nuc == ct
        # Both should index the same cells (same ordering)
        if imp_mask.sum() < 5:
            continue

        ct_corrs = []
        for g in gene_idx:
            imp_vals = imputed[imp_mask, g]
            nuc_vals = X_nuc[nuc_mask, g]
            if len(imp_vals) != len(nuc_vals):
                continue
            if np.std(imp_vals) > 0 and np.std(nuc_vals) > 0:
                r, _ = stats.spearmanr(imp_vals, nuc_vals)
                ct_corrs.append(r)

        if ct_corrs:
            per_ct[ct] = float(np.median(ct_corrs))
            all_corrs.extend(ct_corrs)

    summary = {
        'overall_median': float(np.median(all_corrs)) if all_corrs else 0.0,
        'overall_mean': float(np.mean(all_corrs)) if all_corrs else 0.0,
    }
    return per_ct, summary


def per_cell_correlation(imputed, ct_imputed, target_means, common_cts, method='pearson'):
    """Per-cell correlation: each imputed cell vs its CT mean in target (raw cell).

    Parameters
    ----------
    imputed : [N, G]
    ct_imputed : [N] labels
    target_means : dict, ct -> [G] (raw cell CT means)
    method : 'pearson' or 'spearman'

    Returns
    -------
    per_ct : dict, ct -> median correlation
    summary : dict with overall_median
    """
    corr_fn = stats.pearsonr if method == 'pearson' else stats.spearmanr

    per_ct = {}
    all_corrs = []

    for ct in common_cts:
        if ct not in target_means:
            continue
        mask = ct_imputed == ct
        ct_mean = target_means[ct]
        ct_corrs = []
        for i in np.where(mask)[0]:
            if np.std(imputed[i]) > 0:
                r, _ = corr_fn(imputed[i], ct_mean)
                ct_corrs.append(r)
        if ct_corrs:
            per_ct[ct] = float(np.median(ct_corrs))
            all_corrs.extend(ct_corrs)

    summary = {
        'overall_median': float(np.median(all_corrs)) if all_corrs else 0.0,
        'overall_mean': float(np.mean(all_corrs)) if all_corrs else 0.0,
    }
    return per_ct, summary


def run_all_sample_level(sources, eval_data, model_config):
    """Run all sample-level metrics.

    Parameters
    ----------
    sources : dict from inference.run_5source_inference
    eval_data : EvalData
    model_config : dict from registry

    Returns
    -------
    dict of all results
    """
    from ..inference import get_imputed
    from .ct_level import compute_celltype_means

    imputed = get_imputed(sources, model_config)
    X_cell = sources['raw_cell']
    X_nuc = sources['raw_nuclei']
    ct_cell = eval_data.ct_cell
    ct_nuc = eval_data.ct_nuc
    common_cts = eval_data.common_cts

    results = {}

    # Variance ratio
    print("    Computing variance ratio...")
    vr_per_ct, vr_summary = variance_ratio(
        imputed, ct_nuc, X_cell, ct_cell, common_cts
    )
    results['variance_ratio'] = vr_summary
    results['variance_ratio_per_ct'] = vr_per_ct

    # Within-CT Spearman
    print("    Computing within-CT Spearman...")
    sp_per_ct, sp_summary = within_ct_spearman(
        imputed, ct_nuc, X_nuc, ct_nuc, common_cts
    )
    results['within_ct_spearman'] = sp_summary
    results['within_ct_spearman_per_ct'] = sp_per_ct

    # Per-cell correlation to raw cell CT mean
    print("    Computing per-cell correlation...")
    cell_means = compute_celltype_means(X_cell, ct_cell, common_cts)
    pc_per_ct, pc_summary = per_cell_correlation(
        imputed, ct_nuc, cell_means, common_cts
    )
    results['per_cell_corr'] = pc_summary
    results['per_cell_corr_per_ct'] = pc_per_ct

    # Also: raw nuclei per-cell correlation (baseline)
    pc_nuc_per_ct, pc_nuc_summary = per_cell_correlation(
        X_nuc, ct_nuc, cell_means, common_cts
    )
    results['per_cell_corr_raw_nuc'] = pc_nuc_summary

    # Per-cell correlation of imputed to raw nuclei (reconstruction fidelity)
    # Each imputed cell vs the same raw nucleus
    imp_vs_nuc_corrs = []
    for i in range(len(ct_nuc)):
        if np.std(imputed[i]) > 0 and np.std(X_nuc[i]) > 0:
            r, _ = stats.pearsonr(imputed[i], X_nuc[i])
            imp_vs_nuc_corrs.append(r)
    results['per_cell_corr_imp_vs_raw_nuc'] = {
        'overall_median': float(np.median(imp_vs_nuc_corrs)) if imp_vs_nuc_corrs else 0.0,
    }

    return results
