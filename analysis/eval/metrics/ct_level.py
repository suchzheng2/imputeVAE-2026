"""
Cell-type-level metrics. All functions are pure: arrays in, numbers out.

Metrics:
  - gene_correlation: per-gene Pearson r across CT means
  - r_squared: per-gene R^2 across CT means
  - shift_scores: correlation shift (decoded_vs_cell - decoded_vs_nuclei)
  - classification: logistic regression trained on raw cell
  - de_preservation: one-vs-rest Wilcoxon DE, top-K overlap
  - coexpression: gene-gene correlation matrix preservation
"""

import numpy as np
from scipy import stats
from scipy.stats import ranksums


def compute_celltype_means(X, cell_types, cell_type_list=None):
    """Compute per-cell-type mean expression.

    Parameters
    ----------
    X : np.ndarray [N, G]
    cell_types : np.ndarray [N] str
    cell_type_list : list of str or None
        If provided, only compute for these cell types.

    Returns
    -------
    dict : ct -> [G] array
    """
    cts = cell_type_list or sorted(set(cell_types))
    means = {}
    for ct in cts:
        mask = cell_types == ct
        if mask.sum() > 0:
            means[ct] = X[mask].mean(axis=0)
    return means


def gene_correlation(imputed_means, target_means, common_cts):
    """Per-gene Pearson correlation across cell-type means.

    Parameters
    ----------
    imputed_means : dict, ct -> [G] array
    target_means : dict, ct -> [G] array (ground truth, usually raw cell)
    common_cts : list of str

    Returns
    -------
    per_gene : np.ndarray [G] of Pearson r values
    summary : dict with median, mean, std, pct_above_0.8, pct_above_0.5
    """
    imp_matrix = np.array([imputed_means[ct] for ct in common_cts])
    tgt_matrix = np.array([target_means[ct] for ct in common_cts])
    n_genes = imp_matrix.shape[1]

    per_gene = np.zeros(n_genes)
    for g in range(n_genes):
        if np.std(imp_matrix[:, g]) > 0 and np.std(tgt_matrix[:, g]) > 0:
            per_gene[g], _ = stats.pearsonr(imp_matrix[:, g], tgt_matrix[:, g])
        else:
            per_gene[g] = 0.0

    summary = {
        'median': float(np.nanmedian(per_gene)),
        'mean': float(np.nanmean(per_gene)),
        'std': float(np.nanstd(per_gene)),
        'pct_above_0.8': float((per_gene > 0.8).mean() * 100),
        'pct_above_0.5': float((per_gene > 0.5).mean() * 100),
    }
    return per_gene, summary


def r_squared(imputed_means, target_means, common_cts):
    """Per-gene R^2 across cell-type means.

    Returns
    -------
    per_gene : np.ndarray [G]
    summary : dict with median, mean
    """
    imp_matrix = np.array([imputed_means[ct] for ct in common_cts])
    tgt_matrix = np.array([target_means[ct] for ct in common_cts])
    n_genes = imp_matrix.shape[1]

    per_gene = np.zeros(n_genes)
    for g in range(n_genes):
        ss_res = np.sum((imp_matrix[:, g] - tgt_matrix[:, g])**2)
        ss_tot = np.sum((tgt_matrix[:, g] - tgt_matrix[:, g].mean())**2)
        per_gene[g] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    summary = {
        'median': float(np.nanmedian(per_gene)),
        'mean': float(np.nanmean(per_gene)),
    }
    return per_gene, summary


def shift_scores(decoded_means, raw_cell_means, raw_nuc_means, common_cts):
    """Shift score per cell type: corr(decoded, raw_cell) - corr(decoded, raw_nuc).

    Positive = shifted toward cell profile (good for imputation).

    Returns
    -------
    per_ct : dict, ct -> shift score
    summary : dict with mean_shift, median_shift
    """
    per_ct = {}
    for ct in common_cts:
        if ct not in decoded_means or ct not in raw_cell_means or ct not in raw_nuc_means:
            continue
        dec = decoded_means[ct]
        rc = raw_cell_means[ct]
        rn = raw_nuc_means[ct]
        if np.std(dec) > 0:
            r_cell, _ = stats.pearsonr(dec, rc)
            r_nuc, _ = stats.pearsonr(dec, rn)
            per_ct[ct] = float(r_cell - r_nuc)

    shifts = list(per_ct.values())
    summary = {
        'mean_shift': float(np.mean(shifts)) if shifts else 0.0,
        'median_shift': float(np.median(shifts)) if shifts else 0.0,
    }
    return per_ct, summary


def classification(X_cell, ct_cell, X_test, ct_test, common_cts, n_pca=50):
    """Logistic regression: train on raw cell, predict on test data.

    Uses PCA to reduce dimensionality before fitting (10k features is too slow).

    Parameters
    ----------
    X_cell : [N_cell, G] training data (raw cell)
    ct_cell : [N_cell] training labels
    X_test : [N_test, G] test data (raw nuclei or imputed)
    ct_test : [N_test] test labels
    n_pca : int, number of PCA components (None to skip PCA)

    Returns
    -------
    dict with accuracy, macro_f1, per_class report
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    # Filter to common cell types
    cell_mask = np.isin(ct_cell, common_cts)
    test_mask = np.isin(ct_test, common_cts)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_cell[cell_mask])
    X_test_scaled = scaler.transform(X_test[test_mask])

    # PCA reduction for speed (10k features -> n_pca)
    if n_pca and X_train_scaled.shape[1] > n_pca:
        pca = PCA(n_components=n_pca, random_state=42)
        X_train_scaled = pca.fit_transform(X_train_scaled)
        X_test_scaled = pca.transform(X_test_scaled)

    clf = LogisticRegression(
        multi_class='multinomial', solver='lbfgs',
        max_iter=1000, random_state=42, n_jobs=-1, C=1.0,
    )
    clf.fit(X_train_scaled, ct_cell[cell_mask])

    pred = clf.predict(X_test_scaled)

    acc = accuracy_score(ct_test[test_mask], pred)
    f1 = f1_score(ct_test[test_mask], pred, average='macro')
    report = classification_report(ct_test[test_mask], pred, output_dict=True)

    return {
        'accuracy': float(acc),
        'macro_f1': float(f1),
        'report': report,
    }


def de_preservation(X_source, ct_source, X_target, ct_target, common_cts, top_k=100):
    """DE gene preservation: one-vs-rest Wilcoxon, top-K overlap.

    Parameters
    ----------
    X_source : [N, G] ground truth (raw cell)
    ct_source : [N] labels
    X_target : [N, G] test data (imputed or raw nuclei)
    ct_target : [N] labels

    Returns
    -------
    per_ct : dict, ct -> overlap fraction
    summary : dict with mean_overlap
    """
    def _run_de(X, cts, top_k):
        de_genes = {}
        for ct in sorted(set(cts)):
            mask = cts == ct
            n_in, n_out = mask.sum(), (~mask).sum()
            if n_in < 5 or n_out < 5:
                continue
            X_in = X[mask]
            X_out = X[~mask]
            # Vectorized: compute fold changes all at once
            fcs = X_in.mean(axis=0) - X_out.mean(axis=0)
            # Wilcoxon rank-sum per gene (still needs loop but much less overhead)
            pvals = np.ones(X.shape[1])
            for g in range(X.shape[1]):
                _, pvals[g] = ranksums(X_in[:, g], X_out[:, g])
            order = np.lexsort((-np.abs(fcs), pvals))
            de_genes[ct] = set(order[:top_k])
        return de_genes

    de_source = _run_de(X_source, ct_source, top_k)
    de_target = _run_de(X_target, ct_target, top_k)

    per_ct = {}
    for ct in common_cts:
        if ct in de_source and ct in de_target:
            overlap = len(de_source[ct] & de_target[ct])
            per_ct[ct] = overlap / top_k

    overlaps = list(per_ct.values())
    summary = {
        'mean_overlap': float(np.mean(overlaps)) if overlaps else 0.0,
    }
    return per_ct, summary


def coexpression(X_ref, X_test, top_k_genes=500):
    """Gene-gene coexpression preservation (Spearman on correlation matrices).

    Parameters
    ----------
    X_ref : [N, G] reference (raw cell)
    X_test : [N, G] test (imputed or raw nuclei)
    top_k_genes : int
        Use top-K variable genes for tractability.

    Returns
    -------
    dict with spearman_r
    """
    k = min(top_k_genes, X_ref.shape[1])
    gene_var = X_ref.var(axis=0)
    top_idx = np.argsort(gene_var)[-k:]

    ref_corr = np.corrcoef(X_ref[:, top_idx].T)
    test_corr = np.corrcoef(X_test[:, top_idx].T)

    triu = np.triu_indices(k, k=1)
    ref_flat = ref_corr[triu]
    test_flat = test_corr[triu]

    valid = ~(np.isnan(ref_flat) | np.isnan(test_flat))
    if valid.sum() < 10:
        return {'spearman_r': 0.0}

    r, _ = stats.spearmanr(ref_flat[valid], test_flat[valid])
    return {'spearman_r': float(r)}


def run_all_ct_level(sources, eval_data, model_config):
    """Run all CT-level metrics on 5-source dict.

    Parameters
    ----------
    sources : dict from inference.run_5source_inference
    eval_data : EvalData from data.load_eval_data
    model_config : dict from registry

    Returns
    -------
    dict of all results
    """
    from ..inference import get_imputed

    imputed = get_imputed(sources, model_config)
    X_cell = sources['raw_cell']
    X_nuc = sources['raw_nuclei']
    ct_cell = eval_data.ct_cell
    ct_nuc = eval_data.ct_nuc
    common_cts = eval_data.common_cts

    # Cell-type means
    cell_means = compute_celltype_means(X_cell, ct_cell, common_cts)
    nuc_means = compute_celltype_means(X_nuc, ct_nuc, common_cts)
    imputed_means = compute_celltype_means(imputed, ct_nuc, common_cts)

    # Also compute for decoded_cell (reconstruction quality)
    dec_cell_means = compute_celltype_means(sources['decoded_cell'], ct_cell, common_cts)

    results = {}

    # Gene correlation
    gc_per_gene, gc_summary = gene_correlation(imputed_means, cell_means, common_cts)
    results['gene_corr'] = gc_summary
    results['gene_corr_per_gene'] = gc_per_gene

    # Also: raw nuclei baseline
    _, gc_nuc_summary = gene_correlation(nuc_means, cell_means, common_cts)
    results['gene_corr_raw_nuc'] = gc_nuc_summary

    # Cell reconstruction quality
    _, gc_recon_summary = gene_correlation(dec_cell_means, cell_means, common_cts)
    results['gene_corr_cell_recon'] = gc_recon_summary

    # R-squared
    r2_per_gene, r2_summary = r_squared(imputed_means, cell_means, common_cts)
    results['r_squared'] = r2_summary
    results['r_squared_per_gene'] = r2_per_gene

    # Shift scores (for imputed)
    imp_shift, shift_summary = shift_scores(imputed_means, cell_means, nuc_means, common_cts)
    results['shift'] = shift_summary
    results['shift_per_ct'] = imp_shift

    # Classification
    print("    Computing classification...")
    clf_imputed = classification(X_cell, ct_cell, imputed, ct_nuc, common_cts)
    clf_raw_nuc = classification(X_cell, ct_cell, X_nuc, ct_nuc, common_cts)
    results['clf_imputed'] = {
        'accuracy': clf_imputed['accuracy'],
        'macro_f1': clf_imputed['macro_f1'],
    }
    results['clf_raw_nuc'] = {
        'accuracy': clf_raw_nuc['accuracy'],
        'macro_f1': clf_raw_nuc['macro_f1'],
    }
    results['clf_improvement'] = clf_imputed['accuracy'] - clf_raw_nuc['accuracy']
    results['clf_report'] = clf_imputed['report']

    # DE preservation
    print("    Computing DE preservation...")
    de_imp_per_ct, de_imp_summary = de_preservation(
        X_cell, ct_cell, imputed, ct_nuc, common_cts
    )
    de_nuc_per_ct, de_nuc_summary = de_preservation(
        X_cell, ct_cell, X_nuc, ct_nuc, common_cts
    )
    results['de_imputed'] = de_imp_summary
    results['de_imputed_per_ct'] = de_imp_per_ct
    results['de_raw_nuc'] = de_nuc_summary
    results['de_improvement'] = de_imp_summary['mean_overlap'] - de_nuc_summary['mean_overlap']

    # Coexpression
    print("    Computing coexpression...")
    results['coexp_imputed'] = coexpression(X_cell, imputed)
    results['coexp_raw_nuc'] = coexpression(X_cell, X_nuc)

    # Store means for downstream use
    results['_means'] = {
        'cell': cell_means,
        'nuclei': nuc_means,
        'imputed': imputed_means,
    }

    return results
