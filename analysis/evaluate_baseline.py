"""
Unified evaluation for any baseline method.

Takes an imputed expression matrix + raw cell/nucleus data and computes all
metrics for cross-method comparison. Works with any method that produces
a [N_nuc, G] imputed cell expression matrix from nucleus input.

Input: either
  (a) --imputed_npy path to .npy file with imputed matrix
  (b) --imputed_csv path to .csv file
  Along with --obs_csv for cell type labels of the imputed nuclei

Usage:
    python analysis/evaluate_baseline.py \
        --imputed_npy run/scvi_baseline/imputed_cell_from_nucleus.npy \
        --obs_csv run/scvi_baseline/nucleus_obs.csv \
        --name scvi \
        --output_dir run/scvi_baseline/eval

Metrics computed:
  CT-level:   gene_corr, r2, shift_score, classification, DE overlap, coexpression
  Sample-level: variance_ratio, within_ct_spearman, per_cell_corr
  Label-free:  per_gene_corr_allcells, wasserstein, gene_mse
  Biological: delta_vs_groundtruth, compartment_scores, ct_specific_delta
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse, stats

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from analysis.eval.metrics.ct_level import (
    compute_celltype_means, gene_correlation, r_squared,
    shift_scores, classification, de_preservation, coexpression,
)
from analysis.eval.metrics.sample_level import (
    variance_ratio, within_ct_spearman, per_cell_correlation,
)


# ---------------------------------------------------------------------------
# Data loading (matches baseline_linear_regression.py / baseline_scvi.py)
# ---------------------------------------------------------------------------
DATA_ROOT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi"
DEFAULT_GENE_LIST = "data/training_gene_list_v3.csv"


def _resolve_raw_path(data_dir):
    return os.path.join(DATA_ROOT, data_dir, "data.h5ad")


def load_raw_data(gene_list_path=DEFAULT_GENE_LIST, normalization="log1p"):
    """Load raw cell and nucleus data with same filtering as training.

    Returns normalized data for metric computation.
    normalization: "log1p" (default) or "cpm" (log1p of CPM).
    """
    gene_list = pd.read_csv(gene_list_path, header=None)[0].tolist()
    common_genes = pd.read_csv('data/common_genes.csv', header=0)['common_genes'].tolist()
    common_ct = pd.read_csv('data/common_cell_types.csv', header=0)['common_cell_types'].tolist()

    results = {}
    for name, data_dir, batch in [("cell", "10X_cells_v2_v3_AIBS", "c3A"),
                                   ("nucleus", "10X_nuclei_v2_v3_AIBS", "n3A")]:
        adata = ad.read_h5ad(_resolve_raw_path(data_dir))
        if 'meta_batch_id' not in adata.obs:
            adata.obs['meta_batch_id'] = adata.obs['batch_id'].apply(lambda x: x.split('.')[0])
        # Apply all filters on the view BEFORE .copy() to avoid
        # materialising the full dense matrix (39 GiB as int64).
        row_mask = (adata.obs['meta_batch_id'].str.endswith(batch) &
                    adata.obs['cell_type'].isin(common_ct))
        col_mask = (adata.var['gene_names'].isin(common_genes) &
                    adata.var['gene_names'].isin(gene_list))
        adata = adata[row_mask, col_mask].copy()
        adata = adata[:, adata.var['gene_names'].argsort()].copy()

        X = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
        if normalization == "cpm":
            lib_size = np.clip(X.sum(axis=1, keepdims=True), 1.0, None)
            adata.X = np.log1p(X / lib_size * 1e6).astype(np.float32)
        else:
            adata.X = np.log1p(X).astype(np.float32)
        print(f"  Loaded {name} ({normalization}): {adata.shape}")
        results[name] = adata

    return results["cell"], results["nucleus"]


# ---------------------------------------------------------------------------
# Label-free metrics (address reviewer concern about circular evaluation)
# ---------------------------------------------------------------------------

def per_gene_correlation_allcells(imputed, X_cell, n_quantiles=500):
    """Per-gene Pearson correlation across ALL cells (no cell-type labels).

    Compares distributions via quantile-quantile matching: for each gene,
    compute quantiles at n_quantiles evenly spaced points, then correlate.
    """
    n_genes = imputed.shape[1]
    quantiles = np.linspace(0, 1, n_quantiles)
    # Vectorized: compute all quantiles at once [n_quantiles, n_genes]
    imp_q = np.quantile(imputed, quantiles, axis=0)
    cell_q = np.quantile(X_cell, quantiles, axis=0)
    # Pearson correlation per gene (column-wise)
    corrs = np.zeros(n_genes)
    imp_std = imp_q.std(axis=0)
    cell_std = cell_q.std(axis=0)
    valid = (imp_std > 0) & (cell_std > 0)
    if valid.any():
        imp_centered = imp_q[:, valid] - imp_q[:, valid].mean(axis=0)
        cell_centered = cell_q[:, valid] - cell_q[:, valid].mean(axis=0)
        numer = (imp_centered * cell_centered).sum(axis=0)
        denom = np.sqrt((imp_centered**2).sum(axis=0) * (cell_centered**2).sum(axis=0))
        corrs[valid] = numer / denom
    return {
        'median': float(np.nanmedian(corrs)),
        'mean': float(np.nanmean(corrs)),
        'per_gene': corrs,
    }


def wasserstein_per_gene(imputed, X_cell, n_genes_sample=200):
    """Per-gene Wasserstein-1 distance between imputed and real cell distributions."""
    n_genes = imputed.shape[1]
    if n_genes_sample and n_genes_sample < n_genes:
        gene_var = X_cell.var(axis=0)
        gene_idx = np.argsort(gene_var)[-n_genes_sample:]
    else:
        gene_idx = np.arange(n_genes)

    # Wasserstein-1 on sorted arrays is just mean(|sort(a) - sort(b)|)
    # Vectorize by presorting and interpolating to common length
    n_common = min(imputed.shape[0], X_cell.shape[0], 5000)
    quantiles = np.linspace(0, 1, n_common)
    dists = []
    for g in gene_idx:
        imp_q = np.quantile(imputed[:, g], quantiles)
        cell_q = np.quantile(X_cell[:, g], quantiles)
        dists.append(np.mean(np.abs(imp_q - cell_q)))

    return {
        'median': float(np.median(dists)),
        'mean': float(np.mean(dists)),
        'n_genes': len(dists),
    }


# ---------------------------------------------------------------------------
# Biological validation metrics
# ---------------------------------------------------------------------------

# Known gene sets for cell-vs-nucleus biology (mouse gene names)
RIBOSOMAL_GENES = [
    # RPL family
    'Rpl3', 'Rpl4', 'Rpl5', 'Rpl6', 'Rpl7', 'Rpl7a', 'Rpl8', 'Rpl9',
    'Rpl10', 'Rpl10a', 'Rpl11', 'Rpl12', 'Rpl13', 'Rpl13a', 'Rpl14',
    'Rpl15', 'Rpl17', 'Rpl18', 'Rpl18a', 'Rpl19', 'Rpl21', 'Rpl22',
    'Rpl23', 'Rpl23a', 'Rpl24', 'Rpl26', 'Rpl27', 'Rpl27a', 'Rpl28',
    'Rpl29', 'Rpl30', 'Rpl31', 'Rpl32', 'Rpl34', 'Rpl35', 'Rpl35a',
    'Rpl36', 'Rpl36a', 'Rpl37', 'Rpl37a', 'Rpl38', 'Rpl39', 'Rpl41',
    # RPS family
    'Rps2', 'Rps3', 'Rps3a1', 'Rps4x', 'Rps5', 'Rps6', 'Rps7', 'Rps8',
    'Rps9', 'Rps10', 'Rps11', 'Rps12', 'Rps13', 'Rps14', 'Rps15',
    'Rps15a', 'Rps16', 'Rps17', 'Rps18', 'Rps19', 'Rps20', 'Rps21',
    'Rps23', 'Rps24', 'Rps25', 'Rps26', 'Rps27', 'Rps27a', 'Rps28',
    'Rps29',
]

# Data-driven nuclear-enriched lncRNAs (delta < -0.2 in ground truth cell vs nuclei,
# identified by analysis/find_nuclear_lncrnas.py on n3A-filtered data).
# Replaces the old 6-gene list which included Meg3 and Kcnq1ot1 — those are actually
# cell-enriched in this neuron-dominated dataset (positive ground truth delta).
NUCLEAR_LNCRNA_GENES = [
    # Known nuclear lncRNAs (confirmed nuclear-enriched in our data)
    'Xist', 'Malat1', 'Tsix',
    # Antisense transcripts
    'Nrg3os', 'Sorbs2os', 'Uckl1os',
    # RIKEN clones (likely lncRNAs)
    '1700024B05Rik', '1700024B18Rik', '1700085D07Rik', '1700111E14Rik',
    '4930415C11Rik', '4930419G24Rik', '4933413L06Rik', 'D830024N08Rik',
    # Gm* predicted genes (likely lncRNAs, nuclear-enriched)
    'Gm12027', 'Gm13883', 'Gm13944', 'Gm13963', 'Gm14330', 'Gm15478',
    'Gm15520', 'Gm15563', 'Gm16351', 'Gm19710', 'Gm1992', 'Gm20275',
    'Gm20642', 'Gm21954', 'Gm26699', 'Gm26798', 'Gm26804', 'Gm26854',
    'Gm26905', 'Gm28198', 'Gm28375', 'Gm28376', 'Gm32250', 'Gm35188',
    'Gm36975', 'Gm37679', 'Gm42439', 'Gm42477', 'Gm48678', 'Gm48742',
    'Gm6994',
]

CYTOPLASMIC_GENES = [
    # Cytoskeletal / actin
    'Actb', 'Tmsb4x', 'Actg1',
    # Calmodulin / calcium signaling (cytoplasmic)
    'Calm1', 'Calm2', 'Calm3', 'Camk2n1',
    # Translation factors (cytoplasmic)
    'Eef1a1', 'Eef2', 'Eif4a2',
    # Synaptic vesicle (cytoplasmic, neuron-enriched)
    'Vamp2', 'Snap25', 'Syt1', 'Nrgn',
    # 14-3-3 family (cytoplasmic)
    'Ywhab', 'Ywhaz', 'Ywhah', 'Ywhag',
]

NEURONAL_MARKER_GENES = [
    # Should show larger delta in neurons than in glia
    'Snap25', 'Syt1', 'Nrgn', 'Camk2n1', 'Vamp2', 'Rgs4',
    'Calm1', 'Calm3',
]

GLIAL_MARKER_GENES = [
    # Astrocyte/Oligo-specific genes with expected deltas
    'Plp1',   # myelin, Oligo
    'Mt1',    # metallothionein, Astro
    'Aldoc',  # aldolase C, Astro
]


def delta_vs_ground_truth(imputed, X_nuc, X_cell, ct_nuc, ct_cell, gene_names, common_cts):
    """Compare predicted delta to ground truth delta from real data.

    Ground truth: for each CT, delta_gt = mean(real_cell) - mean(real_nuc)
    Predicted:    for each CT, delta_pred = mean(imputed) - mean(raw_nuc)

    Returns correlation between delta_gt and delta_pred per CT, and overall.
    This is the KEY biological metric: does the method learn the right
    direction and magnitude of change for each gene?
    """
    cell_means = compute_celltype_means(X_cell, ct_cell, common_cts)
    nuc_means = compute_celltype_means(X_nuc, ct_nuc, common_cts)
    imp_means = compute_celltype_means(imputed, ct_nuc, common_cts)

    per_ct_corr = {}
    per_ct_r2 = {}
    all_gt = []
    all_pred = []

    for ct in common_cts:
        if ct not in cell_means or ct not in nuc_means or ct not in imp_means:
            continue
        delta_gt = cell_means[ct] - nuc_means[ct]     # what should change
        delta_pred = imp_means[ct] - nuc_means[ct]     # what method predicted

        if np.std(delta_gt) > 0 and np.std(delta_pred) > 0:
            r, _ = stats.pearsonr(delta_gt, delta_pred)
            per_ct_corr[ct] = float(r)

            ss_res = np.sum((delta_pred - delta_gt) ** 2)
            ss_tot = np.sum((delta_gt - delta_gt.mean()) ** 2)
            per_ct_r2[ct] = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        all_gt.append(delta_gt)
        all_pred.append(delta_pred)

    # Overall (concatenate all CTs)
    all_gt = np.concatenate(all_gt)
    all_pred = np.concatenate(all_pred)
    overall_r, _ = stats.pearsonr(all_gt, all_pred)
    ss_res = np.sum((all_pred - all_gt) ** 2)
    ss_tot = np.sum((all_gt - all_gt.mean()) ** 2)
    overall_r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        'overall_corr': float(overall_r),
        'overall_r2': overall_r2,
        'median_ct_corr': float(np.median(list(per_ct_corr.values()))),
        'per_ct_corr': per_ct_corr,
        'per_ct_r2': per_ct_r2,
    }


def compartment_scores(imputed, X_nuc, gene_names):
    """Score delta direction for known compartment gene sets.

    Cytoplasmic/ribosomal genes should have POSITIVE delta (imputed > nucleus).
    Nuclear lncRNAs should have NEGATIVE delta (imputed < nucleus).

    A method that treats modality as batch noise will shrink both toward zero.
    Our cVAE should show large positive for cytoplasmic, negative for nuclear.
    """
    delta = imputed.mean(axis=0) - X_nuc.mean(axis=0)  # global mean delta
    gene_list = list(gene_names)

    def _score_set(gene_set, expected_sign):
        """Return fraction of genes with correct sign and mean delta."""
        idx = [gene_list.index(g) for g in gene_set if g in gene_list]
        if not idx:
            return {'n_found': 0}
        deltas = delta[idx]
        if expected_sign > 0:
            frac_correct = float((deltas > 0).mean())
        else:
            frac_correct = float((deltas < 0).mean())
        return {
            'n_found': len(idx),
            'mean_delta': float(deltas.mean()),
            'median_delta': float(np.median(deltas)),
            'frac_correct_sign': frac_correct,
        }

    return {
        'ribosomal': _score_set(RIBOSOMAL_GENES, expected_sign=+1),
        'cytoplasmic': _score_set(CYTOPLASMIC_GENES, expected_sign=+1),
        'nuclear_lncrna': _score_set(NUCLEAR_LNCRNA_GENES, expected_sign=-1),
    }


def ct_specific_delta(imputed, X_nuc, ct_nuc, gene_names, common_cts):
    """Test whether neuronal genes have larger delta in neurons vs glia.

    Biological expectation: synaptic genes (Snap25, Syt1, Nrgn) should show
    larger imputation delta in excitatory/inhibitory neurons than in astrocytes
    or oligodendrocytes.
    """
    gene_list = list(gene_names)
    neuron_idx = [gene_list.index(g) for g in NEURONAL_MARKER_GENES if g in gene_list]

    if not neuron_idx:
        return {'n_genes_found': 0}

    # Define neuron vs glia CTs
    neuron_cts = [ct for ct in common_cts if any(
        x in ct for x in ['IT', 'ET', 'NP', 'CT', 'L6b', 'Pvalb', 'Sst', 'Vip', 'Lamp5', 'Sncg']
    )]
    glia_cts = [ct for ct in common_cts if any(
        x in ct for x in ['Astro', 'Oligo', 'OPC', 'Endo', 'Macrophage']
    )]

    if not neuron_cts or not glia_cts:
        return {'n_genes_found': len(neuron_idx), 'note': 'no neuron/glia CT separation'}

    # Compute mean delta for neuronal markers in neurons vs glia
    neuron_deltas = []
    glia_deltas = []

    for ct in neuron_cts:
        mask = ct_nuc == ct
        if mask.sum() < 5:
            continue
        d = imputed[mask][:, neuron_idx].mean() - X_nuc[mask][:, neuron_idx].mean()
        neuron_deltas.append(d)

    for ct in glia_cts:
        mask = ct_nuc == ct
        if mask.sum() < 5:
            continue
        d = imputed[mask][:, neuron_idx].mean() - X_nuc[mask][:, neuron_idx].mean()
        glia_deltas.append(d)

    return {
        'n_genes_found': len(neuron_idx),
        'mean_delta_neurons': float(np.mean(neuron_deltas)) if neuron_deltas else None,
        'mean_delta_glia': float(np.mean(glia_deltas)) if glia_deltas else None,
        'neuron_gt_glia': bool(
            neuron_deltas and glia_deltas and np.mean(neuron_deltas) > np.mean(glia_deltas)
        ),
        'ratio': float(np.mean(neuron_deltas) / np.mean(glia_deltas))
            if neuron_deltas and glia_deltas and np.mean(glia_deltas) != 0 else None,
    }


def gene_mse(imputed_means, target_means, common_cts):
    """Mean squared error at CT-mean level, per gene."""
    imp_matrix = np.array([imputed_means[ct] for ct in common_cts])
    tgt_matrix = np.array([target_means[ct] for ct in common_cts])
    per_gene = np.mean((imp_matrix - tgt_matrix) ** 2, axis=0)
    return {
        'median': float(np.median(per_gene)),
        'mean': float(np.mean(per_gene)),
        'per_gene': per_gene,
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(imputed, adata_cell, adata_nuc, ct_nuc_imputed, name="baseline"):
    """Run all metrics on imputed expression matrix.

    Parameters
    ----------
    imputed : np.ndarray [N_nuc, G] — predicted cell expression (log1p scale)
    adata_cell : AnnData — raw cell data (log1p in .X)
    adata_nuc : AnnData — raw nucleus data (log1p in .X)
    ct_nuc_imputed : np.ndarray [N_nuc] — cell type labels for imputed cells
    name : str — method name for display
    """
    X_cell = adata_cell.X if not sparse.issparse(adata_cell.X) else adata_cell.X.toarray()
    X_nuc = adata_nuc.X if not sparse.issparse(adata_nuc.X) else adata_nuc.X.toarray()
    ct_cell = adata_cell.obs['cell_type'].values.astype(str)
    ct_nuc = adata_nuc.obs['cell_type'].values.astype(str)

    common_cts = sorted(set(ct_cell) & set(ct_nuc) & set(ct_nuc_imputed))
    print(f"\nEvaluating '{name}': {imputed.shape[0]} imputed, "
          f"{X_cell.shape[0]} cells, {X_nuc.shape[0]} nuclei, "
          f"{len(common_cts)} common CTs")

    results = {'method': name}

    # ----- CT-level means -----
    cell_means = compute_celltype_means(X_cell, ct_cell, common_cts)
    nuc_means = compute_celltype_means(X_nuc, ct_nuc, common_cts)
    imp_means = compute_celltype_means(imputed, ct_nuc_imputed, common_cts)

    # Gene correlation
    print("  Gene correlation...")
    gc_per_gene, gc_summary = gene_correlation(imp_means, cell_means, common_cts)
    results['gene_corr'] = gc_summary
    _, gc_nuc = gene_correlation(nuc_means, cell_means, common_cts)
    results['gene_corr_raw_nuc'] = gc_nuc

    # R-squared
    _, r2_summary = r_squared(imp_means, cell_means, common_cts)
    results['r_squared'] = r2_summary

    # Shift scores
    _, shift_summary = shift_scores(imp_means, cell_means, nuc_means, common_cts)
    results['shift'] = shift_summary

    # Subsample for expensive metrics (classification, DE)
    MAX_PER_CT = 2000
    def _subsample(X, cts, max_per_ct=MAX_PER_CT):
        np.random.seed(42)
        idx = []
        for ct in sorted(set(cts)):
            ct_idx = np.where(cts == ct)[0]
            if len(ct_idx) > max_per_ct:
                ct_idx = np.random.choice(ct_idx, max_per_ct, replace=False)
            idx.extend(ct_idx)
        idx = sorted(idx)
        return X[idx], cts[idx]

    X_cell_sub, ct_cell_sub = _subsample(X_cell, ct_cell)
    imp_sub, ct_imp_sub = _subsample(imputed, ct_nuc_imputed)
    nuc_sub, ct_nuc_sub = _subsample(X_nuc, ct_nuc)
    print(f"  Subsampled for clf/DE: {X_cell_sub.shape[0]} cells, "
          f"{imp_sub.shape[0]} imputed, {nuc_sub.shape[0]} nuclei")

    # Classification
    print("  Classification...")
    clf_imp = classification(X_cell_sub, ct_cell_sub, imp_sub, ct_imp_sub, common_cts)
    clf_nuc = classification(X_cell_sub, ct_cell_sub, nuc_sub, ct_nuc_sub, common_cts)
    results['clf_imputed'] = {'accuracy': clf_imp['accuracy'], 'macro_f1': clf_imp['macro_f1']}
    results['clf_raw_nuc'] = {'accuracy': clf_nuc['accuracy'], 'macro_f1': clf_nuc['macro_f1']}

    # DE preservation
    print("  DE preservation...")
    _, de_imp = de_preservation(X_cell_sub, ct_cell_sub, imp_sub, ct_imp_sub, common_cts)
    _, de_nuc = de_preservation(X_cell_sub, ct_cell_sub, nuc_sub, ct_nuc_sub, common_cts)
    results['de_imputed'] = de_imp
    results['de_raw_nuc'] = de_nuc

    # Coexpression
    print("  Coexpression...")
    results['coexp_imputed'] = coexpression(X_cell, imputed)
    results['coexp_raw_nuc'] = coexpression(X_cell, X_nuc)

    # Gene MSE
    results['gene_mse'] = gene_mse(imp_means, cell_means, common_cts)
    del results['gene_mse']['per_gene']  # don't save huge array to JSON

    # ----- Sample-level -----
    print("  Variance ratio...")
    _, vr_summary = variance_ratio(imputed, ct_nuc_imputed, X_cell, ct_cell, common_cts)
    results['variance_ratio'] = vr_summary

    print("  Within-CT Spearman...")
    # Note: imputed and X_nuc must be same cells in same order
    if imputed.shape[0] == X_nuc.shape[0]:
        _, sp_summary = within_ct_spearman(
            imputed, ct_nuc_imputed, X_nuc, ct_nuc, common_cts, n_genes_sample=1000
        )
        results['within_ct_spearman'] = sp_summary
    else:
        print("    SKIPPED — imputed/nucleus cell counts don't match")
        results['within_ct_spearman'] = {'overall_median': None, 'note': 'size_mismatch'}

    print("  Per-cell correlation...")
    _, pc_summary = per_cell_correlation(imputed, ct_nuc_imputed, cell_means, common_cts)
    results['per_cell_corr'] = pc_summary

    # ----- Label-free metrics -----
    print("  Label-free: per-gene QQ correlation...")
    qq = per_gene_correlation_allcells(imputed, X_cell)
    results['qq_gene_corr'] = {'median': qq['median'], 'mean': qq['mean']}

    print("  Label-free: Wasserstein distance...")
    results['wasserstein'] = wasserstein_per_gene(imputed, X_cell)

    # ----- Biological validation -----
    gene_names = adata_nuc.var['gene_names'].values if 'gene_names' in adata_nuc.var else adata_nuc.var_names

    print("  Biology: delta vs ground truth...")
    results['delta_gt'] = delta_vs_ground_truth(
        imputed, X_nuc, X_cell, ct_nuc, ct_cell, gene_names, common_cts
    )

    print("  Biology: compartment scores...")
    results['compartment'] = compartment_scores(imputed, X_nuc, gene_names)

    print("  Biology: CT-specific delta...")
    results['ct_specific'] = ct_specific_delta(
        imputed, X_nuc, ct_nuc, gene_names, common_cts
    )

    return results


def print_summary(results):
    """Print a clean comparison table."""
    name = results['method']
    print(f"\n{'=' * 60}")
    print(f"  {name} — Evaluation Summary")
    print(f"{'=' * 60}")

    rows = [
        ("Gene correlation (CT means)", f"{results['gene_corr']['median']:.3f}",
         f"(raw nuc: {results['gene_corr_raw_nuc']['median']:.3f})"),
        ("R-squared (CT means)", f"{results['r_squared']['median']:.3f}", ""),
        ("Shift score", f"{results['shift']['mean_shift']:.4f}", ""),
        ("Classification accuracy", f"{results['clf_imputed']['accuracy']:.3f}",
         f"(raw nuc: {results['clf_raw_nuc']['accuracy']:.3f})"),
        ("DE overlap (top 100)", f"{results['de_imputed']['mean_overlap']:.3f}",
         f"(raw nuc: {results['de_raw_nuc']['mean_overlap']:.3f})"),
        ("Coexpression", f"{results['coexp_imputed']['spearman_r']:.3f}",
         f"(raw nuc: {results['coexp_raw_nuc']['spearman_r']:.3f})"),
        ("Gene MSE (CT means)", f"{results['gene_mse']['mean']:.4f}", ""),
        ("", "", ""),
        ("Variance ratio", f"{results['variance_ratio']['median_ratio']:.3f}",
         "(1.0 = perfect, <<1 = collapse)"),
        ("Within-CT Spearman",
         f"{results['within_ct_spearman']['overall_median']:.3f}"
         if results['within_ct_spearman'].get('overall_median') is not None else "N/A",
         "(1.0 = perfect ranking)"),
        ("Per-cell corr to CT mean", f"{results['per_cell_corr']['overall_median']:.3f}", ""),
        ("", "", ""),
        ("QQ gene corr (label-free)", f"{results['qq_gene_corr']['median']:.3f}", ""),
        ("Wasserstein (label-free)", f"{results['wasserstein']['median']:.4f}", ""),
        ("", "", ""),
        ("Delta vs ground truth (r)", f"{results['delta_gt']['overall_corr']:.3f}",
         f"(median per-CT: {results['delta_gt']['median_ct_corr']:.3f})"),
        ("Delta vs ground truth (R2)", f"{results['delta_gt']['overall_r2']:.3f}", ""),
        ("Ribosomal genes: correct sign",
         f"{results['compartment']['ribosomal'].get('frac_correct_sign', 0):.0%}",
         f"(mean delta: {results['compartment']['ribosomal'].get('mean_delta', 0):+.2f})"),
        ("Cytoplasmic genes: correct sign",
         f"{results['compartment']['cytoplasmic'].get('frac_correct_sign', 0):.0%}",
         f"(mean delta: {results['compartment']['cytoplasmic'].get('mean_delta', 0):+.2f})"),
        ("Nuclear lncRNA: correct sign",
         f"{results['compartment']['nuclear_lncrna'].get('frac_correct_sign', 0):.0%}",
         f"(mean delta: {results['compartment']['nuclear_lncrna'].get('mean_delta', 0):+.2f})"),
        ("Neuronal markers: neuron > glia",
         f"{'YES' if results['ct_specific'].get('neuron_gt_glia') else 'NO'}",
         f"(ratio: {results['ct_specific'].get('ratio', 0):.2f})"
         if results['ct_specific'].get('ratio') else ""),
    ]

    for label, value, note in rows:
        if label == "":
            print(f"  {'—' * 50}")
        else:
            print(f"  {label:<35s} {value:>8s}  {note}")

    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline imputation method")
    parser.add_argument("--imputed_npy", type=str, required=True,
                        help="Path to .npy file with imputed expression [N_nuc, G]")
    parser.add_argument("--obs_csv", type=str, default=None,
                        help="CSV with 'cell_type' column for imputed nuclei")
    parser.add_argument("--name", type=str, default="baseline",
                        help="Method name for display")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save results JSON (default: same as imputed_npy)")
    parser.add_argument("--gene_list", type=str, default=DEFAULT_GENE_LIST)
    args = parser.parse_args()

    # Load imputed matrix
    print(f"Loading imputed matrix: {args.imputed_npy}")
    imputed = np.load(args.imputed_npy).astype(np.float32)
    print(f"  Shape: {imputed.shape}")

    # Load cell type labels
    if args.obs_csv:
        obs = pd.read_csv(args.obs_csv, index_col=0)
        ct_nuc_imputed = obs['cell_type'].values.astype(str)
    else:
        # If no obs provided, load from raw nucleus data
        print("  No --obs_csv provided, loading CT labels from raw nucleus data")
        ct_nuc_imputed = None  # will be set after loading

    # Load raw data
    print("Loading raw data...")
    adata_cell, adata_nuc = load_raw_data(args.gene_list)

    if ct_nuc_imputed is None:
        ct_nuc_imputed = adata_nuc.obs['cell_type'].values.astype(str)

    # Verify shapes
    assert imputed.shape[1] == adata_cell.shape[1], (
        f"Gene count mismatch: imputed {imputed.shape[1]} vs cell {adata_cell.shape[1]}"
    )

    # Run evaluation
    results = evaluate(imputed, adata_cell, adata_nuc, ct_nuc_imputed, name=args.name)
    print_summary(results)

    # Save results
    out_dir = args.output_dir or os.path.dirname(args.imputed_npy)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"eval_{args.name}.json")

    # Make JSON-serializable
    serializable = {}
    for k, v in results.items():
        if isinstance(v, dict):
            serializable[k] = {kk: vv for kk, vv in v.items()
                              if not isinstance(vv, np.ndarray)}
        else:
            serializable[k] = v

    with open(out_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
