#!/usr/bin/env python
"""
JASPAR mouse transcription-factor contrast analysis for ImputeVAE validation.

Tests whether ImputeVAE creates the same strong neuronal > non-neuronal shift
for TF genes as it does for SynGO synaptic genes, or whether the SynGO pattern
is more specific to synaptic biology.

JASPAR TFs are a nuclear-function contrast set, not a perfect negative control.
TF mRNAs can still be cytoplasmic, but encoded TF proteins are nuclear/regulatory
rather than synaptic.

Usage:
    python -u validation/tf/scripts/jaspar_tf_validation.py

Requires conda env: impute2
"""

import os
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats, sparse
import anndata as ad

warnings.filterwarnings('ignore', category=UserWarning)

SEED = 42
N_BOOT = 1000
CI_LO, CI_HI = 2.5, 97.5

# ============================================================
# Paths
# ============================================================
PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
DATA_ROOT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi"

GENE_LIST_PATH = os.path.join(PROJECT, "data/training_gene_list_v3.csv")
SPLIT_MANIFEST = os.path.join(PROJECT, "data/split_manifest.tsv")
COMMON_CT_PATH = os.path.join(PROJECT, "data/common_cell_types.csv")
COMMON_GENES_PATH = os.path.join(PROJECT, "data/common_genes.csv")
JASPAR_TSV = os.path.join(PROJECT, "validation/tf/external/ultimate_metadata_table_CORE_2026.tsv")
IMPUTED_NPY = os.path.join(PROJECT, "run/eval_v3_splitaware/cvae_imputed_valid.npy")
IMPUTED_OBS = os.path.join(PROJECT, "run/eval_v3_splitaware/valid_nuc_obs.csv")

CHAR_DIR = os.path.join(PROJECT, "validation/tf/jaspar_characterization")
OUT_DIR = os.path.join(PROJECT, "validation/tf/tf_brain_validation")
COMP_DIR = os.path.join(PROJECT, "validation/tf/syngo_vs_tf")
SYNGO_DIR = os.path.join(PROJECT, "validation/syngo/syngo_brain_validation")

NEURONAL = ['L2/3 IT', 'L5 ET', 'L5 IT', 'L5/6 NP', 'L6 CT', 'L6 IT', 'L6b',
            'Lamp5', 'Pvalb', 'Sncg', 'Sst', 'Vip']
NON_NEURONAL = ['Astro', 'Endo', 'Macrophage', 'OPC', 'Oligo']

LOW_SAMPLE_THRESHOLD = 20

SENSITIVITY_SETTINGS = {
    'all': {'exclude_cts': []},
    'exclude_l5et': {'exclude_cts': ['L5 ET']},
    'exclude_low_sample': {'exclude_cts': []},  # filled dynamically
    'exclude_l5et_and_low_sample': {'exclude_cts': ['L5 ET']},  # + low sample
}


# ============================================================
# Step 1: JASPAR characterization
# ============================================================

def characterize_jaspar():
    """Load and characterize JASPAR CORE 2026 mouse TFs."""
    print("=" * 70)
    print("Step 1: JASPAR characterization")
    print("=" * 70)

    jaspar = pd.read_csv(JASPAR_TSV, sep="\t", dtype=str)
    print(f"  JASPAR CORE total rows: {len(jaspar)}")

    # Human reference count
    human = jaspar[(jaspar['collection'] == 'CORE') & (jaspar['tax_id'] == '9606')]
    print(f"  Human CORE rows (reference only): {len(human)}")

    # Mouse primary filter
    mouse_tf = jaspar[
        (jaspar['collection'] == 'CORE') &
        (jaspar['tax_id'] == '10090') &
        (jaspar['species'] == 'Mus musculus')
    ].copy()
    print(f"  Mouse CORE rows (primary): {len(mouse_tf)}")
    print(f"  Mouse unique motif names: {mouse_tf['name'].nunique()}")
    print(f"  Mouse unique base IDs: {mouse_tf['base_id'].nunique()}")

    # Singleton vs complex
    is_complex = mouse_tf['name'].str.contains('::', regex=False)
    singleton_rows = mouse_tf[~is_complex]
    complex_rows = mouse_tf[is_complex]
    print(f"  Mouse singleton motif rows: {len(singleton_rows)}")
    print(f"  Mouse complex/dimer motif rows: {len(complex_rows)}")

    # Extract all component symbols
    all_components = set()
    singleton_symbols = set()
    complex_contributed = set()

    for name in mouse_tf['name'].unique():
        parts = name.split('::')
        for p in parts:
            all_components.add(p)
        if '::' not in name:
            singleton_symbols.add(name)
        else:
            for p in parts:
                complex_contributed.add(p)

    # Symbols only from complexes (not also singleton)
    complex_only = complex_contributed - singleton_symbols

    print(f"  Unique component TF symbols after :: split: {len(all_components)}")
    print(f"  Unique singleton-only TF symbols: {len(singleton_symbols)}")
    print(f"  Symbols contributed only by complex motifs: {len(complex_only)}")

    # Save characterization CSVs
    mouse_tf.to_csv(os.path.join(CHAR_DIR, "jaspar_core_2026_mouse_rows.csv"), index=False)
    complex_rows.to_csv(os.path.join(CHAR_DIR, "jaspar_core_2026_mouse_complex_rows.csv"), index=False)

    # All component symbols with source info
    comp_records = []
    for name in mouse_tf['name'].unique():
        parts = name.split('::')
        for p in parts:
            comp_records.append({
                'tf_symbol': p,
                'motif_name': name,
                'is_complex': '::' in name,
                'is_singleton': '::' not in name,
            })
    comp_df = pd.DataFrame(comp_records)
    comp_df.to_csv(os.path.join(CHAR_DIR, "jaspar_core_2026_mouse_tf_components.csv"), index=False)

    # Model gene overlap
    gene_list = pd.read_csv(GENE_LIST_PATH, header=None)[0].tolist()
    gene_set = set(gene_list)
    gene_lower = {g.lower(): g for g in gene_list}
    print(f"\n  Model gene count: {len(gene_list)}")

    # Map TF symbols to model genes
    tf_primary = sorted(all_components)
    tf_singleton = sorted(singleton_symbols)

    def map_symbol(sym):
        if sym in gene_set:
            return sym, 'exact_model_match'
        lower = sym.lower()
        if lower in gene_lower:
            return gene_lower[lower], 'case_insensitive_model_match'
        mouse = sym[0].upper() + sym[1:].lower() if len(sym) > 1 else sym.upper()
        if mouse in gene_set:
            return mouse, 'mouse_style_fallback'
        return None, 'not_in_model'

    primary_records = []
    for sym in tf_primary:
        model_gene, method = map_symbol(sym)
        is_ribo = (model_gene or '').startswith(('Rpl', 'Rps'))
        primary_records.append({
            'jaspar_symbol': sym,
            'model_gene': model_gene,
            'mapping_method': method,
            'in_model': model_gene is not None,
            'is_ribosomal': is_ribo,
            'in_singleton_set': sym in singleton_symbols,
        })
    primary_df = pd.DataFrame(primary_records)

    # Count mapping methods
    in_model = primary_df[primary_df['in_model']]
    n_exact = (in_model['mapping_method'] == 'exact_model_match').sum()
    n_case = (in_model['mapping_method'] == 'case_insensitive_model_match').sum()
    n_fallback = (in_model['mapping_method'] == 'mouse_style_fallback').sum()
    n_ribo = primary_df['is_ribosomal'].sum()
    n_primary_nonribo = len(in_model) - in_model['is_ribosomal'].sum()

    # Singleton non-ribosomal
    singleton_df = primary_df[primary_df['in_singleton_set'] & primary_df['in_model']]
    n_singleton_nonribo = len(singleton_df) - singleton_df['is_ribosomal'].sum()

    print(f"  Matched exact model symbol: {n_exact}")
    print(f"  Matched case-insensitive: {n_case}")
    print(f"  Matched mouse-style fallback: {n_fallback}")
    print(f"  In model total: {len(in_model)}")
    print(f"  Ribosomal Rpl/Rps: {n_ribo}")
    print(f"  TF primary non-ribosomal: {n_primary_nonribo}")
    print(f"  TF singleton non-ribosomal: {n_singleton_nonribo}")

    primary_df.to_csv(os.path.join(CHAR_DIR, "jaspar_core_2026_mouse_tf_primary_model_genes.csv"), index=False)

    # TF symbol lists
    pd.DataFrame({'tf_symbol': tf_primary}).to_csv(
        os.path.join(CHAR_DIR, "jaspar_core_2026_mouse_tf_symbols.csv"), index=False)
    pd.DataFrame({'tf_symbol': tf_singleton}).to_csv(
        os.path.join(CHAR_DIR, "jaspar_core_2026_mouse_tf_singleton_symbols.csv"), index=False)

    # Filtering counts
    filtering = pd.DataFrame([
        {'stage': 'jaspar_core_total_rows', 'count': len(jaspar)},
        {'stage': 'human_core_rows_reference_only', 'count': len(human)},
        {'stage': 'mouse_core_rows_primary', 'count': len(mouse_tf)},
        {'stage': 'mouse_unique_motif_names', 'count': mouse_tf['name'].nunique()},
        {'stage': 'mouse_unique_base_ids', 'count': mouse_tf['base_id'].nunique()},
        {'stage': 'mouse_singleton_motif_rows', 'count': len(singleton_rows)},
        {'stage': 'mouse_complex_motif_rows', 'count': len(complex_rows)},
        {'stage': 'unique_component_symbols_after_complex_split', 'count': len(all_components)},
        {'stage': 'unique_singleton_symbols', 'count': len(singleton_symbols)},
        {'stage': 'unique_symbols_contributed_by_complex_motifs', 'count': len(complex_only)},
        {'stage': 'model_gene_count', 'count': len(gene_list)},
        {'stage': 'matched_exact_model_symbol', 'count': n_exact},
        {'stage': 'matched_case_insensitive_model_symbol', 'count': n_case},
        {'stage': 'matched_mouse_style_fallback', 'count': n_fallback},
        {'stage': 'in_model', 'count': len(in_model)},
        {'stage': 'ribosomal_Rpl_Rps', 'count': n_ribo},
        {'stage': 'tf_primary_non_ribosomal', 'count': n_primary_nonribo},
        {'stage': 'tf_singleton_non_ribosomal', 'count': n_singleton_nonribo},
    ])
    filtering.to_csv(os.path.join(CHAR_DIR, "jaspar_core_2026_mouse_tf_filtering_counts.csv"), index=False)

    # Class summary
    class_summary = mouse_tf.groupby('class').agg(
        n_motifs=('matrix_id', 'count'),
        n_unique_names=('name', 'nunique'),
    ).reset_index().sort_values('n_motifs', ascending=False)
    class_summary.to_csv(os.path.join(CHAR_DIR, "jaspar_core_2026_mouse_tf_class_summary.csv"), index=False)

    # Family summary
    family_summary = mouse_tf.groupby('family').agg(
        n_motifs=('matrix_id', 'count'),
        n_unique_names=('name', 'nunique'),
    ).reset_index().sort_values('n_motifs', ascending=False)
    family_summary.to_csv(os.path.join(CHAR_DIR, "jaspar_core_2026_mouse_tf_family_summary.csv"), index=False)

    # Get final gene lists
    tf_primary_genes = sorted(in_model[~in_model['is_ribosomal']]['model_gene'].tolist())
    tf_singleton_genes = sorted(
        singleton_df[~singleton_df['is_ribosomal']]['model_gene'].tolist())

    print(f"\n  Final TF primary genes: {len(tf_primary_genes)}")
    print(f"  Final TF singleton genes: {len(tf_singleton_genes)}")

    return tf_primary_genes, tf_singleton_genes, primary_df, filtering


# ============================================================
# Step 2: Load validation data
# ============================================================

def load_data():
    """Load all required matrices with validation-only subsetting."""
    print("\n" + "=" * 70)
    print("Step 2: Loading data")
    print("=" * 70)

    gene_list = pd.read_csv(GENE_LIST_PATH, header=None)[0].tolist()
    manifest = pd.read_csv(SPLIT_MANIFEST, sep='\t')
    common_ct = pd.read_csv(COMMON_CT_PATH)['common_cell_types'].tolist()
    common_genes = pd.read_csv(COMMON_GENES_PATH)['common_genes'].tolist()

    print(f"  Gene list: {len(gene_list)} genes")
    print(f"  Common cell types: {len(common_ct)}")

    def get_indices(modality):
        mod = manifest[manifest.modality == modality].sort_values('obs_id').reset_index(drop=True)
        valid_mask = mod.partition == 'valid'
        return np.where(valid_mask)[0], mod[valid_mask].reset_index(drop=True)

    valid_nuc_idx, valid_nuc_meta = get_indices('nuclei')
    valid_cell_idx, valid_cell_meta = get_indices('cell')
    print(f"  Valid nuclei: {len(valid_nuc_idx)}, Valid cells: {len(valid_cell_idx)}")

    def load_h5ad(data_dir, batch):
        path = os.path.join(DATA_ROOT, data_dir, "data.h5ad")
        print(f"  Loading {path}...")
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
        adata.X = np.log1p(X).astype(np.float32)
        gene_names = adata.var['gene_names'].values.tolist()
        print(f"    Shape after filtering: {adata.shape}")
        return adata, gene_names

    cell_adata, gn_cell = load_h5ad("10X_cells_v2_v3_AIBS", "c3A")
    nuc_adata, gn_nuc = load_h5ad("10X_nuclei_v2_v3_AIBS", "n3A")
    assert gn_cell == gn_nuc, "Gene names mismatch"
    gene_names = gn_cell
    print(f"  Gene alignment verified: {len(gene_names)} genes")

    valid_cell_X = cell_adata.X[valid_cell_idx]
    valid_cell_ct = cell_adata.obs.iloc[valid_cell_idx]['cell_type'].values
    valid_nuc_X = nuc_adata.X[valid_nuc_idx]
    valid_nuc_ct = nuc_adata.obs.iloc[valid_nuc_idx]['cell_type'].values

    del cell_adata, nuc_adata

    print(f"  Loading imputed: {IMPUTED_NPY}")
    imputed_X = np.load(IMPUTED_NPY)
    imputed_obs = pd.read_csv(IMPUTED_OBS)
    imputed_ct = imputed_obs['cell_type'].values

    print(f"    Imputed shape: {imputed_X.shape}")
    assert imputed_X.shape[0] == valid_nuc_X.shape[0]
    assert imputed_X.shape[1] == len(gene_names)
    ct_match = (imputed_ct == valid_nuc_ct).mean()
    print(f"    CT alignment: {ct_match*100:.1f}%")
    assert ct_match > 0.99

    print(f"    Cell range: [{valid_cell_X.min():.2f}, {valid_cell_X.max():.2f}]")
    print(f"    Nuc range: [{valid_nuc_X.min():.2f}, {valid_nuc_X.max():.2f}]")
    print(f"    Imputed range: [{imputed_X.min():.2f}, {imputed_X.max():.2f}]")

    return (gene_names, valid_cell_X, valid_cell_ct,
            valid_nuc_X, valid_nuc_ct, imputed_X, imputed_ct)


# ============================================================
# Step 3: Mean-shift analysis
# ============================================================

def compute_mean_shift(tf_genes, gene_names, valid_cell_X, valid_cell_ct,
                       valid_nuc_X, valid_nuc_ct, imputed_X, imputed_ct):
    """Compute mean-shift deltas for TF genes."""
    print("\n" + "=" * 70)
    print("Step 3: Mean-shift analysis")
    print("=" * 70)

    gene_idx_map = {g: i for i, g in enumerate(gene_names)}
    tf_indices = [gene_idx_map[g] for g in tf_genes if g in gene_idx_map]
    tf_in_model = [gene_names[i] for i in tf_indices]
    print(f"  TF genes in model gene list: {len(tf_in_model)}")

    cell_types = sorted(set(valid_cell_ct) & set(valid_nuc_ct) &
                        set(NEURONAL + NON_NEURONAL))
    print(f"  Cell types: {len(cell_types)}")

    records = []
    for ct in cell_types:
        cell_mask = valid_cell_ct == ct
        nuc_mask = valid_nuc_ct == ct
        imp_mask = imputed_ct == ct

        n_cell = cell_mask.sum()
        n_nuc = nuc_mask.sum()

        cell_mean = valid_cell_X[cell_mask][:, tf_indices].mean(axis=0)
        nuc_mean = valid_nuc_X[nuc_mask][:, tf_indices].mean(axis=0)
        imp_mean = imputed_X[imp_mask][:, tf_indices].mean(axis=0)

        neuronal_status = 'neuronal' if ct in NEURONAL else 'non-neuronal'

        for j, gene in enumerate(tf_in_model):
            emp_delta = float(cell_mean[j] - nuc_mean[j])
            pred_delta = float(imp_mean[j] - nuc_mean[j])
            records.append({
                'gene': gene,
                'cell_type': ct,
                'neuronal_status': neuronal_status,
                'empirical_delta': emp_delta,
                'predicted_delta': pred_delta,
                'cell_mean': float(cell_mean[j]),
                'nucleus_mean': float(nuc_mean[j]),
                'imputed_mean': float(imp_mean[j]),
                'n_cell': n_cell,
                'n_nucleus': n_nuc,
            })

    deltas_df = pd.DataFrame(records)
    print(f"  Gene x CT pairs: {len(deltas_df)}")
    return deltas_df, cell_types, tf_in_model


def compute_quality_metrics(deltas_df):
    """Compute overall and per-CT quality metrics."""
    emp = deltas_df['empirical_delta'].values
    pred = deltas_df['predicted_delta'].values

    def metrics(e, p):
        n = len(e)
        pearson_r = np.corrcoef(e, p)[0, 1]
        spearman_r = stats.spearmanr(e, p).statistic
        ss_res = np.sum((p - e) ** 2)
        ss_tot = np.sum((e - np.mean(e)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        mae = np.mean(np.abs(p - e))
        rmse = np.sqrt(np.mean((p - e) ** 2))
        sign_agree = np.mean(np.sign(e) == np.sign(p))
        slope, intercept = np.polyfit(e, p, 1)
        sign_pp = np.sum((e > 0) & (p > 0))
        sign_nn = np.sum((e < 0) & (p < 0))
        sign_pn = np.sum((e > 0) & (p < 0))
        sign_np = np.sum((e < 0) & (p > 0))
        return {
            'pearson_r': pearson_r, 'spearman_r': spearman_r, 'R2': r2,
            'MAE': mae, 'RMSE': rmse, 'sign_agreement': sign_agree,
            'slope': slope, 'intercept': intercept, 'n': n,
            'sign_pp': sign_pp, 'sign_nn': sign_nn,
            'sign_pn': sign_pn, 'sign_np': sign_np,
        }

    overall = metrics(emp, pred)
    overall['label'] = 'all'
    overall['n_genes'] = deltas_df['gene'].nunique()
    overall['n_cell_types'] = deltas_df['cell_type'].nunique()
    overall['n_observations'] = len(deltas_df)

    return overall


def compute_neuronal_nonneuronal(deltas_df, rng):
    """Compute neuronal vs non-neuronal with bootstrap CIs."""
    ct_means = deltas_df.groupby('cell_type')[['empirical_delta', 'predicted_delta']].mean()
    ct_means['neuronal_status'] = ct_means.index.map(
        lambda x: 'neuronal' if x in NEURONAL else 'non-neuronal')

    neuro_ct = ct_means[ct_means['neuronal_status'] == 'neuronal']
    non_neuro_ct = ct_means[ct_means['neuronal_status'] == 'non-neuronal']

    emp_neuronal = neuro_ct['empirical_delta'].mean()
    emp_non_neuronal = non_neuro_ct['empirical_delta'].mean()
    pred_neuronal = neuro_ct['predicted_delta'].mean()
    pred_non_neuronal = non_neuro_ct['predicted_delta'].mean()

    boot_emp_diff, boot_pred_diff = [], []
    for _ in range(N_BOOT):
        n_idx = rng.choice(len(neuro_ct), size=len(neuro_ct), replace=True)
        nn_idx = rng.choice(len(non_neuro_ct), size=len(non_neuro_ct), replace=True)
        boot_emp_diff.append(
            neuro_ct['empirical_delta'].values[n_idx].mean() -
            non_neuro_ct['empirical_delta'].values[nn_idx].mean())
        boot_pred_diff.append(
            neuro_ct['predicted_delta'].values[n_idx].mean() -
            non_neuro_ct['predicted_delta'].values[nn_idx].mean())

    boot_emp_diff = np.array(boot_emp_diff)
    boot_pred_diff = np.array(boot_pred_diff)

    return {
        'analysis': 'ct_level_equal_weight',
        'emp_neuronal': emp_neuronal,
        'emp_non_neuronal': emp_non_neuronal,
        'emp_diff': emp_neuronal - emp_non_neuronal,
        'pred_neuronal': pred_neuronal,
        'pred_non_neuronal': pred_non_neuronal,
        'pred_diff': pred_neuronal - pred_non_neuronal,
        'emp_diff_ci_lo': np.percentile(boot_emp_diff, CI_LO),
        'emp_diff_ci_hi': np.percentile(boot_emp_diff, CI_HI),
        'pred_diff_ci_lo': np.percentile(boot_pred_diff, CI_LO),
        'pred_diff_ci_hi': np.percentile(boot_pred_diff, CI_HI),
        'n_boot': N_BOOT,
        'n_neuronal_cts': len(neuro_ct),
        'n_non_neuronal_cts': len(non_neuro_ct),
    }


def compute_celltype_summary(deltas_df, rng):
    """Per-cell-type summary with CIs."""
    rows = []
    for ct in sorted(deltas_df['cell_type'].unique()):
        sub = deltas_df[deltas_df['cell_type'] == ct]
        emp = sub['empirical_delta'].values
        pred = sub['predicted_delta'].values
        err = pred - emp
        n_cell = sub['n_cell'].iloc[0]
        n_nuc = sub['n_nucleus'].iloc[0]
        low_sample = n_cell < LOW_SAMPLE_THRESHOLD or n_nuc < LOW_SAMPLE_THRESHOLD
        neuronal_status = 'neuronal' if ct in NEURONAL else 'non-neuronal'

        # Bootstrap CIs on mean shift (resample genes)
        boot_emp, boot_pred = [], []
        for _ in range(N_BOOT):
            idx = rng.choice(len(emp), size=len(emp), replace=True)
            boot_emp.append(emp[idx].mean())
            boot_pred.append(pred[idx].mean())
        boot_emp = np.array(boot_emp)
        boot_pred = np.array(boot_pred)

        rows.append({
            'cell_type': ct,
            'neuronal_status': neuronal_status,
            'n_cell': n_cell,
            'n_nucleus': n_nuc,
            'low_sample': low_sample,
            'empirical_mean_shift': emp.mean(),
            'predicted_mean_shift': pred.mean(),
            'prediction_error': err.mean(),
            'emp_ci_lo': np.percentile(boot_emp, CI_LO),
            'emp_ci_hi': np.percentile(boot_emp, CI_HI),
            'pred_ci_lo': np.percentile(boot_pred, CI_LO),
            'pred_ci_hi': np.percentile(boot_pred, CI_HI),
        })

    return pd.DataFrame(rows)


# ============================================================
# Step 4: Distribution-level analysis (Wasserstein)
# ============================================================

def compute_distribution_metrics(tf_genes, gene_names, valid_cell_X, valid_cell_ct,
                                 valid_nuc_X, valid_nuc_ct, imputed_X, imputed_ct):
    """Compute per-gene per-CT Wasserstein distances."""
    print("\n" + "=" * 70)
    print("Step 4: Distribution-level analysis (Wasserstein)")
    print("=" * 70)

    gene_idx_map = {g: i for i, g in enumerate(gene_names)}
    cell_types = sorted(set(valid_cell_ct) & set(valid_nuc_ct) &
                        set(NEURONAL + NON_NEURONAL))

    records = []
    for ct_i, ct in enumerate(cell_types):
        cell_mask = valid_cell_ct == ct
        nuc_mask = valid_nuc_ct == ct
        imp_mask = imputed_ct == ct

        n_cell = cell_mask.sum()
        n_nuc = nuc_mask.sum()
        low_sample = n_cell < LOW_SAMPLE_THRESHOLD or n_nuc < LOW_SAMPLE_THRESHOLD
        neuronal_status = 'neuronal' if ct in NEURONAL else 'non-neuronal'

        cell_data = valid_cell_X[cell_mask]
        nuc_data = valid_nuc_X[nuc_mask]
        imp_data = imputed_X[imp_mask]

        for gene in tf_genes:
            gi = gene_idx_map.get(gene)
            if gi is None:
                continue

            raw_nuc = nuc_data[:, gi]
            raw_cell = cell_data[:, gi]
            imp_cell = imp_data[:, gi]

            W_raw = stats.wasserstein_distance(raw_nuc, raw_cell)
            W_imputed = stats.wasserstein_distance(imp_cell, raw_cell)
            abs_imp = W_raw - W_imputed
            norm_imp = (1 - W_imputed / W_raw) if W_raw >= 0.01 else np.nan

            records.append({
                'gene': gene,
                'cell_type': ct,
                'neuronal_status': neuronal_status,
                'low_sample': low_sample,
                'n_cell': n_cell,
                'n_nucleus': n_nuc,
                'W_raw': W_raw,
                'W_imputed': W_imputed,
                'absolute_improvement': abs_imp,
                'normalized_improvement': norm_imp,
                'norm_eligible': W_raw >= 0.01,
            })

        if (ct_i + 1) % 5 == 0:
            print(f"  Processed {ct_i + 1}/{len(cell_types)} cell types")

    print(f"  Total gene-CT pairs: {len(records)}")
    return pd.DataFrame(records)


def summarize_distribution(dist_df):
    """Summarize Wasserstein by cell type and group."""
    # By cell type
    ct_rows = []
    for ct in sorted(dist_df['cell_type'].unique()):
        sub = dist_df[dist_df['cell_type'] == ct]
        abs_imp = sub['absolute_improvement'].values

        # Bootstrap CI for median absolute improvement
        boot_medians = []
        for _ in range(N_BOOT):
            idx = np.random.choice(len(abs_imp), size=len(abs_imp), replace=True)
            boot_medians.append(np.median(abs_imp[idx]))
        boot_medians = np.array(boot_medians)

        ct_rows.append({
            'cell_type': ct,
            'neuronal_status': sub['neuronal_status'].iloc[0],
            'low_sample': sub['low_sample'].iloc[0],
            'n_cell': sub['n_cell'].iloc[0],
            'n_nucleus': sub['n_nucleus'].iloc[0],
            'n_genes': sub['gene'].nunique(),
            'median_W_raw': np.median(sub['W_raw']),
            'median_W_imputed': np.median(sub['W_imputed']),
            'median_absolute_improvement': np.median(abs_imp),
            'abs_imp_ci_lo': np.percentile(boot_medians, CI_LO),
            'abs_imp_ci_hi': np.percentile(boot_medians, CI_HI),
            'proportion_improved_W': np.mean(abs_imp > 0),
        })
    ct_df = pd.DataFrame(ct_rows)

    # By group
    group_rows = []
    for label, mask_fn in [
        ('all', lambda d: d),
        ('neuronal', lambda d: d[d['neuronal_status'] == 'neuronal']),
        ('non_neuronal', lambda d: d[d['neuronal_status'] == 'non-neuronal']),
        ('low_sample', lambda d: d[d['low_sample'] == True]),
        ('excluding_low_sample', lambda d: d[d['low_sample'] == False]),
    ]:
        sub = mask_fn(dist_df)
        if len(sub) == 0:
            continue
        abs_imp = sub['absolute_improvement'].values
        group_rows.append({
            'group': label,
            'n_gene_ct_pairs': len(sub),
            'median_W_raw': np.median(sub['W_raw']),
            'median_W_imputed': np.median(sub['W_imputed']),
            'median_absolute_improvement': np.median(abs_imp),
            'proportion_improved_W': np.mean(abs_imp > 0),
            'mean_absolute_improvement': np.mean(abs_imp),
        })
    group_df = pd.DataFrame(group_rows)

    return ct_df, group_df


# ============================================================
# Step 5: Sensitivity analyses
# ============================================================

def run_sensitivity(deltas_df, dist_df, ct_summary_df):
    """Run sensitivity analyses across cell-type exclusion settings."""
    print("\n" + "=" * 70)
    print("Step 5: Sensitivity analyses")
    print("=" * 70)

    rng = np.random.default_rng(SEED)

    low_cts = ct_summary_df[ct_summary_df['low_sample']]['cell_type'].tolist()
    print(f"  Low-sample CTs: {low_cts}")
    SENSITIVITY_SETTINGS['exclude_low_sample']['exclude_cts'] = low_cts
    SENSITIVITY_SETTINGS['exclude_l5et_and_low_sample']['exclude_cts'] = list(
        set(['L5 ET'] + low_cts))

    ms_rows = []
    neuro_rows = []
    dist_rows = []

    for setting_name, config in SENSITIVITY_SETTINGS.items():
        exclude = config['exclude_cts']
        df = deltas_df[~deltas_df['cell_type'].isin(exclude)]
        ddf = dist_df[~dist_df['cell_type'].isin(exclude)]

        # Mean-shift metrics
        emp = df['empirical_delta'].values
        pred = df['predicted_delta'].values
        pearson_r = np.corrcoef(emp, pred)[0, 1]
        spearman_r = stats.spearmanr(emp, pred).statistic
        ss_res = np.sum((pred - emp) ** 2)
        ss_tot = np.sum((emp - np.mean(emp)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        mae = np.mean(np.abs(pred - emp))
        rmse = np.sqrt(np.mean((pred - emp) ** 2))
        sign_agree = np.mean(np.sign(emp) == np.sign(pred))
        slope, intercept = np.polyfit(emp, pred, 1)

        ms_rows.append({
            'setting': setting_name,
            'excluded_cts': '; '.join(sorted(exclude)) if exclude else 'none',
            'n_genes': df['gene'].nunique(),
            'n_cell_types': df['cell_type'].nunique(),
            'n_gene_celltype_pairs': len(df),
            'pearson': pearson_r,
            'spearman': spearman_r,
            'R2': r2,
            'MAE': mae,
            'RMSE': rmse,
            'sign_agreement': sign_agree,
            'slope': slope,
            'intercept': intercept,
        })

        # Neuronal vs non-neuronal
        ct_means = df.groupby('cell_type')[['empirical_delta', 'predicted_delta']].mean()
        ct_means['neuronal_status'] = ct_means.index.map(
            lambda x: 'neuronal' if x in NEURONAL else 'non-neuronal')
        neuro_ct = ct_means[ct_means['neuronal_status'] == 'neuronal']
        non_neuro_ct = ct_means[ct_means['neuronal_status'] == 'non-neuronal']

        emp_neuronal = neuro_ct['empirical_delta'].mean()
        emp_non_neuronal = non_neuro_ct['empirical_delta'].mean()
        pred_neuronal = neuro_ct['predicted_delta'].mean()
        pred_non_neuronal = non_neuro_ct['predicted_delta'].mean()

        boot_emp_diff, boot_pred_diff = [], []
        for _ in range(N_BOOT):
            n_idx = rng.choice(len(neuro_ct), size=len(neuro_ct), replace=True)
            nn_idx = rng.choice(len(non_neuro_ct), size=len(non_neuro_ct), replace=True)
            boot_emp_diff.append(
                neuro_ct['empirical_delta'].values[n_idx].mean() -
                non_neuro_ct['empirical_delta'].values[nn_idx].mean())
            boot_pred_diff.append(
                neuro_ct['predicted_delta'].values[n_idx].mean() -
                non_neuro_ct['predicted_delta'].values[nn_idx].mean())
        boot_emp_diff = np.array(boot_emp_diff)
        boot_pred_diff = np.array(boot_pred_diff)

        neuro_rows.append({
            'setting': setting_name,
            'excluded_cts': '; '.join(sorted(exclude)) if exclude else 'none',
            'n_neuronal_cts': len(neuro_ct),
            'n_non_neuronal_cts': len(non_neuro_ct),
            'emp_neuronal': emp_neuronal,
            'emp_non_neuronal': emp_non_neuronal,
            'emp_diff': emp_neuronal - emp_non_neuronal,
            'emp_diff_ci_lo': np.percentile(boot_emp_diff, CI_LO),
            'emp_diff_ci_hi': np.percentile(boot_emp_diff, CI_HI),
            'pred_neuronal': pred_neuronal,
            'pred_non_neuronal': pred_non_neuronal,
            'pred_diff': pred_neuronal - pred_non_neuronal,
            'pred_diff_ci_lo': np.percentile(boot_pred_diff, CI_LO),
            'pred_diff_ci_hi': np.percentile(boot_pred_diff, CI_HI),
        })

        # Distribution metrics
        neuro_dist = ddf[ddf['neuronal_status'] == 'neuronal']
        non_neuro_dist = ddf[ddf['neuronal_status'] == 'non-neuronal']

        dist_rows.append({
            'setting': setting_name,
            'excluded_cts': '; '.join(sorted(exclude)) if exclude else 'none',
            'n_genes': ddf['gene'].nunique(),
            'n_cell_types': ddf['cell_type'].nunique(),
            'n_gene_celltype_pairs': len(ddf),
            'median_W_raw': np.median(ddf['W_raw']),
            'median_W_imputed': np.median(ddf['W_imputed']),
            'median_absolute_improvement': np.median(ddf['absolute_improvement']),
            'mean_absolute_improvement': np.mean(ddf['absolute_improvement']),
            'proportion_improved_W': np.mean(ddf['absolute_improvement'] > 0),
            'neuronal_median_abs_imp': np.median(neuro_dist['absolute_improvement']),
            'neuronal_proportion_improved': np.mean(neuro_dist['absolute_improvement'] > 0),
            'non_neuronal_median_abs_imp': np.median(non_neuro_dist['absolute_improvement']),
            'non_neuronal_proportion_improved': np.mean(non_neuro_dist['absolute_improvement'] > 0),
        })

    ms_df = pd.DataFrame(ms_rows)
    neuro_df = pd.DataFrame(neuro_rows)
    dist_sens_df = pd.DataFrame(dist_rows)

    return ms_df, neuro_df, dist_sens_df


def run_singleton_sensitivity(deltas_df, tf_singleton_genes):
    """Repeat core mean-shift and neuronal/non-neuronal for singleton-only."""
    print("\n  Running singleton sensitivity...")
    rng = np.random.default_rng(SEED)
    df = deltas_df[deltas_df['gene'].isin(tf_singleton_genes)]
    print(f"    Singleton genes in deltas: {df['gene'].nunique()}")

    if len(df) == 0:
        print("    WARNING: No singleton genes found in deltas!")
        return pd.DataFrame()

    rows = []
    for setting_name, config in SENSITIVITY_SETTINGS.items():
        exclude = config['exclude_cts']
        sub = df[~df['cell_type'].isin(exclude)]
        emp = sub['empirical_delta'].values
        pred = sub['predicted_delta'].values

        pearson_r = np.corrcoef(emp, pred)[0, 1]
        spearman_r = stats.spearmanr(emp, pred).statistic
        ss_res = np.sum((pred - emp) ** 2)
        ss_tot = np.sum((emp - np.mean(emp)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        # Neuronal vs non-neuronal
        ct_means = sub.groupby('cell_type')[['empirical_delta', 'predicted_delta']].mean()
        ct_means['neuronal_status'] = ct_means.index.map(
            lambda x: 'neuronal' if x in NEURONAL else 'non-neuronal')
        neuro_ct = ct_means[ct_means['neuronal_status'] == 'neuronal']
        non_neuro_ct = ct_means[ct_means['neuronal_status'] == 'non-neuronal']

        rows.append({
            'setting': setting_name,
            'excluded_cts': '; '.join(sorted(exclude)) if exclude else 'none',
            'n_genes': sub['gene'].nunique(),
            'n_cell_types': sub['cell_type'].nunique(),
            'n_gene_celltype_pairs': len(sub),
            'pearson': pearson_r,
            'spearman': spearman_r,
            'R2': r2,
            'emp_neuronal': neuro_ct['empirical_delta'].mean(),
            'emp_non_neuronal': non_neuro_ct['empirical_delta'].mean(),
            'emp_diff': neuro_ct['empirical_delta'].mean() - non_neuro_ct['empirical_delta'].mean(),
            'pred_neuronal': neuro_ct['predicted_delta'].mean(),
            'pred_non_neuronal': non_neuro_ct['predicted_delta'].mean(),
            'pred_diff': neuro_ct['predicted_delta'].mean() - non_neuro_ct['predicted_delta'].mean(),
        })

    return pd.DataFrame(rows)


# ============================================================
# Step 6: Plotting
# ============================================================

def plot_main_validation(deltas_df, ct_summary_df, neuro_result):
    """Main validation figure: scatter + bar."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel 1: scatter empirical vs predicted delta
    ax = axes[0]
    emp = deltas_df['empirical_delta'].values
    pred = deltas_df['predicted_delta'].values
    ax.scatter(emp, pred, alpha=0.1, s=3, color='#2196F3', rasterized=True)
    lims = [min(emp.min(), pred.min()) - 0.1, max(emp.max(), pred.max()) + 0.1]
    ax.plot(lims, lims, 'k--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Empirical delta (cell - nucleus)')
    ax.set_ylabel('Predicted delta (imputed - nucleus)')
    r2 = np.corrcoef(emp, pred)[0, 1] ** 2
    ax.set_title(f'TF gene mean-shift (r={np.corrcoef(emp, pred)[0, 1]:.3f})')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel 2: per-CT mean shift
    ax = axes[1]
    ct_df = ct_summary_df.sort_values('empirical_mean_shift', ascending=True)
    colors = ['#2196F3' if s == 'neuronal' else '#FF9800'
              for s in ct_df['neuronal_status']]
    y = np.arange(len(ct_df))
    ax.barh(y, ct_df['empirical_mean_shift'], height=0.4, color=colors,
            alpha=0.7, label='Empirical')
    ax.barh(y + 0.4, ct_df['predicted_mean_shift'], height=0.4,
            color=colors, alpha=0.4, hatch='//', label='Predicted')
    ax.set_yticks(y + 0.2)
    ax.set_yticklabels(ct_df['cell_type'], fontsize=8)
    ax.set_xlabel('Mean TF delta')
    ax.set_title('Per-CT TF mean shift')
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel 3: neuronal vs non-neuronal
    ax = axes[2]
    x = np.arange(2)
    width = 0.35
    ax.bar(x - width/2,
           [neuro_result['emp_neuronal'], neuro_result['emp_non_neuronal']],
           width, label='Empirical', color=['#2196F3', '#FF9800'], alpha=0.85)
    ax.bar(x + width/2,
           [neuro_result['pred_neuronal'], neuro_result['pred_non_neuronal']],
           width, label='Predicted', color=['#2196F3', '#FF9800'], alpha=0.4,
           hatch='//')
    ax.set_xticks(x)
    ax.set_xticklabels(['Neuronal', 'Non-neuronal'])
    ax.set_ylabel('Mean TF delta')
    emp_d = neuro_result['emp_diff']
    pred_d = neuro_result['pred_diff']
    ax.set_title(f'Neuronal vs non-neuronal\nemp_diff={emp_d:.3f}, pred_diff={pred_d:.3f}')
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.suptitle('JASPAR mouse TF contrast analysis — ImputeVAE validation',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(OUT_DIR, f'tf_validation_main.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved tf_validation_main.png/pdf")


def plot_distribution(dist_ct_df, dist_group_df, dist_df):
    """Distribution-level validation figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel 1: median W_raw vs W_imputed by CT
    ax = axes[0]
    ct_sorted = dist_ct_df.sort_values('median_absolute_improvement')
    colors = ['#2196F3' if s == 'neuronal' else '#FF9800'
              for s in ct_sorted['neuronal_status']]
    y = np.arange(len(ct_sorted))
    ax.barh(y, ct_sorted['median_absolute_improvement'], color=colors, alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(ct_sorted['cell_type'], fontsize=8)
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Median Wasserstein improvement')
    ax.set_title('Per-CT distribution improvement')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel 2: proportion improved by group
    ax = axes[1]
    groups = dist_group_df[dist_group_df['group'].isin(['all', 'neuronal', 'non_neuronal'])]
    x = np.arange(len(groups))
    ax.bar(x, groups['proportion_improved_W'],
           color=['#9E9E9E', '#2196F3', '#FF9800'], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(groups['group'], fontsize=9)
    ax.set_ylabel('Proportion improved')
    ax.axhline(0.5, color='gray', linewidth=0.5, linestyle='--')
    ax.set_title('Fraction of gene-CT pairs improved')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel 3: distribution of absolute improvement
    ax = axes[2]
    neuro = dist_df[dist_df['neuronal_status'] == 'neuronal']['absolute_improvement']
    non_neuro = dist_df[dist_df['neuronal_status'] == 'non-neuronal']['absolute_improvement']
    ax.hist(neuro, bins=80, alpha=0.6, color='#2196F3', label='Neuronal', density=True)
    ax.hist(non_neuro, bins=80, alpha=0.6, color='#FF9800', label='Non-neuronal', density=True)
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Absolute Wasserstein improvement')
    ax.set_ylabel('Density')
    ax.set_title('Improvement distribution')
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.suptitle('JASPAR TF distribution-level validation',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(OUT_DIR, f'tf_distribution_validation.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved tf_distribution_validation.png/pdf")


def plot_sensitivity(ms_df, neuro_df, dist_sens_df):
    """Sensitivity plots (same format as SynGO)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Neuronal vs non-neuronal sensitivity
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels = ['All CTs', 'Excl. L5 ET', 'Excl. low-n', 'Excl. both']
    x = np.arange(len(labels))
    width = 0.35

    for ax_idx, (prefix, title) in enumerate([
        ('emp', 'Empirical (ground truth)'),
        ('pred', 'Predicted (ImputeVAE)')
    ]):
        ax = axes[ax_idx]
        neuronal_vals = neuro_df[f'{prefix}_neuronal'].values
        nonneuronal_vals = neuro_df[f'{prefix}_non_neuronal'].values

        ax.bar(x - width/2, neuronal_vals, width, label='Neuronal',
               color='#2196F3', alpha=0.85)
        ax.bar(x + width/2, nonneuronal_vals, width, label='Non-neuronal',
               color='#FF9800', alpha=0.85)

        for i in range(len(neuro_df)):
            diff = neuro_df[f'{prefix}_diff'].values[i]
            ci_lo = neuro_df[f'{prefix}_diff_ci_lo'].values[i]
            ci_hi = neuro_df[f'{prefix}_diff_ci_hi'].values[i]
            y_max = max(neuronal_vals[i], nonneuronal_vals[i])
            ax.annotate(f'diff={diff:.3f}\n[{ci_lo:.3f}, {ci_hi:.3f}]',
                        xy=(x[i], y_max + 0.02), ha='center', va='bottom',
                        fontsize=7.5, color='#333333')

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel('Mean TF delta (cell - nucleus)')
        ax.set_title(title)
        ax.legend(loc='upper right', fontsize=9)
        ax.set_ylim(0, max(neuronal_vals.max(), nonneuronal_vals.max()) + 0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('JASPAR TF contrast: neuronal vs non-neuronal mean shift\n'
                 'sensitivity to cell-type exclusion',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(OUT_DIR, f'tf_sensitivity_neuronal_vs_nonneuronal.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close()

    # Distribution sensitivity panel
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Median absolute improvement
    ax = axes[0]
    vals = dist_sens_df['median_absolute_improvement'].values
    colors = ['#4CAF50' if v > 0 else '#F44336' for v in vals]
    ax.bar(range(len(vals)), vals, color=colors, alpha=0.85)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9, rotation=15)
    ax.set_ylabel('Median absolute improvement')
    ax.set_title('Wasserstein improvement')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel 2: Proportion improved
    ax = axes[1]
    x = np.arange(len(labels))
    w = 0.25
    ax.bar(x - w, dist_sens_df['neuronal_proportion_improved'].values, w,
           label='Neuronal', color='#2196F3', alpha=0.85)
    ax.bar(x, dist_sens_df['proportion_improved_W'].values, w,
           label='All', color='#9E9E9E', alpha=0.85)
    ax.bar(x + w, dist_sens_df['non_neuronal_proportion_improved'].values, w,
           label='Non-neuronal', color='#FF9800', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15)
    ax.set_ylabel('Proportion improved')
    ax.axhline(0.5, color='gray', linewidth=0.5, linestyle='--')
    ax.set_title('Fraction improved')
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel 3: Neuronal vs non-neuronal
    ax = axes[2]
    ax.bar(x - w/2, dist_sens_df['neuronal_median_abs_imp'].values, w,
           label='Neuronal', color='#2196F3', alpha=0.85)
    ax.bar(x + w/2, dist_sens_df['non_neuronal_median_abs_imp'].values, w,
           label='Non-neuronal', color='#FF9800', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15)
    ax.set_ylabel('Median Wasserstein improvement')
    ax.set_title('Neuronal vs non-neuronal')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.suptitle('JASPAR TF distribution-level sensitivity analysis',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(OUT_DIR, f'tf_distribution_sensitivity_panel.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved sensitivity plots")


# ============================================================
# Step 7: SynGO vs TF comparison
# ============================================================

def create_comparison():
    """Create SynGO vs TF comparison outputs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("\n" + "=" * 70)
    print("Step 7: SynGO vs TF comparison")
    print("=" * 70)

    # Load SynGO results
    syngo_sens = pd.read_csv(os.path.join(SYNGO_DIR, 'syngo_sensitivity_summary.csv'))
    syngo_neuro = pd.read_csv(os.path.join(SYNGO_DIR, 'syngo_sensitivity_exclude_l5et.csv'))
    syngo_dist = pd.read_csv(os.path.join(SYNGO_DIR, 'syngo_distribution_sensitivity_summary.csv'))

    # Load TF results
    tf_sens = pd.read_csv(os.path.join(OUT_DIR, 'tf_sensitivity_summary.csv'))
    tf_neuro = pd.read_csv(os.path.join(OUT_DIR, 'tf_sensitivity_neuronal_vs_nonneuronal.csv'))
    tf_dist = pd.read_csv(os.path.join(OUT_DIR, 'tf_distribution_sensitivity_summary.csv'))
    tf_single = pd.read_csv(os.path.join(OUT_DIR, 'tf_singleton_sensitivity_summary.csv'))

    # Combined summary table
    rows = []
    for setting in ['all', 'exclude_l5et']:
        sg = syngo_neuro[syngo_neuro['setting'] == setting].iloc[0]
        tf = tf_neuro[tf_neuro['setting'] == setting].iloc[0]
        sg_s = syngo_sens[syngo_sens['setting'] == setting].iloc[0]
        tf_s = tf_sens[tf_sens['setting'] == setting].iloc[0]
        sg_d = syngo_dist[syngo_dist['setting'] == setting].iloc[0]
        tf_d = tf_dist[tf_dist['setting'] == setting].iloc[0]

        rows.append({
            'setting': setting,
            'gene_set': 'SynGO',
            'n_genes': int(sg_s['n_genes']),
            'n_cts': int(sg_s['n_cell_types']),
            'pearson': sg_s['pearson'],
            'R2': sg_s['R2'],
            'emp_neuronal': sg['emp_neuronal'],
            'emp_non_neuronal': sg['emp_non_neuronal'],
            'emp_diff': sg['emp_diff'],
            'emp_diff_ci_lo': sg['emp_diff_ci_lo'],
            'emp_diff_ci_hi': sg['emp_diff_ci_hi'],
            'pred_diff': sg['pred_diff'],
            'pred_diff_ci_lo': sg['pred_diff_ci_lo'],
            'pred_diff_ci_hi': sg['pred_diff_ci_hi'],
            'median_W_improvement': sg_d['median_absolute_improvement'],
            'proportion_improved': sg_d['proportion_improved_W'],
        })
        rows.append({
            'setting': setting,
            'gene_set': 'JASPAR_TF',
            'n_genes': int(tf_s['n_genes']),
            'n_cts': int(tf_s['n_cell_types']),
            'pearson': tf_s['pearson'],
            'R2': tf_s['R2'],
            'emp_neuronal': tf['emp_neuronal'],
            'emp_non_neuronal': tf['emp_non_neuronal'],
            'emp_diff': tf['emp_diff'],
            'emp_diff_ci_lo': tf['emp_diff_ci_lo'],
            'emp_diff_ci_hi': tf['emp_diff_ci_hi'],
            'pred_diff': tf['pred_diff'],
            'pred_diff_ci_lo': tf['pred_diff_ci_lo'],
            'pred_diff_ci_hi': tf['pred_diff_ci_hi'],
            'median_W_improvement': tf_d['median_absolute_improvement'],
            'proportion_improved': tf_d['proportion_improved_W'],
        })

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(os.path.join(COMP_DIR, 'syngo_vs_jaspar_tf_summary.csv'), index=False)
    print("  Saved syngo_vs_jaspar_tf_summary.csv")
    print(comp_df.to_string(index=False))

    # --- Comparison figure ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Panel 1: Empirical and predicted neuronal - non-neuronal (all CTs)
    ax = axes[0, 0]
    sg_all = syngo_neuro[syngo_neuro['setting'] == 'all'].iloc[0]
    tf_all = tf_neuro[tf_neuro['setting'] == 'all'].iloc[0]
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w/2, [sg_all['emp_diff'], tf_all['emp_diff']], w,
           label='Empirical', color='#4CAF50', alpha=0.85)
    ax.bar(x + w/2, [sg_all['pred_diff'], tf_all['pred_diff']], w,
           label='Predicted', color='#2196F3', alpha=0.85)
    # Error bars
    ax.errorbar(x - w/2,
                [sg_all['emp_diff'], tf_all['emp_diff']],
                yerr=[[sg_all['emp_diff'] - sg_all['emp_diff_ci_lo'],
                       tf_all['emp_diff'] - tf_all['emp_diff_ci_lo']],
                      [sg_all['emp_diff_ci_hi'] - sg_all['emp_diff'],
                       tf_all['emp_diff_ci_hi'] - tf_all['emp_diff']]],
                fmt='none', color='black', capsize=4)
    ax.errorbar(x + w/2,
                [sg_all['pred_diff'], tf_all['pred_diff']],
                yerr=[[sg_all['pred_diff'] - sg_all['pred_diff_ci_lo'],
                       tf_all['pred_diff'] - tf_all['pred_diff_ci_lo']],
                      [sg_all['pred_diff_ci_hi'] - sg_all['pred_diff'],
                       tf_all['pred_diff_ci_hi'] - tf_all['pred_diff']]],
                fmt='none', color='black', capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(['SynGO\n(synaptic)', 'JASPAR TF\n(nuclear)'], fontsize=10)
    ax.set_ylabel('Neuronal - non-neuronal mean shift')
    ax.set_title('Neuronal vs non-neuronal contrast\n(all cell types)')
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel 2: Wasserstein improvement
    ax = axes[0, 1]
    sg_d_all = syngo_dist[syngo_dist['setting'] == 'all'].iloc[0]
    tf_d_all = tf_dist[tf_dist['setting'] == 'all'].iloc[0]
    x = np.arange(2)
    w = 0.25
    ax.bar(x - w, [sg_d_all['neuronal_median_abs_imp'], tf_d_all['neuronal_median_abs_imp']],
           w, label='Neuronal', color='#2196F3', alpha=0.85)
    ax.bar(x, [sg_d_all['median_absolute_improvement'], tf_d_all['median_absolute_improvement']],
           w, label='All', color='#9E9E9E', alpha=0.85)
    ax.bar(x + w, [sg_d_all['non_neuronal_median_abs_imp'], tf_d_all['non_neuronal_median_abs_imp']],
           w, label='Non-neuronal', color='#FF9800', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(['SynGO', 'JASPAR TF'], fontsize=10)
    ax.set_ylabel('Median Wasserstein improvement')
    ax.set_title('Distribution-level improvement')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel 3: Sensitivity excluding L5 ET
    ax = axes[1, 0]
    sg_excl = syngo_neuro[syngo_neuro['setting'] == 'exclude_l5et'].iloc[0]
    tf_excl = tf_neuro[tf_neuro['setting'] == 'exclude_l5et'].iloc[0]
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w/2, [sg_excl['emp_diff'], tf_excl['emp_diff']], w,
           label='Empirical', color='#4CAF50', alpha=0.85)
    ax.bar(x + w/2, [sg_excl['pred_diff'], tf_excl['pred_diff']], w,
           label='Predicted', color='#2196F3', alpha=0.85)
    ax.errorbar(x - w/2,
                [sg_excl['emp_diff'], tf_excl['emp_diff']],
                yerr=[[sg_excl['emp_diff'] - sg_excl['emp_diff_ci_lo'],
                       tf_excl['emp_diff'] - tf_excl['emp_diff_ci_lo']],
                      [sg_excl['emp_diff_ci_hi'] - sg_excl['emp_diff'],
                       tf_excl['emp_diff_ci_hi'] - tf_excl['emp_diff']]],
                fmt='none', color='black', capsize=4)
    ax.errorbar(x + w/2,
                [sg_excl['pred_diff'], tf_excl['pred_diff']],
                yerr=[[sg_excl['pred_diff'] - sg_excl['pred_diff_ci_lo'],
                       tf_excl['pred_diff'] - tf_excl['pred_diff_ci_lo']],
                      [sg_excl['pred_diff_ci_hi'] - sg_excl['pred_diff'],
                       tf_excl['pred_diff_ci_hi'] - tf_excl['pred_diff']]],
                fmt='none', color='black', capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(['SynGO', 'JASPAR TF'], fontsize=10)
    ax.set_ylabel('Neuronal - non-neuronal mean shift')
    ax.set_title('Sensitivity: excluding L5 ET')
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel 4: TF primary vs singleton
    ax = axes[1, 1]
    tf_prim = tf_neuro[tf_neuro['setting'].isin(['all', 'exclude_l5et'])]
    tf_sing_all = tf_single[tf_single['setting'] == 'all'].iloc[0]
    tf_sing_excl = tf_single[tf_single['setting'] == 'exclude_l5et'].iloc[0]

    x = np.arange(2)
    w = 0.35
    ax.bar(x - w/2,
           [tf_neuro[tf_neuro['setting'] == 'all'].iloc[0]['emp_diff'],
            tf_neuro[tf_neuro['setting'] == 'exclude_l5et'].iloc[0]['emp_diff']],
           w, label=f'TF primary (n={tf_sens.iloc[0]["n_genes"]})',
           color='#9C27B0', alpha=0.85)
    ax.bar(x + w/2,
           [tf_sing_all['emp_diff'], tf_sing_excl['emp_diff']],
           w, label=f'TF singleton (n={int(tf_sing_all["n_genes"])})',
           color='#E91E63', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(['All CTs', 'Excl. L5 ET'], fontsize=10)
    ax.set_ylabel('Neuronal - non-neuronal emp diff')
    ax.set_title('TF primary vs singleton-only')
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.suptitle('SynGO (synaptic) vs JASPAR TF (nuclear) contrast',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(COMP_DIR, f'syngo_vs_jaspar_tf_summary.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved syngo_vs_jaspar_tf_summary.png/pdf")


# ============================================================
# README generators
# ============================================================

def write_characterization_readme(filtering_df, primary_df):
    """Write JASPAR characterization README."""
    in_model = primary_df[primary_df['in_model']]
    readme = f"""# JASPAR CORE 2026 Mouse TF Characterization

## Source
JASPAR CORE 2026 metadata: ultimate_metadata_table_CORE.tsv
Downloaded from: https://mencius.uio.no/JASPAR/JASPAR_metadata/2026/

## Important notes
- JASPAR is a motif database; rows are motif models, not one-to-one gene records.
- Multiple motifs can correspond to the same TF; we collapse to unique TF symbols.
- Complex motifs (e.g., Fos::Jun) are split on "::" for the primary set.
- Hyphenated symbols (e.g., Nkx2-5) are NOT split.

## Filtering pipeline
"""
    for _, row in filtering_df.iterrows():
        readme += f"- {row['stage']}: {int(row['count'])}\n"

    readme += f"""
## Gene sets
- **TF_primary**: All unique component symbols after splitting "::" complexes
- **TF_singleton_only**: Motif names without "::" (sensitivity analysis)

## Mapping methods
- exact_model_match: {(in_model['mapping_method'] == 'exact_model_match').sum()}
- case_insensitive_model_match: {(in_model['mapping_method'] == 'case_insensitive_model_match').sum()}
- mouse_style_fallback: {(in_model['mapping_method'] == 'mouse_style_fallback').sum()}

## Files
- jaspar_core_2026_mouse_rows.csv: All mouse CORE motif rows
- jaspar_core_2026_mouse_complex_rows.csv: Complex/dimer motif rows only
- jaspar_core_2026_mouse_tf_components.csv: All component symbols with source motif
- jaspar_core_2026_mouse_tf_symbols.csv: Primary TF symbol list
- jaspar_core_2026_mouse_tf_singleton_symbols.csv: Singleton-only TF symbol list
- jaspar_core_2026_mouse_tf_primary_model_genes.csv: Mapping to model genes
- jaspar_core_2026_mouse_tf_filtering_counts.csv: Filtering pipeline counts
- jaspar_core_2026_mouse_tf_class_summary.csv: TF class distribution
- jaspar_core_2026_mouse_tf_family_summary.csv: TF family distribution
"""
    with open(os.path.join(CHAR_DIR, "JASPAR_MOUSE_TF_CHARACTERIZATION.md"), 'w') as f:
        f.write(readme)
    print("  Saved JASPAR_MOUSE_TF_CHARACTERIZATION.md")


def write_validation_readme(overall, neuro_result, n_tf_genes, n_singleton):
    """Write validation README."""
    readme = f"""# JASPAR Mouse TF Brain Validation

## Purpose
Nuclear-function contrast analysis for ImputeVAE. Tests whether ImputeVAE creates
the same strong neuronal > non-neuronal shift for TF genes as for SynGO synaptic genes,
or whether the SynGO pattern is more specific to synaptic biology.

## Important caveats
1. TFs are not a perfect negative control; TF mRNAs can still be cytoplasmic.
2. JASPAR is a motif database; rows are motif models, not one-to-one gene records.
3. Multiple motifs can correspond to the same TF; collapsed to unique TF symbols.
4. Complex motifs split on "::" for primary set, excluded in singleton-only sensitivity.
5. All expression comparisons are validation-only, unpaired population comparisons.
6. SynGO and TFs annotate encoded protein function/class, not RNA subcellular localization.

## Gene set
- Primary TF genes (non-ribosomal, in model): {n_tf_genes}
- Singleton-only TF genes: {n_singleton}

## Mean-shift results (all CTs)
- Pearson r: {overall['pearson_r']:.4f}
- Spearman r: {overall['spearman_r']:.4f}
- R2: {overall['R2']:.4f}
- MAE: {overall['MAE']:.4f}
- RMSE: {overall['RMSE']:.4f}
- Sign agreement: {overall['sign_agreement']:.4f}

## Neuronal vs non-neuronal
- Empirical neuronal mean shift: {neuro_result['emp_neuronal']:.4f}
- Empirical non-neuronal mean shift: {neuro_result['emp_non_neuronal']:.4f}
- Empirical diff: {neuro_result['emp_diff']:.4f} [{neuro_result['emp_diff_ci_lo']:.4f}, {neuro_result['emp_diff_ci_hi']:.4f}]
- Predicted diff: {neuro_result['pred_diff']:.4f} [{neuro_result['pred_diff_ci_lo']:.4f}, {neuro_result['pred_diff_ci_hi']:.4f}]

## Validation data
- Split-aware validation only (held-out)
- Imputed: run/eval_v3_splitaware/cvae_imputed_valid.npy
- 17 cell types (12 neuronal, 5 non-neuronal)

## Files
- tf_gene_metadata.csv: TF gene metadata
- tf_gene_celltype_deltas.csv: Per-gene per-CT mean-shift deltas
- tf_overall_quality.csv: Overall quality metrics
- tf_quality_by_celltype.csv: (same as tf_celltype_summary.csv)
- tf_celltype_summary.csv: Per-CT summary with CIs
- tf_neuronal_vs_nonneuronal.csv: Neuronal vs non-neuronal comparison
- tf_validation_main.png/pdf: Main validation figure
- tf_sensitivity_*.csv: Sensitivity analysis results
- tf_distribution_*.csv: Distribution-level results
"""
    with open(os.path.join(OUT_DIR, "README.md"), 'w') as f:
        f.write(readme)
    print("  Saved README.md")


def write_distribution_readme():
    """Write distribution analysis README."""
    readme = """# JASPAR TF Distribution-Level Analysis

## Method
For each TF gene g and cell type c:

W_raw[g,c] = Wasserstein(raw_nucleus_log1p[g,c], raw_cell_log1p[g,c])
W_imputed[g,c] = Wasserstein(imputed_cell_log1p[g,c], raw_cell_log1p[g,c])
absolute_improvement[g,c] = W_raw[g,c] - W_imputed[g,c]
normalized_improvement[g,c] = 1 - W_imputed / W_raw (W_raw >= 0.01)

Positive absolute_improvement means imputation moved the distribution closer
to observed whole-cell expression.

## Notes
- All comparisons use log1p-normalized values (same scale as model input/output).
- Because ImputeVAE deterministic decoder outputs are dense, detection at
  threshold 0 is not very informative.

## Files
- tf_distribution_gene_celltype.csv: Per-gene per-CT Wasserstein distances
- tf_distribution_by_celltype.csv: Per-CT summary
- tf_distribution_by_group.csv: By neuronal/non-neuronal group
- tf_distribution_validation.png/pdf: Visualization
- tf_distribution_sensitivity_summary.csv: Sensitivity across CT exclusion settings
- tf_distribution_sensitivity_panel.png/pdf: Sensitivity panel
"""
    with open(os.path.join(OUT_DIR, "DISTRIBUTION_README.md"), 'w') as f:
        f.write(readme)
    print("  Saved DISTRIBUTION_README.md")


def write_comparison_readme():
    """Write comparison README."""
    readme = """# SynGO vs JASPAR TF Comparison

## Purpose
Test whether ImputeVAE recovers a strong neuronal whole-cell-minus-nucleus shift
specifically for SynGO synaptic-protein genes, or whether the same pattern appears
indiscriminately in JASPAR mouse transcription factors.

## Gene sets
- SynGO: Non-ribosomal genes under SynGO CC synapse (positive control for synaptic biology)
- JASPAR TF: Mouse transcription factors from JASPAR CORE 2026 (nuclear-function contrast)

## Key biological framing
- SynGO is a positive synaptic-protein validation set.
- JASPAR TFs are a nuclear-function contrast set, not a perfect negative control.
- TF mRNAs can still be cytoplasmic, but encoded TF proteins are nuclear/regulatory.
- SynGO and TFs annotate protein function/location, not RNA localization.

## Interpretation guide
A strong specificity result would be:
- SynGO: large neuronal > non-neuronal shift, recovered by ImputeVAE
- TF: smaller or absent neuronal > non-neuronal contrast

If TFs also show a strong neuronal shift, interpret cautiously: that may indicate
ImputeVAE learns a broad neuronal shift rather than a SynGO-specific pattern.

## Files
- syngo_vs_jaspar_tf_summary.csv: Side-by-side metrics
- syngo_vs_jaspar_tf_summary.png/pdf: 4-panel comparison figure
"""
    with open(os.path.join(COMP_DIR, "README.md"), 'w') as f:
        f.write(readme)
    print("  Saved comparison README.md")


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(CHAR_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(COMP_DIR, exist_ok=True)

    # Step 1: Characterize JASPAR
    tf_primary_genes, tf_singleton_genes, primary_df, filtering_df = characterize_jaspar()
    write_characterization_readme(filtering_df, primary_df)

    # Step 2: Load validation data
    (gene_names, valid_cell_X, valid_cell_ct,
     valid_nuc_X, valid_nuc_ct, imputed_X, imputed_ct) = load_data()

    # Step 3: Mean-shift analysis
    deltas_df, cell_types, tf_in_model = compute_mean_shift(
        tf_primary_genes, gene_names, valid_cell_X, valid_cell_ct,
        valid_nuc_X, valid_nuc_ct, imputed_X, imputed_ct)

    overall = compute_quality_metrics(deltas_df)
    rng = np.random.default_rng(SEED)
    neuro_result = compute_neuronal_nonneuronal(deltas_df, rng)
    ct_summary_df = compute_celltype_summary(deltas_df, rng)

    # Save gene metadata
    gene_meta = primary_df[primary_df['in_model'] & ~primary_df['is_ribosomal']].copy()
    gene_meta.to_csv(os.path.join(OUT_DIR, 'tf_gene_metadata.csv'), index=False)

    # Save mean-shift outputs
    deltas_df.to_csv(os.path.join(OUT_DIR, 'tf_gene_celltype_deltas.csv'), index=False)

    overall_df = pd.DataFrame([overall])
    overall_df.to_csv(os.path.join(OUT_DIR, 'tf_overall_quality.csv'), index=False)

    ct_summary_df.to_csv(os.path.join(OUT_DIR, 'tf_celltype_summary.csv'), index=False)
    ct_summary_df.to_csv(os.path.join(OUT_DIR, 'tf_quality_by_celltype.csv'), index=False)

    neuro_df = pd.DataFrame([neuro_result])
    neuro_df.to_csv(os.path.join(OUT_DIR, 'tf_neuronal_vs_nonneuronal.csv'), index=False)

    print(f"\n  Overall quality: r={overall['pearson_r']:.4f}, R2={overall['R2']:.4f}")
    print(f"  Neuronal vs non-neuronal:")
    print(f"    emp_diff = {neuro_result['emp_diff']:.4f} "
          f"[{neuro_result['emp_diff_ci_lo']:.4f}, {neuro_result['emp_diff_ci_hi']:.4f}]")
    print(f"    pred_diff = {neuro_result['pred_diff']:.4f} "
          f"[{neuro_result['pred_diff_ci_lo']:.4f}, {neuro_result['pred_diff_ci_hi']:.4f}]")

    # Step 4: Distribution analysis
    dist_df = compute_distribution_metrics(
        tf_primary_genes, gene_names, valid_cell_X, valid_cell_ct,
        valid_nuc_X, valid_nuc_ct, imputed_X, imputed_ct)
    dist_ct_df, dist_group_df = summarize_distribution(dist_df)

    dist_df.to_csv(os.path.join(OUT_DIR, 'tf_distribution_gene_celltype.csv'), index=False)
    dist_ct_df.to_csv(os.path.join(OUT_DIR, 'tf_distribution_by_celltype.csv'), index=False)
    dist_group_df.to_csv(os.path.join(OUT_DIR, 'tf_distribution_by_group.csv'), index=False)

    # Step 5: Sensitivity
    ms_sens, neuro_sens, dist_sens = run_sensitivity(deltas_df, dist_df, ct_summary_df)
    ms_sens.to_csv(os.path.join(OUT_DIR, 'tf_sensitivity_summary.csv'), index=False)
    neuro_sens.to_csv(os.path.join(OUT_DIR, 'tf_sensitivity_neuronal_vs_nonneuronal.csv'), index=False)
    dist_sens.to_csv(os.path.join(OUT_DIR, 'tf_distribution_sensitivity_summary.csv'), index=False)

    # Singleton sensitivity
    singleton_sens = run_singleton_sensitivity(deltas_df, tf_singleton_genes)
    singleton_sens.to_csv(os.path.join(OUT_DIR, 'tf_singleton_sensitivity_summary.csv'), index=False)

    print("\n  Sensitivity summary:")
    print(ms_sens[['setting', 'n_genes', 'n_cell_types', 'pearson', 'R2']].to_string(index=False))
    print("\n  Neuronal vs non-neuronal sensitivity:")
    print(neuro_sens[['setting', 'emp_diff', 'emp_diff_ci_lo', 'emp_diff_ci_hi',
                       'pred_diff']].to_string(index=False))

    # Step 6: Plots
    print("\n" + "=" * 70)
    print("Step 6: Plotting")
    print("=" * 70)
    plot_main_validation(deltas_df, ct_summary_df, neuro_result)
    plot_distribution(dist_ct_df, dist_group_df, dist_df)
    plot_sensitivity(ms_sens, neuro_sens, dist_sens)

    # READMEs
    write_validation_readme(overall, neuro_result, len(tf_primary_genes), len(tf_singleton_genes))
    write_distribution_readme()

    # Step 7: SynGO vs TF comparison
    create_comparison()
    write_comparison_readme()

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
