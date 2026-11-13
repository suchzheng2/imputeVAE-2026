#!/usr/bin/env python
"""
L5 ET diagnostic analysis for ImputeVAE SynGO and TF validation.

Reads existing validation CSVs to diagnose why L5 ET is a recurrent outlier.
Does NOT modify primary SynGO or TF results.

Usage:
    python -u validation/l5et_diagnostic/l5et_diagnostic.py [--skip-reference]

Requires conda env: impute2
"""

import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd
from scipy import stats, sparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')

# ============================================================
# Paths
# ============================================================
PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
DATA_ROOT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi"

SYNGO_DIR = os.path.join(PROJECT, "validation/syngo/syngo_brain_validation")
TF_DIR = os.path.join(PROJECT, "validation/tf/tf_brain_validation")
OUT_DIR = os.path.join(PROJECT, "validation/l5et_diagnostic")

GENE_LIST_PATH = os.path.join(PROJECT, "data/training_gene_list_v3.csv")
SPLIT_MANIFEST = os.path.join(PROJECT, "data/split_manifest.tsv")
COMMON_CT_PATH = os.path.join(PROJECT, "data/common_cell_types.csv")
COMMON_GENES_PATH = os.path.join(PROJECT, "data/common_genes.csv")

NEURONAL = ['L2/3 IT', 'L5 ET', 'L5 IT', 'L5/6 NP', 'L6 CT', 'L6 IT', 'L6b',
            'Lamp5', 'Pvalb', 'Sncg', 'Sst', 'Vip']
NON_NEURONAL = ['Astro', 'Endo', 'Macrophage', 'OPC', 'Oligo']

FOCUS_CTS = ['L5 ET', 'L5 IT', 'L2/3 IT', 'L6 IT', 'L6 CT']


# ============================================================
# Helpers
# ============================================================

def compute_metrics(emp, pred):
    """Compute quality metrics from empirical and predicted arrays."""
    n = len(emp)
    if n < 3:
        return {k: np.nan for k in ['pearson', 'spearman', 'R2', 'MAE', 'RMSE',
                                     'sign_agreement', 'slope', 'intercept']}
    pearson_r = np.corrcoef(emp, pred)[0, 1]
    spearman_r = stats.spearmanr(emp, pred).statistic
    ss_res = np.sum((pred - emp) ** 2)
    ss_tot = np.sum((emp - np.mean(emp)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    mae = np.mean(np.abs(pred - emp))
    rmse = np.sqrt(np.mean((pred - emp) ** 2))
    sign_agree = np.mean(np.sign(emp) == np.sign(pred))
    slope, intercept = np.polyfit(emp, pred, 1)
    return {'pearson': pearson_r, 'spearman': spearman_r, 'R2': r2,
            'MAE': mae, 'RMSE': rmse, 'sign_agreement': sign_agree,
            'slope': slope, 'intercept': intercept}


# ============================================================
# Analysis 1: Cell-type comparison summary
# ============================================================

def celltype_comparison_summary():
    """Compare L5 ET against related and broad cell types for both gene sets."""
    print("=" * 70)
    print("Analysis 1: Cell-type comparison summary")
    print("=" * 70)

    rows = []
    for gene_set, deltas_path, dist_path in [
        ('SynGO', os.path.join(SYNGO_DIR, 'syngo_gene_celltype_deltas.csv'),
         os.path.join(SYNGO_DIR, 'syngo_distribution_gene_celltype.csv')),
        ('JASPAR_TF', os.path.join(TF_DIR, 'tf_gene_celltype_deltas.csv'),
         os.path.join(TF_DIR, 'tf_distribution_gene_celltype.csv')),
    ]:
        deltas = pd.read_csv(deltas_path)
        dist = pd.read_csv(dist_path)

        all_cts = sorted(deltas['cell_type'].unique())

        for ct in all_cts:
            sub = deltas[deltas['cell_type'] == ct]
            dsub = dist[dist['cell_type'] == ct]
            emp = sub['empirical_delta'].values
            pred = sub['predicted_delta'].values
            error = pred - emp

            m = compute_metrics(emp, pred)
            neuronal_status = 'neuronal' if ct in NEURONAL else 'non-neuronal'

            row = {
                'gene_set': gene_set,
                'cell_type': ct,
                'neuronal_status': neuronal_status,
                'n_cell': int(sub['n_cell'].iloc[0]),
                'n_nucleus': int(sub['n_nucleus'].iloc[0]),
                'n_genes': len(sub),
                'empirical_mean_shift': emp.mean(),
                'predicted_mean_shift': pred.mean(),
                'prediction_error': error.mean(),
                'MAE': m['MAE'],
                'RMSE': m['RMSE'],
                'pearson': m['pearson'],
                'spearman': m['spearman'],
                'R2': m['R2'],
                'sign_agreement': m['sign_agreement'],
                'slope': m['slope'],
                'intercept': m['intercept'],
                'median_W_raw': dsub['W_raw'].median(),
                'median_W_imputed': dsub['W_imputed'].median(),
                'median_W_improvement': dsub['absolute_improvement'].median(),
                'proportion_improved': (dsub['absolute_improvement'] > 0).mean(),
            }
            rows.append(row)

        # Add group summaries
        for group, cts in [('all_neuronal_excl_l5et',
                            [c for c in NEURONAL if c != 'L5 ET' and c in all_cts]),
                           ('all_non_neuronal',
                            [c for c in NON_NEURONAL if c in all_cts])]:
            sub = deltas[deltas['cell_type'].isin(cts)]
            dsub = dist[dist['cell_type'].isin(cts)]
            emp = sub['empirical_delta'].values
            pred = sub['predicted_delta'].values
            error = pred - emp
            m = compute_metrics(emp, pred)

            rows.append({
                'gene_set': gene_set,
                'cell_type': f'[{group}]',
                'neuronal_status': 'group',
                'n_cell': -1,
                'n_nucleus': -1,
                'n_genes': sub['gene'].nunique(),
                'empirical_mean_shift': emp.mean(),
                'predicted_mean_shift': pred.mean(),
                'prediction_error': error.mean(),
                'MAE': m['MAE'],
                'RMSE': m['RMSE'],
                'pearson': m['pearson'],
                'spearman': m['spearman'],
                'R2': m['R2'],
                'sign_agreement': m['sign_agreement'],
                'slope': m['slope'],
                'intercept': m['intercept'],
                'median_W_raw': dsub['W_raw'].median(),
                'median_W_imputed': dsub['W_imputed'].median(),
                'median_W_improvement': dsub['absolute_improvement'].median(),
                'proportion_improved': (dsub['absolute_improvement'] > 0).mean(),
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, 'l5et_celltype_comparison_summary.csv'), index=False)
    print(f"  Saved l5et_celltype_comparison_summary.csv ({len(df)} rows)")

    # Print L5 ET vs L5 IT comparison
    for gs in ['SynGO', 'JASPAR_TF']:
        print(f"\n  {gs} — L5 ET vs L5 IT:")
        for ct in ['L5 ET', 'L5 IT']:
            r = df[(df['gene_set'] == gs) & (df['cell_type'] == ct)].iloc[0]
            print(f"    {ct}: error={r['prediction_error']:.4f}, "
                  f"R2={r['R2']:.4f}, slope={r['slope']:.3f}")

    return df


# ============================================================
# Analysis 2: Gene-level L5 ET error tables
# ============================================================

def gene_level_errors():
    """Extract L5 ET gene-level errors for SynGO and TF."""
    print("\n" + "=" * 70)
    print("Analysis 2: Gene-level L5 ET error tables")
    print("=" * 70)

    for label, deltas_path, dist_path, prefix in [
        ('SynGO', os.path.join(SYNGO_DIR, 'syngo_gene_celltype_deltas.csv'),
         os.path.join(SYNGO_DIR, 'syngo_distribution_gene_celltype.csv'), 'syngo'),
        ('JASPAR_TF', os.path.join(TF_DIR, 'tf_gene_celltype_deltas.csv'),
         os.path.join(TF_DIR, 'tf_distribution_gene_celltype.csv'), 'tf'),
    ]:
        deltas = pd.read_csv(deltas_path)
        dist = pd.read_csv(dist_path)

        l5et = deltas[deltas['cell_type'] == 'L5 ET'].copy()
        l5et_dist = dist[dist['cell_type'] == 'L5 ET'].copy()

        l5et['prediction_error'] = l5et['predicted_delta'] - l5et['empirical_delta']
        l5et['abs_error'] = l5et['prediction_error'].abs()

        # Merge distribution metrics
        if len(l5et_dist) > 0:
            merge_cols = ['gene']
            dist_cols = ['gene', 'W_raw', 'W_imputed', 'absolute_improvement']
            l5et = l5et.merge(l5et_dist[dist_cols], on='gene', how='left')

        # Rank
        l5et['rank_abs_error'] = l5et['abs_error'].rank(ascending=False).astype(int)
        l5et['rank_prediction_error'] = l5et['prediction_error'].rank(ascending=False).astype(int)

        l5et_sorted = l5et.sort_values('abs_error', ascending=False)
        l5et_sorted.to_csv(os.path.join(OUT_DIR, f'{prefix}_l5et_gene_errors.csv'), index=False)

        # Top overpredicted (positive error)
        over = l5et.sort_values('prediction_error', ascending=False).head(20)
        over.to_csv(os.path.join(OUT_DIR, f'{prefix}_l5et_top_overpredicted.csv'), index=False)

        # Top underpredicted (negative error)
        under = l5et.sort_values('prediction_error', ascending=True).head(20)
        under.to_csv(os.path.join(OUT_DIR, f'{prefix}_l5et_top_underpredicted.csv'), index=False)

        print(f"  {label}: {len(l5et)} genes")
        print(f"    Mean error: {l5et['prediction_error'].mean():.4f}")
        print(f"    Frac overpredicted: {(l5et['prediction_error'] > 0).mean():.3f}")
        print(f"    Top overpredicted: {over['gene'].values[:5].tolist()}")
        print(f"    Top underpredicted: {under['gene'].values[:5].tolist()}")


# ============================================================
# Analysis 3: Global bias vs gene-specific
# ============================================================

def global_bias_analysis():
    """Check if L5 ET is globally over-shifted or gene-specific."""
    print("\n" + "=" * 70)
    print("Analysis 3: Global bias summary")
    print("=" * 70)

    rows = []
    for label, deltas_path, prefix in [
        ('SynGO', os.path.join(SYNGO_DIR, 'syngo_gene_celltype_deltas.csv'), 'syngo'),
        ('JASPAR_TF', os.path.join(TF_DIR, 'tf_gene_celltype_deltas.csv'), 'tf'),
    ]:
        deltas = pd.read_csv(deltas_path)
        neuronal_cts = [c for c in NEURONAL if c in deltas['cell_type'].unique()]

        for ct in ['L5 ET', 'L5 IT']:
            sub = deltas[deltas['cell_type'] == ct]
            emp = sub['empirical_delta'].values
            pred = sub['predicted_delta'].values
            error = pred - emp
            slope, intercept = np.polyfit(emp, pred, 1)
            rows.append({
                'gene_set': label,
                'cell_type': ct,
                'slope': slope,
                'intercept': intercept,
                'median_prediction_error': np.median(error),
                'mean_prediction_error': np.mean(error),
                'fraction_overpredicted': (error > 0).mean(),
                'n_genes': len(sub),
            })

        # Median across all neuronal CTs (excluding L5 ET)
        other_neuro = deltas[deltas['cell_type'].isin(
            [c for c in neuronal_cts if c != 'L5 ET'])]
        ct_errors = other_neuro.groupby('cell_type').apply(
            lambda g: pd.Series({
                'slope': np.polyfit(g['empirical_delta'], g['predicted_delta'], 1)[0],
                'intercept': np.polyfit(g['empirical_delta'], g['predicted_delta'], 1)[1],
                'median_error': (g['predicted_delta'] - g['empirical_delta']).median(),
                'mean_error': (g['predicted_delta'] - g['empirical_delta']).mean(),
                'frac_over': ((g['predicted_delta'] - g['empirical_delta']) > 0).mean(),
            })
        )
        rows.append({
            'gene_set': label,
            'cell_type': '[median_other_neuronal]',
            'slope': ct_errors['slope'].median(),
            'intercept': ct_errors['intercept'].median(),
            'median_prediction_error': ct_errors['median_error'].median(),
            'mean_prediction_error': ct_errors['mean_error'].median(),
            'fraction_overpredicted': ct_errors['frac_over'].median(),
            'n_genes': int(np.median([
                len(deltas[deltas['cell_type'] == c]) for c in ct_errors.index])),
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, 'l5et_global_bias_summary.csv'), index=False)
    print(df.to_string(index=False))
    return df


# ============================================================
# Analysis 4: Profile similarity
# ============================================================

def profile_similarity():
    """Check if L5 ET imputed profile is closer to L5 IT than L5 ET."""
    print("\n" + "=" * 70)
    print("Analysis 4: Profile similarity")
    print("=" * 70)

    for label, deltas_path, prefix in [
        ('SynGO', os.path.join(SYNGO_DIR, 'syngo_gene_celltype_deltas.csv'), 'syngo'),
        ('JASPAR_TF', os.path.join(TF_DIR, 'tf_gene_celltype_deltas.csv'), 'tf'),
    ]:
        deltas = pd.read_csv(deltas_path)
        all_cts = sorted(deltas['cell_type'].unique())
        genes = sorted(deltas[deltas['cell_type'] == 'L5 ET']['gene'].unique())

        # Build gene-level mean vectors per CT
        ct_cell_means = {}
        ct_nuc_means = {}
        ct_imp_means = {}
        for ct in all_cts:
            sub = deltas[(deltas['cell_type'] == ct) & (deltas['gene'].isin(genes))]\
                .sort_values('gene')
            ct_cell_means[ct] = sub['cell_mean'].values
            ct_nuc_means[ct] = sub['nucleus_mean'].values
            ct_imp_means[ct] = sub['imputed_mean'].values

        l5et_imp = ct_imp_means['L5 ET']
        l5et_nuc = ct_nuc_means['L5 ET']
        l5et_cell = ct_cell_means['L5 ET']

        rows = []
        for query_name, query_vec in [
            ('L5ET_imputed', l5et_imp),
            ('L5ET_nucleus', l5et_nuc),
            ('L5ET_empirical_cell', l5et_cell),
        ]:
            for ref_ct in all_cts:
                ref_vec = ct_cell_means[ref_ct]
                pearson_r = np.corrcoef(query_vec, ref_vec)[0, 1]
                spearman_r = stats.spearmanr(query_vec, ref_vec).statistic
                euclidean = np.sqrt(np.sum((query_vec - ref_vec) ** 2))
                mean_diff = np.mean(query_vec - ref_vec)
                rows.append({
                    'query_profile': query_name,
                    'reference_cell_type': ref_ct,
                    'reference_profile': 'empirical_cell_mean',
                    'pearson': pearson_r,
                    'spearman': spearman_r,
                    'euclidean_distance': euclidean,
                    'mean_difference': mean_diff,
                })

        sim_df = pd.DataFrame(rows)
        sim_df.to_csv(os.path.join(OUT_DIR, f'{prefix}_l5et_profile_similarity.csv'), index=False)

        # Print key comparisons
        imp_row = sim_df[sim_df['query_profile'] == 'L5ET_imputed'].sort_values(
            'pearson', ascending=False)
        print(f"\n  {label} — L5 ET imputed most similar to:")
        for _, r in imp_row.head(5).iterrows():
            print(f"    {r['reference_cell_type']}: pearson={r['pearson']:.4f}, "
                  f"euclidean={r['euclidean_distance']:.3f}")


# ============================================================
# Analysis 5: Sample-size diagnostic
# ============================================================

def sample_size_diagnostic():
    """Collect and verify sample counts across gene sets."""
    print("\n" + "=" * 70)
    print("Analysis 5: Sample-size diagnostic")
    print("=" * 70)

    syngo = pd.read_csv(os.path.join(SYNGO_DIR, 'syngo_gene_celltype_deltas.csv'))
    tf = pd.read_csv(os.path.join(TF_DIR, 'tf_gene_celltype_deltas.csv'))

    syngo_counts = syngo.groupby('cell_type')[['n_cell', 'n_nucleus']].first().reset_index()
    syngo_counts.columns = ['cell_type', 'syngo_n_cell', 'syngo_n_nucleus']
    tf_counts = tf.groupby('cell_type')[['n_cell', 'n_nucleus']].first().reset_index()
    tf_counts.columns = ['cell_type', 'tf_n_cell', 'tf_n_nucleus']

    merged = syngo_counts.merge(tf_counts, on='cell_type', how='outer')
    merged['counts_match'] = (
        (merged['syngo_n_cell'] == merged['tf_n_cell']) &
        (merged['syngo_n_nucleus'] == merged['tf_n_nucleus']))
    merged['neuronal_status'] = merged['cell_type'].map(
        lambda x: 'neuronal' if x in NEURONAL else 'non-neuronal')

    n_cell_col = 'syngo_n_cell'
    n_nuc_col = 'syngo_n_nucleus'
    merged['low_n_cell'] = merged[n_cell_col] < 50
    merged['low_n_nucleus'] = merged[n_nuc_col] < 50
    merged['low_sample_existing'] = (merged[n_cell_col] < 20) | (merged[n_nuc_col] < 20)

    merged = merged.sort_values('cell_type')
    merged.to_csv(os.path.join(OUT_DIR, 'validation_sample_counts_by_celltype.csv'), index=False)
    print(merged.to_string(index=False))

    # Note about L5 ET
    l5et = merged[merged['cell_type'] == 'L5 ET'].iloc[0]
    print(f"\n  L5 ET: {int(l5et[n_cell_col])} cells, {int(l5et[n_nuc_col])} nuclei")
    print(f"    low_sample_existing (<20): {l5et['low_sample_existing']}")
    print(f"    low_n_cell (<50): {l5et['low_n_cell']}")
    print("    Note: 32 whole cells is small for stable cell-type mean estimation")

    return merged


# ============================================================
# Analysis 6: Optional all-cell reference sensitivity
# ============================================================

def reference_sensitivity(skip=False):
    """Compare validation-only vs all-cell L5 ET cell mean."""
    print("\n" + "=" * 70)
    print("Analysis 6: All-cell reference sensitivity")
    print("=" * 70)

    if skip:
        print("  SKIPPED (--skip-reference)")
        return False

    import anndata as ad

    gene_list = pd.read_csv(GENE_LIST_PATH, header=None)[0].tolist()
    manifest = pd.read_csv(SPLIT_MANIFEST, sep='\t')
    common_ct = pd.read_csv(COMMON_CT_PATH)['common_cell_types'].tolist()
    common_genes = pd.read_csv(COMMON_GENES_PATH)['common_genes'].tolist()

    cell_path = os.path.join(DATA_ROOT, "10X_cells_v2_v3_AIBS/data.h5ad")
    if not os.path.exists(cell_path):
        print(f"  Raw data not found: {cell_path}")
        print("  SKIPPED")
        return False

    print(f"  Loading {cell_path}...")
    adata = ad.read_h5ad(cell_path)
    if 'meta_batch_id' not in adata.obs:
        adata.obs['meta_batch_id'] = adata.obs['batch_id'].apply(lambda x: x.split('.')[0])
    row_mask = (adata.obs['meta_batch_id'].str.endswith('c3A') &
                adata.obs['cell_type'].isin(common_ct))
    col_mask = (adata.var['gene_names'].isin(common_genes) &
                adata.var['gene_names'].isin(gene_list))
    adata = adata[row_mask, col_mask].copy()
    adata = adata[:, adata.var['gene_names'].argsort()].copy()
    X = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
    X_log1p = np.log1p(X).astype(np.float32)
    gene_names = adata.var['gene_names'].values.tolist()
    cell_types = adata.obs['cell_type'].values

    # Split into validation vs all
    cell_manifest = manifest[manifest.modality == 'cell'].sort_values('obs_id').reset_index(drop=True)
    valid_mask = cell_manifest.partition == 'valid'
    valid_idx = np.where(valid_mask)[0]
    train_idx = np.where(~valid_mask)[0]

    print(f"  Total L5 ET cells (c3A): {(cell_types == 'L5 ET').sum()}")
    print(f"  Validation L5 ET: {((cell_types[valid_idx] == 'L5 ET')).sum()}")
    print(f"  All L5 ET: {(cell_types == 'L5 ET').sum()}")

    # Gene index maps for SynGO and TF
    syngo_deltas = pd.read_csv(os.path.join(SYNGO_DIR, 'syngo_gene_celltype_deltas.csv'))
    tf_deltas = pd.read_csv(os.path.join(TF_DIR, 'tf_gene_celltype_deltas.csv'))
    syngo_genes = sorted(syngo_deltas['gene'].unique())
    tf_genes = sorted(tf_deltas['gene'].unique())

    gene_idx_map = {g: i for i, g in enumerate(gene_names)}

    rows = []
    for gene_set_name, gene_set in [('SynGO', syngo_genes), ('JASPAR_TF', tf_genes)]:
        idxs = [gene_idx_map[g] for g in gene_set if g in gene_idx_map]
        gs_genes = [gene_names[i] for i in idxs]

        # L5 ET validation-only
        l5et_valid = X_log1p[valid_idx][cell_types[valid_idx] == 'L5 ET'][:, idxs]
        # L5 ET all cells
        l5et_all = X_log1p[cell_types == 'L5 ET'][:, idxs]

        valid_mean = l5et_valid.mean(axis=0)
        all_mean = l5et_all.mean(axis=0)

        pearson_r = np.corrcoef(valid_mean, all_mean)[0, 1]
        mae = np.mean(np.abs(valid_mean - all_mean))
        max_diff_idx = np.argmax(np.abs(valid_mean - all_mean))

        rows.append({
            'gene_set': gene_set_name,
            'n_valid_l5et_cells': len(l5et_valid),
            'n_all_l5et_cells': len(l5et_all),
            'n_genes': len(idxs),
            'pearson_valid_vs_all_mean': pearson_r,
            'mae_valid_vs_all_mean': mae,
            'max_abs_diff': np.abs(valid_mean - all_mean).max(),
            'max_diff_gene': gs_genes[max_diff_idx],
            'valid_mean_overall': valid_mean.mean(),
            'all_mean_overall': all_mean.mean(),
            'diff_overall': valid_mean.mean() - all_mean.mean(),
        })

        print(f"\n  {gene_set_name}:")
        print(f"    Valid L5 ET cells: {len(l5et_valid)}, All: {len(l5et_all)}")
        print(f"    Pearson(valid_mean, all_mean): {pearson_r:.6f}")
        print(f"    MAE: {mae:.6f}")
        print(f"    Max diff gene: {gs_genes[max_diff_idx]} "
              f"({np.abs(valid_mean - all_mean).max():.4f})")

    ref_df = pd.DataFrame(rows)
    ref_df.to_csv(os.path.join(OUT_DIR, 'l5et_validation_vs_all_cells_reference.csv'), index=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax_i, (gene_set_name, gene_set) in enumerate(
            [('SynGO', syngo_genes), ('JASPAR_TF', tf_genes)]):
        ax = axes[ax_i]
        idxs = [gene_idx_map[g] for g in gene_set if g in gene_idx_map]
        l5et_valid = X_log1p[valid_idx][cell_types[valid_idx] == 'L5 ET'][:, idxs]
        l5et_all = X_log1p[cell_types == 'L5 ET'][:, idxs]
        vm = l5et_valid.mean(axis=0)
        am = l5et_all.mean(axis=0)
        r = np.corrcoef(vm, am)[0, 1]

        ax.scatter(am, vm, alpha=0.5, s=15, color='#2196F3')
        lims = [0, max(vm.max(), am.max()) * 1.05]
        ax.plot(lims, lims, 'k--', linewidth=0.8, alpha=0.5)
        ax.set_xlabel(f'All L5 ET cells mean (n={len(l5et_all)})')
        ax.set_ylabel(f'Validation L5 ET cells mean (n={len(l5et_valid)})')
        ax.set_title(f'{gene_set_name}: r={r:.4f}')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('L5 ET reference sensitivity: validation vs all cells\n'
                 '(population reference, NOT primary validation)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(OUT_DIR, f'l5et_reference_sensitivity.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close()
    print("\n  Saved l5et_reference_sensitivity.png/pdf")

    del adata, X, X_log1p
    return True


# ============================================================
# Figures
# ============================================================

def plot_main_diagnostic():
    """Main 6-panel diagnostic figure."""
    print("\n" + "=" * 70)
    print("Plotting: l5et_diagnostic_main")
    print("=" * 70)

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

    syngo_deltas = pd.read_csv(os.path.join(SYNGO_DIR, 'syngo_gene_celltype_deltas.csv'))
    tf_deltas = pd.read_csv(os.path.join(TF_DIR, 'tf_gene_celltype_deltas.csv'))
    syngo_dist = pd.read_csv(os.path.join(SYNGO_DIR, 'syngo_distribution_gene_celltype.csv'))
    tf_dist = pd.read_csv(os.path.join(TF_DIR, 'tf_distribution_gene_celltype.csv'))

    # Panel A: SynGO L5 ET scatter
    ax = fig.add_subplot(gs[0, 0])
    l5et_s = syngo_deltas[syngo_deltas['cell_type'] == 'L5 ET']
    emp, pred = l5et_s['empirical_delta'].values, l5et_s['predicted_delta'].values
    ax.scatter(emp, pred, alpha=0.3, s=8, color='#E53935')
    lims = [min(emp.min(), pred.min()) - 0.1, max(emp.max(), pred.max()) + 0.1]
    ax.plot(lims, lims, 'k--', linewidth=0.8, alpha=0.5)
    m = compute_metrics(emp, pred)
    ax.set_title(f'A) SynGO L5 ET\nslope={m["slope"]:.2f}, int={m["intercept"]:.2f}, '
                 f'R²={m["R2"]:.3f}', fontsize=10)
    ax.set_xlabel('Empirical delta')
    ax.set_ylabel('Predicted delta')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel B: TF L5 ET scatter
    ax = fig.add_subplot(gs[0, 1])
    l5et_t = tf_deltas[tf_deltas['cell_type'] == 'L5 ET']
    emp, pred = l5et_t['empirical_delta'].values, l5et_t['predicted_delta'].values
    ax.scatter(emp, pred, alpha=0.5, s=15, color='#E53935')
    lims = [min(emp.min(), pred.min()) - 0.1, max(emp.max(), pred.max()) + 0.1]
    ax.plot(lims, lims, 'k--', linewidth=0.8, alpha=0.5)
    m = compute_metrics(emp, pred)
    ax.set_title(f'B) TF L5 ET\nslope={m["slope"]:.2f}, int={m["intercept"]:.2f}, '
                 f'R²={m["R2"]:.3f}', fontsize=10)
    ax.set_xlabel('Empirical delta')
    ax.set_ylabel('Predicted delta')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel C: SynGO prediction error by CT
    ax = fig.add_subplot(gs[0, 2])
    ct_errors_s = syngo_deltas.groupby('cell_type').apply(
        lambda g: (g['predicted_delta'] - g['empirical_delta']).mean()).sort_values()
    colors_s = ['#E53935' if ct == 'L5 ET' else '#FF9800' if ct == 'L5 IT'
                else '#2196F3' if ct in NEURONAL else '#9E9E9E'
                for ct in ct_errors_s.index]
    y = np.arange(len(ct_errors_s))
    ax.barh(y, ct_errors_s.values, color=colors_s, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(ct_errors_s.index, fontsize=8)
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Mean prediction error')
    ax.set_title('C) SynGO prediction error by CT', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel D: TF prediction error by CT
    ax = fig.add_subplot(gs[1, 0])
    ct_errors_t = tf_deltas.groupby('cell_type').apply(
        lambda g: (g['predicted_delta'] - g['empirical_delta']).mean()).sort_values()
    colors_t = ['#E53935' if ct == 'L5 ET' else '#FF9800' if ct == 'L5 IT'
                else '#2196F3' if ct in NEURONAL else '#9E9E9E'
                for ct in ct_errors_t.index]
    y = np.arange(len(ct_errors_t))
    ax.barh(y, ct_errors_t.values, color=colors_t, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(ct_errors_t.index, fontsize=8)
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Mean prediction error')
    ax.set_title('D) TF prediction error by CT', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel E: SynGO Wasserstein improvement by CT
    ax = fig.add_subplot(gs[1, 1])
    ct_wimp_s = syngo_dist.groupby('cell_type')['absolute_improvement'].median().sort_values()
    colors_ws = ['#E53935' if ct == 'L5 ET' else '#FF9800' if ct == 'L5 IT'
                 else '#4CAF50' if v > 0 else '#F44336'
                 for ct, v in zip(ct_wimp_s.index, ct_wimp_s.values)]
    y = np.arange(len(ct_wimp_s))
    ax.barh(y, ct_wimp_s.values, color=colors_ws, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(ct_wimp_s.index, fontsize=8)
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Median Wasserstein improvement')
    ax.set_title('E) SynGO distribution improvement', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel F: TF Wasserstein improvement by CT
    ax = fig.add_subplot(gs[1, 2])
    ct_wimp_t = tf_dist.groupby('cell_type')['absolute_improvement'].median().sort_values()
    colors_wt = ['#E53935' if ct == 'L5 ET' else '#FF9800' if ct == 'L5 IT'
                 else '#4CAF50' if v > 0 else '#F44336'
                 for ct, v in zip(ct_wimp_t.index, ct_wimp_t.values)]
    y = np.arange(len(ct_wimp_t))
    ax.barh(y, ct_wimp_t.values, color=colors_wt, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(ct_wimp_t.index, fontsize=8)
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Median Wasserstein improvement')
    ax.set_title('F) TF distribution improvement', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.suptitle('L5 ET diagnostic: ImputeVAE SynGO & TF validation',
                 fontsize=14, fontweight='bold')
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(OUT_DIR, f'l5et_diagnostic_main.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved l5et_diagnostic_main.png/pdf")


def plot_top_gene_errors():
    """Top gene error bar plots."""
    print("  Plotting l5et_top_gene_errors...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for col, (prefix, label) in enumerate([('syngo', 'SynGO'), ('tf', 'TF')]):
        over = pd.read_csv(os.path.join(OUT_DIR, f'{prefix}_l5et_top_overpredicted.csv'))
        under = pd.read_csv(os.path.join(OUT_DIR, f'{prefix}_l5et_top_underpredicted.csv'))

        # Overpredicted
        ax = axes[0, col]
        y = np.arange(min(20, len(over)))
        ax.barh(y, over['prediction_error'].values[:20], color='#E53935', alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(over['gene'].values[:20], fontsize=7)
        ax.set_xlabel('Prediction error (pred - emp)')
        ax.set_title(f'{label}: Top 20 overpredicted in L5 ET', fontsize=10)
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Underpredicted
        ax = axes[1, col]
        y = np.arange(min(20, len(under)))
        ax.barh(y, under['prediction_error'].values[:20], color='#1565C0', alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(under['gene'].values[:20], fontsize=7)
        ax.set_xlabel('Prediction error (pred - emp)')
        ax.set_title(f'{label}: Top 20 underpredicted in L5 ET', fontsize=10)
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('L5 ET gene-level prediction errors', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(OUT_DIR, f'l5et_top_gene_errors.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved l5et_top_gene_errors.png/pdf")


def plot_profile_similarity():
    """Profile similarity heatmap."""
    print("  Plotting l5et_profile_similarity...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_i, (prefix, label) in enumerate([('syngo', 'SynGO'), ('tf', 'TF')]):
        sim = pd.read_csv(os.path.join(OUT_DIR, f'{prefix}_l5et_profile_similarity.csv'))
        ax = axes[ax_i]

        queries = ['L5ET_imputed', 'L5ET_nucleus', 'L5ET_empirical_cell']
        ref_cts = sorted(sim['reference_cell_type'].unique())

        # Build matrix
        matrix = np.zeros((len(queries), len(ref_cts)))
        for qi, q in enumerate(queries):
            for ri, rct in enumerate(ref_cts):
                row = sim[(sim['query_profile'] == q) & (sim['reference_cell_type'] == rct)]
                matrix[qi, ri] = row['pearson'].values[0] if len(row) > 0 else np.nan

        im = ax.imshow(matrix, cmap='RdYlBu_r', vmin=0.7, vmax=1.0, aspect='auto')
        ax.set_xticks(range(len(ref_cts)))
        ax.set_xticklabels(ref_cts, fontsize=7, rotation=45, ha='right')
        ax.set_yticks(range(len(queries)))
        ax.set_yticklabels(['Imputed', 'Nucleus', 'Emp. cell'], fontsize=9)

        # Annotate
        for qi in range(len(queries)):
            for ri in range(len(ref_cts)):
                val = matrix[qi, ri]
                color = 'white' if val < 0.85 else 'black'
                ax.text(ri, qi, f'{val:.3f}', ha='center', va='center',
                        fontsize=6, color=color)

        # Highlight L5 ET and L5 IT columns
        for ri, rct in enumerate(ref_cts):
            if rct == 'L5 ET':
                ax.axvline(ri, color='red', linewidth=2, alpha=0.5)
            elif rct == 'L5 IT':
                ax.axvline(ri, color='orange', linewidth=2, alpha=0.5)

        ax.set_title(f'{label}', fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.6, label='Pearson r')

    fig.suptitle('L5 ET profile similarity to empirical cell-type means\n'
                 '(red line = L5 ET, orange line = L5 IT)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(OUT_DIR, f'l5et_profile_similarity.{ext}'),
                    dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved l5et_profile_similarity.png/pdf")


# ============================================================
# README
# ============================================================

def write_readme(comp_df, bias_df, ref_ran):
    """Write diagnostic README."""

    # Extract key numbers
    syngo_l5et = comp_df[(comp_df['gene_set'] == 'SynGO') &
                         (comp_df['cell_type'] == 'L5 ET')].iloc[0]
    syngo_l5it = comp_df[(comp_df['gene_set'] == 'SynGO') &
                         (comp_df['cell_type'] == 'L5 IT')].iloc[0]
    tf_l5et = comp_df[(comp_df['gene_set'] == 'JASPAR_TF') &
                      (comp_df['cell_type'] == 'L5 ET')].iloc[0]

    bias_syngo = bias_df[(bias_df['gene_set'] == 'SynGO') &
                         (bias_df['cell_type'] == 'L5 ET')].iloc[0]
    bias_syngo_l5it = bias_df[(bias_df['gene_set'] == 'SynGO') &
                              (bias_df['cell_type'] == 'L5 IT')].iloc[0]

    readme = f"""# L5 ET Diagnostic Analysis

## Purpose
Diagnose why L5 ET is a recurrent outlier in ImputeVAE SynGO and TF validation.
This analysis does NOT modify primary SynGO or TF results.

## Inputs
All primary comparisons use existing validation-only CSVs from:
- `validation/syngo/syngo_brain_validation/`
- `validation/tf/tf_brain_validation/`

Cells and nuclei are unpaired populations; mean-shift is computed from
independent cell-type-level averages.

## Key L5 ET Numbers

### SynGO
| Metric | L5 ET | L5 IT |
|--------|-------|-------|
| n_cell | {int(syngo_l5et['n_cell'])} | {int(syngo_l5it['n_cell'])} |
| n_nucleus | {int(syngo_l5et['n_nucleus'])} | {int(syngo_l5it['n_nucleus'])} |
| Empirical mean shift | {syngo_l5et['empirical_mean_shift']:.4f} | {syngo_l5it['empirical_mean_shift']:.4f} |
| Predicted mean shift | {syngo_l5et['predicted_mean_shift']:.4f} | {syngo_l5it['predicted_mean_shift']:.4f} |
| Prediction error | {syngo_l5et['prediction_error']:.4f} | {syngo_l5it['prediction_error']:.4f} |
| Pearson | {syngo_l5et['pearson']:.4f} | {syngo_l5it['pearson']:.4f} |
| R2 | {syngo_l5et['R2']:.4f} | {syngo_l5it['R2']:.4f} |
| Slope | {syngo_l5et['slope']:.3f} | {syngo_l5it['slope']:.3f} |
| Intercept | {syngo_l5et['intercept']:.4f} | {syngo_l5it['intercept']:.4f} |

### TF
| Metric | L5 ET | L5 IT (ref) |
|--------|-------|-------------|
| Prediction error | {tf_l5et['prediction_error']:.4f} | {comp_df[(comp_df['gene_set'] == 'JASPAR_TF') & (comp_df['cell_type'] == 'L5 IT')].iloc[0]['prediction_error']:.4f} |
| Pearson | {tf_l5et['pearson']:.4f} | {comp_df[(comp_df['gene_set'] == 'JASPAR_TF') & (comp_df['cell_type'] == 'L5 IT')].iloc[0]['pearson']:.4f} |
| R2 | {tf_l5et['R2']:.4f} | {comp_df[(comp_df['gene_set'] == 'JASPAR_TF') & (comp_df['cell_type'] == 'L5 IT')].iloc[0]['R2']:.4f} |

## Is L5 ET globally over-shifted or gene-specific?
- SynGO L5 ET: slope={bias_syngo['slope']:.3f}, intercept={bias_syngo['intercept']:.4f}
- SynGO L5 IT: slope={bias_syngo_l5it['slope']:.3f}, intercept={bias_syngo_l5it['intercept']:.4f}
- L5 ET fraction overpredicted: {bias_syngo['fraction_overpredicted']:.1%}

The high intercept ({bias_syngo['intercept']:.3f}) and fraction overpredicted ({bias_syngo['fraction_overpredicted']:.1%})
indicate a **global positive bias** rather than a few gene-specific outliers.
The model systematically over-amplifies the nucleus-to-cell shift for L5 ET.

## Profile similarity
See `syngo_l5et_profile_similarity.csv` and `tf_l5et_profile_similarity.csv`.
These show whether the L5 ET imputed profile more closely resembles L5 IT or another
neuronal population than true L5 ET whole cells.

## Sample-size caveat
L5 ET has {int(syngo_l5et['n_cell'])} validation whole cells and {int(syngo_l5et['n_nucleus'])} validation nuclei.
While not flagged by the existing <20 low-sample threshold, 32 whole cells is still
small for stable cell-type mean estimation, especially for a biologically distinct
population like L5 ET (corticospinal projection neurons).

## All-cell reference sensitivity
{"This analysis was run (see l5et_validation_vs_all_cells_reference.csv)." if ref_ran else "SKIPPED (raw data not loaded)."}
{"This is diagnostic only, NOT primary validation, because training cells are included." if ref_ran else ""}

## Conclusion
L5 ET is a recurrent calibration stress-test population. ImputeVAE preserves much
of the gene ranking but over-amplifies the nucleus-to-cell shift magnitude in L5 ET,
especially for SynGO and TF genes. This likely reflects a combination of rare-cell-type
calibration, small validation whole-cell sample size, and unpaired cell/nucleus
composition differences. The main SynGO and TF conclusions are not driven by L5 ET
because sensitivity analyses excluding L5 ET remain consistent.

## Files
- `l5et_celltype_comparison_summary.csv` — CT-level metrics for SynGO and TF
- `syngo_l5et_gene_errors.csv` / `tf_l5et_gene_errors.csv` — Gene-level L5 ET errors
- `syngo_l5et_top_overpredicted.csv` / `syngo_l5et_top_underpredicted.csv` — Top SynGO errors
- `tf_l5et_top_overpredicted.csv` / `tf_l5et_top_underpredicted.csv` — Top TF errors
- `l5et_global_bias_summary.csv` — Global bias vs gene-specific
- `syngo_l5et_profile_similarity.csv` / `tf_l5et_profile_similarity.csv` — Profile similarity
- `validation_sample_counts_by_celltype.csv` — Sample-size diagnostic
- `l5et_validation_vs_all_cells_reference.csv` — Reference sensitivity (if run)
- `l5et_diagnostic_main.png/pdf` — Main 6-panel figure
- `l5et_top_gene_errors.png/pdf` — Top gene error bar plots
- `l5et_profile_similarity.png/pdf` — Profile similarity heatmap
- `l5et_reference_sensitivity.png/pdf` — Reference sensitivity scatter (if run)
"""
    with open(os.path.join(OUT_DIR, 'README.md'), 'w') as f:
        f.write(readme)
    print("  Saved README.md")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-reference', action='store_true',
                        help='Skip optional all-cell reference sensitivity')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # Check inputs
    missing = []
    for path in [
        os.path.join(SYNGO_DIR, 'syngo_gene_celltype_deltas.csv'),
        os.path.join(SYNGO_DIR, 'syngo_distribution_gene_celltype.csv'),
        os.path.join(TF_DIR, 'tf_gene_celltype_deltas.csv'),
        os.path.join(TF_DIR, 'tf_distribution_gene_celltype.csv'),
    ]:
        if not os.path.exists(path):
            missing.append(path)
    if missing:
        print("ERROR: Missing input files:")
        for p in missing:
            print(f"  {p}")
        sys.exit(1)

    # Run analyses
    comp_df = celltype_comparison_summary()
    gene_level_errors()
    bias_df = global_bias_analysis()
    profile_similarity()
    sample_size_diagnostic()
    ref_ran = reference_sensitivity(skip=args.skip_reference)

    # Plots
    print("\n" + "=" * 70)
    print("Generating figures")
    print("=" * 70)
    plot_main_diagnostic()
    plot_top_gene_errors()
    plot_profile_similarity()

    # README
    write_readme(comp_df, bias_df, ref_ran)

    # Final summary
    print("\n" + "=" * 70)
    print("Quality checks")
    print("=" * 70)
    generated = [f for f in os.listdir(OUT_DIR)
                 if f != 'l5et_diagnostic.py' and not f.startswith('.')]
    print(f"  Files generated: {len(generated)}")
    for f in sorted(generated):
        print(f"    {f}")
    print(f"\n  Missing input files: {len(missing)}")
    print(f"  Reference sensitivity run: {ref_ran}")

    # L5 ET vs L5 IT summary
    print("\n  L5 ET vs L5 IT summary:")
    for gs in ['SynGO', 'JASPAR_TF']:
        for ct in ['L5 ET', 'L5 IT']:
            r = comp_df[(comp_df['gene_set'] == gs) & (comp_df['cell_type'] == ct)]
            if len(r) > 0:
                r = r.iloc[0]
                print(f"    {gs} {ct}: error={r['prediction_error']:.4f}, "
                      f"R2={r['R2']:.4f}, slope={r['slope']:.3f}, "
                      f"W_imp={r['median_W_improvement']:.4f}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
