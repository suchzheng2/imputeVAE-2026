#!/usr/bin/env python3
"""
Sample-Level Gene Stratification Analysis.

Complements gene_stratification.py (gene-level) with per-cell metrics:
1. Cell reconstruction: per-cell corr(decoded, raw) across gene subsets
2. Nuclei cross-decode: per-cell corr vs cell-type mean across gene subsets
3. Variance ratio within cell types across gene subsets

Answers: does the model reconstruct/impute certain gene categories better
at the individual cell level? Is per-cell variation preserved differently
for different gene categories?
"""

import gc
import json
import os
import sys
import argparse
import warnings

import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats, sparse

warnings.filterwarnings('ignore')

sys.path.insert(0, '/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE')
from imputeVAE.models import VAE_JSD
from imputeVAE.models_cond import VAE_JSD_Cond

CELL_PATH = "10X_cells_v2_v3_AIBS/"
NUC_PATH = "10X_nuclei_v2_v3_AIBS/"
CELL_ID = 0
NUCLEI_ID = 1


# =============================================================================
# Data loading (same pattern as comprehensive_eval.py / recon_eval.py)
# =============================================================================

def _resolve_raw_path(data_path):
    return os.path.join(
        "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi",
        data_path, "data.h5ad",
    )


def load_and_filter_data(data_path, batch, gene_list_path):
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


def load_model(run_folder, model_class, device):
    with open(os.path.join(run_folder, 'arguments.json')) as f:
        args = json.load(f)
    if model_class == 'VAE_JSD_Cond':
        model = VAE_JSD_Cond(
            input_size=args['num_feature'],
            num_internal=args['num_internal'],
            num_latent=args['num_latent'],
            embed_dim=args.get('embed_dim', 8),
        )
    else:
        model = VAE_JSD(
            input_size=args['num_feature'],
            num_internal=args['num_internal'],
            num_latent=args['num_latent'],
        )
    model.load_state_dict(torch.load(os.path.join(run_folder, 'models', 'model.pt'),
                                      map_location=device))
    model.to(device)
    model.eval()
    return model


def run_inference(model, X, modality_id, has_cond, device, batch_size=256):
    """Run same-modality reconstruction."""
    X_tensor = torch.tensor(X, dtype=torch.float32)
    recon_list = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            batch = X_tensor[i:i+batch_size].to(device)
            if has_cond:
                _, decoded, _, _ = model(batch, modality_id=modality_id)
            else:
                _, decoded, _, _ = model(batch)
            recon_list.append(decoded.cpu().numpy())
    return np.concatenate(recon_list, axis=0)


def run_cross_decode(model, X, source_id, target_id, device, batch_size=256):
    """Cross-decode: encode with source modality, decode with target."""
    X_tensor = torch.tensor(X, dtype=torch.float32)
    cross_list = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            batch = X_tensor[i:i+batch_size].to(device)
            out = model.decode_cross(batch, source_modality_id=source_id,
                                      target_modality_id=target_id)
            cross_list.append(out.cpu().numpy())
    return np.concatenate(cross_list, axis=0)


# =============================================================================
# Gene strata definitions
# =============================================================================

IEGS = {'Fos', 'Fosb', 'Jun', 'Junb', 'Jund', 'Arc', 'Egr1', 'Egr2',
        'Egr3', 'Egr4', 'Nr4a1', 'Nr4a2', 'Nr4a3', 'Npas4', 'Bdnf',
        'Homer1', 'Dusp1', 'Dusp5', 'Dusp6', 'Btg2'}

NEURONAL_MARKERS = {'Rbfox3', 'Snap25', 'Syt1', 'Stmn2', 'Map2', 'Tubb3',
                    'Dcx', 'Tbr1', 'Satb2', 'Bcl11b', 'Fezf2'}

GLIAL_MARKERS = {'Gfap', 'Aqp4', 'Aldh1l1', 'Slc1a3',
                 'Mbp', 'Plp1', 'Mog', 'Mag', 'Cnp',
                 'Cx3cr1', 'Csf1r', 'P2ry12', 'Tmem119',
                 'Pdgfra', 'Cspg4', 'Sox10'}


def build_gene_strata(gene_names, X_cell, ct_cell, X_nuc, ct_nuc):
    """Build boolean masks for gene strata.

    Returns dict: stratum_name -> boolean array of length n_genes.
    """
    n_genes = len(gene_names)

    # Compute per-gene mean expression across all cells/nuclei
    cell_mean = X_cell.mean(axis=0)
    nuc_mean = X_nuc.mean(axis=0)
    abs_diff = np.abs(cell_mean - nuc_mean)

    # Rank by absolute difference
    diff_rank = np.argsort(np.argsort(-abs_diff))  # 0 = most different

    # Expression quintiles (based on cell mean)
    expr_quintiles = pd.qcut(cell_mean, 5, labels=False, duplicates='drop')

    # Ribosomal
    ribo_mask = np.array([g.startswith(('Rps', 'Rpl')) for g in gene_names])

    # Named gene sets
    ieg_mask = np.array([g in IEGS for g in gene_names])
    neuronal_mask = np.array([g in NEURONAL_MARKERS for g in gene_names])
    glial_mask = np.array([g in GLIAL_MARKERS for g in gene_names])

    strata = {
        'all': np.ones(n_genes, dtype=bool),
        'top_1k_diff': diff_rank < 1000,
        'top_2k_diff': diff_rank < 2000,
        'bottom_1k_diff': diff_rank >= (n_genes - 1000),
        'IEG': ieg_mask,
        'ribosomal': ribo_mask,
        'neuronal_marker': neuronal_mask,
        'glial_marker': glial_mask,
    }

    # Expression quintiles
    for q in range(5):
        label = f'expr_Q{q+1}'
        strata[label] = expr_quintiles == q

    # Report gene counts
    for name, mask in strata.items():
        n = mask.sum()
        if n > 0:
            print(f"  Stratum '{name}': {n} genes")
        else:
            print(f"  Stratum '{name}': 0 genes (SKIPPING)")

    # Report IEG presence
    present_iegs = [g for g in gene_names if g in IEGS]
    missing_iegs = IEGS - set(gene_names)
    if missing_iegs:
        print(f"  IEGs missing from gene list: {sorted(missing_iegs)}")
    if present_iegs:
        print(f"  IEGs present: {sorted(present_iegs)}")

    return strata


# =============================================================================
# Per-cell metrics
# =============================================================================

def _vectorized_rowwise_corr(A, B):
    """Vectorized per-row Pearson correlation between matrices A and B.

    Returns 1D array of correlations, one per row.
    """
    A_centered = A - A.mean(axis=1, keepdims=True)
    B_centered = B - B.mean(axis=1, keepdims=True)
    num = (A_centered * B_centered).sum(axis=1)
    den = np.sqrt((A_centered**2).sum(axis=1) * (B_centered**2).sum(axis=1))
    den[den == 0] = np.nan
    return num / den


def percell_corr(X_pred, X_true, gene_mask, min_genes=10):
    """Per-cell Pearson correlation across genes in gene_mask.

    Returns 1D array of correlations, one per cell. Vectorized.
    """
    if gene_mask.sum() < min_genes:
        return np.full(X_pred.shape[0], np.nan)

    return _vectorized_rowwise_corr(X_pred[:, gene_mask], X_true[:, gene_mask])


def percell_corr_to_ctmean(X_pred, ct_means, cell_types, gene_mask, min_genes=10):
    """Per-cell correlation of predicted profile vs its cell-type mean in raw cells.
    Vectorized: builds a reference matrix from cell-type means.
    """
    if gene_mask.sum() < min_genes:
        return np.full(X_pred.shape[0], np.nan)

    # Build reference matrix: each row is the cell-type mean for that cell's type
    ref = np.zeros_like(X_pred[:, gene_mask])
    valid = np.ones(X_pred.shape[0], dtype=bool)
    for i, ct in enumerate(cell_types):
        if ct in ct_means:
            ref[i] = ct_means[ct][gene_mask]
        else:
            valid[i] = False

    corrs = _vectorized_rowwise_corr(X_pred[:, gene_mask], ref)
    corrs[~valid] = np.nan
    return corrs


def variance_ratio_per_ct(X_pred, X_ref, ct_pred, ct_ref, gene_mask, min_cells=10):
    """Variance ratio (pred / ref) per cell type for genes in gene_mask.

    Returns dict: ct -> ratio.
    """
    if gene_mask.sum() < 2:
        return {}

    common_cts = set(ct_pred) & set(ct_ref)
    results = {}
    for ct in sorted(common_cts):
        pred_mask = ct_pred == ct
        ref_mask = ct_ref == ct
        if pred_mask.sum() < min_cells or ref_mask.sum() < min_cells:
            continue
        var_pred = X_pred[pred_mask][:, gene_mask].var(axis=0).mean()
        var_ref = X_ref[ref_mask][:, gene_mask].var(axis=0).mean()
        if var_ref > 0:
            results[ct] = float(var_pred / var_ref)
        else:
            results[ct] = np.nan
    return results


# =============================================================================
# Aggregation
# =============================================================================

def aggregate_percell(corrs, cell_types):
    """Aggregate per-cell correlations by cell type."""
    rows = []
    valid = ~np.isnan(corrs)

    # Overall
    if valid.sum() > 0:
        rows.append({
            'cell_type': 'ALL',
            'n_cells': int(valid.sum()),
            'median_corr': float(np.nanmedian(corrs)),
            'mean_corr': float(np.nanmean(corrs)),
            'std_corr': float(np.nanstd(corrs)),
            'q25_corr': float(np.nanpercentile(corrs, 25)),
            'q75_corr': float(np.nanpercentile(corrs, 75)),
        })

    for ct in sorted(set(cell_types)):
        mask = (cell_types == ct) & valid
        if mask.sum() == 0:
            continue
        c = corrs[mask]
        rows.append({
            'cell_type': ct,
            'n_cells': int(mask.sum()),
            'median_corr': float(np.median(c)),
            'mean_corr': float(np.mean(c)),
            'std_corr': float(np.std(c)),
            'q25_corr': float(np.percentile(c, 25)),
            'q75_corr': float(np.percentile(c, 75)),
        })
    return pd.DataFrame(rows)


# =============================================================================
# Visualization
# =============================================================================

def plot_percell_by_stratum(results_dict, title, output_path):
    """Box plot of per-cell correlations across gene strata.

    results_dict: {stratum_name: corr_array}
    """
    # Filter to strata with enough data
    plot_strata = [(name, corrs) for name, corrs in results_dict.items()
                   if not np.all(np.isnan(corrs)) and len(corrs) > 0]
    if not plot_strata:
        print(f"  No data to plot for {title}")
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    data = [corrs[~np.isnan(corrs)] for _, corrs in plot_strata]
    labels = [name for name, _ in plot_strata]

    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False,
                    medianprops=dict(color='black', linewidth=2))

    colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel('Per-cell Pearson r')
    ax.set_title(title, fontweight='bold')
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.axhline(0.8, color='gray', ls='--', lw=0.8, alpha=0.5)

    # Add median values
    for i, d in enumerate(data):
        med = np.median(d)
        ax.text(i + 1, med + 0.01, f'{med:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(output_path)}")


def plot_heatmap_ct_stratum(all_results, cell_types_list, strata_names, title, output_path):
    """Heatmap: rows=cell types, cols=gene strata, fill=median per-cell correlation."""
    cts = sorted(set(cell_types_list))
    matrix = np.full((len(cts), len(strata_names)), np.nan)

    for j, stratum in enumerate(strata_names):
        corrs = all_results.get(stratum)
        if corrs is None:
            continue
        for i, ct in enumerate(cts):
            mask = (cell_types_list == ct) & ~np.isnan(corrs)
            if mask.sum() > 0:
                matrix[i, j] = np.median(corrs[mask])

    fig, ax = plt.subplots(figsize=(max(10, len(strata_names) * 1.2), max(6, len(cts) * 0.4)))

    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0.3, vmax=1.0)
    ax.set_xticks(range(len(strata_names)))
    ax.set_xticklabels(strata_names, rotation=35, ha='right', fontsize=9)
    ax.set_yticks(range(len(cts)))
    ax.set_yticklabels(cts, fontsize=9)
    ax.set_title(title, fontweight='bold')

    # Add text annotations
    for i in range(len(cts)):
        for j in range(len(strata_names)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', fontsize=7)

    plt.colorbar(im, ax=ax, label='Median per-cell Pearson r')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(output_path)}")


def plot_variance_ratio(var_results, strata_names, title, output_path):
    """Grouped bar chart of variance ratios by stratum."""
    # Collect data: for each stratum, median variance ratio across cell types
    medians = []
    q25s = []
    q75s = []
    valid_strata = []

    for name in strata_names:
        ratios = var_results.get(name, {})
        vals = [v for v in ratios.values() if not np.isnan(v)]
        if len(vals) > 0:
            medians.append(np.median(vals))
            q25s.append(np.percentile(vals, 25))
            q75s.append(np.percentile(vals, 75))
            valid_strata.append(name)

    if not valid_strata:
        print(f"  No variance ratio data to plot")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(valid_strata))
    medians = np.array(medians)
    q25s = np.array(q25s)
    q75s = np.array(q75s)

    ax.bar(x, medians, color='#FF9800', alpha=0.85, edgecolor='black', lw=0.5)
    ax.errorbar(x, medians, yerr=[medians - q25s, q75s - medians],
                fmt='none', color='black', capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels(valid_strata, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Variance ratio (imputed / real cell)')
    ax.set_title(title, fontweight='bold')
    ax.axhline(1.0, color='red', ls='--', lw=1, label='Perfect match')
    ax.legend()

    for i, m in enumerate(medians):
        ax.text(i, m + 0.01, f'{m:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(output_path)}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Sample-level gene stratification")
    p.add_argument("--run_folder", required=True)
    p.add_argument("--model_class", default="VAE_JSD_Cond",
                   choices=["VAE_JSD", "VAE_JSD_Cond"])
    p.add_argument("--gene_list", default="data/training_gene_list_v3.csv")
    p.add_argument("--max_per_type", type=int, default=2000)
    p.add_argument("--output_dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    run_folder = args.run_folder
    if not os.path.isabs(run_folder):
        run_folder = os.path.join('/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE',
                                   run_folder)

    has_cond = args.model_class == 'VAE_JSD_Cond'
    output_dir = args.output_dir or os.path.join(run_folder, 'evaluation', 'sample_stratification')
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("SAMPLE-LEVEL GENE STRATIFICATION")
    print("=" * 80)
    print(f"Run: {run_folder}")
    print(f"Model: {args.model_class}")
    print(f"Output: {output_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- Load data ---
    print("\n--- Loading data ---")
    cell_adata = load_and_filter_data(CELL_PATH, 'c3A', args.gene_list)
    nuc_adata = load_and_filter_data(NUC_PATH, 'n3A', args.gene_list)

    cell_adata = stratified_subsample(cell_adata, args.max_per_type)
    nuc_adata = stratified_subsample(nuc_adata, args.max_per_type)

    X_cell = cell_adata.X if not sparse.issparse(cell_adata.X) else cell_adata.X.toarray()
    X_nuc = nuc_adata.X if not sparse.issparse(nuc_adata.X) else nuc_adata.X.toarray()
    ct_cell = cell_adata.obs['cell_type'].values.astype(str)
    ct_nuc = nuc_adata.obs['cell_type'].values.astype(str)
    gene_names = cell_adata.var['gene_names'].values.astype(str)

    # Free AnnData objects — we only need the numpy arrays from here
    del cell_adata, nuc_adata
    gc.collect()

    print(f"  Cells: {X_cell.shape}, Nuclei: {X_nuc.shape}")

    # --- Load model and run inference ---
    print("\n--- Loading model ---")
    model = load_model(run_folder, args.model_class, device)

    print("\n--- Running inference ---")
    # Cell reconstruction
    cell_recon = run_inference(model, X_cell, CELL_ID, has_cond, device)
    print(f"  Cell recon: {cell_recon.shape}")

    if has_cond:
        # Nuclei cross-decode
        nuc_cross = run_cross_decode(model, X_nuc, NUCLEI_ID, CELL_ID, device)
        print(f"  Nuclei cross-decode: {nuc_cross.shape}")
    else:
        nuc_cross = run_inference(model, X_nuc, NUCLEI_ID, has_cond, device)
        print(f"  Nuclei decoded (shared): {nuc_cross.shape}")

    # Free model
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Cell-type means from raw cells
    ct_means = {}
    for ct in sorted(set(ct_cell)):
        ct_means[ct] = X_cell[ct_cell == ct].mean(axis=0)

    # --- Build gene strata ---
    print("\n--- Building gene strata ---")
    strata = build_gene_strata(gene_names, X_cell, ct_cell, X_nuc, ct_nuc)

    # Define order for display
    strata_order = ['all', 'top_1k_diff', 'top_2k_diff', 'bottom_1k_diff',
                    'IEG', 'ribosomal', 'neuronal_marker', 'glial_marker',
                    'expr_Q1', 'expr_Q2', 'expr_Q3', 'expr_Q4', 'expr_Q5']
    strata_order = [s for s in strata_order if s in strata and strata[s].sum() >= 10]

    # =========================================================================
    # 1. Cell reconstruction: per-cell corr across gene subsets
    # =========================================================================
    print("\n--- Cell reconstruction per-cell correlations ---")
    cell_recon_results = {}
    cell_recon_tables = []

    for sname in strata_order:
        mask = strata[sname]
        corrs = percell_corr(cell_recon, X_cell, mask)
        cell_recon_results[sname] = corrs
        df = aggregate_percell(corrs, ct_cell)
        df['stratum'] = sname
        df['n_genes'] = int(mask.sum())
        cell_recon_tables.append(df)
        med = np.nanmedian(corrs)
        print(f"  {sname:25s}: median={med:.4f} ({mask.sum()} genes)")

    cell_recon_df = pd.concat(cell_recon_tables, ignore_index=True)
    cell_recon_df.to_csv(os.path.join(output_dir, 'cell_recon_by_stratum.csv'), index=False)

    # =========================================================================
    # 2. Nuclei cross-decode: per-cell corr vs cell-type mean
    # =========================================================================
    print("\n--- Nuclei cross-decode per-cell corr vs cell-type mean ---")
    crossdec_results = {}
    crossdec_tables = []

    for sname in strata_order:
        mask = strata[sname]
        corrs = percell_corr_to_ctmean(nuc_cross, ct_means, ct_nuc, mask)
        crossdec_results[sname] = corrs
        df = aggregate_percell(corrs, ct_nuc)
        df['stratum'] = sname
        df['n_genes'] = int(mask.sum())
        crossdec_tables.append(df)
        med = np.nanmedian(corrs)
        print(f"  {sname:25s}: median={med:.4f} ({mask.sum()} genes)")

    crossdec_df = pd.concat(crossdec_tables, ignore_index=True)
    crossdec_df.to_csv(os.path.join(output_dir, 'crossdec_corr_by_stratum.csv'), index=False)

    # =========================================================================
    # 3. Variance ratio per cell type per stratum
    # =========================================================================
    print("\n--- Variance ratio (imputed / real cell) ---")
    var_results = {}
    var_tables = []

    for sname in strata_order:
        mask = strata[sname]
        ratios = variance_ratio_per_ct(nuc_cross, X_cell, ct_nuc, ct_cell, mask)
        var_results[sname] = ratios
        vals = [v for v in ratios.values() if not np.isnan(v)]
        med = np.median(vals) if vals else np.nan
        print(f"  {sname:25s}: median ratio={med:.4f} ({mask.sum()} genes)")
        for ct, ratio in ratios.items():
            var_tables.append({
                'cell_type': ct, 'stratum': sname,
                'n_genes': int(mask.sum()), 'variance_ratio': ratio,
            })

    var_df = pd.DataFrame(var_tables)
    var_df.to_csv(os.path.join(output_dir, 'variance_ratio_by_stratum.csv'), index=False)

    # =========================================================================
    # Visualizations
    # =========================================================================
    print("\n--- Generating plots ---")

    # Figure 1: Box plots of per-cell correlations
    plot_percell_by_stratum(
        cell_recon_results,
        'Cell reconstruction: per-cell correlation by gene stratum',
        os.path.join(output_dir, 'cell_recon_percell_by_stratum.png'))

    plot_percell_by_stratum(
        crossdec_results,
        'Nuclei cross-decode: per-cell corr vs CT mean by gene stratum',
        os.path.join(output_dir, 'crossdec_percell_by_stratum.png'))

    # Figure 2: Heatmaps (cell type x stratum)
    plot_heatmap_ct_stratum(
        cell_recon_results, ct_cell, strata_order,
        'Cell recon: median per-cell corr (cell type x gene stratum)',
        os.path.join(output_dir, 'cell_recon_heatmap_ct_stratum.png'))

    plot_heatmap_ct_stratum(
        crossdec_results, ct_nuc, strata_order,
        'Cross-decode: median per-cell corr vs CT mean (cell type x gene stratum)',
        os.path.join(output_dir, 'crossdec_heatmap_ct_stratum.png'))

    # Figure 3: Variance ratio
    plot_variance_ratio(
        var_results, strata_order,
        'Cross-decode variance ratio by gene stratum',
        os.path.join(output_dir, 'variance_ratio_by_stratum.png'))

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Print compact summary table
    summary_rows = []
    for sname in strata_order:
        cell_med = np.nanmedian(cell_recon_results.get(sname, [np.nan]))
        cross_med = np.nanmedian(crossdec_results.get(sname, [np.nan]))
        var_vals = [v for v in var_results.get(sname, {}).values() if not np.isnan(v)]
        var_med = np.median(var_vals) if var_vals else np.nan
        n_genes = int(strata[sname].sum())
        summary_rows.append({
            'stratum': sname,
            'n_genes': n_genes,
            'cell_recon_median_r': cell_med,
            'crossdec_vs_ctmean_median_r': cross_med,
            'variance_ratio_median': var_med,
        })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(os.path.join(output_dir, 'summary.csv'), index=False)

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()
