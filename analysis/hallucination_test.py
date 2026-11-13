"""
Hallucination vs Genuine Recovery Tests

Tests whether the conditioned decoder genuinely uses per-cell info from z,
or collapses to cell-type mean predictions.

Tests:
A1. Per-cell variance: Var(imputed | cell type) vs Var(real cells | cell type)
A2. Within-cell-type ranking preservation: Spearman(raw nuc, imputed) within each CT
B1. z ablation: replace z with cell-type mean z, check how much output changes
B2. z variance decomposition: between-CT vs within-CT variance in latent space
"""

import gc
import json
import os
import sys
import warnings

import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats, sparse
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

sys.path.insert(0, '/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE')
from imputeVAE.models import VAE_JSD
from imputeVAE.models_cond import VAE_JSD_Cond

import argparse

# ---- CLI args (all optional, backward-compatible defaults) ----
_parser = argparse.ArgumentParser(description='Hallucination vs Genuine Recovery Tests')
_parser.add_argument('--models', type=str, default=None,
                     help='Comma-separated model preset names to test (default: all)')
_parser.add_argument('--gene_list', type=str, default=None,
                     help='Override gene list path')
_parser.add_argument('--output_dir', type=str, default=None,
                     help='Override output directory')
_cli_args = _parser.parse_args()

# ---- All model presets ----
ALL_MODELS = {
    'MSE_cond': {
        'run_folder': 'run/run_2026-04-02-02-02-35_3345839',
        'gene_list': 'data/training_gene_list_v3.csv',
    },
    'MSE_cond_xr': {
        'run_folder': 'run/run_2026-04-02-10-51-32_3738876',
        'gene_list': 'data/training_gene_list_v3.csv',
    },
    'pct5_cond_xr': {
        'run_folder': 'run/run_2026-04-03-03-44-41_2957436',
        'gene_list': 'data/gene_list_perct_5pct.csv',
    },
    'pct5v2_cond': {
        'run_folder': 'run/run_2026-04-07-02-12-05_3779214',
        'gene_list': 'data/gene_list_perct_5pct.csv',
    },
    'pct5v2_cond_xr': {
        'run_folder': 'run/run_2026-04-07-02-12-05_3551465',
        'gene_list': 'data/gene_list_perct_5pct.csv',
    },
}

# Select models to run
if _cli_args.models:
    _selected = [m.strip() for m in _cli_args.models.split(',')]
    MODELS = {k: v for k, v in ALL_MODELS.items() if k in _selected}
    if not MODELS:
        raise ValueError(f"No matching models found. Available: {list(ALL_MODELS.keys())}")
else:
    # Default: original two models only (backward compatible)
    MODELS = {k: ALL_MODELS[k] for k in ['MSE_cond', 'MSE_cond_xr']}

# Gene list: CLI override > per-model default (applied per model below)
_cli_gene_list = _cli_args.gene_list

# Output directory
OUTPUT_DIR = _cli_args.output_dir or 'notes/hallucination_tests'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CELL_PATH = "10X_cells_v2_v3_AIBS/"
NUC_PATH = "10X_nuclei_v2_v3_AIBS/"
# GENE_LIST set per-model in the loop below
CELL_ID = 0
NUCLEI_ID = 1

# =====================================================================
# Data loading (reuse from comprehensive_eval)
# =====================================================================
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

def load_model(run_folder, device):
    args_path = os.path.join(run_folder, 'arguments.json')
    model_path = os.path.join(run_folder, 'models', 'model.pt')
    with open(args_path) as f:
        args = json.load(f)
    model = VAE_JSD_Cond(
        input_size=args['num_feature'],
        num_internal=args['num_internal'],
        num_latent=args['num_latent'],
        embed_dim=args.get('embed_dim', 8),
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, args['num_latent']

# =====================================================================
# Main
# =====================================================================
print("=" * 80)
print("HALLUCINATION vs GENUINE RECOVERY TESTS")
print("=" * 80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

MAX_PER_TYPE = 2000

def stratified_subsample(adata, max_per_type):
    idx = []
    for ct in adata.obs['cell_type'].unique():
        ct_idx = np.where(adata.obs['cell_type'].values == ct)[0]
        if len(ct_idx) > max_per_type:
            ct_idx = np.random.choice(ct_idx, max_per_type, replace=False)
        idx.extend(ct_idx)
    return adata[sorted(idx)].copy()

# Cache loaded data by gene_list to avoid reloading for models with same gene set
_data_cache = {}

def _load_data_for_gene_list(gene_list_path):
    if gene_list_path in _data_cache:
        return _data_cache[gene_list_path]
    print(f"\n--- Loading data (gene_list={gene_list_path}) ---")
    np.random.seed(42)
    cell_adata = load_and_filter_data(CELL_PATH, 'c3A', gene_list_path)
    nuc_adata = load_and_filter_data(NUC_PATH, 'n3A', gene_list_path)
    cell_adata = stratified_subsample(cell_adata, MAX_PER_TYPE)
    nuc_adata = stratified_subsample(nuc_adata, MAX_PER_TYPE)
    X_cell = cell_adata.X if isinstance(cell_adata.X, np.ndarray) else cell_adata.X.toarray()
    X_nuc = nuc_adata.X if isinstance(nuc_adata.X, np.ndarray) else nuc_adata.X.toarray()
    ct_cell = cell_adata.obs['cell_type'].values
    ct_nuc = nuc_adata.obs['cell_type'].values
    common_cts = sorted(set(ct_cell) & set(ct_nuc))
    cell_ct_means = {}
    for ct in common_cts:
        cell_ct_means[ct] = X_cell[ct_cell == ct].mean(axis=0)
    print(f"Cell: {X_cell.shape}, Nuclei: {X_nuc.shape}, Common CTs: {len(common_cts)}")
    result = (X_cell, X_nuc, ct_cell, ct_nuc, common_cts, cell_ct_means)
    _data_cache[gene_list_path] = result
    return result

# =====================================================================
# Run tests per model
# =====================================================================
for model_name, model_info in MODELS.items():
    gene_list = _cli_gene_list or model_info.get('gene_list', 'data/training_gene_list_v3.csv')
    X_cell, X_nuc, ct_cell, ct_nuc, common_cts, cell_ct_means = _load_data_for_gene_list(gene_list)
    n_genes = X_cell.shape[1]
    print(f"\n{'=' * 70}")
    print(f"MODEL: {model_name}")
    print(f"{'=' * 70}")

    model, num_latent = load_model(model_info['run_folder'], device)

    # --- Encode all nuclei → get z vectors ---
    print("\n  Encoding nuclei...")
    X_nuc_tensor = torch.tensor(X_nuc, dtype=torch.float32)
    z_list = []
    imputed_list = []
    with torch.no_grad():
        for i in range(0, X_nuc.shape[0], 256):
            batch = X_nuc_tensor[i:i+256].to(device)
            encoded = model.encoder(batch)
            mu = model.mu(encoded)
            z_list.append(mu.cpu().numpy())

            # Cross-decode: nuclei → cell
            emb_cell = model._get_modality_emb(batch.size(0), CELL_ID, device)
            z_cond = torch.cat([mu, emb_cell], dim=1)
            decoded = model.decoder(z_cond)
            imputed_list.append(decoded.cpu().numpy())

    z_all = np.concatenate(z_list, axis=0)  # [N_nuc, latent_dim]
    imputed = np.concatenate(imputed_list, axis=0)  # [N_nuc, n_genes]
    print(f"  z shape: {z_all.shape}, imputed shape: {imputed.shape}")

    # =================================================================
    # TEST B2: z variance decomposition (between-CT vs within-CT)
    # =================================================================
    print("\n  [B2] z variance decomposition")

    grand_mean_z = z_all.mean(axis=0)
    ss_total = np.sum((z_all - grand_mean_z) ** 2, axis=0)  # per dimension

    ss_between = np.zeros(num_latent)
    ss_within = np.zeros(num_latent)
    for ct in common_cts:
        mask = ct_nuc == ct
        ct_z = z_all[mask]
        ct_mean = ct_z.mean(axis=0)
        n_ct = mask.sum()
        ss_between += n_ct * (ct_mean - grand_mean_z) ** 2
        ss_within += np.sum((ct_z - ct_mean) ** 2, axis=0)

    frac_between = ss_between / (ss_total + 1e-12)
    mean_frac_between = frac_between.mean()
    median_frac_between = np.median(frac_between)

    print(f"  Fraction of z variance that is between-CT:")
    print(f"    Mean across dimensions:   {mean_frac_between:.4f}")
    print(f"    Median across dimensions: {median_frac_between:.4f}")
    print(f"    Min: {frac_between.min():.4f}, Max: {frac_between.max():.4f}")

    # Histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(frac_between, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(mean_frac_between, color='red', linestyle='--', label=f'Mean={mean_frac_between:.3f}')
    ax.set_xlabel('Fraction of variance (between-CT)')
    ax.set_ylabel('# latent dimensions')
    ax.set_title(f'{model_name}: z variance decomposition')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/B2_z_variance_{model_name}.png', dpi=150)
    plt.close()

    # =================================================================
    # TEST B1: z ablation — replace z with cell-type mean z
    # =================================================================
    print("\n  [B1] z ablation — replace z with cell-type mean z")

    # Compute cell-type mean z
    ct_mean_z = {}
    for ct in common_cts:
        ct_mean_z[ct] = z_all[ct_nuc == ct].mean(axis=0)

    # Build ablated z: each nucleus gets its cell-type mean z
    z_ablated = np.zeros_like(z_all)
    for ct in common_cts:
        mask = ct_nuc == ct
        z_ablated[mask] = ct_mean_z[ct]

    # Decode ablated z with cell embedding
    z_ablated_tensor = torch.tensor(z_ablated, dtype=torch.float32)
    imputed_ablated_list = []
    with torch.no_grad():
        for i in range(0, z_ablated.shape[0], 256):
            z_batch = z_ablated_tensor[i:i+256].to(device)
            emb_cell = model._get_modality_emb(z_batch.size(0), CELL_ID, device)
            z_cond = torch.cat([z_batch, emb_cell], dim=1)
            decoded = model.decoder(z_cond)
            imputed_ablated_list.append(decoded.cpu().numpy())

    imputed_ablated = np.concatenate(imputed_ablated_list, axis=0)

    # Compare: how much does ablation change the output?
    # Per-cell: corr(original_imputed, ablated_imputed)
    cell_corrs_orig_vs_ablated = []
    for i in range(imputed.shape[0]):
        r, _ = stats.pearsonr(imputed[i], imputed_ablated[i])
        cell_corrs_orig_vs_ablated.append(r)
    cell_corrs_orig_vs_ablated = np.array(cell_corrs_orig_vs_ablated)

    print(f"  Per-cell corr(original imputed, ablated imputed):")
    print(f"    Median: {np.median(cell_corrs_orig_vs_ablated):.6f}")
    print(f"    Mean:   {np.mean(cell_corrs_orig_vs_ablated):.6f}")
    print(f"    Min:    {np.min(cell_corrs_orig_vs_ablated):.6f}")
    print(f"    Max:    {np.max(cell_corrs_orig_vs_ablated):.6f}")

    # Per-cell MSE between original and ablated
    mse_orig_vs_ablated = np.mean((imputed - imputed_ablated) ** 2, axis=1)
    mse_orig_vs_zero = np.mean(imputed ** 2, axis=1)  # scale reference

    print(f"  Per-cell MSE(original, ablated):")
    print(f"    Median: {np.median(mse_orig_vs_ablated):.6f}")
    print(f"    Relative to signal: {np.median(mse_orig_vs_ablated / (mse_orig_vs_zero + 1e-12)):.6f}")

    # Per-cell-type breakdown
    print(f"\n  Per-cell-type ablation effect:")
    print(f"  {'Cell Type':<20} {'Corr(orig,ablated)':>20} {'Relative MSE':>15}")
    ablation_rows = []
    for ct in common_cts:
        mask = ct_nuc == ct
        med_corr = np.median(cell_corrs_orig_vs_ablated[mask])
        med_rel_mse = np.median(mse_orig_vs_ablated[mask] / (mse_orig_vs_zero[mask] + 1e-12))
        print(f"  {ct:<20} {med_corr:>20.6f} {med_rel_mse:>15.6f}")
        ablation_rows.append({'cell_type': ct, 'n': mask.sum(),
                              'corr_orig_ablated': med_corr,
                              'relative_mse': med_rel_mse})
    pd.DataFrame(ablation_rows).to_csv(f'{OUTPUT_DIR}/B1_ablation_{model_name}.csv', index=False)

    # =================================================================
    # TEST A1: Per-cell variance comparison
    # =================================================================
    print("\n  [A1] Per-cell variance (imputed vs real cell)")

    variance_rows = []
    print(f"  {'Cell Type':<20} {'Var(imputed)':>14} {'Var(real cell)':>14} {'Ratio':>10}")
    for ct in common_cts:
        nuc_mask = ct_nuc == ct
        cell_mask = ct_cell == ct

        # Per-gene variance within cell type, averaged across genes
        imp_var = imputed[nuc_mask].var(axis=0).mean()
        cell_var = X_cell[cell_mask].var(axis=0).mean()
        ratio = imp_var / (cell_var + 1e-12)

        print(f"  {ct:<20} {imp_var:>14.6f} {cell_var:>14.6f} {ratio:>10.4f}")
        variance_rows.append({'cell_type': ct, 'n_nuc': nuc_mask.sum(), 'n_cell': cell_mask.sum(),
                              'var_imputed': imp_var, 'var_real_cell': cell_var, 'ratio': ratio})

    var_df = pd.DataFrame(variance_rows)
    var_df.to_csv(f'{OUTPUT_DIR}/A1_variance_{model_name}.csv', index=False)
    overall_ratio = var_df['var_imputed'].mean() / var_df['var_real_cell'].mean()
    print(f"  Overall variance ratio (imputed/real): {overall_ratio:.4f}")

    # =================================================================
    # TEST A2: Within-cell-type ranking preservation
    # =================================================================
    print("\n  [A2] Within-cell-type ranking preservation (Spearman)")

    # For each cell type, for each gene, compute Spearman(raw_nuc, imputed) within that CT
    ranking_rows = []
    print(f"  {'Cell Type':<20} {'Median Spearman':>16} {'Frac > 0':>10} {'N genes tested':>15}")
    for ct in common_cts:
        nuc_mask = ct_nuc == ct
        raw_nuc_ct = X_nuc[nuc_mask]
        imp_ct = imputed[nuc_mask]

        if nuc_mask.sum() < 10:
            continue

        gene_spearmans = []
        for g in range(n_genes):
            # Only test genes with some variance in raw nuclei
            if raw_nuc_ct[:, g].std() > 0 and imp_ct[:, g].std() > 0:
                rho, _ = spearmanr(raw_nuc_ct[:, g], imp_ct[:, g])
                if not np.isnan(rho):
                    gene_spearmans.append(rho)

        gene_spearmans = np.array(gene_spearmans)
        med = np.median(gene_spearmans) if len(gene_spearmans) > 0 else 0
        frac_pos = (gene_spearmans > 0).mean() if len(gene_spearmans) > 0 else 0

        print(f"  {ct:<20} {med:>16.4f} {frac_pos:>10.3f} {len(gene_spearmans):>15}")
        ranking_rows.append({'cell_type': ct, 'n_nuc': nuc_mask.sum(),
                             'median_spearman': med, 'frac_positive': frac_pos,
                             'n_genes_tested': len(gene_spearmans)})

    rank_df = pd.DataFrame(ranking_rows)
    rank_df.to_csv(f'{OUTPUT_DIR}/A2_ranking_{model_name}.csv', index=False)
    overall_spearman = rank_df['median_spearman'].mean()
    overall_frac_pos = rank_df['frac_positive'].mean()
    print(f"  Overall: median Spearman={overall_spearman:.4f}, frac>0={overall_frac_pos:.3f}")

    # =================================================================
    # Summary figure
    # =================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{model_name}: Hallucination Tests', fontsize=14)

    # A1: Variance ratio bar chart
    ax = axes[0, 0]
    x_pos = range(len(common_cts))
    ratios = [var_df.loc[var_df['cell_type'] == ct, 'ratio'].values[0] for ct in common_cts]
    ax.bar(x_pos, ratios, color='steelblue', edgecolor='black')
    ax.axhline(1.0, color='red', linestyle='--', label='Ratio=1 (same variance)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(common_cts, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Var(imputed) / Var(real cell)')
    ax.set_title('A1: Per-CT variance ratio')
    ax.legend(fontsize=8)

    # B1: Ablation correlation histogram
    ax = axes[0, 1]
    ax.hist(cell_corrs_orig_vs_ablated, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(np.median(cell_corrs_orig_vs_ablated), color='red', linestyle='--',
               label=f'Median={np.median(cell_corrs_orig_vs_ablated):.4f}')
    ax.set_xlabel('Corr(original imputed, ablated imputed)')
    ax.set_ylabel('# cells')
    ax.set_title('B1: z ablation effect')
    ax.legend(fontsize=8)

    # A2: Within-CT Spearman bar chart
    ax = axes[1, 0]
    spearmans = [rank_df.loc[rank_df['cell_type'] == ct, 'median_spearman'].values[0]
                 for ct in common_cts if ct in rank_df['cell_type'].values]
    ct_labels = [ct for ct in common_cts if ct in rank_df['cell_type'].values]
    ax.bar(range(len(ct_labels)), spearmans, color='coral', edgecolor='black')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(range(len(ct_labels)))
    ax.set_xticklabels(ct_labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Median Spearman (within CT)')
    ax.set_title('A2: Within-CT ranking preservation')

    # B2: z variance decomposition histogram
    ax = axes[1, 1]
    ax.hist(frac_between, bins=30, edgecolor='black', alpha=0.7, color='mediumpurple')
    ax.axvline(mean_frac_between, color='red', linestyle='--',
               label=f'Mean={mean_frac_between:.3f}')
    ax.set_xlabel('Fraction between-CT variance')
    ax.set_ylabel('# latent dimensions')
    ax.set_title('B2: z variance decomposition')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/summary_{model_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Summary figure saved: {OUTPUT_DIR}/summary_{model_name}.png")

    del model
    torch.cuda.empty_cache()
    gc.collect()

# =====================================================================
# Cross-model comparison summary
# =====================================================================
print("\n" + "=" * 80)
print("CROSS-MODEL SUMMARY")
print("=" * 80)

for model_name in MODELS:
    var_df = pd.read_csv(f'{OUTPUT_DIR}/A1_variance_{model_name}.csv')
    rank_df = pd.read_csv(f'{OUTPUT_DIR}/A2_ranking_{model_name}.csv')
    abl_df = pd.read_csv(f'{OUTPUT_DIR}/B1_ablation_{model_name}.csv')

    print(f"\n  {model_name}:")
    print(f"    A1 variance ratio (imputed/real): {var_df['ratio'].mean():.4f}")
    print(f"    A2 within-CT Spearman:            {rank_df['median_spearman'].mean():.4f}")
    print(f"    A2 frac genes with positive rank:  {rank_df['frac_positive'].mean():.3f}")
    print(f"    B1 ablation corr(orig, ablated):   {abl_df['corr_orig_ablated'].median():.6f}")

print("\nDone!")
