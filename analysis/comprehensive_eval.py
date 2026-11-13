"""
Comprehensive evaluation of MSE conditioned decoder models.

Computes:
1. Cell-type mean baseline (correlation, R²)
2. Multi-class logistic regression (train on raw cell → predict raw nuclei vs cross-decoded nuclei)
3. DE gene preservation (one-vs-rest Wilcoxon, top-K overlap)
4. Per-cell correlation (each imputed nucleus vs its cell-type mean in raw cell)
5. Gene-gene coexpression preservation

Runs on three MSE models:
- MSE shared (run_2026-04-01-14-01-16_1946018)
- MSE cond no-xr (run_2026-04-02-02-02-35_3345839)
- MSE cond+xr (run_2026-04-02-10-51-32_3738876)
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

import argparse

sys.path.insert(0, '/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE')
from imputeVAE.models import VAE_JSD
from imputeVAE.models_cond import VAE_JSD_Cond
from imputeVAE.data import load_raw_data, filter_by_protocol
from imputeVAE.utils import load_args_from_json

# ---- CLI args ----
_parser = argparse.ArgumentParser(description='Comprehensive evaluation')
_parser.add_argument('--models', type=str, default=None,
                     help='Comma-separated model preset names to test (default: all)')
_parser.add_argument('--gene_list', type=str, default=None,
                     help='Override gene list path')
_parser.add_argument('--output_dir', type=str, default=None,
                     help='Override output directory')
_cli_args = _parser.parse_args()

# ---- All model presets ----
ALL_MODELS = {
    'MSE_shared': {
        'run_folder': 'run/run_2026-04-01-14-01-16_1946018',
        'model_class': 'VAE_JSD',
        'has_cross_decode': False,
        'gene_list': 'data/training_gene_list_v3.csv',
    },
    'MSE_cond': {
        'run_folder': 'run/run_2026-04-02-02-02-35_3345839',
        'model_class': 'VAE_JSD_Cond',
        'has_cross_decode': True,
        'gene_list': 'data/training_gene_list_v3.csv',
    },
    'MSE_cond_xr': {
        'run_folder': 'run/run_2026-04-02-10-51-32_3738876',
        'model_class': 'VAE_JSD_Cond',
        'has_cross_decode': True,
        'gene_list': 'data/training_gene_list_v3.csv',
    },
    'pct5v2_cond': {
        'run_folder': 'run/run_2026-04-07-02-12-05_3779214',
        'model_class': 'VAE_JSD_Cond',
        'has_cross_decode': True,
        'gene_list': 'data/gene_list_perct_5pct.csv',
    },
    'pct5v2_cond_xr': {
        'run_folder': 'run/run_2026-04-07-02-12-05_3551465',
        'model_class': 'VAE_JSD_Cond',
        'has_cross_decode': True,
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
    # Default: original three models only (backward compatible)
    MODELS = {k: ALL_MODELS[k] for k in ['MSE_shared', 'MSE_cond', 'MSE_cond_xr']}

# Gene list: CLI override > per-model default (applied per model below)
_cli_gene_list = _cli_args.gene_list

OUTPUT_DIR = _cli_args.output_dir or 'notes/comprehensive_eval'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CELL_PATH = "10X_cells_v2_v3_AIBS/"
NUC_PATH = "10X_nuclei_v2_v3_AIBS/"

CELL_ID = 0
NUCLEI_ID = 1


def _resolve_raw_path(data_path):
    return os.path.join(
        "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi",
        data_path, "data.h5ad",
    )


def load_and_filter_data(data_path, batch, gene_list_path):
    """Load raw data, filter to v3 + gene list, return adata with log1p in .X"""
    adata = ad.read_h5ad(_resolve_raw_path(data_path))

    # Add meta_batch_id
    if 'meta_batch_id' not in adata.obs:
        adata.obs['meta_batch_id'] = adata.obs['batch_id'].apply(lambda x: x.split('.')[0])

    # Filter to v3
    adata = adata[adata.obs['meta_batch_id'].str.endswith(batch)].copy()

    # Common genes
    common_genes = pd.read_csv('data/common_genes.csv', header=0)['common_genes'].tolist()
    adata = adata[:, adata.var['gene_names'].isin(common_genes)].copy()

    # Common cell types
    common_ct = pd.read_csv('data/common_cell_types.csv', header=0)['common_cell_types'].tolist()
    adata = adata[adata.obs['cell_type'].isin(common_ct)].copy()

    # Gene list filter
    gene_list = pd.read_csv(gene_list_path, header=None)[0].tolist()
    adata = adata[:, adata.var['gene_names'].isin(gene_list)].copy()

    # Sort by gene name
    adata = adata[:, adata.var['gene_names'].argsort()].copy()

    # Compute log1p
    X = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
    adata.X = np.log1p(X).astype(np.float32)

    print(f"  Loaded {data_path} batch={batch}: {adata.shape}")
    return adata


def load_model(model_name, model_info, device):
    """Load trained model from checkpoint."""
    run_folder = model_info['run_folder']
    args_path = os.path.join(run_folder, 'arguments.json')
    model_path = os.path.join(run_folder, 'models', 'model.pt')

    with open(args_path) as f:
        args = json.load(f)

    num_feature = args['num_feature']
    num_internal = args['num_internal']
    num_latent = args['num_latent']

    if model_info['model_class'] == 'VAE_JSD_Cond':
        embed_dim = args.get('embed_dim', 8)
        model = VAE_JSD_Cond(
            input_size=num_feature,
            num_internal=num_internal,
            num_latent=num_latent,
            embed_dim=embed_dim,
        )
    else:
        model = VAE_JSD(
            input_size=num_feature,
            num_internal=num_internal,
            num_latent=num_latent,
        )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"  Loaded model: {model_name} from {run_folder}")
    return model


def run_inference(model, data, model_info, device, batch_size=256):
    """Run model inference, return reconstructed and (optionally) cross-decoded data."""
    X = torch.tensor(data, dtype=torch.float32)
    n = X.shape[0]

    recon_list = []
    cross_dec_list = []
    has_cross = model_info['has_cross_decode']

    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = X[i:i+batch_size].to(device)

            if has_cross:
                # Forward with nuclei embedding
                _, decoded, mu, _ = model(batch, modality_id=NUCLEI_ID)
                recon_list.append(decoded.cpu().numpy())

                # Cross-decode: nuclei input → cell embedding
                cross_dec = model.decode_cross(batch, source_modality_id=NUCLEI_ID,
                                                target_modality_id=CELL_ID)
                cross_dec_list.append(cross_dec.cpu().numpy())
            else:
                _, decoded, mu, _ = model(batch)
                recon_list.append(decoded.cpu().numpy())

    result = {'recon': np.concatenate(recon_list, axis=0)}
    if has_cross:
        result['cross_dec'] = np.concatenate(cross_dec_list, axis=0)
    return result


def compute_celltype_means(X, cell_types):
    """Compute per-cell-type mean expression. Returns dict: ct -> 1D array."""
    means = {}
    for ct in sorted(set(cell_types)):
        mask = cell_types == ct
        means[ct] = X[mask].mean(axis=0)
    return means


# =====================================================================
# Main
# =====================================================================
print("="*80)
print("COMPREHENSIVE EVALUATION")
print("="*80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# =====================================================================
# 1. Data loading with caching (supports per-model gene lists)
# =====================================================================
np.random.seed(42)
MAX_PER_TYPE = 2000

def stratified_subsample(adata, max_per_type):
    idx = []
    for ct in adata.obs['cell_type'].unique():
        ct_idx = np.where(adata.obs['cell_type'].values == ct)[0]
        if len(ct_idx) > max_per_type:
            ct_idx = np.random.choice(ct_idx, max_per_type, replace=False)
        idx.extend(ct_idx)
    return adata[sorted(idx)].copy()

_data_cache = {}  # gene_list_path -> (X_cell, X_nuc, ct_cell, ct_nuc, gene_names, common_cts, cell_means, nuc_means, cell_percell_to_mean)

def _load_data_for_gene_list(gene_list_path):
    """Load and cache data for a given gene list."""
    if gene_list_path in _data_cache:
        return _data_cache[gene_list_path]

    print(f"\n--- Loading data for gene list: {gene_list_path} ---")
    cell_adata = load_and_filter_data(CELL_PATH, 'c3A', gene_list_path)
    nuc_adata = load_and_filter_data(NUC_PATH, 'n3A', gene_list_path)

    cell_adata = stratified_subsample(cell_adata, MAX_PER_TYPE)
    nuc_adata = stratified_subsample(nuc_adata, MAX_PER_TYPE)
    print(f"After subsampling: Cell={cell_adata.shape}, Nuclei={nuc_adata.shape}")

    X_cell = cell_adata.X if not sparse.issparse(cell_adata.X) else cell_adata.X.toarray()
    X_nuc = nuc_adata.X if not sparse.issparse(nuc_adata.X) else nuc_adata.X.toarray()
    ct_cell = cell_adata.obs['cell_type'].values
    ct_nuc = nuc_adata.obs['cell_type'].values
    gene_names = cell_adata.var['gene_names'].values

    common_cts = sorted(set(ct_cell) & set(ct_nuc))
    print(f"Common cell types: {len(common_cts)}")

    # Cell-type means
    cell_means = compute_celltype_means(X_cell, ct_cell)
    nuc_means = compute_celltype_means(X_nuc, ct_nuc)

    # Baseline: per-cell corr of real cell to its cell-type mean (ceiling)
    cell_percell_to_mean = []
    for i, ct in enumerate(ct_cell):
        if ct in cell_means and np.std(X_cell[i]) > 0:
            r, _ = stats.pearsonr(X_cell[i], cell_means[ct])
            cell_percell_to_mean.append(r)

    # Baseline: raw nuclei means vs raw cell means
    baseline_mean_matrix = np.array([cell_means[ct] for ct in common_cts])
    nuc_mean_matrix = np.array([nuc_means[ct] for ct in common_cts])
    n_genes = baseline_mean_matrix.shape[1]
    nuc_vs_cell_gene_corr = []
    for g in range(n_genes):
        if np.std(nuc_mean_matrix[:, g]) > 0 and np.std(baseline_mean_matrix[:, g]) > 0:
            r, _ = stats.pearsonr(nuc_mean_matrix[:, g], baseline_mean_matrix[:, g])
        else:
            r = 0
        nuc_vs_cell_gene_corr.append(r)

    print(f"Raw nuclei means vs raw cell means (per-gene median corr): "
          f"{np.nanmedian(nuc_vs_cell_gene_corr):.4f}")
    print(f"Real cells: per-cell corr to own cell-type mean (median): "
          f"{np.nanmedian(cell_percell_to_mean):.4f}")

    cached = (X_cell, X_nuc, ct_cell, ct_nuc, gene_names, common_cts,
              cell_means, nuc_means, cell_percell_to_mean)
    _data_cache[gene_list_path] = cached
    return cached

# =====================================================================
# 2. Run inference for each model
# =====================================================================
all_results = {}
all_data = {}  # model_name -> cached data tuple

for model_name, model_info in MODELS.items():
    print(f"\n--- Model: {model_name} ---")

    # Resolve gene list
    gene_list_path = _cli_gene_list or model_info['gene_list']
    data = _load_data_for_gene_list(gene_list_path)
    all_data[model_name] = data
    X_cell, X_nuc, ct_cell, ct_nuc, gene_names, common_cts, cell_means, nuc_means, cell_percell_to_mean = data

    model = load_model(model_name, model_info, device)

    # Inference on nuclei
    results = run_inference(model, X_nuc, model_info, device)
    all_results[model_name] = results

    # Also run on cell data for reconstruction quality
    X_cell_tensor = torch.tensor(X_cell, dtype=torch.float32)
    cell_recon_list = []
    with torch.no_grad():
        for i in range(0, X_cell.shape[0], 256):
            batch = X_cell_tensor[i:i+256].to(device)
            if model_info['has_cross_decode']:
                _, decoded, _, _ = model(batch, modality_id=CELL_ID)
            else:
                _, decoded, _, _ = model(batch)
            cell_recon_list.append(decoded.cpu().numpy())
    results['cell_recon'] = np.concatenate(cell_recon_list, axis=0)

    del model
    torch.cuda.empty_cache()
    gc.collect()

# =====================================================================
# 4. Evaluation metrics
# =====================================================================
summary_rows = []

for model_name, results in all_results.items():
    print(f"\n{'='*60}")
    print(f"EVALUATION: {model_name}")
    print(f"{'='*60}")

    model_info = MODELS[model_name]
    has_cross = model_info['has_cross_decode']

    # Unpack per-model data
    X_cell, X_nuc, ct_cell, ct_nuc, gene_names, common_cts, cell_means, nuc_means, cell_percell_to_mean = all_data[model_name]
    n_genes = X_cell.shape[1]

    # The imputed data: cross-decoded for cond models, reconstructed for shared
    if has_cross:
        imputed = results['cross_dec']  # nuclei → cell embedding
        imputed_label = 'cross_dec'
    else:
        imputed = results['recon']  # shared decoder output
        imputed_label = 'recon'

    nuc_recon = results['recon']  # nuclei reconstruction (same-modality)
    cell_recon = results['cell_recon']  # cell reconstruction

    # --- 4a. Cell-type mean correlation ---
    print("\n  [Cell-type mean correlation]")
    imputed_means = compute_celltype_means(imputed, ct_nuc)
    nuc_recon_means = compute_celltype_means(nuc_recon, ct_nuc)
    cell_recon_means = compute_celltype_means(cell_recon, ct_cell)

    # Per-gene correlation: imputed means vs raw cell means
    imp_mean_matrix = np.array([imputed_means.get(ct, np.zeros(n_genes)) for ct in common_cts])
    cell_mean_matrix = np.array([cell_means[ct] for ct in common_cts])
    nuc_raw_mean_matrix = np.array([nuc_means[ct] for ct in common_cts])

    imp_gene_corr = []
    imp_gene_r2 = []
    for g in range(n_genes):
        if np.std(imp_mean_matrix[:, g]) > 0 and np.std(cell_mean_matrix[:, g]) > 0:
            r, _ = stats.pearsonr(imp_mean_matrix[:, g], cell_mean_matrix[:, g])
            ss_res = np.sum((imp_mean_matrix[:, g] - cell_mean_matrix[:, g])**2)
            ss_tot = np.sum((cell_mean_matrix[:, g] - cell_mean_matrix[:, g].mean())**2)
            r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        else:
            r, r2 = 0, 0
        imp_gene_corr.append(r)
        imp_gene_r2.append(r2)

    # Cell reconstruction quality
    cell_recon_gene_corr = []
    for g in range(n_genes):
        cr_means = np.array([cell_recon_means.get(ct, np.zeros(n_genes))[g] for ct in common_cts])
        raw_means = cell_mean_matrix[:, g]
        if np.std(cr_means) > 0 and np.std(raw_means) > 0:
            r, _ = stats.pearsonr(cr_means, raw_means)
        else:
            r = 0
        cell_recon_gene_corr.append(r)

    print(f"  Imputed nuc→raw cell (per-gene median corr): {np.nanmedian(imp_gene_corr):.4f}")
    print(f"  Imputed nuc→raw cell (per-gene median R²):   {np.nanmedian(imp_gene_r2):.4f}")
    print(f"  Cell recon→raw cell (per-gene median corr):   {np.nanmedian(cell_recon_gene_corr):.4f}")

    # --- 4b. Per-cell correlation ---
    print("\n  [Per-cell correlation]")
    # Each imputed nucleus vs its cell-type mean in raw cell data
    percell_corr_to_cellmean = []
    for i, ct in enumerate(ct_nuc):
        if ct in cell_means and np.std(imputed[i]) > 0:
            r, _ = stats.pearsonr(imputed[i], cell_means[ct])
            percell_corr_to_cellmean.append(r)

    # Each imputed nucleus vs corresponding raw nucleus (same-sample reconstruction fidelity)
    percell_corr_to_raw_nuc = []
    for i in range(len(ct_nuc)):
        if np.std(imputed[i]) > 0 and np.std(X_nuc[i]) > 0:
            r, _ = stats.pearsonr(imputed[i], X_nuc[i])
            percell_corr_to_raw_nuc.append(r)

    # Cell reconstruction: per-cell correlation
    percell_cell_recon = []
    for i in range(len(ct_cell)):
        if np.std(cell_recon[i]) > 0 and np.std(X_cell[i]) > 0:
            r, _ = stats.pearsonr(cell_recon[i], X_cell[i])
            percell_cell_recon.append(r)

    print(f"  Imputed nuc vs raw cell-type mean (median): {np.nanmedian(percell_corr_to_cellmean):.4f}")
    print(f"  Imputed nuc vs raw nucleus (median):        {np.nanmedian(percell_corr_to_raw_nuc):.4f}")
    print(f"  Cell recon vs raw cell (median):             {np.nanmedian(percell_cell_recon):.4f}")
    print(f"  [Baseline] Real cell vs own ct mean (median): {np.nanmedian(cell_percell_to_mean):.4f}")

    # Per-cell-type breakdown
    print("\n  [Per-cell-type breakdown: imputed nuc vs raw cell-type mean]")
    ct_percell = {}
    for ct in common_cts:
        ct_mask = ct_nuc == ct
        ct_corrs = []
        for i in np.where(ct_mask)[0]:
            if ct in cell_means and np.std(imputed[i]) > 0:
                r, _ = stats.pearsonr(imputed[i], cell_means[ct])
                ct_corrs.append(r)
        if ct_corrs:
            ct_percell[ct] = np.median(ct_corrs)
            print(f"    {ct:<20} median corr: {np.median(ct_corrs):.4f} (n={len(ct_corrs)})")

    # --- 4c. Logistic regression classification ---
    print("\n  [Cell-type classification]")
    # Train on raw cell data, predict on raw nuclei and imputed nuclei
    scaler = StandardScaler()
    X_cell_scaled = scaler.fit_transform(X_cell)

    # Filter to common cell types for classification
    cell_ct_mask = np.isin(ct_cell, common_cts)
    nuc_ct_mask = np.isin(ct_nuc, common_cts)

    clf = LogisticRegression(
        multi_class='multinomial', solver='lbfgs',
        max_iter=1000, random_state=42, n_jobs=-1, C=1.0,
    )
    clf.fit(X_cell_scaled[cell_ct_mask], ct_cell[cell_ct_mask])

    # Predict on raw nuclei
    X_nuc_scaled = scaler.transform(X_nuc)
    raw_nuc_pred = clf.predict(X_nuc_scaled[nuc_ct_mask])
    raw_nuc_acc = accuracy_score(ct_nuc[nuc_ct_mask], raw_nuc_pred)
    raw_nuc_f1 = f1_score(ct_nuc[nuc_ct_mask], raw_nuc_pred, average='macro')

    # Predict on imputed nuclei
    imputed_scaled = scaler.transform(imputed)
    imp_pred = clf.predict(imputed_scaled[nuc_ct_mask])
    imp_acc = accuracy_score(ct_nuc[nuc_ct_mask], imp_pred)
    imp_f1 = f1_score(ct_nuc[nuc_ct_mask], imp_pred, average='macro')

    # Predict on raw cell (sanity check — should be very high)
    cell_pred = clf.predict(X_cell_scaled[cell_ct_mask])
    cell_acc = accuracy_score(ct_cell[cell_ct_mask], cell_pred)

    print(f"  Train on raw cell, predict on:")
    print(f"    Raw cell (sanity):  accuracy={cell_acc:.4f}")
    print(f"    Raw nuclei:         accuracy={raw_nuc_acc:.4f}, macro-F1={raw_nuc_f1:.4f}")
    print(f"    Imputed nuclei:     accuracy={imp_acc:.4f}, macro-F1={imp_f1:.4f}")
    print(f"    Improvement:        {(imp_acc - raw_nuc_acc)*100:+.1f}% accuracy")

    # Per-class report for imputed
    report = classification_report(ct_nuc[nuc_ct_mask], imp_pred, output_dict=True)
    report_df = pd.DataFrame(report).T
    report_df.to_csv(f'{OUTPUT_DIR}/classification_report_{model_name}.csv')

    # --- 4d. DE gene preservation ---
    print("\n  [DE gene preservation]")
    from scipy.stats import rankdata, ranksums

    def run_de_one_vs_rest(X, cell_types, top_k=100):
        """One-vs-rest Wilcoxon rank-sum DE. Returns dict: ct -> list of top-K gene indices."""
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
            # Rank by p-value (ascending), break ties by fold change
            order = np.lexsort((-np.abs(fold_changes), pvals))
            de_genes[ct] = order[:top_k]
        return de_genes

    TOP_K = 100

    # DE on raw cell data (ground truth)
    de_raw_cell = run_de_one_vs_rest(X_cell, ct_cell, top_k=TOP_K)

    # DE on imputed nuclei
    de_imputed = run_de_one_vs_rest(imputed, ct_nuc, top_k=TOP_K)

    # DE on raw nuclei (for comparison)
    de_raw_nuc = run_de_one_vs_rest(X_nuc, ct_nuc, top_k=TOP_K)

    # Compute overlap
    de_overlaps_imp = {}
    de_overlaps_raw_nuc = {}
    for ct in common_cts:
        if ct in de_raw_cell and ct in de_imputed:
            overlap = len(set(de_raw_cell[ct]) & set(de_imputed[ct]))
            de_overlaps_imp[ct] = overlap / TOP_K
        if ct in de_raw_cell and ct in de_raw_nuc:
            overlap = len(set(de_raw_cell[ct]) & set(de_raw_nuc[ct]))
            de_overlaps_raw_nuc[ct] = overlap / TOP_K

    mean_de_overlap_imp = np.mean(list(de_overlaps_imp.values())) if de_overlaps_imp else 0
    mean_de_overlap_raw = np.mean(list(de_overlaps_raw_nuc.values())) if de_overlaps_raw_nuc else 0

    print(f"  Top-{TOP_K} DE overlap with raw cell DE:")
    print(f"    Imputed nuclei:  {mean_de_overlap_imp:.3f} (mean across cell types)")
    print(f"    Raw nuclei:      {mean_de_overlap_raw:.3f} (mean across cell types)")
    print(f"    Improvement:     {(mean_de_overlap_imp - mean_de_overlap_raw)*100:+.1f}%")

    print(f"\n  Per-cell-type DE overlap (imputed):")
    for ct in common_cts:
        imp_ov = de_overlaps_imp.get(ct, 0)
        raw_ov = de_overlaps_raw_nuc.get(ct, 0)
        print(f"    {ct:<20} imputed={imp_ov:.2f}  raw_nuc={raw_ov:.2f}  delta={imp_ov-raw_ov:+.2f}")

    # --- 4e. Gene-gene coexpression preservation ---
    print("\n  [Gene-gene coexpression preservation]")
    # Top-K gene-gene correlations in raw cell, compare to imputed
    # Use cell-type means to avoid cell-count effects
    K_GENES = min(500, n_genes)  # Use top variable genes for tractability
    gene_var = X_cell.var(axis=0)
    top_var_idx = np.argsort(gene_var)[-K_GENES:]

    cell_coexp = np.corrcoef(X_cell[:, top_var_idx].T)
    imp_coexp = np.corrcoef(imputed[:, top_var_idx].T)
    nuc_coexp = np.corrcoef(X_nuc[:, top_var_idx].T)

    # Upper triangle only
    triu_idx = np.triu_indices(K_GENES, k=1)
    cell_flat = cell_coexp[triu_idx]
    imp_flat = imp_coexp[triu_idx]
    nuc_flat = nuc_coexp[triu_idx]

    # Remove NaN
    valid = ~(np.isnan(cell_flat) | np.isnan(imp_flat) | np.isnan(nuc_flat))
    coexp_corr_imp, _ = stats.spearmanr(cell_flat[valid], imp_flat[valid])
    coexp_corr_nuc, _ = stats.spearmanr(cell_flat[valid], nuc_flat[valid])

    print(f"  Coexpression matrix correlation (Spearman, top {K_GENES} variable genes):")
    print(f"    Imputed vs raw cell:  {coexp_corr_imp:.4f}")
    print(f"    Raw nuclei vs raw cell: {coexp_corr_nuc:.4f}")

    # --- Collect summary ---
    summary_rows.append({
        'model': model_name,
        'imputed_mode': imputed_label,
        # Cell-type mean correlation
        'gene_corr_imp_vs_cell': np.nanmedian(imp_gene_corr),
        'gene_r2_imp_vs_cell': np.nanmedian(imp_gene_r2),
        'gene_corr_cellrecon': np.nanmedian(cell_recon_gene_corr),
        # Per-cell
        'percell_imp_vs_cellmean': np.nanmedian(percell_corr_to_cellmean),
        'percell_imp_vs_rawnuc': np.nanmedian(percell_corr_to_raw_nuc),
        'percell_cellrecon': np.nanmedian(percell_cell_recon),
        'percell_realcell_ceiling': np.nanmedian(cell_percell_to_mean),
        # Classification
        'clf_raw_cell_acc': cell_acc,
        'clf_raw_nuc_acc': raw_nuc_acc,
        'clf_raw_nuc_f1': raw_nuc_f1,
        'clf_imp_acc': imp_acc,
        'clf_imp_f1': imp_f1,
        'clf_improvement': imp_acc - raw_nuc_acc,
        # DE preservation
        'de_overlap_imputed': mean_de_overlap_imp,
        'de_overlap_raw_nuc': mean_de_overlap_raw,
        'de_improvement': mean_de_overlap_imp - mean_de_overlap_raw,
        # Coexpression
        'coexp_imp_vs_cell': coexp_corr_imp,
        'coexp_nuc_vs_cell': coexp_corr_nuc,
    })

# =====================================================================
# 5. Summary table
# =====================================================================
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(f'{OUTPUT_DIR}/summary_table.csv', index=False)

# Pretty print
for _, row in summary_df.iterrows():
    print(f"\n  Model: {row['model']}")
    print(f"  --- Cell-type mean correlation ---")
    print(f"    Imputed→raw cell gene corr: {row['gene_corr_imp_vs_cell']:.4f}")
    print(f"    Imputed→raw cell gene R²:   {row['gene_r2_imp_vs_cell']:.4f}")
    print(f"    Cell recon gene corr:        {row['gene_corr_cellrecon']:.4f}")
    print(f"  --- Per-cell ---")
    print(f"    Imputed vs cell-type mean:   {row['percell_imp_vs_cellmean']:.4f}")
    print(f"    Imputed vs raw nucleus:      {row['percell_imp_vs_rawnuc']:.4f}")
    print(f"    Cell recon vs raw cell:       {row['percell_cellrecon']:.4f}")
    print(f"    [Ceiling] Real cell vs mean:  {row['percell_realcell_ceiling']:.4f}")
    print(f"  --- Classification ---")
    print(f"    Raw nuclei accuracy:          {row['clf_raw_nuc_acc']:.4f}")
    print(f"    Imputed nuclei accuracy:      {row['clf_imp_acc']:.4f} ({row['clf_improvement']*100:+.1f}%)")
    print(f"  --- DE preservation ---")
    print(f"    Imputed DE overlap:           {row['de_overlap_imputed']:.3f}")
    print(f"    Raw nuclei DE overlap:        {row['de_overlap_raw_nuc']:.3f}")
    print(f"  --- Coexpression ---")
    print(f"    Imputed vs raw cell:          {row['coexp_imp_vs_cell']:.4f}")
    print(f"    Raw nuclei vs raw cell:       {row['coexp_nuc_vs_cell']:.4f}")

# =====================================================================
# 6. Save per-cell-type details
# =====================================================================
# DE overlap per cell type for each model
de_detail_rows = []
for model_name in all_results:
    # Recompute DE for this model (already done above, just save)
    pass  # Already saved in the loop above

print(f"\nAll results saved to: {OUTPUT_DIR}/")
print("Done!")
