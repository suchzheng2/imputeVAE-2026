"""
Reconstruction-only evaluation.

Evaluates same-modality reconstruction quality (NOT cross-decode):
- Reconstructed cell: encode cell → decode with cell embedding → compare to raw cell
- Reconstructed nuclei: encode nuc → decode with nuc embedding → compare to raw nuc

Metrics: per-cell correlation, classification, DE preservation, coexpression.
This complements comprehensive_eval.py which focused on cross-decode imputation.
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
from scipy.stats import ranksums
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

sys.path.insert(0, '/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE')
from imputeVAE.models import VAE_JSD
from imputeVAE.models_cond import VAE_JSD_Cond

OUTPUT_DIR = 'notes/recon_eval'
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = {
    'MSE_shared': {
        'run_folder': 'run/run_2026-04-01-14-01-16_1946018',
        'model_class': 'VAE_JSD',
        'has_cond': False,
    },
    'MSE_cond': {
        'run_folder': 'run/run_2026-04-02-02-02-35_3345839',
        'model_class': 'VAE_JSD_Cond',
        'has_cond': True,
    },
    'MSE_cond_xr': {
        'run_folder': 'run/run_2026-04-02-10-51-32_3738876',
        'model_class': 'VAE_JSD_Cond',
        'has_cond': True,
    },
}

CELL_PATH = "10X_cells_v2_v3_AIBS/"
NUC_PATH = "10X_nuclei_v2_v3_AIBS/"
GENE_LIST = "data/training_gene_list_v3.csv"
CELL_ID = 0
NUCLEI_ID = 1


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


def load_model(model_info, device):
    run_folder = model_info['run_folder']
    with open(os.path.join(run_folder, 'arguments.json')) as f:
        args = json.load(f)
    if model_info['has_cond']:
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


def reconstruct(model, X, modality_id, has_cond, device, batch_size=256):
    """Reconstruct data with same-modality embedding."""
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


def run_de_one_vs_rest(X, cell_types, top_k=100):
    """One-vs-rest Wilcoxon rank-sum DE. Returns dict: ct -> top-K gene indices."""
    de_genes = {}
    for ct in sorted(set(cell_types)):
        mask = cell_types == ct
        if mask.sum() < 5 or (~mask).sum() < 5:
            continue
        pvals = []
        fold_changes = []
        for g in range(X.shape[1]):
            _, p = ranksums(X[mask, g], X[~mask, g])
            fc = X[mask, g].mean() - X[~mask, g].mean()
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
print("=" * 80)
print("RECONSTRUCTION EVALUATION")
print("=" * 80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load data
print("\n--- Loading data ---")
cell_adata = load_and_filter_data(CELL_PATH, 'c3A', GENE_LIST)
nuc_adata = load_and_filter_data(NUC_PATH, 'n3A', GENE_LIST)

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

cell_adata = stratified_subsample(cell_adata, MAX_PER_TYPE)
nuc_adata = stratified_subsample(nuc_adata, MAX_PER_TYPE)

X_cell = cell_adata.X if isinstance(cell_adata.X, np.ndarray) else cell_adata.X.toarray()
X_nuc = nuc_adata.X if isinstance(nuc_adata.X, np.ndarray) else nuc_adata.X.toarray()
ct_cell = cell_adata.obs['cell_type'].values
ct_nuc = nuc_adata.obs['cell_type'].values
n_genes = X_cell.shape[1]
common_cts = sorted(set(ct_cell) & set(ct_nuc))

print(f"Cell: {X_cell.shape}, Nuclei: {X_nuc.shape}, Common CTs: {len(common_cts)}")

# Compute raw DE (ground truth)
print("\n--- Computing raw DE genes ---")
de_raw_cell = run_de_one_vs_rest(X_cell, ct_cell, top_k=100)
de_raw_nuc = run_de_one_vs_rest(X_nuc, ct_nuc, top_k=100)
print(f"  Raw cell DE: {len(de_raw_cell)} cell types")
print(f"  Raw nuc DE:  {len(de_raw_nuc)} cell types")

# =====================================================================
# Per-model evaluation
# =====================================================================
for model_name, model_info in MODELS.items():
    print(f"\n{'=' * 70}")
    print(f"MODEL: {model_name}")
    print(f"{'=' * 70}")

    model = load_model(model_info, device)

    # Reconstruct both modalities
    print("  Reconstructing cells...")
    recon_cell = reconstruct(model, X_cell, CELL_ID, model_info['has_cond'], device)
    print("  Reconstructing nuclei...")
    recon_nuc = reconstruct(model, X_nuc, NUCLEI_ID, model_info['has_cond'], device)

    # ----- Per-cell correlation -----
    print("\n  [Per-cell correlation]")

    # Cell recon vs raw cell
    cell_corrs = []
    for i in range(X_cell.shape[0]):
        r, _ = stats.pearsonr(recon_cell[i], X_cell[i])
        cell_corrs.append(r)
    cell_corrs = np.array(cell_corrs)

    # Nuc recon vs raw nuc
    nuc_corrs = []
    for i in range(X_nuc.shape[0]):
        r, _ = stats.pearsonr(recon_nuc[i], X_nuc[i])
        nuc_corrs.append(r)
    nuc_corrs = np.array(nuc_corrs)

    print(f"  Recon cell vs raw cell (median):  {np.median(cell_corrs):.4f}")
    print(f"  Recon nuc vs raw nuc (median):    {np.median(nuc_corrs):.4f}")

    # Per-cell-type breakdown
    print(f"\n  Per-cell-type recon correlation:")
    print(f"  {'Cell Type':<20} {'Cell recon':>12} {'Nuc recon':>12}")
    for ct in common_cts:
        cell_mask = ct_cell == ct
        nuc_mask = ct_nuc == ct
        mc = np.median(cell_corrs[cell_mask]) if cell_mask.any() else 0
        mn = np.median(nuc_corrs[nuc_mask]) if nuc_mask.any() else 0
        print(f"  {ct:<20} {mc:>12.4f} {mn:>12.4f}")

    # ----- Classification -----
    print("\n  [Classification — train on raw cell]")
    scaler = StandardScaler()
    X_cell_scaled = scaler.fit_transform(X_cell)

    cell_ct_mask = np.isin(ct_cell, common_cts)
    nuc_ct_mask = np.isin(ct_nuc, common_cts)

    clf = LogisticRegression(
        multi_class='multinomial', solver='lbfgs',
        max_iter=2000, random_state=42, n_jobs=-1, C=1.0,
    )
    clf.fit(X_cell_scaled[cell_ct_mask], ct_cell[cell_ct_mask])

    # Predict on raw cell (sanity)
    cell_pred = clf.predict(X_cell_scaled[cell_ct_mask])
    cell_acc = accuracy_score(ct_cell[cell_ct_mask], cell_pred)

    # Predict on recon cell
    recon_cell_scaled = scaler.transform(recon_cell)
    rc_pred = clf.predict(recon_cell_scaled[cell_ct_mask])
    rc_acc = accuracy_score(ct_cell[cell_ct_mask], rc_pred)
    rc_f1 = f1_score(ct_cell[cell_ct_mask], rc_pred, average='macro')

    # Predict on raw nuc
    X_nuc_scaled = scaler.transform(X_nuc)
    rn_pred = clf.predict(X_nuc_scaled[nuc_ct_mask])
    rn_acc = accuracy_score(ct_nuc[nuc_ct_mask], rn_pred)

    # Predict on recon nuc
    recon_nuc_scaled = scaler.transform(recon_nuc)
    rnr_pred = clf.predict(recon_nuc_scaled[nuc_ct_mask])
    rnr_acc = accuracy_score(ct_nuc[nuc_ct_mask], rnr_pred)
    rnr_f1 = f1_score(ct_nuc[nuc_ct_mask], rnr_pred, average='macro')

    print(f"  Train on raw cell, predict on:")
    print(f"    Raw cell (sanity):        accuracy={cell_acc:.4f}")
    print(f"    Recon cell:               accuracy={rc_acc:.4f}, F1={rc_f1:.4f}")
    print(f"    Raw nuclei:               accuracy={rn_acc:.4f}")
    print(f"    Recon nuclei:             accuracy={rnr_acc:.4f}, F1={rnr_f1:.4f}")
    print(f"    Recon cell delta:         {(rc_acc - cell_acc)*100:+.1f}%")
    print(f"    Recon nuc delta vs raw:   {(rnr_acc - rn_acc)*100:+.1f}%")

    # ----- DE preservation -----
    print("\n  [DE preservation — recon vs raw]")

    de_recon_cell = run_de_one_vs_rest(recon_cell, ct_cell, top_k=100)
    de_recon_nuc = run_de_one_vs_rest(recon_nuc, ct_nuc, top_k=100)

    # Overlap: recon cell DE vs raw cell DE
    cell_de_overlaps = {}
    for ct in common_cts:
        if ct in de_raw_cell and ct in de_recon_cell:
            overlap = len(set(de_raw_cell[ct]) & set(de_recon_cell[ct]))
            cell_de_overlaps[ct] = overlap / 100

    # Overlap: recon nuc DE vs raw nuc DE
    nuc_de_overlaps = {}
    for ct in common_cts:
        if ct in de_raw_nuc and ct in de_recon_nuc:
            overlap = len(set(de_raw_nuc[ct]) & set(de_recon_nuc[ct]))
            nuc_de_overlaps[ct] = overlap / 100

    mean_cell_de = np.mean(list(cell_de_overlaps.values())) if cell_de_overlaps else 0
    mean_nuc_de = np.mean(list(nuc_de_overlaps.values())) if nuc_de_overlaps else 0

    print(f"  Top-100 DE overlap:")
    print(f"    Recon cell vs raw cell DE: {mean_cell_de:.3f}")
    print(f"    Recon nuc vs raw nuc DE:   {mean_nuc_de:.3f}")

    print(f"\n  Per-cell-type DE overlap:")
    print(f"  {'Cell Type':<20} {'Cell recon':>12} {'Nuc recon':>12}")
    de_rows = []
    for ct in common_cts:
        co = cell_de_overlaps.get(ct, 0)
        no = nuc_de_overlaps.get(ct, 0)
        print(f"  {ct:<20} {co:>12.2f} {no:>12.2f}")
        de_rows.append({'cell_type': ct, 'cell_recon_de': co, 'nuc_recon_de': no})
    pd.DataFrame(de_rows).to_csv(f'{OUTPUT_DIR}/de_recon_{model_name}.csv', index=False)

    # ----- Coexpression -----
    print("\n  [Coexpression — recon vs raw]")
    K_GENES = min(500, n_genes)
    gene_var = X_cell.var(axis=0)
    top_var_idx = np.argsort(gene_var)[-K_GENES:]

    raw_cell_coexp = np.corrcoef(X_cell[:, top_var_idx].T)
    recon_cell_coexp = np.corrcoef(recon_cell[:, top_var_idx].T)
    raw_nuc_coexp = np.corrcoef(X_nuc[:, top_var_idx].T)
    recon_nuc_coexp = np.corrcoef(recon_nuc[:, top_var_idx].T)

    triu = np.triu_indices(K_GENES, k=1)

    rc_coexp_corr = stats.spearmanr(raw_cell_coexp[triu], recon_cell_coexp[triu])[0]
    rn_coexp_corr = stats.spearmanr(raw_nuc_coexp[triu], recon_nuc_coexp[triu])[0]

    print(f"  Recon cell coexp vs raw cell coexp: {rc_coexp_corr:.4f}")
    print(f"  Recon nuc coexp vs raw nuc coexp:   {rn_coexp_corr:.4f}")

    # ----- Per-cell variance (recon vs raw) -----
    print("\n  [Variance preservation]")
    print(f"  {'Cell Type':<20} {'Raw cell var':>14} {'Recon cell var':>14} {'Ratio':>8}  {'Raw nuc var':>14} {'Recon nuc var':>14} {'Ratio':>8}")
    for ct in common_cts:
        cell_mask = ct_cell == ct
        nuc_mask = ct_nuc == ct
        raw_cv = X_cell[cell_mask].var(axis=0).mean()
        rec_cv = recon_cell[cell_mask].var(axis=0).mean()
        raw_nv = X_nuc[nuc_mask].var(axis=0).mean()
        rec_nv = recon_nuc[nuc_mask].var(axis=0).mean()
        print(f"  {ct:<20} {raw_cv:>14.4f} {rec_cv:>14.4f} {rec_cv/(raw_cv+1e-12):>8.4f}  {raw_nv:>14.4f} {rec_nv:>14.4f} {rec_nv/(raw_nv+1e-12):>8.4f}")

    del model
    torch.cuda.empty_cache()
    gc.collect()

# =====================================================================
# Summary
# =====================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("All results saved to:", OUTPUT_DIR)
print("Done!")
