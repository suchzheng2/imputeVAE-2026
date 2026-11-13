"""Per-cell-type DE overlap: imputed vs raw nuclei vs raw cells.

Loads model, cross-decodes nuclei, computes DE overlap per cell type.
Requires GPU (or CPU with patience).
"""

import os
import sys
import json
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sparse
from scipy.stats import ranksums
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from imputeVAE.models import VAE_JSD
from imputeVAE.models_cond import VAE_JSD_Cond

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "notes", "qualify-exam", "pre-qual", "figures")
DATA_ROOT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi"

CELL_ID = 0
NUCLEI_ID = 1
TOP_K = 100  # top DE genes per cell type


def load_data(data_dir, batch_suffix, gene_list_path):
    path = os.path.join(DATA_ROOT, data_dir, "data.h5ad")
    print(f"Loading {path}...")
    adata = ad.read_h5ad(path)

    if 'meta_batch_id' not in adata.obs:
        adata.obs['meta_batch_id'] = adata.obs['batch_id'].apply(lambda x: x.split('.')[0])
    adata = adata[adata.obs['meta_batch_id'].str.endswith(batch_suffix)].copy()

    common_genes = pd.read_csv(os.path.join(PROJECT_ROOT, 'data/common_genes.csv'),
                               header=0)['common_genes'].tolist()
    adata = adata[:, adata.var['gene_names'].isin(common_genes)].copy()

    common_ct = pd.read_csv(os.path.join(PROJECT_ROOT, 'data/common_cell_types.csv'),
                            header=0)['common_cell_types'].tolist()
    adata = adata[adata.obs['cell_type'].isin(common_ct)].copy()

    gene_list = pd.read_csv(gene_list_path, header=None)[0].tolist()
    adata = adata[:, adata.var['gene_names'].isin(gene_list)].copy()
    adata = adata[:, adata.var['gene_names'].argsort()].copy()

    X = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
    adata.X = np.log1p(X).astype(np.float32)

    print(f"  {adata.shape[0]} cells, {adata.shape[1]} genes")
    return adata


def load_model(run_folder, device):
    args_path = os.path.join(run_folder, 'arguments.json')
    model_path = os.path.join(run_folder, 'models', 'model.pt')

    with open(args_path) as f:
        args = json.load(f)

    embed_dim = args.get('embed_dim', 8)
    model = VAE_JSD_Cond(
        input_size=args['num_feature'],
        num_internal=args['num_internal'],
        num_latent=args['num_latent'],
        embed_dim=embed_dim,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def cross_decode(model, X_nuc, device, batch_size=256):
    X = torch.tensor(X_nuc, dtype=torch.float32)
    results = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            batch = X[i:i+batch_size].to(device)
            xd = model.decode_cross(batch, source_modality_id=NUCLEI_ID,
                                    target_modality_id=CELL_ID)
            results.append(xd.cpu().numpy())
    return np.concatenate(results, axis=0)


def run_de(X, cts, top_k=TOP_K):
    """Rank-sum DE: each CT vs rest. Returns {ct: set of top gene indices}."""
    de_genes = {}
    unique_cts = sorted(set(cts))
    for ct in unique_cts:
        mask = cts == ct
        if mask.sum() < 5 or (~mask).sum() < 5:
            continue
        pvals = np.ones(X.shape[1])
        fcs = np.zeros(X.shape[1])
        for g in range(X.shape[1]):
            _, p = ranksums(X[mask, g], X[~mask, g])
            pvals[g] = p
            fcs[g] = X[mask, g].mean() - X[~mask, g].mean()
        order = np.lexsort((-np.abs(fcs), pvals))
        de_genes[ct] = set(order[:top_k])
    return de_genes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='pct5v2_cond_xr',
                        choices=['pct5v2_cond_xr', 'MSE_cond_xr'])
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    models_info = {
        'pct5v2_cond_xr': {
            'run_folder': 'run/run_2026-04-07-02-12-05_3551465',
            'gene_list': 'data/gene_list_perct_5pct.csv',
            'label': 'PerCT5 cond+xr',
        },
        'MSE_cond_xr': {
            'run_folder': 'run/run_2026-04-02-10-51-32_3738876',
            'gene_list': 'data/training_gene_list_v3.csv',
            'label': 'ConfigB cond+xr',
        },
    }

    info = models_info[args.model]
    gene_list_path = os.path.join(PROJECT_ROOT, info['gene_list'])

    # Load data
    adata_cell = load_data("10X_cells_v2_v3_AIBS", "c3A", gene_list_path)
    adata_nuc = load_data("10X_nuclei_v2_v3_AIBS", "n3A", gene_list_path)

    X_cell = adata_cell.X
    ct_cell = adata_cell.obs['cell_type'].values
    X_nuc = adata_nuc.X
    ct_nuc = adata_nuc.obs['cell_type'].values

    common_cts = sorted(set(ct_cell) & set(ct_nuc))
    print(f"\nCommon cell types: {common_cts}")

    # Load model and cross-decode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    model = load_model(os.path.join(PROJECT_ROOT, info['run_folder']), device)
    X_imputed = cross_decode(model, X_nuc, device)
    print(f"Cross-decoded: {X_imputed.shape}")

    # Run DE for all three sources
    print("\nRunning DE (raw cell)...")
    de_cell = run_de(X_cell, ct_cell)
    print("Running DE (raw nuclei)...")
    de_nuc = run_de(X_nuc, ct_nuc)
    print("Running DE (imputed)...")
    de_imp = run_de(X_imputed, ct_nuc)

    # Compute per-CT overlap (vs raw cell as ground truth)
    rows = []
    for ct in common_cts:
        if ct not in de_cell:
            continue
        nuc_overlap = len(de_cell[ct] & de_nuc.get(ct, set())) / TOP_K if ct in de_nuc else 0
        imp_overlap = len(de_cell[ct] & de_imp.get(ct, set())) / TOP_K if ct in de_imp else 0
        rows.append({
            'cell_type': ct,
            'raw_nuclei': nuc_overlap,
            'imputed': imp_overlap,
            'improvement': imp_overlap - nuc_overlap,
        })
        print(f"  {ct:15s}  nuc={nuc_overlap:.3f}  imp={imp_overlap:.3f}  Δ={imp_overlap-nuc_overlap:+.3f}")

    df = pd.DataFrame(rows)
    df = df.sort_values('imputed', ascending=True)

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, f"de_per_ct_{args.model}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # --- Plot: grouped horizontal bar chart ---
    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.5)))

    y = np.arange(len(df))
    height = 0.35

    bars_nuc = ax.barh(y + height/2, df['raw_nuclei'].values * 100,
                       height, color='#999999', label='Raw nuclei')
    bars_imp = ax.barh(y - height/2, df['imputed'].values * 100,
                       height, color='#e8912d', label=f'Imputed ({info["label"]})')

    ax.set_yticks(y)
    ax.set_yticklabels(df['cell_type'].values, fontsize=10)
    ax.set_xlabel(f'DE overlap with real cells (%, top {TOP_K} genes)', fontsize=12)
    ax.set_title('Per-cell-type DE gene recovery', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.set_xlim(0, 100)

    # Add improvement annotations
    for i, (_, row) in enumerate(df.iterrows()):
        imp_pct = row['imputed'] * 100
        delta = row['improvement'] * 100
        if delta > 0:
            ax.text(imp_pct + 1, i - height/2, f"+{delta:.0f}%",
                    va='center', fontsize=8, color='#2ca02c', fontweight='bold')
        elif delta < 0:
            ax.text(imp_pct + 1, i - height/2, f"{delta:.0f}%",
                    va='center', fontsize=8, color='#dc3545', fontweight='bold')

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"slide_de_per_ct_{args.model}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
