"""
mu / log_var range and distribution diagnostic.

Loads each specified run, runs a forward pass on a subsample of data,
and reports the range/distribution of mu and log_var — useful for diagnosing
whether CPM normalization is causing latent explosion.

Usage:
    python analysis/mu_logvar_range.py \
        --runs "run/run_XXX:log1p_s1" "run/run_YYY:cpm_s1" ... \
        --output_dir notes/experiments/SUC-6_variance_retention/mu_logvar_range
"""

import argparse
import gc
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import sparse

warnings.filterwarnings('ignore')

sys.path.insert(0, '/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE')

BASE_DATA_PATH = '/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi'
CELL_DATA_DIR  = '10X_cells_v2_v3_AIBS'
NUC_DATA_DIR   = '10X_nuclei_v2_v3_AIBS'
BATCH_CELL     = 'c3A'
BATCH_NUC      = 'n3A'
MAX_CELLS      = 8000


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data_for_run(run_args, max_cells=MAX_CELLS):
    """Load and normalise data matching the run's training configuration."""
    import anndata as ad
    import h5py
    import pandas as pd

    data_path = run_args['data_path']
    norm      = run_args.get('raw_data_normalization', 'log1p')
    gene_list_path = run_args.get('gene_list', None)

    if 'nuclei' in data_path.lower():
        data_dir = NUC_DATA_DIR
        batch    = BATCH_NUC
    else:
        data_dir = CELL_DATA_DIR
        batch    = BATCH_CELL

    h5ad_path = os.path.join(BASE_DATA_PATH, data_dir, 'data.h5ad')
    print(f"  Loading data from {h5ad_path}  (batch={batch}, norm={norm})")

    # Read metadata only
    adata_obs = ad.read_h5ad(h5ad_path, backed='r')
    obs = adata_obs.obs.copy()
    var = adata_obs.var.copy()
    if hasattr(adata_obs, 'file') and adata_obs.file is not None:
        adata_obs.file.close()
    del adata_obs
    gc.collect()

    gene_names = var['gene_names'].values if 'gene_names' in var.columns else var.index.values

    # Batch filter
    if 'meta_batch_id' not in obs.columns:
        obs['meta_batch_id'] = obs['batch_id'].apply(lambda x: x.split('.')[0])
    batch_mask = obs['meta_batch_id'].str.endswith(batch).values
    batch_idx  = np.where(batch_mask)[0]
    print(f"  {len(batch_idx)} cells in batch {batch}")

    # Subsample
    rng = np.random.RandomState(42)
    if len(batch_idx) > max_cells:
        batch_idx = rng.choice(batch_idx, max_cells, replace=False)
        batch_idx = np.sort(batch_idx)

    # Gene filter
    if gene_list_path and os.path.exists(gene_list_path):
        gene_list = pd.read_csv(gene_list_path, header=None)[0].tolist()
        gene_set  = set(gene_list)
    else:
        gene_set = set(gene_names)

    gene_idx = np.array(
        [i for i, g in enumerate(gene_names) if g in gene_set], dtype=np.int64
    )
    print(f"  {len(gene_idx)} genes matched")

    # Read raw counts via h5py
    import h5py
    with h5py.File(h5ad_path, 'r') as f:
        x_data = f['X']
        if isinstance(x_data, h5py.Group):
            data_    = x_data['data'][:]
            indices_ = x_data['indices'][:]
            indptr_  = x_data['indptr'][:]
            n_rows   = len(indptr_) - 1
            X_sparse = sparse.csr_matrix(
                (data_, indices_, indptr_), shape=(n_rows, len(gene_names))
            )
            X = X_sparse[batch_idx][:, gene_idx].toarray().astype(np.float32)
            del X_sparse, data_, indices_, indptr_
        else:
            rows = [x_data[idx, :][gene_idx] for idx in batch_idx]
            X = np.array(rows, dtype=np.float32)

    gc.collect()

    # Normalise
    if norm == 'log1p':
        X_norm = np.log1p(X)
    elif norm == 'cpm':
        lib_size = X.sum(axis=1, keepdims=True).clip(1.0)
        X_norm   = np.log1p(X / lib_size * 1e6).astype(np.float32)
    elif norm == 'raw':
        X_norm = X
    else:
        raise ValueError(f"Unknown normalization: {norm}")

    print(f"  X_norm shape={X_norm.shape}  min={X_norm.min():.3f}  "
          f"max={X_norm.max():.3f}  mean={X_norm.mean():.3f}")
    del X
    gc.collect()
    return X_norm


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_from_run(run_folder, n_genes):
    from imputeVAE.models import VAE_JSD
    from imputeVAE.models_cond import VAE_JSD_Cond

    with open(os.path.join(run_folder, 'arguments.json')) as f:
        run_args = json.load(f)

    num_internal = run_args.get('num_internal', [1024, 512])
    if isinstance(num_internal, str):
        num_internal = [int(x) for x in num_internal.split(',')]
    num_latent = run_args.get('num_latent', 32)
    embed_dim  = run_args.get('embed_dim', 8)
    is_cond    = bool(run_args.get('fine_tune_cond_stage3', False))

    if is_cond:
        model = VAE_JSD_Cond(
            input_size=n_genes,
            num_internal=num_internal,
            num_latent=num_latent,
            embed_dim=embed_dim,
        )
        print(f"  Model: VAE_JSD_Cond  latent={num_latent}  embed={embed_dim}  internal={num_internal}")
    else:
        model = VAE_JSD(
            input_size=n_genes,
            num_internal=num_internal,
            num_latent=num_latent,
        )
        print(f"  Model: VAE_JSD  latent={num_latent}  internal={num_internal}")

    model_path = os.path.join(run_folder, 'models', 'model.pt')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def extract_mu_logvar(model, X_norm, batch_size=512):
    mus, lvs = [], []
    with torch.no_grad():
        for i in range(0, len(X_norm), batch_size):
            x       = torch.tensor(X_norm[i:i + batch_size], dtype=torch.float32)
            encoded = model.encoder(x)
            mu      = model.mu(encoded)
            lv      = torch.clamp(model.log_var(encoded), min=-20, max=20)
            mus.append(mu.numpy())
            lvs.append(lv.numpy())
    return np.concatenate(mus), np.concatenate(lvs)


# ---------------------------------------------------------------------------
# Statistics + reporting
# ---------------------------------------------------------------------------

PCTS = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]


def print_run_stats(mu, log_var, label):
    sep = '=' * 72
    print(f"\n{sep}")
    print(f"  {label}")
    print(sep)

    flat_mu = mu.flatten()
    flat_lv = log_var.flatten()

    print(f"\n  {'Statistic':<14} {'mu':>14} {'log_var':>14}")
    print(f"  {'-'*14} {'-'*14} {'-'*14}")
    for name, mu_v, lv_v in [
        ('min',      flat_mu.min(),            flat_lv.min()),
        ('p0.1',     np.percentile(flat_mu, 0.1),  np.percentile(flat_lv, 0.1)),
        ('p1',       np.percentile(flat_mu, 1),    np.percentile(flat_lv, 1)),
        ('p5',       np.percentile(flat_mu, 5),    np.percentile(flat_lv, 5)),
        ('p25',      np.percentile(flat_mu, 25),   np.percentile(flat_lv, 25)),
        ('p50',      np.percentile(flat_mu, 50),   np.percentile(flat_lv, 50)),
        ('p75',      np.percentile(flat_mu, 75),   np.percentile(flat_lv, 75)),
        ('p95',      np.percentile(flat_mu, 95),   np.percentile(flat_lv, 95)),
        ('p99',      np.percentile(flat_mu, 99),   np.percentile(flat_lv, 99)),
        ('p99.9',    np.percentile(flat_mu, 99.9), np.percentile(flat_lv, 99.9)),
        ('max',      flat_mu.max(),            flat_lv.max()),
        ('mean',     flat_mu.mean(),           flat_lv.mean()),
        ('std',      flat_mu.std(),            flat_lv.std()),
        ('|mean|',   np.abs(flat_mu).mean(),   np.abs(flat_lv).mean()),
        ('|max|',    np.abs(flat_mu).max(),    np.abs(flat_lv).max()),
    ]:
        print(f"  {name:<14} {mu_v:>14.4f} {lv_v:>14.4f}")

    # Clamp saturation
    pct_lo = 100.0 * (flat_lv <= -19.9).mean()
    pct_hi = 100.0 * (flat_lv >=  19.9).mean()
    print(f"\n  log_var at clamp -20: {pct_lo:.3f}%   at clamp +20: {pct_hi:.3f}%")

    # Per-dim summary
    n_dims       = mu.shape[1]
    abs_mu_dim   = np.abs(mu).mean(axis=0)
    mean_lv_dim  = log_var.mean(axis=0)
    post_std_dim = np.exp(0.5 * mean_lv_dim)

    print(f"\n  Per-dim summary ({n_dims} dims):")
    print(f"    mean |mu|  per dim — min={abs_mu_dim.min():.3f}  max={abs_mu_dim.max():.3f}  "
          f"mean={abs_mu_dim.mean():.3f}")
    print(f"    mean logvar per dim — min={mean_lv_dim.min():.3f}  max={mean_lv_dim.max():.3f}  "
          f"mean={mean_lv_dim.mean():.3f}")
    print(f"    posterior std per dim — min={post_std_dim.min():.4f}  "
          f"max={post_std_dim.max():.4f}  mean={post_std_dim.mean():.4f}")
    print(f"    Dims with mean |mu| > 10:   {(abs_mu_dim > 10).sum()}/{n_dims}")
    print(f"    Dims with mean |mu| > 100:  {(abs_mu_dim > 100).sum()}/{n_dims}")
    print(f"    Dims with post_std < 0.01:  {(post_std_dim < 0.01).sum()}/{n_dims}")
    print(f"    Dims with post_std < 0.1:   {(post_std_dim < 0.1).sum()}/{n_dims}")

    # Top 5 dims by |mu|
    top5 = np.argsort(-abs_mu_dim)[:5]
    print(f"\n  Top 5 dims by mean |mu|:")
    print(f"    {'Dim':>5} {'mean|mu|':>10} {'meanLogVar':>12} {'postStd':>9}")
    for d in top5:
        print(f"    {d:>5} {abs_mu_dim[d]:>10.3f} {mean_lv_dim[d]:>12.3f} "
              f"{post_std_dim[d]:>9.4f}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_run(res, output_dir):
    """One figure per run: mu histogram + log_var histogram side by side."""
    label   = res['label']
    flat_mu = res['mu'].flatten()
    flat_lv = res['log_var'].flatten()

    pct_lo = 100.0 * (flat_lv <= -19.9).mean()
    pct_hi = 100.0 * (flat_lv >=  19.9).mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(label, fontsize=13, fontweight='bold')

    # mu histogram
    ax = axes[0]
    ax.hist(flat_mu, bins=300, density=True, color='steelblue', alpha=0.85)
    ax.axvline(0, color='black', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xlabel('mu')
    ax.set_ylabel('density')
    ax.set_title(f'mu  [min={flat_mu.min():.1f}, max={flat_mu.max():.1f}, '
                 f'mean|mu|={np.abs(flat_mu).mean():.2f}]')

    # log_var histogram
    ax = axes[1]
    ax.hist(flat_lv, bins=200, density=True, color='seagreen', alpha=0.85)
    ax.axvline(-20, color='red', linestyle='--', linewidth=1.2, alpha=0.8, label='clamp ±20')
    ax.axvline( 20, color='red', linestyle='--', linewidth=1.2, alpha=0.8)
    ax.axvline(  0, color='black', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xlabel('log_var')
    ax.set_ylabel('density')
    ax.set_title(f'log_var  [at -20: {pct_lo:.1f}%,  at +20: {pct_hi:.1f}%]')
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, f'{label}_dist.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--runs', nargs='+', required=True,
        help='"run_folder:label" pairs, e.g. run/run_XXX:log1p_s1',
    )
    parser.add_argument(
        '--output_dir', type=str,
        default='notes/experiments/SUC-6_variance_retention/mu_logvar_range',
    )
    parser.add_argument('--max_cells', type=int, default=MAX_CELLS)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []

    for run_spec in args.runs:
        run_folder, label = (run_spec.rsplit(':', 1) if ':' in run_spec
                             else (run_spec, os.path.basename(run_spec)))

        print(f"\n{'#'*72}")
        print(f"# {label}  —  {run_folder}")
        print(f"{'#'*72}")

        with open(os.path.join(run_folder, 'arguments.json')) as f:
            run_args = json.load(f)

        X_norm = load_data_for_run(run_args, max_cells=args.max_cells)
        model  = load_model_from_run(run_folder, X_norm.shape[1])

        print("  Extracting mu and log_var...")
        mu, log_var = extract_mu_logvar(model, X_norm)
        print(f"  mu: {mu.shape}   log_var: {log_var.shape}")

        print_run_stats(mu, log_var, label)

        # Save arrays
        np.save(os.path.join(args.output_dir, f'{label}_mu.npy'),     mu)
        np.save(os.path.join(args.output_dir, f'{label}_logvar.npy'), log_var)

        all_results.append({'label': label, 'mu': mu, 'log_var': log_var})

        del X_norm, model
        gc.collect()

    # Cross-run summary table
    print(f"\n{'='*80}")
    print("  CROSS-RUN SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Label':<22} {'mu_min':>9} {'mu_max':>9} {'|mu|mean':>9} "
          f"{'lv_min':>8} {'lv_max':>8} {'%@-20':>7} {'%@+20':>7}")
    print(f"  {'-'*22} {'-'*9} {'-'*9} {'-'*9} {'-'*8} {'-'*8} {'-'*7} {'-'*7}")
    for res in all_results:
        fm = res['mu'].flatten()
        fl = res['log_var'].flatten()
        print(f"  {res['label']:<22} {fm.min():>9.2f} {fm.max():>9.2f} "
              f"{np.abs(fm).mean():>9.3f} "
              f"{fl.min():>8.2f} {fl.max():>8.2f} "
              f"{100*(fl<=-19.9).mean():>7.2f} {100*(fl>=19.9).mean():>7.2f}")

    print("\nGenerating plots...")
    for res in all_results:
        plot_run(res, args.output_dir)

    print(f"\nAll outputs written to: {args.output_dir}")
    print("Done.")


if __name__ == '__main__':
    main()
