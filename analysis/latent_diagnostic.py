"""
Latent Space Diagnostic

Analyzes the 32-dim latent to answer:
1. Per-dimension total variance — are any dims collapsed/dead?
2. Per-dimension between-CT vs within-CT variance — what does each dim encode?
3. Per-dimension effective usage — does the encoder saturate all dims?
4. Cumulative within-CT variance — how many dims carry per-cell info?
5. Log-var analysis — is the encoder posteriors collapsed (log_var → -inf)?

Usage:
    python analysis/latent_diagnostic.py --run_folder run/run_XXXX --gene_list data/gene_list_perct_5pct.csv
"""

import argparse
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import sparse

warnings.filterwarnings('ignore')

sys.path.insert(0, '/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE')
from imputeVAE.models_cond import VAE_JSD_Cond

# ---- Config ----
CELL_DATA = '10X_cells_v2_v3_AIBS'
NUC_DATA = '10X_nuclei_v2_v3_AIBS'
BASE_PATH = '/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi'
BATCH_CELL = 'c3A'
BATCH_NUC = 'n3A'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_folder', type=str, required=True)
    parser.add_argument('--gene_list', type=str, default='data/gene_list_perct_5pct.csv')
    parser.add_argument('--output_dir', type=str, default=None)
    return parser.parse_args()


def load_data(gene_list_path, batch_cell=BATCH_CELL, batch_nuc=BATCH_NUC,
              max_cells=10000):
    """Load cell and nuclei data filtered to gene list, subsampled."""
    import anndata as ad
    import h5py
    import gc

    genes = pd.read_csv(gene_list_path, header=None)[0].tolist()

    datasets = {}
    for name, data_dir, batch in [('cell', CELL_DATA, batch_cell),
                                   ('nuclei', NUC_DATA, batch_nuc)]:
        path = os.path.join(BASE_PATH, data_dir, 'data.h5ad')

        # Step 1: read obs + var only to get batch + cell type + gene info
        adata_obs = ad.read_h5ad(path, backed='r')
        obs = adata_obs.obs.copy()
        var = adata_obs.var.copy()
        if hasattr(adata_obs, 'file') and adata_obs.file is not None:
            adata_obs.file.close()
        del adata_obs
        gc.collect()

        # Get gene names — could be in var_names or var['gene_names']
        if 'gene_names' in var.columns:
            gene_names = var['gene_names'].values
        else:
            gene_names = var.index.values

        # Filter to batch (endswith, matching training code in data.py)
        if 'meta_batch_id' not in obs.columns:
            obs['meta_batch_id'] = obs['batch_id'].apply(lambda x: x.split('.')[0])
        batch_mask = obs['meta_batch_id'].str.endswith(batch).values
        batch_idx = np.where(batch_mask)[0]
        print(f"  {name}: {len(batch_idx)} cells in batch {batch}")

        # Subsample
        rng = np.random.RandomState(42)
        if len(batch_idx) > max_cells:
            batch_idx = rng.choice(batch_idx, max_cells, replace=False)
            batch_idx.sort()

        # Gene indices (int64 for h5py indexing)
        gene_set = set(genes)
        gene_idx = np.array([i for i, g in enumerate(gene_names) if g in gene_set],
                            dtype=np.int64)
        print(f"  {name}: {len(gene_idx)} genes matched from gene list")

        # Step 2: read only needed rows/cols from h5ad via h5py
        cell_types = obs['cell_type'].values[batch_idx]

        with h5py.File(path, 'r') as f:
            X_key = 'X'
            if 'raw' in f and 'X' in f['raw']:
                X_key = 'raw/X'

            x_data = f[X_key]

            # Check if sparse (CSR/CSC)
            if isinstance(x_data, h5py.Group):
                # Sparse matrix — read full sparse and slice
                data = x_data['data'][:]
                indices = x_data['indices'][:]
                indptr = x_data['indptr'][:]
                shape = x_data.attrs.get('shape', None)
                if shape is None:
                    shape = (len(indptr) - 1, var_names.shape[0])
                X_sparse = sparse.csr_matrix((data, indices, indptr), shape=shape)
                X = X_sparse[batch_idx][:, gene_idx].toarray().astype(np.float32)
                del X_sparse, data, indices, indptr
            else:
                # Dense — read row by row
                rows = []
                for idx in batch_idx:
                    row = x_data[idx, :][gene_idx]
                    rows.append(row)
                X = np.array(rows, dtype=np.float32)

        gc.collect()

        X_log = np.log1p(X)

        datasets[name] = {
            'X_log': X_log,
            'cell_types': cell_types,
            'n_genes': len(gene_idx),
        }
        print(f"  Loaded {name}: {X_log.shape}")
        del X
        gc.collect()

    return datasets


def load_model(run_folder, n_genes):
    """Load trained model."""
    args_path = os.path.join(run_folder, 'arguments.json')
    with open(args_path) as f:
        run_args = json.load(f)

    num_internal = run_args.get('num_internal', [1024, 512])
    if isinstance(num_internal, str):
        num_internal = [int(x) for x in num_internal.split(',')]
    num_latent = run_args.get('num_latent', 32)
    embed_dim = run_args.get('embed_dim', 8)

    model = VAE_JSD_Cond(
        input_size=n_genes,
        num_internal=num_internal,
        num_latent=num_latent,
        embed_dim=embed_dim,
    )

    model_path = os.path.join(run_folder, 'models', 'model.pt')
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    print(f"  Loaded model: latent_size={num_latent}, internal={num_internal}, embed_dim={embed_dim}")
    return model, num_latent


def extract_latents(model, X_log, batch_size=512):
    """Extract mu and log_var from encoder."""
    model.eval()
    mus = []
    log_vars = []

    with torch.no_grad():
        for i in range(0, len(X_log), batch_size):
            batch = torch.tensor(X_log[i:i+batch_size], dtype=torch.float32)
            encoded = model.encoder(batch)
            mu = model.mu(encoded)
            log_var = model.log_var(encoded)
            log_var = torch.clamp(log_var, min=-20, max=20)
            mus.append(mu.numpy())
            log_vars.append(log_var.numpy())

    return np.concatenate(mus), np.concatenate(log_vars)


def analyze_latent(mu, log_var, cell_types, label, output_dir):
    """Full per-dimension analysis."""
    n_cells, n_dims = mu.shape
    unique_cts = np.unique(cell_types)
    n_cts = len(unique_cts)

    print(f"\n{'='*70}")
    print(f"  LATENT DIAGNOSTIC: {label}")
    print(f"  {n_cells} cells, {n_dims} dims, {n_cts} cell types")
    print(f"{'='*70}")

    # ---- 1. Per-dimension total variance ----
    total_var = np.var(mu, axis=0)  # [n_dims]

    # ---- 2. Between-CT vs within-CT variance decomposition ----
    grand_mean = mu.mean(axis=0)
    ss_total = np.sum((mu - grand_mean) ** 2, axis=0)

    ss_between = np.zeros(n_dims)
    ss_within = np.zeros(n_dims)
    ct_means = {}
    ct_counts = {}

    for ct in unique_cts:
        mask = cell_types == ct
        ct_mu = mu[mask]
        n_ct = ct_mu.shape[0]
        ct_mean = ct_mu.mean(axis=0)
        ct_means[ct] = ct_mean
        ct_counts[ct] = n_ct

        ss_between += n_ct * (ct_mean - grand_mean) ** 2
        ss_within += np.sum((ct_mu - ct_mean) ** 2, axis=0)

    frac_between = ss_between / (ss_total + 1e-10)
    frac_within = 1 - frac_between

    # ---- 3. Log-var analysis (posterior collapse check) ----
    mean_log_var = log_var.mean(axis=0)  # per-dim average log_var
    std_from_logvar = np.exp(0.5 * mean_log_var)  # corresponding std

    # ---- 4. Sort dims by total variance ----
    sort_idx = np.argsort(-total_var)  # descending

    # ---- Print per-dimension table ----
    print(f"\n  Per-dimension breakdown (sorted by total variance, descending):")
    print(f"  {'Dim':>4} {'TotalVar':>10} {'FracBetween':>12} {'FracWithin':>11} "
          f"{'WithinVar':>10} {'MeanLogVar':>11} {'PostStd':>8}")
    print(f"  {'-'*4} {'-'*10} {'-'*12} {'-'*11} {'-'*10} {'-'*11} {'-'*8}")

    within_var = ss_within / (n_cells - n_cts)  # unbiased within-CT variance per dim

    for rank, dim in enumerate(sort_idx):
        print(f"  {dim:>4} {total_var[dim]:>10.4f} {frac_between[dim]:>12.4f} "
              f"{frac_within[dim]:>11.4f} {within_var[dim]:>10.4f} "
              f"{mean_log_var[dim]:>11.4f} {std_from_logvar[dim]:>8.4f}")

    # ---- Summary stats ----
    n_active = np.sum(total_var > 0.01)  # dims with non-trivial variance
    n_dead = n_dims - n_active
    total_within_var = ss_within.sum()
    total_total_var = ss_total.sum()
    overall_frac_within = total_within_var / (total_total_var + 1e-10)

    print(f"\n  Summary:")
    print(f"    Active dims (var > 0.01):     {n_active}/{n_dims}")
    print(f"    Dead dims (var <= 0.01):      {n_dead}/{n_dims}")
    print(f"    Overall frac between-CT:      {1 - overall_frac_within:.4f}")
    print(f"    Overall frac within-CT:       {overall_frac_within:.4f}")
    print(f"    Mean posterior std:            {std_from_logvar.mean():.4f}")
    print(f"    Median posterior std:          {np.median(std_from_logvar):.4f}")
    print(f"    Min posterior std:             {std_from_logvar.min():.4f}")
    print(f"    Max posterior std:             {std_from_logvar.max():.4f}")

    # How many dims needed to capture 90% of within-CT variance?
    within_var_sorted = np.sort(within_var)[::-1]
    cumsum = np.cumsum(within_var_sorted) / (within_var_sorted.sum() + 1e-10)
    n_for_90 = np.searchsorted(cumsum, 0.9) + 1
    print(f"    Dims for 90% within-CT var:   {n_for_90}/{n_dims}")

    # Dims where within-CT variance dominates
    n_within_dominant = np.sum(frac_within > 0.5)
    print(f"    Dims where within > between:  {n_within_dominant}/{n_dims}")

    # ---- 5. Plots ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Latent Diagnostic: {label}', fontsize=14)

    # 5a. Total variance per dim (sorted)
    ax = axes[0, 0]
    ax.bar(range(n_dims), total_var[sort_idx], color='steelblue', alpha=0.8)
    ax.axhline(0.01, color='red', linestyle='--', alpha=0.5, label='active threshold')
    ax.set_xlabel('Dimension (sorted by variance)')
    ax.set_ylabel('Total variance')
    ax.set_title('Per-dim total variance')
    ax.legend(fontsize=8)

    # 5b. Stacked bar: between vs within
    ax = axes[0, 1]
    ax.bar(range(n_dims), frac_between[sort_idx], color='coral', alpha=0.8, label='Between-CT')
    ax.bar(range(n_dims), frac_within[sort_idx], bottom=frac_between[sort_idx],
           color='steelblue', alpha=0.8, label='Within-CT')
    ax.axhline(0.5, color='black', linestyle=':', alpha=0.3)
    ax.set_xlabel('Dimension (sorted by total var)')
    ax.set_ylabel('Fraction of variance')
    ax.set_title('Between-CT vs Within-CT per dim')
    ax.legend(fontsize=8)

    # 5c. Within-CT variance per dim
    ax = axes[0, 2]
    within_sorted_idx = np.argsort(-within_var)
    ax.bar(range(n_dims), within_var[within_sorted_idx], color='seagreen', alpha=0.8)
    ax.set_xlabel('Dimension (sorted by within-CT var)')
    ax.set_ylabel('Within-CT variance')
    ax.set_title('Per-dim within-CT variance')

    # 5d. Posterior std per dim
    ax = axes[1, 0]
    ax.bar(range(n_dims), std_from_logvar[sort_idx], color='mediumpurple', alpha=0.8)
    ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='prior std')
    ax.set_xlabel('Dimension (sorted by total var)')
    ax.set_ylabel('Mean posterior std')
    ax.set_title('Posterior collapse check')
    ax.legend(fontsize=8)

    # 5e. Scatter: total var vs frac_within
    ax = axes[1, 1]
    ax.scatter(total_var, frac_within, alpha=0.7, s=50, c='darkorange', edgecolor='black')
    for i in range(n_dims):
        ax.annotate(str(i), (total_var[i], frac_within[i]), fontsize=6, alpha=0.7)
    ax.set_xlabel('Total variance')
    ax.set_ylabel('Fraction within-CT')
    ax.set_title('Which dims carry per-cell info?')
    ax.axhline(0.5, color='black', linestyle=':', alpha=0.3)

    # 5f. Cumulative within-CT variance
    ax = axes[1, 2]
    ax.plot(range(1, n_dims + 1), cumsum, 'o-', color='seagreen', markersize=4)
    ax.axhline(0.9, color='red', linestyle='--', alpha=0.5, label='90%')
    ax.set_xlabel('# dimensions (sorted by within-CT var)')
    ax.set_ylabel('Cumulative fraction')
    ax.set_title(f'Cumulative within-CT variance ({n_for_90} dims for 90%)')
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, f'latent_diagnostic_{label}.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\n  Figure saved: {fig_path}")

    # ---- Save CSV ----
    df = pd.DataFrame({
        'dim': range(n_dims),
        'total_var': total_var,
        'frac_between': frac_between,
        'frac_within': frac_within,
        'within_var': within_var,
        'mean_log_var': mean_log_var,
        'posterior_std': std_from_logvar,
    })
    csv_path = os.path.join(output_dir, f'latent_diagnostic_{label}.csv')
    df.to_csv(csv_path, index=False)
    print(f"  CSV saved: {csv_path}")

    return {
        'n_active': int(n_active),
        'n_dead': int(n_dead),
        'overall_frac_within': float(overall_frac_within),
        'mean_posterior_std': float(std_from_logvar.mean()),
        'dims_for_90pct_within': int(n_for_90),
        'n_within_dominant': int(n_within_dominant),
    }


def main():
    args = parse_args()

    output_dir = args.output_dir or os.path.join(
        'notes/experiments/SUC-6_variance_retention', 'latent_diagnostic')
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    datasets = load_data(args.gene_list)

    print("Loading model...")
    model, n_latent = load_model(args.run_folder, datasets['nuclei']['n_genes'])

    # Extract latents for both modalities
    results = {}
    for modality in ['nuclei', 'cell']:
        print(f"\nExtracting latents for {modality}...")
        mu, log_var = extract_latents(model, datasets[modality]['X_log'])
        cell_types = datasets[modality]['cell_types']

        summary = analyze_latent(mu, log_var, cell_types, f"{modality}", output_dir)
        results[modality] = summary

    # ---- Cross-modality comparison ----
    print(f"\n{'='*70}")
    print("  CROSS-MODALITY COMPARISON")
    print(f"{'='*70}")
    for key in results['nuclei']:
        print(f"  {key:30s}  nuclei={results['nuclei'][key]}  cell={results['cell'][key]}")

    # ---- Inter-modality variance decomposition ----
    # Re-extract latents to get raw mu arrays
    print(f"\n{'='*70}")
    print("  INTER-MODALITY VARIANCE ANALYSIS")
    print(f"{'='*70}")

    mu_cell, _ = extract_latents(model, datasets['cell']['X_log'])
    mu_nuc, _ = extract_latents(model, datasets['nuclei']['X_log'])

    n_dims = mu_cell.shape[1]
    cell_mean = mu_cell.mean(axis=0)
    nuc_mean = mu_nuc.mean(axis=0)
    shift = cell_mean - nuc_mean

    # Pool both modalities, compute between-modality variance fraction
    all_mu = np.concatenate([mu_cell, mu_nuc], axis=0)
    n_c, n_n = len(mu_cell), len(mu_nuc)
    grand_mean = all_mu.mean(axis=0)
    ss_total = np.sum((all_mu - grand_mean) ** 2, axis=0)
    ss_between_mod = (n_c * (cell_mean - grand_mean) ** 2 +
                      n_n * (nuc_mean - grand_mean) ** 2)
    frac_inter_mod = ss_between_mod / (ss_total + 1e-10)

    print(f"\n  Per-dimension inter-modality analysis (sorted by inter-mod fraction):")
    print(f"  {'Dim':>4} {'CellMean':>9} {'NucMean':>8} {'Shift':>7} "
          f"{'FracInterMod':>13} {'NucVar':>8} {'CellVar':>9}")
    print(f"  {'-'*4} {'-'*9} {'-'*8} {'-'*7} {'-'*13} {'-'*8} {'-'*9}")

    # Load per-modality variance from CSVs for reference
    nuc_csv = pd.read_csv(os.path.join(output_dir, 'latent_diagnostic_nuclei.csv'))
    cell_csv = pd.read_csv(os.path.join(output_dir, 'latent_diagnostic_cell.csv'))

    order = np.argsort(-frac_inter_mod)
    for dim in order:
        nv = nuc_csv.loc[nuc_csv['dim'] == dim, 'total_var'].values[0]
        cv = cell_csv.loc[cell_csv['dim'] == dim, 'total_var'].values[0]
        print(f"  {dim:>4} {cell_mean[dim]:>9.4f} {nuc_mean[dim]:>8.4f} "
              f"{shift[dim]:>7.3f} {frac_inter_mod[dim]:>13.4f} "
              f"{nv:>8.4f} {cv:>9.4f}")

    print(f"\n  Summary:")
    print(f"    Dims with >10% inter-modality var: {(frac_inter_mod > 0.1).sum()}/{n_dims}")
    print(f"    Dims with >50% inter-modality var: {(frac_inter_mod > 0.5).sum()}/{n_dims}")
    print(f"    Mean frac inter-modality:          {frac_inter_mod.mean():.4f}")
    print(f"    Max frac inter-modality:           {frac_inter_mod.max():.4f} (dim {np.argmax(frac_inter_mod)})")

    dead_nuc_dims = [d for d in range(n_dims)
                     if nuc_csv.loc[nuc_csv['dim'] == d, 'total_var'].values[0] <= 0.01]
    if dead_nuc_dims:
        print(f"    Dead nuclei dims {dead_nuc_dims}:")
        for d in dead_nuc_dims:
            print(f"      dim {d}: inter-mod frac={frac_inter_mod[d]:.4f}, "
                  f"shift={shift[d]:.4f}")

    # Save inter-modality CSV
    inter_df = pd.DataFrame({
        'dim': range(n_dims),
        'cell_mean': cell_mean,
        'nuc_mean': nuc_mean,
        'shift': shift,
        'frac_inter_modality': frac_inter_mod,
    })
    inter_csv = os.path.join(output_dir, 'latent_inter_modality.csv')
    inter_df.to_csv(inter_csv, index=False)
    print(f"\n  Inter-modality CSV saved: {inter_csv}")

    # ---- Within-CT PCA on gene expression space ----
    print(f"\n{'='*70}")
    print("  WITHIN-CT DIMENSIONALITY (PCA on gene expression residuals)")
    print(f"{'='*70}")

    from sklearn.decomposition import PCA

    for modality in ['cell', 'nuclei']:
        X_log = datasets[modality]['X_log']
        ct = datasets[modality]['cell_types']
        unique_cts = np.unique(ct)
        n_cells, n_genes = X_log.shape

        print(f"\n  --- {modality} ({n_cells} cells, {n_genes} genes) ---")

        # Subtract cell-type means to get residuals
        residuals = np.zeros_like(X_log)
        for c in unique_cts:
            mask = ct == c
            ct_mean = X_log[mask].mean(axis=0)
            residuals[mask] = X_log[mask] - ct_mean

        # Total within-CT variance (sum across genes)
        total_var = np.var(residuals, axis=0).sum()
        print(f"  Total within-CT variance (sum across genes): {total_var:.2f}")
        print(f"  Mean within-CT variance per gene: {np.var(residuals, axis=0).mean():.6f}")

        # PCA on residuals
        max_components = min(500, n_cells - 1, n_genes)
        pca = PCA(n_components=max_components, random_state=42)
        pca.fit(residuals)

        explained = np.cumsum(pca.explained_variance_ratio_)

        # Key thresholds
        for target in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
            n_needed = np.searchsorted(explained, target) + 1
            if n_needed <= max_components:
                print(f"  PCs for {target*100:.0f}% within-CT variance: {n_needed}")
            else:
                print(f"  PCs for {target*100:.0f}% within-CT variance: >{max_components}")

        # How much can 32 dims capture?
        if len(explained) >= 32:
            print(f"  Variance captured by 32 PCs: {explained[31]*100:.1f}%")
        if len(explained) >= 64:
            print(f"  Variance captured by 64 PCs: {explained[63]*100:.1f}%")
        if len(explained) >= 128:
            print(f"  Variance captured by 128 PCs: {explained[127]*100:.1f}%")

        # Top eigenvalues
        print(f"  Top 10 eigenvalues (explained var ratio):")
        for i in range(min(10, len(pca.explained_variance_ratio_))):
            print(f"    PC{i+1}: {pca.explained_variance_ratio_[i]:.4f} "
                  f"(cumulative: {explained[i]:.4f})")

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'Within-CT Dimensionality: {modality}', fontsize=14)

        ax = axes[0]
        ax.plot(range(1, max_components + 1), explained, 'b-', linewidth=1.5)
        ax.axhline(0.9, color='red', linestyle='--', alpha=0.5, label='90%')
        ax.axhline(0.5, color='orange', linestyle='--', alpha=0.5, label='50%')
        ax.axvline(32, color='green', linestyle='--', alpha=0.5, label='32 dims')
        if len(explained) >= 32:
            ax.plot(32, explained[31], 'go', markersize=10,
                    label=f'32 PCs = {explained[31]*100:.1f}%')
        ax.set_xlabel('Number of PCs')
        ax.set_ylabel('Cumulative variance explained')
        ax.set_title('Cumulative explained variance')
        ax.legend(fontsize=8)
        ax.set_xlim(0, min(200, max_components))

        ax = axes[1]
        n_show = min(100, max_components)
        ax.bar(range(1, n_show + 1), pca.explained_variance_ratio_[:n_show],
               color='steelblue', alpha=0.8)
        ax.axvline(32, color='green', linestyle='--', alpha=0.5, label='32 dims')
        ax.set_xlabel('PC')
        ax.set_ylabel('Variance explained (fraction)')
        ax.set_title('Scree plot')
        ax.legend(fontsize=8)

        plt.tight_layout()
        fig_path = os.path.join(output_dir, f'within_ct_pca_{modality}.png')
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"  Figure saved: {fig_path}")

        # Save explained variance
        pca_df = pd.DataFrame({
            'pc': range(1, max_components + 1),
            'explained_var_ratio': pca.explained_variance_ratio_,
            'cumulative': explained,
        })
        pca_csv = os.path.join(output_dir, f'within_ct_pca_{modality}.csv')
        pca_df.to_csv(pca_csv, index=False)
        print(f"  CSV saved: {pca_csv}")

        del residuals, pca
        import gc; gc.collect()

    print("\nDone!")


if __name__ == '__main__':
    main()
