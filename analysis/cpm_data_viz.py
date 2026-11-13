"""
CPM normalization data visualization.

Compares raw counts, log1p, and CPM (log1p of counts-per-million) normalizations
on the nuclei v3 dataset used for CPM training experiments.

Figures:
  1. Expression value distribution (histograms): raw vs log1p vs CPM
  2. Library size distribution and CPM effect
  3. Per-gene mean expression: log1p vs CPM scatter
  4. Per-gene std comparison: log1p vs CPM (across cells, nonzero only)
  5. Value range summary across normalizations

Usage:
    python -m analysis.cpm_data_viz [--n_cells 5000] [--out_dir notes/experiments/SUC-6_variance_retention/cpm_viz]
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi/10X_nuclei_v2_v3_AIBS/data.h5ad"
GENE_LIST = os.path.join(_PROJECT_DIR, "data", "gene_list_perct_5pct.csv")


def load_data(n_cells=5000, seed=42):
    """Load nuclei h5ad, filter to v3/n3A, apply gene list, subsample.

    Uses backed='r' to filter obs/var before materializing the dense matrix,
    avoiding OOM on the full 116k × 25k array.
    """
    import anndata as ad

    print(f"Loading metadata from {DATA_PATH} ...")
    adata = ad.read_h5ad(DATA_PATH, backed='r')
    print(f"  Full shape: {adata.shape}")

    # Build meta_batch_id from batch_id (e.g. '1-n3A.2' -> '1-n3A')
    obs = adata.obs.copy()
    if 'meta_batch_id' not in obs.columns and 'batch_id' in obs.columns:
        obs['meta_batch_id'] = obs['batch_id'].apply(lambda x: x.split('.')[0])

    # Filter to n3A (protocol v3, sample A nuclei): meta_batch_id ends with 'n3A'
    cell_mask = obs['meta_batch_id'].str.endswith('n3A')
    cell_idx = np.where(cell_mask)[0]
    print(f"  n3A cells: {len(cell_idx)}")

    # Gene list filter on var
    gene_list = set(pd.read_csv(GENE_LIST, header=None)[0].tolist())
    if 'gene_names' in adata.var.columns:
        gene_mask = adata.var['gene_names'].isin(gene_list).values
    else:
        gene_mask = np.array([g in gene_list for g in adata.var_names])
    gene_idx = np.where(gene_mask)[0]
    print(f"  Genes after filter: {len(gene_idx)}")

    # Subsample cells before materializing
    rng = np.random.default_rng(seed)
    n = min(n_cells, len(cell_idx))
    chosen = rng.choice(len(cell_idx), size=n, replace=False)
    chosen = np.sort(chosen)
    cell_idx_sub = cell_idx[chosen]

    # Load obs metadata for chosen cells
    obs_sub = obs.iloc[cell_idx_sub].copy()

    # Materialize only the needed slice — backed h5ad supports integer indexing
    print(f"  Materializing {len(cell_idx_sub)} cells × {len(gene_idx)} genes ...")
    X_slice = adata.X[cell_idx_sub]  # still sparse or dense depending on storage
    if hasattr(X_slice, 'toarray'):
        X_slice = X_slice.toarray()
    X_sub = X_slice[:, gene_idx].astype(np.float32)

    adata.file.close()

    print(f"  Final shape: {X_sub.shape}")
    # Build a lightweight AnnData for downstream use
    import anndata as ad2
    adata_sub = ad2.AnnData(X=X_sub, obs=obs_sub)
    return adata_sub


def compute_normalizations(adata):
    """Return dense matrices for raw, log1p, and CPM (log1p) normalizations."""
    X = adata.X
    if sp.issparse(X):
        X_raw = X.toarray().astype(np.float32)
    else:
        X_raw = np.array(X, dtype=np.float32)

    X_log1p = np.log1p(X_raw)

    lib_size = X_raw.sum(axis=1)
    lib_size = np.clip(lib_size, 1.0, None)
    X_cpm = np.log1p(X_raw / lib_size[:, None] * 1e6)

    return X_raw, X_log1p, X_cpm, lib_size


def fig1_value_distributions(X_raw, X_log1p, X_cpm, out_dir):
    """Histogram of all nonzero expression values for each normalization."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Expression value distributions (nonzero entries)', fontsize=13)

    configs = [
        (X_raw, 'Raw counts', '#d62728', 100),
        (X_log1p, 'log1p(counts)', '#1f77b4', 80),
        (X_cpm, 'log1p(CPM)', '#2ca02c', 80),
    ]

    for ax, (X, label, color, bins) in zip(axes, configs):
        vals = X[X > 0].ravel()
        # Subsample for speed
        if len(vals) > 500_000:
            vals = np.random.default_rng(0).choice(vals, 500_000, replace=False)
        ax.hist(vals, bins=bins, color=color, alpha=0.8, density=True)
        ax.set_xlabel('Expression value', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{label}\np50={np.median(vals):.2f}, p99={np.percentile(vals,99):.2f}', fontsize=11)
        ax.axvline(np.mean(vals), color='k', lw=1.5, ls='--', label=f'mean={np.mean(vals):.2f}')
        ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, 'fig1_value_distributions.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


def fig2_library_size(X_raw, X_log1p, X_cpm, out_dir):
    """Show that log1p preserves library size variation while CPM removes it.

    Per-cell effective library size = sum(expm1(X)) recovers the original
    total counts from log1p, and ~10^6 from CPM (flat by construction).
    """
    lib_from_log1p = np.expm1(X_log1p).sum(axis=1)   # should match raw lib size
    lib_from_cpm   = np.expm1(X_cpm).sum(axis=1)     # should be ~10^6 per cell

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Per-cell effective library size after normalization', fontsize=13)

    ax = axes[0]
    ax.hist(lib_from_log1p, bins=80, color='#1f77b4', alpha=0.8)
    ax.set_xlabel('sum(expm1(log1p values)) per cell', fontsize=11)
    ax.set_ylabel('# cells', fontsize=11)
    cv = lib_from_log1p.std() / lib_from_log1p.mean()
    ax.set_title(f'log1p(counts): library size still varies\nmean={lib_from_log1p.mean():.0f}, CV={cv:.3f}', fontsize=11)
    ax.axvline(lib_from_log1p.mean(), color='k', lw=1.5, ls='--', label=f'mean')
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.hist(lib_from_cpm, bins=80, color='#2ca02c', alpha=0.8)
    ax.set_xlabel('sum(expm1(CPM values)) per cell', fontsize=11)
    ax.set_ylabel('# cells', fontsize=11)
    cv2 = lib_from_cpm.std() / lib_from_cpm.mean()
    ax.set_title(f'log1p(CPM): all cells normalized to same total\nmean={lib_from_cpm.mean():.0f}, CV={cv2:.4f}', fontsize=11)
    ax.axvline(lib_from_cpm.mean(), color='k', lw=1.5, ls='--', label=f'mean')
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, 'fig2_library_size.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


def fig3_per_gene_means(X_log1p, X_cpm, out_dir):
    """Scatter: per-gene mean in log1p vs CPM."""
    mean_log1p = X_log1p.mean(axis=0)
    mean_cpm = X_cpm.mean(axis=0)

    # Pearson on nonzero-detected genes
    detected = (X_log1p > 0).mean(axis=0) > 0.01
    r = np.corrcoef(mean_log1p[detected], mean_cpm[detected])[0, 1]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(mean_log1p, mean_cpm, s=2, alpha=0.3, color='#8c564b')
    # Diagonal reference isn't meaningful here (different scales); add regression line
    lims_x = [mean_log1p.min(), mean_log1p.max()]
    fit = np.polyfit(mean_log1p[detected], mean_cpm[detected], 1)
    ax.plot(lims_x, np.polyval(fit, lims_x), 'r--', lw=1.5, alpha=0.8,
            label=f'fit: slope={fit[0]:.2f}, r={r:.3f}')
    ax.set_xlabel('Per-gene mean: log1p(counts)', fontsize=12)
    ax.set_ylabel('Per-gene mean: log1p(CPM)', fontsize=12)
    ax.set_title('Per-gene mean expression: log1p vs CPM\n(each dot = one gene)', fontsize=12)
    ax.legend(fontsize=10)

    plt.tight_layout()
    path = os.path.join(out_dir, 'fig3_pergene_means.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


def fig4_value_ranges(X_raw, X_log1p, X_cpm, out_dir):
    """Boxplot comparing value ranges of the three normalizations (nonzero only)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    rng = np.random.default_rng(1)
    n_sample = 200_000

    def _sample_nonzero(X):
        vals = X[X > 0].ravel()
        if len(vals) > n_sample:
            vals = rng.choice(vals, n_sample, replace=False)
        return vals

    data = [_sample_nonzero(X_raw), _sample_nonzero(X_log1p), _sample_nonzero(X_cpm)]
    labels = ['Raw counts', 'log1p(counts)', 'log1p(CPM)']
    colors = ['#d62728', '#1f77b4', '#2ca02c']

    bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.5)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Expression value (nonzero entries)', fontsize=12)
    ax.set_title('Value range comparison across normalizations\n(nonzero entries, outliers hidden)', fontsize=12)

    # Print stats
    for i, (vals, lbl) in enumerate(zip(data, labels)):
        print(f"  {lbl}: mean={vals.mean():.3f}, std={vals.std():.3f}, p99={np.percentile(vals,99):.3f}, max={vals.max():.2f}")

    plt.tight_layout()
    path = os.path.join(out_dir, 'fig4_value_ranges.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


def fig5_sparsity_comparison(X_raw, X_log1p, X_cpm, out_dir):
    """Per-gene sparsity (fraction zeros): log1p vs CPM should be identical."""
    sparsity_raw = (X_raw == 0).mean(axis=0)
    sparsity_log1p = (X_log1p == 0).mean(axis=0)
    sparsity_cpm = (X_cpm == 0).mean(axis=0)

    assert np.allclose(sparsity_log1p, sparsity_cpm, atol=1e-6), \
        "CPM and log1p sparsity differ — unexpected!"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Per-gene sparsity (fraction zeros)', fontsize=13)

    ax = axes[0]
    ax.hist(sparsity_raw, bins=60, color='#9467bd', alpha=0.8)
    ax.set_xlabel('Fraction zero cells', fontsize=11)
    ax.set_ylabel('# genes', fontsize=11)
    ax.set_title(f'Sparsity distribution\nmean={sparsity_raw.mean():.3f}, median={np.median(sparsity_raw):.3f}', fontsize=11)

    ax = axes[1]
    ax.scatter(sparsity_raw, sparsity_cpm, s=2, alpha=0.3, color='#17becf')
    lim = [0, 1]
    ax.plot(lim, lim, 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('Sparsity: raw', fontsize=11)
    ax.set_ylabel('Sparsity: log1p(CPM)', fontsize=11)
    ax.set_title('Sparsity identical between normalizations\n(zeros → zeros through CPM)', fontsize=11)

    plt.tight_layout()
    path = os.path.join(out_dir, 'fig5_sparsity.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


def fig6_cell_type_distributions(X_log1p, X_cpm, adata, out_dir):
    """Per-cell-type mean expression: how CPM affects cell type separation."""
    if 'cell_type' not in adata.obs.columns:
        print("  No cell_type column, skipping fig6")
        return

    cts = adata.obs['cell_type'].values
    unique_cts = sorted(set(cts))
    if len(unique_cts) > 15:
        # Pick top 8 by frequency
        from collections import Counter
        freq = Counter(cts)
        unique_cts = [ct for ct, _ in freq.most_common(8)]

    n_genes_show = 2000
    rng = np.random.default_rng(2)
    gene_idx = rng.choice(X_log1p.shape[1], min(n_genes_show, X_log1p.shape[1]), replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Per-cell-type mean expression (random 2000 genes)', fontsize=13)

    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_cts)))

    for ax, (X, title) in zip(axes, [(X_log1p, 'log1p(counts)'), (X_cpm, 'log1p(CPM)')]):
        ct_means = []
        for ct in unique_cts:
            mask = cts == ct
            ct_means.append(X[mask][:, gene_idx].mean(axis=0))

        # Sort genes by variance across CTs
        ct_means_arr = np.array(ct_means)
        gene_var = ct_means_arr.var(axis=0)
        order = np.argsort(-gene_var)[:500]

        for i, (ct, ct_mean) in enumerate(zip(unique_cts, ct_means)):
            ax.plot(np.arange(500), ct_mean[order], lw=0.8, alpha=0.7, color=colors[i], label=ct)

        ax.set_xlabel('Gene (sorted by CT variance)', fontsize=11)
        ax.set_ylabel('Mean expression', fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=7, loc='upper right', ncol=2)

    plt.tight_layout()
    path = os.path.join(out_dir, 'fig6_ct_mean_expression.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_cells', type=int, default=5000)
    parser.add_argument('--out_dir', type=str,
                        default='notes/experiments/SUC-6_variance_retention/cpm_viz')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    out_dir = os.path.join(_PROJECT_DIR, args.out_dir) if not os.path.isabs(args.out_dir) else args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir: {out_dir}")

    adata = load_data(n_cells=args.n_cells, seed=args.seed)
    X_raw, X_log1p, X_cpm, lib_size = compute_normalizations(adata)

    print(f"\nData shape: {X_raw.shape}")
    print(f"Raw:   mean(nonzero)={X_raw[X_raw>0].mean():.2f}, max={X_raw.max():.0f}")
    print(f"log1p: mean(nonzero)={X_log1p[X_log1p>0].mean():.3f}, max={X_log1p.max():.3f}")
    print(f"CPM:   mean(nonzero)={X_cpm[X_cpm>0].mean():.3f}, max={X_cpm.max():.3f}")
    print()

    print("Generating figures...")
    fig1_value_distributions(X_raw, X_log1p, X_cpm, out_dir)
    fig2_library_size(X_raw, X_log1p, X_cpm, out_dir)
    fig3_per_gene_means(X_log1p, X_cpm, out_dir)
    fig4_value_ranges(X_raw, X_log1p, X_cpm, out_dir)
    fig5_sparsity_comparison(X_raw, X_log1p, X_cpm, out_dir)
    fig6_cell_type_distributions(X_log1p, X_cpm, adata, out_dir)

    print(f"\nAll figures saved to {out_dir}")


if __name__ == '__main__':
    main()
