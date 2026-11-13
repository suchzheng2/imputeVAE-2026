"""
Library size analysis for SUC-6 variance retention.

Investigates how much within-CT variance is driven by library size,
whether the decoder preserves library-size information, and whether
proportion normalization changes the variance collapse picture.

Analyses:
  1. Library size preservation: raw vs decoded lib size per cell
  2. Variance decomposition: per-gene R² of expression ~ log(libsize)
  3. Proportion-normalized std_ratio comparison
  4. Per-gene correlation of decoded expression with raw library size
  5. Latent dimension correlation with library size

Usage:
    python -m analysis.library_size_analysis --model pct5v2_cond
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_DIR)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_library_sizes(X_log1p):
    """Recover raw counts via expm1, sum across genes. Returns [N] array."""
    raw = np.expm1(np.clip(X_log1p, 0, None))  # clip negatives from decoder
    return raw.sum(axis=1)


def _pearson_r(x, y):
    """Pearson correlation, returns 0 for constant inputs."""
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return np.corrcoef(x, y)[0, 1]


def _spearman_r(x, y):
    if len(x) < 3:
        return 0.0
    r, _ = stats.spearmanr(x, y)
    return r if np.isfinite(r) else 0.0


# ---------------------------------------------------------------------------
# Analysis 1: Library size preservation
# ---------------------------------------------------------------------------

def analysis1_libsize_preservation(X_cell, decoded_cell, ct_cell, common_cts):
    """Global and per-CT correlation of raw vs decoded library sizes."""
    lib_raw = compute_library_sizes(X_cell)
    lib_dec = compute_library_sizes(decoded_cell)

    results = {
        'global_pearson': _pearson_r(lib_raw, lib_dec),
        'global_spearman': _spearman_r(lib_raw, lib_dec),
        'global_lib_raw_mean': lib_raw.mean(),
        'global_lib_dec_mean': lib_dec.mean(),
        'global_lib_raw_std': lib_raw.std(),
        'global_lib_dec_std': lib_dec.std(),
    }

    per_ct = []
    for ct in common_cts:
        mask = ct_cell == ct
        lr, ld = lib_raw[mask], lib_dec[mask]
        per_ct.append({
            'ct': ct,
            'n': mask.sum(),
            'lib_raw_mean': lr.mean(),
            'lib_dec_mean': ld.mean(),
            'lib_raw_std': lr.std(),
            'lib_dec_std': ld.std(),
            'pearson': _pearson_r(lr, ld),
            'spearman': _spearman_r(lr, ld),
        })

    return results, pd.DataFrame(per_ct), lib_raw, lib_dec


# ---------------------------------------------------------------------------
# Analysis 2: Variance decomposition via library size regression
# ---------------------------------------------------------------------------

def analysis2_variance_decomposition(X_cell, decoded_cell, lib_raw, lib_dec,
                                      ct_cell, common_cts, gene_names):
    """Per-gene R² of expression ~ log(libsize), raw vs decoded."""
    log_lib_raw = np.log1p(lib_raw)
    log_lib_dec = np.log1p(lib_dec)

    rows = []
    for ct in common_cts:
        mask = ct_cell == ct
        n = mask.sum()
        if n < 10:
            continue

        X_ct = X_cell[mask]
        D_ct = decoded_cell[mask]
        ll_raw = log_lib_raw[mask]
        ll_dec = log_lib_dec[mask]

        # Vectorized: corr of each gene with log_libsize
        # R² = pearson(gene, log_lib)²
        # Center the data
        X_centered = X_ct - X_ct.mean(axis=0, keepdims=True)
        ll_raw_c = ll_raw - ll_raw.mean()
        D_centered = D_ct - D_ct.mean(axis=0, keepdims=True)
        ll_dec_c = ll_dec - ll_dec.mean()

        # Pearson correlations (vectorized across genes)
        x_norms = np.sqrt((X_centered ** 2).sum(axis=0))
        d_norms = np.sqrt((D_centered ** 2).sum(axis=0))
        ll_raw_norm = np.sqrt((ll_raw_c ** 2).sum())
        ll_dec_norm = np.sqrt((ll_dec_c ** 2).sum())

        # Avoid division by zero
        x_norms = np.where(x_norms == 0, 1, x_norms)
        d_norms = np.where(d_norms == 0, 1, d_norms)

        corr_raw = (X_centered * ll_raw_c[:, None]).sum(axis=0) / (x_norms * ll_raw_norm)
        corr_dec = (D_centered * ll_dec_c[:, None]).sum(axis=0) / (d_norms * ll_dec_norm)

        r2_raw = corr_raw ** 2
        r2_dec = corr_dec ** 2

        # Per-gene total variance
        var_raw = X_ct.var(axis=0)
        var_dec = D_ct.var(axis=0)

        # Frac nonzero
        frac_nz = (X_ct > 0).mean(axis=0)

        for g in range(len(gene_names)):
            rows.append({
                'ct': ct,
                'gene': gene_names[g],
                'frac_nonzero': frac_nz[g],
                'var_raw': var_raw[g],
                'var_dec': var_dec[g],
                'r2_raw_libsize': r2_raw[g],
                'r2_dec_libsize': r2_dec[g],
                'var_raw_lib_explained': var_raw[g] * r2_raw[g],
                'var_raw_residual': var_raw[g] * (1 - r2_raw[g]),
                'var_dec_lib_explained': var_dec[g] * r2_dec[g],
                'var_dec_residual': var_dec[g] * (1 - r2_dec[g]),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis 3: Proportion-normalized std comparison
# ---------------------------------------------------------------------------

def analysis3_proportion_normalized(X_cell, decoded_cell, lib_raw, lib_dec,
                                     ct_cell, common_cts, gene_names):
    """Within-CT std_ratio on proportion-normalized data vs log1p."""
    # Proportion normalize: counts / libsize, then log1p
    raw_counts = np.expm1(np.clip(X_cell, 0, None))
    dec_counts = np.expm1(np.clip(decoded_cell, 0, None))

    # Avoid div by zero
    lib_raw_safe = np.where(lib_raw == 0, 1, lib_raw)
    lib_dec_safe = np.where(lib_dec == 0, 1, lib_dec)

    prop_raw = np.log1p(raw_counts / lib_raw_safe[:, None])
    prop_dec = np.log1p(dec_counts / lib_dec_safe[:, None])

    rows = []
    for ct in common_cts:
        mask = ct_cell == ct
        n = mask.sum()
        if n < 10:
            continue

        # log1p space
        std_raw_log1p = X_cell[mask].std(axis=0)
        std_dec_log1p = decoded_cell[mask].std(axis=0)

        # Proportion space
        std_raw_prop = prop_raw[mask].std(axis=0)
        std_dec_prop = prop_dec[mask].std(axis=0)

        frac_nz = (X_cell[mask] > 0).mean(axis=0)

        for g in range(len(gene_names)):
            sr_log1p = std_dec_log1p[g] / std_raw_log1p[g] if std_raw_log1p[g] > 0 else 0
            sr_prop = std_dec_prop[g] / std_raw_prop[g] if std_raw_prop[g] > 0 else 0
            rows.append({
                'ct': ct,
                'gene': gene_names[g],
                'frac_nonzero': frac_nz[g],
                'std_raw_log1p': std_raw_log1p[g],
                'std_dec_log1p': std_dec_log1p[g],
                'std_ratio_log1p': sr_log1p,
                'std_raw_prop': std_raw_prop[g],
                'std_dec_prop': std_dec_prop[g],
                'std_ratio_prop': sr_prop,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis 5: Latent dimension correlation with library size
# ---------------------------------------------------------------------------

def analysis5_latent_libsize(latent_cell, lib_raw_cell, latent_nuc, lib_raw_nuc):
    """Correlation of each latent dimension with log(library size)."""
    results = []
    for label, latent, lib in [('cell', latent_cell, lib_raw_cell),
                                ('nuclei', latent_nuc, lib_raw_nuc)]:
        log_lib = np.log1p(lib)
        n_dims = latent.shape[1]
        for d in range(n_dims):
            r = _pearson_r(latent[:, d], log_lib)
            results.append({
                'modality': label,
                'dim': d,
                'pearson_with_log_libsize': r,
                'abs_pearson': abs(r),
            })
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_libsize_scatter(lib_raw, lib_dec, ct_cell, common_cts, out_dir):
    """Scatter of raw vs decoded library size, colored by CT."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 8))

    cmap = plt.cm.get_cmap('tab20', len(common_cts))
    for i, ct in enumerate(common_cts):
        mask = ct_cell == ct
        ax.scatter(lib_raw[mask], lib_dec[mask], s=3, alpha=0.3,
                   c=[cmap(i)], label=ct)

    mx = max(np.percentile(lib_raw, 99.5), np.percentile(lib_dec, 99.5))
    ax.plot([0, mx], [0, mx], 'k--', lw=1, alpha=0.5, label='y = x')
    ax.set_xlabel('Raw library size (sum of counts)', fontsize=13)
    ax.set_ylabel('Decoded library size (sum of expm1(decoded))', fontsize=13)
    ax.set_title('Per-cell library size: raw vs decoded', fontsize=14)
    ax.legend(fontsize=7, markerscale=4, ncol=2, loc='upper left')
    r = _pearson_r(lib_raw, lib_dec)
    ax.text(0.97, 0.03, f'Pearson r = {r:.3f}', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=12,
            bbox=dict(fc='white', alpha=0.8))

    fig.tight_layout()
    path = os.path.join(out_dir, 'fig_libsize_scatter.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig_variance_decomposition(df, out_dir):
    """Box plots: R² of expression ~ log(libsize) by sparsity bin, raw vs decoded."""
    bins = [
        (0.7, 1.01, '>70%'),
        (0.3, 0.7, '30-70%'),
        (0.1, 0.3, '10-30%'),
        (0, 0.1, '<10%'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, col, title in [(axes[0], 'r2_raw_libsize', 'Raw expression'),
                            (axes[1], 'r2_dec_libsize', 'Decoded expression')]:
        positions = []
        data_list = []
        labels = []
        for i, (lo, hi, label) in enumerate(bins):
            mask = (df['frac_nonzero'] >= lo) & (df['frac_nonzero'] < hi)
            vals = df.loc[mask, col].dropna()
            data_list.append(vals.values)
            positions.append(i)
            labels.append(f'{label}\n(n={len(vals):,})')

        bp = ax.boxplot(data_list, positions=positions, widths=0.6,
                        patch_artist=True, showfliers=False)
        for patch in bp['boxes']:
            patch.set_facecolor('#1f77b4' if 'Raw' in title else '#d62728')
            patch.set_alpha(0.5)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_title(title, fontsize=13)
        ax.set_ylabel('R² (expression ~ log library size)', fontsize=12)
        # Add median labels
        for i, d in enumerate(data_list):
            if len(d) > 0:
                med = np.median(d)
                ax.text(i, med + 0.02, f'{med:.3f}', ha='center', fontsize=9,
                        fontweight='bold')

    fig.suptitle('How much within-CT variance is explained by library size?\n'
                 'Per (CT, gene) R² by sparsity bin',
                 fontsize=14)
    fig.tight_layout()
    path = os.path.join(out_dir, 'fig_variance_decomposition.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig_proportion_std_comparison(df, out_dir):
    """Side-by-side box plots: std_ratio in log1p space vs proportion space."""
    bins = [
        (0.7, 1.01, '>70%'),
        (0.3, 0.7, '30-70%'),
        (0.1, 0.3, '10-30%'),
        (0, 0.1, '<10%'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, col, title in [(axes[0], 'std_ratio_log1p', 'log1p space (current)'),
                            (axes[1], 'std_ratio_prop', 'Proportion-normalized')]:
        data_list = []
        labels = []
        for lo, hi, label in bins:
            mask = (df['frac_nonzero'] >= lo) & (df['frac_nonzero'] < hi)
            vals = df.loc[mask, col].dropna()
            vals = vals[np.isfinite(vals)]
            data_list.append(vals.values)
            labels.append(f'{label}\n(n={len(vals):,})')

        bp = ax.boxplot(data_list, widths=0.6, patch_artist=True, showfliers=False)
        for patch in bp['boxes']:
            patch.set_facecolor('#2ca02c')
            patch.set_alpha(0.5)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_title(title, fontsize=13)
        ax.set_ylabel('std_ratio (decoded / raw)', fontsize=12)
        ax.axhline(1.0, color='k', ls='--', lw=1, alpha=0.3)
        for i, d in enumerate(data_list):
            if len(d) > 0:
                med = np.median(d)
                ax.text(i + 1, med + 0.02, f'{med:.3f}', ha='center', fontsize=9,
                        fontweight='bold')

    fig.suptitle('Does proportion normalization change the std collapse?\n'
                 'std_ratio by sparsity bin, log1p vs proportion space',
                 fontsize=14)
    fig.tight_layout()
    path = os.path.join(out_dir, 'fig_proportion_std_comparison.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig_latent_libsize(df, out_dir):
    """Bar chart of latent dimension correlations with log(library size)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

    for ax, modality in [(axes[0], 'cell'), (axes[1], 'nuclei')]:
        sub = df[df['modality'] == modality].sort_values('dim')
        colors = ['#d62728' if v > 0 else '#1f77b4'
                  for v in sub['pearson_with_log_libsize']]
        ax.bar(sub['dim'], sub['pearson_with_log_libsize'], color=colors, alpha=0.7)
        ax.set_xlabel('Latent dimension', fontsize=12)
        ax.set_ylabel('Pearson r with log(library size)', fontsize=12)
        ax.set_title(f'{modality.capitalize()} latent', fontsize=13)
        ax.axhline(0, color='k', lw=0.5)

        # Annotate top correlated dim
        top = sub.loc[sub['abs_pearson'].idxmax()]
        ax.annotate(f'dim {int(top["dim"])}: r={top["pearson_with_log_libsize"]:.3f}',
                    xy=(top['dim'], top['pearson_with_log_libsize']),
                    xytext=(top['dim'] + 5, top['pearson_with_log_libsize'] * 0.8),
                    fontsize=10, arrowprops=dict(arrowstyle='->', color='black'))

    fig.suptitle('Which latent dimensions encode library size?', fontsize=14)
    fig.tight_layout()
    path = os.path.join(out_dir, 'fig_latent_libsize.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig_r2_raw_vs_dec_scatter(df, out_dir):
    """Scatter: R²_raw vs R²_decoded, colored by sparsity."""
    bins = [
        (0, 0.1, '<10%', '#d62728'),
        (0.1, 0.3, '10-30%', '#ff7f0e'),
        (0.3, 0.7, '30-70%', '#2ca02c'),
        (0.7, 1.01, '>70%', '#1f77b4'),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    for lo, hi, label, color in bins:
        mask = (df['frac_nonzero'] >= lo) & (df['frac_nonzero'] < hi)
        s = df[mask]
        if len(s) > 5000:
            s = s.sample(5000, random_state=42)
        ax.scatter(s['r2_raw_libsize'], s['r2_dec_libsize'], s=2, alpha=0.3,
                   c=color, label=label)

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('R² raw expression ~ log(libsize)', fontsize=13)
    ax.set_ylabel('R² decoded expression ~ log(libsize)', fontsize=13)
    ax.set_title('Library size dependence: raw vs decoded\n'
                 '(each dot = one gene in one CT)', fontsize=13)
    ax.legend(title='Sparsity', fontsize=10, markerscale=5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')

    fig.tight_layout()
    path = os.path.join(out_dir, 'fig_r2_scatter_raw_vs_dec.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="pct5v2_cond")
    parser.add_argument("--out", type=str,
                        default="notes/experiments/SUC-6_variance_retention/library_size/")
    args = parser.parse_args()

    out_dir = os.path.join(_PROJECT_DIR, args.out)
    os.makedirs(out_dir, exist_ok=True)

    # --- Load data and run inference ---
    from analysis.eval.registry import get_model
    from analysis.eval.data import load_eval_data
    from analysis.eval.inference import load_model, run_5source_inference

    model_config = get_model(args.model)
    data = load_eval_data(model_config['gene_list'], max_per_type=5000)
    model, device = load_model(model_config)

    print("\nRunning inference...")
    sources = run_5source_inference(model, model_config,
                                     data.X_cell, data.X_nuc, device)

    import gc
    del model
    gc.collect()

    common_cts = data.common_cts

    # ===========================================================
    # Analysis 1: Library size preservation
    # ===========================================================
    print("\n=== Analysis 1: Library size preservation ===")
    global_stats, ct_stats, lib_raw, lib_dec = analysis1_libsize_preservation(
        data.X_cell, sources['decoded_cell'], data.ct_cell, common_cts
    )
    print(f"  Global Pearson:  {global_stats['global_pearson']:.4f}")
    print(f"  Global Spearman: {global_stats['global_spearman']:.4f}")
    print(f"  Raw lib mean:    {global_stats['global_lib_raw_mean']:.1f} "
          f"(std {global_stats['global_lib_raw_std']:.1f})")
    print(f"  Dec lib mean:    {global_stats['global_lib_dec_mean']:.1f} "
          f"(std {global_stats['global_lib_dec_std']:.1f})")
    print("\n  Per-CT library size correlation:")
    for _, row in ct_stats.iterrows():
        print(f"    {row['ct']:20s}  n={row['n']:5d}  "
              f"raw={row['lib_raw_mean']:8.0f}  dec={row['lib_dec_mean']:8.0f}  "
              f"r={row['pearson']:.3f}")

    ct_stats.to_csv(os.path.join(out_dir, 'libsize_per_ct.csv'), index=False)
    fig_libsize_scatter(lib_raw, lib_dec, data.ct_cell, common_cts, out_dir)

    # Also compute nuclei library sizes for analysis 5
    lib_raw_nuc = compute_library_sizes(data.X_nuc)

    # ===========================================================
    # Analysis 2: Variance decomposition
    # ===========================================================
    print("\n=== Analysis 2: Variance decomposition ===")
    df_decomp = analysis2_variance_decomposition(
        data.X_cell, sources['decoded_cell'], lib_raw, lib_dec,
        data.ct_cell, common_cts, data.gene_names
    )

    # Summary by sparsity
    bins = [(0.7, 1.01, '>70%'), (0.3, 0.7, '30-70%'),
            (0.1, 0.3, '10-30%'), (0, 0.1, '<10%')]
    print("\n  Median R² (expression ~ log library size) by sparsity:")
    print(f"  {'Bin':>10s}  {'R²_raw':>8s}  {'R²_dec':>8s}  {'n':>8s}")
    for lo, hi, label in bins:
        mask = (df_decomp['frac_nonzero'] >= lo) & (df_decomp['frac_nonzero'] < hi)
        sub = df_decomp[mask]
        print(f"  {label:>10s}  {sub['r2_raw_libsize'].median():>8.4f}  "
              f"{sub['r2_dec_libsize'].median():>8.4f}  {len(sub):>8d}")

    df_decomp.to_csv(os.path.join(out_dir, 'variance_decomposition.csv'), index=False)
    fig_variance_decomposition(df_decomp, out_dir)
    fig_r2_raw_vs_dec_scatter(df_decomp, out_dir)

    # ===========================================================
    # Analysis 3: Proportion-normalized std
    # ===========================================================
    print("\n=== Analysis 3: Proportion-normalized std comparison ===")
    df_prop = analysis3_proportion_normalized(
        data.X_cell, sources['decoded_cell'], lib_raw, lib_dec,
        data.ct_cell, common_cts, data.gene_names
    )

    print("\n  Median std_ratio by sparsity: log1p vs proportion-normalized")
    print(f"  {'Bin':>10s}  {'log1p':>8s}  {'prop':>8s}  {'n':>8s}")
    for lo, hi, label in bins:
        mask = (df_prop['frac_nonzero'] >= lo) & (df_prop['frac_nonzero'] < hi)
        sub = df_prop[mask]
        sr_log1p = sub['std_ratio_log1p'].median()
        sr_prop = sub['std_ratio_prop'].median()
        print(f"  {label:>10s}  {sr_log1p:>8.4f}  {sr_prop:>8.4f}  {len(sub):>8d}")

    df_prop.to_csv(os.path.join(out_dir, 'proportion_std_comparison.csv'), index=False)
    fig_proportion_std_comparison(df_prop, out_dir)

    # ===========================================================
    # Analysis 5: Latent-libsize correlation
    # ===========================================================
    print("\n=== Analysis 5: Latent-libsize correlation ===")
    latent_cell = sources.get('latent_cell')
    latent_nuc = sources.get('latent_nuc')
    if latent_cell is not None and latent_nuc is not None:
        df_latent = analysis5_latent_libsize(
            latent_cell, lib_raw, latent_nuc, lib_raw_nuc
        )

        for modality in ['cell', 'nuclei']:
            sub = df_latent[df_latent['modality'] == modality]
            top = sub.loc[sub['abs_pearson'].idxmax()]
            print(f"  {modality:7s}: max |r| = {top['abs_pearson']:.4f} "
                  f"(dim {int(top['dim'])}, r = {top['pearson_with_log_libsize']:.4f})")
            # Top 5
            top5 = sub.nlargest(5, 'abs_pearson')
            for _, row in top5.iterrows():
                print(f"           dim {int(row['dim']):3d}  r = {row['pearson_with_log_libsize']:+.4f}")

        df_latent.to_csv(os.path.join(out_dir, 'latent_libsize_correlation.csv'), index=False)
        fig_latent_libsize(df_latent, out_dir)
    else:
        print("  Skipped (latent_cell not available)")

    # ===========================================================
    # Summary
    # ===========================================================
    print("\n=== Summary ===")
    print(f"  Library size preservation:  r = {global_stats['global_pearson']:.3f}")
    mask_dense = (df_decomp['frac_nonzero'] >= 0.7)
    mask_sparse = (df_decomp['frac_nonzero'] < 0.1)
    print(f"  R² libsize (dense genes):  raw {df_decomp.loc[mask_dense, 'r2_raw_libsize'].median():.3f}  "
          f"dec {df_decomp.loc[mask_dense, 'r2_dec_libsize'].median():.3f}")
    print(f"  R² libsize (sparse genes): raw {df_decomp.loc[mask_sparse, 'r2_raw_libsize'].median():.3f}  "
          f"dec {df_decomp.loc[mask_sparse, 'r2_dec_libsize'].median():.3f}")
    mask_dense_p = (df_prop['frac_nonzero'] >= 0.7)
    print(f"  std_ratio dense (log1p):  {df_prop.loc[mask_dense_p, 'std_ratio_log1p'].median():.3f}")
    print(f"  std_ratio dense (prop):   {df_prop.loc[mask_dense_p, 'std_ratio_prop'].median():.3f}")

    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
