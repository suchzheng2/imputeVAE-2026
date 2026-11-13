"""
Figures for within-CT nonzero analysis (SUC-6).

Figures:
  1. Per-gene scatter: raw_std vs dec_std, colored by sparsity bucket
  2. Distribution examples: 3 genes at different sparsity in L2/3 IT
  3. Heatmap: CT × sparsity bucket, colored by std_ratio
  4a. Nonzero value scatter: raw vs decoded, colored by cell type (subsampled)
  4b. Nonzero value scatter: raw vs decoded, colored by gene (subsampled)

Usage:
    python -m analysis.within_ct_nonzero_figures --model pct5v2_cond
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

warnings.filterwarnings("ignore", category=FutureWarning)

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_DIR)


def fig1_std_scatter(df, out_dir):
    """Per-gene scatter: raw_std vs dec_std, colored by sparsity."""
    sub = df[df['source'] == 'cell_recon'].copy()

    bins = [
        (0, 0.1, '<10%', '#d62728'),
        (0.1, 0.3, '10-30%', '#ff7f0e'),
        (0.3, 0.7, '30-70%', '#2ca02c'),
        (0.7, 1.01, '>70%', '#1f77b4'),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    for lo, hi, label, color in bins:
        mask = (sub['frac_nonzero'] >= lo) & (sub['frac_nonzero'] < hi)
        s = sub[mask]
        # Subsample for visibility
        if len(s) > 5000:
            s = s.sample(5000, random_state=42)
        ax.scatter(s['raw_std'], s['dec_std'], s=2, alpha=0.3, c=color, label=label)

    # Diagonal
    mx = max(sub['raw_std'].quantile(0.99), sub['dec_std'].quantile(0.99))
    ax.plot([0, mx], [0, mx], 'k--', lw=1, alpha=0.5, label='y = x')

    ax.set_xlabel('Raw std (nonzero cells)', fontsize=13)
    ax.set_ylabel('Decoded std (nonzero cells)', fontsize=13)
    ax.set_title('Per-gene within-CT: raw vs decoded std\n(each dot = one gene in one cell type)', fontsize=13)
    ax.legend(title='Gene sparsity\n(frac nonzero)', fontsize=10, markerscale=5)
    ax.set_xlim(0, mx)
    ax.set_ylim(0, mx)
    ax.set_aspect('equal')

    fig.tight_layout()
    path = os.path.join(out_dir, 'fig1_std_scatter.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig1b_std_scatter_by_ct(df, out_dir):
    """Per-gene scatter: raw_std vs dec_std, colored by cell type."""
    sub = df[df['source'] == 'cell_recon'].copy()

    cts = sorted(sub['ct'].unique())
    cmap = plt.cm.get_cmap('tab20', len(cts))
    ct_colors = {ct: cmap(i) for i, ct in enumerate(cts)}

    fig, ax = plt.subplots(1, 1, figsize=(9, 8))

    for ct in cts:
        s = sub[sub['ct'] == ct]
        if len(s) > 3000:
            s = s.sample(3000, random_state=42)
        ax.scatter(s['raw_std'], s['dec_std'], s=2, alpha=0.3,
                   c=[ct_colors[ct]], label=ct)

    # Diagonal
    mx = max(sub['raw_std'].quantile(0.99), sub['dec_std'].quantile(0.99))
    ax.plot([0, mx], [0, mx], 'k--', lw=1, alpha=0.5)

    ax.set_xlabel('Raw std (nonzero cells)', fontsize=13)
    ax.set_ylabel('Decoded std (nonzero cells)', fontsize=13)
    ax.set_title('Per-gene within-CT: raw vs decoded std\n(colored by cell type)', fontsize=13)
    ax.legend(fontsize=7, markerscale=5, ncol=2, loc='upper left',
              title='Cell type', title_fontsize=9)
    ax.set_xlim(0, mx)
    ax.set_ylim(0, mx)
    ax.set_aspect('equal')

    fig.tight_layout()
    path = os.path.join(out_dir, 'fig1b_std_scatter_by_ct.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_dist_row_pair(axes_nz, axes_z, raw_ct, dec_ct, targets, gene_names,
                        frac_nz, row_label, legend=False, xlabel=False,
                        extra_x_arrays=None):
    """Plot nonzero/zero distribution rows for one source (cell recon or imputed).

    Parameters
    ----------
    axes_nz : array of Axes — nonzero row (one per target gene)
    axes_z : array of Axes — zero row (one per target gene)
    raw_ct : ndarray [N, G] — raw cell expression for this CT
    dec_ct : ndarray [N, G] — decoded/imputed expression for this CT
    targets : list of (label, gene_idx)
    gene_names : array of str
    frac_nz : ndarray [G] — fraction nonzero per gene (from raw_ct)
    row_label : str — e.g. 'Cell recon' or 'Imputed (nuc→cell)'
    legend : bool — show legend on first panel
    xlabel : bool — show x-axis label on zero row
    extra_x_arrays : list of ndarray or None — additional arrays to include in
        x-axis range computation (e.g. imputed array when sharing scale with recon)
    """
    for col, (label, gidx) in enumerate(targets):
        raw_vals = raw_ct[:, gidx]
        dec_vals = dec_ct[:, gidx]
        gname = gene_names[gidx] if gene_names is not None else f'Gene {gidx}'
        frac = frac_nz[gidx]

        nz_mask = raw_vals > 0
        raw_nz = raw_vals[nz_mask]
        dec_nz = dec_vals[nz_mask]
        raw_z = raw_vals[~nz_mask]
        dec_z = dec_vals[~nz_mask]

        # Shared x-axis range per column
        all_vals = [raw_vals, dec_vals]
        if extra_x_arrays is not None:
            all_vals.append(extra_x_arrays[col])
        x_lo = min(v.min() for v in all_vals) - 0.05
        x_hi = max(v.max() for v in all_vals) + 0.05
        bins_h = np.linspace(x_lo, x_hi, 50)

        # Nonzero row
        ax_nz = axes_nz[col]
        ax_nz.hist(raw_nz, bins=bins_h, alpha=0.5, density=True, label='Raw cell', color='#1f77b4')
        ax_nz.hist(dec_nz, bins=bins_h, alpha=0.5, density=True, label=row_label, color='#d62728')
        ax_nz.set_title(f'{label}\n{gname} ({frac:.0%} nonzero)', fontsize=11)
        ax_nz.set_xlim(x_lo, x_hi)
        n_nz = nz_mask.sum()
        n_total = len(raw_vals)
        # Summary stats for nonzero entries
        raw_nz_mean = raw_nz.mean() if len(raw_nz) > 0 else 0
        raw_nz_std = raw_nz.std() if len(raw_nz) > 0 else 0
        dec_nz_mean = dec_nz.mean() if len(dec_nz) > 0 else 0
        dec_nz_std = dec_nz.std() if len(dec_nz) > 0 else 0
        std_ratio = dec_nz_std / raw_nz_std if raw_nz_std > 0 else 0
        stat_text = (f'n={n_nz} ({n_nz/n_total:.0%})\n'
                     f'raw:  μ={raw_nz_mean:.2f}, σ={raw_nz_std:.2f}\n'
                     f'dec:  μ={dec_nz_mean:.2f}, σ={dec_nz_std:.2f}\n'
                     f'σ ratio: {std_ratio:.2f}')
        ax_nz.text(0.97, 0.95, stat_text,
                   transform=ax_nz.transAxes, ha='right', va='top', fontsize=9,
                   fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

        # Zero row
        ax_z = axes_z[col]
        ax_z.hist(raw_z, bins=bins_h, alpha=0.5, density=True, label='Raw cell', color='#1f77b4')
        ax_z.hist(dec_z, bins=bins_h, alpha=0.5, density=True, label=row_label, color='#d62728')
        ax_z.set_xlim(x_lo, x_hi)
        n_z = (~nz_mask).sum()
        ax_z.text(0.97, 0.95, f'n={n_z} ({n_z/n_total:.0%})',
                  transform=ax_z.transAxes, ha='right', va='top', fontsize=9,
                  bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
        if xlabel:
            ax_z.set_xlabel('Expression (log1p)', fontsize=11)

    if legend:
        axes_nz[0].legend(fontsize=10, loc='upper left')


def fig2_distribution_examples(data, sources, out_dir):
    """Distribution examples: 4 genes at different sparsity in one CT.

    Layout: 4 rows x 4 columns
      Row 0: cell recon — nonzero entries (raw cell vs decoded cell)
      Row 1: cell recon — zero entries
      Row 2: imputed nuc→cell — nonzero entries (raw cell vs cross-decoded)
      Row 3: imputed nuc→cell — zero entries
    Same genes selected by cell sparsity. Imputed row uses nuclei from same CT.
    """
    ct = 'L2/3 IT'

    # Cell data for this CT
    ct_mask_cell = data.ct_cell == ct
    raw_cell_ct = data.X_cell[ct_mask_cell]

    # Decoded cell (same-modality reconstruction)
    dec_cell_ct = sources['decoded_cell'][ct_mask_cell]

    # Imputed nuclei→cell for this CT
    ct_mask_nuc = data.ct_nuc == ct
    imp_ct = sources['cross_decoded'][ct_mask_nuc]

    gene_names = data.gene_names

    # Compute sparsity per gene within this CT (from raw cell)
    frac_nz = (raw_cell_ct > 0).mean(axis=0)

    # Pick representative genes: one per sparsity bin
    targets = [
        ('Dense (>70% nonzero)', np.argmin(np.abs(frac_nz - 0.85))),
        ('Medium (30-70%)', np.argmin(np.abs(frac_nz - 0.50))),
        ('Low-medium (10-30%)', np.argmin(np.abs(frac_nz - 0.20))),
        ('Sparse (<10%)', np.argmin(np.abs(frac_nz - 0.05))),
    ]

    n_cols = len(targets)
    fig, axes = plt.subplots(4, n_cols, figsize=(5.5 * n_cols, 16))

    # For imputed row: use raw_cell as reference but imputed cells come from
    # nuclei (different cells, same CT). Use a random subsample of raw_cell
    # matched to imputed size for fair visual comparison.
    n_imp = imp_ct.shape[0]
    n_cell = raw_cell_ct.shape[0]
    rng = np.random.RandomState(42)
    if n_cell > n_imp:
        raw_cell_sub = raw_cell_ct[rng.choice(n_cell, n_imp, replace=False)]
    else:
        raw_cell_sub = raw_cell_ct

    # Compute extra arrays for x-axis range sharing within each column
    # (so cell-recon and imputed rows use the same x range per gene)
    extra_for_recon = [imp_ct[:, gidx] for _, gidx in targets]
    extra_for_imp = [dec_cell_ct[:, gidx] for _, gidx in targets]

    # Rows 0-1: cell reconstruction
    _plot_dist_row_pair(axes[0], axes[1], raw_cell_ct, dec_cell_ct,
                        targets, gene_names, frac_nz,
                        row_label='Cell recon', legend=True,
                        extra_x_arrays=extra_for_recon)
    axes[0, 0].set_ylabel('Density (nonzero)', fontsize=11)
    axes[1, 0].set_ylabel('Density (zero)', fontsize=11)

    # Rows 2-3: imputed nuclei→cell
    # Sparsity fractions from raw_cell_sub (same CT, subsampled)
    frac_nz_sub = (raw_cell_sub > 0).mean(axis=0)
    _plot_dist_row_pair(axes[2], axes[3], raw_cell_sub, imp_ct,
                        targets, gene_names, frac_nz_sub,
                        row_label='Imputed (nuc→cell)', legend=True,
                        xlabel=True, extra_x_arrays=extra_for_imp)
    axes[2, 0].set_ylabel('Density (nonzero)', fontsize=11)
    axes[3, 0].set_ylabel('Density (zero)', fontsize=11)

    # Clear column titles on rows 2-3 (gene names already shown in row 0)
    for col in range(n_cols):
        axes[2, col].set_title('')

    fig.suptitle(
        f'Raw vs Decoded/Imputed distributions — {ct}\n'
        f'Rows 1-2: cell reconstruction | Rows 3-4: imputed (nuc→cell)',
        fontsize=14, y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, 'fig2_distribution_examples.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def fig3_heatmap(df, out_dir):
    """Heatmap: CT × sparsity bucket, colored by std_ratio."""
    sub = df[df['source'] == 'cell_recon'].copy()

    bins = [(0, 0.1, '<10%'), (0.1, 0.3, '10-30%'), (0.3, 0.7, '30-70%'), (0.7, 1.01, '>70%')]
    cts = sorted(sub['ct'].unique())

    mat = np.full((len(cts), len(bins)), np.nan)
    for j, (lo, hi, _) in enumerate(bins):
        for i, ct in enumerate(cts):
            mask = (sub['ct'] == ct) & (sub['frac_nonzero'] >= lo) & (sub['frac_nonzero'] < hi)
            vals = sub.loc[mask, 'std_ratio']
            if len(vals) > 0:
                mat[i, j] = vals.median()

    fig, ax = plt.subplots(1, 1, figsize=(8, 10))
    im = ax.imshow(mat, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels([b[2] for b in bins], fontsize=11)
    ax.set_yticks(range(len(cts)))
    ax.set_yticklabels(cts, fontsize=10)
    ax.set_xlabel('Gene sparsity (frac nonzero)', fontsize=12)
    ax.set_ylabel('Cell type', fontsize=12)
    ax.set_title('Median std_ratio (decoded / raw)\nPer gene, within CT, nonzero only', fontsize=13)

    # Annotate cells
    for i in range(len(cts)):
        for j in range(len(bins)):
            v = mat[i, j]
            if not np.isnan(v):
                color = 'white' if v < 0.3 or v > 0.8 else 'black'
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=9, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, label='std_ratio')
    fig.tight_layout()
    path = os.path.join(out_dir, 'fig3_heatmap_ct_sparsity.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig4_nonzero_scatter(data, sources, out_dir, n_sample=200000):
    """Scatter of all nonzero values: raw vs decoded.

    4a: colored by cell type
    4b: colored by gene (top 20 most variable)
    """
    raw = data.X_cell
    dec = sources['decoded_cell']
    ct_array = data.ct_cell
    gene_names = data.gene_names

    # Get nonzero positions
    nz_mask = raw > 0
    nz_rows, nz_cols = np.where(nz_mask)

    # Subsample
    rng = np.random.RandomState(42)
    n_total = len(nz_rows)
    if n_total > n_sample:
        idx = rng.choice(n_total, n_sample, replace=False)
        nz_rows = nz_rows[idx]
        nz_cols = nz_cols[idx]

    raw_vals = raw[nz_rows, nz_cols]
    dec_vals = dec[nz_rows, nz_cols]
    ct_vals = ct_array[nz_rows]
    gene_vals = nz_cols

    # --- 4a: colored by cell type ---
    fig, ax = plt.subplots(1, 1, figsize=(9, 8))

    cts = sorted(np.unique(ct_vals))
    cmap = plt.cm.get_cmap('tab20', len(cts))
    ct_to_idx = {ct: i for i, ct in enumerate(cts)}

    for ct in cts:
        mask = ct_vals == ct
        ax.scatter(raw_vals[mask], dec_vals[mask], s=1, alpha=0.15,
                   c=[cmap(ct_to_idx[ct])], label=ct)

    mx = max(np.percentile(raw_vals, 99.5), np.percentile(dec_vals, 99.5))
    ax.plot([0, mx], [0, mx], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('Raw expression (log1p, nonzero)', fontsize=13)
    ax.set_ylabel('Decoded expression (log1p)', fontsize=13)
    ax.set_title(f'Nonzero entries: raw vs decoded\n(n={len(raw_vals):,} subsampled)', fontsize=13)
    ax.legend(fontsize=7, markerscale=8, ncol=2, loc='upper left')
    ax.set_xlim(0, mx)
    ax.set_ylim(-0.2, mx)

    fig.tight_layout()
    path = os.path.join(out_dir, 'fig4a_nonzero_scatter_by_ct.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    # --- 4b: colored by gene (top 20 most variable genes) ---
    # Find top 20 genes by within-CT nonzero variance
    gene_var = np.zeros(raw.shape[1])
    for g in range(raw.shape[1]):
        nz = raw[:, g] > 0
        if nz.sum() > 10:
            gene_var[g] = np.var(raw[nz, g])
    top20 = np.argsort(gene_var)[-20:]

    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    cmap_g = plt.cm.get_cmap('tab20', 20)

    # Plot background (other genes) in gray
    other_mask = ~np.isin(gene_vals, top20)
    ax.scatter(raw_vals[other_mask], dec_vals[other_mask], s=1, alpha=0.05, c='gray', label='other')

    for rank, g in enumerate(top20):
        mask = gene_vals == g
        gname = gene_names[g] if gene_names is not None else f'Gene {g}'
        ax.scatter(raw_vals[mask], dec_vals[mask], s=3, alpha=0.4,
                   c=[cmap_g(rank)], label=gname)

    ax.plot([0, mx], [0, mx], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('Raw expression (log1p, nonzero)', fontsize=13)
    ax.set_ylabel('Decoded expression (log1p)', fontsize=13)
    ax.set_title(f'Nonzero entries: raw vs decoded\nTop 20 most variable genes highlighted', fontsize=13)
    ax.legend(fontsize=6, markerscale=5, ncol=2, loc='upper left')
    ax.set_xlim(0, mx)
    ax.set_ylim(-0.2, mx)

    fig.tight_layout()
    path = os.path.join(out_dir, 'fig4b_nonzero_scatter_by_gene.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def fig5_all_ct_distributions(data, sources, out_dir, source_key='decoded_cell',
                               source_label='Cell recon'):
    """Small-multiples grid: rows=CTs, columns=sparsity bins.

    Each panel shows overlaid histograms (raw cell vs decoded/imputed) for
    the nonzero entries of a representative gene picked near the bin center.

    Parameters
    ----------
    data : eval data object with X_cell, ct_cell, gene_names
    sources : dict with decoded arrays
    out_dir : str
    source_key : 'decoded_cell' or 'cross_decoded'
    source_label : legend label for decoded curve
    """
    bins = [
        (0.7, 1.01, '>70%', 0.85),
        (0.3, 0.7, '30-70%', 0.50),
        (0.1, 0.3, '10-30%', 0.20),
        (0, 0.1, '<10%', 0.05),
    ]
    n_bins = len(bins)

    gene_names = data.gene_names

    # Determine whether source is cross-decoded (nuclei→cell) or same-modality
    is_cross = (source_key == 'cross_decoded')
    if is_cross:
        ct_array_dec = data.ct_nuc
        dec_array = sources[source_key]
    else:
        ct_array_dec = data.ct_cell
        dec_array = sources[source_key]

    cts = sorted(np.unique(data.ct_cell))
    n_cts = len(cts)

    fig, axes = plt.subplots(n_cts, n_bins, figsize=(4 * n_bins, 2.2 * n_cts))

    for row_idx, ct in enumerate(cts):
        # Raw cell data for this CT
        ct_mask_cell = data.ct_cell == ct
        raw_ct = data.X_cell[ct_mask_cell]

        # Decoded data for this CT
        ct_mask_dec = ct_array_dec == ct
        dec_ct = dec_array[ct_mask_dec]

        if dec_ct.shape[0] == 0:
            # No decoded data for this CT (e.g. CT not present in nuclei)
            for col_idx in range(n_bins):
                ax = axes[row_idx, col_idx]
                ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                        ha='center', va='center', fontsize=8, color='gray')
                ax.set_xticks([]); ax.set_yticks([])
                if row_idx == 0:
                    ax.set_title(bins[col_idx][2], fontsize=10, fontweight='bold')
                if col_idx == 0:
                    ax.set_ylabel(ct, fontsize=9)
            continue

        # Per-gene frac nonzero from raw cell (always use full raw for gene selection)
        frac_nz = (raw_ct > 0).mean(axis=0)

        for col_idx, (lo, hi, bin_label, target_frac) in enumerate(bins):
            ax = axes[row_idx, col_idx]

            # Find representative gene: closest to target_frac within bin
            in_bin = (frac_nz >= lo) & (frac_nz < hi)
            if not in_bin.any():
                ax.text(0.5, 0.5, 'no genes', transform=ax.transAxes,
                        ha='center', va='center', fontsize=8, color='gray')
                ax.set_xticks([])
                ax.set_yticks([])
                if row_idx == 0:
                    ax.set_title(bin_label, fontsize=10, fontweight='bold')
                if col_idx == 0:
                    ax.set_ylabel(ct, fontsize=9)
                continue

            bin_indices = np.where(in_bin)[0]
            best_idx = bin_indices[np.argmin(np.abs(frac_nz[bin_indices] - target_frac))]

            raw_vals = raw_ct[:, best_idx]
            dec_vals = dec_ct[:, best_idx]
            gname = gene_names[best_idx] if gene_names is not None else f'Gene {best_idx}'
            frac = frac_nz[best_idx]

            # For cross-decoded, data is unpaired (different cells).
            # Show all raw nonzero vs all decoded values (no per-cell mask sharing).
            if is_cross:
                raw_nz = raw_vals[raw_vals > 0]
                # For imputed, show all decoded values (they're from nuclei cells)
                dec_nz = dec_vals[dec_vals > 0] if (len(dec_vals) > 0 and dec_vals.max() > 0) else dec_vals
            else:
                nz_mask = raw_vals > 0
                raw_nz = raw_vals[nz_mask]
                dec_nz = dec_vals[nz_mask]

            if len(raw_nz) < 3:
                ax.text(0.5, 0.5, f'{gname}\nn<3', transform=ax.transAxes,
                        ha='center', va='center', fontsize=7, color='gray')
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                x_lo = min(raw_vals.min(), dec_vals.min()) - 0.05
                x_hi = max(raw_vals.max(), dec_vals.max()) + 0.05
                hist_bins = np.linspace(x_lo, x_hi, 35)

                ax.hist(raw_nz, bins=hist_bins, alpha=0.5, density=True,
                        color='#1f77b4', label='Raw cell')
                ax.hist(dec_nz, bins=hist_bins, alpha=0.5, density=True,
                        color='#d62728', label=source_label)
                ax.set_xlim(x_lo, x_hi)

                # Stats annotation
                raw_mean = raw_nz.mean() if len(raw_nz) > 0 else 0
                dec_mean = dec_nz.mean() if len(dec_nz) > 0 else 0
                raw_std = raw_nz.std() if len(raw_nz) > 1 else 0
                dec_std = dec_nz.std() if len(dec_nz) > 1 else 0
                mr = dec_mean / raw_mean if raw_mean > 0 else 0
                sr = dec_std / raw_std if raw_std > 0 else 0
                n_raw = len(raw_nz)
                n_dec = len(dec_nz)
                n_label = f'n={n_raw}' if n_raw == n_dec else f'n={n_raw}/{n_dec}'
                ax.text(0.97, 0.95,
                        f'{gname} ({frac:.0%})\n'
                        f'{n_label}\n'
                        f'\u03bcr={mr:.2f} \u03c3r={sr:.2f}',
                        transform=ax.transAxes, ha='right', va='top',
                        fontsize=7, fontfamily='monospace',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

            ax.tick_params(labelsize=6)
            ax.set_yticks([])

            # Column headers on first row
            if row_idx == 0:
                ax.set_title(bin_label, fontsize=10, fontweight='bold')
            # Row labels on first column
            if col_idx == 0:
                ax.set_ylabel(ct, fontsize=9)

            # Legend on top-left panel only
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=6, loc='upper left')

    suffix = 'imputed' if is_cross else 'recon'
    fig.suptitle(
        f'Raw vs {source_label} distributions — all cell types\n'
        f'Columns: sparsity bins | Each panel: representative gene (nonzero entries)',
        fontsize=14, y=1.005)
    fig.tight_layout()
    path = os.path.join(out_dir, f'fig5_all_ct_distributions_{suffix}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="pct5v2_cond")
    parser.add_argument("--out", type=str,
                        default="notes/experiments/SUC-6_variance_retention/figures/")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to within_ct_nonzero_gene_stats.csv (auto-detected if omitted)")
    args = parser.parse_args()

    out_dir = os.path.join(_PROJECT_DIR, args.out)
    os.makedirs(out_dir, exist_ok=True)

    # Load CSV for figures 1 and 3
    csv_path = args.csv or os.path.join(
        _PROJECT_DIR, "notes/experiments/zero_nonzero_analysis",
        args.model, "within_ct_nonzero_gene_stats.csv"
    )
    print(f"Loading stats CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  {len(df)} rows, sources: {df['source'].unique()}")

    # Figure 1: std scatter (by sparsity)
    print("\nFigure 1: std scatter (by sparsity)...")
    fig1_std_scatter(df, out_dir)

    # Figure 1b: std scatter (by cell type)
    print("Figure 1b: std scatter (by cell type)...")
    fig1b_std_scatter_by_ct(df, out_dir)

    # Figure 3: heatmap
    print("Figure 3: heatmap...")
    fig3_heatmap(df, out_dir)

    # For figures 2, 4, 5 we need actual data + decoded arrays
    from analysis.eval.registry import get_model
    from analysis.eval.data import load_eval_data
    from analysis.eval.inference import load_model, run_5source_inference

    model_config = get_model(args.model)
    data = load_eval_data(model_config['gene_list'], max_per_type=5000)
    model, device = load_model(model_config)

    print(f"\nRunning inference for figures 2, 4, 5...")
    sources = run_5source_inference(model, model_config,
                                     data.X_cell, data.X_nuc, device)

    import gc
    del model
    gc.collect()

    # Figure 2: distribution examples
    print("Figure 2: distribution examples...")
    fig2_distribution_examples(data, sources, out_dir)

    # Figure 4: nonzero value scatter
    print("Figure 4a,b: nonzero scatter...")
    fig4_nonzero_scatter(data, sources, out_dir)

    # Figure 5: all CT distributions — cell reconstruction
    print("Figure 5: all CT distributions (cell recon)...")
    fig5_all_ct_distributions(data, sources, out_dir,
                              source_key='decoded_cell',
                              source_label='Cell recon')

    # Figure 5: all CT distributions — imputed (nuc→cell)
    print("Figure 5: all CT distributions (imputed)...")
    fig5_all_ct_distributions(data, sources, out_dir,
                              source_key='cross_decoded',
                              source_label='Imputed (nuc\u2192cell)')

    print("\nAll figures saved to:", out_dir)


if __name__ == "__main__":
    main()
