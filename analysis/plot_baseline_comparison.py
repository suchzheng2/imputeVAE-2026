"""
Generate baseline comparison figure panels for ISMB poster.

Panels:
  A — Grouped bar: CT-level metrics (gene corr, R², shift)
  B — Bar: Per-cell variance ratio
  C — Scatter: Delta (ground truth vs predicted) for cVAE and scVI
  D — GO enrichment dot plot
  E — Compartment scores (ribosomal/cytoplasmic/nuclear lncRNA)

Usage:
    python analysis/plot_baseline_comparison.py [--output_dir notes/presentation/baseline_figures]

Requires: run/ctmean_baseline/eval/eval_ctmean.json, run/scvi_baseline/eval/eval_scvi.json
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec

# ---- Style ----
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})

COLORS = {
    'cVAE': '#2166AC',      # blue
    'CT-mean': '#B2182B',   # red
    'scVI': '#F4A582',      # light salmon
    'Raw nuclei': '#999999', # gray
}


def load_results():
    """Load eval JSONs and hardcoded cVAE values."""
    with open('run/ctmean_baseline/eval/eval_ctmean.json') as f:
        ctmean = json.load(f)
    with open('run/scvi_baseline/eval/eval_scvi.json') as f:
        scvi = json.load(f)

    # cVAE values from inline eval of best model (run_2026-05-11-19-52-25_134850)
    cvae = {
        'gene_corr': {'median': 0.986},
        'r_squared': {'median': 0.792},
        'shift': {'mean_shift': 0.200},
        'variance_ratio': {'median_ratio': 0.45},
        'delta_gt': {
            'overall_corr': 0.918,
            'overall_r2': 0.792,
            'median_ct_corr': 0.968,
            'per_ct_corr': {},  # populated below if available
        },
        'compartment': {
            'ribosomal': {'frac_correct_sign': 1.0, 'mean_delta': 1.71},
            'cytoplasmic': {'frac_correct_sign': 1.0, 'mean_delta': 2.89},
            'nuclear_lncrna': {'frac_correct_sign': 1.0, 'mean_delta': -0.54},
        },
    }

    return cvae, ctmean, scvi


def panel_a_ct_metrics(ax, cvae, ctmean, scvi):
    """Grouped bar chart of CT-level metrics."""
    metrics = ['Gene\ncorrelation', 'R²', 'Shift\nscore']
    cvae_vals = [
        cvae['gene_corr']['median'],
        cvae['r_squared']['median'],
        cvae['shift']['mean_shift'],
    ]
    ctmean_vals = [
        min(ctmean['gene_corr']['median'], 1.0),
        min(ctmean['r_squared']['median'], 1.0),
        ctmean['shift']['mean_shift'],
    ]
    scvi_vals = [
        scvi['gene_corr']['median'],
        scvi['r_squared']['median'],
        scvi['shift']['mean_shift'],
    ]
    raw_nuc_vals = [
        scvi['gene_corr_raw_nuc']['median'],
        None,
        None,
    ]

    x = np.arange(len(metrics))
    w = 0.2

    bars_cvae = ax.bar(x - 1.5*w, cvae_vals, w, color=COLORS['cVAE'], label='cVAE (ours)', zorder=3)
    bars_ctm = ax.bar(x - 0.5*w, ctmean_vals, w, color=COLORS['CT-mean'], label='CT-mean', zorder=3)
    bars_scvi = ax.bar(x + 0.5*w, scvi_vals, w, color=COLORS['scVI'], label='scVI', zorder=3)

    # Raw nuclei only for gene corr
    raw_bar_vals = [raw_nuc_vals[0], 0, 0]
    bars_raw = ax.bar(x[0] + 1.5*w, raw_nuc_vals[0], w, color=COLORS['Raw nuclei'], label='Raw nuclei', zorder=3)

    # Value labels on bars
    for bars, vals in [(bars_cvae, cvae_vals), (bars_ctm, ctmean_vals), (bars_scvi, scvi_vals)]:
        for bar, val in zip(bars, vals):
            if val is not None and val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
            elif val is not None and val < 0:
                ax.text(bar.get_x() + bar.get_width()/2, 0.02,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold',
                        color='red')

    ax.text(bars_raw[0].get_x() + bars_raw[0].get_width()/2, bars_raw[0].get_height() + 0.02,
            f'{raw_nuc_vals[0]:.2f}', ha='center', va='bottom', fontsize=8)

    # scVI R² is negative — draw hatched bar below zero
    ax.bar(x[1] + 0.5*w, scvi_vals[1], w, color=COLORS['scVI'], zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Score')
    ax.set_title('A. Cell-type-level metrics')
    ax.axhline(0, color='black', linewidth=0.5, zorder=1)
    ax.set_ylim(-0.8, 1.15)
    ax.legend(loc='lower left', framealpha=0.9, fontsize=9)
    ax.grid(axis='y', alpha=0.3, zorder=0)


def panel_b_variance(ax, cvae, ctmean, scvi):
    """Bar chart of variance ratio."""
    methods = ['cVAE\n(ours)', 'CT-mean', 'scVI']
    vals = [
        cvae['variance_ratio']['median_ratio'],
        ctmean['variance_ratio']['median_ratio'],
        scvi['variance_ratio']['median_ratio'],
    ]
    colors = [COLORS['cVAE'], COLORS['CT-mean'], COLORS['scVI']]

    bars = ax.bar(methods, vals, color=colors, width=0.6, zorder=3, edgecolor='white', linewidth=1)

    for bar, val in zip(bars, vals):
        label = f'{val:.3f}' if val < 0.01 else f'{val:.2f}'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                label, ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.axhline(1.0, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Perfect (1.0)')
    ax.set_ylabel('Variance ratio')
    ax.set_title('B. Within-CT variance\npreservation')
    ax.set_ylim(0, 0.65)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3, zorder=0)


def panel_c_delta_scatter(axes, scvi):
    """Delta scatter: ground truth vs predicted, cVAE and scVI side by side.

    Uses per-CT correlation values. For full scatter we'd need to recompute —
    here we show per-CT delta correlation as a bar chart instead.
    """
    ax_cvae, ax_scvi = axes

    # cVAE per-CT delta correlations (from validation analysis)
    cvae_per_ct = {
        'L6 CT': 0.991, 'L5 IT': 0.986, 'L2/3 IT': 0.987, 'L6 IT': 0.978,
        'Sst': 0.981, 'L6b': 0.968, 'Lamp5': 0.968, 'Pvalb': 0.972,
        'L5 ET': 0.956, 'Macrophage': 0.883, 'L5/6 NP': 0.974, 'Vip': 0.966,
        'Oligo': 0.909, 'Sncg': 0.917, 'Endo': 0.659, 'OPC': 0.858, 'Astro': 0.656,
    }
    scvi_per_ct = scvi['delta_gt']['per_ct_corr']

    # Get common CTs sorted by cVAE performance
    common_cts = sorted(set(cvae_per_ct.keys()) & set(scvi_per_ct.keys()),
                        key=lambda ct: cvae_per_ct.get(ct, 0), reverse=True)

    cvae_vals = [cvae_per_ct[ct] for ct in common_cts]
    scvi_vals = [scvi_per_ct[ct] for ct in common_cts]

    # Short labels
    short = [ct.replace('Macrophage', 'Macro').replace('L2/3 IT', 'L2/3')
             .replace('L5/6 NP', 'NP') for ct in common_cts]

    y = np.arange(len(common_cts))

    # cVAE panel
    ax_cvae.barh(y, cvae_vals, color=COLORS['cVAE'], height=0.7, zorder=3)
    ax_cvae.set_yticks(y)
    ax_cvae.set_yticklabels(short, fontsize=9)
    ax_cvae.set_xlabel('Delta correlation (r)')
    ax_cvae.set_title('cVAE (ours)', fontsize=11, fontweight='bold', color=COLORS['cVAE'])
    ax_cvae.set_xlim(0.4, 1.05)
    ax_cvae.axvline(0.9, color='gray', linestyle=':', alpha=0.5)
    ax_cvae.invert_yaxis()
    ax_cvae.grid(axis='x', alpha=0.3, zorder=0)
    # Mean annotation
    mean_cvae = np.mean(cvae_vals)
    ax_cvae.axvline(mean_cvae, color=COLORS['cVAE'], linestyle='--', linewidth=1.5, alpha=0.7)
    ax_cvae.text(mean_cvae + 0.01, len(common_cts) - 0.5, f'mean={mean_cvae:.3f}',
                 fontsize=9, color=COLORS['cVAE'], fontweight='bold')

    # scVI panel
    ax_scvi.barh(y, scvi_vals, color=COLORS['scVI'], height=0.7, zorder=3)
    ax_scvi.set_yticks(y)
    ax_scvi.set_yticklabels([])
    ax_scvi.set_xlabel('Delta correlation (r)')
    ax_scvi.set_title('scVI', fontsize=11, fontweight='bold', color='#D6604D')
    ax_scvi.set_xlim(0.4, 1.05)
    ax_scvi.axvline(0.9, color='gray', linestyle=':', alpha=0.5)
    ax_scvi.invert_yaxis()
    ax_scvi.grid(axis='x', alpha=0.3, zorder=0)
    mean_scvi = np.mean(scvi_vals)
    ax_scvi.axvline(mean_scvi, color='#D6604D', linestyle='--', linewidth=1.5, alpha=0.7)
    ax_scvi.text(mean_scvi + 0.01, len(common_cts) - 0.5, f'mean={mean_scvi:.3f}',
                 fontsize=9, color='#D6604D', fontweight='bold')


def panel_d_go(ax):
    """GO enrichment dot plot for top terms."""
    # Curated top terms from go_enrichment analysis
    terms = [
        ('Structural constituent\nof ribosome', 'GO:MF', 3.18e-26, 22, 69),
        ('Ribosome', 'KEGG', 5.02e-25, 21, 68),
        ('Ribosomal subunit', 'GO:CC', 5.49e-24, 21, 72),
        ('Synapse', 'GO:CC', 3.4e-14, 43, 1200),
        ('Cytoplasmic\ntranslation', 'GO:BP', 1.9e-15, 20, 145),
        ('Cytoplasm', 'GO:CC', 7.5e-08, 79, 5548),
        ('Cytosol', 'GO:CC', 6.0e-09, 49, 2215),
        ('Postsynapse', 'GO:CC', 1.2e-10, 28, 646),
        ('Synaptic vesicle\ncycle', 'GO:BP', 2.1e-06, 13, 172),
    ]

    names = [t[0] for t in terms]
    sources = [t[1] for t in terms]
    pvals = [-np.log10(t[2]) for t in terms]
    hits = [t[3] for t in terms]
    term_sizes = [t[4] for t in terms]
    gene_ratio = [t[3]/t[4] for t in terms]

    # Sort by p-value
    order = np.argsort(pvals)[::-1]
    names = [names[i] for i in order]
    pvals = [pvals[i] for i in order]
    hits = [hits[i] for i in order]
    gene_ratio = [gene_ratio[i] for i in order]

    y = np.arange(len(names))

    # Color by -log10(p), size by gene ratio
    scatter = ax.scatter(
        [gene_ratio[i] for i in range(len(names))],
        y,
        s=[hits[i] * 12 for i in range(len(names))],
        c=pvals,
        cmap='YlOrRd',
        edgecolors='black',
        linewidths=0.5,
        zorder=3,
        vmin=5, vmax=26,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Gene ratio (hits / term size)')
    ax.set_title('D. GO enrichment\n(genes increased by imputation)')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3, zorder=0)

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('-log₁₀(p-value)', fontsize=9)

    # Size legend
    for size, label in [(10, '10'), (25, '25'), (45, '45')]:
        ax.scatter([], [], s=size*12, c='gray', edgecolors='black',
                   linewidths=0.5, label=f'{label} genes')
    ax.legend(title='Hit count', loc='lower right', fontsize=8, title_fontsize=9,
              framealpha=0.9)


def panel_e_compartment(ax, cvae, ctmean, scvi):
    """Compartment scores: correct sign fraction for each gene set."""
    gene_sets = ['Ribosomal\n(expect ↑)', 'Cytoplasmic\n(expect ↑)', 'Nuclear lncRNA\n(expect ↓)']
    keys = ['ribosomal', 'cytoplasmic', 'nuclear_lncrna']

    x = np.arange(len(gene_sets))
    w = 0.25

    cvae_vals = [cvae['compartment'][k]['frac_correct_sign'] for k in keys]
    ctmean_vals = [ctmean['compartment'][k]['frac_correct_sign'] for k in keys]
    scvi_vals = [scvi['compartment'][k]['frac_correct_sign'] for k in keys]

    ax.bar(x - w, cvae_vals, w, color=COLORS['cVAE'], label='cVAE (ours)', zorder=3)
    ax.bar(x, ctmean_vals, w, color=COLORS['CT-mean'], label='CT-mean', zorder=3)
    ax.bar(x + w, scvi_vals, w, color=COLORS['scVI'], label='scVI', zorder=3)

    # Value labels
    for xpos, vals, offset in [(x - w, cvae_vals, 0), (x, ctmean_vals, 0), (x + w, scvi_vals, 0)]:
        for xi, val in zip(xpos, vals):
            ax.text(xi, val + 0.02, f'{val:.0%}', ha='center', va='bottom', fontsize=8,
                    fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(gene_sets)
    ax.set_ylabel('Fraction correct sign')
    ax.set_title('E. Compartment gene validation')
    ax.set_ylim(0, 1.15)
    ax.axhline(1.0, color='green', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(axis='y', alpha=0.3, zorder=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='notes/presentation/baseline_figures')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cvae, ctmean, scvi = load_results()

    # ---- Composite figure ----
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35,
                           height_ratios=[1, 1])

    # Row 1: A (CT metrics), B (variance), E (compartment)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_e = fig.add_subplot(gs[0, 2])

    # Row 2: C (delta, two subpanels), D (GO enrichment)
    gs_c = gs[1, 0:2].subgridspec(1, 2, wspace=0.08)
    ax_c1 = fig.add_subplot(gs_c[0])
    ax_c2 = fig.add_subplot(gs_c[1])
    ax_d = fig.add_subplot(gs[1, 2])

    panel_a_ct_metrics(ax_a, cvae, ctmean, scvi)
    panel_b_variance(ax_b, cvae, ctmean, scvi)
    panel_e_compartment(ax_e, cvae, ctmean, scvi)

    ax_c1.set_ylabel('')
    fig.text(0.25, 0.48, 'C. Per-cell-type delta correlation (imputed − nucleus vs ground truth)',
             ha='center', fontsize=13, fontweight='bold')
    panel_c_delta_scatter([ax_c1, ax_c2], scvi)

    panel_d_go(ax_d)

    fig.suptitle('Baseline Comparison: cVAE vs CT-mean and scVI',
                 fontsize=16, fontweight='bold', y=0.98)

    # Save
    for ext in ['png', 'pdf']:
        path = os.path.join(args.output_dir, f'baseline_comparison.{ext}')
        fig.savefig(path)
        print(f'Saved: {path}')
    plt.close()

    # ---- Also save individual panels ----
    for panel_name, panel_func, panel_size, panel_args in [
        ('panel_A_ct_metrics', panel_a_ct_metrics, (7, 5), (cvae, ctmean, scvi)),
        ('panel_B_variance', panel_b_variance, (5, 5), (cvae, ctmean, scvi)),
        ('panel_E_compartment', panel_e_compartment, (7, 5), (cvae, ctmean, scvi)),
        ('panel_D_go', panel_d_go, (7, 6), ()),
    ]:
        fig, ax = plt.subplots(figsize=panel_size)
        panel_func(ax, *panel_args)
        fig.tight_layout()
        for ext in ['png', 'pdf']:
            path = os.path.join(args.output_dir, f'{panel_name}.{ext}')
            fig.savefig(path)
        plt.close()
        print(f'Saved: {panel_name}')

    # Panel C (two subplots)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
    panel_c_delta_scatter([ax1, ax2], scvi)
    fig.suptitle('Per-cell-type delta correlation', fontsize=13, fontweight='bold')
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        path = os.path.join(args.output_dir, f'panel_C_delta.{ext}')
        fig.savefig(path)
    plt.close()
    print('Saved: panel_C_delta')

    print('\nDone!')


if __name__ == '__main__':
    main()
