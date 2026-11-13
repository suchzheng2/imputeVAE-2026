#!/usr/bin/env python
"""Generate supplementary figures from saved SynGO tables (no data reload needed)."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RESULTS = 'validation/syngo/results'
TABLES = os.path.join(RESULTS, 'tables')
FIGS = os.path.join(RESULTS, 'figures')

colors = {'neuronal': '#4477AA', 'non-neuronal': '#228833'}


def fig_evidence_composition():
    """S1: SynGO evidence composition."""
    ev = pd.read_csv(os.path.join(TABLES, 'syngo_evidence_summary.csv'))
    ev = ev.sort_values('count', ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(range(len(ev)), ev['count'].values, color='steelblue')
    ax.set_yticks(range(len(ev)))
    ax.set_yticklabels(ev.evidence_type.values, fontsize=8)
    ax.set_xlabel('Count')
    ax.set_title('SynGO evidence composition (top 15)')
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(os.path.join(FIGS, 'syngo_evidence_composition.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIGS, 'syngo_evidence_composition.pdf'), bbox_inches='tight')
    plt.close(fig)


def fig_subgroup_overlap_upset():
    """S3: UpSet-style plot of SynGO subgroup overlaps."""
    mapping = pd.read_csv(os.path.join(TABLES, 'syngo_mouse_mapping.csv'))
    in_model = mapping[mapping.present_in_model & ~mapping.is_ribosomal].copy()

    # Parse memberships
    sets = {}
    key_terms = {
        'GO:0098793': 'Presynaptic',
        'GO:0098794': 'Postsynaptic',
        'GO:0014069': 'PSD',
        'GO:0048786': 'Active zone',
        'GO:0008021': 'Syn vesicle',
    }

    for _, row in in_model.iterrows():
        ms = str(row['syngo_memberships'])
        gene = row['mouse_symbol']
        for go_id, label in key_terms.items():
            if go_id in ms:
                if label not in sets:
                    sets[label] = set()
                sets[label].add(gene)

    # Simple bar chart of set sizes + pairwise overlaps
    labels = list(key_terms.values())
    sizes = [len(sets.get(l, set())) for l in labels]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Set sizes
    ax = axes[0]
    ax.barh(range(len(labels)), sizes, color='steelblue')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Gene count')
    ax.set_title('SynGO subgroup sizes (non-ribosomal)')
    ax.invert_yaxis()

    # Overlap matrix
    ax = axes[1]
    n = len(labels)
    overlap = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            s1 = sets.get(labels[i], set())
            s2 = sets.get(labels[j], set())
            if i == j:
                overlap[i, j] = len(s1)
            else:
                overlap[i, j] = len(s1 & s2)

    im = ax.imshow(overlap, cmap='Blues')
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{int(overlap[i,j])}', ha='center', va='center', fontsize=7)
    ax.set_title('Pairwise overlap')
    plt.colorbar(im, ax=ax, label='Genes')

    plt.tight_layout()
    fig.savefig(os.path.join(FIGS, 'syngo_subgroup_overlap_upset.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIGS, 'syngo_subgroup_overlap_upset.pdf'), bbox_inches='tight')
    plt.close(fig)


def fig_subgroup_detail():
    """S6/S7: Pre/post/both and distal subgroup enrichment."""
    sg = pd.read_csv(os.path.join(TABLES, 'syngo_subgroup_results.csv'))
    computed = sg[sg['status'] == 'computed'].copy()
    if len(computed) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Location groups
    loc_groups = computed[computed['type'].isin(['location'])]
    if len(loc_groups) > 0:
        ax = axes[0]
        y = np.arange(len(loc_groups))
        width = 0.35
        ax.barh(y - width/2, loc_groups['empirical_neuro_diff'].values, width,
               label='Empirical', color='steelblue', alpha=0.8)
        ax.barh(y + width/2, loc_groups['predicted_neuro_diff'].values, width,
               label='Predicted', color='coral', alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(loc_groups['subgroup'].values, fontsize=9)
        ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
        ax.set_xlabel('Neuronal - non-neuronal enrichment')
        ax.set_title('Location subgroups')
        ax.legend(fontsize=8)

    # Distal groups
    distal_groups = computed[computed['type'].isin(['distal'])]
    if len(distal_groups) > 0:
        ax = axes[1]
        y = np.arange(len(distal_groups))
        width = 0.35
        ax.barh(y - width/2, distal_groups['empirical_neuro_diff'].values, width,
               label='Empirical', color='steelblue', alpha=0.8)
        ax.barh(y + width/2, distal_groups['predicted_neuro_diff'].values, width,
               label='Predicted', color='coral', alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(distal_groups['subgroup'].values, fontsize=9)
        ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
        ax.set_xlabel('Neuronal - non-neuronal enrichment')
        ax.set_title('Distal structure subgroups')
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGS, 'syngo_subgroup_detail.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIGS, 'syngo_subgroup_detail.pdf'), bbox_inches='tight')
    plt.close(fig)


def fig_rare_ct_sensitivity():
    """S10: Rare cell-type sensitivity comparison."""
    sens = pd.read_csv(os.path.join(TABLES, 'syngo_sensitivity_results.csv'))

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = sens['subgroup'].values
    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width/2, sens['empirical_neuro_diff'].values, width,
           label='Empirical', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, sens['predicted_neuro_diff'].values, width,
           label='Predicted', color='coral', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([l.replace('synapse_', '').replace('_', ' ') for l in labels],
                       rotation=20, ha='right', fontsize=9)
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_ylabel('Neuronal - non-neuronal enrichment')
    ax.set_title('Sensitivity analyses: neuronal enrichment difference')
    ax.legend()

    # Add Pearson r annotations
    for i, (_, row) in enumerate(sens.iterrows()):
        ax.text(i, max(row['empirical_neuro_diff'], row['predicted_neuro_diff']) + 0.01,
                f"r={row['pearson_r']:.3f}", ha='center', fontsize=7)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGS, 'syngo_rare_ct_sensitivity.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIGS, 'syngo_rare_ct_sensitivity.pdf'), bbox_inches='tight')
    plt.close(fig)


def fig_baseline_comparison():
    """S12: Baseline comparison figure."""
    mc = pd.read_csv(os.path.join(TABLES, 'syngo_method_comparison.csv'))
    computed = mc[mc['status'] == 'computed'].copy()
    if len(computed) == 0:
        return

    # Add ImputeVAE from recovery metrics
    rec = pd.read_csv(os.path.join(TABLES, 'syngo_recovery_metrics.csv'))
    imputevae_row = rec[rec['label'] == 'all_gene_celltype_pairs'].iloc[0]

    methods = ['ImputeVAE'] + computed['method'].tolist()
    pearson_vals = [imputevae_row['pearson_r']] + computed['delta_pearson'].tolist()
    r2_vals = [imputevae_row['R2']] + computed['R2'].tolist()
    sign_vals = [imputevae_row['sign_agreement']] + computed['sign_agreement'].tolist()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    method_colors = ['#E63946'] + ['#457B9D'] * len(computed)

    for ax, vals, title in zip(axes,
                                [pearson_vals, r2_vals, sign_vals],
                                ['Pearson r', 'R^2', 'Sign agreement']):
        y = np.arange(len(methods))
        bars = ax.barh(y, vals, color=method_colors, alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(methods, fontsize=9)
        ax.set_xlabel(title)
        ax.set_title(f'SynGO delta {title}')
        ax.invert_yaxis()
        if title == 'R^2':
            ax.axvline(0, color='black', linewidth=0.5, linestyle='--')

        for i, v in enumerate(vals):
            if not np.isnan(v):
                ax.text(max(v + 0.02, 0.05), i, f'{v:.3f}', va='center', fontsize=7)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGS, 'syngo_baseline_comparison.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIGS, 'syngo_baseline_comparison.pdf'), bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    print("Generating supplementary figures...")
    fig_evidence_composition()
    print("  evidence_composition done")
    fig_subgroup_overlap_upset()
    print("  subgroup_overlap done")
    fig_subgroup_detail()
    print("  subgroup_detail done")
    fig_rare_ct_sensitivity()
    print("  rare_ct_sensitivity done")
    fig_baseline_comparison()
    print("  baseline_comparison done")
    print("All supplementary figures generated.")
