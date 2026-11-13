#!/usr/bin/env python3
"""
Gene Stratification Analysis for imputeVAE.

Stratifies genes by expression level and cell-vs-nucleus difference,
then evaluates imputation quality per stratum. Addresses the critical
question: does the model add value on genes that actually differ between
cell and nucleus?

Reads gene-level correlation CSVs from the evaluation pipeline.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def parse_args():
    p = argparse.ArgumentParser(description="Gene stratification analysis")
    p.add_argument("--run_folder", required=True, help="Run folder name or path")
    p.add_argument("--epoch", type=int, default=599)
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output dir (default: run_folder/evaluation/gene_stratification)")
    return p.parse_args()


def load_gene_corr(run_path, epoch, decoder_label):
    """Load gene correlation CSV for a given decoder."""
    csv_path = os.path.join(run_path, "evaluation", "gene_correlation",
                            f"epoch_{epoch}", f"{decoder_label}_gene_correlation.csv")
    if not os.path.exists(csv_path):
        print(f"WARNING: {csv_path} not found")
        return None
    df = pd.read_csv(csv_path)
    return df


def assign_expression_quintiles(df, col='mean_expr_raw_cell'):
    """Assign genes to expression level quintiles."""
    df = df.copy()
    df['expr_quintile'] = pd.qcut(df[col], 5,
                                   labels=['Q1 (lowest)', 'Q2', 'Q3', 'Q4', 'Q5 (highest)'])
    return df


def assign_cell_nucleus_diff(df):
    """Compute cell-vs-nucleus expression difference and assign strata."""
    df = df.copy()
    df['cell_nuc_diff'] = df['mean_expr_raw_cell'] - df['mean_expr_raw_nuclei']
    df['cell_nuc_abs_diff'] = df['cell_nuc_diff'].abs()
    df['cell_nuc_log_ratio'] = np.log2(
        (df['mean_expr_raw_cell'] + 0.01) / (df['mean_expr_raw_nuclei'] + 0.01))

    # Rank by absolute difference
    df['diff_rank'] = df['cell_nuc_abs_diff'].rank(ascending=False)
    n = len(df)

    conditions = [
        df['diff_rank'] <= 1000,
        df['diff_rank'] <= 2000,
        df['diff_rank'] > n - 1000,
    ]
    choices = ['Top 1K most different', 'Top 2K most different', 'Bottom 1K (housekeeping)']
    df['diff_stratum'] = np.select(conditions, choices, default='Middle genes')

    # Also create a finer-grained quintile
    df['diff_quintile'] = pd.qcut(df['cell_nuc_abs_diff'], 5,
                                   labels=['Q1 (most similar)', 'Q2', 'Q3', 'Q4',
                                           'Q5 (most different)'])
    return df


def assign_biological_categories(df):
    """Tag genes with known biological categories."""
    df = df.copy()

    # Immediate early genes (known cytoplasm-enriched, lost in snRNA-seq)
    iegs = {'Fos', 'Fosb', 'Jun', 'Junb', 'Jund', 'Arc', 'Egr1', 'Egr2',
            'Egr3', 'Egr4', 'Nr4a1', 'Nr4a2', 'Nr4a3', 'Npas4', 'Bdnf',
            'Homer1', 'Dusp1', 'Dusp5', 'Dusp6', 'Btg2'}

    # Ribosomal protein genes
    ribo = {g for g in df['gene_name'] if g.startswith(('Rps', 'Rpl'))}

    # Neuronal markers
    neuronal = {'Rbfox3', 'Snap25', 'Syt1', 'Stmn2', 'Map2', 'Tubb3',
                'Dcx', 'Tbr1', 'Satb2', 'Bcl11b', 'Fezf2'}

    # Glial markers
    glial = {'Gfap', 'Aqp4', 'Aldh1l1', 'Slc1a3',  # astrocyte
             'Mbp', 'Plp1', 'Mog', 'Mag', 'Cnp',     # oligodendrocyte
             'Cx3cr1', 'Csf1r', 'P2ry12', 'Tmem119',  # microglia
             'Pdgfra', 'Cspg4', 'Sox10'}               # OPC

    # Cell-type-specific TFs
    tfs = {'Lhx6', 'Dlx1', 'Dlx2', 'Dlx5', 'Nkx2-1', 'Sox6',  # inhibitory
           'Neurod2', 'Neurod6', 'Tbr1', 'Satb2', 'Fezf2',      # excitatory
           'Olig1', 'Olig2', 'Sox10', 'Nkx6-2',                  # oligo
           'Sox9', 'Nfia', 'Nfib'}                                 # astrocyte

    categories = []
    for g in df['gene_name']:
        cats = []
        if g in iegs:
            cats.append('IEG')
        if g in ribo:
            cats.append('Ribosomal')
        if g in neuronal:
            cats.append('Neuronal marker')
        if g in glial:
            cats.append('Glial marker')
        if g in tfs:
            cats.append('TF')
        categories.append('; '.join(cats) if cats else 'Other')

    df['bio_category'] = categories
    return df


def compute_stratum_stats(df, stratum_col, corr_col='corr_vs_raw_cell'):
    """Compute summary statistics per stratum."""
    rows = []
    for name, grp in df.groupby(stratum_col, observed=True):
        r = grp[corr_col]
        rows.append({
            'stratum': name,
            'n_genes': len(grp),
            'median_corr': r.median(),
            'mean_corr': r.mean(),
            'std_corr': r.std(),
            'q25_corr': r.quantile(0.25),
            'q75_corr': r.quantile(0.75),
            'pct_above_0.8': (r > 0.8).mean() * 100,
            'pct_above_0.5': (r > 0.5).mean() * 100,
            'mean_expr_cell': grp['mean_expr_raw_cell'].mean(),
            'mean_expr_nuclei': grp['mean_expr_raw_nuclei'].mean(),
            'mean_abs_diff': grp['cell_nuc_abs_diff'].mean() if 'cell_nuc_abs_diff' in grp else np.nan,
        })
    return pd.DataFrame(rows)


# =============================================================================
# Visualization
# =============================================================================

def plot_stratification_overview(df_cell, df_nuc, output_dir):
    """Main overview figure: 2x3 grid showing stratification results."""
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

    # --- Panel A: Expression quintile vs correlation (cell decoder) ---
    ax = fig.add_subplot(gs[0, 0])
    quintiles = ['Q1 (lowest)', 'Q2', 'Q3', 'Q4', 'Q5 (highest)']
    cell_medians = [df_cell[df_cell['expr_quintile'] == q]['corr_vs_raw_cell'].median()
                    for q in quintiles]
    nuc_medians = [df_nuc[df_nuc['expr_quintile'] == q]['corr_vs_raw_cell'].median()
                   for q in quintiles]
    x = np.arange(5)
    ax.bar(x - 0.17, cell_medians, 0.3, label='Cell decoder', color='#2196F3', alpha=0.85)
    ax.bar(x + 0.17, nuc_medians, 0.3, label='Nuclei decoder (cross)', color='#FF9800', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(['Q1\n(low)', 'Q2', 'Q3', 'Q4', 'Q5\n(high)'])
    ax.set_ylabel('Median gene-level Pearson r')
    ax.set_title('A. Imputation quality by expression level', fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.axhline(0.8, color='gray', ls='--', lw=0.8, alpha=0.5)

    # --- Panel B: Cell-nucleus difference quintile vs correlation ---
    ax = fig.add_subplot(gs[0, 1])
    diff_qs = ['Q1 (most similar)', 'Q2', 'Q3', 'Q4', 'Q5 (most different)']
    cell_diff_med = [df_cell[df_cell['diff_quintile'] == q]['corr_vs_raw_cell'].median()
                     for q in diff_qs]
    nuc_diff_med = [df_nuc[df_nuc['diff_quintile'] == q]['corr_vs_raw_cell'].median()
                    for q in diff_qs]
    ax.bar(x - 0.17, cell_diff_med, 0.3, label='Cell decoder', color='#2196F3', alpha=0.85)
    ax.bar(x + 0.17, nuc_diff_med, 0.3, label='Nuclei decoder (cross)', color='#FF9800', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(['Q1\n(similar)', 'Q2', 'Q3', 'Q4', 'Q5\n(different)'])
    ax.set_ylabel('Median gene-level Pearson r')
    ax.set_title('B. Imputation quality by cell-nucleus difference', fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.axhline(0.8, color='gray', ls='--', lw=0.8, alpha=0.5)

    # --- Panel C: Scatter — cell-nuc abs diff vs gene corr ---
    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(df_nuc['cell_nuc_abs_diff'], df_nuc['corr_vs_raw_cell'],
               s=3, alpha=0.15, color='#FF9800', label='Nuclei decoder (cross)')
    ax.scatter(df_cell['cell_nuc_abs_diff'], df_cell['corr_vs_raw_cell'],
               s=3, alpha=0.15, color='#2196F3', label='Cell decoder')
    ax.set_xlabel('|mean_cell - mean_nuclei| (log1p)')
    ax.set_ylabel('Gene-level Pearson r vs raw cell')
    ax.set_title('C. Gene correlation vs cell-nucleus gap', fontweight='bold')
    ax.legend(fontsize=8, markerscale=5)
    ax.axhline(0.8, color='gray', ls='--', lw=0.8, alpha=0.5)

    # --- Panel D: Top 1K vs Bottom 1K bar chart ---
    ax = fig.add_subplot(gs[1, 0])
    strata = ['Top 1K most different', 'Middle genes', 'Bottom 1K (housekeeping)']
    colors_cell = ['#1565C0', '#42A5F5', '#90CAF9']
    colors_nuc = ['#E65100', '#FF9800', '#FFE0B2']
    cell_vals = [df_cell[df_cell['diff_stratum'] == s]['corr_vs_raw_cell'].median()
                 for s in strata]
    nuc_vals = [df_nuc[df_nuc['diff_stratum'] == s]['corr_vs_raw_cell'].median()
                for s in strata]
    x3 = np.arange(3)
    bars1 = ax.bar(x3 - 0.17, cell_vals, 0.3, color=colors_cell, edgecolor='black', lw=0.5)
    bars2 = ax.bar(x3 + 0.17, nuc_vals, 0.3, color=colors_nuc, edgecolor='black', lw=0.5)
    ax.set_xticks(x3)
    ax.set_xticklabels(['Top 1K\n(most different)', 'Middle\ngenes', 'Bottom 1K\n(housekeeping)'],
                        fontsize=9)
    ax.set_ylabel('Median gene-level Pearson r')
    ax.set_title('D. Critical test: most-different vs housekeeping', fontweight='bold')
    ax.set_ylim(0, 1.05)
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.3f}',
                    ha='center', va='bottom', fontsize=8)
    ax.legend([bars1[0], bars2[0]], ['Cell decoder', 'Nuclei decoder (cross)'], fontsize=8)

    # --- Panel E: Biological category analysis ---
    ax = fig.add_subplot(gs[1, 1])
    bio_cats = ['IEG', 'Ribosomal', 'Neuronal marker', 'Glial marker', 'TF', 'Other']
    cell_bio = []
    nuc_bio = []
    n_genes = []
    for cat in bio_cats:
        mask_cell = df_cell['bio_category'].str.contains(cat) if cat != 'Other' else (df_cell['bio_category'] == 'Other')
        mask_nuc = df_nuc['bio_category'].str.contains(cat) if cat != 'Other' else (df_nuc['bio_category'] == 'Other')
        cell_bio.append(df_cell[mask_cell]['corr_vs_raw_cell'].median())
        nuc_bio.append(df_nuc[mask_nuc]['corr_vs_raw_cell'].median())
        n_genes.append(mask_cell.sum())
    x6 = np.arange(len(bio_cats))
    ax.bar(x6 - 0.17, cell_bio, 0.3, label='Cell decoder', color='#2196F3', alpha=0.85)
    ax.bar(x6 + 0.17, nuc_bio, 0.3, label='Nuclei decoder (cross)', color='#FF9800', alpha=0.85)
    ax.set_xticks(x6)
    ax.set_xticklabels([f'{c}\n(n={n})' for c, n in zip(bio_cats, n_genes)],
                        fontsize=8, rotation=15, ha='right')
    ax.set_ylabel('Median gene-level Pearson r')
    ax.set_title('E. Imputation quality by gene category', fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.axhline(0.8, color='gray', ls='--', lw=0.8, alpha=0.5)

    # --- Panel F: Distribution of gene correlations by stratum ---
    ax = fig.add_subplot(gs[1, 2])
    for stratum, color, label in [
        ('Top 1K most different', '#E65100', 'Cross-dec: Top 1K diff'),
        ('Bottom 1K (housekeeping)', '#FFE0B2', 'Cross-dec: Bottom 1K'),
    ]:
        vals = df_nuc[df_nuc['diff_stratum'] == stratum]['corr_vs_raw_cell'].dropna()
        ax.hist(vals, bins=50, alpha=0.6, color=color, label=label, density=True)
    for stratum, color, label in [
        ('Top 1K most different', '#1565C0', 'Cell-dec: Top 1K diff'),
        ('Bottom 1K (housekeeping)', '#90CAF9', 'Cell-dec: Bottom 1K'),
    ]:
        vals = df_cell[df_cell['diff_stratum'] == stratum]['corr_vs_raw_cell'].dropna()
        ax.hist(vals, bins=50, alpha=0.4, color=color, label=label, density=True)
    ax.set_xlabel('Gene-level Pearson r vs raw cell')
    ax.set_ylabel('Density')
    ax.set_title('F. Correlation distributions: top 1K vs bottom 1K', fontweight='bold')
    ax.legend(fontsize=7, loc='upper left')

    plt.savefig(os.path.join(output_dir, 'gene_stratification_overview.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: gene_stratification_overview.png")


def plot_expression_vs_corr_2d(df_cell, df_nuc, output_dir):
    """2D heatmap: expression level x cell-nuc diff → median correlation."""
    import seaborn as sns

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, df, title in [
        (axes[0], df_cell, 'Cell decoder → raw cell'),
        (axes[1], df_nuc, 'Nuclei decoder (cross) → raw cell'),
    ]:
        # Create 2D bins
        df = df.copy()
        df['expr_bin'] = pd.qcut(df['mean_expr_raw_cell'], 5,
                                  labels=['Very low', 'Low', 'Medium', 'High', 'Very high'])
        df['diff_bin'] = pd.qcut(df['cell_nuc_abs_diff'], 5,
                                  labels=['Very similar', 'Similar', 'Moderate', 'Different', 'Very different'])

        pivot = df.pivot_table(values='corr_vs_raw_cell', index='diff_bin',
                                columns='expr_bin', aggfunc='median')
        # Reverse so "Very different" is on top
        pivot = pivot.iloc[::-1]

        # Count genes per bin for annotation
        count_pivot = df.pivot_table(values='corr_vs_raw_cell', index='diff_bin',
                                      columns='expr_bin', aggfunc='count')
        count_pivot = count_pivot.iloc[::-1]

        # Build annotation strings: value + count, or "n/a" for empty bins
        annot_strings = pivot.copy().astype(str)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.iloc[i, j]
                cnt = count_pivot.iloc[i, j] if not np.isnan(count_pivot.iloc[i, j]) else 0
                if np.isnan(val) or cnt == 0:
                    annot_strings.iloc[i, j] = 'n/a'
                else:
                    annot_strings.iloc[i, j] = f'{val:.3f}\n(n={int(cnt)})'

        # Use mask for truly empty cells
        mask = pivot.isna()

        sns.heatmap(pivot, annot=annot_strings, fmt='', cmap='RdYlGn', center=0.7,
                    vmin=0.3, vmax=1.0, ax=ax, linewidths=0.5, mask=mask,
                    annot_kws={'fontsize': 9, 'fontweight': 'bold'},
                    cbar_kws={'label': 'Median Pearson r'})
        # Color empty cells light gray
        ax.set_facecolor('#f0f0f0')
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_ylabel('Cell-nucleus difference')
        ax.set_xlabel('Expression level')

    plt.suptitle('Gene imputation quality: Expression level × Cell-nucleus difference',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'expression_vs_diff_heatmap.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: expression_vs_diff_heatmap.png")


def plot_top_bottom_genes(df_nuc, output_dir, n=30):
    """Bar chart of top and bottom N genes by cross-decode correlation."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    # Top N
    top = df_nuc.nlargest(n, 'corr_vs_raw_cell')
    ax = axes[0]
    colors = ['#E65100' if d > df_nuc['cell_nuc_abs_diff'].quantile(0.8)
              else '#FF9800' if d > df_nuc['cell_nuc_abs_diff'].quantile(0.5)
              else '#FFE0B2' for d in top['cell_nuc_abs_diff']]
    ax.barh(range(n), top['corr_vs_raw_cell'].values, color=colors, edgecolor='black', lw=0.3)
    ax.set_yticks(range(n))
    ax.set_yticklabels(top['gene_name'].values, fontsize=8)
    ax.set_xlabel('Pearson r (nuclei decoded vs raw cell)')
    ax.set_title(f'Top {n} genes: best cross-decode quality', fontweight='bold')
    ax.set_xlim(0.95, 1.0)
    ax.invert_yaxis()

    # Bottom N
    bottom = df_nuc.nsmallest(n, 'corr_vs_raw_cell')
    ax = axes[1]
    colors = ['#E65100' if d > df_nuc['cell_nuc_abs_diff'].quantile(0.8)
              else '#FF9800' if d > df_nuc['cell_nuc_abs_diff'].quantile(0.5)
              else '#FFE0B2' for d in bottom['cell_nuc_abs_diff']]
    ax.barh(range(n), bottom['corr_vs_raw_cell'].values, color=colors, edgecolor='black', lw=0.3)
    ax.set_yticks(range(n))
    ax.set_yticklabels(bottom['gene_name'].values, fontsize=8)
    ax.set_xlabel('Pearson r (nuclei decoded vs raw cell)')
    ax.set_title(f'Bottom {n} genes: worst cross-decode quality', fontweight='bold')
    ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#E65100', label='High cell-nuc diff (top 20%)'),
        Patch(facecolor='#FF9800', label='Medium cell-nuc diff'),
        Patch(facecolor='#FFE0B2', label='Low cell-nuc diff (bottom 50%)'),
    ]
    axes[1].legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_bottom_genes_crossdecode.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: top_bottom_genes_crossdecode.png")


def plot_ieg_detail(df_cell, df_nuc, output_dir):
    """Detailed view of IEG and marker gene imputation quality."""
    iegs = {'Fos', 'Fosb', 'Jun', 'Junb', 'Jund', 'Arc', 'Egr1', 'Egr2',
            'Egr3', 'Nr4a1', 'Nr4a2', 'Nr4a3', 'Npas4', 'Bdnf', 'Homer1'}

    markers = {'Rbfox3', 'Snap25', 'Syt1', 'Map2',       # pan-neuronal
               'Gfap', 'Aqp4', 'Aldh1l1',                 # astrocyte
               'Mbp', 'Plp1', 'Mog', 'Cnp',               # oligo
               'Cx3cr1', 'Csf1r', 'P2ry12',               # microglia
               'Pdgfra', 'Cspg4',                          # OPC
               'Gad1', 'Gad2', 'Slc17a7', 'Slc17a6'}      # Exc/Inh

    genes_of_interest = iegs | markers
    present = genes_of_interest & set(df_nuc['gene_name'])
    missing = genes_of_interest - present

    if missing:
        print(f"  Genes not in gene list: {sorted(missing)}")

    df_subset = df_nuc[df_nuc['gene_name'].isin(present)].copy()
    df_subset_cell = df_cell[df_cell['gene_name'].isin(present)].copy()

    if len(df_subset) == 0:
        print("  No IEG/marker genes found in data, skipping plot")
        return

    # Categorize
    df_subset['category'] = df_subset['gene_name'].apply(
        lambda g: 'IEG' if g in iegs else 'Cell-type marker')
    df_subset = df_subset.sort_values(['category', 'corr_vs_raw_cell'], ascending=[True, False])

    df_subset_cell = df_subset_cell.set_index('gene_name')

    fig, ax = plt.subplots(figsize=(14, max(8, len(df_subset) * 0.35)))

    y = np.arange(len(df_subset))
    gene_names = df_subset['gene_name'].values

    # Cross-decode (nuclei→cell)
    ax.barh(y + 0.15, df_subset['corr_vs_raw_cell'].values, 0.3,
            color='#FF9800', alpha=0.85, label='Nuclei decoder (cross) → raw cell')
    # Cell decoder
    cell_corrs = [df_subset_cell.loc[g, 'corr_vs_raw_cell'] if g in df_subset_cell.index else np.nan
                  for g in gene_names]
    ax.barh(y - 0.15, cell_corrs, 0.3,
            color='#2196F3', alpha=0.85, label='Cell decoder → raw cell')

    ax.set_yticks(y)
    ylabels = []
    for _, row in df_subset.iterrows():
        diff = row['mean_expr_raw_cell'] - row['mean_expr_raw_nuclei']
        ylabels.append(f"{row['gene_name']}  (Δ={diff:.2f})")
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_xlabel('Gene-level Pearson r vs raw cell')
    ax.set_title('IEGs and cell-type markers: imputation quality\n'
                 '(Δ = mean_cell − mean_nuclei in log1p)', fontweight='bold')
    ax.legend(fontsize=9)
    ax.axvline(0.8, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax.invert_yaxis()

    # Add category separators
    categories = df_subset['category'].values
    for i in range(1, len(categories)):
        if categories[i] != categories[i-1]:
            ax.axhline(i - 0.5, color='black', ls='-', lw=1)
            ax.text(0.02, i - 0.5, categories[i], fontsize=10, fontweight='bold',
                    transform=ax.get_yaxis_transform(), va='bottom')
    ax.text(0.02, -0.3, categories[0], fontsize=10, fontweight='bold',
            transform=ax.get_yaxis_transform(), va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ieg_marker_detail.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: ieg_marker_detail.png")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # Resolve run path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.isabs(args.run_folder):
        run_path = args.run_folder
    else:
        run_path = os.path.join(project_root, "run", args.run_folder)

    if not os.path.isdir(run_path):
        print(f"ERROR: Run folder not found: {run_path}")
        sys.exit(1)

    output_dir = args.output_dir or os.path.join(run_path, "evaluation", "gene_stratification")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Run: {run_path}")
    print(f"Output: {output_dir}")

    # Load gene correlation data
    df_cell = load_gene_corr(run_path, args.epoch, "decoded_cell")
    df_nuc = load_gene_corr(run_path, args.epoch, "decoded_nuclei")

    if df_cell is None or df_nuc is None:
        print("ERROR: Could not load gene correlation CSVs")
        sys.exit(1)

    print(f"Cell decoder: {len(df_cell)} genes")
    print(f"Nuclei decoder: {len(df_nuc)} genes")

    # Assign strata to both dataframes
    for df in [df_cell, df_nuc]:
        assign_expression_quintiles(df, col='mean_expr_raw_cell')
        # Need to copy columns since assign functions return copies
    df_cell = assign_expression_quintiles(df_cell)
    df_cell = assign_cell_nucleus_diff(df_cell)
    df_cell = assign_biological_categories(df_cell)

    df_nuc = assign_expression_quintiles(df_nuc)
    df_nuc = assign_cell_nucleus_diff(df_nuc)
    df_nuc = assign_biological_categories(df_nuc)

    # =========================================================================
    # Compute and print summary tables
    # =========================================================================

    print("\n" + "="*80)
    print("STRATIFICATION BY EXPRESSION LEVEL")
    print("="*80)
    for label, df in [("Cell decoder", df_cell), ("Nuclei decoder (cross)", df_nuc)]:
        print(f"\n  {label}:")
        stats_df = compute_stratum_stats(df, 'expr_quintile')
        print(stats_df[['stratum', 'n_genes', 'median_corr', 'mean_corr', 'q25_corr',
                         'q75_corr', 'pct_above_0.8']].to_string(index=False))

    print("\n" + "="*80)
    print("STRATIFICATION BY CELL-NUCLEUS DIFFERENCE")
    print("="*80)
    for label, df in [("Cell decoder", df_cell), ("Nuclei decoder (cross)", df_nuc)]:
        print(f"\n  {label}:")
        stats_df = compute_stratum_stats(df, 'diff_quintile')
        print(stats_df[['stratum', 'n_genes', 'median_corr', 'mean_corr', 'mean_abs_diff',
                         'pct_above_0.8']].to_string(index=False))

    print("\n" + "="*80)
    print("TOP 1K vs BOTTOM 1K (CRITICAL TEST)")
    print("="*80)
    for label, df in [("Cell decoder", df_cell), ("Nuclei decoder (cross)", df_nuc)]:
        print(f"\n  {label}:")
        stats_df = compute_stratum_stats(df, 'diff_stratum')
        print(stats_df[['stratum', 'n_genes', 'median_corr', 'mean_corr', 'mean_abs_diff',
                         'pct_above_0.8', 'pct_above_0.5']].to_string(index=False))

    print("\n" + "="*80)
    print("BIOLOGICAL CATEGORIES")
    print("="*80)
    for label, df in [("Cell decoder", df_cell), ("Nuclei decoder (cross)", df_nuc)]:
        print(f"\n  {label}:")
        bio_stats = []
        for cat in ['IEG', 'Ribosomal', 'Neuronal marker', 'Glial marker', 'TF', 'Other']:
            if cat == 'Other':
                mask = df['bio_category'] == 'Other'
            else:
                mask = df['bio_category'].str.contains(cat)
            sub = df[mask]
            if len(sub) > 0:
                bio_stats.append({
                    'category': cat,
                    'n_genes': len(sub),
                    'median_corr': sub['corr_vs_raw_cell'].median(),
                    'mean_abs_diff': sub['cell_nuc_abs_diff'].mean(),
                    'pct_above_0.8': (sub['corr_vs_raw_cell'] > 0.8).mean() * 100,
                })
        print(pd.DataFrame(bio_stats).to_string(index=False))

    # =========================================================================
    # Generate plots
    # =========================================================================
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    plot_stratification_overview(df_cell, df_nuc, output_dir)
    plot_expression_vs_corr_2d(df_cell, df_nuc, output_dir)
    plot_top_bottom_genes(df_nuc, output_dir)
    plot_ieg_detail(df_cell, df_nuc, output_dir)

    # Save full stratified table
    df_nuc_out = df_nuc[['gene_name', 'corr_vs_raw_cell', 'corr_vs_raw_nuclei',
                          'mean_expr_raw_cell', 'mean_expr_raw_nuclei',
                          'cell_nuc_diff', 'cell_nuc_abs_diff', 'cell_nuc_log_ratio',
                          'expr_quintile', 'diff_quintile', 'diff_stratum',
                          'bio_category']].copy()
    df_nuc_out.to_csv(os.path.join(output_dir, 'gene_stratification_full.csv'), index=False)
    print(f"Saved: gene_stratification_full.csv")

    # Save summary tables
    for stratum_col, fname in [
        ('expr_quintile', 'summary_by_expression.csv'),
        ('diff_quintile', 'summary_by_cell_nuc_diff.csv'),
        ('diff_stratum', 'summary_top_bottom_1k.csv'),
    ]:
        for label, df in [("cell_decoder", df_cell), ("nuclei_crossdecode", df_nuc)]:
            s = compute_stratum_stats(df, stratum_col)
            s['decoder'] = label
            s.to_csv(os.path.join(output_dir, f'{label}_{fname}'), index=False)

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()
