#!/usr/bin/env python3
"""Simplified gene expression comparison plots."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def create_simple_gene_plots():
    """Create simple gene expression plots using pre-computed delta data."""

    # Output directory
    output_dir = Path("notes/experiments/gene_expression_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load validation comparison data
    comp_file = "notes/experiments/delta_analysis/valid_set/train_vs_valid_comparison.csv"
    df = pd.read_csv(comp_file)

    # Set plotting style
    plt.style.use('seaborn-v0_8')

    # 1. Delta correlation by cell type and sample size
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Scatter plot: sample size vs correlation
    colors = plt.cm.viridis(np.log10(df['n_valid']) / np.log10(df['n_valid'].max()))
    scatter = ax1.scatter(df['n_valid'], df['delta_correlation'],
                         c=df['n_valid'], s=100, alpha=0.8,
                         cmap='viridis', edgecolors='black', linewidth=0.5)

    # Label outliers
    for _, row in df.iterrows():
        if row['n_valid'] <= 15 or row['delta_correlation'] < 0.85:
            ax1.annotate(row['cell_type'],
                        (row['n_valid'], row['delta_correlation']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, alpha=0.9)

    ax1.set_xlabel('Validation Sample Size (n cells)')
    ax1.set_ylabel('Delta Pattern Correlation (Train vs Valid)')
    ax1.set_title('Gene Expression Pattern Consistency', fontweight='bold')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)

    # Add trend line
    log_n = np.log10(df['n_valid'])
    z = np.polyfit(log_n, df['delta_correlation'], 1)
    p = np.poly1d(z)
    x_trend = np.logspace(np.log10(df['n_valid'].min()), np.log10(df['n_valid'].max()), 100)
    ax1.plot(x_trend, p(np.log10(x_trend)), "r--", alpha=0.8, linewidth=2,
             label=f'Trend: r² = {np.corrcoef(log_n, df["delta_correlation"])[0,1]**2:.3f}')
    ax1.legend()

    plt.colorbar(scatter, ax=ax1, label='Sample Size')

    # 2. Top gene overlap analysis
    ax2.bar(range(len(df)), df['top20_overlap'],
           color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    # Add horizontal line at mean
    mean_overlap = df['top20_overlap'].mean()
    ax2.axhline(mean_overlap, color='red', linestyle='--', linewidth=2, alpha=0.8,
                label=f'Mean: {mean_overlap:.1f}/20')

    ax2.set_xlabel('Cell Types (ordered by correlation)')
    ax2.set_ylabel('Top-20 Gene Overlap (out of 20)')
    ax2.set_title('Gene Ranking Consistency', fontweight='bold')
    ax2.set_xticks(range(len(df)))
    ax2.set_xticklabels(df.sort_values('delta_correlation')['cell_type'],
                       rotation=45, ha='right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()
    ax2.set_ylim(8, 20)

    plt.tight_layout()
    plt.savefig(output_dir / "gene_expression_consistency.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 3. Load some actual delta data for top genes visualization
    delta_dir = Path("notes/experiments/delta_analysis/tables")
    if delta_dir.exists():
        # Find a good cell type with delta data
        sample_ct = "L5_IT"  # High sample size
        delta_file = delta_dir / f"{sample_ct}_delta.csv"

        if delta_file.exists():
            delta_df = pd.read_csv(delta_file)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Top increased genes
            top_inc = delta_df.head(15)
            ax1.barh(range(len(top_inc)), top_inc['median_delta'],
                    color='darkred', alpha=0.8)
            ax1.set_yticks(range(len(top_inc)))
            ax1.set_yticklabels(top_inc['gene'], fontsize=10)
            ax1.set_xlabel('Expression Delta (Imputed - Original)')
            ax1.set_title(f'Top Increased Genes in {sample_ct}', fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='x')

            # Top decreased genes
            top_dec = delta_df.tail(15)
            ax2.barh(range(len(top_dec)), top_dec['median_delta'],
                    color='darkblue', alpha=0.8)
            ax2.set_yticks(range(len(top_dec)))
            ax2.set_yticklabels(top_dec['gene'], fontsize=10)
            ax2.set_xlabel('Expression Delta (Imputed - Original)')
            ax2.set_title(f'Top Decreased Genes in {sample_ct}', fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='x')

            plt.tight_layout()
            plt.savefig(output_dir / "top_delta_genes_example.png", dpi=300, bbox_inches='tight')
            plt.show()

    # 4. Summary statistics
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create summary stats table
    high_n = df[df['n_valid'] > 50]
    low_n = df[df['n_valid'] <= 50]

    stats_data = {
        'Metric': [
            'Overall Mean Correlation',
            'Overall Median Correlation',
            'High-n Types Mean Corr (>50 cells)',
            'Low-n Types Mean Corr (≤50 cells)',
            'Overall Mean Gene Overlap',
            'High-n Types Mean Overlap',
            'Low-n Types Mean Overlap',
            'Types with Excellent Overlap (≥17/20)',
            'Types with Poor Overlap (≤12/20)'
        ],
        'Value': [
            f"{df['delta_correlation'].mean():.3f}",
            f"{df['delta_correlation'].median():.3f}",
            f"{high_n['delta_correlation'].mean():.3f}" if len(high_n) > 0 else "N/A",
            f"{low_n['delta_correlation'].mean():.3f}" if len(low_n) > 0 else "N/A",
            f"{df['top20_overlap'].mean():.1f}/20",
            f"{high_n['top20_overlap'].mean():.1f}/20" if len(high_n) > 0 else "N/A",
            f"{low_n['top20_overlap'].mean():.1f}/20" if len(low_n) > 0 else "N/A",
            f"{len(df[df['top20_overlap'] >= 17])}/{len(df)}",
            f"{len(df[df['top20_overlap'] <= 12])}/{len(df)}"
        ]
    }

    # Remove axis
    ax.axis('tight')
    ax.axis('off')

    # Create table
    table = ax.table(cellText=[[row['Metric'], row['Value']] for row in
                              [dict(zip(stats_data.keys(), vals)) for vals in
                               zip(*stats_data.values())]],
                    colLabels=['Metric', 'Value'],
                    cellLoc='left',
                    loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(stats_data['Metric']) + 1):
        color = '#f0f0f0' if i % 2 == 0 else 'white'
        table[(i, 0)].set_facecolor(color)
        table[(i, 1)].set_facecolor(color)

    ax.set_title('Gene Expression Analysis Summary\n(Validation Set Consistency)',
                fontsize=16, fontweight='bold', pad=20)

    plt.savefig(output_dir / "summary_table.png", dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n✓ Gene expression plots saved to: {output_dir}")
    print(f"✓ Created plots:")
    print(f"  - gene_expression_consistency.png")
    print(f"  - top_delta_genes_example.png")
    print(f"  - summary_table.png")

if __name__ == "__main__":
    create_simple_gene_plots()
