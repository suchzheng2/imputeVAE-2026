#!/usr/bin/env python3
"""Enhanced visualization of validation delta analysis results."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def create_validation_plots():
    """Create enhanced plots for validation delta analysis."""

    # Load data
    data_file = "notes/experiments/delta_analysis/valid_set/train_vs_valid_comparison.csv"
    df = pd.read_csv(data_file)

    # Create output directory
    output_dir = Path("notes/experiments/delta_analysis/valid_set/enhanced_plots")
    output_dir.mkdir(exist_ok=True)

    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    # 1. Correlation vs Sample Size scatter plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Color by sample size
    scatter = ax1.scatter(df['n_valid'], df['delta_correlation'],
                         c=df['n_valid'], s=100, alpha=0.7,
                         cmap='viridis', edgecolors='black', linewidth=0.5)

    # Add cell type labels for interesting points
    for _, row in df.iterrows():
        if row['n_valid'] <= 15 or row['delta_correlation'] < 0.9:
            ax1.annotate(row['cell_type'],
                        (row['n_valid'], row['delta_correlation']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)

    ax1.set_xlabel('Validation Set Size (n cells)', fontsize=12)
    ax1.set_ylabel('Delta Correlation (Train vs Valid)', fontsize=12)
    ax1.set_title('Delta Consistency vs Sample Size', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')

    # Add trend line
    log_n = np.log10(df['n_valid'])
    z = np.polyfit(log_n, df['delta_correlation'], 1)
    p = np.poly1d(z)
    x_trend = np.logspace(np.log10(df['n_valid'].min()),
                         np.log10(df['n_valid'].max()), 100)
    ax1.plot(x_trend, p(np.log10(x_trend)), "r--", alpha=0.7, linewidth=2)

    plt.colorbar(scatter, ax=ax1, label='Sample Size')

    # 2. Top-20 overlap scatter
    scatter2 = ax2.scatter(df['n_valid'], df['top20_overlap'],
                          c=df['delta_correlation'], s=100, alpha=0.7,
                          cmap='plasma', edgecolors='black', linewidth=0.5)

    # Add labels for low-overlap points
    for _, row in df.iterrows():
        if row['top20_overlap'] <= 12:
            ax2.annotate(row['cell_type'],
                        (row['n_valid'], row['top20_overlap']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)

    ax2.set_xlabel('Validation Set Size (n cells)', fontsize=12)
    ax2.set_ylabel('Top-20 Gene Overlap (out of 20)', fontsize=12)
    ax2.set_title('Gene Overlap vs Sample Size', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_ylim(8, 20.5)

    plt.colorbar(scatter2, ax=ax2, label='Delta Correlation')

    plt.tight_layout()
    plt.savefig(output_dir / "validation_consistency_analysis.png",
                dpi=300, bbox_inches='tight')
    plt.show()

    # 3. Summary bar chart with error indicators
    fig, ax = plt.subplots(figsize=(12, 8))

    # Sort by correlation
    df_sorted = df.sort_values('delta_correlation', ascending=True)

    # Create color map based on sample size
    colors = plt.cm.viridis(np.log10(df_sorted['n_valid']) / np.log10(df_sorted['n_valid'].max()))

    bars = ax.barh(range(len(df_sorted)), df_sorted['delta_correlation'],
                   color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    # Add sample size annotations
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        ax.text(row['delta_correlation'] + 0.01, i, f"n={row['n_valid']}",
                va='center', fontsize=9, alpha=0.7)

    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted['cell_type'])
    ax.set_xlabel('Delta Correlation (Train vs Validation)', fontsize=12)
    ax.set_title('Cell Type-Specific Delta Consistency\n(Train-Validation Agreement)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Add vertical line at mean
    mean_corr = df['delta_correlation'].mean()
    ax.axvline(mean_corr, color='red', linestyle='--', alpha=0.8, linewidth=2)
    ax.text(mean_corr + 0.02, len(df_sorted)/2, f'Mean: {mean_corr:.3f}',
            rotation=90, va='center', fontweight='bold', color='red')

    plt.tight_layout()
    plt.savefig(output_dir / "cell_type_consistency_ranking.png",
                dpi=300, bbox_inches='tight')
    plt.show()

    # 4. Summary statistics table plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')

    # Create summary stats
    summary_stats = {
        'Metric': [
            'Mean Correlation',
            'Median Correlation',
            'Mean Top-20 Overlap',
            'Median Top-20 Overlap',
            'High-n Types (>50 cells)',
            'Low-n Types (≤50 cells)',
            'Perfect Overlap (19-20/20)',
            'Poor Overlap (≤12/20)'
        ],
        'Value': [
            f"{df['delta_correlation'].mean():.3f}",
            f"{df['delta_correlation'].median():.3f}",
            f"{df['top20_overlap'].mean():.1f}/20",
            f"{df['top20_overlap'].median():.1f}/20",
            f"{len(df[df['n_valid'] > 50])}/{len(df)} types",
            f"{len(df[df['n_valid'] <= 50])}/{len(df)} types",
            f"{len(df[df['top20_overlap'] >= 19])}/{len(df)} types",
            f"{len(df[df['top20_overlap'] <= 12])}/{len(df)} types"
        ]
    }

    summary_df = pd.DataFrame(summary_stats)

    table = ax.table(cellText=summary_df.values,
                    colLabels=summary_df.columns,
                    cellLoc='left',
                    loc='center',
                    colWidths=[0.6, 0.4])

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)

    # Style the table
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax.set_title('Validation Delta Analysis Summary',
                 fontsize=16, fontweight='bold', pad=20)

    plt.savefig(output_dir / "summary_statistics.png",
                dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n✓ Enhanced validation plots saved to: {output_dir}")
    print(f"✓ Created 4 visualization files:")
    print(f"  - validation_consistency_analysis.png")
    print(f"  - cell_type_consistency_ranking.png")
    print(f"  - summary_statistics.png")

if __name__ == "__main__":
    create_validation_plots()
