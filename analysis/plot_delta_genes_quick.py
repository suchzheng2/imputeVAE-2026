#!/usr/bin/env python3
"""Quick plot of top delta genes."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_delta_genes():
    """Plot top increased and decreased genes."""

    # Load delta data for L2/3 IT (high sample size)
    delta_file = "notes/experiments/delta_analysis/tables/L2_3 IT_delta.csv"
    df = pd.read_csv(delta_file)

    # Output directory
    output_dir = Path("notes/experiments/gene_expression_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Top increased genes
    top_inc = df.head(15)
    bars1 = ax1.barh(range(len(top_inc)), top_inc['median_delta'],
                    color='darkred', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.set_yticks(range(len(top_inc)))
    ax1.set_yticklabels(top_inc['gene'], fontsize=11)
    ax1.set_xlabel('Expression Delta (Imputed - Original)', fontsize=12)
    ax1.set_title('Top 15 Increased Genes\n(L2/3 IT Cell Type)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.invert_yaxis()

    # Add values on bars
    for i, (idx, row) in enumerate(top_inc.iterrows()):
        ax1.text(row['median_delta'] + 0.05, i, f"{row['median_delta']:.2f}",
                va='center', fontsize=9)

    # Top decreased genes
    top_dec = df.tail(15)
    bars2 = ax2.barh(range(len(top_dec)), top_dec['median_delta'],
                    color='darkblue', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.set_yticks(range(len(top_dec)))
    ax2.set_yticklabels(top_dec['gene'], fontsize=11)
    ax2.set_xlabel('Expression Delta (Imputed - Original)', fontsize=12)
    ax2.set_title('Top 15 Decreased Genes\n(L2/3 IT Cell Type)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.invert_yaxis()

    # Add values on bars
    for i, (idx, row) in enumerate(top_dec.iterrows()):
        ax2.text(row['median_delta'] - 0.1, i, f"{row['median_delta']:.2f}",
                va='center', ha='right', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "top_delta_genes_L2_3_IT.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary
    print(f"\n✓ Delta genes plot saved: {output_dir}/top_delta_genes_L2_3_IT.png")
    print(f"\nTop increased genes (L2/3 IT):")
    for gene, delta in zip(top_inc['gene'].head(10), top_inc['median_delta'].head(10)):
        print(f"  {gene:<15} +{delta:.2f}")

    print(f"\nTop decreased genes (L2/3 IT):")
    for gene, delta in zip(top_dec['gene'].tail(10), top_dec['median_delta'].tail(10)):
        print(f"  {gene:<15} {delta:.2f}")

if __name__ == "__main__":
    plot_delta_genes()
