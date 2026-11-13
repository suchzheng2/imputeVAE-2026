#!/usr/bin/env python3
"""Comprehensive GO analysis visualization for training and validation sets."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse
import networkx as nx
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

def load_go_results(results_dir):
    """Load GO enrichment results from CSV files."""
    results = {}

    files = {
        'increased': 'increased_go_results.csv',
        'decreased': 'decreased_go_results.csv',
        'decreased_named': 'decreased_named_go_results.csv'
    }

    for key, filename in files.items():
        filepath = Path(results_dir) / filename
        if filepath.exists():
            results[key] = pd.read_csv(filepath)
        else:
            print(f"Warning: {filepath} not found")
            results[key] = pd.DataFrame()

    return results

def plot_go_barplot(results_dict, output_dir, max_terms=15):
    """Create horizontal bar plots of top GO terms."""

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()

    # Define categories and colors
    categories = {
        'GO:CC': '#E31A1C',  # Red
        'GO:BP': '#1F78B4',  # Blue
        'GO:MF': '#33A02C',  # Green
        'KEGG': '#FF7F00'    # Orange
    }

    plot_configs = [
        ('increased', 'GO:CC', 'Increased Genes - Cellular Component'),
        ('increased', 'GO:BP', 'Increased Genes - Biological Process'),
        ('decreased_named', 'GO:CC', 'Decreased Genes - Cellular Component'),
        ('decreased_named', 'GO:BP', 'Decreased Genes - Biological Process')
    ]

    for i, (gene_set, go_cat, title) in enumerate(plot_configs):
        ax = axes[i]

        if gene_set not in results_dict or results_dict[gene_set].empty:
            ax.text(0.5, 0.5, f'No data for\n{title}',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(title, fontweight='bold')
            continue

        # Filter by category
        df = results_dict[gene_set]
        df_cat = df[df['source'] == go_cat].head(max_terms)

        if df_cat.empty:
            ax.text(0.5, 0.5, f'No {go_cat} terms found',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(title, fontweight='bold')
            continue

        # Calculate -log10(p_value) for plotting
        df_cat = df_cat.copy()
        df_cat['neg_log_p'] = -np.log10(df_cat['p_value'])

        # Sort by significance
        df_cat = df_cat.sort_values('neg_log_p', ascending=True)

        # Create horizontal bar plot
        bars = ax.barh(range(len(df_cat)), df_cat['neg_log_p'],
                      color=categories[go_cat], alpha=0.8,
                      edgecolor='black', linewidth=0.5)

        # Customize y-axis
        ax.set_yticks(range(len(df_cat)))
        # Truncate long term names
        labels = [name[:50] + '...' if len(name) > 50 else name
                 for name in df_cat['name']]
        ax.set_yticklabels(labels, fontsize=10)

        # Add gene count annotations
        for j, (idx, row) in enumerate(df_cat.iterrows()):
            ax.text(row['neg_log_p'] + 0.1, j,
                   f"n={row['intersection_size']}",
                   va='center', fontsize=9, alpha=0.8)

        ax.set_xlabel('-log₁₀(p-value)', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        # Add significance threshold line
        ax.axvline(-np.log10(0.05), color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax.text(-np.log10(0.05) + 0.1, len(df_cat)/2, 'p=0.05',
               rotation=90, va='center', color='red', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / "go_terms_barplot.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_go_dotplot(results_dict, output_dir):
    """Create dot plot showing GO terms with size=gene_ratio, color=p_value."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))

    plot_data = [
        ('increased', ax1, 'Increased Genes GO Terms'),
        ('decreased_named', ax2, 'Decreased Genes GO Terms')
    ]

    for gene_set, ax, title in plot_data:
        if gene_set not in results_dict or results_dict[gene_set].empty:
            ax.text(0.5, 0.5, f'No data for {title}',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(title, fontweight='bold')
            continue

        df = results_dict[gene_set].copy()

        # Filter top terms from each category
        top_terms = []
        for source in ['GO:CC', 'GO:BP', 'GO:MF', 'KEGG']:
            subset = df[df['source'] == source].head(8)
            top_terms.append(subset)

        if not top_terms:
            continue

        plot_df = pd.concat(top_terms, ignore_index=True)

        if plot_df.empty:
            continue

        # Calculate metrics
        plot_df['gene_ratio'] = plot_df['intersection_size'] / plot_df['term_size']
        plot_df['neg_log_p'] = -np.log10(plot_df['p_value'])

        # Create scatter plot
        scatter = ax.scatter(plot_df['gene_ratio'],
                           range(len(plot_df)),
                           c=plot_df['neg_log_p'],
                           s=plot_df['intersection_size'] * 10,
                           alpha=0.8, cmap='plasma',
                           edgecolors='black', linewidth=0.5)

        # Customize y-axis
        ax.set_yticks(range(len(plot_df)))
        labels = [f"{row['source']}: {row['name'][:40]}{'...' if len(row['name']) > 40 else ''}"
                 for _, row in plot_df.iterrows()]
        ax.set_yticklabels(labels, fontsize=9)

        ax.set_xlabel('Gene Ratio (genes in term / total genes in term)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('-log₁₀(p-value)', fontsize=11)

        # Add size legend
        sizes = [20, 50, 100]
        labels = ['20', '50', '100']
        legend_elements = [plt.scatter([], [], s=s*10, c='gray', alpha=0.6,
                                     edgecolors='black', linewidth=0.5)
                          for s in sizes]
        ax.legend(legend_elements, labels,
                 title='Gene Count', loc='lower right',
                 frameon=True, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "go_terms_dotplot.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_go_comparison(train_results, valid_results, output_dir):
    """Compare GO enrichment between training and validation sets."""

    if not train_results or not valid_results:
        print("Need both training and validation GO results for comparison")
        return

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    comparisons = [
        ('increased', 'GO:CC', 'Cellular Component - Increased'),
        ('increased', 'GO:BP', 'Biological Process - Increased'),
        ('decreased_named', 'GO:CC', 'Cellular Component - Decreased'),
        ('decreased_named', 'GO:BP', 'Biological Process - Decreased')
    ]

    for i, (gene_set, go_cat, title) in enumerate(comparisons):
        ax = axes[i//2, i%2]

        # Get data for both sets
        train_df = train_results.get(gene_set, pd.DataFrame())
        valid_df = valid_results.get(gene_set, pd.DataFrame())

        if train_df.empty or valid_df.empty:
            ax.text(0.5, 0.5, f'Insufficient data\nfor {title}',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(title, fontweight='bold')
            continue

        # Filter by GO category
        train_cat = train_df[train_df['source'] == go_cat].head(15)
        valid_cat = valid_df[valid_df['source'] == go_cat].head(15)

        if train_cat.empty and valid_cat.empty:
            ax.text(0.5, 0.5, f'No {go_cat} terms',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(title, fontweight='bold')
            continue

        # Find overlapping terms
        train_terms = set(train_cat['name'])
        valid_terms = set(valid_cat['name'])
        common_terms = train_terms & valid_terms

        # Prepare plotting data
        all_terms = sorted(train_terms | valid_terms)[:20]  # Limit for visibility

        train_pvals = []
        valid_pvals = []

        for term in all_terms:
            # Training p-value
            train_match = train_cat[train_cat['name'] == term]
            train_p = train_match['p_value'].iloc[0] if not train_match.empty else 1.0
            train_pvals.append(-np.log10(train_p))

            # Validation p-value
            valid_match = valid_cat[valid_cat['name'] == term]
            valid_p = valid_match['p_value'].iloc[0] if not valid_match.empty else 1.0
            valid_pvals.append(-np.log10(valid_p))

        # Create comparison plot
        x = np.arange(len(all_terms))
        width = 0.35

        bars1 = ax.bar(x - width/2, train_pvals, width,
                      label='Training', alpha=0.8, color='skyblue', edgecolor='black')
        bars2 = ax.bar(x + width/2, valid_pvals, width,
                      label='Validation', alpha=0.8, color='lightcoral', edgecolor='black')

        # Customize
        ax.set_xticks(x)
        ax.set_xticklabels([term[:30] + '...' if len(term) > 30 else term
                           for term in all_terms], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('-log₁₀(p-value)', fontsize=11)
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Add significance threshold
        ax.axhline(-np.log10(0.05), color='red', linestyle='--', alpha=0.7)

        # Highlight common terms
        for j, term in enumerate(all_terms):
            if term in common_terms:
                rect = Rectangle((j-0.4, 0), 0.8, max(max(train_pvals), max(valid_pvals)),
                               facecolor='yellow', alpha=0.2, zorder=0)
                ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig(output_dir / "go_training_vs_validation.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Print overlap statistics
    print(f"\n=== Training vs Validation GO Overlap ===")
    for gene_set, go_cat, title in comparisons:
        train_df = train_results.get(gene_set, pd.DataFrame())
        valid_df = valid_results.get(gene_set, pd.DataFrame())

        if not train_df.empty and not valid_df.empty:
            train_terms = set(train_df[train_df['source'] == go_cat]['name'])
            valid_terms = set(valid_df[valid_df['source'] == go_cat]['name'])
            overlap = len(train_terms & valid_terms)
            total = len(train_terms | valid_terms)
            if total > 0:
                print(f"{title}: {overlap}/{total} terms overlap ({overlap/total*100:.1f}%)")

def plot_go_network(results_dict, output_dir, gene_set='increased', max_nodes=30):
    """Create network plot of GO term relationships."""

    if gene_set not in results_dict or results_dict[gene_set].empty:
        print(f"No data for {gene_set} gene set")
        return

    df = results_dict[gene_set]

    # Get top terms
    top_terms = df.head(max_nodes)

    # Create network
    G = nx.Graph()

    # Add nodes (GO terms)
    for _, row in top_terms.iterrows():
        G.add_node(row['name'],
                  size=row['intersection_size'],
                  pval=row['p_value'],
                  source=row['source'])

    # Add edges based on gene overlap (simplified)
    # In a real implementation, you'd use GO hierarchy or gene overlap
    nodes = list(G.nodes())
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes[i+1:], i+1):
            # Simple rule: connect terms from same source with some probability
            row1 = top_terms[top_terms['name'] == node1].iloc[0]
            row2 = top_terms[top_terms['name'] == node2].iloc[0]
            if row1['source'] == row2['source']:
                G.add_edge(node1, node2, weight=0.5)

    # Plot network
    fig, ax = plt.subplots(figsize=(16, 12))

    # Position nodes
    pos = nx.spring_layout(G, k=3, iterations=50)

    # Color by source
    colors = {'GO:CC': '#E31A1C', 'GO:BP': '#1F78B4', 'GO:MF': '#33A02C', 'KEGG': '#FF7F00'}
    node_colors = [colors.get(G.nodes[node]['source'], 'gray') for node in G.nodes()]

    # Size by gene count
    node_sizes = [G.nodes[node]['size'] * 50 for node in G.nodes()]

    # Draw network
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                          alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)

    # Add labels (shortened)
    labels = {node: node[:20] + '...' if len(node) > 20 else node
             for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)

    ax.set_title(f'GO Term Network - {gene_set.title()} Genes',
                fontsize=16, fontweight='bold')
    ax.axis('off')

    # Add legend
    legend_elements = [plt.scatter([], [], c=color, s=100, alpha=0.8, label=source)
                      for source, color in colors.items()]
    ax.legend(handles=legend_elements, title='GO Source', loc='upper right')

    plt.tight_layout()
    plt.savefig(output_dir / f"go_network_{gene_set}.png", dpi=300, bbox_inches='tight')
    plt.show()

def create_go_plots(train_dir=None, valid_dir=None, output_dir=None):
    """Main function to create all GO plots."""

    if output_dir is None:
        output_dir = Path("notes/experiments/go_analysis_plots")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default directories
    if train_dir is None:
        train_dir = "notes/experiments/delta_analysis/go_enrichment"
    if valid_dir is None:
        valid_dir = "notes/experiments/delta_analysis/valid_set/go_enrichment"

    # Load results
    train_results = load_go_results(train_dir) if Path(train_dir).exists() else {}
    valid_results = load_go_results(valid_dir) if Path(valid_dir).exists() else {}

    print(f"Loaded training results: {list(train_results.keys())}")
    print(f"Loaded validation results: {list(valid_results.keys())}")

    # Create plots
    if train_results:
        print("\n1. Creating GO bar plots (training)...")
        plot_go_barplot(train_results, output_dir)

        print("\n2. Creating GO dot plots (training)...")
        plot_go_dotplot(train_results, output_dir)

        print("\n3. Creating GO network plots (training)...")
        plot_go_network(train_results, output_dir, 'increased')

    # Comparison plots if both datasets exist
    if train_results and valid_results:
        print("\n4. Creating training vs validation comparison...")
        plot_go_comparison(train_results, valid_results, output_dir)

    print(f"\n✓ GO analysis plots saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create GO analysis visualizations")
    parser.add_argument("--train_dir", default=None, help="Training GO results directory")
    parser.add_argument("--valid_dir", default=None, help="Validation GO results directory")
    parser.add_argument("--output_dir", default=None, help="Output directory for plots")

    args = parser.parse_args()
    create_go_plots(args.train_dir, args.valid_dir, args.output_dir)
