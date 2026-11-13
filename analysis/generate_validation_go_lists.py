#!/usr/bin/env python3
"""Generate gene lists for GO analysis from validation delta results."""

import pandas as pd
import numpy as np
from pathlib import Path
import os

def generate_validation_gene_lists():
    """Generate top increased/decreased gene lists from validation delta analysis."""

    # Input and output directories
    valid_tables_dir = Path("notes/experiments/delta_analysis/valid_set/tables")
    go_output_dir = Path("notes/experiments/delta_analysis/valid_set/go_enrichment")
    go_output_dir.mkdir(exist_ok=True)

    # Find all delta CSV files
    delta_files = list(valid_tables_dir.glob("*_delta.csv"))
    print(f"Found {len(delta_files)} delta files")

    # Collect all genes and their average deltas
    all_gene_deltas = {}

    for delta_file in delta_files:
        print(f"Processing {delta_file.name}")
        df = pd.read_csv(delta_file)

        for _, row in df.iterrows():
            gene = row['gene']
            delta = row['median_delta']
            n_cells = row['n_cells']

            # Weight by number of cells for averaging
            if gene in all_gene_deltas:
                old_delta, old_weight = all_gene_deltas[gene]
                total_weight = old_weight + n_cells
                avg_delta = (old_delta * old_weight + delta * n_cells) / total_weight
                all_gene_deltas[gene] = (avg_delta, total_weight)
            else:
                all_gene_deltas[gene] = (delta, n_cells)

    # Convert to sorted list
    gene_delta_list = [(gene, delta_weight[0]) for gene, delta_weight in all_gene_deltas.items()]
    gene_delta_list.sort(key=lambda x: x[1], reverse=True)

    # Extract top/bottom genes
    n_genes = 100

    top_increased = [gene for gene, delta in gene_delta_list[:n_genes]]
    top_decreased = [gene for gene, delta in gene_delta_list[-n_genes:]]

    # All genes as background
    all_genes = [gene for gene, delta in gene_delta_list]

    # Save gene lists
    with open(go_output_dir / "top100_increased_genes.txt", 'w') as f:
        for gene in top_increased:
            f.write(f"{gene}\n")

    with open(go_output_dir / "top100_decreased_genes.txt", 'w') as f:
        for gene in top_decreased:
            f.write(f"{gene}\n")

    with open(go_output_dir / "all_genes_background.txt", 'w') as f:
        for gene in all_genes:
            f.write(f"{gene}\n")

    print(f"\n✓ Generated gene lists in {go_output_dir}:")
    print(f"  - Top {n_genes} increased genes")
    print(f"  - Top {n_genes} decreased genes")
    print(f"  - Background: {len(all_genes)} total genes")

    print(f"\nTop 10 increased genes (validation):")
    for i, gene in enumerate(top_increased[:10], 1):
        delta = next(d for g, d in gene_delta_list if g == gene)
        print(f"  {i:2d}. {gene:<15} (+{delta:.2f})")

    print(f"\nTop 10 decreased genes (validation):")
    for i, gene in enumerate(top_decreased[-10:], 1):
        delta = next(d for g, d in gene_delta_list if g == gene)
        print(f"  {i:2d}. {gene:<15} ({delta:.2f})")

    return go_output_dir

if __name__ == "__main__":
    generate_validation_gene_lists()
