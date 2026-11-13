"""Find nuclear-enriched lncRNAs from ground truth cell-vs-nuclei expression.

Approach:
1. Compute mean expression per gene for cells vs nuclei (log1p, n3A filtered)
2. Identify genes with large negative delta (nuclear-enriched = higher in nuclei)
3. Flag likely lncRNAs by name pattern (Gm*, *os, *Rik, known lncRNAs)
4. Cross-reference with known nuclear lncRNA lists

Output: CSV with gene, cell_mean, nuc_mean, delta, is_lncrna, source
"""
import os, sys
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_ROOT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi"

# Known nuclear-retained lncRNAs (curated from literature)
KNOWN_NUCLEAR_LNCRNAS = {
    'Malat1', 'Neat1', 'Xist', 'Meg3', 'Snhg11', 'Kcnq1ot1',
    'Gomafu', 'Miat',  # same gene, neuronal nuclear lncRNA
    'Firre', 'Tsix',
    'Fendrr', 'Airn', 'Rian',
    'Mirg',  # maternally expressed, Dlk1-Dio3 locus with Meg3
    'Pvt1', 'Gas5',
    'Snhg1', 'Snhg3', 'Snhg4', 'Snhg5', 'Snhg6', 'Snhg7', 'Snhg8',
    'Snhg9', 'Snhg10', 'Snhg12', 'Snhg14', 'Snhg15', 'Snhg16',
    'Rmrp', 'Rpph1',  # nuclear RNA processing
}


def is_likely_lncrna(gene_name):
    """Heuristic: flag likely lncRNAs/predicted genes by naming pattern."""
    name = gene_name
    # Gm* predicted genes (many are lncRNAs)
    if name.startswith('Gm') and name[2:].isdigit():
        return True, 'Gm_predicted'
    # *Rik RIKEN clones (many are lncRNAs)
    if name.endswith('Rik'):
        return True, 'RIKEN'
    # Antisense transcripts (*os suffix)
    if name.endswith('os') and len(name) > 3:
        return True, 'antisense'
    # Known nuclear lncRNAs
    if name in KNOWN_NUCLEAR_LNCRNAS:
        return True, 'known_nuclear'
    # Snhg* family
    if name.startswith('Snhg'):
        return True, 'Snhg_family'
    return False, 'protein_coding'


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene_list", default="data/training_gene_list_v3.csv")
    parser.add_argument("--output_dir", default="notes/experiments/nuclear_lncrna_analysis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    gene_list = pd.read_csv(args.gene_list, header=None)[0].tolist()
    common_genes = pd.read_csv("data/common_genes.csv", header=0)["common_genes"].tolist()
    common_ct = pd.read_csv("data/common_cell_types.csv", header=0)["common_cell_types"].tolist()

    means = {}
    for name, ddir, batch in [("cell", "10X_cells_v2_v3_AIBS", "c3A"),
                               ("nucleus", "10X_nuclei_v2_v3_AIBS", "n3A")]:
        adata = ad.read_h5ad(os.path.join(DATA_ROOT, ddir, "data.h5ad"))
        if 'meta_batch_id' not in adata.obs:
            adata.obs['meta_batch_id'] = adata.obs['batch_id'].apply(lambda x: x.split('.')[0])
        adata = adata[adata.obs['meta_batch_id'].str.endswith(batch)].copy()
        adata = adata[:, adata.var['gene_names'].isin(common_genes)].copy()
        adata = adata[adata.obs['cell_type'].isin(common_ct)].copy()
        adata = adata[:, adata.var['gene_names'].isin(gene_list)].copy()
        adata = adata[:, adata.var['gene_names'].argsort()].copy()

        X = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
        X = np.log1p(X).astype(np.float32)
        gene_names = list(adata.var['gene_names'].values)
        means[name] = pd.Series(X.mean(axis=0), index=gene_names)
        print(f"{name}: {adata.shape[0]} cells, {len(gene_names)} genes")
        del adata, X

    # Build results table
    all_genes = means['cell'].index
    rows = []
    for g in all_genes:
        cell_m = means['cell'][g]
        nuc_m = means['nucleus'][g]
        delta = cell_m - nuc_m
        is_lnc, source = is_likely_lncrna(g)
        rows.append({
            'gene': g,
            'cell_mean': round(float(cell_m), 5),
            'nuc_mean': round(float(nuc_m), 5),
            'delta': round(float(delta), 5),
            'abs_delta': round(abs(float(delta)), 5),
            'is_lncrna': is_lnc,
            'lncrna_source': source,
        })

    df = pd.DataFrame(rows).sort_values('delta')
    df.to_csv(os.path.join(args.output_dir, "all_genes_delta.csv"), index=False)

    # Nuclear-enriched genes (negative delta = higher in nuclei)
    nuclear_enriched = df[df['delta'] < -0.1].copy()
    print(f"\nNuclear-enriched genes (delta < -0.1): {len(nuclear_enriched)}")

    # Nuclear-enriched lncRNAs
    nuc_lncrnas = nuclear_enriched[nuclear_enriched['is_lncrna']].copy()
    print(f"Nuclear-enriched lncRNAs: {len(nuc_lncrnas)}")
    nuc_lncrnas.to_csv(os.path.join(args.output_dir, "nuclear_enriched_lncrnas.csv"), index=False)

    # Summary stats
    print(f"\n{'='*70}")
    print("Top 30 nuclear-enriched genes (most negative delta):")
    print(f"{'Gene':<18} {'cell_mean':>10} {'nuc_mean':>10} {'delta':>10} {'lncRNA?':>8} {'source':<15}")
    print("-" * 72)
    for _, row in df.head(30).iterrows():
        lnc = "YES" if row['is_lncrna'] else ""
        print(f"{row['gene']:<18} {row['cell_mean']:>10.4f} {row['nuc_mean']:>10.4f} "
              f"{row['delta']:>10.4f} {lnc:>8} {row['lncrna_source']:<15}")

    print(f"\n{'='*70}")
    print("Top 30 nuclear-enriched lncRNAs:")
    print(f"{'Gene':<18} {'cell_mean':>10} {'nuc_mean':>10} {'delta':>10} {'source':<15}")
    print("-" * 68)
    for _, row in nuc_lncrnas.head(30).iterrows():
        print(f"{row['gene']:<18} {row['cell_mean']:>10.4f} {row['nuc_mean']:>10.4f} "
              f"{row['delta']:>10.4f} {row['lncrna_source']:<15}")

    # Proposed expanded list for eval metric
    # Take nuclear-enriched lncRNAs with delta < -0.2 (strong signal)
    strong = nuc_lncrnas[nuc_lncrnas['delta'] < -0.2]
    proposed = sorted(strong['gene'].tolist())
    print(f"\n{'='*70}")
    print(f"Proposed expanded nuclear lncRNA list ({len(proposed)} genes, delta < -0.2):")
    print(proposed)

    with open(os.path.join(args.output_dir, "proposed_nuclear_lncrna_list.txt"), 'w') as f:
        for g in proposed:
            f.write(g + '\n')

    # Also check: which known nuclear lncRNAs are in the gene list and what's their delta?
    print(f"\n{'='*70}")
    print("Known nuclear lncRNAs — status in our data:")
    print(f"{'Gene':<18} {'in_genelist':>12} {'delta':>10}")
    print("-" * 42)
    for g in sorted(KNOWN_NUCLEAR_LNCRNAS):
        in_list = g in set(all_genes)
        if in_list:
            delta = float(df[df['gene'] == g]['delta'].values[0])
            print(f"{g:<18} {'YES':>12} {delta:>10.4f}")
        else:
            print(f"{g:<18} {'NO':>12} {'—':>10}")


if __name__ == "__main__":
    main()
