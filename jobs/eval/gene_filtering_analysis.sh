#!/bin/bash
#SBATCH --job-name=gene_filter_analysis
#SBATCH --output=jobs/logs/slurm-%j_gene_filter.out
#SBATCH --error=jobs/logs/slurm-%j_gene_filter.err
#SBATCH --partition=pi_gerstein
#SBATCH --mem=100G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00

source ~/.bashrc
conda activate impute2

cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

python << 'PYEOF'
import numpy as np
import pandas as pd
import anndata as ad
import os
import gc

output_dir = 'notes/gene_filtering_analysis'
os.makedirs(output_dir, exist_ok=True)

# =====================================================================
# 1. Load data (v3 only)
# =====================================================================
cell_path = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi/10X_cells_v2_v3_AIBS/data.h5ad"
nuc_path = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi/10X_nuclei_v2_v3_AIBS/data.h5ad"

print("Loading cell data...")
cell_adata = ad.read_h5ad(cell_path)
cell_adata.obs['meta_batch_id'] = cell_adata.obs['batch_id'].apply(lambda x: x.split('.')[0])
cell_v3 = cell_adata[cell_adata.obs['meta_batch_id'].str.endswith('c3A')].copy()
del cell_adata; gc.collect()
print(f"Cell v3: {cell_v3.shape}")

print("Loading nuclei data...")
nuc_adata = ad.read_h5ad(nuc_path)
nuc_adata.obs['meta_batch_id'] = nuc_adata.obs['batch_id'].apply(lambda x: x.split('.')[0])
nuc_v3 = nuc_adata[nuc_adata.obs['meta_batch_id'].str.endswith('n3A')].copy()
del nuc_adata; gc.collect()
print(f"Nuclei v3: {nuc_v3.shape}")

# Common genes (must be present in both modalities)
common_genes_df = pd.read_csv('data/common_genes.csv', header=0)
common_genes = set(common_genes_df['common_genes'].tolist())
cell_v3 = cell_v3[:, cell_v3.var['gene_names'].isin(common_genes)].copy()
nuc_v3 = nuc_v3[:, nuc_v3.var['gene_names'].isin(common_genes)].copy()
print(f"After common gene filter - Cell: {cell_v3.shape}, Nuclei: {nuc_v3.shape}")

# MitoCarta genes (always excluded)
mitocarta = set(pd.read_csv('data/Mouse.MitoCarta.csv', header=0)['Symbol'].tolist())
print(f"MitoCarta genes: {len(mitocarta)}")

# Config B for comparison
config_b = set(pd.read_csv('data/training_gene_list_v3.csv', header=None)[0].tolist())
print(f"Config B genes: {len(config_b)}")

# =====================================================================
# 2. Marker genes (Allen + PsychENCODE, mouse orthologs)
# =====================================================================
markers = {
    'Slc17a7': 'Excitatory (pan)', 'Slc17a6': 'Excitatory (pan)',
    'Gad1': 'Inhibitory (pan)', 'Gad2': 'Inhibitory (pan)', 'Slc32a1': 'Inhibitory (pan)',
    'Snap25': 'Pan-neuronal', 'Syt1': 'Pan-neuronal', 'Rbfox3': 'Pan-neuronal',
    'Cux2': 'L2/3 IT', 'Cux1': 'L2/3 IT',
    'Rorb': 'L5 IT', 'Fezf2': 'L5 ET', 'Bcl11b': 'L5 ET', 'Crym': 'L5 ET',
    'Tshz2': 'L5/6 NP', 'Syt6': 'L6 CT', 'Tle4': 'L6 CT/L6b', 'Foxp2': 'L6 CT/L6b',
    'Ctgf': 'L6b',
    'Pvalb': 'Pvalb', 'Sst': 'Sst', 'Vip': 'Vip', 'Lamp5': 'Lamp5', 'Sncg': 'Sncg',
    'Cck': 'Sncg', 'Npy': 'Sst', 'Nos1': 'Sst/Lamp5', 'Calb2': 'Vip', 'Ndnf': 'Lamp5',
    'Aqp4': 'Astrocyte', 'Gfap': 'Astrocyte', 'Aldh1l1': 'Astrocyte',
    'Slc1a3': 'Astrocyte', 'Sox9': 'Astrocyte', 'Aldoc': 'Astrocyte', 'Apoe': 'Astrocyte',
    'Mog': 'Oligodendrocyte', 'Mbp': 'Oligodendrocyte', 'Plp1': 'Oligodendrocyte',
    'Mobp': 'Oligodendrocyte', 'Mag': 'Oligodendrocyte', 'Cnp': 'Oligodendrocyte',
    'Pdgfra': 'OPC', 'Cspg4': 'OPC', 'Vcan': 'OPC', 'Olig1': 'OPC', 'Olig2': 'OPC',
    'Flt1': 'Endothelial', 'Cldn5': 'Endothelial', 'Pecam1': 'Endothelial',
    'Cx3cr1': 'Microglia', 'C1qa': 'Microglia', 'C1qb': 'Microglia', 'C1qc': 'Microglia',
    'Ctss': 'Microglia', 'P2ry12': 'Microglia', 'Tmem119': 'Microglia',
    'Csf1r': 'Microglia', 'Hexb': 'Microglia',
}
marker_set = set(markers.keys())

# =====================================================================
# 3. Compute per-cell-type detection rates (separate for each modality)
# =====================================================================
print("\nComputing per-cell-type detection rates...")

X_cell = cell_v3.X.toarray() if hasattr(cell_v3.X, 'toarray') else cell_v3.X
gene_names_cell = cell_v3.var['gene_names'].values
cell_types_cell = cell_v3.obs['cell_type'].values
unique_ct_cell = sorted(set(cell_types_cell))

X_nuc = nuc_v3.X.toarray() if hasattr(nuc_v3.X, 'toarray') else nuc_v3.X
gene_names_nuc = nuc_v3.var['gene_names'].values
cell_types_nuc = nuc_v3.obs['cell_type'].values
unique_ct_nuc = sorted(set(cell_types_nuc))

def compute_per_celltype_det(X, gene_names, cell_types, unique_cts):
    """Returns DataFrame: genes x cell_types, values = detection rate."""
    results = {}
    for ct in unique_cts:
        mask = cell_types == ct
        n_cells = mask.sum()
        if n_cells == 0:
            continue
        det = (X[mask] > 0).sum(axis=0) / n_cells
        # Handle sparse result
        det = np.asarray(det).flatten()
        results[ct] = det
    return pd.DataFrame(results, index=gene_names)

print("  Cell modality...")
cell_det_per_ct = compute_per_celltype_det(X_cell, gene_names_cell, cell_types_cell, unique_ct_cell)
print(f"  Cell: {cell_det_per_ct.shape} (genes x cell_types)")

print("  Nuclei modality...")
nuc_det_per_ct = compute_per_celltype_det(X_nuc, gene_names_nuc, cell_types_nuc, unique_ct_nuc)
print(f"  Nuclei: {nuc_det_per_ct.shape} (genes x cell_types)")

# Save full per-celltype detection tables
cell_det_per_ct.to_csv(f'{output_dir}/cell_per_celltype_detection.csv')
nuc_det_per_ct.to_csv(f'{output_dir}/nuclei_per_celltype_detection.csv')

# Global detection rates
cell_global_det = pd.Series((X_cell > 0).sum(axis=0) / X_cell.shape[0], index=gene_names_cell)
nuc_global_det = pd.Series((X_nuc > 0).sum(axis=0) / X_nuc.shape[0], index=gene_names_nuc)

# Max per-celltype detection rate across any cell type (separate per modality)
cell_max_ct_det = cell_det_per_ct.max(axis=1)
nuc_max_ct_det = nuc_det_per_ct.max(axis=1)

# =====================================================================
# 4. Strategy (c2): keep gene if max per-celltype det > threshold
#    in EITHER cell or nuclei modality. Exclude MitoCarta.
# =====================================================================
print("\n" + "="*80)
print("FILTERING STRATEGY COMPARISON")
print("="*80)

# Align gene names
all_genes = sorted(set(gene_names_cell) & set(gene_names_nuc) & common_genes)

# Build master table
master = pd.DataFrame({'gene': all_genes}).set_index('gene')
master['cell_global_det'] = master.index.map(lambda g: cell_global_det.get(g, 0))
master['nuc_global_det'] = master.index.map(lambda g: nuc_global_det.get(g, 0))
master['cell_max_ct_det'] = master.index.map(lambda g: cell_max_ct_det.get(g, 0))
master['nuc_max_ct_det'] = master.index.map(lambda g: nuc_max_ct_det.get(g, 0))
master['either_max_ct_det'] = master[['cell_max_ct_det', 'nuc_max_ct_det']].max(axis=1)
master['min_global_det'] = master[['cell_global_det', 'nuc_global_det']].min(axis=1)
master['is_mitocarta'] = master.index.isin(mitocarta)
master['in_config_b'] = master.index.isin(config_b)
master['is_marker'] = master.index.isin(marker_set)

# Cell type where max detection occurs
master['cell_best_ct'] = master.index.map(
    lambda g: cell_det_per_ct.loc[g].idxmax() if g in cell_det_per_ct.index else ''
)
master['nuc_best_ct'] = master.index.map(
    lambda g: nuc_det_per_ct.loc[g].idxmax() if g in nuc_det_per_ct.index else ''
)

master.to_csv(f'{output_dir}/master_gene_table.csv')

# =====================================================================
# 5. Compare filtering strategies at multiple thresholds
# =====================================================================
strategies = {}

# Config B (current): min_global_det >= 0.05, exclude mitocarta
# Note: Config B was precomputed; here we replicate the logic
strategies['ConfigB_replica'] = (master['min_global_det'] >= 0.05) & (~master['is_mitocarta'])

# Strategy C2: per-cell-type det >= threshold in either modality, exclude mitocarta
for thresh in [0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20]:
    label = f'PerCT_{thresh*100:.0f}pct'
    strategies[label] = (master['either_max_ct_det'] >= thresh) & (~master['is_mitocarta'])

# Also: global det >= 1% (very permissive)
strategies['Global_1pct'] = (master['min_global_det'] >= 0.01) & (~master['is_mitocarta'])

# Also: global det >= 2%
strategies['Global_2pct'] = (master['min_global_det'] >= 0.02) & (~master['is_mitocarta'])

# =====================================================================
# 6. Compute metrics for each strategy
# =====================================================================

# Variance (log1p space)
X_cell_log = np.log1p(X_cell)
gene_var_cell = pd.Series(X_cell_log.var(axis=0), index=gene_names_cell)
total_var_cell = gene_var_cell.sum()

X_nuc_log = np.log1p(X_nuc)
gene_var_nuc = pd.Series(X_nuc_log.var(axis=0), index=gene_names_nuc)
total_var_nuc = gene_var_nuc.sum()

print(f"\n{'Strategy':<20} {'Genes':>7} {'Markers':>8} {'Miss':>5} "
      f"{'CellVar%':>9} {'NucVar%':>9} {'New vs B':>9} {'Drop vs B':>10}")
print("-" * 90)

comparison_rows = []
for name, mask in sorted(strategies.items()):
    genes = set(master.index[mask])
    n_genes = len(genes)
    markers_in = marker_set & genes & set(master.index)  # markers that exist in data
    markers_in_data = marker_set & set(master.index)
    n_markers = len(markers_in)
    n_markers_missing = len(markers_in_data) - n_markers

    # Variance captured
    cell_var_frac = gene_var_cell[gene_var_cell.index.isin(genes)].sum() / total_var_cell
    nuc_var_frac = gene_var_nuc[gene_var_nuc.index.isin(genes)].sum() / total_var_nuc

    # Comparison to Config B
    new_vs_b = len(genes - config_b)
    dropped_vs_b = len(config_b - genes)

    print(f"{name:<20} {n_genes:>7} {n_markers:>5}/{len(markers_in_data):<2} {n_markers_missing:>5} "
          f"{cell_var_frac*100:>8.1f}% {nuc_var_frac*100:>8.1f}% {new_vs_b:>+9} {dropped_vs_b:>10}")

    # Which markers are missing?
    missing = sorted(markers_in_data - markers_in)

    comparison_rows.append({
        'strategy': name,
        'n_genes': n_genes,
        'n_markers_retained': n_markers,
        'n_markers_total': len(markers_in_data),
        'n_markers_missing': n_markers_missing,
        'missing_markers': '; '.join(missing),
        'cell_var_fraction': cell_var_frac,
        'nuc_var_fraction': nuc_var_frac,
        'new_vs_configb': new_vs_b,
        'dropped_vs_configb': dropped_vs_b,
    })

comp_df = pd.DataFrame(comparison_rows)
comp_df.to_csv(f'{output_dir}/strategy_comparison.csv', index=False)

# =====================================================================
# 7. Print missing markers per strategy
# =====================================================================
print("\n" + "="*80)
print("MISSING MARKERS PER STRATEGY")
print("="*80)
for _, row in comp_df.iterrows():
    if row['n_markers_missing'] > 0:
        print(f"\n{row['strategy']} ({row['n_markers_missing']} missing):")
        for m in row['missing_markers'].split('; '):
            if m:
                ct = markers.get(m, '?')
                print(f"  {m:<12} ({ct})")

# =====================================================================
# 8. Detailed look at recommended strategy: PerCT_5pct
# =====================================================================
rec = 'PerCT_5pct'
rec_mask = strategies[rec]
rec_genes = set(master.index[rec_mask])

print("\n" + "="*80)
print(f"RECOMMENDED STRATEGY: {rec}")
print("="*80)

added = sorted(rec_genes - config_b)
removed = sorted(config_b - rec_genes)

print(f"\nGenes added vs Config B: {len(added)}")
print(f"Genes removed vs Config B: {len(removed)}")

# Show added genes that are markers
added_markers = [g for g in added if g in marker_set]
print(f"\nMarker genes ADDED back: {len(added_markers)}")
for g in sorted(added_markers):
    ct = markers.get(g, '?')
    cell_d = master.loc[g, 'cell_max_ct_det'] * 100
    nuc_d = master.loc[g, 'nuc_max_ct_det'] * 100
    cell_ct = master.loc[g, 'cell_best_ct']
    nuc_ct = master.loc[g, 'nuc_best_ct']
    print(f"  {g:<12} ({ct:<20}) cell: {cell_d:>5.1f}% in {cell_ct:<15} nuc: {nuc_d:>5.1f}% in {nuc_ct}")

# Save the gene list
rec_gene_list = sorted(rec_genes)
pd.Series(rec_gene_list).to_csv(f'{output_dir}/gene_list_perct_5pct.csv', index=False, header=False)
print(f"\nGene list saved: {output_dir}/gene_list_perct_5pct.csv ({len(rec_gene_list)} genes)")

# Also save PerCT_10pct and PerCT_20pct for comparison
for t in ['PerCT_10pct', 'PerCT_20pct']:
    t_genes = sorted(master.index[strategies[t]])
    pd.Series(t_genes).to_csv(f'{output_dir}/gene_list_{t.lower()}.csv', index=False, header=False)
    print(f"Gene list saved: {output_dir}/gene_list_{t.lower()}.csv ({len(t_genes)} genes)")

# =====================================================================
# 9. Marker gene detection rate table (for presentation)
# =====================================================================
print("\n" + "="*80)
print("MARKER GENE DETECTION RATES (for presentation)")
print("="*80)

marker_detail = []
for gene, ct in sorted(markers.items(), key=lambda x: (x[1], x[0])):
    if gene not in master.index:
        marker_detail.append({
            'gene': gene, 'cell_type': ct,
            'cell_global_det': np.nan, 'nuc_global_det': np.nan,
            'cell_max_ct_det': np.nan, 'nuc_max_ct_det': np.nan,
            'cell_best_ct': '', 'nuc_best_ct': '',
            'in_config_b': False, 'in_perct5': False,
            'in_common_genes': False,
        })
        continue

    r = master.loc[gene]
    marker_detail.append({
        'gene': gene, 'cell_type': ct,
        'cell_global_det': r['cell_global_det'],
        'nuc_global_det': r['nuc_global_det'],
        'cell_max_ct_det': r['cell_max_ct_det'],
        'nuc_max_ct_det': r['nuc_max_ct_det'],
        'cell_best_ct': r['cell_best_ct'],
        'nuc_best_ct': r['nuc_best_ct'],
        'in_config_b': r['in_config_b'],
        'in_perct5': gene in rec_genes,
        'in_common_genes': True,
    })

marker_detail_df = pd.DataFrame(marker_detail)
marker_detail_df.to_csv(f'{output_dir}/marker_gene_detail.csv', index=False)

print(f"\n{'Gene':<12} {'CellType':<20} {'GlobCell%':>10} {'GlobNuc%':>10} "
      f"{'MaxCTcell%':>11} {'MaxCTnuc%':>10} {'ConfigB':>8} {'PerCT5':>7}")
print("-" * 95)
for _, r in marker_detail_df.iterrows():
    if pd.isna(r['cell_global_det']):
        print(f"{r['gene']:<12} {r['cell_type']:<20}  NOT IN COMMON GENES")
        continue
    print(f"{r['gene']:<12} {r['cell_type']:<20} {r['cell_global_det']*100:>9.2f}% {r['nuc_global_det']*100:>9.2f}% "
          f"{r['cell_max_ct_det']*100:>10.1f}% {r['nuc_max_ct_det']*100:>9.1f}% "
          f"{'YES' if r['in_config_b'] else 'NO':>8} {'YES' if r['in_perct5'] else 'NO':>7}")

print("\n\nDone! All results saved to:", output_dir)
PYEOF
