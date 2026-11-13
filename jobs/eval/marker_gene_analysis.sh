#!/bin/bash
#SBATCH --job-name=marker_gene_analysis
#SBATCH --output=jobs/logs/slurm-%j_marker_gene.out
#SBATCH --error=jobs/logs/slurm-%j_marker_gene.err
#SBATCH --partition=pi_gerstein
#SBATCH --mem=200G
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
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from umap import UMAP

output_dir = 'notes/marker_gene_analysis'
os.makedirs(output_dir, exist_ok=True)

# =====================================================================
# 1. Load raw data (v3 only)
# =====================================================================
cell_path = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi/10X_cells_v2_v3_AIBS/data.h5ad"
nuc_path = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi/10X_nuclei_v2_v3_AIBS/data.h5ad"

print("Loading cell data...")
cell_adata = ad.read_h5ad(cell_path)
cell_adata.obs['meta_batch_id'] = cell_adata.obs['batch_id'].apply(lambda x: x.split('.')[0])
cell_v3 = cell_adata[cell_adata.obs['meta_batch_id'].str.endswith('c3A')].copy()
del cell_adata
print(f"Cell v3: {cell_v3.shape}")

print("Loading nuclei data...")
nuc_adata = ad.read_h5ad(nuc_path)
nuc_adata.obs['meta_batch_id'] = nuc_adata.obs['batch_id'].apply(lambda x: x.split('.')[0])
nuc_v3 = nuc_adata[nuc_adata.obs['meta_batch_id'].str.endswith('n3A')].copy()
del nuc_adata
print(f"Nuclei v3: {nuc_v3.shape}")

# Common genes
common_genes_df = pd.read_csv('data/common_genes.csv', header=0)
common_genes = common_genes_df['common_genes'].tolist()
cell_v3 = cell_v3[:, cell_v3.var['gene_names'].isin(common_genes)].copy()
nuc_v3 = nuc_v3[:, nuc_v3.var['gene_names'].isin(common_genes)].copy()
print(f"After common gene filter - Cell: {cell_v3.shape}, Nuclei: {nuc_v3.shape}")

# Config B gene list
config_b = set(pd.read_csv('data/training_gene_list_v3.csv', header=None)[0].tolist())

# =====================================================================
# 2. Marker gene list (Allen + PsychENCODE, mouse orthologs)
# =====================================================================
markers = {
    # Pan-neuronal
    'Slc17a7': 'Excitatory (pan)', 'Slc17a6': 'Excitatory (pan)',
    'Gad1': 'Inhibitory (pan)', 'Gad2': 'Inhibitory (pan)', 'Slc32a1': 'Inhibitory (pan)',
    'Snap25': 'Pan-neuronal', 'Syt1': 'Pan-neuronal', 'Rbfox3': 'Pan-neuronal',
    # Excitatory subtypes
    'Cux2': 'L2/3 IT', 'Cux1': 'L2/3 IT',
    'Rorb': 'L5 IT', 'Fezf2': 'L5 ET', 'Bcl11b': 'L5 ET', 'Crym': 'L5 ET',
    'Tshz2': 'L5/6 NP', 'Syt6': 'L6 CT', 'Tle4': 'L6 CT/L6b', 'Foxp2': 'L6 CT/L6b',
    'Ctgf': 'L6b',
    # Inhibitory subtypes
    'Pvalb': 'Pvalb', 'Sst': 'Sst', 'Vip': 'Vip', 'Lamp5': 'Lamp5', 'Sncg': 'Sncg',
    'Cck': 'Sncg', 'Npy': 'Sst', 'Nos1': 'Sst/Lamp5', 'Calb2': 'Vip', 'Ndnf': 'Lamp5',
    # Non-neuronal
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

# =====================================================================
# 3. Compute global detection rates
# =====================================================================
X_cell = cell_v3.X.toarray() if hasattr(cell_v3.X, 'toarray') else cell_v3.X
X_nuc = nuc_v3.X.toarray() if hasattr(nuc_v3.X, 'toarray') else nuc_v3.X

gene_names_cell = cell_v3.var['gene_names'].values
gene_names_nuc = nuc_v3.var['gene_names'].values

cell_det = pd.Series((X_cell > 0).sum(axis=0) / X_cell.shape[0], index=gene_names_cell)
nuc_det = pd.Series((X_nuc > 0).sum(axis=0) / X_nuc.shape[0], index=gene_names_nuc)

# Combine
common_in_both = sorted(set(gene_names_cell) & set(gene_names_nuc))
det_df = pd.DataFrame({
    'gene': common_in_both,
    'cell_det': [cell_det.get(g, 0) for g in common_in_both],
    'nuc_det': [nuc_det.get(g, 0) for g in common_in_both],
}).set_index('gene')
det_df['min_det'] = det_df[['cell_det', 'nuc_det']].min(axis=1)
det_df['max_det'] = det_df[['cell_det', 'nuc_det']].max(axis=1)
det_df['in_config_b'] = det_df.index.isin(config_b)

# =====================================================================
# 4. Marker gene detection rate table
# =====================================================================
print("\n" + "="*80)
print("MARKER GENE DETECTION RATES")
print("="*80)
print(f"{'Gene':<12} {'CellType':<20} {'CellDet%':>9} {'NucDet%':>9} {'MinDet%':>9} {'InConfigB':>10}")
print("-" * 75)

marker_rows = []
for gene, ct in sorted(markers.items(), key=lambda x: (x[1], x[0])):
    in_data = gene in det_df.index
    if in_data:
        cd = det_df.loc[gene, 'cell_det']
        nd = det_df.loc[gene, 'nuc_det']
        md = det_df.loc[gene, 'min_det']
        in_b = gene in config_b
        print(f"{gene:<12} {ct:<20} {cd*100:>8.2f}% {nd*100:>8.2f}% {md*100:>8.2f}% {'YES' if in_b else 'NO':>10}")
        marker_rows.append({'gene': gene, 'cell_type': ct, 'cell_det': cd, 'nuc_det': nd, 'min_det': md, 'in_config_b': in_b})
    else:
        print(f"{gene:<12} {ct:<20}  NOT IN COMMON GENES")
        marker_rows.append({'gene': gene, 'cell_type': ct, 'cell_det': np.nan, 'nuc_det': np.nan, 'min_det': np.nan, 'in_config_b': False})

marker_df = pd.DataFrame(marker_rows)
marker_df.to_csv(f'{output_dir}/marker_detection_rates.csv', index=False)

# =====================================================================
# 5. Per-cell-type detection rates for key missing markers
# =====================================================================
print("\n" + "="*80)
print("PER-CELL-TYPE DETECTION RATES FOR MISSING MARKERS")
print("="*80)

missing_markers = marker_df[~marker_df['in_config_b'] & marker_df['min_det'].notna()]['gene'].tolist()

ct_det_rows = []
for gene in missing_markers:
    if gene not in gene_names_cell:
        continue
    gene_idx = np.where(gene_names_cell == gene)[0]
    if len(gene_idx) == 0:
        continue
    gene_idx = gene_idx[0]

    print(f"\n{gene} ({markers[gene]}):")
    for ct in sorted(cell_v3.obs['cell_type'].unique()):
        ct_mask = cell_v3.obs['cell_type'].values == ct
        ct_data = X_cell[ct_mask, gene_idx]
        det = (ct_data > 0).sum() / ct_data.shape[0]
        n_cells = ct_mask.sum()
        if det > 0.01:
            print(f"  {ct:<20} {det*100:>6.1f}% ({n_cells} cells)")
        ct_det_rows.append({'gene': gene, 'cell_type': ct, 'det_rate': det, 'n_cells': n_cells})

ct_det_df = pd.DataFrame(ct_det_rows)
ct_det_df.to_csv(f'{output_dir}/marker_per_celltype_detection.csv', index=False)

# =====================================================================
# 6. Threshold analysis: how many genes at each threshold?
# =====================================================================
print("\n" + "="*80)
print("THRESHOLD ANALYSIS")
print("="*80)

# What threshold keeps all markers?
valid_markers = marker_df[marker_df['min_det'].notna() & ~marker_df['in_config_b']]
if len(valid_markers) > 0:
    lowest = valid_markers['min_det'].min()
    lowest_gene = valid_markers.loc[valid_markers['min_det'].idxmin(), 'gene']
    print(f"Lowest min-det among missing markers: {lowest*100:.3f}% ({lowest_gene})")

thresholds = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10]
thresh_rows = []
for thresh in thresholds:
    n_genes = (det_df['min_det'] >= thresh).sum()
    n_markers_missing_kept = (valid_markers['min_det'] >= thresh).sum() if len(valid_markers) > 0 else 0
    n_markers_missing_total = len(valid_markers)
    print(f"  min_det >= {thresh*100:>5.1f}%: {n_genes:>6} genes, "
          f"{n_markers_missing_kept}/{n_markers_missing_total} missing markers retained")
    thresh_rows.append({'threshold': thresh, 'n_genes': n_genes,
                        'markers_retained': n_markers_missing_kept,
                        'markers_total': n_markers_missing_total})

pd.DataFrame(thresh_rows).to_csv(f'{output_dir}/threshold_analysis.csv', index=False)

# =====================================================================
# 7. Variance analysis: what fraction of variance do Config B genes capture?
# =====================================================================
print("\n" + "="*80)
print("VARIANCE ANALYSIS")
print("="*80)

# Use log1p for variance calculation
X_cell_log = np.log1p(X_cell)
gene_var_cell = X_cell_log.var(axis=0)

total_var = gene_var_cell.sum()
config_b_mask = np.array([g in config_b for g in gene_names_cell])
config_b_var = gene_var_cell[config_b_mask].sum()
print(f"Total genes: {len(gene_names_cell)}")
print(f"Config B genes: {config_b_mask.sum()}")
print(f"Total variance (log1p): {total_var:.2f}")
print(f"Config B variance: {config_b_var:.2f}")
print(f"Fraction of variance captured: {config_b_var/total_var*100:.1f}%")

# Same for nuclei
X_nuc_log = np.log1p(X_nuc)
gene_var_nuc = X_nuc_log.var(axis=0)
total_var_nuc = gene_var_nuc.sum()
config_b_mask_nuc = np.array([g in config_b for g in gene_names_nuc])
config_b_var_nuc = gene_var_nuc[config_b_mask_nuc].sum()
print(f"\nNuclei:")
print(f"Total variance (log1p): {total_var_nuc:.2f}")
print(f"Config B variance: {config_b_var_nuc:.2f}")
print(f"Fraction of variance captured: {config_b_var_nuc/total_var_nuc*100:.1f}%")

# =====================================================================
# 8. UMAP of raw data with Config B genes
# =====================================================================
print("\n" + "="*80)
print("GENERATING UMAPs")
print("="*80)

# Get Config B gene indices
cell_gene_names = cell_v3.var['gene_names'].values
nuc_gene_names = nuc_v3.var['gene_names'].values
configb_cell_mask = np.array([g in config_b for g in cell_gene_names])
configb_nuc_mask = np.array([g in config_b for g in nuc_gene_names])

# Subsample
np.random.seed(42)
n_sub = min(10000, X_cell.shape[0])
cell_idx = np.random.choice(X_cell.shape[0], n_sub, replace=False)
nuc_idx = np.random.choice(X_nuc.shape[0], min(n_sub, X_nuc.shape[0]), replace=False)

# Get Config B data, log-normalize
X_cell_cb = X_cell[np.ix_(cell_idx, configb_cell_mask)]
X_nuc_cb = X_nuc[np.ix_(nuc_idx, configb_nuc_mask)]

# Normalize: divide by total, multiply by 1e4, log1p
X_cell_norm = np.log1p(X_cell_cb / (X_cell_cb.sum(axis=1, keepdims=True) + 1e-8) * 1e4)
X_nuc_norm = np.log1p(X_nuc_cb / (X_nuc_cb.sum(axis=1, keepdims=True) + 1e-8) * 1e4)

# Combine
X_combined = np.vstack([X_cell_norm, X_nuc_norm])
modality = np.array(['cell'] * len(cell_idx) + ['nuclei'] * len(nuc_idx))
ct_cell = cell_v3.obs['cell_type'].values[cell_idx]
ct_nuc = nuc_v3.obs['cell_type'].values[nuc_idx]
cell_types_combined = np.concatenate([ct_cell, ct_nuc])

# Scale, PCA, UMAP
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X_combined)
X_scaled = np.clip(X_scaled, -10, 10)
pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X_scaled)
umap_model = UMAP(n_components=2, n_neighbors=30, random_state=42)
umap_coords = umap_model.fit_transform(X_pca[:, :30])

print(f"UMAP computed: {X_combined.shape[0]} cells, {X_combined.shape[1]} genes")

# Plot by modality
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

for m, color in [('cell', '#1f77b4'), ('nuclei', '#ff7f0e')]:
    mask = modality == m
    axes[0].scatter(umap_coords[mask, 0], umap_coords[mask, 1], c=color, s=1, alpha=0.3, label=m, rasterized=True)
axes[0].legend(markerscale=5)
axes[0].set_title('Config B genes (10,134) — by modality')
axes[0].set_xlabel('UMAP1')
axes[0].set_ylabel('UMAP2')

unique_cts = sorted(set(cell_types_combined))
cmap = plt.cm.get_cmap('tab20', len(unique_cts))
for i, ct in enumerate(unique_cts):
    mask = cell_types_combined == ct
    axes[1].scatter(umap_coords[mask, 0], umap_coords[mask, 1], c=[cmap(i)], s=1, alpha=0.3, label=ct, rasterized=True)
axes[1].legend(markerscale=5, fontsize=6, ncol=2, loc='upper right')
axes[1].set_title('Config B genes (10,134) — by cell type')
axes[1].set_xlabel('UMAP1')
axes[1].set_ylabel('UMAP2')

plt.tight_layout()
plt.savefig(f'{output_dir}/umap_configB_raw.png', dpi=150, bbox_inches='tight')
print(f"Saved: {output_dir}/umap_configB_raw.png")

# =====================================================================
# 9. Dot plot of marker genes (present ones) — manual implementation
# =====================================================================
print("\nGenerating marker dot plot...")

present_markers = [g for g in markers.keys() if g in config_b and g in set(cell_gene_names)]
if present_markers:
    # Group by cell type category for ordering
    marker_order = sorted(present_markers, key=lambda g: (markers[g], g))

    # Get gene indices in cell data
    gene_to_idx = {g: i for i, g in enumerate(cell_gene_names)}

    # Compute dot plot data: fraction expressing + mean expression per cell type
    ct_order = sorted(set(ct_cell))
    frac_mat = np.zeros((len(ct_order), len(marker_order)))
    mean_mat = np.zeros((len(ct_order), len(marker_order)))

    for j, gene in enumerate(marker_order):
        gidx = gene_to_idx[gene]
        for i, ct in enumerate(ct_order):
            ct_mask = ct_cell == ct
            vals = X_cell_cb[ct_mask, np.where(cell_gene_names[configb_cell_mask] == gene)[0][0]] if gene in cell_gene_names[configb_cell_mask] else np.zeros(ct_mask.sum())
            # Use raw counts for fraction, log-normed for color
            raw_vals = X_cell[np.ix_(cell_idx[ct_mask], [gidx])][:, 0]
            frac_mat[i, j] = (raw_vals > 0).mean()
            mean_mat[i, j] = np.log1p(raw_vals[raw_vals > 0].mean()) if (raw_vals > 0).any() else 0

    # Scale mean expression per gene (0-1)
    for j in range(mean_mat.shape[1]):
        col = mean_mat[:, j]
        if col.max() > col.min():
            mean_mat[:, j] = (col - col.min()) / (col.max() - col.min())

    fig, ax = plt.subplots(figsize=(max(12, len(marker_order)*0.6), max(6, len(ct_order)*0.4)))
    for i in range(len(ct_order)):
        for j in range(len(marker_order)):
            size = frac_mat[i, j] * 200
            color = plt.cm.Reds(mean_mat[i, j])
            ax.scatter(j, i, s=size, c=[color], edgecolors='gray', linewidth=0.5)

    ax.set_xticks(range(len(marker_order)))
    ax.set_xticklabels([f"{g}\n({markers[g]})" for g in marker_order], rotation=90, fontsize=7)
    ax.set_yticks(range(len(ct_order)))
    ax.set_yticklabels(ct_order, fontsize=8)
    ax.set_title('Marker genes in Config B — Dot plot (cell data)')
    ax.set_xlim(-0.5, len(marker_order) - 0.5)
    ax.set_ylim(-0.5, len(ct_order) - 0.5)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/dotplot_present_markers.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir}/dotplot_present_markers.png")

print("\nDone!")
PYEOF
