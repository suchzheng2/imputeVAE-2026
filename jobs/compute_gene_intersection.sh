#!/bin/bash
#SBATCH --job-name=gene_intersect
#SBATCH --partition=pi_gerstein
#SBATCH --mem=200G
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --output=jobs/logs/slurm-%j_gene_intersect.out
#SBATCH --error=jobs/logs/slurm-%j_gene_intersect.err

source ~/.bashrc
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

python3 << 'PYEOF'
import anndata as ad
import pandas as pd
import numpy as np

common_genes = pd.read_csv("data/common_genes.csv", header=0)['common_genes'].tolist()
mitocarta = set(pd.read_csv("data/Mouse.MitoCarta.csv", header=0)['Symbol'].tolist())

# Cell v3
cell = ad.read_h5ad("/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi/10X_cells_v2_v3_AIBS/data.h5ad")
cell = cell[:, cell.var['gene_names'].isin(common_genes)].copy()
cell.obs['meta_batch_id'] = cell.obs['batch_id'].str.split('.').str[0]
cell = cell[cell.obs['meta_batch_id'].str.contains('c3')].copy()
print(f"Cell v3: {cell.shape}")
cell_det = np.asarray((cell.X > 0).sum(axis=0) / cell.X.shape[0]).flatten()
cell_pass = set(cell.var['gene_names'][cell_det >= 0.05].tolist())
print(f"Cell genes >=5% det: {len(cell_pass)}")
del cell

# Nuclei v3
nuc = ad.read_h5ad("/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi/10X_nuclei_v2_v3_AIBS/data.h5ad")
nuc = nuc[:, nuc.var['gene_names'].isin(common_genes)].copy()
nuc.obs['meta_batch_id'] = nuc.obs['batch_id'].str.split('.').str[0]
nuc = nuc[nuc.obs['meta_batch_id'].str.contains('n3')].copy()
print(f"Nuclei v3: {nuc.shape}")
nuc_det = np.asarray((nuc.X > 0).sum(axis=0) / nuc.X.shape[0]).flatten()
nuc_pass = set(nuc.var['gene_names'][nuc_det >= 0.05].tolist())
print(f"Nuclei genes >=5% det: {len(nuc_pass)}")
del nuc

# Stats
print(f"\nCell-only (det>=5% in cell, <5% in nuclei): {len(cell_pass - nuc_pass)}")
print(f"Nuclei-only (det>=5% in nuclei, <5% in cell): {len(nuc_pass - cell_pass)}")
print(f"Intersection (det>=5% in both): {len(cell_pass & nuc_pass)}")

# Intersection minus MitoCarta
intersection = sorted((cell_pass & nuc_pass) - mitocarta)
mito_removed = len(cell_pass & nuc_pass) - len(intersection)
print(f"MitoCarta removed: {mito_removed}")
print(f"Final training gene set: {len(intersection)}")

pd.Series(intersection).to_csv("data/training_gene_list_v3.csv", index=False, header=False)
print(f"Saved to data/training_gene_list_v3.csv")

# Also save the cell-only and nuclei-only lists for reference
pd.Series(sorted(cell_pass - nuc_pass)).to_csv("data/cell_only_genes_v3.csv", index=False, header=False)
pd.Series(sorted(nuc_pass - cell_pass)).to_csv("data/nuclei_only_genes_v3.csv", index=False, header=False)
print("Saved cell-only and nuclei-only gene lists for reference")
PYEOF
