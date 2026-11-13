# imputeVAE

Staged VAE framework for aligning cell and nucleus transcriptomics.

## External Data Dependencies

The following data files are accessed from outside this repository and must be
placed (or symlinked) under the `data/` directory before running.

### Raw h5ad files (`data/scvi/`)

These are the raw count matrices used when `--use_raw_data` is enabled.

| Local path | Original location |
|---|---|
| `data/scvi/10X_cells_v2_v3_AIBS/data.h5ad` | `Cytoplasm_imputation/scvi/10X_cells_v2_v3_AIBS/data.h5ad` |
| `data/scvi/10X_nuclei_v2_v3_AIBS/data.h5ad` | `Cytoplasm_imputation/scvi/10X_nuclei_v2_v3_AIBS/data.h5ad` |

### ABC integrated results (`data/ABC_results/`)

Pre-processed integrated h5ad files used when `--use_raw_data` is **not** set.

| Local path | Original location |
|---|---|
| `data/ABC_results/10X_cells_v2_v3_AIBS_g10k/integrated.h5ad` | `Cytoplasm_imputation/ABC/results/10X_cells_v2_v3_AIBS_g10k/integrated.h5ad` |
| `data/ABC_results/10X_nuclei_v2_v3_AIBS_g10k/integrated.h5ad` | `Cytoplasm_imputation/ABC/results/10X_nuclei_v2_v3_AIBS_g10k/integrated.h5ad` |

### MitoCarta gene list (`data/`)

Used when `--exclude_mitocarta` is enabled to filter mitochondrial genes.

| Local path | Original location |
|---|---|
| `data/Mouse.MitoCarta.csv` | `Cytoplasm_imputation/mouse.miticarta.analysis/Mouse.MitoCarta.csv` |

### Quick setup with symlinks

```bash
cd data/

# Raw h5ad files
mkdir -p scvi
ln -s /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi/10X_cells_v2_v3_AIBS scvi/10X_cells_v2_v3_AIBS
ln -s /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi/10X_nuclei_v2_v3_AIBS scvi/10X_nuclei_v2_v3_AIBS

# ABC integrated results
mkdir -p ABC_results
ln -s /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/ABC/results/10X_cells_v2_v3_AIBS_g10k ABC_results/10X_cells_v2_v3_AIBS_g10k
ln -s /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/ABC/results/10X_nuclei_v2_v3_AIBS_g10k ABC_results/10X_nuclei_v2_v3_AIBS_g10k

# MitoCarta
ln -s /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/mouse.miticarta.analysis/Mouse.MitoCarta.csv Mouse.MitoCarta.csv
```
