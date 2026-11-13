#!/bin/bash
# per-gene density figures (10 genes, 6 series) for every cell type; outputs armA/figures/gene_densities_<CT>/
source ~/.bashrc 2>/dev/null; conda activate impute2; export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
for CT in "Astro" "Endo" "L2/3 IT" "L5 ET" "L5 IT" "L5/6 NP" "L6 CT" "L6 IT" "L6b" "Lamp5" "Macrophage" "OPC" "Pvalb" "Sncg" "Sst" "Vip"; do
  echo "=== $CT $(date +%H:%M:%S)"; python analysis/benchmark/gene_density_ct.py --ct "$CT" 2>&1 | tail -1
done
echo "=== all done $(date)"
