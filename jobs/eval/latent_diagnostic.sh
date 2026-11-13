#!/bin/bash
#SBATCH --job-name=latent_diag
#SBATCH --output=jobs/logs/slurm-%j_latent_diagnostic.out
#SBATCH --error=jobs/logs/slurm-%j_latent_diagnostic.err
#SBATCH --partition=scavenge
#SBATCH --mem=100G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00

source ~/.bashrc
conda activate impute2

cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

# Delete stale bytecode
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# Run latent diagnostic on pct5v2_cond (no xr — retains more variance)
python analysis/latent_diagnostic.py \
  --run_folder run/run_2026-04-07-02-12-05_3779214 \
  --gene_list data/gene_list_perct_5pct.csv \
  2>&1
