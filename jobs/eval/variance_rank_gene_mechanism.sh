#!/bin/bash
#SBATCH --job-name=var_gene_mechanism
#SBATCH --partition=scavenge
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=jobs/logs/slurm-%j_var_gene_mechanism.out
#SBATCH --error=jobs/logs/slurm-%j_var_gene_mechanism.err
# Latent-loading + partner-gene tests for L5 IT example genes, 2026-09-25.
module load miniconda
conda activate impute2
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
python analysis/variance_rank_gene_mechanism.py
