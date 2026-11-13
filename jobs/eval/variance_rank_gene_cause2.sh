#!/bin/bash
#SBATCH --job-name=var_gene_cause2
#SBATCH --partition=scavenge
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=jobs/logs/slurm-%j_var_gene_cause2.out
#SBATCH --error=jobs/logs/slurm-%j_var_gene_cause2.err
# Technical vs biological cause of variance collapse for Gm49164, Gfra1 (L5 IT), 2026-09-25.
module load miniconda
conda activate impute2
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
python analysis/variance_rank_gene_cause2.py
