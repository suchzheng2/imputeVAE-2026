#!/bin/bash
#SBATCH --job-name=var_gene_examples
#SBATCH --partition=scavenge
#SBATCH --time=1:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=jobs/logs/slurm-%j_var_gene_examples.out
#SBATCH --error=jobs/logs/slurm-%j_var_gene_examples.err
# Per-gene value distributions for L5 IT example genes (real cell / raw nucleus / imputed), 2026-09-25.
module load miniconda
conda activate impute2
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
python analysis/variance_rank_gene_examples.py
