#!/bin/bash
#SBATCH --job-name=latent_usage
#SBATCH --partition=scavenge
#SBATCH --time=2:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --output=jobs/logs/slurm-%j_latent_usage.out
#SBATCH --error=jobs/logs/slurm-%j_latent_usage.err
module load miniconda
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
python analysis/latent_usage_check.py
