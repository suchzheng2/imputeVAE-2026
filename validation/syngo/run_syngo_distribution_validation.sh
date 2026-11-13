#!/bin/bash
#SBATCH --job-name=syngo_dist_val
#SBATCH --partition=scavenge
#SBATCH --time=4:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --output=jobs/logs/slurm-%j_syngo_dist_val.out
#SBATCH --error=jobs/logs/slurm-%j_syngo_dist_val.err

module load miniconda
conda activate impute2

cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

python -u validation/syngo/syngo_brain_validation/syngo_distribution_validation.py
