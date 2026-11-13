#!/bin/bash
#SBATCH --job-name=var_zero_nz
#SBATCH --partition=scavenge
#SBATCH --time=4:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=4
#SBATCH --output=jobs/logs/slurm-%j_var_zero_nz.out
#SBATCH --error=jobs/logs/slurm-%j_var_zero_nz.err
module load miniconda
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
export PYTHONUNBUFFERED=1
python analysis/variance_zero_nonzero_fullmodel.py
