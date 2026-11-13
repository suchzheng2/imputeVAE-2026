#!/bin/bash
#SBATCH --job-name=nz_figures
#SBATCH --partition=pi_gerstein,scavenge
#SBATCH --time=02:00:00
#SBATCH --mem=150G
#SBATCH --cpus-per-task=4
#SBATCH --output=jobs/logs/slurm-%j_nz_figures.out
#SBATCH --error=jobs/logs/slurm-%j_nz_figures.err

cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

source ~/.bashrc
conda activate impute2

python -m analysis.within_ct_nonzero_figures --model pct5v2_cond
