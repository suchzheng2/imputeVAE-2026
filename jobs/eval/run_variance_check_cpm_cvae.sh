#!/bin/bash
#SBATCH --job-name=var_cpm_cvae
#SBATCH --partition=pi_gerstein
#SBATCH --mem=400G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=jobs/logs/slurm-%j_var_cpm_cvae.out
#SBATCH --error=jobs/logs/slurm-%j_var_cpm_cvae.err

source ~/.bashrc
conda activate impute2

cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

python analysis/quick_variance_check.py run/run_2026-05-12-15-07-59_1408472
