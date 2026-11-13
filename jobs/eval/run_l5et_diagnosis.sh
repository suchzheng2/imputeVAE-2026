#!/bin/bash
#SBATCH --job-name=l5et_diag
#SBATCH --partition=pi_gerstein
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=jobs/logs/slurm-%j_l5et_diag.out
#SBATCH --error=jobs/logs/slurm-%j_l5et_diag.err

source ~/.bashrc
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
export PYTHONUNBUFFERED=1

find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

python analysis/l5et_diagnosis.py
