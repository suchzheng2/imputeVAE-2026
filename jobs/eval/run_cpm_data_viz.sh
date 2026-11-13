#!/bin/bash
#SBATCH --job-name=cpm_data_viz
#SBATCH --partition=pi_gerstein,scavenge
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=jobs/logs/slurm-%j_cpm_data_viz.out
#SBATCH --error=jobs/logs/slurm-%j_cpm_data_viz.err

source ~/.bashrc
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

python -m analysis.cpm_data_viz --n_cells 8000
