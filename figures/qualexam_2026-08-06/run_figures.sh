#!/bin/bash
#SBATCH --job-name=qualexam_figs
#SBATCH --partition=scavenge
#SBATCH --time=02:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=4
#SBATCH --output=jobs/logs/slurm-%j_qualexam_figs.out
#SBATCH --error=jobs/logs/slurm-%j_qualexam_figs.err

source ~/.bashrc
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

python figures/qualexam_2026-08-06/generate_figures.py 4 5

echo "Done: $(date)"
