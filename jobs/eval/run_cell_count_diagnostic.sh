#!/bin/bash
#SBATCH --job-name=cell_count
#SBATCH --partition=pi_gerstein
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=jobs/logs/slurm-%j_cell_count.out
#SBATCH --error=jobs/logs/slurm-%j_cell_count.err

module purge
source ~/.bashrc
conda activate impute2

echo "Running cell count diagnostic"
echo "Start: $(date)"

python analysis/cell_count_diagnostic.py --subset_ratios 0.2,1.0

echo "Done: $(date)"
