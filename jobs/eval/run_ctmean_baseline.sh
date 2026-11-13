#!/bin/bash
#SBATCH --job-name=ctmean_baseline
#SBATCH --partition=pi_gerstein
#SBATCH --mem=200G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=jobs/logs/slurm-%j_ctmean_baseline.out
#SBATCH --error=jobs/logs/slurm-%j_ctmean_baseline.err

source ~/.bashrc
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

echo "CT-mean baseline — $(date)"
python analysis/baseline_ctmean.py \
    --output_dir run/ctmean_baseline \
    --gene_list data/training_gene_list_v3.csv

echo "Job finished at $(date)"
