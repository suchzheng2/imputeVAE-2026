#!/bin/bash
#SBATCH --job-name=baseline_linreg
#SBATCH --output=jobs/logs/slurm-%j_baseline_linreg.out
#SBATCH --error=jobs/logs/slurm-%j_baseline_linreg.err
#SBATCH --partition=scavenge,pi_gerstein
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=02:00:00

echo "Job started: $(date)"
echo "Node: $(hostname)"

module load miniconda
conda activate impute2

cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

# Run with PerCT_5pct gene list (current standard)
python analysis/baseline_linear_regression.py \
    --gene_list data/gene_list_perct_5pct.csv \
    --output_dir notes/experiments/SUC-8_baseline/results

echo "Job finished: $(date)"
