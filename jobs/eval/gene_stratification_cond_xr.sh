#!/bin/bash
#SBATCH --job-name=gene_strat
#SBATCH --partition=pi_gerstein
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=jobs/logs/slurm-%j_gene_strat.out
#SBATCH --error=jobs/logs/slurm-%j_gene_strat.err

source ~/.bashrc
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# Gene stratification for MSE Cond+XR (run E, best model)
python analysis/gene_stratification.py \
  --run_folder run_2026-04-02-10-51-32_3738876 \
  --epoch 599

echo "Gene stratification complete."
