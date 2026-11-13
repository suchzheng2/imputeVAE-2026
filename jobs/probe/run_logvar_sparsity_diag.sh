#!/bin/bash
#SBATCH --job-name=lv_sparsity
#SBATCH --partition=pi_gerstein
#SBATCH --mem=200G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=jobs/logs/slurm-%j_logvar_sparsity.out
#SBATCH --error=jobs/logs/slurm-%j_logvar_sparsity.err

source ~/.bashrc
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# Check: do extreme log_var values map to zero vs nonzero expression?
# Uses existing Config B stage 1 model (the one that feeds into stage 2)
# No GPU needed -- CPU inference only

python analysis/logvar_sparsity_diagnostic.py \
  --run_path "run/run_2026-04-01-09-29-26_3250320" \
  --data_path "10X_cells_v2_v3_AIBS/" \
  --output_dir "notes/experiments/logvar_sparsity" \
  --max_cells 5000
