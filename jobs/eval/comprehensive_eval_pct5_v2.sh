#!/bin/bash
#SBATCH --job-name=compeval_pct5v2
#SBATCH --output=jobs/logs/slurm-%j_comprehensive_eval_pct5_v2.out
#SBATCH --error=jobs/logs/slurm-%j_comprehensive_eval_pct5_v2.err
#SBATCH --partition=pi_gerstein
#SBATCH --mem=200G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00

source ~/.bashrc
conda activate impute2

cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

# Delete stale bytecode
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# Comprehensive eval on both v2 models (cond and cond+xr)
python analysis/comprehensive_eval.py \
  --models pct5v2_cond,pct5v2_cond_xr \
  --output_dir notes/experiments/SUC-5_perct5_retraining/comprehensive_eval_v2 \
  2>&1

echo "Comprehensive eval pct5 v2 complete."
