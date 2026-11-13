#!/bin/bash
#SBATCH --job-name=halluc_pct5v2
#SBATCH --output=jobs/logs/slurm-%j_hallucination_test_pct5_v2.out
#SBATCH --error=jobs/logs/slurm-%j_hallucination_test_pct5_v2.err
#SBATCH --partition=scavenge_gpu
#SBATCH --mem=200G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate impute2

cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

# Delete stale bytecode
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# Hallucination tests on both v2 models (cond and cond+xr)
python analysis/hallucination_test.py \
  --models pct5v2_cond,pct5v2_cond_xr \
  --output_dir notes/experiments/SUC-5_perct5_retraining/hallucination_tests_v2 \
  2>&1
