#!/bin/bash
#SBATCH --job-name=eval_splitaware
#SBATCH --output=jobs/logs/slurm-%j_eval_splitaware.out
#SBATCH --error=jobs/logs/slurm-%j_eval_splitaware.err
#SBATCH --partition=scavenge
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00

set -euo pipefail

echo "=== Split-Aware Evaluation ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Start: $(date)"
echo ""

cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

# Clean pycache
find imputeVAE/ analysis/ -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

source activate impute2

python analysis/eval_splitaware.py \
    --cvae_run_dir run/run_2026-07-18-19-40-51_570984 \
    --output_dir run/eval_v3_splitaware

echo ""
echo "=== Done: $(date) ==="
