#!/bin/bash
#SBATCH --job-name=eval_1m
#SBATCH --partition=scavenge_gpu
#SBATCH --time=1:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --output=jobs/logs/slurm-%j_eval_%x.out
#SBATCH --error=jobs/logs/slurm-%j_eval_%x.err

# Usage: sbatch --export=MODEL=model_name jobs/eval/eval_one_model.sh

source ~/.bashrc
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
export PYTHONUNBUFFERED=1

find imputeVAE/ analysis/ -name "__pycache__" -exec rm -rf {} + 2>/dev/null

echo "=== Evaluating: ${MODEL} ==="
echo "Started: $(date)"

python -m analysis.eval.runner \
    --model "${MODEL}" \
    --eval all \
    --output_dir notes/eval_results

echo "Finished: $(date)"
