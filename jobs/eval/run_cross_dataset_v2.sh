#!/bin/bash
#SBATCH --job-name=xdataset_v2
#SBATCH --partition=priority_gpu
#SBATCH --account=prio_gerstein
#SBATCH --gpus=a5000:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=jobs/logs/slurm-%j_xdataset_v2.out
#SBATCH --error=jobs/logs/slurm-%j_xdataset_v2.err

source ~/.bashrc
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
export PYTHONUNBUFFERED=1

find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# Best full-data cVAE (trained on v3: n3A -> c3A)
RUN_DIR="run/run_2026-07-18-19-40-51_570984"

# Test 1: v2 batch (protocol generalization)
echo "=== Cross-dataset eval: v2 batch (n2A -> c2A) ==="
python analysis/cross_dataset_eval.py \
    --run_dir "$RUN_DIR" \
    --nuc_batch n2A \
    --cell_batch c2A \
    --output_dir run/cross_dataset_eval/v2_batch \
    --name cvae_v2

# Test 2: Also run on v3 (same protocol as training) for comparison
echo ""
echo "=== Reference eval: v3 batch (n3A -> c3A) ==="
python analysis/cross_dataset_eval.py \
    --run_dir "$RUN_DIR" \
    --nuc_batch n3A \
    --cell_batch c3A \
    --output_dir run/cross_dataset_eval/v3_batch \
    --name cvae_v3
