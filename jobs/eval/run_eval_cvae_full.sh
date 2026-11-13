#!/bin/bash
#SBATCH --job-name=eval_cvae_full
#SBATCH --partition=priority_gpu
#SBATCH --account=prio_gerstein
#SBATCH --gpus=a100:1
#SBATCH --mem=400G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=jobs/logs/slurm-%j_eval_cvae_full.out
#SBATCH --error=jobs/logs/slurm-%j_eval_cvae_full.err

source ~/.bashrc
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
export PYTHONUNBUFFERED=1

find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# Find the run folder from job 7714771 (full-data single-stage cVAE normloss)
# Update RUN_DIR after job completes — check with:
#   ls -td run/run_2026-07-* | head -1
RUN_DIR="run/run_2026-07-18-19-40-51_570984"
OUT_DIR="run/cvae_full_eval"

# Step 1: Export imputed .npy from cVAE model
echo "=== Step 1: Exporting imputed cell expression from cVAE ==="
python analysis/export_cvae_imputed.py \
    --run_dir "$RUN_DIR" \
    --output_dir "$OUT_DIR"

# Step 2: Run full eval pipeline
echo "=== Step 2: Running evaluation pipeline ==="
python analysis/evaluate_baseline_incremental.py \
    --imputed_npy "$OUT_DIR/imputed_cell_from_nucleus.npy" \
    --obs_csv "$OUT_DIR/nucleus_obs.csv" \
    --name cvae_full \
    --gene_list data/training_gene_list_v3.csv \
    --output_dir "$OUT_DIR/eval"

# Step 3: Quick variance check
echo "=== Step 3: Variance ratio ==="
python analysis/quick_variance_check.py "$RUN_DIR"
