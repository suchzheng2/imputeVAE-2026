#!/bin/bash
#SBATCH --job-name=eval_cpm_full
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --mem=140G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=jobs/logs/slurm-%j_eval_cpm_full.out
#SBATCH --error=jobs/logs/slurm-%j_eval_cpm_full.err

source ~/.bashrc
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
export PYTHONUNBUFFERED=1

find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

RUN_DIR="run/run_2026-05-20-00-46-21_2182434"
OUT_DIR="run/cpm_full_eval"

# Step 1: Export imputed .npy
echo "=== Step 1: Exporting imputed cell expression from CPM cVAE ==="
python analysis/export_cvae_imputed.py \
    --run_dir "$RUN_DIR" \
    --output_dir "$OUT_DIR"

# Step 2: Run full eval pipeline
echo "=== Step 2: Running evaluation pipeline ==="
python analysis/evaluate_baseline_incremental.py \
    --imputed_npy "$OUT_DIR/imputed_cell_from_nucleus.npy" \
    --obs_csv "$OUT_DIR/nucleus_obs.csv" \
    --name cpm_full \
    --norm cpm \
    --gene_list data/training_gene_list_v3.csv \
    --output_dir "$OUT_DIR/eval"

# Step 3: Quick variance check
echo "=== Step 3: Variance ratio ==="
python analysis/quick_variance_check.py "$RUN_DIR"
