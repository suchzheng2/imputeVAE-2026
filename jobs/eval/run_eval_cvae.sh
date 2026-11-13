#!/bin/bash
#SBATCH --job-name=eval_cvae
#SBATCH --partition=pi_gerstein
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=jobs/logs/slurm-%j_eval_cvae.out
#SBATCH --error=jobs/logs/slurm-%j_eval_cvae.err

source ~/.bashrc
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
export PYTHONUNBUFFERED=1

find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

RUN_DIR="run/run_2026-05-11-19-52-25_134850"
OUT_DIR="run/cvae_baseline"

# Step 1: Export imputed .npy from cVAE model
echo "=== Step 1: Exporting imputed cell expression from cVAE ==="
python analysis/export_cvae_imputed.py \
    --run_dir "$RUN_DIR" \
    --output_dir "$OUT_DIR"

# Step 2: Run full eval pipeline (same as scVI/CT-mean)
echo "=== Step 2: Running evaluation pipeline ==="
python analysis/evaluate_baseline_incremental.py \
    --imputed_npy "$OUT_DIR/imputed_cell_from_nucleus.npy" \
    --obs_csv "$OUT_DIR/nucleus_obs.csv" \
    --name cvae \
    --gene_list data/training_gene_list_v3.csv \
    --output_dir "$OUT_DIR/eval"
