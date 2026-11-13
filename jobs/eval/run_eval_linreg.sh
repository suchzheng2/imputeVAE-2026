#!/bin/bash
#SBATCH --job-name=eval_linreg
#SBATCH --partition=pi_gerstein
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=jobs/logs/slurm-%j_eval_linreg.out
#SBATCH --error=jobs/logs/slurm-%j_eval_linreg.err

source ~/.bashrc
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
export PYTHONUNBUFFERED=1

find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

OUT_DIR="run/linreg_baseline"

# Step 1: Export imputed .npy via linear regression
echo "=== Step 1: Fitting linear regression and exporting imputed matrix ==="
python analysis/baseline_linreg_export.py \
    --gene_list data/training_gene_list_v3.csv \
    --output_dir "$OUT_DIR"

# Step 2: Run full eval pipeline (same as other baselines)
echo "=== Step 2: Running evaluation pipeline ==="
python analysis/evaluate_baseline_incremental.py \
    --imputed_npy "$OUT_DIR/imputed_cell_from_nucleus.npy" \
    --obs_csv "$OUT_DIR/nucleus_obs.csv" \
    --name linreg \
    --gene_list data/training_gene_list_v3.csv \
    --output_dir "$OUT_DIR/eval"
