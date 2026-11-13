#!/bin/bash
#SBATCH --job-name=baseline_eval
#SBATCH --partition=pi_gerstein
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=jobs/logs/slurm-%j_baseline_eval.out
#SBATCH --error=jobs/logs/slurm-%j_baseline_eval.err

source ~/.bashrc
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
export PYTHONUNBUFFERED=1

find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

GENE_LIST=data/training_gene_list_v3.csv

# --- 1. CT-mean baseline ---
echo "=== CT-mean baseline === $(date)"
python analysis/evaluate_baseline_incremental.py \
    --imputed_npy run/ctmean_baseline/imputed_cell_from_nucleus.npy \
    --obs_csv run/ctmean_baseline/nucleus_obs.csv \
    --name ctmean \
    --gene_list $GENE_LIST \
    --output_dir run/ctmean_baseline/eval

# --- 2. scVI baseline ---
echo ""
echo "=== scVI evaluation === $(date)"
python analysis/evaluate_baseline_incremental.py \
    --imputed_npy run/scvi_baseline/imputed_cell_from_nucleus.npy \
    --obs_csv run/scvi_baseline/nucleus_obs.csv \
    --name scvi \
    --gene_list $GENE_LIST \
    --output_dir run/scvi_baseline/eval

echo ""
echo "All done at $(date)"
