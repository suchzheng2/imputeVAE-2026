#!/bin/bash
#SBATCH --job-name=eval_linreg
#SBATCH --output=jobs/logs/slurm-%j_eval_linreg.out
#SBATCH --error=jobs/logs/slurm-%j_eval_linreg.err
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=02:00:00

echo "Job started: $(date)"
echo "Node: $(hostname)"

module load miniconda
conda activate impute2

cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

# Clear stale bytecode
find imputeVAE/ analysis/ -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null

# Step 1: Export imputed matrix (.npy + obs .csv)
echo "=== Step 1: Exporting linreg imputed matrix ==="
python analysis/baseline_linreg_export.py \
    --gene_list data/training_gene_list_v3.csv \
    --output_dir run/linreg_baseline

echo ""
echo "=== Step 2: Running eval pipeline ==="
python analysis/evaluate_baseline_incremental.py \
    --imputed_npy run/linreg_baseline/imputed_cell_from_nucleus.npy \
    --obs_csv run/linreg_baseline/nucleus_obs.csv \
    --name linreg \
    --output_dir run/linreg_baseline/eval \
    --gene_list data/training_gene_list_v3.csv

echo ""
echo "Job finished: $(date)"
