#!/bin/bash
#SBATCH --job-name=xt_impute
#SBATCH --partition=scavenge_gpu
#SBATCH --requeue
#SBATCH --gpus=1
#SBATCH --mem=300G
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --output=jobs/logs/slurm-%j_xt_impute.out
#SBATCH --error=jobs/logs/slurm-%j_xt_impute.err

source ~/.bashrc
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
export PYTHONUNBUFFERED=1

find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

MODEL_RUN="run/run_2026-07-18-19-40-51_570984"
CELL_H5AD="data/external/lake2023_cells.h5ad"
NUC_H5AD="data/external/lake2023_nuclei.h5ad"
OUTPUT_DIR="notes/experiments/cross_tissue_imputation"

echo "Running cross-tissue imputation (brain cVAE -> kidney)..."
python analysis/cross_tissue_imputation.py \
    --model_run "$MODEL_RUN" \
    --cell_h5ad "$CELL_H5AD" \
    --nuc_h5ad "$NUC_H5AD" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 512

echo "Done!"
