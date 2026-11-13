#!/bin/bash
#SBATCH --job-name=xdec_heatmaps
#SBATCH --output=jobs/logs/slurm-%j_xdec_heatmaps.out
#SBATCH --error=jobs/logs/slurm-%j_xdec_heatmaps.err
#SBATCH --partition=pi_gerstein
#SBATCH --mem=200G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00

source ~/.bashrc
conda activate impute2

cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

# Delete stale bytecode
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# Best model: MSE cond+xr
RUN=run_2026-04-02-10-51-32_3738876

# Selected epochs: early, mid, late, final
python analysis/eval_cross_decoded_heatmaps.py \
    --run_folder "$RUN" \
    --epochs 0 100 300 599 \
    2>&1
