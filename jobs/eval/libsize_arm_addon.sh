#!/bin/bash
#SBATCH --job-name=libsize_addon
#SBATCH --partition=scavenge
#SBATCH --array=1-2
#SBATCH --time=1:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=4
#SBATCH --output=jobs/logs/slurm-%A_%a_libsize_addon.out
#SBATCH --error=jobs/logs/slurm-%A_%a_libsize_addon.err
# Dimensionality + library-size R2 add-on (analysis/variance_zero_nonzero_addon.py) for the covariate arms (2026-10-23).
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
source ~/.bashrc 2>/dev/null || true; conda activate impute2
set -eo pipefail; export PYTHONUNBUFFERED=1
case "${SLURM_ARRAY_TASK_ID:-1}" in 1) ARM=armA ;; 2) ARM=armB ;; 3) ARM=latent8 ;; 4) ARM=latent4 ;; esac  # 3/4 = latent-width controls (2026-10-23)
export IMPUTED=run/libsize_${ARM}_eval/imputed_cell_from_nucleus.npy OUT_DIR=notes/experiments/libsize_covariate/${ARM}/tables
python analysis/variance_zero_nonzero_addon.py
echo "=== Done $(date)"
