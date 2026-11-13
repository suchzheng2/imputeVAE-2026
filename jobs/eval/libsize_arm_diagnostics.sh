#!/bin/bash
#SBATCH --job-name=libsize_diag
#SBATCH --partition=scavenge
#SBATCH --array=1-2
#SBATCH --time=4:00:00
#SBATCH --mem=140G
#SBATCH --cpus-per-task=8
#SBATCH --output=jobs/logs/slurm-%A_%a_libsize_diag.out
#SBATCH --error=jobs/logs/slurm-%A_%a_libsize_diag.err
# Library-size covariate arms (notes/libsize_covariate_review_2026-10-09.md §5, steps 2-4):
#   export all-data imputed matrix at ct_median_cell -> latent_usage_check -> variance_zero_nonzero_fullmodel.
# Array task 1 = arm A (JSD 0.1/0.1), 2 = arm B (JSD 0.01/0.01). CPU only (paper-model runs took ~10 min at <90 GB each).
# 3 = latent-width control 8 dims, 4 = 4 dims (arm A config; runs of 2026-10-23, job 13784373) -> submit with --array=3-4.
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
source ~/.bashrc 2>/dev/null || true; conda activate impute2
set -eo pipefail
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
find imputeVAE analysis -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

case "${SLURM_ARRAY_TASK_ID:-1}" in
  1) ARM=armA; RUN=run/run_2026-10-16_libsize_armA_jsd0.1_seed1 ;;
  2) ARM=armB; RUN=run/run_2026-10-16_libsize_armB_jsd0.01_seed1 ;;
  3) ARM=latent8; RUN=run/run_2026-10-23_libsize_armA_latent8_seed1 ;;
  4) ARM=latent4; RUN=run/run_2026-10-23_libsize_armA_latent4_seed1 ;;
  *) echo "bad array id"; exit 2 ;;
esac
TARGET=${LIBSIZE_TARGET:-ct_median_cell}
EXPORT=run/libsize_${ARM}_eval
NOTES=notes/experiments/libsize_covariate/${ARM}
mkdir -p "$EXPORT" "$NOTES"
echo "=== $ARM run=$RUN target=$TARGET host=$(hostname) $(date)"

echo "=== 1/3 export"
python analysis/export_cvae_imputed.py --run_dir "$RUN" --output_dir "$EXPORT" --target_libsize "$TARGET"

echo "=== 2/3 latent usage"
python analysis/latent_usage_check.py --run_dir "$RUN" --out "$NOTES/latent_usage" \
    --imputed "$EXPORT/imputed_cell_from_nucleus.npy" --target_libsize "$TARGET"

echo "=== 3/3 variance decomposition"
python analysis/variance_zero_nonzero_fullmodel.py --imputed "$EXPORT/imputed_cell_from_nucleus.npy" --out "$NOTES/tables"

echo "=== Done $(date)"
