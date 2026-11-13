#!/bin/bash
#SBATCH --job-name=bench_cvae
#SBATCH --partition=scavenge
#SBATCH --time=01:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --output=jobs/logs/slurm-%j_bench_cvae.out
#SBATCH --error=jobs/logs/slurm-%j_bench_cvae.err
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
source ~/.bashrc 2>/dev/null || true; conda activate impute2; export PYTHONUNBUFFERED=1
set -eo pipefail
find imputeVAE analysis -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
EVAL=${EVAL_PARTITION:-test}
RUN_DIR=${CVAE_RUN_DIR:-run/run_2026-07-18-19-40-51_570984}
# CVAE_TARGET_LIBSIZE: auto (default) | ct_median_cell | quantile_map | source | global_quantile_map | global_shift
#   (libsize_cond runs only; the global_* modes use no cell-type label, added 2026-10-23)
# CVAE_TRAIN_JOB / CVAE_TRAIN_TIME_S: provenance of the training job (defaults = paper run 7714771)
python -m analysis.benchmark.predict_cvae --run_dir $RUN_DIR --eval_partition $EVAL \
  --output_dir ${CVAE_OUT:-run/benchmark/$EVAL/cvae} --target_libsize ${CVAE_TARGET_LIBSIZE:-auto} \
  --training_job_id ${CVAE_TRAIN_JOB:-7714771} --training_time_s ${CVAE_TRAIN_TIME_S:-44102.0}
