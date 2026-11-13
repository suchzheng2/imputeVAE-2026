#!/bin/bash
#SBATCH --job-name=bench_eval
#SBATCH --partition=scavenge
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=jobs/logs/slurm-%j_bench_eval.out
#SBATCH --error=jobs/logs/slurm-%j_bench_eval.err
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
source ~/.bashrc 2>/dev/null || true; conda activate impute2; export PYTHONUNBUFFERED=1
set -eo pipefail
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
find analysis/benchmark -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
EVAL=${EVAL_PARTITION:-test}
python -m analysis.benchmark.evaluate_all --eval_partition $EVAL --base_dir run/benchmark/$EVAL \
    --output_dir run/benchmark/$EVAL/results "$@"
