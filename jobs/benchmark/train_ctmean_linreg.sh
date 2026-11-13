#!/bin/bash
#SBATCH --job-name=bench_trivial
#SBATCH --partition=scavenge
#SBATCH --time=01:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --output=jobs/logs/slurm-%j_bench_trivial.out
#SBATCH --error=jobs/logs/slurm-%j_bench_trivial.err
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
source ~/.bashrc 2>/dev/null || true; conda activate impute2; export PYTHONUNBUFFERED=1
set -eo pipefail
EVAL=${EVAL_PARTITION:-test}
echo "=== CT-mean + raw nuclei ($EVAL) ==="
python -m analysis.benchmark.train_ctmean --eval_partition $EVAL --output_root run/benchmark/$EVAL
echo "=== Linear regression ($EVAL) ==="
python -m analysis.benchmark.train_linreg --eval_partition $EVAL --output_root run/benchmark/$EVAL
echo "Done."
