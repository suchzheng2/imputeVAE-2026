#!/bin/bash
#SBATCH --job-name=bench_cache
#SBATCH --partition=scavenge
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --output=jobs/logs/slurm-%j_bench_cache.out
#SBATCH --error=jobs/logs/slurm-%j_bench_cache.err

cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
source ~/.bashrc 2>/dev/null || true
conda activate impute2
set -eo pipefail
export PYTHONUNBUFFERED=1
find analysis/benchmark -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

echo "=== Benchmark cache build: $(date) on $(hostname) ==="
python -m analysis.benchmark.build_cache "$@"
echo "=== Done: $(date) ==="
