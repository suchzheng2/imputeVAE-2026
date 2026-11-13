#!/bin/bash
#SBATCH --job-name=bench_scbutterfly
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0
#SBATCH --output=jobs/logs/slurm-%A_%a_bench_scbutterfly.out
#SBATCH --error=jobs/logs/slurm-%A_%a_bench_scbutterfly.err
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
source ~/.bashrc 2>/dev/null || true; conda activate impute2; export PYTHONUNBUFFERED=1
set -eo pipefail
find analysis -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
EVAL=${EVAL_PARTITION:-test}
SEED=${SLURM_ARRAY_TASK_ID:-0}
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -m analysis.benchmark.train_scbutterfly --seed $((19193 + SEED)) --eval_partition $EVAL --output_dir run/benchmark/$EVAL/scbutterfly_seed$SEED
