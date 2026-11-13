#!/bin/bash
#SBATCH --job-name=eval-cond
#SBATCH --out="jobs/logs/slurm-%j_eval_cond_500560.out"
#SBATCH --partition=pi_gerstein,scavenge_gpu
#SBATCH --gpus=1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --requeue
#SBATCH --mem=300G
#SBATCH --mail-type=ALL

set -euo pipefail

cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

module load miniconda
conda activate impute2

RUN_FOLDER="run/run_2026-03-26-23-28-20_500560"

echo "=== Conditioned model evaluation ==="
echo "Run: ${RUN_FOLDER}"
echo "Start: $(date)"

# Generate epoch dumps and run UMAPs + heatmaps + correlation
# (scIB runs as subprocess in abc_env)
python analysis/eval_probe_cond.py \
    --run_folder "${RUN_FOLDER}" \
    --splits train,valid \
    --skip_umap

echo "=== Done: $(date) ==="
