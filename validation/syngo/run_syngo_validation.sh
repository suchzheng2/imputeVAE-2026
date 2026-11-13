#!/bin/bash
#SBATCH --job-name=syngo_val
#SBATCH --partition=scavenge
#SBATCH --time=4:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --output=jobs/logs/slurm-%j_syngo_val.out
#SBATCH --error=jobs/logs/slurm-%j_syngo_val.err

# SynGO biological consistency analysis for ImputeVAE
# Usage: sbatch validation/syngo/run_syngo_validation.sh

set -e

cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

# Clear stale bytecode
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

module load miniconda
conda activate impute2

echo "Starting SynGO validation analysis at $(date)"
echo "Python: $(which python)"
echo "Working dir: $(pwd)"

python -u validation/syngo/scripts/syngo_validation.py \
    --output_dir validation/syngo/results \
    --n_bootstrap 2000 \
    --priority 3 \
    --seed 42

echo "Finished at $(date)"
