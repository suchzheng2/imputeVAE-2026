#!/bin/bash
#SBATCH --job-name=syngo_audit
#SBATCH --partition=scavenge
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=jobs/logs/slurm-%j_syngo_audit.out
#SBATCH --error=jobs/logs/slurm-%j_syngo_audit.err

# SynGO second-pass audit - works from existing CSV tables only (no h5ad)
# Usage: sbatch validation/syngo/run_syngo_audit.sh

set -e

cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

# Clear stale bytecode
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

module load miniconda
conda activate impute2

echo "Starting SynGO audit at $(date)"
echo "Python: $(which python)"
echo "Working dir: $(pwd)"

python -u validation/syngo/scripts/syngo_audit.py

echo "Finished at $(date)"
