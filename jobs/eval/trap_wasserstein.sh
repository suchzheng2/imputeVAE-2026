#!/bin/bash
#SBATCH --job-name=trap_wasserstein
#SBATCH --output=jobs/logs/slurm-%j_trap_wasserstein.out
#SBATCH --error=jobs/logs/slurm-%j_trap_wasserstein.err
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00

set -euo pipefail

echo "=== TRAP Validation with Wasserstein Distance ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo ""

cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

# Clean pycache
find imputeVAE/ analysis/ -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

source activate impute2

python analysis/trap_crossvalidation.py \
    --output_dir compartment_classification/trap_validation

echo ""
echo "=== Done: $(date) ==="
