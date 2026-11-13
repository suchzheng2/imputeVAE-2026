#!/bin/bash
#SBATCH --job-name=frac_valid
#SBATCH --output=jobs/logs/slurm-%j_frac_valid.out
#SBATCH --error=jobs/logs/slurm-%j_frac_valid.err
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail

echo "=== Fractionation Validation ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo ""

cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

find imputeVAE/ analysis/ -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

source activate impute2

python analysis/fractionation_validation.py \
    --output_dir compartment_classification/fractionation_validation

echo ""
echo "=== Done: $(date) ==="
