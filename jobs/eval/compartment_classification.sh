#!/bin/bash
#SBATCH --job-name=compartment_class
#SBATCH --output=jobs/logs/slurm-%j_compartment_class.out
#SBATCH --error=jobs/logs/slurm-%j_compartment_class.err
#SBATCH --partition=pi_gerstein
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00

set -euo pipefail

echo "=== Compartment Classification ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo ""

cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

# Clean pycache
find imputeVAE/ analysis/ -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

source activate impute2

python analysis/compartment_classification.py \
    --output_dir compartment_classification \
    --taus 0 0.1 0.2

echo ""
echo "=== Done: $(date) ==="
