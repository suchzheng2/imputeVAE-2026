#!/bin/bash
#SBATCH --job-name=l5et_diag
#SBATCH --partition=scavenge
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=jobs/logs/slurm-%j_l5et_diag.out
#SBATCH --error=jobs/logs/slurm-%j_l5et_diag.err

cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

source activate impute2

echo "Starting L5 ET diagnostic: $(date)"
python -u validation/l5et_diagnostic/l5et_diagnostic.py
echo "Finished: $(date)"
