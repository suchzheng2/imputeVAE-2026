#!/bin/bash
#SBATCH --job-name=jaspar_tf
#SBATCH --partition=scavenge
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=jobs/logs/slurm-%j_jaspar_tf.out
#SBATCH --error=jobs/logs/slurm-%j_jaspar_tf.err

cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

# Clean stale bytecode
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

source activate impute2

echo "Starting JASPAR TF validation: $(date)"
python -u validation/tf/scripts/jaspar_tf_validation.py
echo "Finished: $(date)"
