#!/bin/bash
#SBATCH --job-name=trap_pseudobulk
#SBATCH --partition=scavenge
#SBATCH --time=2:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --output=jobs/logs/slurm-%j_trap_pseudobulk.out
#SBATCH --error=jobs/logs/slurm-%j_trap_pseudobulk.err
set -euo pipefail
module load miniconda
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
echo "Job $SLURM_JOB_ID on $(hostname), start $(date)"
python validation/trap_ribotag_external_validation/scripts/01_build_model_pseudobulk.py
echo "end $(date)"
