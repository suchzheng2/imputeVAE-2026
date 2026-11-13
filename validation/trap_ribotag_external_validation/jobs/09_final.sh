#!/bin/bash
#SBATCH --job-name=trap_secondary
#SBATCH --partition=scavenge
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=jobs/logs/slurm-%j_trap_secondary.out
#SBATCH --error=jobs/logs/slurm-%j_trap_secondary.err
module load miniconda
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE/validation/trap_ribotag_external_validation/scripts
export PYTHONUNBUFFERED=1
python 09_secondary_datasets.py --n_perm 1000 2>&1 | grep -v "Warning\|warn(" | tee ../outputs/logs/09_final.log
