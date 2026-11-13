#!/bin/bash
#SBATCH --job-name=trap_extension
#SBATCH --partition=scavenge
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=jobs/logs/slurm-%j_trap_extension.out
#SBATCH --error=jobs/logs/slurm-%j_trap_extension.err
module load miniconda
conda activate impute2
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE/validation/trap_ribotag_external_validation/scripts
python 14_build_baseline_pseudobulks.py 2>&1 | tee ../outputs/extension/logs/14_baseline_pseudobulks.log
python 15_extension_h2_h3.py --n_perm 1000 2>&1 | grep -v "Warning\|warn(" | tee ../outputs/extension/logs/15_h2_h3_final.log
