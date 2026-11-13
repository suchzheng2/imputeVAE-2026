#!/bin/bash
#SBATCH --job-name=trap_ext_remap
#SBATCH --partition=scavenge
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=jobs/logs/slurm-%j_trap_ext_remap.out
#SBATCH --error=jobs/logs/slurm-%j_trap_ext_remap.err
# Re-run extension H2/H3 after the L5 mapping audit (2026-09-24): Scnn1a_L4 excluded, Rbp4 -> POOL_L5_IT_ET.
module load miniconda
conda activate impute2
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE/validation/trap_ribotag_external_validation/scripts
python 15_extension_h2_h3.py --n_perm 1000 2>&1 | grep -v "Warning\|warn(" | tee ../outputs/extension/logs/15_h2_h3_after_mapping_audit.log
python 16_extension_figures.py 2>&1 | grep -v "Warning\|warn(" | tee ../outputs/extension/logs/16_figures_after_mapping_audit.log
