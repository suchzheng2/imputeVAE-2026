#!/bin/bash
#SBATCH --job-name=trap_final_stats
#SBATCH --partition=scavenge
#SBATCH --time=4:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --output=jobs/logs/slurm-%j_trap_final_stats.out
#SBATCH --error=jobs/logs/slurm-%j_trap_final_stats.err
module load miniconda
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
T=validation/trap_ribotag_external_validation
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
echo "=== 08 H2/H3 final (1000) ==="; python $T/scripts/08_profile_improvement.py --n_perm 1000 2>&1 | grep -v "Warning\|warn(msg)" | tee $T/outputs/logs/08_final.log
echo "=== 07 H1/H4/H5 final (1000) ==="; python $T/scripts/07_primary_concordance.py --datasets GSE210071,GSE121199 --n_perm 1000 2>&1 | grep -v Warning | tee $T/outputs/logs/07_final.log
echo "done $(date)"
