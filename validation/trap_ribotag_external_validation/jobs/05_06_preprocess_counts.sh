#!/bin/bash
#SBATCH --job-name=trap_counts_R
#SBATCH --partition=scavenge
#SBATCH --time=0:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=jobs/logs/slurm-%j_trap_counts_R.out
#SBATCH --error=jobs/logs/slurm-%j_trap_counts_R.err
module purge
module load R-bundle-Bioconductor/3.18-foss-2022b-R-4.3.2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
T=validation/trap_ribotag_external_validation
echo "=== 05 GSE121199 ==="; Rscript $T/scripts/05_preprocess_gse121199.R 2>&1 | tee $T/outputs/logs/05_gse121199.log
echo "=== 06 GSE263082 ==="; Rscript $T/scripts/06_preprocess_gse263082.R 2>&1 | tee $T/outputs/logs/06_gse263082.log
