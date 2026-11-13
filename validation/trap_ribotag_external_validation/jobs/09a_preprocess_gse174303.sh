#!/bin/bash
#SBATCH --job-name=trap_gse174303_R
#SBATCH --partition=scavenge
#SBATCH --time=0:20:00
#SBATCH --mem=8G
#SBATCH --output=jobs/logs/slurm-%j_trap_gse174303_R.out
#SBATCH --error=jobs/logs/slurm-%j_trap_gse174303_R.err
module purge
module load R-bundle-Bioconductor/3.18-foss-2022b-R-4.3.2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
Rscript validation/trap_ribotag_external_validation/scripts/09a_preprocess_gse174303.R 2>&1 | tee validation/trap_ribotag_external_validation/outputs/logs/09a_gse174303.log
