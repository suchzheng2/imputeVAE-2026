#!/bin/bash
#SBATCH --job-name=yscorch_inspect
#SBATCH --partition=scavenge
#SBATCH --mem=120G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=jobs/logs/slurm-%j_yscorch_inspect.out
#SBATCH --error=jobs/logs/slurm-%j_yscorch_inspect.err
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
module load R/4.4.1-foss-2022b R-bundle-Bioconductor/3.19-foss-2022b-R-4.4.1
Rscript notes/experiments/yscorch_scoping/inspect_rds.R
