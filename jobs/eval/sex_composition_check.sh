#!/bin/bash
#SBATCH --job-name=sex_composition
#SBATCH --partition=scavenge
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=jobs/logs/slurm-%j_sex_composition.out
#SBATCH --error=jobs/logs/slurm-%j_sex_composition.err
# Sex composition of cells vs nuclei from Xist and Y-linked genes, per library, 2026-09-25.
module load miniconda
conda activate impute2
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
python analysis/sex_composition_check.py
