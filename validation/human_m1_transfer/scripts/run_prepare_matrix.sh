#!/bin/bash
#SBATCH --job-name=prep_human_m1
#SBATCH --partition=scavenge
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --output=/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE/jobs/logs/slurm-%j_prep_human_m1.out
#SBATCH --error=/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE/jobs/logs/slurm-%j_prep_human_m1.err

cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
source activate impute2

python validation/human_m1_transfer/scripts/03_prepare_model_matrix.py
