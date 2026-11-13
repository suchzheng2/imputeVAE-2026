#!/bin/bash
#SBATCH --job-name=named_lncrna
#SBATCH --partition=pi_gerstein
#SBATCH --mem=100G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=jobs/logs/slurm-%j_named_lncrna.out
#SBATCH --error=jobs/logs/slurm-%j_named_lncrna.err

source ~/.bashrc
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

python analysis/named_lncrna_localization.py

echo "Done — $(date)"
