#!/bin/bash
#SBATCH --job-name=qual_fig1
#SBATCH --partition=scavenge
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=jobs/logs/slurm-%j_qual_fig1.out
#SBATCH --error=jobs/logs/slurm-%j_qual_fig1.err

source ~/.bashrc
conda activate impute2

cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
python analysis/qual_fig1.py
