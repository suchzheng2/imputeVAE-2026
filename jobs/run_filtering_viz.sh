#!/bin/bash
#SBATCH --job-name=filter_viz
#SBATCH --partition=pi_gerstein
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=jobs/logs/filter_viz_%j.out
#SBATCH --error=jobs/logs/filter_viz_%j.err

source ~/.bashrc
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
python analysis/visualize_filtering_cutoffs_v3.py
