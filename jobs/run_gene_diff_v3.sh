#!/bin/bash
#SBATCH --job-name=gene_diff_v3
#SBATCH --partition=pi_gerstein
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=jobs/logs/gene_diff_v3_%j.out
#SBATCH --error=jobs/logs/gene_diff_v3_%j.err

source ~/.bashrc
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
python analysis/cell_vs_nuclei_gene_diff_v3.py
