#!/bin/bash
#SBATCH --job-name=libsize_analysis
#SBATCH --partition=pi_gerstein,scavenge
#SBATCH --time=02:00:00
#SBATCH --mem=150G
#SBATCH --cpus-per-task=4
#SBATCH --output=jobs/logs/slurm-%j_libsize_analysis.out
#SBATCH --error=jobs/logs/slurm-%j_libsize_analysis.err

cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

source ~/.bashrc
conda activate impute2

python -m analysis.library_size_analysis --model pct5v2_cond
