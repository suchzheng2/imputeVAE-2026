#!/bin/bash
#SBATCH --job-name=cmp_s1
#SBATCH --partition=pi_gerstein,scavenge_gpu
#SBATCH --gpus=1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=jobs/logs/slurm-%j_compare_stage1.out
#SBATCH --error=jobs/logs/slurm-%j_compare_stage1.err

source ~/.bashrc
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

python analysis/compare_stage1_models.py
