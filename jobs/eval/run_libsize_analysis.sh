#!/bin/bash
#SBATCH --job-name=libsize_cmp
#SBATCH --partition=priority_gpu
#SBATCH --account=prio_gerstein
#SBATCH --gpus=a100:1
#SBATCH --mem=400G
#SBATCH --cpus-per-task=4
#SBATCH --time=3:00:00
#SBATCH --output=jobs/logs/slurm-%j_libsize_cmp.out
#SBATCH --error=jobs/logs/slurm-%j_libsize_cmp.err

source ~/.bashrc
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

# Library size analysis on two models:
# 1. Best log1p: dec-only learned 8d (job 4514645)
# 2. Best CPM: single-stage cVAE (job 5442588, normloss fix)

python analysis/libsize_comparison.py \
    --runs \
        run/run_2026-05-08-00-23-35_3604390 \
        run/run_2026-05-20-00-46-21_2182434 \
    --labels \
        "log1p_dec8d" \
        "cpm_ss_cvae" \
    --device cuda
