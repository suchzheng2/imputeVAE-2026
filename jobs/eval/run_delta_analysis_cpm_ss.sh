#!/bin/bash
#SBATCH --job-name=delta_cpm_ss
#SBATCH --partition=priority_gpu
#SBATCH --account=prio_gerstein
#SBATCH --gpus=a100:1
#SBATCH --mem=400G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=jobs/logs/slurm-%j_delta_cpm_ss.out
#SBATCH --error=jobs/logs/slurm-%j_delta_cpm_ss.err

source ~/.bashrc
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

# Delta analysis for CPM single-stage cVAE (job 5442588, normloss fix)
RUN=run/run_2026-05-20-00-46-21_2182434

python analysis/delta_analysis.py "$RUN" \
    --output_dir "$RUN/delta_analysis" \
    --device cuda
