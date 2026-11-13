#!/bin/bash
#SBATCH --job-name=delta_full
#SBATCH --partition=pi_gerstein
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=jobs/logs/slurm-%j_delta_full.out
#SBATCH --error=jobs/logs/slurm-%j_delta_full.err

source ~/.bashrc
conda activate impute2

cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
export PYTHONUNBUFFERED=1

python analysis/delta_analysis_from_precomputed.py \
    --imputed run/cvae_full_eval/imputed_cell_from_nucleus.npy \
    --nucleus_obs run/cvae_full_eval/nucleus_obs.csv \
    --run_folder run/run_2026-07-18-19-40-51_570984 \
    --output_dir notes/experiments/delta_analysis_full
