#!/bin/bash
#SBATCH --job-name=delta_analysis
#SBATCH --partition=priority
#SBATCH --account=prio_gerstein
#SBATCH --mem=200G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=jobs/logs/slurm-%j_delta_analysis.out
#SBATCH --error=jobs/logs/slurm-%j_delta_analysis.err

source ~/.bashrc
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

# Best model: log1p, learned 8d embed, dec-only (job 4514645)
RUN=run/run_2026-05-08-00-23-35_3604390
OUT=notes/experiments/delta_analysis

python -u analysis/delta_analysis.py "$RUN" --output_dir "$OUT"
