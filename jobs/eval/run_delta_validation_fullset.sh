#!/bin/bash
#SBATCH --job-name=delta_valid_fullset
#SBATCH --output=jobs/logs/slurm-%j_delta_valid_fullset.out
#SBATCH --error=jobs/logs/slurm-%j_delta_valid_fullset.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --partition=pi_gerstein

cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

source ~/.bashrc
conda activate impute2

echo "Starting validation delta analysis with full-dataset model..."
echo "Model: run_2026-05-20-10-18-35_2542283"
echo "Date: $(date)"

python analysis/delta_analysis_validation.py run/run_2026-05-20-10-18-35_2542283 \
    --output_dir notes/experiments/delta_analysis/valid_set_fullmodel \
    --device cpu \
    --split valid

echo "Completed at: $(date)"
