#!/bin/bash
#SBATCH --job-name=delta_valid
#SBATCH --partition=pi_gerstein
#SBATCH --time=02:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=4
#SBATCH --output=jobs/logs/slurm-%j_delta_valid.out
#SBATCH --error=jobs/logs/slurm-%j_delta_valid.err

module purge
source ~/.bashrc
conda activate impute2

RUN_FOLDER="run/run_2026-05-11-19-52-25_134850"
OUTPUT_DIR="notes/experiments/delta_analysis/valid_set"

echo "Running validation delta analysis on $RUN_FOLDER"
echo "Output: $OUTPUT_DIR"
echo "Start: $(date)"

python analysis/delta_analysis_validation.py \
    "$RUN_FOLDER" \
    --output_dir "$OUTPUT_DIR" \
    --split valid \
    --device cpu

echo "Done: $(date)"
