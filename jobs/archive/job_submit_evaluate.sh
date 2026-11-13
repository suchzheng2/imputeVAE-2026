#!/bin/bash
#SBATCH --job-name=eval-run
#SBATCH --partition=bigmem
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --output=jobs/logs/eval_%j.out
#SBATCH --error=jobs/logs/eval_%j.err
#SBATCH --mail-type=ALL

# Usage:
#   sbatch jobs/job_submit_evaluate.sh <run_folder_name> [modality] [analyses]
#
# Examples:
#   sbatch jobs/job_submit_evaluate.sh run_2026-03-04-16-37-04_2199483
#   sbatch jobs/job_submit_evaluate.sh run_2026-03-04-16-37-04_2199483 nuclei all
#   sbatch jobs/job_submit_evaluate.sh run_2026-03-04-16-37-04_2199483 nuclei loss,scale,correlation

module load miniconda
conda activate impute2

RUN_FOLDER=${1:?"Error: must provide run folder name as first argument"}
MODALITY=${2:-"nuclei"}
ANALYSES=${3:-"all"}

cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

python analysis/evaluate.py \
    --run_folder "$RUN_FOLDER" \
    --modality "$MODALITY" \
    --analyses "$ANALYSES"
