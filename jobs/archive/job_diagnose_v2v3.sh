#!/bin/bash
#SBATCH --job-name=v2v3-diag
#SBATCH --partition=bigmem
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --output=jobs/logs/v2v3_diag_%j.out
#SBATCH --error=jobs/logs/v2v3_diag_%j.err
#SBATCH --mail-type=ALL

# Usage:
#   sbatch jobs/job_diagnose_v2v3.sh <run_path> [epochs...]
#
# Examples:
#   sbatch jobs/job_diagnose_v2v3.sh /home/sz477/project/cell-imputing/Cytoplasm_imputation/run/run_2026-03-04-12-59-01_308825
#   sbatch jobs/job_diagnose_v2v3.sh /home/sz477/project/cell-imputing/Cytoplasm_imputation/run/run_2026-03-04-12-59-01_308825 200 350 500

module load miniconda
conda activate impute2

RUN_PATH=${1:?"Error: must provide full run path as first argument"}
shift
EPOCHS=("$@")

cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

if [ ${#EPOCHS[@]} -gt 0 ]; then
    python analysis/diagnose_v2v3_batch.py \
        --run_path "$RUN_PATH" \
        --epochs "${EPOCHS[@]}"
else
    python analysis/diagnose_v2v3_batch.py \
        --run_path "$RUN_PATH"
fi
