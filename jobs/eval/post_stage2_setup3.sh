#!/bin/bash
#SBATCH --job-name=post-s2-c2both
#SBATCH --partition=bigmem
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --output=jobs/logs/post_stage2_setup3_%j.out
#SBATCH --error=jobs/logs/post_stage2_setup3_%j.err
#SBATCH --mail-type=ALL

set -euo pipefail

module load miniconda
conda activate impute2

cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

STAGE2_JOB_ID="${1:?need stage2 job id}"
LOG_PATH="jobs/logs/slurm-${STAGE2_JOB_ID}_stage2_cell2both_v3.out"
RUN_PATH="$(grep -o '/gpfs[^ ]*/run/run_[^ ]*' "$LOG_PATH" | head -n 1)"

if [[ -z "${RUN_PATH}" ]]; then
  echo "Could not determine run path from ${LOG_PATH}"
  exit 1
fi

RUN_FOLDER="$(basename "$RUN_PATH")"

python3 analysis/evaluate.py --run_folder "$RUN_FOLDER" --modality nuclei --analyses all
