#!/bin/bash
#SBATCH --job-name=post-s3-n2c
#SBATCH --partition=bigmem
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --output=jobs/logs/post_stage3_setup2_%j.out
#SBATCH --error=jobs/logs/post_stage3_setup2_%j.err
#SBATCH --mail-type=ALL

set -euo pipefail

module load miniconda
conda activate impute2

cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

STAGE3_JOB_ID="${1:?need stage3 job id}"
LOG_PATH="jobs/logs/slurm-${STAGE3_JOB_ID}_stage3_nuclei2cell_v3.out"
RUN_PATH="$(grep -o '/gpfs[^ ]*/run/run_[^ ]*' "$LOG_PATH" | head -n 1)"

if [[ -z "${RUN_PATH}" ]]; then
  echo "Could not determine run path from ${LOG_PATH}"
  exit 1
fi

RUN_FOLDER="$(basename "$RUN_PATH")"

python3 analysis/evaluate.py --run_folder "$RUN_FOLDER" --modality cell --analyses all
