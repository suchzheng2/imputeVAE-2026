#!/bin/bash
#SBATCH --job-name=post-s2-c2n
#SBATCH --partition=bigmem
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --output=jobs/logs/post_stage2_setup1_%j.out
#SBATCH --error=jobs/logs/post_stage2_setup1_%j.err
#SBATCH --mail-type=ALL

set -euo pipefail

module load miniconda
conda activate impute2

cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

STAGE2_JOB_ID="${1:?need stage2 job id}"
LOG_PATH="jobs/logs/slurm-${STAGE2_JOB_ID}_stage2_cell2nuclei_v3.out"
RUN_PATH="$(grep -o '/gpfs[^ ]*/run/run_[^ ]*' "$LOG_PATH" | head -n 1)"

if [[ -z "${RUN_PATH}" ]]; then
  echo "Could not determine run path from ${LOG_PATH}"
  exit 1
fi

RUN_FOLDER="$(basename "$RUN_PATH")"
STAGE2_MODEL="${RUN_PATH}/models/model.pt"

python3 analysis/evaluate.py --run_folder "$RUN_FOLDER" --modality nuclei --analyses all

STAGE3_SUBMIT_OUTPUT="$(sbatch --export=ALL,NUCLEI_STAGE2_MODEL="$STAGE2_MODEL" jobs/train_stage3_cell2nuclei_v3.sh)"
echo "$STAGE3_SUBMIT_OUTPUT"
STAGE3_JOB_ID="$(echo "$STAGE3_SUBMIT_OUTPUT" | awk '{print $4}')"
sbatch --dependency=afterok:"$STAGE3_JOB_ID" jobs/post_stage3_setup1.sh "$STAGE3_JOB_ID"
