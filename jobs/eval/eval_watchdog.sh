#!/bin/bash

#SBATCH --job-name=eval-watch
#SBATCH --out="jobs/logs/slurm-%j_eval_watchdog.out"
#SBATCH --partition=pi_gerstein,scavenge
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --requeue
#SBATCH --mem=300G
#SBATCH --mail-type=ALL

module load miniconda
source activate impute2

# ============================================================
# Epoch Dump Watchdog
#
# Polls a run folder for new epoch dumps, runs evaluation
# (correlation + heatmaps), saves results, and deletes the dump.
#
# Usage:
#   RUN_FOLDER=run/run_2026-03-19-... sbatch jobs/eval_watchdog.sh
#
# Optional env vars:
#   ANALYSES  - comma-separated (default: correlation,heatmaps)
#   INTERVAL  - minutes between polls (default: 10)
#   WATCH     - set to "true" for watch mode (default: true)
# ============================================================

RUN_FOLDER="${RUN_FOLDER:?ERROR: set RUN_FOLDER env var}"
ANALYSES="${ANALYSES:-correlation,heatmaps}"
INTERVAL="${INTERVAL:-10}"
WATCH="${WATCH:-true}"

WATCH_FLAG=""
if [ "$WATCH" = "true" ]; then
    WATCH_FLAG="--watch"
fi

python analysis/eval_watchdog.py \
    --run_folder "$RUN_FOLDER" \
    --analyses "$ANALYSES" \
    --interval "$INTERVAL" \
    $WATCH_FLAG
