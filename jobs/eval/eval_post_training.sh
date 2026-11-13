#!/bin/bash

#SBATCH --job-name=eval-post
#SBATCH --out="jobs/logs/slurm-%j_eval_post_training.out"
#SBATCH --partition=pi_gerstein,scavenge
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --requeue
#SBATCH --mem=300G
#SBATCH --mail-type=ALL

module load miniconda
source activate impute2

# ============================================================
# Post-training evaluation on epoch dumps
#
# Generates UMAPs and runs full evaluation for each epoch dump.
#
# Usage:
#   RUN_FOLDER=run_2026-03-19-... sbatch jobs/eval_post_training.sh
#
# Optional env vars:
#   EPOCHS      - comma-separated epochs (default: all)
#   SKIP_UMAP   - set to "true" to skip UMAP generation
#   SKIP_EVAL   - set to "true" to skip full evaluation
#   SKIP_SCIB   - set to "true" to skip scIB metrics
#   DELETE_DUMPS - set to "true" to delete dumps after processing
# ============================================================

RUN_FOLDER="${RUN_FOLDER:?ERROR: set RUN_FOLDER env var}"

EXTRA_FLAGS=""
if [ -n "$EPOCHS" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --epochs $EPOCHS"
fi
if [ "$SKIP_UMAP" = "true" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --skip_umap"
fi
if [ "$SKIP_EVAL" = "true" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --skip_eval"
fi
if [ "$SKIP_SCIB" = "true" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --skip_scib"
fi
if [ "$DELETE_DUMPS" = "true" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --delete_dumps"
fi

python -u analysis/eval_post_training.py \
    --run_folder "$RUN_FOLDER" \
    $EXTRA_FLAGS
