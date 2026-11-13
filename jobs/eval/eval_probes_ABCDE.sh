#!/bin/bash

#SBATCH --job-name=eval-probes
#SBATCH --out="jobs/logs/slurm-%j_eval_probes_ABCDE.out"
#SBATCH --partition=pi_gerstein,scavenge
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=2
#SBATCH --requeue
#SBATCH --mem=300G
#SBATCH --mail-type=ALL

module load miniconda
source activate impute2

# ============================================================
# Quick evaluation for all 5 probe runs (A-E)
# Generates epoch dump from saved model, then runs UMAPs,
# heatmaps, and correlation analysis. Skips scIB by default.
# ============================================================

PROBES=(
    "A:run_2026-03-20-10-27-02_2073068"
    "B:run_2026-03-20-10-57-36_2707095"
    "C:run_2026-03-20-10-26-23_2685978"
    "D:run_2026-03-20-10-27-02_2958891"
    "E:run_2026-03-20-10-27-02_3292804"
)

for entry in "${PROBES[@]}"; do
    PROBE="${entry%%:*}"
    RUN="${entry##*:}"
    echo ""
    echo "========================================"
    echo "  Probe $PROBE: $RUN"
    echo "========================================"
    python -u analysis/eval_probe.py \
        --run_folder "$RUN" \
        --skip_scib \
        --delete_dump
    echo "  Probe $PROBE done (exit code: $?)"
    echo ""
done

echo "All probes evaluated."
