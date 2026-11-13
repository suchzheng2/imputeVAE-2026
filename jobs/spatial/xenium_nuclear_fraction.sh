#!/bin/bash
#SBATCH --job-name=xen_nucfrac
#SBATCH --partition=scavenge
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=jobs/logs/slurm-%j_xen_nucfrac.out
#SBATCH --error=jobs/logs/slurm-%j_xen_nucfrac.err
# Per-gene / per-cell nuclear fraction of Xenium Prime 5K transcripts (2026-11-13); see analysis/spatial/xenium_nuclear_fraction.py
source ~/.bashrc 2>/dev/null || true
conda activate impute2; export PYTHONUNBUFFERED=1
set -e
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
find analysis -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
python analysis/spatial/xenium_nuclear_fraction.py
echo "=== Done: $(date)"
