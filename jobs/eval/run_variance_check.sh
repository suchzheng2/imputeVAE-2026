#!/bin/bash
#SBATCH --job-name=var_check
#SBATCH --partition=pi_gerstein
#SBATCH --cpus-per-task=2
#SBATCH --mem=400G
#SBATCH --time=01:00:00
#SBATCH --output=jobs/logs/slurm-%j_var_check.out
#SBATCH --error=jobs/logs/slurm-%j_var_check.err

cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
module load miniconda
conda activate impute2

# Run all log1p models in one go (share data load)
python analysis/quick_variance_check.py \
    run/run_2026-05-11-19-52-25_134850 \
    run/run_2026-05-08-00-23-35_3604390

echo ""
echo "=== CPM model (separate data load) ==="
python analysis/quick_variance_check.py \
    run/run_2026-05-11-04-43-20_620735

echo "Done."
