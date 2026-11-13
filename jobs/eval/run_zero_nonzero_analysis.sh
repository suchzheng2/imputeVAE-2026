#!/bin/bash
#SBATCH --job-name=zero_nz
#SBATCH --partition=scavenge_gpu
#SBATCH --requeue
#SBATCH --gpus=1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=jobs/logs/slurm-%j_zero_nz.out
#SBATCH --error=jobs/logs/slurm-%j_zero_nz.err

source ~/.bashrc
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

# Delete stale bytecode
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

python -m analysis.zero_nonzero_analysis \
    --model pct5v2_cond \
    --out notes/experiments/zero_nonzero_analysis
