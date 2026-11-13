#!/bin/bash
#SBATCH --job-name=de_per_ct
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=jobs/logs/slurm-%j_de_per_ct.out
#SBATCH --error=jobs/logs/slurm-%j_de_per_ct.err

source ~/.bashrc
conda activate impute2

cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

find imputeVAE -name "__pycache__" -exec rm -rf {} + 2>/dev/null

python analysis/presentation_de_per_ct.py --model pct5v2_cond_xr
