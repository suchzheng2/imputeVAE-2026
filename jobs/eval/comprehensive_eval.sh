#!/bin/bash
#SBATCH --job-name=comprehensive_eval
#SBATCH --output=jobs/logs/slurm-%j_comprehensive_eval.out
#SBATCH --error=jobs/logs/slurm-%j_comprehensive_eval.err
#SBATCH --partition=scavenge_gpu
#SBATCH --mem=300G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate impute2

cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

# Delete stale bytecode
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

python analysis/comprehensive_eval.py 2>&1
