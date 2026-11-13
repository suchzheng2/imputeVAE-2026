#!/bin/bash
#SBATCH --job-name=hallucination_test
#SBATCH --output=jobs/logs/slurm-%j_hallucination_test.out
#SBATCH --error=jobs/logs/slurm-%j_hallucination_test.err
#SBATCH --partition=scavenge_gpu
#SBATCH --mem=200G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate impute2

cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

# Delete stale bytecode
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

python analysis/hallucination_test.py 2>&1
