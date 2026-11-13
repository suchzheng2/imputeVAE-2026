#!/bin/bash
#SBATCH --job-name=recon_eval
#SBATCH --output=jobs/logs/slurm-%j_recon_eval.out
#SBATCH --error=jobs/logs/slurm-%j_recon_eval.err
#SBATCH --partition=scavenge_gpu
#SBATCH --mem=200G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate impute2

cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

python analysis/recon_eval.py 2>&1
