#!/bin/bash
#SBATCH --job-name=eval_ctmean
#SBATCH --partition=pi_gerstein
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=jobs/logs/slurm-%j_eval_ctmean.out
#SBATCH --error=jobs/logs/slurm-%j_eval_ctmean.err

source ~/.bashrc
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
export PYTHONUNBUFFERED=1

find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

python analysis/evaluate_baseline_incremental.py \
    --imputed_npy run/ctmean_baseline/imputed_cell_from_nucleus.npy \
    --obs_csv run/ctmean_baseline/nucleus_obs.csv \
    --name ctmean \
    --gene_list data/training_gene_list_v3.csv \
    --output_dir run/ctmean_baseline/eval \
    --skip clf,de,coexp,sample
