#!/bin/bash
#SBATCH --job-name=eval_scvi_rerun
#SBATCH --partition=scavenge
#SBATCH --requeue
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=jobs/logs/slurm-%j_eval_scvi_rerun.out
#SBATCH --error=jobs/logs/slurm-%j_eval_scvi_rerun.err

source ~/.bashrc
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
export PYTHONUNBUFFERED=1

find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# Re-run only bio step (updated nuclear lncRNA list)
python analysis/evaluate_baseline_incremental.py \
    --imputed_npy run/scvi_baseline/imputed_cell_from_nucleus.npy \
    --obs_csv run/scvi_baseline/nucleus_obs.csv \
    --name scvi_v2 \
    --gene_list data/training_gene_list_v3.csv \
    --output_dir run/scvi_baseline/eval \
    --skip clf,de,coexp,sample,labelfree
