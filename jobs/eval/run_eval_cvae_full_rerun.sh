#!/bin/bash
#SBATCH --job-name=eval_cvae_rerun
#SBATCH --partition=priority_gpu
#SBATCH --account=prio_gerstein
#SBATCH --gpus=a5000:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=jobs/logs/slurm-%j_eval_cvae_rerun.out
#SBATCH --error=jobs/logs/slurm-%j_eval_cvae_rerun.err

source ~/.bashrc
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
export PYTHONUNBUFFERED=1

find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# Re-run only bio step (updated nuclear lncRNA list)
# Skip everything else — already computed
python analysis/evaluate_baseline_incremental.py \
    --imputed_npy run/cvae_full_eval/imputed_cell_from_nucleus.npy \
    --obs_csv run/cvae_full_eval/nucleus_obs.csv \
    --name cvae_full_v2 \
    --gene_list data/training_gene_list_v3.csv \
    --output_dir run/cvae_full_eval/eval \
    --skip clf,de,coexp,sample,labelfree
