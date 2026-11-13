#!/bin/bash
#SBATCH --job-name=eval_pct5_cond_xr
#SBATCH --partition=pi_gerstein
#SBATCH --mem=200G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=jobs/logs/slurm-%j_eval_pct5_cond_xr.out
#SBATCH --error=jobs/logs/slurm-%j_eval_pct5_cond_xr.err

source ~/.bashrc
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

# Delete stale bytecode
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# PerCT_5pct Stage 2 conditioned decoder + cross-recons
python analysis/evaluate.py \
  --run_folder run_2026-04-03-03-44-41_2957436 \
  --modality nuclei \
  --analyses loss,scale,correlation,gene_corr,heatmaps \
  --epochs 0,50,100,200,300,400,500,599

echo "PerCT_5pct cond+XR eval complete."
