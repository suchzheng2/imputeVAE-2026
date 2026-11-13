#!/bin/bash
#SBATCH --job-name=wilcoxon_fulldata
#SBATCH --partition=scavenge_gpu
#SBATCH --time=1:00:00
#SBATCH --mem=400G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --output=jobs/logs/slurm-%j_wilcoxon_fulldata.out
#SBATCH --error=jobs/logs/slurm-%j_wilcoxon_fulldata.err

source ~/.bashrc
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
export PYTHONUNBUFFERED=1

find imputeVAE/ analysis/ -name "__pycache__" -exec rm -rf {} + 2>/dev/null

python analysis/wilcoxon_gene_test.py \
    --run_folder run/run_2026-05-20-00-46-21_2182434 \
    --output_dir notes/eval_results/fulldata_cpm_scalar_cvae/wilcoxon
