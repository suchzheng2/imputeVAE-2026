#!/bin/bash
#SBATCH --job-name=gene_strat
#SBATCH --output=jobs/logs/slurm-%j_gene_strat.out
#SBATCH --error=jobs/logs/slurm-%j_gene_strat.err
#SBATCH --partition=scavenge
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=01:00:00

echo "Job started: $(date)"
echo "Node: $(hostname)"

module load miniconda
conda activate impute2

cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

find imputeVAE/ analysis/ -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null

python analysis/gene_stratified_gap_eval.py \
    --imputed_npy run/cvae_full_eval/imputed_cell_from_nucleus.npy \
    --obs_csv run/cvae_full_eval/nucleus_obs.csv

echo "Job finished: $(date)"
