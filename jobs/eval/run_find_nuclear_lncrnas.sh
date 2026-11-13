#!/bin/bash
#SBATCH --job-name=find_lncrna
#SBATCH --partition=priority_gpu
#SBATCH --account=prio_gerstein
#SBATCH --gpus=a5000:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=jobs/logs/slurm-%j_find_lncrna.out
#SBATCH --error=jobs/logs/slurm-%j_find_lncrna.err

source ~/.bashrc
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
export PYTHONUNBUFFERED=1

python analysis/find_nuclear_lncrnas.py \
    --gene_list data/training_gene_list_v3.csv \
    --output_dir notes/experiments/nuclear_lncrna_analysis
