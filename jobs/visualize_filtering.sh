#!/bin/bash

#SBATCH --job-name=filter-viz
#SBATCH --out="jobs/logs/slurm-%j_filter_viz.out"
#SBATCH --partition=pi_gerstein,scavenge
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=200G
#SBATCH --mail-type=ALL

module load miniconda
source activate impute2

python analysis/visualize_filtering_cutoffs.py --output_dir analysis/filtering_cutoffs
