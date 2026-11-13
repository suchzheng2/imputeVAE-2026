#!/bin/bash
#SBATCH --job-name=eval_nb_s2_cond
#SBATCH --partition=pi_gerstein,scavenge_gpu
#SBATCH --gpus=1
#SBATCH --mem=400G
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
#SBATCH --output=jobs/logs/slurm-%j_eval_nb_s2_cond.out
#SBATCH --error=jobs/logs/slurm-%j_eval_nb_s2_cond.err

source ~/.bashrc
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

# NB Stage 2 conditioned decoder (joint cell+nuclei, Config B, 10134 genes)
python analysis/evaluate.py \
  --run_folder run_2026-04-02-02-02-36_722169 \
  --modality nuclei \
  --analyses loss,scale,correlation,gene_corr,heatmaps \
  --epochs 0,50,100,200,300,400,500,599

echo "NB stage2 conditioned decoder eval complete."
