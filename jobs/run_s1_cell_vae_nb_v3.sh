#!/bin/bash
#SBATCH --job-name=s1_cell_nb_v3
#SBATCH --partition=pi_gerstein,scavenge_gpu
#SBATCH --gpus=1
#SBATCH --mem=400G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=jobs/logs/slurm-%j_s1_cell_vae_nb_v3.out
#SBATCH --error=jobs/logs/slurm-%j_s1_cell_vae_nb_v3.err

source ~/.bashrc
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

# Stage 1: Cell-only pretraining with NB loss + delayed/annealed JSD
# Same schedule as vae_js run for fair comparison:
#   Epochs   0-299: reconstruction only (beta=0)
#   Epochs 300-399: JSD anneals from 0 → 0.5
#   Epochs 400-599: full reconstruction + JSD (beta=0.5)

python train.py \
  -dp "10X_cells_v2_v3_AIBS/" \
  --model "vae_nb" \
  --batch "c3A" \
  --protocol "v3" \
  --use_raw_data True \
  --raw_data_normalization "raw" \
  --internal_size "10,9" \
  --latent_size 5 \
  --num_epochs 600 \
  --batch_size 128 \
  --learning_rate 1e-4 \
  --beta 0.5 \
  --kl_anneal True \
  --kl_anneal_start_epoch 300 \
  --kl_anneal_epoch 100 \
  --l1_coef 0.01 \
  --subset True \
  --subset_ratio 0.2 \
  --gene_list "data/training_gene_list_v3.csv" \
  --valid True \
  --test True \
  --debug True
