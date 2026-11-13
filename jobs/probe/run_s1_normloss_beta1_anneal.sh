#!/bin/bash
#SBATCH --job-name=s1_nl_b1a
#SBATCH --partition=pi_gerstein,scavenge_gpu
#SBATCH --gpus=1
#SBATCH --mem=400G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=jobs/logs/slurm-%j_s1_normloss_beta1_anneal.out
#SBATCH --error=jobs/logs/slurm-%j_s1_normloss_beta1_anneal.err
#SBATCH --requeue

source ~/.bashrc
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

# Delete stale bytecode
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# Stage 1 probe: Normalized losses, beta=1.0, WITH annealing
# Recon scaled by epoch-0 value, JSD scaled by ln(2) -> both ~[0,1]
# JSD anneals from 0 to beta=1.0 over epochs 0-50
# Config B gene list (10,134 genes), log1p normalization

python train.py \
  -dp "10X_cells_v2_v3_AIBS/" \
  --model "vae_js" \
  --batch "c3A" \
  --protocol "v3" \
  --use_raw_data True \
  --raw_data_normalization "log1p" \
  --internal_size "10,9" \
  --latent_size 5 \
  --num_epochs 200 \
  --batch_size 128 \
  --learning_rate 1e-4 \
  --beta 1.0 \
  --kl_anneal True \
  --kl_anneal_start_epoch 0 \
  --kl_anneal_epoch 50 \
  --recons_loss_type 0 \
  --clamp_recons_loss False \
  --normalize_losses True \
  --subset True \
  --subset_ratio 0.2 \
  --gene_list "data/training_gene_list_v3.csv" \
  --valid True \
  --test True \
  --debug True
