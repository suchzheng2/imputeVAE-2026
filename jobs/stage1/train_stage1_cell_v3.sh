#!/bin/bash

#SBATCH --job-name=s1-cell-v3
#SBATCH --out="jobs/logs/slurm-%j_stage1_cell_v3.out"
#SBATCH --partition=gpu,pi_gerstein
#SBATCH --gpus=1
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=1
#SBATCH --requeue
#SBATCH --mem=300G
#SBATCH --mail-type=ALL

module load miniconda
conda activate impute2

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export CUDA_VISIBLE_DEVICES=0

# Sanity check: fail fast if PyTorch can't see the GPU
python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT AVAILABLE - wrong PyTorch build!'"

# ============================================================
# Stage 1 (combined 1.1+1.2) - Cell, v3 only, NB VAE
# KL annealing: beta=0 for epochs 0-199, ramp to 0.5 over 200-399
# ============================================================

python train.py \
   -dp "10X_cells_v2_v3_AIBS/" \
   --model "vae_nb" \
   -b "c3A" \
   --protocol "v3" \
   --use_raw_data True \
   --raw_data_normalization "raw" \
   --internal_size "10,9" \
   --latent_size 5 \
   --num_epochs 600 \
   --beta 0.5 \
   --kl_anneal True \
   --kl_anneal_start_epoch 200 \
   --kl_anneal_epoch 400 \
   --batch_size 128 \
   --learning_rate 1e-4 \
   --subset True \
   --subset_ratio 0.2 \
   --exclude_mitocarta True \
   --debug True
