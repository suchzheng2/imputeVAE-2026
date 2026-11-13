#!/bin/bash

#SBATCH --job-name=s2-c2n-v3
#SBATCH --out="jobs/logs/slurm-%j_stage2_cell2nuclei_v3.out"
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

python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT AVAILABLE'"

# ============================================================
# Stage 2: Fine-tune CELL-trained model on NUCLEI data (v3)
#
# Strategy:
#   - Load VAE_NB model pretrained on cell data (stage 1)
#   - Fine-tune on nuclei data only
#   - Loss = nuclei reconstruction + annealed nuclei JSD
#   - This is Setup 1 / Stage 2
#
# >> BEFORE RUNNING: update CELL_MODEL if needed
# ============================================================

CELL_MODEL="run/run_2026-03-12-17-29-06_928562/models/model.pt"

python train.py \
   -dp "10X_nuclei_v2_v3_AIBS/" \
   --model "vae_nb" \
   -b "n3A" \
   --protocol "v3" \
   --use_raw_data True \
   --raw_data_normalization "raw" \
   --fine_tune_stage2_v2 True \
   --stage1_model_run "$CELL_MODEL" \
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
