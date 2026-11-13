#!/bin/bash

#SBATCH --job-name=s3-a1.0
#SBATCH --out="jobs/logs/slurm-%j_s3_align_1.0.out"
#SBATCH --partition=gpu,pi_gerstein,scavenge_gpu
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --requeue
#SBATCH --mem=500G
#SBATCH --mail-type=ALL

module load miniconda
conda activate impute2

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export CUDA_VISIBLE_DEVICES=0

python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT AVAILABLE'"

# ============================================================
# Full run: alignment_weight=1.0, annealed over 100 epochs
# Standard weights: recons=1.0, jsd=0.5 for both modalities
# ============================================================

CELL_STAGE1_MODEL="run/run_2026-03-12-17-29-06_928562/models/model.pt"

python train.py \
   -dp "10X_nuclei_v2_v3_AIBS/" \
   --model "vae_nb" \
   -b "n3A" \
   --protocol "v3" \
   --use_raw_data True \
   --raw_data_normalization "raw" \
   --fine_tune_stage3_v2 True \
   --stage1_model_run "$CELL_STAGE1_MODEL" \
   --internal_size "10,9" \
   --latent_size 5 \
   --num_epochs 300 \
   --cell_recons_weight 1.0 \
   --cell_jsd_weight 0.5 \
   --nuclei_recons_weight 1.0 \
   --nuclei_jsd_weight 0.5 \
   --alignment_weight 1.0 \
   --anneal_alignment True \
   --anneal_alignment_start_epoch 0 \
   --anneal_alignment_epoch 100 \
   --cross_recons_weight 0.0 \
   --batch_size 128 \
   --learning_rate 1e-4 \
   --exclude_mitocarta True \
   --stage3_save_epoch_dump True \
   --stage3_dump_every 50 \
   --stage3_eval_every 50 \
   --stage3_skip_full_eval True
