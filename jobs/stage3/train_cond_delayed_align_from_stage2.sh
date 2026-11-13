#!/bin/bash

#SBATCH --job-name=cond-s2-delay
#SBATCH --out="jobs/logs/slurm-%j_cond_delayed_align_from_stage2.out"
#SBATCH --partition=gpu,pi_gerstein,scavenge_gpu
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --requeue
#SBATCH --mem=500G
#SBATCH --mail-type=ALL

module load miniconda
source activate impute2

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export CUDA_VISIBLE_DEVICES=0

python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT AVAILABLE'"

# ============================================================
# Delayed-Alignment Conditioned Decoder — from stage2 cell2nuclei
#   Init: stage2 model (cell stage1 -> nuclei stage2, run 1260842)
#     Already knows both cell and nuclei reconstruction
#   300 epochs total
#   Epochs 0-99:   pure reconstruction (alignment_weight=0)
#     -> both decoder heads refine with modality embeddings
#   Epochs 100-149: alignment ramps 0 -> 1.5 (50-epoch anneal)
#   Epochs 150-299: alignment at 1.5 (steady state, longer training)
# ============================================================

python train.py \
   -dp "10X_nuclei_v2_v3_AIBS/" \
   --model "vae_nb" \
   -b "n3A" \
   --protocol "v3" \
   --use_raw_data True \
   --raw_data_normalization "raw" \
   --fine_tune_cond_stage3 True \
   --stage1_model_run "run/run_2026-03-13-00-55-11_1260842/models/model.pt" \
   --internal_size "10,9" \
   --latent_size 5 \
   --embed_dim 8 \
   --num_epochs 300 \
   --cell_recons_weight 1.0 \
   --cell_jsd_weight 0.5 \
   --nuclei_recons_weight 1.0 \
   --nuclei_jsd_weight 0.5 \
   --alignment_weight 1.5 \
   --anneal_alignment True \
   --anneal_alignment_start_epoch 100 \
   --anneal_alignment_epoch 50 \
   --batch_size 128 \
   --learning_rate 1e-4 \
   --subset True \
   --subset_ratio 0.1 \
   --exclude_mitocarta True \
   --stage3_eval_every 25 \
   --stage3_save_epoch_dump False \
   --stage3_skip_full_eval True \
   --debug True
