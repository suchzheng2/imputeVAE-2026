#!/bin/bash

#SBATCH --job-name=s3-c2n-v3
#SBATCH --out="jobs/logs/slurm-%j_stage3_cell2nuclei_v3.out"
#SBATCH --partition=gpu,pi_gerstein,scavenge_gpu
#SBATCH --gpus=1
#SBATCH --time=18:00:00
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
# Setup 1 / Stage 3:
#   stage 1: cell
#   stage 2: nuclei
#   stage 3: both modalities + alignment JSD
#
# Loss:
#   nuclei reconstruction
#   + nuclei prior JSD
#   + cell reconstruction
#   + cell prior JSD
#   + alignment JSD
# Recommended anchor:
#   keep cell reconstruction heavier than nuclei reconstruction
# ============================================================

CELL_STAGE1_MODEL="${CELL_STAGE1_MODEL:-run/run_2026-03-12-17-29-06_928562/models/model.pt}"
NUCLEI_STAGE2_MODEL="${NUCLEI_STAGE2_MODEL:-run/run_2026-03-13-00-55-11_1260842/models/model.pt}"

python train.py \
   -dp "10X_nuclei_v2_v3_AIBS/" \
   --model "vae_nb" \
   -b "n3A" \
   --protocol "v3" \
   --use_raw_data True \
   --raw_data_normalization "raw" \
   --fine_tune_stage3_v2 True \
   --stage1_model_run "$CELL_STAGE1_MODEL" \
   --stage2_model_run "$NUCLEI_STAGE2_MODEL" \
   --internal_size "10,9" \
   --latent_size 5 \
   --num_epochs 600 \
   --cell_recons_weight 1.0 \
   --cell_jsd_weight 0.5 \
   --nuclei_recons_weight 1.0 \
   --nuclei_jsd_weight 0.5 \
   --alignment_weight 1.0 \
   --anneal_cell_jsd_weight False \
   --anneal_nuclei_jsd_weight False \
   --anneal_alignment True \
   --anneal_alignment_start_epoch 0 \
   --anneal_alignment_epoch 100 \
   --batch_size 128 \
   --learning_rate 1e-4 \
   --subset True \
   --subset_ratio 0.2 \
   --exclude_mitocarta True \
   --stage3_save_epoch_dump True \
   --stage3_dump_every 50 \
   --stage3_skip_full_eval True \
   --debug True
