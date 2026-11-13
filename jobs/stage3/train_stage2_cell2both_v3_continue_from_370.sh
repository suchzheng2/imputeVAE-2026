#!/bin/bash

#SBATCH --job-name=s2-c2both-v3-r370
#SBATCH --out="jobs/logs/slurm-%j_stage2_cell2both_v3_continue_from_370.out"
#SBATCH --partition=gpu,pi_gerstein
#SBATCH --gpus=1
#SBATCH --time=8:00:00
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
# Setup 3 / Stage 2 resume:
#   Continue the timed-out joint cell->both run from the saved
#   epoch-370 checkpoint in the original run directory.
# ============================================================

RESUME_RUN_DIR="run/run_2026-03-13-00-55-52_997499"
RESUME_MODEL="${RESUME_RUN_DIR}/models/model.pt"
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
   --stage2_model_run "$RESUME_MODEL" \
   --continue_training True \
   --continue_training_run_path "$RESUME_RUN_DIR" \
   --continue_training_epoch 371 \
   --internal_size "10,9" \
   --latent_size 5 \
   --num_epochs 600 \
   --cell_recons_weight 2.0 \
   --cell_jsd_weight 0.5 \
   --nuclei_recons_weight 1.0 \
   --nuclei_jsd_weight 0.5 \
   --alignment_weight 0.5 \
   --anneal_cell_jsd_weight True \
   --cell_jsd_weight_start_epoch 200 \
   --cell_jsd_weight_anneal_epoch 400 \
   --anneal_nuclei_jsd_weight True \
   --nuclei_jsd_weight_start_epoch 200 \
   --nuclei_jsd_weight_anneal_epoch 400 \
   --anneal_alignment True \
   --anneal_alignment_start_epoch 200 \
   --anneal_alignment_epoch 200 \
   --batch_size 128 \
   --learning_rate 1e-4 \
   --subset True \
   --subset_ratio 0.2 \
   --exclude_mitocarta True \
   --debug True
