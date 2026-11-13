#!/bin/bash
#SBATCH --job-name=s2_fcvae
#SBATCH --partition=pi_gerstein,scavenge_gpu
#SBATCH --gpus=1
#SBATCH --mem=400G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=jobs/logs/slurm-%j_s2_full_cvae_normloss.out
#SBATCH --error=jobs/logs/slurm-%j_s2_full_cvae_normloss.err
#SBATCH --requeue

source ~/.bashrc
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# Stage 2 full cVAE: encoder + decoder conditioned, log1p + normalized losses
# Conditioning: fixed scalar 1=cell, 0=nuclei on BOTH encoder and decoder
# Stage 1: run_2026-05-01-12-01-55_1984590 (log1p normloss beta=0.1 winner)
# Config B gene list (10,134 genes), log1p
#
# Loss schedule — different from decoder-only cond:
# With encoder conditioning, the model knows which modality from the start,
# so no need for cascading anneal on reconstruction/JSD terms.
#   cell_recons (1.0)     - ON from epoch 0 (no anneal)
#   cell_jsd (0.1)        - ON from epoch 0 (no anneal)
#   nuclei_recons (1.0)   - ON from epoch 0 (no anneal)
#   nuclei_jsd (0.1)      - ON from epoch 0 (no anneal)
#   alignment_jsd (1.0)   - anneals 0->1.0 over epochs 50-100
#     (short delay so both modalities establish latent representations first)

STAGE1_MODEL="run/run_2026-05-01-12-01-55_1984590/models/model.pt"
echo "Stage 1 model: ${STAGE1_MODEL}"
if [ ! -f "$STAGE1_MODEL" ]; then
    echo "ERROR: Stage 1 model not found: ${STAGE1_MODEL}"
    exit 1
fi

python train.py \
  -dp "10X_nuclei_v2_v3_AIBS/" \
  --model "vae_js" \
  --batch "n3A" \
  --protocol "v3" \
  --use_raw_data True \
  --raw_data_normalization "log1p" \
  --fine_tune_cond_stage3 True \
  --stage1_model_run "$STAGE1_MODEL" \
  --scalar_cond True \
  --cond_encoder True \
  --internal_size "10,9" \
  --latent_size 5 \
  --num_epochs 600 \
  --batch_size 128 \
  --learning_rate 1e-4 \
  --beta 0.1 \
  --recons_loss_type 0 \
  --clamp_recons_loss False \
  --normalize_losses True \
  --l1_coef 0.01 \
  --subset True \
  --subset_ratio 0.2 \
  --gene_list "data/training_gene_list_v3.csv" \
  --valid True \
  --test True \
  --debug True \
  \
  --cell_recons_weight 1.0 \
  --cell_jsd_weight 0.1 \
  --nuclei_recons_weight 1.0 \
  --nuclei_jsd_weight 0.1 \
  --alignment_weight 1.0 \
  --anneal_alignment True \
  --anneal_alignment_start_epoch 50 \
  --anneal_alignment_epoch 50 \
  --cross_recons_weight 0.0 \
  \
  --stage3_eval_every 50 \
  --stage3_save_epoch_dump True \
  --stage3_dump_every 50 \
  --stage3_skip_full_eval True
