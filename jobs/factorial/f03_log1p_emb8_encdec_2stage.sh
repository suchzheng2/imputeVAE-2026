#!/bin/bash
#SBATCH --job-name=f03_log1p_emb8_encdec_2stage
#SBATCH --partition=priority_gpu
#SBATCH --account=prio_gerstein
#SBATCH --gpus=a100:1
#SBATCH --mem=400G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=jobs/logs/slurm-%j_f03_log1p_emb8_encdec_2stage.out
#SBATCH --error=jobs/logs/slurm-%j_f03_log1p_emb8_encdec_2stage.err

source ~/.bashrc
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# Factorial experiment run 3/16
# Norm=log1p, Cond=emb8, Target=encdec, Stage=2stage
# Epochs=400

STAGE1_MODEL="run/run_2026-05-01-12-01-55_1984590/models/model.pt"
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
  --embed_dim 8 \
  --scalar_cond False \
  --cond_encoder True \
  --internal_size "10,9" \
  --latent_size 5 \
  --num_epochs 400 \
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
  --anneal_nuclei_recons_weight True \
  --nuclei_recons_weight_start_epoch 0 \
  --nuclei_recons_weight_anneal_epoch 100 \
  --nuclei_jsd_weight 0.1 \
  --anneal_nuclei_jsd_weight True \
  --nuclei_jsd_weight_start_epoch 100 \
  --nuclei_jsd_weight_anneal_epoch 100 \
  --alignment_weight 1.0 \
  --anneal_alignment True \
  --anneal_alignment_start_epoch 200 \
  --anneal_alignment_epoch 100 \
  --cross_recons_weight 0.0 \
  \
  --stage3_eval_every 50 \
  --stage3_save_epoch_dump True \
  --stage3_dump_every 50 \
  --stage3_skip_full_eval True
