#!/bin/bash
#SBATCH --job-name=probe_cpm_sc001
#SBATCH --partition=pi_gerstein,scavenge_gpu
#SBATCH --gpus=1
#SBATCH --mem=400G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=jobs/logs/slurm-%j_probe_cpm_softcon_l001.out
#SBATCH --error=jobs/logs/slurm-%j_probe_cpm_softcon_l001.err

source ~/.bashrc
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# Probe: does soft constraint (lambda=0.01) un-stick JSD for CPM model?
# Starting from existing CPM stage 1.  Only 100 epochs — watching JSD values.
# If JSD drops below ln(2)=0.693, the fix works.
#
# CPM recons loss ~1.5-2.2, mu values potentially large (~50+).
# lambda=0.01: penalty = 0.01 * mu^2.  For mu=10: 1.0.  For mu=50: 25.
# Should be strong enough to pull mu back toward 0.

STAGE1_MODEL="run/run_2026-04-23-23-59-34_1163351/models/model.pt"

echo "Stage 1 model: ${STAGE1_MODEL}"

python train.py \
  -dp "10X_nuclei_v2_v3_AIBS/" \
  --model "vae_js" \
  --batch "n3A" \
  --protocol "v3" \
  --use_raw_data True \
  --raw_data_normalization "cpm" \
  --fine_tune_cond_stage3 True \
  --stage1_model_run "$STAGE1_MODEL" \
  --embed_dim 8 \
  --internal_size "10,9" \
  --latent_size 5 \
  --num_epochs 100 \
  --batch_size 128 \
  --learning_rate 1e-4 \
  --beta 0.1 \
  --recons_loss_type 0 \
  --l1_coef 0.01 \
  --subset True \
  --subset_ratio 0.2 \
  --gene_list "data/gene_list_perct_5pct.csv" \
  --valid True \
  --test True \
  --clamp_recons_loss False \
  --debug False \
  \
  --use_soft_constraint True \
  --soft_constraint_lambda 0.01 \
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
  --stage3_eval_every 10 \
  --stage3_save_epoch_dump False \
  --stage3_skip_full_eval True
