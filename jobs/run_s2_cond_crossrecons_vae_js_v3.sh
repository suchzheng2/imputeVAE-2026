#!/bin/bash
#SBATCH --job-name=s2_cond_xr_js_v3
#SBATCH --partition=pi_gerstein,scavenge_gpu
#SBATCH --gpus=1
#SBATCH --mem=400G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=jobs/logs/slurm-%j_s2_cond_xr_vae_js_v3.out
#SBATCH --error=jobs/logs/slurm-%j_s2_cond_xr_vae_js_v3.err

source ~/.bashrc
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

# Stage 2 conditioned decoder + cross-recons: MSE (vae_js)
# Cross-recons: encode nuclei → decode with cell embedding → compare to raw cell
#
# Loss schedule:
#   cell_recons (1.0)     — ON from epoch 0
#   cell_jsd (0.5)        — ON from epoch 0
#   nuclei_recons (1.0)   — anneals 0→1.0 over epochs 0-100
#   cross_recons (1.0)    — delayed to epoch 100, anneals over epochs 100-200
#   nuclei_jsd (0.5)      — delayed to epoch 100, anneals over epochs 100-200
#   alignment_jsd (1.0)   — delayed to epoch 200, anneals over epochs 200-300

# Stage 1 run: vae_js (MSE), cell v3
STAGE1_MODEL="run/run_2026-04-01-09-29-26_3250320/models/model.pt"
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
  --embed_dim 8 \
  --internal_size "10,9" \
  --latent_size 5 \
  --num_epochs 600 \
  --batch_size 128 \
  --learning_rate 1e-4 \
  --beta 0.5 \
  --recons_loss_type 0 \
  --l1_coef 0.01 \
  --subset True \
  --subset_ratio 0.2 \
  --gene_list "data/training_gene_list_v3.csv" \
  --valid True \
  --test True \
  --debug True \
  \
  --cell_recons_weight 1.0 \
  --cell_jsd_weight 0.5 \
  --nuclei_recons_weight 1.0 \
  --anneal_nuclei_recons_weight True \
  --nuclei_recons_weight_start_epoch 0 \
  --nuclei_recons_weight_anneal_epoch 100 \
  --nuclei_jsd_weight 0.5 \
  --anneal_nuclei_jsd_weight True \
  --nuclei_jsd_weight_start_epoch 100 \
  --nuclei_jsd_weight_anneal_epoch 100 \
  --alignment_weight 1.0 \
  --anneal_alignment True \
  --anneal_alignment_start_epoch 200 \
  --anneal_alignment_epoch 100 \
  --cross_recons_weight 1.0 \
  --anneal_cross_recons_weight True \
  --cross_recons_weight_start_epoch 100 \
  --cross_recons_weight_anneal_epoch 100 \
  \
  --stage3_eval_every 50 \
  --stage3_save_epoch_dump True \
  --stage3_dump_every 50 \
  --stage3_skip_full_eval False
