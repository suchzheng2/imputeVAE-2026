#!/bin/bash
#SBATCH --job-name=s2_xr_cpm
#SBATCH --partition=pi_gerstein,scavenge_gpu
#SBATCH --gpus=1
#SBATCH --mem=400G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=jobs/logs/slurm-%j_s2_cond_xr_js_perct5_cpm.out
#SBATCH --error=jobs/logs/slurm-%j_s2_cond_xr_js_perct5_cpm.err
#SBATCH --requeue

source ~/.bashrc
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

# Delete stale bytecode
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# Stage 2 conditioned decoder + cross-recons: CPM + log1p, PerCT_5pct genes
# Same cascading annealing as pct5v2_cond_xr, only normalization changes
# JSD weights 0.1 (matching stage 1 beta and v2 convention)
#
# Loss schedule:
#   cell_recons (1.0)     — ON from epoch 0
#   cell_jsd (0.1)        — ON from epoch 0
#   nuclei_recons (1.0)   — anneals 0→1.0 over epochs 0-100
#   cross_recons (1.0)    — delayed to epoch 100, anneals over epochs 100-200
#   nuclei_jsd (0.1)      — delayed to epoch 100, anneals over epochs 100-200
#   alignment_jsd (1.0)   — delayed to epoch 200, anneals over epochs 200-300

# *** FILL IN stage 1 CPM job ID after it completes ***
STAGE1_JOBID="3964307"
STAGE1_LOG="jobs/logs/slurm-${STAGE1_JOBID}_s1_cell_js_perct5_cpm.out"
STAGE1_RUN_DIR=$(grep -m1 "^/gpfs.*run/run_" "$STAGE1_LOG" | grep -o 'run/run_[^ ]*')

if [ -z "$STAGE1_RUN_DIR" ]; then
    echo "ERROR: Could not detect stage 1 run folder from $STAGE1_LOG"
    exit 1
fi

STAGE1_MODEL="${STAGE1_RUN_DIR}/models/model.pt"
echo "Stage 1 model: ${STAGE1_MODEL}"
if [ ! -f "$STAGE1_MODEL" ]; then
    echo "ERROR: Stage 1 model not found: ${STAGE1_MODEL}"
    exit 1
fi

# Verify stage 1 didn't collapse
FINAL_RECONS=$(tail -1 "${STAGE1_RUN_DIR}/train_epoch_loss/loss.txt" | awk -F',' '{print $3}')
echo "Stage 1 final recons loss: ${FINAL_RECONS}"
COLLAPSED=$(echo "$FINAL_RECONS > 0.5" | bc -l 2>/dev/null)
if [ "$COLLAPSED" = "1" ]; then
    echo "WARNING: Stage 1 recons loss > 0.5 — model may have collapsed!"
    echo "Proceeding anyway, but check results carefully."
fi

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
  --num_epochs 600 \
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
  --cross_recons_weight 1.0 \
  --anneal_cross_recons_weight True \
  --cross_recons_weight_start_epoch 100 \
  --cross_recons_weight_anneal_epoch 100 \
  \
  --stage3_eval_every 50 \
  --stage3_save_epoch_dump True \
  --stage3_dump_every 50 \
  --stage3_skip_full_eval False
