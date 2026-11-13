#!/bin/bash
#SBATCH --job-name=s2_joint_nb_v3
#SBATCH --partition=pi_gerstein,scavenge_gpu
#SBATCH --gpus=1
#SBATCH --mem=400G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=jobs/logs/slurm-%j_s2_joint_vae_nb_v3.out
#SBATCH --error=jobs/logs/slurm-%j_s2_joint_vae_nb_v3.err

source ~/.bashrc
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

# Stage 2 (code's stage3_v2): Joint cell + nuclei training with NB loss
# Same annealing schedule as vae_js run for fair comparison

# Stage 1 run: vae_nb (NB), cell v3 — UPDATE PATH after new stage 1 completes
STAGE1_MODEL="run/run_2026-04-01-09-29-27_2189144/models/model.pt"
echo "Stage 1 model: ${STAGE1_MODEL}"
if [ ! -f "$STAGE1_MODEL" ]; then
    echo "ERROR: Stage 1 model not found: ${STAGE1_MODEL}"
    exit 1
fi

python train.py \
  -dp "10X_nuclei_v2_v3_AIBS/" \
  --model "vae_nb" \
  --batch "n3A" \
  --protocol "v3" \
  --use_raw_data True \
  --raw_data_normalization "raw" \
  --fine_tune_stage3_v2 True \
  --stage1_model_run "$STAGE1_MODEL" \
  --internal_size "10,9" \
  --latent_size 5 \
  --num_epochs 600 \
  --batch_size 128 \
  --learning_rate 1e-4 \
  --beta 0.5 \
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
  --cross_recons_weight 0.0 \
  \
  --stage3_eval_every 50 \
  --stage3_save_epoch_dump True \
  --stage3_dump_every 50 \
  --stage3_skip_full_eval False
