#!/bin/bash
#SBATCH --job-name=s2_sc_cpm_400
#SBATCH --partition=priority_gpu
#SBATCH --account=prio_gerstein
#SBATCH --gpus=a100:1
#SBATCH --mem=400G
#SBATCH --cpus-per-task=4
#SBATCH --time=18:00:00
#SBATCH --output=jobs/logs/slurm-%j_s2_scalar_cvae_cpm_normloss_v2_400ep.out
#SBATCH --error=jobs/logs/slurm-%j_s2_scalar_cvae_cpm_normloss_v2_400ep.err

source ~/.bashrc
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

# 2-stage scalar cVAE, CPM, normloss fix, 400 epochs
# Same as job 5442589 but with reduced epochs (losses plateau by ~385)
python train.py \
    --model vae_js \
    --data_path "10X_nuclei_v2_v3_AIBS/" \
    --use_raw_data True \
    --raw_data_normalization cpm \
    --protocol v3 \
    --batch n3A \
    --gene_list data/training_gene_list_v3.csv \
    --clamp_recons_loss False \
    --internal_size 10,9 \
    --latent_size 5 \
    --num_epochs 400 \
    --batch_size 128 \
    --learning_rate 0.0001 \
    --beta 0.1 \
    --normalize_losses True \
    --fine_tune_cond_stage3 True \
    --scalar_cond True \
    --cond_encoder True \
    --embed_dim 8 \
    --stage1_model_run run/run_2026-05-08-13-22-30_3907651/models/model.pt \
    --nuclei_recons_weight 1.0 \
    --nuclei_jsd_weight 0.1 \
    --cell_recons_weight 1.0 \
    --cell_jsd_weight 0.1 \
    --alignment_weight 1.0 \
    --anneal_alignment True \
    --anneal_alignment_start_epoch 50 \
    --anneal_alignment_epoch 50 \
    --cross_recons_weight 0.0 \
    --stage3_eval_every 50 \
    --stage3_skip_full_eval True
