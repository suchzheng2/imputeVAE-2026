#!/bin/bash

#SBATCH --job-name=nb-1.2
#SBATCH --out="jobs/logs/slurm-%j_vae_nb_cell_stage1.out"
#SBATCH --partition=gpu,pi_gerstein
#SBATCH --gpus=1
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=1
#SBATCH --requeue
#SBATCH --mem=300G
#SBATCH --mail-type=ALL

module load miniconda
conda activate impute2

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export CUDA_VISIBLE_DEVICES=0

# Sanity check: fail fast if PyTorch can't see the GPU
python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT AVAILABLE - wrong PyTorch build!'"

# ============================================================
# Stage 1 - NB VAE on raw cell data
# ============================================================

# stage 1.1 - cell, NB reconstruction only (beta=0)
# python train.py \
#    -dp "10X_cells_v2_v3_AIBS/" \
#    --model "vae_nb" \
#    -b "c3A"\
#    --use_raw_data True\
#    --raw_data_normalization "raw"\
#    --internal_size "10,9" \
#    --latent_size 5 \
#    --num_epochs 300 \
#    --beta 0 \
#    --batch_size 128\
#    --learning_rate 1e-4\
#    --subset True\
#    --subset_ratio 0.2\
#    --exclude_mitocarta True\
#    --debug True

   # stage 1.1 - nuclei, NB reconstruction only (beta=0)
   # python train.py \
   # -dp "10X_nuclei_v2_v3_AIBS/" \
   # --model "vae_nb" \
   # -b "n3A"\
   # --use_raw_data True\
   # --raw_data_normalization "raw"\
   # --internal_size "10,9" \
   # --latent_size 5 \
   # --num_epochs 300 \
   # --beta 0 \
   # --batch_size 128\
   # --learning_rate 1e-4\
   # --subset True\
   # --subset_ratio 0.2\
   # --exclude_mitocarta True\
   # --debug True

# stage 1.2 - cell, NB reconstruction + KL with annealing
# Uncomment after stage 1.1 finishes and set --load_trained_model_path
# python train.py \
#    -dp "10X_cells_v2_v3_AIBS/" \
#    --model "vae_nb" \
#    -b "c3A"\
#    --use_raw_data True\
#    --raw_data_normalization "raw"\
#    --load_trained_model True\
#    --load_trained_model_path "run/<STAGE_1.1_RUN_FOLDER>/models/model.pt"\
#    --internal_size "10,9" \
#    --latent_size 5 \
#    --num_epochs 500 \
#    --beta 0.5 \
#    --kl_anneal True \
#    --kl_anneal_start_epoch 0\
#    --kl_anneal_epoch 200\
#    --batch_size 128\
#    --learning_rate 1e-4\
#    --subset True\
#    --subset_ratio 0.2\
#    --exclude_mitocarta True\
#    --debug True

# stage 1.2 ZINB variant - same as 1.2 but with zero-inflation
# python train.py \
#    -dp "10X_cells_v2_v3_AIBS/" \
#    --model "vae_nb" \
#    --use_zinb True \
#    -b "c3A"\
#    --use_raw_data True\
#    --raw_data_normalization "raw"\
#    --load_trained_model True\
#    --load_trained_model_path "<STAGE_1.1_RUN_PATH>/models/model.pt"\
#    --internal_size "10,9" \
#    --latent_size 5 \
#    --num_epochs 500 \
#    --beta 0.5 \
#    --kl_anneal True \
#    --kl_anneal_start_epoch 0\
#    --kl_anneal_epoch 200\
#    --batch_size 128\
#    --learning_rate 1e-4\
#    --subset True\
#    --subset_ratio 0.2\
#    --exclude_mitocarta True\
#    --debug True

# ============================================================
# Stage 1 - NB VAE on raw cell data, combined 1.1 and 1.2
# ============================================================

python train.py \
   -dp "10X_cells_v2_v3_AIBS/" \
   --model "vae_nb" \
   -b "c3A"\
   --use_raw_data True\
   --raw_data_normalization "raw"\
   --internal_size "10,9" \
   --latent_size 5 \
   --num_epochs 600 \
   --beta 0.5 \
   --kl_anneal True \
   --kl_anneal_start_epoch 200\
   --kl_anneal_epoch 400\
   --batch_size 128\
   --learning_rate 1e-4\
   --subset True\
   --subset_ratio 0.2\
   --exclude_mitocarta True\
   --debug True


# python train.py \
#    -dp "10X_nuclei_v2_v3_AIBS/" \
#    --model "vae_nb" \
#    -b "n3A"\
#    --use_raw_data True\
#    --raw_data_normalization "raw"\
#    --internal_size "10,9" \
#    --latent_size 5 \
#    --num_epochs 600 \
#    --beta 0.1 \
#    --kl_anneal True \
#    --kl_anneal_start_epoch 300\
#    --kl_anneal_epoch 500\
#    --batch_size 128\
#    --learning_rate 1e-4\
#    --subset True\
#    --subset_ratio 0.2\
#    --exclude_mitocarta True\
#    --debug True
# ============================================================
# Stage 2 - NB VAE on raw nuclei data
# ============================================================

# stage 2.1 - nuclei, NB reconstruction only (beta=0)
# python train.py \
#    -dp "10X_nuclei_v2_v3_AIBS/" \
#    --model "vae_nb" \
#    -b "n3A"\
#    --use_raw_data True\
#    --raw_data_normalization "raw"\
#    --internal_size "10,9" \
#    --latent_size 5 \
#    --num_epochs 300 \
#    --beta 0 \
#    --batch_size 128\
#    --learning_rate 1e-4\
#    --subset True\
#    --subset_ratio 0.2\
#    --exclude_mitocarta True\
#    --debug True

# stage 2.2 - nuclei, NB reconstruction + KL with annealing
# Uncomment after stage 2.1 finishes and set --load_trained_model_path
# python train.py \
#    -dp "10X_nuclei_v2_v3_AIBS/" \
#    --model "vae_nb" \
#    -b "n3A"\
#    --use_raw_data True\
#    --raw_data_normalization "raw"\
#    --load_trained_model True\
#    --load_trained_model_path "<STAGE_2.1_RUN_PATH>/models/model.pt"\
#    --internal_size "10,9" \
#    --latent_size 5 \
#    --num_epochs 500 \
#    --beta 0.5 \
#    --kl_anneal True \
#    --kl_anneal_start_epoch 0\
#    --kl_anneal_epoch 200\
#    --batch_size 128\
#    --learning_rate 1e-4\
#    --subset True\
#    --subset_ratio 0.2\
#    --exclude_mitocarta True\
#    --debug True

# ============================================================
# Stage 3 - Finetune: align cell & nuclei latent spaces with JSD
# ============================================================

# stage 3 - cell recon + nuclei recon + JSD alignment
# Uncomment after stage 1.2 and 2.2 finish, set model paths
# python train.py \
#    -dp "10X_nuclei_v2_v3_AIBS/" \
#    --model "vae_nb" \
#    -b "n3A"\
#    --use_raw_data True\
#    --raw_data_normalization "raw"\
#    --fine_tune_stage3_v2 True \
#    --stage1_model_run "<STAGE_1.2_RUN_PATH>/models/model.pt"\
#    --stage2_model_run "<STAGE_2.2_RUN_PATH>/models/model.pt"\
#    --was_train_by_cell_type True \
#    --use_all_cell_mu_log_var False \
#    --alpha 0.5\
#    --anneal_alpha True\
#    --anneal_alpha_start_epoch 0\
#    --anneal_alpha_epoch 100\
#    --internal_size "10,9" \
#    --latent_size 5 \
#    --num_epochs 500 \
#    --beta 0.5 \
#    --batch_size 128\
#    --learning_rate 1e-5\
#    --subset True\
#    --subset_ratio 0.2\
#    --exclude_mitocarta True\
#    --ft_metrics "jsd"\
#    --debug True
