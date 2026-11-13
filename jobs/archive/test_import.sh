#!/bin/bash
#SBATCH --job-name=test-vae
#SBATCH --out="jobs/logs/slurm-%j_test_import.out"
#SBATCH --partition=gpu,pi_gerstein
#SBATCH --gpus=1
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=500G
#SBATCH --mail-type=ALL

module load miniconda
conda activate impute2

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export CUDA_VISIBLE_DEVICES=0

# Sanity check: fail fast if PyTorch can't see the GPU
python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT AVAILABLE - wrong PyTorch build!'"

# Short smoke test: 5 epochs, small subset
python train.py \
   -dp "10X_cells_v2_v3_AIBS/" \
   --model "vae_nb" \
   -b "c3A"\
   --use_raw_data True\
   --raw_data_normalization "raw"\
   --internal_size "10,9" \
   --latent_size 5 \
   --num_epochs 5 \
   --beta 0.5 \
   --kl_anneal True \
   --kl_anneal_start_epoch 2\
   --kl_anneal_epoch 4\
   --batch_size 128\
   --learning_rate 1e-4\
   --subset True\
   --subset_ratio 0.05\
   --exclude_mitocarta True\
   --debug True
