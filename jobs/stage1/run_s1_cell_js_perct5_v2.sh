#!/bin/bash
#SBATCH --job-name=s1_pct5_v2
#SBATCH --partition=pi_gerstein,scavenge_gpu
#SBATCH --gpus=1
#SBATCH --mem=400G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=jobs/logs/slurm-%j_s1_cell_js_perct5_v2.out
#SBATCH --error=jobs/logs/slurm-%j_s1_cell_js_perct5_v2.err

source ~/.bashrc
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

# Delete stale bytecode
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# Stage 1 v2: Cell-only pretraining with MSE + delayed/annealed JSD
# Gene list: PerCT_5pct (16,439 genes)
# FIX: beta 0.5 → 0.1, anneal over 200 epochs (not 100)
# v1 collapsed at epoch 400 when JSD overwhelmed reconstruction

python train.py \
  -dp "10X_cells_v2_v3_AIBS/" \
  --model "vae_js" \
  --batch "c3A" \
  --protocol "v3" \
  --use_raw_data True \
  --raw_data_normalization "log1p" \
  --internal_size "10,9" \
  --latent_size 5 \
  --num_epochs 600 \
  --batch_size 128 \
  --learning_rate 1e-4 \
  --beta 0.1 \
  --kl_anneal True \
  --kl_anneal_start_epoch 300 \
  --kl_anneal_epoch 200 \
  --recons_loss_type 0 \
  --l1_coef 0.01 \
  --subset True \
  --subset_ratio 0.2 \
  --gene_list "data/gene_list_perct_5pct.csv" \
  --valid True \
  --test True \
  --debug True
