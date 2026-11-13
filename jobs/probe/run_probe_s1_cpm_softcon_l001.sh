#!/bin/bash
#SBATCH --job-name=probe_s1_cpm_sc001
#SBATCH --partition=priority_gpu
#SBATCH --account=prio_gerstein
#SBATCH --gpus=a100:1
#SBATCH --mem=400G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=jobs/logs/slurm-%j_probe_s1_cpm_softcon_l001.out
#SBATCH --error=jobs/logs/slurm-%j_probe_s1_cpm_softcon_l001.err

source ~/.bashrc
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# Probe: Stage 1 cell-only CPM training WITH soft constraint (lambda=0.01)
# Starting from scratch — no pretrained model.
# Goal: does soft constraint prevent mu from growing large, allowing JSD to stay < ln(2)?
#
# Hypothesis: CPM stage 1 without soft constraint lets mu drift to ~100-500
# because JSD gradient = 0 when mu is large. With soft constraint, the L2 penalty
# pulls mu back (gradient = 2*lambda*mu, never zero), keeping JSD functional.
#
# If JSD drops below 0.693 within first 20-30 epochs -> soft constraint works at stage 1 too.
# This would allow training a better CPM stage 1 model as foundation for stage 2.

python train.py \
  -dp "10X_cells_v2_v3_AIBS/" \
  --model "vae_js" \
  --batch "c3A" \
  --protocol "v3" \
  --use_raw_data True \
  --raw_data_normalization "cpm" \
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
  --soft_constraint_lambda 0.01
