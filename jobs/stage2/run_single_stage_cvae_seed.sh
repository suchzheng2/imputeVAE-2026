#!/bin/bash
#SBATCH --job-name=ss_cvae_seed
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --mem=400G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --array=1-2
#SBATCH --requeue
#SBATCH --exclude=r105u17n01,r106u40n01,r106u37n01,r202u17n01
#SBATCH --output=jobs/logs/slurm-%A_%a_single_stage_cvae_seed.out
#SBATCH --error=jobs/logs/slurm-%A_%a_single_stage_cvae_seed.err

# Seed replicates of the paper model (run_2026-07-18-19-40-51_570984, job 7714771).
# Identical config to jobs/stage2/run_single_stage_cvae_normloss_full.sh except:
#   --seed $SLURM_ARRAY_TASK_ID        (weight init / shuffling; data split unchanged)
#   --stage3_save_epoch_dump False     (saves ~166 GB per run; group quota is at 98%)
# Evaluate with: python -m analysis.benchmark.predict_cvae --run_dir run/<new run> --output_dir run/benchmark/test/cvae_seed<k>

source ~/.bashrc 2>/dev/null || true
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
SEED=${SLURM_ARRAY_TASK_ID:-1}
echo "=== cVAE seed replicate, seed=$SEED, $(date), $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

python train.py \
  -dp "10X_nuclei_v2_v3_AIBS/" \
  --model "vae_js" \
  --batch "n3A" \
  --protocol "v3" \
  --use_raw_data True \
  --raw_data_normalization "log1p" \
  --fine_tune_cond_stage3 True \
  --scalar_cond True \
  --cond_encoder True \
  --internal_size "10,9" \
  --latent_size 5 \
  --num_epochs 600 \
  --batch_size 128 \
  --learning_rate 1e-4 \
  --beta 0.1 \
  --recons_loss_type 0 \
  --clamp_recons_loss False \
  --normalize_losses True \
  --l1_coef 0.01 \
  --gene_list "data/training_gene_list_v3.csv" \
  --valid True \
  --test True \
  --debug True \
  --seed $SEED \
  --cell_recons_weight 1.0 \
  --cell_jsd_weight 0.1 \
  --nuclei_recons_weight 1.0 \
  --nuclei_jsd_weight 0.1 \
  --alignment_weight 1.0 \
  --cross_recons_weight 0.0 \
  --stage3_eval_every 50 \
  --stage3_save_epoch_dump False \
  --stage3_dump_every 50 \
  --stage3_skip_full_eval True
echo "=== Done: $(date)"
