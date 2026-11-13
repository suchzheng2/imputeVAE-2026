#!/bin/bash
#SBATCH --job-name=cvae_libcond
#SBATCH --partition=pi_gerstein_gpu,scavenge_gpu
#SBATCH --gpus=2
#SBATCH --mem=400G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --exclude=r105u17n01,r106u40n01,r106u37n01,r202u17n01
#SBATCH --output=jobs/logs/slurm-%j_cvae_libcond.out
#SBATCH --error=jobs/logs/slurm-%j_cvae_libcond.err
# Library (batch) covariate on the final model (2026-11-06): arm A config (covariate on, JSD 0.1/0.1, alignment 1.0,
# seed 1, 600 epochs) + --library_cond True: a learned 4-d embedding of the sequencing library (17 libraries: 12 cell,
# 5 nuclei; <run>/library_vocab.json) on encoder and decoder, id 0 = UNKNOWN trained by 10 % dropout and used as the
# default decode target. Question: does an explicit library/batch condition change the benchmark, make the latent
# library-free, and let the output be placed in a named (female / male) cell library (Xist)?
# Record: notes/experiments/ablations_final_model/README.md (library covariate section). Resumable via --run_name.
source ~/.bashrc 2>/dev/null || true
conda activate impute2; export PYTHONUNBUFFERED=1
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
find imputeVAE analysis -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
SEED=1
RUN_NAME="run_2026-11-06_libcond_seed${SEED}"
RUN_DIR="run/${RUN_NAME}"
RESUME_ARGS=""
if [ -f "$RUN_DIR/run_state.json" ] && [ -f "$RUN_DIR/models/model.pt" ]; then
  LAST=$(python -c "import json;print(json.load(open('$RUN_DIR/run_state.json'))['last_checkpoint_epoch'])")
  RESUME_ARGS="--continue_training True --continue_training_run_path $RUN_DIR --continue_training_epoch $((LAST+1))"
  echo "=== Resuming $RUN_NAME from checkpoint epoch $LAST"
fi
echo "=== cVAE library covariate seed=$SEED, $(date), $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
HEALTHY=$(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null | grep '^GPU-' | head -1)
if [ -z "$HEALTHY" ]; then
  echo "!!! no usable GPU in this allocation (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES); aborting (exit 3) instead of training on CPU"
  exit 3
fi
export CUDA_VISIBLE_DEVICES=$HEALTHY
python -c "import torch,sys; ok=torch.cuda.is_available(); print('torch sees GPU:', torch.cuda.get_device_name(0) if ok else None, flush=True); sys.exit(0 if ok else 3)" || exit 3
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
  --libsize_cond True \
  --libsize_target ct_median_cell \
  --library_cond True \
  --library_dim 4 \
  --library_capacity 32 \
  --library_dropout 0.1 \
  --internal_size "10,9" \
  --latent_size 5 \
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
  --seed 1 \
  --cell_recons_weight 1.0 \
  --cell_jsd_weight 0.1 \
  --nuclei_recons_weight 1.0 \
  --nuclei_jsd_weight 0.1 \
  --alignment_weight 1.0 \
  --cross_recons_weight 0.0 \
  --stage3_save_epoch_dump False \
  --stage3_dump_every 50 \
  --stage3_skip_full_eval True \
  --run_name "$RUN_NAME" \
  --num_epochs 600 \
  --stage3_eval_every 50 \
  --debug False \
  $RESUME_ARGS
echo "=== Done: $(date)"
