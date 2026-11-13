#!/bin/bash
#SBATCH --job-name=ss_cvae_libsize
#SBATCH --partition=pi_gerstein_gpu,scavenge_gpu
#SBATCH --gpus=1
#SBATCH --mem=400G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --array=1-2
#SBATCH --requeue
#SBATCH --exclude=r105u17n01,r106u40n01,r106u37n01,r202u17n01
#SBATCH --output=jobs/logs/slurm-%A_%a_single_stage_cvae_libsize.out
#SBATCH --error=jobs/logs/slurm-%A_%a_single_stage_cvae_libsize.err

# Library-size covariate retrain (plan: notes/libsize_covariate_review_2026-10-09.md, section 3).
# Same config as the paper model / seed replicates (jobs/stage2/run_single_stage_cvae_seed.sh, seed 1)
# plus --libsize_cond True. Array index selects the regulariser arm:
#   1 = arm A: covariate only, per-modality JSD weights 0.1 / 0.1 (paper values)   -> isolates the covariate
#   2 = arm B: covariate + per-modality JSD weights 0.01 / 0.01 (alignment stays 1.0) -> is the collapse regulariser-driven?
# Inline cross-decode eval decodes at the per-type median training-cell library (--libsize_target ct_median_cell).
#
# Partition: pi_gerstein_gpu (group-owned, non-preemptible; r813u29n03 = 1 TB A100 node fits the 400G request;
# peak RSS of this config is 326 GiB, sacct 12646907) with scavenge_gpu as fallback. First submission on
# scavenge_gpu alone (13442818) had an estimated start 4 days out.
# Resumable: the run folder is fixed by --run_name, and a requeued task resumes from the last checkpoint
# (models/model.pt + run_state.json, written every 10 epochs) instead of starting over in a new folder
# (the seed-2 replicate was preempted 9 times and restarted from scratch each time, 2026-10-09..12).
#
# Evaluate with:
#   CVAE_RUN_DIR=run/<run_name> CVAE_OUT=run/benchmark/test/cvae_libsize_armA CVAE_TARGET_LIBSIZE=ct_median_cell \
#     sbatch --export=ALL jobs/benchmark/predict_cvae.sh      (then jobs/benchmark/evaluate_all.sh)

source ~/.bashrc 2>/dev/null || true
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

ARM_IDX=${SLURM_ARRAY_TASK_ID:-1}
case "$ARM_IDX" in
  1) ARM=A; JSD_W=0.1 ;;
  2) ARM=B; JSD_W=0.01 ;;
  *) echo "unknown array index $ARM_IDX"; exit 1 ;;
esac
SEED=1
RUN_NAME="run_2026-10-16_libsize_arm${ARM}_jsd${JSD_W}_seed${SEED}"
RUN_DIR="run/${RUN_NAME}"

RESUME_ARGS=""
if [ -f "$RUN_DIR/run_state.json" ] && [ -f "$RUN_DIR/models/model.pt" ]; then
  LAST=$(python -c "import json;print(json.load(open('$RUN_DIR/run_state.json'))['last_checkpoint_epoch'])")
  RESUME_ARGS="--continue_training True --continue_training_run_path $RUN_DIR --continue_training_epoch $((LAST+1))"
  echo "=== Resuming $RUN_NAME from checkpoint epoch $LAST"
fi

echo "=== cVAE libsize covariate, arm=$ARM jsd=$JSD_W seed=$SEED, $(date), $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

# ---- GPU health guard (2026-10-16): r813u29n03 GPU 0000:31:00.0 is broken (nvidia-smi "Unknown Error") and the
# training code falls back to CPU silently. Pick the first healthy GPU in the allocation by UUID (submit with
# `sbatch --gpus=2` on that node so a healthy one is included) and abort instead of training on CPU.
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
  --run_name "$RUN_NAME" \
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
  --cell_jsd_weight $JSD_W \
  --nuclei_recons_weight 1.0 \
  --nuclei_jsd_weight $JSD_W \
  --alignment_weight 1.0 \
  --cross_recons_weight 0.0 \
  --stage3_eval_every 50 \
  --stage3_save_epoch_dump False \
  --stage3_dump_every 50 \
  --stage3_skip_full_eval True \
  $RESUME_ARGS
echo "=== Done: $(date)"
