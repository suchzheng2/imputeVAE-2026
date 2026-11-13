#!/bin/bash
#SBATCH --job-name=ss_cvae_latent
#SBATCH --partition=pi_gerstein_gpu,scavenge_gpu
#SBATCH --gpus=2
#SBATCH --mem=400G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --array=1-2
#SBATCH --requeue
#SBATCH --exclude=r105u17n01,r106u40n01,r106u37n01,r202u17n01
#SBATCH --output=jobs/logs/slurm-%A_%a_single_stage_cvae_latent.out
#SBATCH --error=jobs/logs/slurm-%A_%a_single_stage_cvae_latent.err

# Latent-size control for the library-size covariate model (2026-10-23; plan: notes/libsize_covariate_review_2026-10-09.md §6).
# Identical to arm A of jobs/stage2/run_single_stage_cvae_libsize.sh (covariate on, JSD 0.1/0.1, alignment 1.0, seed 1)
# except the latent width. The 32-dim arms use ~5 dims (latent-usage check); this asks whether capacity matters at all.
#   1 = 8 dims  (--latent_size 3): at/above the active count -> CT-level metrics should be unchanged
#   2 = 4 dims  (--latent_size 2): below the active count     -> does identity coding need more than 4 dims?
# --gpus=2 + the healthy-GPU guard because r813u29n03 GPU 0000:31:00.0 is broken (see the arm A/B script).

source ~/.bashrc 2>/dev/null || true
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

IDX=${SLURM_ARRAY_TASK_ID:-1}
case "$IDX" in
  1) LS=3; DIMS=8 ;;
  2) LS=2; DIMS=4 ;;
  *) echo "unknown array index $IDX"; exit 1 ;;
esac
SEED=1
JSD_W=0.1
RUN_NAME="run_2026-10-23_libsize_armA_latent${DIMS}_seed${SEED}"
RUN_DIR="run/${RUN_NAME}"

RESUME_ARGS=""
if [ -f "$RUN_DIR/run_state.json" ] && [ -f "$RUN_DIR/models/model.pt" ]; then
  LAST=$(python -c "import json;print(json.load(open('$RUN_DIR/run_state.json'))['last_checkpoint_epoch'])")
  RESUME_ARGS="--continue_training True --continue_training_run_path $RUN_DIR --continue_training_epoch $((LAST+1))"
  echo "=== Resuming $RUN_NAME from checkpoint epoch $LAST"
fi

echo "=== cVAE libsize covariate latent control, dims=$DIMS seed=$SEED, $(date), $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

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
  --latent_size $LS \
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
