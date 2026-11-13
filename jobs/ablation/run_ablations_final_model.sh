#!/bin/bash
#SBATCH --job-name=cvae_ablation
#SBATCH --partition=pi_gerstein_gpu,scavenge_gpu
#SBATCH --gpus=2
#SBATCH --mem=400G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --array=1-6
#SBATCH --requeue
#SBATCH --exclude=r105u17n01,r106u40n01,r106u37n01,r202u17n01
#SBATCH --output=jobs/logs/slurm-%A_%a_cvae_ablation.out
#SBATCH --error=jobs/logs/slurm-%A_%a_cvae_ablation.err

# Ablations on the final (library-size covariate) model, 2026-11-04. Plan: notes/prashant_update_2026-10-30.md §4.3,
# notes/paper_storyline_2026-10-23.md item 5 (A4); record: notes/experiments/ablations_final_model/README.md.
# Base config = arm A of jobs/stage2/run_single_stage_cvae_libsize.sh (covariate on, JSD 0.1/0.1, alignment 1.0,
# seed 1, 600 epochs, run run_2026-10-16_libsize_armA_jsd0.1_seed1). Each array task changes ONE thing:
#   1 nojsd    per-modality JSD weights 0/0 (alignment 1.0)        -> is the prior-matching regulariser needed?
#   2 noalign  alignment weight 0 (JSD 0.1/0.1)                     -> is cross-modality latent alignment needed?
#   3 deccond  --cond_encoder False (decoder-only modality signal)  -> does the encoder need to know the modality?
#   4 n1_L5IT  --exclude_cell_types "L5 IT"  (2,320 test nuclei; neighbours L2/3 IT, L6 IT = interpolation case)
#   5 n1_Pvalb --exclude_cell_types "Pvalb"  (520 test nuclei; neighbour Sst = the known confusion pair)
#   6 donor5   --holdout_donors "nuclei:5"   (7,594 nuclei of donor 5 leave train/valid; test unchanged)
# Exclusions are applied AFTER the fixed stratified split (imputeVAE/data.py apply_train_exclusions), so the
# test partition is identical to data/split_manifest.tsv and the held-out observations are scored by the benchmark.
# --gpus=2 + healthy-GPU guard: r813u29n03 GPU 0000:31:00.0 is broken (see the arm A/B script). Resumable via --run_name.
# Evaluate: jobs/ablation/predict_ablations.sh (array, --dependency=aftercorr:<this job>) then
#           jobs/ablation/evaluate_ablations.sh (--dependency=afterok:<predict job>).

source ~/.bashrc 2>/dev/null || true
conda activate impute2; export PYTHONUNBUFFERED=1
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
find imputeVAE analysis -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

IDX=${SLURM_ARRAY_TASK_ID:-1}
JSD_W=0.1; ALIGN_W=1.0; COND_ENC=True; EXTRA=""
case "$IDX" in
  1) ARM=nojsd;    JSD_W=0.0 ;;
  2) ARM=noalign;  ALIGN_W=0.0 ;;
  3) ARM=deccond;  COND_ENC=False ;;
  4) ARM=n1_L5IT;  EXTRA='--exclude_cell_types L5 IT' ;;
  5) ARM=n1_Pvalb; EXTRA='--exclude_cell_types Pvalb' ;;
  6) ARM=donor5;   EXTRA='--holdout_donors nuclei:5' ;;
  *) echo "unknown array index $IDX"; exit 1 ;;
esac
SEED=1
RUN_NAME="run_2026-11-04_ablation_${ARM}_seed${SEED}"
RUN_DIR="run/${RUN_NAME}"

RESUME_ARGS=""
if [ -f "$RUN_DIR/run_state.json" ] && [ -f "$RUN_DIR/models/model.pt" ]; then
  LAST=$(python -c "import json;print(json.load(open('$RUN_DIR/run_state.json'))['last_checkpoint_epoch'])")
  RESUME_ARGS="--continue_training True --continue_training_run_path $RUN_DIR --continue_training_epoch $((LAST+1))"
  echo "=== Resuming $RUN_NAME from checkpoint epoch $LAST"
fi

echo "=== cVAE ablation arm=$ARM (jsd=$JSD_W align=$ALIGN_W cond_encoder=$COND_ENC extra='$EXTRA') seed=$SEED, $(date), $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

HEALTHY=$(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null | grep '^GPU-' | head -1)
if [ -z "$HEALTHY" ]; then
  echo "!!! no usable GPU in this allocation (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES); aborting (exit 3) instead of training on CPU"
  exit 3
fi
export CUDA_VISIBLE_DEVICES=$HEALTHY
python -c "import torch,sys; ok=torch.cuda.is_available(); print('torch sees GPU:', torch.cuda.get_device_name(0) if ok else None, flush=True); sys.exit(0 if ok else 3)" || exit 3

# EXTRA holds a flag whose value may contain a space ("L5 IT"); split it into flag / value explicitly.
EXTRA_ARGS=()
if [ -n "$EXTRA" ]; then
  EXTRA_ARGS=("${EXTRA%% *}" "${EXTRA#* }")
fi

python train.py \
  -dp "10X_nuclei_v2_v3_AIBS/" \
  --model "vae_js" \
  --batch "n3A" \
  --protocol "v3" \
  --use_raw_data True \
  --raw_data_normalization "log1p" \
  --fine_tune_cond_stage3 True \
  --scalar_cond True \
  --cond_encoder $COND_ENC \
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
  --alignment_weight $ALIGN_W \
  --cross_recons_weight 0.0 \
  --stage3_eval_every 50 \
  --stage3_save_epoch_dump False \
  --stage3_dump_every 50 \
  --stage3_skip_full_eval True \
  "${EXTRA_ARGS[@]}" \
  $RESUME_ARGS
echo "=== Done: $(date)"
