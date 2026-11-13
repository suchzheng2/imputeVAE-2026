#!/bin/bash
#SBATCH --job-name=libcond_smoke
#SBATCH --partition=scavenge
#SBATCH --mem=400G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=jobs/logs/slurm-%j_libcond_smoke.out
#SBATCH --error=jobs/logs/slurm-%j_libcond_smoke.err
# Smoke test of the library (batch) covariate path (2026-11-06): 10 % of the data, 2 epochs, inline eval every epoch,
# CPU only; then the four benchmark decodes + the three library-target variants through predict_cvae on the result.
# The real training job (run_library_cond.sh) is submitted with --dependency=afterok on this job.
source ~/.bashrc 2>/dev/null || true
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
set -eo pipefail
export PYTHONUNBUFFERED=1
find imputeVAE analysis -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
RUN_NAME=run_2026-11-06_libcond_smoke
rm -rf run/$RUN_NAME
echo "=== smoke start $(date) $(hostname)"
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
  --num_epochs 2 \
  --stage3_eval_every 1 \
  --subset True --subset_ratio 0.1 \
  --debug False
echo "=== training done $(date); run folder:"; ls run/$RUN_NAME; cat run/$RUN_NAME/library_vocab.json | head -40
for SPEC in ct_median_cell:unknown quantile_map:unknown knn_type_quantile_map:unknown knn_quantile_map:unknown ct_median_cell:mean_cell ct_median_cell:4-c3A; do
  T=${SPEC%%:*}; L=${SPEC##*:}
  echo "=== predict $T target_library=$L"
  python -m analysis.benchmark.predict_cvae --run_dir run/$RUN_NAME --eval_partition test \
    --output_dir run/benchmark/test/_smoke_libcond_${T}_${L} --target_libsize $T --target_library $L \
    --training_job_id smoke --training_time_s 0
done
rm -rf run/benchmark/test/_smoke_libcond_*
echo "=== Done: $(date)"
