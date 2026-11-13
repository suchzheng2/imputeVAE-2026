#!/bin/bash
#SBATCH --job-name=abl_predict
#SBATCH --partition=scavenge
#SBATCH --array=1-6
#SBATCH --time=03:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=jobs/logs/slurm-%A_%a_abl_predict.out
#SBATCH --error=jobs/logs/slurm-%A_%a_abl_predict.err
# Cross-decode the test nuclei through each ablation run (jobs/ablation/run_ablations_final_model.sh, same array
# index) in the four decode modes used for the covariate model: ct_median_cell (label), quantile_map (label),
# knn_type_quantile_map (reported label-free), knn_quantile_map (no labels anywhere).
# Submit with --dependency=aftercorr:<training array job id>.
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
source ~/.bashrc 2>/dev/null || true; conda activate impute2; export PYTHONUNBUFFERED=1
set -eo pipefail
find imputeVAE analysis -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
case "${SLURM_ARRAY_TASK_ID:-1}" in
  1) ARM=nojsd ;; 2) ARM=noalign ;; 3) ARM=deccond ;; 4) ARM=n1_L5IT ;; 5) ARM=n1_Pvalb ;; 6) ARM=donor5 ;;
  *) echo "bad array id"; exit 2 ;;
esac
RUN_DIR=run/run_2026-11-04_ablation_${ARM}_seed1
JOB=${TRAIN_JOB_ID:-unknown}
test -f "$RUN_DIR/models/model.pt" || { echo "no checkpoint in $RUN_DIR"; exit 3; }
python -c "import json;s=json.load(open('$RUN_DIR/run_state.json'));print('run_state',s);assert s['last_checkpoint_epoch']>=599,'training not finished'"
for MODE in ct_median_cell:ctmed quantile_map:qmap knn_type_quantile_map:ktqmap knn_quantile_map:kqmap; do
  T=${MODE%%:*}; S=${MODE##*:}
  echo "=== $ARM $T -> abl_${ARM}_${S}"
  python -m analysis.benchmark.predict_cvae --run_dir $RUN_DIR --eval_partition test \
    --output_dir run/benchmark/test/abl_${ARM}_${S} --target_libsize $T \
    --training_job_id "${JOB}_${SLURM_ARRAY_TASK_ID}" --training_time_s 0
done
echo "=== Done: $(date)"
