#!/bin/bash
#SBATCH --job-name=libcond_predict
#SBATCH --partition=scavenge
#SBATCH --time=03:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=jobs/logs/slurm-%j_libcond_predict.out
#SBATCH --error=jobs/logs/slurm-%j_libcond_predict.err
# Decodes of the library-covariate run (jobs/ablation/run_library_cond.sh) for the benchmark:
#   the four standard modes at the UNKNOWN (marginal) target library  -> run/benchmark/test/libcond_{ctmed,qmap,ktqmap,kqmap}
#   ct_median_cell at the mean cell-library embedding                  -> libcond_ctmed_meancell
#   ct_median_cell as one female (4-c3A) and one male (1-c3A) library  -> libcond_ctmed_lib4F, libcond_ctmed_lib1M (Xist check)
# Submit with --dependency=afterany:<training job>; exits 3 if the run is not finished (so a chained evaluate does not run).
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
source ~/.bashrc 2>/dev/null || true; conda activate impute2; export PYTHONUNBUFFERED=1
set -eo pipefail
find imputeVAE analysis -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
RUN_DIR=run/run_2026-11-06_libcond_seed1
JOB=${TRAIN_JOB_ID:-unknown}
test -f "$RUN_DIR/models/model.pt" || { echo "no checkpoint in $RUN_DIR"; exit 3; }
python -c "import json;s=json.load(open('$RUN_DIR/run_state.json'));print('run_state',s);assert s['last_checkpoint_epoch']>=599,'training not finished'" || exit 3
for SPEC in ct_median_cell:unknown:ctmed quantile_map:unknown:qmap knn_type_quantile_map:unknown:ktqmap knn_quantile_map:unknown:kqmap \
            ct_median_cell:mean_cell:ctmed_meancell ct_median_cell:4-c3A:ctmed_lib4F ct_median_cell:1-c3A:ctmed_lib1M; do
  IFS=: read -r T L S <<< "$SPEC"
  echo "=== $T target_library=$L -> libcond_$S"
  python -m analysis.benchmark.predict_cvae --run_dir $RUN_DIR --eval_partition test \
    --output_dir run/benchmark/test/libcond_$S --target_libsize $T --target_library $L \
    --training_job_id "$JOB" --training_time_s 0
done
echo "=== Done: $(date)"
