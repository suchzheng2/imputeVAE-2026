#!/bin/bash
#SBATCH --job-name=xdataset_v2_lib
#SBATCH --partition=scavenge
#SBATCH --array=0-4
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=jobs/logs/slurm-%A_%a_xdataset_v2_lib.out
#SBATCH --error=jobs/logs/slurm-%A_%a_xdataset_v2_lib.err
# Cross-chemistry (v2 nuclei -> v2 cells) test of the library-size covariate model (2026-10-23).
# Question: was the v2 drop (gene corr 0.927 -> 0.593, R2 -0.03; notes/experiments/cross_dataset_validation.md)
# a library-scale shift? Arm A decoded at four libraries, plus the paper model re-run through the same script.
#   0 = paper model (reference, no covariate)      -> run/cross_dataset_eval/v2_batch_paper_rerun
#   1 = arm A, source library (nucleus's own)      -> run/cross_dataset_eval/v2_batch_armA_source
#   2 = arm A, v3 training-cell per-type median    -> run/cross_dataset_eval/v2_batch_armA_ctmed_train
#   3 = arm A, v2 cell per-type median (label)     -> run/cross_dataset_eval/v2_batch_armA_ctmed_v2
#   4 = arm A, quantile map onto v2 cells          -> run/cross_dataset_eval/v2_batch_armA_qmap_v2
# Label-free scale modes (added 2026-10-23, submit with --array=5-7; no cell-type label at inference):
#   5 = arm A, pooled quantile map onto v2 cells   -> run/cross_dataset_eval/v2_batch_armA_gqmap_v2
#   6 = arm A, nucleus lib + global v2 shift       -> run/cross_dataset_eval/v2_batch_armA_gshift_v2
#   7 = arm A, nucleus lib + global v3 TRAIN shift -> run/cross_dataset_eval/v2_batch_armA_gshift_train (no v2 cells needed)
# kNN-latent label-free modes (added 2026-10-23, submit with --array=8-10; K=50 nearest cells in the aligned latent):
#   8 = arm A, K nearest unlabelled v2 cells        -> run/cross_dataset_eval/v2_batch_armA_knn_v2
#   9 = arm A, kNN quantile map onto v2 cells       -> run/cross_dataset_eval/v2_batch_armA_knnq_v2
#  10 = arm A, K nearest v3 TRAIN cells + v2 shift  -> run/cross_dataset_eval/v2_batch_armA_knn_train_shift
# predicted-type modes (kNN majority vote over v3 training nuclei; training labels only; --array=11-12):
#  11 = arm A, predicted type -> v2 per-type cell median          -> run/cross_dataset_eval/v2_batch_armA_knntype_v2
#  12 = arm A, predicted type -> v3 per-type median + v2 shift    -> run/cross_dataset_eval/v2_batch_armA_knntype_train_shift
# reference-classifier predicted type (2026-10-27; log-CPM/HVG/PCA/LR on v3 training nuclei, analysis/external/
# type_classifier.py, replaces the latent vote off-chemistry; training labels only; --array=13-14):
#  13 = arm A, pca_lr type -> v2 per-type cell median             -> run/cross_dataset_eval/v2_batch_armA_lrtype_v2
#  14 = arm A, pca_lr type -> v3 per-type median + v2 shift       -> run/cross_dataset_eval/v2_batch_armA_lrtype_train_shift
# CPU: the old GPU run (7758417) took 21 min; decode of 76k nuclei on CPU adds a few minutes.
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
source ~/.bashrc 2>/dev/null || true; conda activate impute2
set -eo pipefail
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
find imputeVAE analysis -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

ARMA=run/run_2026-10-16_libsize_armA_jsd0.1_seed1
PAPER=run/run_2026-07-18-19-40-51_570984
case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) RUN=$PAPER; MODE=auto;                   OUT=run/cross_dataset_eval/v2_batch_paper_rerun;   NAME=cvae_v2_paper ;;
  1) RUN=$ARMA;  MODE=source;                 OUT=run/cross_dataset_eval/v2_batch_armA_source;   NAME=cvae_v2_armA_source ;;
  2) RUN=$ARMA;  MODE=ct_median_cell;         OUT=run/cross_dataset_eval/v2_batch_armA_ctmed_train; NAME=cvae_v2_armA_ctmed_train ;;
  3) RUN=$ARMA;  MODE=ct_median_target_cell;  OUT=run/cross_dataset_eval/v2_batch_armA_ctmed_v2; NAME=cvae_v2_armA_ctmed_v2 ;;
  4) RUN=$ARMA;  MODE=quantile_map_target;    OUT=run/cross_dataset_eval/v2_batch_armA_qmap_v2;  NAME=cvae_v2_armA_qmap_v2 ;;
  5) RUN=$ARMA;  MODE=global_quantile_map_target; OUT=run/cross_dataset_eval/v2_batch_armA_gqmap_v2;  NAME=cvae_v2_armA_gqmap_v2 ;;
  6) RUN=$ARMA;  MODE=global_shift_target;    OUT=run/cross_dataset_eval/v2_batch_armA_gshift_v2; NAME=cvae_v2_armA_gshift_v2 ;;
  7) RUN=$ARMA;  MODE=global_shift_train;     OUT=run/cross_dataset_eval/v2_batch_armA_gshift_train; NAME=cvae_v2_armA_gshift_train ;;
  8) RUN=$ARMA;  MODE=knn_cell_median_target; OUT=run/cross_dataset_eval/v2_batch_armA_knn_v2;    NAME=cvae_v2_armA_knn_v2 ;;
  9) RUN=$ARMA;  MODE=knn_quantile_map_target; OUT=run/cross_dataset_eval/v2_batch_armA_knnq_v2;  NAME=cvae_v2_armA_knnq_v2 ;;
 10) RUN=$ARMA;  MODE=knn_train_shift_target; OUT=run/cross_dataset_eval/v2_batch_armA_knn_train_shift; NAME=cvae_v2_armA_knn_train_shift ;;
 11) RUN=$ARMA;  MODE=knn_type_median_target; OUT=run/cross_dataset_eval/v2_batch_armA_knntype_v2; NAME=cvae_v2_armA_knntype_v2 ;;
 12) RUN=$ARMA;  MODE=knn_type_train_shift_target; OUT=run/cross_dataset_eval/v2_batch_armA_knntype_train_shift; NAME=cvae_v2_armA_knntype_train_shift ;;
 13) RUN=$ARMA;  MODE=knn_type_median_target; TP=pca_lr; OUT=run/cross_dataset_eval/v2_batch_armA_lrtype_v2; NAME=cvae_v2_armA_lrtype_v2 ;;
 14) RUN=$ARMA;  MODE=knn_type_train_shift_target; TP=pca_lr; OUT=run/cross_dataset_eval/v2_batch_armA_lrtype_train_shift; NAME=cvae_v2_armA_lrtype_train_shift ;;
  *) echo bad index; exit 2 ;;
esac
echo "=== $NAME run=$RUN mode=$MODE type_predictor=${TP:-latent_knn} $(hostname) $(date)"
python analysis/cross_dataset_eval.py --run_dir "$RUN" --nuc_batch n2A --cell_batch c2A \
    --output_dir "$OUT" --name "$NAME" --target_libsize "$MODE" --type_predictor "${TP:-latent_knn}"
echo "=== Done $(date)"
