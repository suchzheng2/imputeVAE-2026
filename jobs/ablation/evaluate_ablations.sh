#!/bin/bash
#SBATCH --job-name=abl_eval
#SBATCH --partition=scavenge
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=jobs/logs/slurm-%j_abl_eval.out
#SBATCH --error=jobs/logs/slurm-%j_abl_eval.err
# Full benchmark table for the ablation decodes + the arm A / no-covariate controls -> run/benchmark/test/results_ablations/
# then the per-donor breakdown for the donor hold-out arm. Submit with --dependency=afterok:<predict array job id>.
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
source ~/.bashrc 2>/dev/null || true; conda activate impute2; export PYTHONUNBUFFERED=1
set -eo pipefail
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
find analysis/benchmark -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
cd run/benchmark/test
METHODS=$(ls -d abl_* cvae_libsize_armA cvae_libsize_armA_qmap cvae_libsize_armA_knntypeq cvae_libsize_armA_knnq cvae_seed1 | tr '\n' ',' | sed 's/,$//')
cd ../../..
echo "methods: $METHODS"
python -m analysis.benchmark.evaluate_all --eval_partition test --base_dir run/benchmark/test \
    --output_dir run/benchmark/test/results_ablations --methods "$METHODS" "$@"
python -m analysis.benchmark.eval_by_donor --eval_partition test \
    --methods abl_donor5_ctmed,abl_donor5_qmap,abl_donor5_ktqmap,abl_donor5_kqmap,cvae_libsize_armA,cvae_libsize_armA_qmap,cvae_libsize_armA_knntypeq,cvae_libsize_armA_knnq,cvae_seed1 \
    --output_dir run/benchmark/test/results_ablations/by_donor
echo "=== Done: $(date)"
