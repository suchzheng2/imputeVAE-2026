#!/bin/bash
#SBATCH --job-name=libcond_eval
#SBATCH --partition=scavenge
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=jobs/logs/slurm-%j_libcond_eval.out
#SBATCH --error=jobs/logs/slurm-%j_libcond_eval.err
# Benchmark table for the library-covariate decodes + arm A controls -> run/benchmark/test/results_libcond/
# Submit with --dependency=afterok:<predict job>.
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
source ~/.bashrc 2>/dev/null || true; conda activate impute2; export PYTHONUNBUFFERED=1
set -eo pipefail
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
find analysis/benchmark -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
cd run/benchmark/test
METHODS=$(ls -d libcond_* cvae_libsize_armA cvae_libsize_armA_qmap cvae_libsize_armA_knntypeq cvae_libsize_armA_knnq cvae_seed1 | tr '\n' ',' | sed 's/,$//')
cd ../../..
echo "methods: $METHODS"
python -m analysis.benchmark.evaluate_all --eval_partition test --base_dir run/benchmark/test \
    --output_dir run/benchmark/test/results_libcond --methods "$METHODS" "$@"
python -m analysis.benchmark.latent_library_mixing \
    --run_dirs run/run_2026-10-16_libsize_armA_jsd0.1_seed1,run/run_2026-11-06_libcond_seed1 \
    --output_dir run/benchmark/test/results_libcond/latent_mixing
python -m analysis.benchmark.eval_by_donor --eval_partition test \
    --methods libcond_ctmed,libcond_qmap,libcond_ktqmap,libcond_kqmap,cvae_libsize_armA,cvae_libsize_armA_qmap,cvae_libsize_armA_knntypeq,cvae_libsize_armA_knnq \
    --output_dir run/benchmark/test/results_libcond/by_donor
echo "=== Done: $(date)"
