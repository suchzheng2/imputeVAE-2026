#!/bin/bash
#SBATCH --job-name=bench_libfree
#SBATCH --partition=scavenge
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=jobs/logs/slurm-%j_bench_libfree.out
#SBATCH --error=jobs/logs/slurm-%j_bench_libfree.err
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
source ~/.bashrc 2>/dev/null || true; conda activate scvi2; export PYTHONUNBUFFERED=1
set -eo pipefail
EVAL=${EVAL_PARTITION:-test}
for d in run/benchmark/$EVAL/scvi_seed* run/benchmark/$EVAL/scanvi_seed*; do
  [ -d "$d" ] || continue
  m=$(basename $d | sed -E 's/_seed[0-9]+//'); s=$(basename $d | grep -oE 'seed[0-9]+')
  out=run/benchmark/$EVAL/${m}_libfree_$s
  [ -f $out/imputed_eval.npy ] && { echo "skip $out"; continue; }
  echo "=== $d -> $out"
  python -m analysis.benchmark.redecode_scvi_libfree --method $m --model_dir $d --eval_partition $EVAL --output_dir $out
done
echo "Done."
