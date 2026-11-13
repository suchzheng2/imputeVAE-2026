#!/bin/bash
# Submit the split-aware benchmark.  Usage:  bash jobs/benchmark/submit_all.sh [test|valid] [cache_jobid]
# Dependency chain: cache -> {trivial, cvae, scvi x3, scanvi x3, scgen x3, scbutterfly} -> evaluate (afterany).
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
export EVAL_PARTITION=${1:-test}
CACHE_DEP=${2:-}
mkdir -p jobs/logs run/benchmark/$EVAL_PARTITION
DEP=""
if [ -n "$CACHE_DEP" ]; then DEP="--dependency=afterok:$CACHE_DEP"; fi
echo "=== Split-aware benchmark, eval partition = $EVAL_PARTITION ==="
J1=$(sbatch --parsable $DEP --export=ALL jobs/benchmark/train_ctmean_linreg.sh); echo "trivial baselines: $J1"
J2=$(sbatch --parsable $DEP --export=ALL jobs/benchmark/predict_cvae.sh);        echo "cVAE predict:      $J2"
J3=$(sbatch --parsable $DEP --export=ALL jobs/benchmark/train_scvi.sh);          echo "scVI x3:           $J3"
J4=$(sbatch --parsable $DEP --export=ALL jobs/benchmark/train_scanvi.sh);        echo "scANVI x3:         $J4"
J5=$(sbatch --parsable $DEP --export=ALL jobs/benchmark/train_scgen.sh);         echo "scGen x3:          $J5"
J6=$(sbatch --parsable $DEP --export=ALL jobs/benchmark/train_scbutterfly.sh);   echo "scButterfly:       $J6"
J7=$(sbatch --parsable --dependency=afterany:$J1:$J2:$J3:$J4:$J5:$J6 --export=ALL jobs/benchmark/evaluate_all.sh)
echo "evaluate (afterany): $J7"
echo "$EVAL_PARTITION $J1 $J2 $J3 $J4 $J5 $J6 $J7" >> jobs/benchmark/submitted_jobs.txt
echo "Monitor: sacct -j $J1,$J2,$J3,$J4,$J5,$J6,$J7 --format=JobID,JobName%22,State,Elapsed -n"
