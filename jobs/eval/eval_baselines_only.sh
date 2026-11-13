#!/bin/bash
#SBATCH --job-name=eval_baselines
#SBATCH --partition=pi_gerstein
#SBATCH --time=2:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=4
#SBATCH --output=jobs/logs/slurm-%j_eval_baselines.out
#SBATCH --error=jobs/logs/slurm-%j_eval_baselines.err

source ~/.bashrc
conda activate impute2

cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

find imputeVAE/ analysis/ -name "__pycache__" -exec rm -rf {} + 2>/dev/null

echo "=== Baseline-only evaluation (no GPU needed) ==="
echo "Started: $(date)"

python -m analysis.eval.runner \
    --model baseline_raw_nuclei,baseline_ct_mean,baseline_linear_reg \
    --eval ct_level,sample_level,gene_stratified \
    --output_dir notes/eval_results

echo "Done: $(date)"
