#!/bin/bash
#SBATCH --job-name=eval_paper
#SBATCH --partition=scavenge_gpu
#SBATCH --time=8:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --output=jobs/logs/slurm-%j_eval_paper.out
#SBATCH --error=jobs/logs/slurm-%j_eval_paper.err

source ~/.bashrc
conda activate impute2

cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
export PYTHONUNBUFFERED=1

# Delete stale bytecode
find imputeVAE/ analysis/ -name "__pycache__" -exec rm -rf {} + 2>/dev/null

echo "=== Paper model evaluation ==="
echo "Started: $(date)"
echo ""

# baseline_raw_nuclei already completed — run remaining 7 models
python -m analysis.eval.runner \
    --model baseline_ct_mean,baseline_linear_reg,configB_shared,configB_cond,configB_cond_xr,configB_cond_normloss,configB_single_stage_cVAE \
    --eval all \
    --output_dir notes/eval_results

echo ""
echo "Evaluation complete: $(date)"
