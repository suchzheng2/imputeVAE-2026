#!/bin/bash
#SBATCH --job-name=eval_pct5v2_cond_xr
#SBATCH --partition=scavenge_gpu,pi_gerstein
#SBATCH --time=4:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --output=jobs/logs/slurm-%j_eval.out
#SBATCH --error=jobs/logs/slurm-%j_eval.err
#SBATCH --requeue

source ~/.bashrc
conda activate impute2

cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
rm -rf __pycache__ analysis/__pycache__ analysis/eval/__pycache__ analysis/eval/metrics/__pycache__

python -m analysis.eval.runner \
    --model pct5v2_cond_xr \
    --eval ct_level,sample_level,latent,gene_stratified \
    --output_dir /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE/notes/eval_results

echo "Evaluation complete: $(date)"
