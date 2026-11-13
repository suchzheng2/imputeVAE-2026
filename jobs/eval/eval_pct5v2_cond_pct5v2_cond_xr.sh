#!/bin/bash
#SBATCH --job-name=eval_pct5v2_cond
#SBATCH --partition=pi_gerstein
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --output=jobs/logs/slurm-%j_eval.out
#SBATCH --error=jobs/logs/slurm-%j_eval.err

module purge
module load miniconda
conda activate impute2

cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

python -m analysis.eval.runner \
    --model pct5v2_cond,pct5v2_cond_xr \
    --eval ct_level,sample_level,latent,gene_stratified \
    --output_dir /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE/notes/eval_results

echo "Evaluation complete: $(date)"
