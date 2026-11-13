#!/bin/bash
#SBATCH --job-name=data_audit
#SBATCH --partition=scavenge_gpu
#SBATCH --requeue
#SBATCH --gpus=1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=jobs/logs/slurm-%j_data_audit.out
#SBATCH --error=jobs/logs/slurm-%j_data_audit.err

source ~/.bashrc
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

# Delete stale bytecode
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

python -m analysis.data_flow_audit \
    --model pct5v2_cond \
    --out audit_results \
    --max-per-type 2000
