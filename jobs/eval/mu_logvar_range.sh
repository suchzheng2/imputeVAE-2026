#!/bin/bash
#SBATCH --job-name=mu_logvar_range
#SBATCH --output=jobs/logs/slurm-%j_mu_logvar_range.out
#SBATCH --error=jobs/logs/slurm-%j_mu_logvar_range.err
#SBATCH --partition=scavenge
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00

source ~/.bashrc
conda activate impute2

cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

# Delete stale bytecode
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# Runs to examine:
#   log1p S1 — VAE_JSD, cells,  log1p normalisation
#   log1p S2 — VAE_JSD_Cond, nuclei, log1p normalisation
#   CPM   S1 — VAE_JSD, cells,  CPM normalisation
#   CPM   S2 — VAE_JSD_Cond, nuclei, CPM normalisation (JSD stuck at ln(2))

python analysis/mu_logvar_range.py \
    --runs \
        "run/run_2026-04-02-19-27-35_3767601:log1p_s1" \
        "run/run_2026-04-07-02-12-05_3779214:log1p_s2_cond" \
        "run/run_2026-04-23-23-59-34_1163351:cpm_s1" \
        "run/run_2026-04-28-22-16-14_990383:cpm_s2_cond" \
    --output_dir notes/experiments/SUC-6_variance_retention/mu_logvar_range \
    --max_cells 8000 \
    2>&1
