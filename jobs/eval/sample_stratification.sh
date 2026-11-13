#!/bin/bash
#SBATCH --job-name=sample_strat
#SBATCH --partition=pi_gerstein
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --output=jobs/logs/slurm-%j_sample_strat.out
#SBATCH --error=jobs/logs/slurm-%j_sample_strat.err

source ~/.bashrc
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

echo "========================================="
echo "1. Gene stratification: MSE Cond (re-run with fixed heatmap)"
echo "========================================="
python analysis/gene_stratification.py \
  --run_folder run_2026-04-02-02-02-35_3345839 \
  --epoch 599

echo ""
echo "========================================="
echo "2. Gene stratification: MSE Cond+XR (re-run with fixed heatmap)"
echo "========================================="
python analysis/gene_stratification.py \
  --run_folder run_2026-04-02-10-51-32_3738876 \
  --epoch 599

echo ""
echo "========================================="
echo "3. Sample stratification: MSE Cond+XR"
echo "========================================="
python analysis/sample_stratification.py \
  --run_folder run/run_2026-04-02-10-51-32_3738876 \
  --model_class VAE_JSD_Cond

echo ""
echo "========================================="
echo "4. Sample stratification: MSE Cond"
echo "========================================="
python analysis/sample_stratification.py \
  --run_folder run/run_2026-04-02-02-02-35_3345839 \
  --model_class VAE_JSD_Cond

echo ""
echo "All stratification analyses complete."
