#!/bin/bash
#SBATCH --job-name=eval_v3_runs
#SBATCH --partition=pi_gerstein,scavenge_gpu
#SBATCH --gpus=1
#SBATCH --mem=400G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=jobs/logs/slurm-%j_eval_v3_runs.out
#SBATCH --error=jobs/logs/slurm-%j_eval_v3_runs.err

source ~/.bashrc
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

# Post-training evaluation for all 4 Config B V3 runs:
#   MSE stage1: run_2026-04-01-09-29-26_3250320 (cell only, 10134 genes)
#   NB  stage1: run_2026-04-01-09-29-27_2189144 (cell only, 10134 genes)
#   MSE stage2: run_2026-04-01-14-01-16_1946018 (joint cell+nuclei)
#   NB  stage2: run_2026-04-01-14-43-14_475265  (joint cell+nuclei)
#
# Analyses: loss curves, correlation/shift, gene correlation, heatmaps
# (skip scib for now — requires abc_env subprocess)

echo "========================================"
echo "=== MSE Stage 1 (cell only) ==="
echo "========================================"
python analysis/evaluate.py \
  --run_folder run_2026-04-01-09-29-26_3250320 \
  --modality cell \
  --analyses loss,scale,correlation,gene_corr,heatmaps \
  --epochs 0,250,500,599

echo ""
echo "========================================"
echo "=== NB Stage 1 (cell only) ==="
echo "========================================"
python analysis/evaluate.py \
  --run_folder run_2026-04-01-09-29-27_2189144 \
  --modality cell \
  --analyses loss,scale,correlation,gene_corr,heatmaps \
  --epochs 0,250,500,599

echo ""
echo "========================================"
echo "=== MSE Stage 2 (joint cell+nuclei) ==="
echo "========================================"
python analysis/evaluate.py \
  --run_folder run_2026-04-01-14-01-16_1946018 \
  --modality nuclei \
  --analyses loss,scale,correlation,gene_corr,heatmaps \
  --epochs 0,50,100,200,300,400,500,599

echo ""
echo "========================================"
echo "=== NB Stage 2 (joint cell+nuclei) ==="
echo "========================================"
python analysis/evaluate.py \
  --run_folder run_2026-04-01-14-43-14_475265 \
  --modality nuclei \
  --analyses loss,scale,correlation,gene_corr,heatmaps \
  --epochs 0,50,100,200,300,400,500,599

echo "All evaluations complete."
