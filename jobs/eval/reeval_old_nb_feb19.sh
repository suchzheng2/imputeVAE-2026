#!/bin/bash
#SBATCH --job-name=reeval_nb_feb19
#SBATCH --partition=pi_gerstein
#SBATCH --mem=400G
#SBATCH --cpus-per-task=4
#SBATCH --time=3:00:00
#SBATCH --output=jobs/logs/slurm-%j_reeval_nb_feb19.out
#SBATCH --error=jobs/logs/slurm-%j_reeval_nb_feb19.err

source ~/.bashrc
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

CELL_RUN="/home/sz477/project/cell-imputing/Cytoplasm_imputation/run/run_2026-02-19-16-24-30_3760179"
NUCLEI_RUN="/home/sz477/project/cell-imputing/Cytoplasm_imputation/run/run_2026-02-19-16-22-49_3589992"
OUTPUT="/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/stage1-analysis/reeval_nb_feb19"

python analysis/reeval_old_nb_runs.py \
  --cell_run_path "$CELL_RUN" \
  --nuclei_run_path "$NUCLEI_RUN" \
  --output_dir "$OUTPUT" \
  --subsample_fraction 0.1

echo "Re-evaluation complete."
