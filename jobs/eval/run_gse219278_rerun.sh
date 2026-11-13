#!/bin/bash
#SBATCH --job-name=gse219278_v2
#SBATCH --partition=scavenge
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=jobs/logs/slurm-%j_gse219278_v2.out
#SBATCH --error=jobs/logs/slurm-%j_gse219278_v2.err

cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
source activate impute2

echo "=== Step 5: Harmonize gene spaces (JASPAR fix) ==="
python validation/gse219278_sorted_population_validation/scripts/05_harmonize_gene_spaces.py

echo ""
echo "=== Step 6: Main validation ==="
python validation/gse219278_sorted_population_validation/scripts/06_main_validation.py

echo ""
echo "=== Step 7: Figures ==="
python validation/gse219278_sorted_population_validation/scripts/07_make_figures.py

echo ""
echo "=== DONE ==="
