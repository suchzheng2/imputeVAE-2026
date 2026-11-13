#!/bin/bash
#SBATCH --job-name=gse219278_val
#SBATCH --partition=scavenge
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=jobs/logs/slurm-%j_gse219278.out
#SBATCH --error=jobs/logs/slurm-%j_gse219278.err

cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
source activate impute2

echo "=== Step 4: Human M1 pseudobulk ==="
python validation/gse219278_sorted_population_validation/scripts/04_make_human_m1_pseudobulk.py

echo ""
echo "=== Step 5: Harmonize gene spaces ==="
python validation/gse219278_sorted_population_validation/scripts/05_harmonize_gene_spaces.py

echo ""
echo "=== Step 6: Main validation ==="
python validation/gse219278_sorted_population_validation/scripts/06_main_validation.py

echo ""
echo "=== DONE ==="
