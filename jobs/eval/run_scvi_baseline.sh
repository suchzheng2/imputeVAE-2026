#!/bin/bash
#SBATCH --job-name=scvi_baseline
#SBATCH --partition=pi_gerstein,scavenge_gpu
#SBATCH --gpus=1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=jobs/logs/slurm-%j_scvi_baseline.out
#SBATCH --error=jobs/logs/slurm-%j_scvi_baseline.err

source ~/.bashrc
conda activate scvi2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

# Delete stale bytecode
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

echo "scVI baseline — $(date)"
python -c "import scvi; print(f'scvi-tools {scvi.__version__}')"

OUTPUT_DIR=run/scvi_baseline

# --- Step 1: Train scVI and generate imputed expression ---
python analysis/baseline_scvi.py \
    --n_latent 30 \
    --n_layers 2 \
    --n_hidden 256 \
    --max_epochs 200 \
    --batch_size 256 \
    --lr 1e-3 \
    --gene_list data/training_gene_list_v3.csv \
    --output_dir $OUTPUT_DIR

# --- Step 2: Evaluate ---
if [ ! -f "$OUTPUT_DIR/imputed_cell_from_nucleus.npy" ]; then
    echo "ERROR: Training did not produce output. Skipping evaluation."
    exit 1
fi

echo ""
echo "=== Running evaluation ==="
conda activate impute2

python analysis/evaluate_baseline.py \
    --imputed_npy $OUTPUT_DIR/imputed_cell_from_nucleus.npy \
    --obs_csv $OUTPUT_DIR/nucleus_obs.csv \
    --name scvi \
    --gene_list data/training_gene_list_v3.csv \
    --output_dir $OUTPUT_DIR/eval

echo "Job finished at $(date)"
