#!/bin/bash
#SBATCH --job-name=scbutterfly_baseline
#SBATCH --partition=pi_gerstein,scavenge_gpu
#SBATCH --gpus=1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
#SBATCH --output=jobs/logs/slurm-%j_scbutterfly_baseline.out
#SBATCH --error=jobs/logs/slurm-%j_scbutterfly_baseline.err

source ~/.bashrc
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE

# Delete stale bytecode
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

export PYTHONUNBUFFERED=1
echo "scButterfly baseline (CITE variant, RNA->RNA) — $(date)"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

OUTPUT_DIR=run/scbutterfly_baseline

# --- Step 1: Train scButterfly and generate imputed expression ---
python analysis/baseline_scbutterfly.py \
    --output_dir $OUTPUT_DIR \
    --gene_list data/training_gene_list_v3.csv \
    --R2R_pretrain_epoch 100 \
    --A2A_pretrain_epoch 100 \
    --translator_epoch 200 \
    --patience 50 \
    --batch_size 64 \
    --lr 0.001 \
    --seed 19193

# --- Step 2: Evaluate ---
if [ ! -f "$OUTPUT_DIR/imputed_cell_from_nucleus.npy" ]; then
    echo "ERROR: Training did not produce output. Skipping evaluation."
    exit 1
fi

echo ""
echo "=== Running evaluation ==="

python analysis/evaluate_baseline_incremental.py \
    --imputed_npy $OUTPUT_DIR/imputed_cell_from_nucleus.npy \
    --obs_csv $OUTPUT_DIR/nucleus_obs.csv \
    --name scbutterfly \
    --output_dir $OUTPUT_DIR/eval

echo "Job finished at $(date)"
