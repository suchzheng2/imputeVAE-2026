#!/bin/bash
#SBATCH --job-name=conserved_transform
#SBATCH --partition=scavenge
#SBATCH --requeue
#SBATCH --mem=300G
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --output=jobs/logs/slurm-%j_conserved_transform.out
#SBATCH --error=jobs/logs/slurm-%j_conserved_transform.err

source ~/.bashrc
conda activate impute2
cd /gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
export PYTHONUNBUFFERED=1

find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

CELL_H5AD="data/external/lake2023_cells.h5ad"
NUC_H5AD="data/external/lake2023_nuclei.h5ad"

# Download if not present
mkdir -p data/external
if [ ! -f "$CELL_H5AD" ]; then
    echo "Downloading Lake 2023 cells..."
    wget -q -O "$CELL_H5AD" \
        "https://datasets.cellxgene.cziscience.com/08648660-9898-4644-a9a1-ff3a159a0041.h5ad"
    echo "  Done: $(ls -lh $CELL_H5AD | awk '{print $5}')"
fi

if [ ! -f "$NUC_H5AD" ]; then
    echo "Downloading Lake 2023 nuclei..."
    wget -q -O "$NUC_H5AD" \
        "https://datasets.cellxgene.cziscience.com/02d5db96-d67e-45f2-b556-b6bd99753665.h5ad"
    echo "  Done: $(ls -lh $NUC_H5AD | awk '{print $5}')"
fi

echo "Running conserved transformation analysis..."
python analysis/conserved_transformation.py \
    --cell_h5ad "$CELL_H5AD" \
    --nuc_h5ad "$NUC_H5AD" \
    --output_dir notes/experiments/conserved_transformation
