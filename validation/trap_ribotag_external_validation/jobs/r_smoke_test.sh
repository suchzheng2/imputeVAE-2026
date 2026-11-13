#!/bin/bash
#SBATCH --job-name=r_smoke
#SBATCH --partition=scavenge
#SBATCH --time=0:10:00
#SBATCH --mem=4G
#SBATCH --output=jobs/logs/slurm-%j_r_smoke.out
#SBATCH --error=jobs/logs/slurm-%j_r_smoke.err
module purge
module load R-bundle-Bioconductor/3.18-foss-2022b-R-4.3.2
which Rscript
Rscript -e 'for (p in c("edgeR","limma","DESeq2","statmod")) cat(p, requireNamespace(p, quietly=TRUE), "\n"); cat(R.version.string, "\n")'
