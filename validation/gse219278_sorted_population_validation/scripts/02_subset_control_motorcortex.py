#!/usr/bin/env python3
"""Step 3: Subset Control MotorCortex samples from GSE219278.

Extracts 18 samples (6 donors x 3 populations) of TPM data.
Saves per-population and combined matrices.
"""
import pandas as pd
import numpy as np
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "data", "raw")
PROC = os.path.join(BASE, "data", "processed")
os.makedirs(PROC, exist_ok=True)

# Load TPM
tpm_path = os.path.join(RAW, "GSE219278_allSamples_rsem_genes_results_TPM_annotated.tsv.gz")
print("Loading TPM data...")
df = pd.read_csv(tpm_path, sep="\t")

annot_cols = ["gene_ID", "gene_name", "gene_type"]
sample_cols = [c for c in df.columns if c not in annot_cols]

# Filter to Control MotorCortex
ctrl_mc_cols = [c for c in sample_cols if c.startswith("Control_") and "MotorCortex" in c]
assert len(ctrl_mc_cols) == 18, f"Expected 18 Control_MotorCortex columns, got {len(ctrl_mc_cols)}"

# Create subset dataframe
df_sub = df[annot_cols + ctrl_mc_cols].copy()
print(f"Subset shape: {df_sub.shape}")

# Save combined
df_sub.to_csv(os.path.join(PROC, "gse219278_control_motorcortex_tpm.csv.gz"), index=False)
print(f"Saved combined: {df_sub.shape}")

# Save per-population
for pop in ["Neurons", "OligoLineage", "OtherGlia"]:
    pop_cols = [c for c in ctrl_mc_cols if c.endswith(f"_{pop}")]
    df_pop = df[annot_cols + pop_cols].copy()
    df_pop.to_csv(os.path.join(PROC, f"gse219278_control_motorcortex_{pop.lower()}_tpm.csv.gz"), index=False)
    print(f"  {pop}: {len(pop_cols)} samples, {df_pop.shape}")

# Also save gene annotation separately
df[annot_cols].to_csv(os.path.join(PROC, "gse219278_gene_annotations.csv"), index=False)

print(f"\nAll files saved to {PROC}")
