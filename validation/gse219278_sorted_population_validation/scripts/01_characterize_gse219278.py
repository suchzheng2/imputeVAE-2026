#!/usr/bin/env python3
"""Step 2: Characterize GSE219278 sorted-population RNA-seq data.

Reads the raw TPM and counts files, reports:
- Full sample inventory (condition x region x population x donor)
- Gene counts by gene_type
- TPM distribution summary
- Identifies Control_MotorCortex subset for validation
"""
import pandas as pd
import numpy as np
import os, sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "data", "raw")
OUT = os.path.join(BASE, "outputs", "tables")
os.makedirs(OUT, exist_ok=True)

# ---------- Load TPM ----------
tpm_path = os.path.join(RAW, "GSE219278_allSamples_rsem_genes_results_TPM_annotated.tsv.gz")
print(f"Reading TPM: {tpm_path}")
df_tpm = pd.read_csv(tpm_path, sep="\t")
print(f"  Shape: {df_tpm.shape}")

annot_cols = ["gene_ID", "gene_name", "gene_type"]
sample_cols = [c for c in df_tpm.columns if c not in annot_cols]
print(f"  Annotation columns: {annot_cols}")
print(f"  Sample columns: {len(sample_cols)}")

# ---------- Parse sample metadata ----------
records = []
for col in sample_cols:
    parts = col.split("_")
    # Format: {condition}_{donor}_{region}_{population}
    # But condition can be multi-word like C9ALS
    # Known conditions: C9ALS, Control, sALS
    # Donor: A1, A2, C1-C6, S1-S6
    # Region: FrontalCortex, MotorCortex
    # Population: Neurons, OligoLineage, OtherGlia
    condition = parts[0]
    donor = parts[1]
    region = parts[2]
    population = parts[3]
    records.append({
        "sample_column": col,
        "condition": condition,
        "donor": donor,
        "region": region,
        "population": population,
    })

sample_meta = pd.DataFrame(records)
print(f"\n--- Sample inventory ---")
print(f"Conditions: {sorted(sample_meta['condition'].unique())}")
print(f"Regions: {sorted(sample_meta['region'].unique())}")
print(f"Populations: {sorted(sample_meta['population'].unique())}")
print(f"\nCondition x Region counts:")
print(sample_meta.groupby(["condition", "region"]).size().unstack(fill_value=0))
print(f"\nCondition x Population counts:")
print(sample_meta.groupby(["condition", "population"]).size().unstack(fill_value=0))

# Control MotorCortex subset
ctrl_mc = sample_meta[
    (sample_meta["condition"] == "Control") & (sample_meta["region"] == "MotorCortex")
]
print(f"\n--- Control MotorCortex subset ---")
print(f"  Samples: {len(ctrl_mc)}")
print(f"  Donors: {sorted(ctrl_mc['donor'].unique())}")
print(f"  Populations: {sorted(ctrl_mc['population'].unique())}")
for _, row in ctrl_mc.iterrows():
    print(f"    {row['sample_column']}")

# ---------- Gene types ----------
print(f"\n--- Gene types ---")
gt_counts = df_tpm["gene_type"].value_counts()
print(gt_counts.to_string())

# ---------- TPM summary ----------
ctrl_mc_cols = ctrl_mc["sample_column"].tolist()
tpm_ctrl = df_tpm[ctrl_mc_cols]
print(f"\n--- TPM summary (Control MotorCortex) ---")
for pop in ["Neurons", "OligoLineage", "OtherGlia"]:
    pop_cols = [c for c in ctrl_mc_cols if c.endswith(f"_{pop}")]
    vals = df_tpm[pop_cols].values
    nonzero = (vals > 0).sum(axis=0).mean()
    print(f"  {pop}: mean TPM={vals.mean():.3f}, median={np.median(vals):.3f}, "
          f"mean genes detected={nonzero:.0f}/{len(df_tpm)}")

# ---------- Save metadata ----------
sample_meta.to_csv(os.path.join(OUT, "gse219278_sample_metadata.csv"), index=False)
ctrl_mc.to_csv(os.path.join(OUT, "gse219278_control_motorcortex_samples.csv"), index=False)
gt_counts.to_csv(os.path.join(OUT, "gse219278_gene_type_counts.csv"))
print(f"\nSaved to {OUT}")
