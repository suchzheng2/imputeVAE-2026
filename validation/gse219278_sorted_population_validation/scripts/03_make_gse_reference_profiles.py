#!/usr/bin/env python3
"""Step 4: Make GSE219278 reference profiles.

For each population (Neurons, OligoLineage, OtherGlia):
- Compute mean and median TPM across 6 donors
- Log2(TPM+1) transform
- Compute pairwise donor correlations
"""
import pandas as pd
import numpy as np
from scipy import stats
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")
OUT = os.path.join(BASE, "outputs", "tables")
os.makedirs(OUT, exist_ok=True)

# Load subset
df = pd.read_csv(os.path.join(PROC, "gse219278_control_motorcortex_tpm.csv.gz"))
annot_cols = ["gene_ID", "gene_name", "gene_type"]

populations = ["Neurons", "OligoLineage", "OtherGlia"]
profiles = {}

for pop in populations:
    pop_cols = [c for c in df.columns if c not in annot_cols and c.endswith(f"_{pop}")]
    tpm_mat = df[pop_cols].values  # (genes, donors)

    # Mean/median TPM
    mean_tpm = tpm_mat.mean(axis=1)
    median_tpm = np.median(tpm_mat, axis=1)

    # Log2(TPM+1) per donor
    log2_mat = np.log2(tpm_mat + 1)
    mean_log2 = log2_mat.mean(axis=1)

    profiles[pop] = {
        "mean_tpm": mean_tpm,
        "median_tpm": median_tpm,
        "mean_log2_tpm_p1": mean_log2,
    }

    # Donor-donor correlations
    n_donors = tpm_mat.shape[1]
    donor_corrs = []
    for i in range(n_donors):
        for j in range(i+1, n_donors):
            r, _ = stats.pearsonr(log2_mat[:, i], log2_mat[:, j])
            donor_corrs.append(r)

    print(f"{pop}: {n_donors} donors, mean donor-donor r={np.mean(donor_corrs):.4f} "
          f"(range {np.min(donor_corrs):.4f}-{np.max(donor_corrs):.4f}), "
          f"mean genes detected={(tpm_mat > 0).sum(axis=0).mean():.0f}")

# Build reference table
ref = df[annot_cols].copy()
for pop in populations:
    ref[f"{pop}_mean_tpm"] = profiles[pop]["mean_tpm"]
    ref[f"{pop}_median_tpm"] = profiles[pop]["median_tpm"]
    ref[f"{pop}_mean_log2_tpm_p1"] = profiles[pop]["mean_log2_tpm_p1"]

ref.to_csv(os.path.join(PROC, "gse219278_reference_profiles.csv.gz"), index=False)
print(f"\nSaved reference profiles: {ref.shape}")

# Compute population-vs-population deltas (Neurons - OligoLineage, etc.)
delta_records = []
for pop_a in populations:
    for pop_b in populations:
        if pop_a == pop_b:
            continue
        diff = profiles[pop_a]["mean_log2_tpm_p1"] - profiles[pop_b]["mean_log2_tpm_p1"]
        delta_records.append({
            "comparison": f"{pop_a}_minus_{pop_b}",
            "mean_delta": diff.mean(),
            "median_delta": np.median(diff),
            "n_positive": (diff > 0).sum(),
            "n_negative": (diff < 0).sum(),
        })

delta_df = pd.DataFrame(delta_records)
delta_df.to_csv(os.path.join(OUT, "gse219278_population_deltas_summary.csv"), index=False)
print(f"\nPopulation deltas:")
print(delta_df.to_string(index=False))
