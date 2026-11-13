#!/usr/bin/env python3
"""Step 6: Make human M1 three-group pseudobulk profiles.

Groups Allen Human M1 nuclei into three populations matching GSE219278:
  - Neurons: excitatory_neuron + inhibitory_neuron
  - OligoLineage: oligodendrocyte + opc
  - OtherGlia: astrocyte + microglia_macrophage
  (Excludes: endothelial, vlmc_pericyte_smc — too few cells and no GSE match)

For each group, computes:
  - Raw nuclei pseudobulk: mean log1p expression (model gene space)
  - Imputed cell pseudobulk: mean imputed log1p expression (model gene space)
  - Predicted delta = imputed - raw (population-level mean shift)

Also does per-donor breakdown for robustness analysis.
"""
import pandas as pd
import numpy as np
from scipy import sparse
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HM1 = os.path.join(os.path.dirname(BASE), "human_m1_transfer")
PROC = os.path.join(BASE, "data", "processed")
OUT = os.path.join(BASE, "outputs", "tables")
os.makedirs(PROC, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

# Group mapping
GROUP_MAP = {
    "excitatory_neuron": "Neurons",
    "inhibitory_neuron": "Neurons",
    "oligodendrocyte": "OligoLineage",
    "opc": "OligoLineage",
    "astrocyte": "OtherGlia",
    "microglia_macrophage": "OtherGlia",
}
EXCLUDE = {"endothelial", "vlmc_pericyte_smc"}

# ---------- Load data ----------
print("Loading obs metadata...")
obs = pd.read_csv(os.path.join(HM1, "processed", "human_m1_nuclei_obs.csv"))
print(f"  Total nuclei: {len(obs)}")

print("Loading gene mapping...")
gene_map = pd.read_csv(os.path.join(HM1, "processed", "human_m1_gene_mapping_for_model_order.csv"))

print("Loading raw nuclei (sparse)...")
raw_csr = sparse.load_npz(os.path.join(HM1, "processed", "human_m1_nuclei_mouse_model_order.npz")).tocsr()
n_cells, n_genes = raw_csr.shape
print(f"  Raw shape: {raw_csr.shape}")

print("Loading imputed cell profiles (mmap)...")
imputed = np.load(os.path.join(HM1, "inference", "human_m1_imputed_cell_log1p.npy"), mmap_mode="r")
print(f"  Imputed shape: {imputed.shape}")

assert raw_csr.shape == imputed.shape
assert len(obs) == n_cells

# ---------- Assign groups ----------
obs["gse_group"] = obs["mapped_broad_class"].map(GROUP_MAP)
excluded = obs["mapped_broad_class"].isin(EXCLUDE)
print(f"\nExcluded cell types: {obs.loc[excluded, 'mapped_broad_class'].value_counts().to_dict()}")

valid_idx = np.where(~excluded.values)[0]
obs_valid = obs.iloc[valid_idx].copy()

print(f"Nuclei after exclusion: {len(obs_valid)}")
print(f"Group sizes:")
print(obs_valid["gse_group"].value_counts().to_string())


def sparse_log1p_mean(csr_mat, row_indices):
    """Compute mean(log1p(x)) for selected rows without densifying full matrix."""
    sub = csr_mat[row_indices]
    # Convert chunk-wise to avoid OOM
    chunk_size = 5000
    accum = np.zeros(csr_mat.shape[1], dtype=np.float64)
    n = len(row_indices)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_dense = np.array(sub[start:end].todense(), dtype=np.float32)
        accum += np.log1p(chunk_dense).sum(axis=0).A1 if hasattr(chunk_dense, 'A1') else np.log1p(chunk_dense).sum(axis=0)
    return (accum / n).astype(np.float32)


# ---------- Compute pseudobulk ----------
groups = ["Neurons", "OligoLineage", "OtherGlia"]
donors = sorted(obs_valid["donor"].unique())

pseudobulk_records = []
raw_profiles = {}
imp_profiles = {}

for grp in groups:
    grp_mask = obs_valid["gse_group"].values == grp
    grp_global_idx = valid_idx[grp_mask]

    raw_mean = sparse_log1p_mean(raw_csr, grp_global_idx)
    imp_mean = imputed[grp_global_idx].mean(axis=0)
    delta = imp_mean - raw_mean

    raw_profiles[grp] = raw_mean
    imp_profiles[grp] = imp_mean

    pseudobulk_records.append({
        "group": grp,
        "n_nuclei": int(grp_mask.sum()),
        "mean_raw_log1p": float(raw_mean.mean()),
        "mean_imputed_log1p": float(imp_mean.mean()),
        "mean_predicted_delta": float(delta.mean()),
        "median_predicted_delta": float(np.median(delta)),
        "n_positive_delta": int((delta > 0).sum()),
        "n_negative_delta": int((delta < 0).sum()),
    })
    print(f"\n{grp} (n={grp_mask.sum()}):")
    print(f"  Raw mean: {raw_mean.mean():.4f}")
    print(f"  Imputed mean: {imp_mean.mean():.4f}")
    print(f"  Predicted delta mean: {delta.mean():.4f}")

pb_df = pd.DataFrame(pseudobulk_records)
pb_df.to_csv(os.path.join(OUT, "human_m1_pseudobulk_summary.csv"), index=False)

# Save per-gene profiles (in model gene space)
profile_df = gene_map[["model_gene_index", "mouse_model_gene", "human_gene_symbol"]].copy()
for grp in groups:
    profile_df[f"{grp}_raw_log1p"] = raw_profiles[grp]
    profile_df[f"{grp}_imputed_log1p"] = imp_profiles[grp]
    profile_df[f"{grp}_predicted_delta"] = imp_profiles[grp] - raw_profiles[grp]

profile_df.to_csv(os.path.join(PROC, "human_m1_pseudobulk_per_gene.csv.gz"), index=False)

# ---------- Per-donor pseudobulk ----------
donor_records = []
for grp in groups:
    for donor in donors:
        grp_donor_mask = ((obs_valid["gse_group"].values == grp) &
                          (obs_valid["donor"].values == donor))
        n = grp_donor_mask.sum()
        if n == 0:
            continue
        gd_global_idx = valid_idx[grp_donor_mask]
        raw_mean = sparse_log1p_mean(raw_csr, gd_global_idx)
        imp_mean = imputed[gd_global_idx].mean(axis=0)
        delta = imp_mean - raw_mean
        donor_records.append({
            "group": grp,
            "donor": donor,
            "n_nuclei": int(n),
            "mean_raw_log1p": float(raw_mean.mean()),
            "mean_imputed_log1p": float(imp_mean.mean()),
            "mean_predicted_delta": float(delta.mean()),
        })

donor_df = pd.DataFrame(donor_records)
donor_df.to_csv(os.path.join(OUT, "human_m1_pseudobulk_by_donor.csv"), index=False)
print(f"\nPer-donor breakdown:")
print(donor_df.to_string(index=False))

print(f"\nAll outputs saved to {PROC} and {OUT}")
