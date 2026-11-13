#!/usr/bin/env python3
"""Steps 9-12: Main validation metrics, delta-direction, reference matching, donor robustness.

Validation logic:
  GSE219278 sorted populations give us empirical "whole-cell vs nucleus" differences
  at the population level. ImputeVAE predicts cell-like profiles from nuclei.

  For each gene in the harmonized set:
    - GSE empirical direction: sign(GSE_Neurons_log2TPM - GSE_OligoLineage_log2TPM) etc.
    - Model predicted delta direction: sign(imputed_Neurons - raw_Neurons) etc.

  Key metrics:
    1. Profile correlation: GSE population profile vs M1 imputed pseudobulk
    2. Delta-direction agreement: does the model predict shifts in the right direction?
    3. Reference matching: correlation matrix between GSE and M1 profiles
    4. Gene set enrichment: SynGO vs JASPAR TF shift magnitudes
    5. Donor robustness: leave-one-donor-out stability
"""
import pandas as pd
import numpy as np
from scipy import stats
import os, json

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")
OUT = os.path.join(BASE, "outputs", "tables")
HM1 = os.path.join(os.path.dirname(BASE), "human_m1_transfer")
os.makedirs(OUT, exist_ok=True)

# ---------- Load harmonized data ----------
print("Loading harmonized gene profiles...")
harm = pd.read_csv(os.path.join(PROC, "harmonized_gene_profiles.csv.gz"))
print(f"  {len(harm)} harmonized genes")

populations = ["Neurons", "OligoLineage", "OtherGlia"]

# =====================================================================
# STEP 9: Main Validation Metrics
# =====================================================================
print("\n" + "="*60)
print("STEP 9: Main Validation Metrics")
print("="*60)

results = {}

# 9a. Profile correlation: GSE log2(TPM+1) vs M1 imputed log1p
# These are different scales (log2 vs ln), but correlation is scale-invariant
for pop in populations:
    gse_vals = harm[f"gse_{pop}_mean_log2_tpm_p1"].values
    m1_imp = harm[f"m1_{pop}_imputed_log1p"].values
    m1_raw = harm[f"m1_{pop}_raw_log1p"].values

    r_imp, p_imp = stats.pearsonr(gse_vals, m1_imp)
    rho_imp, _ = stats.spearmanr(gse_vals, m1_imp)
    r_raw, p_raw = stats.pearsonr(gse_vals, m1_raw)
    rho_raw, _ = stats.spearmanr(gse_vals, m1_raw)

    results[f"profile_corr_{pop}"] = {
        "pearson_imputed": r_imp, "spearman_imputed": rho_imp,
        "pearson_raw": r_raw, "spearman_raw": rho_raw,
        "pearson_improvement": r_imp - r_raw,
    }
    print(f"\n{pop}:")
    print(f"  GSE vs M1 imputed: Pearson r={r_imp:.4f}, Spearman rho={rho_imp:.4f}")
    print(f"  GSE vs M1 raw:     Pearson r={r_raw:.4f}, Spearman rho={rho_raw:.4f}")
    print(f"  Improvement:       {r_imp - r_raw:+.4f}")

# 9b. Predicted delta magnitude by gene set
for gene_set, label in [("is_syngo", "SynGO"), ("is_jaspar_tf", "JASPAR_TF")]:
    mask = harm[gene_set].values
    subset = harm[mask]
    print(f"\n{label} (n={mask.sum()}):")
    for pop in populations:
        delta = subset[f"m1_{pop}_predicted_delta"].values
        print(f"  {pop}: mean delta={delta.mean():.4f}, median={np.median(delta):.4f}")

    results[f"geneset_{label}"] = {
        "n_genes": int(mask.sum()),
        "Neurons_mean_delta": float(subset["m1_Neurons_predicted_delta"].mean()),
        "OligoLineage_mean_delta": float(subset["m1_OligoLineage_predicted_delta"].mean()),
        "OtherGlia_mean_delta": float(subset["m1_OtherGlia_predicted_delta"].mean()),
    }

# =====================================================================
# STEP 10: Delta-Direction Validation
# =====================================================================
print("\n" + "="*60)
print("STEP 10: Delta-Direction Validation")
print("="*60)

# For each population, the "empirical" direction from GSE is:
# GSE_Neurons > GSE_OligoLineage for neuron-enriched genes
# The model's predicted shift should be larger for Neurons than OligoLineage
# i.e., if a gene is neuron-enriched in GSE, model should predict bigger delta for Neurons

# Pairwise comparisons
comparisons = [
    ("Neurons", "OligoLineage"),
    ("Neurons", "OtherGlia"),
    ("OligoLineage", "OtherGlia"),
]

direction_results = []
for pop_a, pop_b in comparisons:
    # GSE empirical direction: which population has higher expression?
    gse_diff = harm[f"gse_{pop_a}_mean_log2_tpm_p1"].values - harm[f"gse_{pop_b}_mean_log2_tpm_p1"].values

    # Model predicted delta difference: which population gets a bigger imputation shift?
    model_delta_diff = harm[f"m1_{pop_a}_predicted_delta"].values - harm[f"m1_{pop_b}_predicted_delta"].values

    # Also compare imputed profiles directly
    m1_imp_diff = harm[f"m1_{pop_a}_imputed_log1p"].values - harm[f"m1_{pop_b}_imputed_log1p"].values

    # Direction agreement: does the model's imputed profile rank match GSE?
    sign_agree_imp = np.mean(np.sign(gse_diff) == np.sign(m1_imp_diff))
    # Correlation of differences
    r_diff, _ = stats.pearsonr(gse_diff, m1_imp_diff)
    rho_diff, _ = stats.spearmanr(gse_diff, m1_imp_diff)

    rec = {
        "comparison": f"{pop_a}_vs_{pop_b}",
        "sign_agreement_imputed": sign_agree_imp,
        "pearson_diff": r_diff,
        "spearman_diff": rho_diff,
        "n_genes": len(gse_diff),
        "gse_mean_diff": gse_diff.mean(),
        "m1_imp_mean_diff": m1_imp_diff.mean(),
    }
    direction_results.append(rec)
    print(f"\n{pop_a} vs {pop_b}:")
    print(f"  Sign agreement (imputed vs GSE): {sign_agree_imp:.4f}")
    print(f"  Pearson r (diff): {r_diff:.4f}")
    print(f"  Spearman rho (diff): {rho_diff:.4f}")

dir_df = pd.DataFrame(direction_results)
dir_df.to_csv(os.path.join(OUT, "delta_direction_validation.csv"), index=False)

# Per gene-set direction analysis
for gene_set, label in [("is_syngo", "SynGO"), ("is_jaspar_tf", "JASPAR_TF")]:
    mask = harm[gene_set].values
    if mask.sum() == 0:
        continue
    subset = harm[mask]
    print(f"\n{label} direction analysis (n={mask.sum()}):")
    for pop_a, pop_b in comparisons:
        gse_diff = subset[f"gse_{pop_a}_mean_log2_tpm_p1"].values - subset[f"gse_{pop_b}_mean_log2_tpm_p1"].values
        m1_imp_diff = subset[f"m1_{pop_a}_imputed_log1p"].values - subset[f"m1_{pop_b}_imputed_log1p"].values
        sign_agree = np.mean(np.sign(gse_diff) == np.sign(m1_imp_diff))
        r, _ = stats.pearsonr(gse_diff, m1_imp_diff)
        print(f"  {pop_a} vs {pop_b}: sign agree={sign_agree:.4f}, r={r:.4f}")

# =====================================================================
# STEP 11: Reference Matching Matrix
# =====================================================================
print("\n" + "="*60)
print("STEP 11: Reference Matching Matrix")
print("="*60)

# Build 3x3 correlation matrix: GSE populations vs M1 imputed populations
# Plus 3x3 for GSE vs M1 raw
corr_matrix_imp = np.zeros((3, 3))
corr_matrix_raw = np.zeros((3, 3))

for i, pop_gse in enumerate(populations):
    for j, pop_m1 in enumerate(populations):
        gse_vals = harm[f"gse_{pop_gse}_mean_log2_tpm_p1"].values
        m1_imp = harm[f"m1_{pop_m1}_imputed_log1p"].values
        m1_raw = harm[f"m1_{pop_m1}_raw_log1p"].values

        corr_matrix_imp[i, j], _ = stats.pearsonr(gse_vals, m1_imp)
        corr_matrix_raw[i, j], _ = stats.pearsonr(gse_vals, m1_raw)

print("\nGSE vs M1 Imputed (Pearson r):")
header = "".join(f"{p:>15s}" for p in populations)
print(f"{'':>15s}{header}")
for i, pop in enumerate(populations):
    vals = "".join(f"{corr_matrix_imp[i, j]:>15.4f}" for j in range(3))
    print(f"{pop:>15s}{vals}")

print("\nGSE vs M1 Raw Nuclei (Pearson r):")
print(f"{'':>15s}{header}")
for i, pop in enumerate(populations):
    vals = "".join(f"{corr_matrix_raw[i, j]:>15.4f}" for j in range(3))
    print(f"{pop:>15s}{vals}")

# Diagonal dominance check
diag_imp = np.diag(corr_matrix_imp)
offdiag_imp = corr_matrix_imp[~np.eye(3, dtype=bool)]
print(f"\nImputed diagonal mean: {diag_imp.mean():.4f}, off-diagonal mean: {offdiag_imp.mean():.4f}")
print(f"Diagonal dominance: {'YES' if all(corr_matrix_imp[i, i] == corr_matrix_imp[i, :].max() for i in range(3)) else 'NO'}")

# Save correlation matrices
corr_imp_df = pd.DataFrame(corr_matrix_imp, index=populations, columns=populations)
corr_raw_df = pd.DataFrame(corr_matrix_raw, index=populations, columns=populations)
corr_imp_df.to_csv(os.path.join(OUT, "reference_matching_imputed.csv"))
corr_raw_df.to_csv(os.path.join(OUT, "reference_matching_raw.csv"))

# =====================================================================
# STEP 12: Donor Robustness (Leave-One-Out)
# =====================================================================
print("\n" + "="*60)
print("STEP 12: Donor Robustness")
print("="*60)

# Load per-donor pseudobulk from step 04
donor_pb = pd.read_csv(os.path.join(OUT, "human_m1_pseudobulk_by_donor.csv"))

# For GSE: recompute leave-one-donor-out profiles
gse_tpm = pd.read_csv(os.path.join(PROC, "gse219278_control_motorcortex_tpm.csv.gz"))
annot_cols = ["gene_ID", "gene_name", "gene_type"]
gse_donors = sorted(set(c.split("_")[1] for c in gse_tpm.columns if c not in annot_cols))

print(f"GSE donors: {gse_donors}")
print(f"M1 donors: {sorted(donor_pb['donor'].unique())}")

# M1 leave-one-donor-out: load imputed data (mmap) and recompute per-gene profiles
obs = pd.read_csv(os.path.join(HM1, "processed", "human_m1_nuclei_obs.csv"))
imputed = np.load(os.path.join(HM1, "inference", "human_m1_imputed_cell_log1p.npy"), mmap_mode="r")

GROUP_MAP = {
    "excitatory_neuron": "Neurons",
    "inhibitory_neuron": "Neurons",
    "oligodendrocyte": "OligoLineage",
    "opc": "OligoLineage",
    "astrocyte": "OtherGlia",
    "microglia_macrophage": "OtherGlia",
}
EXCLUDE = {"endothelial", "vlmc_pericyte_smc"}
obs["gse_group"] = obs["mapped_broad_class"].map(GROUP_MAP)
valid_mask = ~obs["mapped_broad_class"].isin(EXCLUDE)

m1_donors = sorted(obs.loc[valid_mask, "donor"].unique())

# Leave-one-out for M1: for each donor, compute profiles without that donor
# Then correlate with full GSE reference
loo_records = []

# Get full GSE reference gene order matching harmonized set
harm_indices = harm["model_gene_index"].values.astype(int)

for loo_donor in m1_donors:
    for grp in populations:
        # Indices for this group excluding this donor
        mask_keep = (valid_mask.values &
                     (obs["gse_group"].values == grp) &
                     (obs["donor"].values != loo_donor))
        mask_full = (valid_mask.values &
                     (obs["gse_group"].values == grp))

        if mask_keep.sum() == 0:
            continue

        imp_loo = np.array(imputed[mask_keep]).mean(axis=0)[harm_indices]
        imp_full = np.array(imputed[mask_full]).mean(axis=0)[harm_indices]
        gse_ref = harm[f"gse_{grp}_mean_log2_tpm_p1"].values

        r_loo, _ = stats.pearsonr(gse_ref, imp_loo)
        r_full, _ = stats.pearsonr(gse_ref, imp_full)

        loo_records.append({
            "excluded_donor": loo_donor,
            "group": grp,
            "n_excluded": int((valid_mask.values & (obs["gse_group"].values == grp) & (obs["donor"].values == loo_donor)).sum()),
            "n_remaining": int(mask_keep.sum()),
            "pearson_loo": r_loo,
            "pearson_full": r_full,
            "delta_r": r_loo - r_full,
        })

loo_df = pd.DataFrame(loo_records)
loo_df.to_csv(os.path.join(OUT, "donor_robustness_loo.csv"), index=False)
print(f"\nLeave-one-out donor robustness:")
print(loo_df.to_string(index=False))

# LOO for GSE donors
gse_loo_records = []
for loo_donor in gse_donors:
    for pop in populations:
        all_cols = [c for c in gse_tpm.columns if c not in annot_cols and c.endswith(f"_{pop}")]
        keep_cols = [c for c in all_cols if loo_donor not in c]

        if len(keep_cols) == 0:
            continue

        # Recompute mean log2(TPM+1) without this donor
        tpm_loo = gse_tpm[keep_cols].values
        log2_loo = np.log2(tpm_loo + 1).mean(axis=1)

        tpm_full = gse_tpm[all_cols].values
        log2_full = np.log2(tpm_full + 1).mean(axis=1)

        # Match to harmonized gene set
        gse_gene_to_idx = {name: i for i, name in enumerate(gse_tpm["gene_name"])}
        gse_indices = []
        valid_harm = []
        for hi, hsym in enumerate(harm["gse_gene_name"].values):
            if hsym in gse_gene_to_idx:
                gse_indices.append(gse_gene_to_idx[hsym])
                valid_harm.append(hi)

        log2_loo_sub = log2_loo[gse_indices]
        log2_full_sub = log2_full[gse_indices]

        r_stability, _ = stats.pearsonr(log2_loo_sub, log2_full_sub)

        gse_loo_records.append({
            "excluded_donor": loo_donor,
            "group": pop,
            "n_donors_remaining": len(keep_cols),
            "pearson_loo_vs_full": r_stability,
        })

gse_loo_df = pd.DataFrame(gse_loo_records)
gse_loo_df.to_csv(os.path.join(OUT, "gse_donor_robustness.csv"), index=False)
print(f"\nGSE donor robustness:")
print(gse_loo_df.to_string(index=False))

# =====================================================================
# Save consolidated results
# =====================================================================
all_results = {
    "profile_correlations": {k: v for k, v in results.items() if k.startswith("profile_corr")},
    "gene_set_deltas": {k: v for k, v in results.items() if k.startswith("geneset")},
    "direction_validation": direction_results,
    "reference_matching": {
        "imputed": corr_matrix_imp.tolist(),
        "raw": corr_matrix_raw.tolist(),
        "populations": populations,
        "diagonal_dominance": all(corr_matrix_imp[i, i] == corr_matrix_imp[i, :].max() for i in range(3)),
    },
    "donor_robustness": {
        "m1_loo_mean_delta_r": float(loo_df["delta_r"].mean()),
        "gse_loo_mean_stability": float(gse_loo_df["pearson_loo_vs_full"].mean()),
    },
}

with open(os.path.join(OUT, "validation_results_consolidated.json"), "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\nAll results saved to {OUT}")
print("\nDONE.")
