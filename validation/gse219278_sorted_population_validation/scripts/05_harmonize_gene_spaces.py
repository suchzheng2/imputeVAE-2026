#!/usr/bin/env python3
"""Step 7-8: Build human-first gene sets and harmonize gene spaces.

1. Build SynGO gene set from human symbols (already have mapping from syngo_gene_metadata.csv)
2. Build JASPAR human TF gene set
3. Harmonize: find genes present in BOTH GSE219278 AND human M1 model space
4. Output: harmonized gene-level table with all profiles and gene set memberships
"""
import pandas as pd
import numpy as np
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")
OUT = os.path.join(BASE, "outputs", "tables")
SYNGO_DIR = os.path.join(os.path.dirname(BASE), "syngo", "syngo_brain_validation")
TF_DIR = os.path.join(os.path.dirname(BASE), "tf")
os.makedirs(OUT, exist_ok=True)

# ---------- Load human M1 pseudobulk (model gene space) ----------
print("Loading human M1 pseudobulk profiles...")
m1_profiles = pd.read_csv(os.path.join(PROC, "human_m1_pseudobulk_per_gene.csv.gz"))
print(f"  Model gene space: {len(m1_profiles)} genes")
# Keep only genes with human orthologs
m1_with_human = m1_profiles[m1_profiles["human_gene_symbol"].notna()].copy()
print(f"  With human orthologs: {len(m1_with_human)}")

# ---------- Load GSE219278 reference profiles ----------
print("Loading GSE219278 reference profiles...")
gse_ref = pd.read_csv(os.path.join(PROC, "gse219278_reference_profiles.csv.gz"))
print(f"  GSE genes: {len(gse_ref)}")

# Build lookup by gene_name (human symbols)
gse_lookup = {}
for _, row in gse_ref.iterrows():
    gse_lookup[row["gene_name"]] = row
# Also try uppercase for case-insensitive matching
gse_lookup_lower = {k.lower(): v for k, v in gse_lookup.items() if isinstance(k, str)}

# ---------- Load SynGO gene set (human symbols) ----------
print("\nLoading SynGO gene metadata...")
syngo_meta = pd.read_csv(os.path.join(SYNGO_DIR, "syngo_gene_metadata.csv"))
syngo_human_symbols = set(syngo_meta["human_symbol"].dropna().unique())
print(f"  SynGO human symbols: {len(syngo_human_symbols)}")

# ---------- Load JASPAR TF gene set (human) ----------
print("Loading JASPAR CORE metadata...")
jaspar_meta = pd.read_csv(os.path.join(TF_DIR, "external", "ultimate_metadata_table_CORE_2026.tsv"),
                          sep="\t")
# Filter to human (tax_id 9606) — column may be string
jaspar_meta["tax_id"] = pd.to_numeric(jaspar_meta["tax_id"], errors="coerce")
jaspar_human = jaspar_meta[jaspar_meta["tax_id"] == 9606]
# Extract unique TF symbols, handling :: complexes
tf_human_symbols = set()
for name in jaspar_human["name"].unique():
    for part in str(name).split("::"):
        tf_human_symbols.add(part.strip())
print(f"  JASPAR human TF symbols (from CORE): {len(tf_human_symbols)}")

# ---------- Harmonize ----------
print("\nHarmonizing gene spaces...")
harmonized_records = []
n_matched = 0
n_syngo = 0
n_tf = 0

for _, row in m1_with_human.iterrows():
    hsym = row["human_gene_symbol"]
    if not isinstance(hsym, str):
        continue

    # Find in GSE reference
    gse_row = None
    if hsym in gse_lookup:
        gse_row = gse_lookup[hsym]
    elif hsym.lower() in gse_lookup_lower:
        gse_row = gse_lookup_lower[hsym.lower()]

    if gse_row is None:
        continue

    # Check if expressed in GSE (mean TPM > 0 in at least one population)
    gse_expressed = any(gse_row[f"{pop}_mean_tpm"] > 0
                       for pop in ["Neurons", "OligoLineage", "OtherGlia"])
    if not gse_expressed:
        continue

    n_matched += 1
    is_syngo = hsym in syngo_human_symbols or hsym.upper() in {s.upper() for s in syngo_human_symbols}
    is_tf = hsym in tf_human_symbols or hsym.upper() in {s.upper() for s in tf_human_symbols}
    if is_syngo:
        n_syngo += 1
    if is_tf:
        n_tf += 1

    rec = {
        "human_gene_symbol": hsym,
        "mouse_model_gene": row["mouse_model_gene"],
        "model_gene_index": row["model_gene_index"],
        "gse_gene_name": gse_row["gene_name"],
        "is_syngo": is_syngo,
        "is_jaspar_tf": is_tf,
    }

    # Add GSE reference values
    for pop in ["Neurons", "OligoLineage", "OtherGlia"]:
        rec[f"gse_{pop}_mean_tpm"] = gse_row[f"{pop}_mean_tpm"]
        rec[f"gse_{pop}_mean_log2_tpm_p1"] = gse_row[f"{pop}_mean_log2_tpm_p1"]

    # Add human M1 model values
    for pop in ["Neurons", "OligoLineage", "OtherGlia"]:
        rec[f"m1_{pop}_raw_log1p"] = row[f"{pop}_raw_log1p"]
        rec[f"m1_{pop}_imputed_log1p"] = row[f"{pop}_imputed_log1p"]
        rec[f"m1_{pop}_predicted_delta"] = row[f"{pop}_predicted_delta"]

    harmonized_records.append(rec)

harm_df = pd.DataFrame(harmonized_records)
print(f"\nHarmonized genes: {n_matched}")
print(f"  SynGO genes in harmonized set: {n_syngo}")
print(f"  JASPAR TF genes in harmonized set: {n_tf}")

harm_df.to_csv(os.path.join(PROC, "harmonized_gene_profiles.csv.gz"), index=False)
print(f"\nSaved: {os.path.join(PROC, 'harmonized_gene_profiles.csv.gz')}")

# ---------- Gene set summary ----------
summary = {
    "total_model_genes": len(m1_profiles),
    "model_genes_with_human_ortholog": len(m1_with_human),
    "gse_total_genes": len(gse_ref),
    "harmonized_genes": n_matched,
    "harmonized_syngo": n_syngo,
    "harmonized_jaspar_tf": n_tf,
}
summary_df = pd.DataFrame([summary])
summary_df.to_csv(os.path.join(OUT, "gene_harmonization_summary.csv"), index=False)
print(f"\nGene harmonization summary:")
for k, v in summary.items():
    print(f"  {k}: {v}")
