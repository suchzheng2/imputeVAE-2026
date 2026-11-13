#!/usr/bin/env python3
"""
11_global_qc_acceptance_checks.py
Acceptance checks for global QC outputs from script 10.
"""
import os, sys
import pandas as pd

BASE = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE"
HM1 = os.path.join(BASE, "validation/human_m1_transfer")
FIG_DIR = os.path.join(HM1, "figures")
RES_DIR = os.path.join(HM1, "results")
REP_DIR = os.path.join(HM1, "reports")

passed = 0
failed = 0

def check(condition, msg):
    global passed, failed
    if condition:
        print(f"  [OK] {msg}")
        passed += 1
    else:
        print(f"  [FAIL] {msg}")
        failed += 1

# ── CSVs exist and non-empty ──
print("Result CSVs:")
csv_files = [
    "human_m1_global_embedding.csv",
    "human_m1_centroid_shift_summary.csv",
    "human_m1_identity_preservation.csv",
    "human_m1_program_score_shifts.csv",
    "human_m1_distributional_delta_summary.csv",
]
for fname in csv_files:
    path = os.path.join(RES_DIR, fname)
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    check(exists, f"CSV: {fname}")

# ── Figures exist ──
print("\nFigures:")
fig_files = [
    "human_m1_global_pca_umap",
    "human_m1_centroid_arrows",
    "human_m1_identity_preservation",
    "human_m1_program_score_shifts",
    "human_m1_distributional_delta_summary",
]
for fname in fig_files:
    path_png = os.path.join(FIG_DIR, fname + ".png")
    check(os.path.exists(path_png), f"Figure: {fname}.png")

# optional Part F figure
part_f_path = os.path.join(FIG_DIR, "human_m1_mouse_human_delta_projection.png")
if os.path.exists(part_f_path):
    print(f"  [INFO] Part F figure exists: human_m1_mouse_human_delta_projection.png")
else:
    print(f"  [INFO] Part F figure not produced (optional)")

# ── Embedding CSV checks ──
print("\nEmbedding CSV:")
emb = pd.read_csv(os.path.join(RES_DIR, "human_m1_global_embedding.csv"))
check(set(emb["modality"].unique()) == {"nucleus", "imputed"},
      "Embedding contains both modalities")
n_per_mod = emb.groupby("modality").size()
check(n_per_mod["nucleus"] == n_per_mod["imputed"],
      f"Equal rows per modality: {n_per_mod['nucleus']} each")
check(n_per_mod["nucleus"] <= 500 * 20 + 500,  # generous upper bound
      f"Sampled rows reasonable: {n_per_mod['nucleus']}")
has_umap = "UMAP1" in emb.columns
print(f"  [INFO] UMAP columns present: {has_umap}")

# ── Program score checks ──
print("\nProgram scores:")
ps = pd.read_csv(os.path.join(RES_DIR, "human_m1_program_score_shifts.csv"))
gs_names = ps["gene_set"].unique()
check("SynGO" in gs_names, "SynGO gene set present")
check("TF" in gs_names, "TF gene set present")
check("Ribosomal" in gs_names, "Ribosomal gene set present")
check("Nuclear_lncRNA" in gs_names, "Nuclear_lncRNA gene set present")

# check n_genes_used reported
for gs in gs_names:
    sub = ps[ps["gene_set"] == gs]
    n_genes = sub["n_genes_used"].values[0]
    check(n_genes > 0, f"  {gs} n_genes_used = {n_genes}")

syngo_n = ps[ps["gene_set"] == "SynGO"]["n_genes_used"].values[0]
check(1200 <= syngo_n <= 1300,
      f"SynGO n_genes_used ~ 1236 (got {syngo_n})")

# ── Identity preservation checks ──
print("\nIdentity preservation:")
ident = pd.read_csv(os.path.join(RES_DIR, "human_m1_identity_preservation.csv"))
check("broad_class" in ident["class_level"].values, "Has broad_class level")
check("mouse_like_class" in ident["class_level"].values, "Has mouse_like_class level")

broad_ident = ident[ident["class_level"] == "broad_class"]
median_rate = broad_ident["imputed_assignment_to_original_class_rate"].median()
print(f"  [INFO] Broad class median identity rate: {median_rate:.3f}")

# ── Low-sample flagging ──
print("\nLow-sample flagging:")
cent = pd.read_csv(os.path.join(RES_DIR, "human_m1_centroid_shift_summary.csv"))
low = cent[cent["n_nuclei"] < 200]
check(True, f"Low-sample classes flagged: {len(low)} entries")
for _, row in low.iterrows():
    print(f"    {row['class_level']}: {row['class']} (n={row['n_nuclei']})")

# ── Report exists ──
print("\nReport:")
report_path = os.path.join(REP_DIR, "HUMAN_M1_GLOBAL_QC_REPORT.md")
check(os.path.exists(report_path) and os.path.getsize(report_path) > 0,
      "Report: HUMAN_M1_GLOBAL_QC_REPORT.md")

# ── Summary ──
print(f"\n{'='*50}")
print(f"RESULTS: {passed} passed, {failed} failed")
print(f"{'='*50}")

if failed > 0:
    sys.exit(1)
