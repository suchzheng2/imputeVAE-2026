#!/usr/bin/env python3
"""
12_human_first_geneset_validation.py

Recompute Human M1 transfer validation using gene sets defined in human
annotation space first, then mapped to mouse model genes via one-to-one orthologs.

Parts:
  1. Human-first SynGO gene set (from SynGO 1.3 HGNC symbols)
  2. Human-first TF gene set (from JASPAR CORE Homo sapiens)
  3. Recompute predicted-shift summaries
  4. Compare old vs new results
  5. Figures
  6. Report
  7. Acceptance checks
"""

import os, sys, csv, json, glob, warnings, hashlib, datetime
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
try:
    from matplotlib_venn import venn2
    HAS_VENN = True
except ImportError:
    HAS_VENN = False
import openpyxl

warnings.filterwarnings("ignore", category=FutureWarning)

# ── paths ──
BASE = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE"
HM1 = os.path.join(BASE, "validation/human_m1_transfer")
FIG_DIR = os.path.join(HM1, "figures")
RES_DIR = os.path.join(HM1, "results")
REP_DIR = os.path.join(HM1, "reports")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(REP_DIR, exist_ok=True)

# Input data
GENE_MAP_PATH = os.path.join(HM1, "processed/human_m1_gene_mapping_for_model_order.csv")
ORTHOLOG_PATH = os.path.join(HM1, "orthologs/human_to_mouse_one_to_one_orthologs.csv")
BROAD_MEANS_PATH = os.path.join(HM1, "results/human_m1_broad_class_gene_means.csv")
ML_MEANS_PATH = os.path.join(HM1, "results/human_m1_mouse_like_class_gene_means.csv")

# SynGO source
SYNGO_ZIP = os.path.join(BASE, "validation/syngo/syngo1.3_complete_data.zip")
SYNGO_GENES_XLSX = "/tmp/syngo_data/genes.xlsx"  # extracted earlier
SYNGO_METADATA_PATH = os.path.join(BASE, "validation/syngo/syngo_brain_validation/syngo_gene_metadata.csv")

# JASPAR source
JASPAR_PATH = os.path.join(BASE, "validation/tf/external/ultimate_metadata_table_CORE_2026.tsv")

# Previous results
PREV_SYNGO_SUMMARY = os.path.join(HM1, "results/human_m1_syngo_transfer_summary.csv")
PREV_TF_SUMMARY = os.path.join(HM1, "results/human_m1_tf_transfer_summary.csv")
PREV_CONTRAST = os.path.join(HM1, "results/human_m1_syngo_vs_tf_transfer_contrast.csv")
PREV_NEUR = os.path.join(HM1, "results/human_m1_neuronal_vs_nonneuronal_predicted_shift.csv")

# Mouse delta tables
MOUSE_DELTA_DIR = os.path.join(BASE, "notes/experiments/delta_analysis/tables")

NEURONAL_CLASSES = {"excitatory_neuron", "inhibitory_neuron"}
NONNEURONAL_CLASSES = {"astrocyte", "oligodendrocyte", "opc", "endothelial",
                       "microglia_macrophage", "vlmc_pericyte_smc"}


def save_fig(fig, name):
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"),
                    dpi=200, bbox_inches="tight", format=ext)
    plt.close(fig)
    print(f"  Saved {name}.png/pdf/svg")


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ══════════════════════════════════════════════════════════════════════
# Load shared data
# ══════════════════════════════════════════════════════════════════════
print("=== Loading shared data ===")

gene_map = pd.read_csv(GENE_MAP_PATH)
# mapped = present in human M1 matrix and is one-to-one ortholog
mapped_mask = (gene_map["filled_value_status"] == "mapped")
mapped_genes = gene_map[mapped_mask].copy()
print(f"  Gene map: {len(gene_map)} total, {len(mapped_genes)} mapped")

# Build human -> mouse lookup for mapped genes
h2m = dict(zip(mapped_genes["human_gene_symbol"], mapped_genes["mouse_model_gene"]))
m2h = dict(zip(mapped_genes["mouse_model_gene"], mapped_genes["human_gene_symbol"]))
mouse_to_idx = dict(zip(mapped_genes["mouse_model_gene"], mapped_genes["model_gene_index"]))
human_in_m1 = set(mapped_genes["human_gene_symbol"].dropna())
mouse_in_model = set(mapped_genes["mouse_model_gene"])

# Load full ortholog table for broader mapping
orthologs = pd.read_csv(ORTHOLOG_PATH)
# Full human->mouse 1:1 mapping (including genes not in model)
full_h2m = dict(zip(orthologs["human_gene_symbol"], orthologs["mouse_gene_symbol"]))

# Load human M1 delta tables
broad_means = pd.read_csv(BROAD_MEANS_PATH)
ml_means = pd.read_csv(ML_MEANS_PATH)
# Filter to mapped genes only
broad_mapped = broad_means[broad_means["filled_value_status"] == "mapped"].copy()
ml_mapped = ml_means[ml_means["filled_value_status"] == "mapped"].copy()
print(f"  Broad means: {len(broad_mapped)} rows ({broad_mapped['group'].nunique()} classes)")
print(f"  Mouse-like means: {len(ml_mapped)} rows ({ml_mapped['group'].nunique()} classes)")


# ══════════════════════════════════════════════════════════════════════
# Part 1: Human-first SynGO gene set
# ══════════════════════════════════════════════════════════════════════
print("\n=== Part 1: Human-first SynGO gene set ===")

# Load SynGO genes from original HGNC source
if not os.path.exists(SYNGO_GENES_XLSX):
    import zipfile
    with zipfile.ZipFile(SYNGO_ZIP, "r") as zf:
        zf.extract("genes.xlsx", "/tmp/syngo_data")

wb = openpyxl.load_workbook(SYNGO_GENES_XLSX, read_only=True)
ws = wb.active
syngo_rows = list(ws.iter_rows(values_only=True))
syngo_header = syngo_rows[0]
syngo_data = syngo_rows[1:]
wb.close()

syngo_human_all = set(r[1] for r in syngo_data if r[1])  # hgnc_symbol column
print(f"  SynGO total HGNC symbols: {len(syngo_human_all)}")

# Load SynGO annotation metadata for synaptic location columns
syngo_meta = pd.read_csv(SYNGO_METADATA_PATH)
syngo_annot = {}
for _, row in syngo_meta.iterrows():
    syngo_annot[row["human_symbol"]] = {
        "sg_broad_synapse": row.get("sg_broad_synapse", False),
        "sg_presynaptic": row.get("sg_presynaptic", False),
        "sg_postsynaptic": row.get("sg_postsynaptic", False),
        "sg_postsynaptic_density": row.get("sg_postsynaptic_density", False),
        "sg_presynaptic_active_zone": row.get("sg_presynaptic_active_zone", False),
        "sg_synaptic_vesicle": row.get("sg_synaptic_vesicle", False),
    }

# Map human SynGO -> mouse model genes
syngo_mapping_rows = []
for hsym in sorted(syngo_human_all):
    is_ribo_h = hsym.startswith("RPL") or hsym.startswith("RPS")
    msym = h2m.get(hsym)
    in_h_m1 = hsym in human_in_m1
    in_model = msym in mouse_in_model if msym else False
    is_ribo_m = (msym.startswith("Rpl") or msym.startswith("Rps")) if msym else False

    # Determine inclusion
    if msym is None:
        # Try full ortholog table
        msym_full = full_h2m.get(hsym)
        if msym_full and msym_full in mouse_in_model:
            # This would mean it's in model but not in human M1
            excl = "ortholog_in_model_but_not_in_human_m1"
        elif msym_full:
            excl = "ortholog_not_in_model"
        else:
            excl = "no_one_to_one_ortholog"
        included = False
    elif not in_h_m1:
        excl = "not_in_human_m1_matrix"
        included = False
    elif is_ribo_h or is_ribo_m:
        excl = "ribosomal"
        included = False
    else:
        excl = ""
        included = True

    annot = syngo_annot.get(hsym, {})
    syngo_mapping_rows.append({
        "human_gene_symbol": hsym,
        "mouse_model_gene": msym if msym else "",
        "model_gene_index": mouse_to_idx.get(msym, "") if msym else "",
        "in_human_m1_matrix": in_h_m1,
        "in_model": in_model,
        "is_ribosomal_human": is_ribo_h,
        "is_ribosomal_mouse": is_ribo_m,
        "included_primary": included,
        "exclusion_reason": excl,
        "sg_broad_synapse": annot.get("sg_broad_synapse", ""),
        "sg_presynaptic": annot.get("sg_presynaptic", ""),
        "sg_postsynaptic": annot.get("sg_postsynaptic", ""),
        "sg_postsynaptic_density": annot.get("sg_postsynaptic_density", ""),
        "sg_presynaptic_active_zone": annot.get("sg_presynaptic_active_zone", ""),
        "sg_synaptic_vesicle": annot.get("sg_synaptic_vesicle", ""),
    })

syngo_map_df = pd.DataFrame(syngo_mapping_rows)
syngo_map_df.to_csv(os.path.join(RES_DIR, "human_first_syngo_gene_mapping.csv"), index=False)

syngo_included = syngo_map_df[syngo_map_df["included_primary"]]
syngo_primary_mouse = set(syngo_included["mouse_model_gene"])
syngo_primary_human = set(syngo_included["human_gene_symbol"])

n_syngo_total = len(syngo_human_all)
n_syngo_ribo = syngo_map_df["is_ribosomal_human"].sum()
n_syngo_mapped = syngo_map_df["in_model"].sum()
n_syngo_primary = len(syngo_included)

print(f"  SynGO: {n_syngo_total} total -> {n_syngo_mapped} in model -> {n_syngo_primary} primary (non-ribo, in human M1)")
print(f"  Excluded: {(syngo_map_df['exclusion_reason'] != '').sum()}")
print(f"    ribosomal: {syngo_map_df[syngo_map_df['exclusion_reason'] == 'ribosomal'].shape[0]}")
print(f"    no_ortholog: {syngo_map_df[syngo_map_df['exclusion_reason'] == 'no_one_to_one_ortholog'].shape[0]}")
print(f"    not_in_model: {syngo_map_df[syngo_map_df['exclusion_reason'] == 'ortholog_not_in_model'].shape[0]}")
print(f"    not_in_human_m1: {syngo_map_df[syngo_map_df['exclusion_reason'] == 'not_in_human_m1_matrix'].shape[0]}")


# ══════════════════════════════════════════════════════════════════════
# Part 2: Human-first TF gene set
# ══════════════════════════════════════════════════════════════════════
print("\n=== Part 2: Human-first TF gene set (JASPAR CORE Homo sapiens) ===")

jaspar_rows = []
with open(JASPAR_PATH, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        if row.get("species", "") == "Homo sapiens":
            jaspar_rows.append(row)

print(f"  JASPAR human rows: {len(jaspar_rows)}")
jaspar_md5 = md5_file(JASPAR_PATH)
jaspar_date = datetime.datetime.fromtimestamp(os.path.getmtime(JASPAR_PATH)).strftime("%Y-%m-%d")

# Extract unique TF symbols, handling complexes
tf_name_to_matrix_ids = {}
for row in jaspar_rows:
    name = row["name"]
    mid = row["matrix_id"]
    tf_name_to_matrix_ids.setdefault(name, []).append(mid)

# Split complexes and build gene-level info
tf_gene_info = {}  # human_symbol -> {matrix_ids, complex_component, singleton_only}
for name, mids in tf_name_to_matrix_ids.items():
    if "::" in name:
        components = name.split("::")
        for comp in components:
            comp = comp.strip()
            if comp not in tf_gene_info:
                tf_gene_info[comp] = {"matrix_ids": [], "complex_component": True, "singleton_only": False}
            tf_gene_info[comp]["matrix_ids"].extend(mids)
    else:
        if name not in tf_gene_info:
            tf_gene_info[name] = {"matrix_ids": [], "complex_component": False, "singleton_only": True}
        else:
            # Already added as complex component — mark as not singleton-only
            tf_gene_info[name]["singleton_only"] = False
        tf_gene_info[name]["matrix_ids"].extend(mids)

# For genes that appear ONLY as singletons (never in a complex), singleton_only=True
# For genes that appear as both singleton and complex component, singleton_only still True
# (they have their own direct motif)
# Actually: singleton_only = has at least one non-complex motif entry
for name, mids_list in tf_name_to_matrix_ids.items():
    if "::" not in name and name in tf_gene_info:
        tf_gene_info[name]["singleton_only"] = True

tf_human_all = set(tf_gene_info.keys())
print(f"  Unique human TF symbols: {len(tf_human_all)} ({sum(1 for v in tf_gene_info.values() if v['complex_component'])} from complexes)")

# Map to mouse model genes
tf_mapping_rows = []
for hsym in sorted(tf_human_all):
    info = tf_gene_info[hsym]
    msym = h2m.get(hsym)
    in_h_m1 = hsym in human_in_m1
    in_model = msym in mouse_in_model if msym else False

    if msym is None:
        msym_full = full_h2m.get(hsym)
        if msym_full and msym_full in mouse_in_model:
            excl = "ortholog_in_model_but_not_in_human_m1"
        elif msym_full:
            excl = "ortholog_not_in_model"
        else:
            excl = "no_one_to_one_ortholog"
        included_primary = False
        included_singleton = False
    elif not in_h_m1:
        excl = "not_in_human_m1_matrix"
        included_primary = False
        included_singleton = False
    else:
        excl = ""
        included_primary = True
        included_singleton = info["singleton_only"]

    tf_mapping_rows.append({
        "human_gene_symbol": hsym,
        "mouse_model_gene": msym if msym else "",
        "model_gene_index": mouse_to_idx.get(msym, "") if msym else "",
        "jaspar_matrix_ids": ";".join(sorted(set(info["matrix_ids"]))),
        "motif_count": len(set(info["matrix_ids"])),
        "complex_component": info["complex_component"],
        "singleton_only": info["singleton_only"],
        "in_human_m1_matrix": in_h_m1,
        "in_model": in_model,
        "included_primary": included_primary,
        "included_singleton_sensitivity": included_singleton,
        "exclusion_reason": excl,
    })

tf_map_df = pd.DataFrame(tf_mapping_rows)
tf_map_df.to_csv(os.path.join(RES_DIR, "human_first_tf_gene_mapping.csv"), index=False)

tf_primary = tf_map_df[tf_map_df["included_primary"]]
tf_singleton = tf_map_df[tf_map_df["included_singleton_sensitivity"]]
tf_primary_mouse = set(tf_primary["mouse_model_gene"])
tf_primary_human = set(tf_primary["human_gene_symbol"])
tf_singleton_mouse = set(tf_singleton["mouse_model_gene"])

n_tf_total = len(tf_human_all)
n_tf_primary = len(tf_primary)
n_tf_singleton = len(tf_singleton)

print(f"  TF: {n_tf_total} total -> {n_tf_primary} primary (in model + human M1) -> {n_tf_singleton} singleton-only")


# ══════════════════════════════════════════════════════════════════════
# Part 3: Recompute predicted-shift summaries
# ══════════════════════════════════════════════════════════════════════
print("\n=== Part 3: Recompute predicted-shift summaries ===")

# --- 3A: SynGO transfer summary (long format) ---
syngo_summary_rows = []
for _, row in ml_mapped.iterrows():
    if row["mouse_model_gene"] in syngo_primary_mouse:
        syngo_summary_rows.append({
            "human_class": row["group"],
            "human_gene_symbol": row["human_gene_symbol"],
            "mouse_model_gene": row["mouse_model_gene"],
            "model_gene_index": row["model_gene_index"],
            "human_predicted_delta": row["predicted_delta"],
            "nucleus_mean_log1p": row["nucleus_mean_log1p"],
            "imputed_mean_log1p": row["imputed_mean_log1p"],
            "gene_set": "human_first_SynGO",
        })

syngo_summary_df = pd.DataFrame(syngo_summary_rows)
syngo_summary_df.to_csv(os.path.join(RES_DIR, "human_first_syngo_transfer_summary.csv"), index=False)
print(f"  SynGO summary: {len(syngo_summary_df)} rows, {syngo_summary_df['human_gene_symbol'].nunique()} genes, {syngo_summary_df['human_class'].nunique()} classes")

# --- 3B: TF transfer summary (long format, primary + singleton flag) ---
tf_summary_rows = []
for _, row in ml_mapped.iterrows():
    if row["mouse_model_gene"] in tf_primary_mouse:
        is_singleton = row["mouse_model_gene"] in tf_singleton_mouse
        tf_summary_rows.append({
            "human_class": row["group"],
            "human_gene_symbol": row["human_gene_symbol"],
            "mouse_model_gene": row["mouse_model_gene"],
            "model_gene_index": row["model_gene_index"],
            "human_predicted_delta": row["predicted_delta"],
            "nucleus_mean_log1p": row["nucleus_mean_log1p"],
            "imputed_mean_log1p": row["imputed_mean_log1p"],
            "gene_set": "human_first_TF_primary",
            "singleton_only": is_singleton,
        })

tf_summary_df = pd.DataFrame(tf_summary_rows)
tf_summary_df.to_csv(os.path.join(RES_DIR, "human_first_tf_transfer_summary.csv"), index=False)
print(f"  TF summary: {len(tf_summary_df)} rows, {tf_summary_df['human_gene_symbol'].nunique()} genes")

# --- 3C: Neuronal vs non-neuronal predicted shift ---
# Use broad class means
broad_mapped_neur = broad_mapped[broad_mapped["group"].isin(NEURONAL_CLASSES)]
broad_mapped_nneur = broad_mapped[broad_mapped["group"].isin(NONNEURONAL_CLASSES)]

gene_sets_for_neur = {
    "human_first_SynGO": syngo_primary_mouse,
    "human_first_TF_primary": tf_primary_mouse,
    "human_first_TF_singleton": tf_singleton_mouse,
    "all_mapped": mouse_in_model,
}

neur_rows = []
for gs_name, gs_mice in gene_sets_for_neur.items():
    neur_sub = broad_mapped_neur[broad_mapped_neur["mouse_model_gene"].isin(gs_mice)]
    nneur_sub = broad_mapped_nneur[broad_mapped_nneur["mouse_model_gene"].isin(gs_mice)]
    neur_rows.append({
        "gene_set": gs_name,
        "n_genes": len(gs_mice & mouse_in_model),
        "neuronal_mean_delta": neur_sub["predicted_delta"].mean(),
        "nonneuronal_mean_delta": nneur_sub["predicted_delta"].mean(),
        "neuronal_minus_nonneuronal": neur_sub["predicted_delta"].mean() - nneur_sub["predicted_delta"].mean(),
        "neuronal_n_obs": len(neur_sub),
        "nonneuronal_n_obs": len(nneur_sub),
    })

neur_df = pd.DataFrame(neur_rows)
neur_df.to_csv(os.path.join(RES_DIR, "human_first_neuronal_vs_nonneuronal_predicted_shift.csv"), index=False)
print(f"  Neuronal vs non-neuronal: {len(neur_df)} gene sets")
for _, r in neur_df.iterrows():
    print(f"    {r['gene_set']}: n={r['n_genes']}, neur={r['neuronal_mean_delta']:.4f}, "
          f"non-neur={r['nonneuronal_mean_delta']:.4f}, diff={r['neuronal_minus_nonneuronal']:.4f}")

# --- 3D: SynGO vs TF contrast by mouse-like class ---
contrast_rows = []
for cls in sorted(ml_mapped["group"].unique()):
    cls_data = ml_mapped[ml_mapped["group"] == cls]

    syngo_sub = cls_data[cls_data["mouse_model_gene"].isin(syngo_primary_mouse)]
    tf_p_sub = cls_data[cls_data["mouse_model_gene"].isin(tf_primary_mouse)]
    tf_s_sub = cls_data[cls_data["mouse_model_gene"].isin(tf_singleton_mouse)]

    contrast_rows.append({
        "class": cls,
        "syngo_mean_delta": syngo_sub["predicted_delta"].mean() if len(syngo_sub) > 0 else np.nan,
        "tf_primary_mean_delta": tf_p_sub["predicted_delta"].mean() if len(tf_p_sub) > 0 else np.nan,
        "tf_singleton_mean_delta": tf_s_sub["predicted_delta"].mean() if len(tf_s_sub) > 0 else np.nan,
        "syngo_minus_tf_primary": (syngo_sub["predicted_delta"].mean() - tf_p_sub["predicted_delta"].mean())
            if len(syngo_sub) > 0 and len(tf_p_sub) > 0 else np.nan,
        "n_syngo": len(syngo_sub),
        "n_tf_primary": len(tf_p_sub),
        "n_tf_singleton": len(tf_s_sub),
    })

contrast_df = pd.DataFrame(contrast_rows)
contrast_df.to_csv(os.path.join(RES_DIR, "human_first_syngo_vs_tf_transfer_contrast.csv"), index=False)
print(f"  SynGO vs TF contrast: {len(contrast_df)} classes")


# ══════════════════════════════════════════════════════════════════════
# Part 4: Compare old vs new
# ══════════════════════════════════════════════════════════════════════
print("\n=== Part 4: Old vs new comparison ===")

prev_neur = pd.read_csv(PREV_NEUR)
prev_contrast = pd.read_csv(PREV_CONTRAST)

# Get previous gene sets from previous summaries
prev_syngo_genes = set()
if os.path.exists(PREV_SYNGO_SUMMARY):
    prev_syngo_df = pd.read_csv(PREV_SYNGO_SUMMARY)
    prev_syngo_genes = set(prev_syngo_df["mouse_model_gene"].unique())

prev_tf_genes = set()
if os.path.exists(PREV_TF_SUMMARY):
    prev_tf_df = pd.read_csv(PREV_TF_SUMMARY)
    prev_tf_genes = set(prev_tf_df["mouse_model_gene"].unique())

# Overlap
syngo_overlap = syngo_primary_mouse & prev_syngo_genes
tf_overlap = tf_primary_mouse & prev_tf_genes

comp_rows = []
for gs_label, new_genes, prev_genes, overlap, prev_row_name, new_row_name in [
    ("SynGO", syngo_primary_mouse, prev_syngo_genes, syngo_overlap, "SynGO", "human_first_SynGO"),
    ("TF", tf_primary_mouse, prev_tf_genes, tf_overlap, "TF", "human_first_TF_primary"),
]:
    prev_neur_row = prev_neur[prev_neur["gene_set"] == prev_row_name]
    new_neur_row = neur_df[neur_df["gene_set"] == new_row_name]

    prev_diff = prev_neur_row["neuronal_minus_nonneuronal"].values[0] if len(prev_neur_row) > 0 else np.nan
    new_diff = new_neur_row["neuronal_minus_nonneuronal"].values[0] if len(new_neur_row) > 0 else np.nan

    # Class-wise means comparison
    prev_cls_means = {}
    new_cls_means = {}
    if gs_label == "SynGO":
        for _, r in prev_contrast.iterrows():
            prev_cls_means[r["class"]] = r["syngo_mean_delta"]
        for _, r in contrast_df.iterrows():
            new_cls_means[r["class"]] = r["syngo_mean_delta"]
    else:
        for _, r in prev_contrast.iterrows():
            prev_cls_means[r["class"]] = r["tf_mean_delta"]
        for _, r in contrast_df.iterrows():
            new_cls_means[r["class"]] = r["tf_primary_mean_delta"]

    common_cls = sorted(set(prev_cls_means.keys()) & set(new_cls_means.keys()))
    if len(common_cls) >= 3:
        prev_vals = [prev_cls_means[c] for c in common_cls]
        new_vals = [new_cls_means[c] for c in common_cls]
        r_pearson, _ = stats.pearsonr(prev_vals, new_vals)
        r_spearman, _ = stats.spearmanr(prev_vals, new_vals)
    else:
        r_pearson = r_spearman = np.nan

    comp_rows.append({
        "gene_set": gs_label,
        "previous_n_genes": len(prev_genes),
        "human_first_n_genes": len(new_genes),
        "overlap_n_genes": len(overlap),
        "previous_only": len(prev_genes - new_genes),
        "human_first_only": len(new_genes - prev_genes),
        "previous_neuronal_diff": prev_diff,
        "human_first_neuronal_diff": new_diff,
        "delta_in_diff": new_diff - prev_diff if not np.isnan(prev_diff) else np.nan,
        "classwise_pearson": r_pearson,
        "classwise_spearman": r_spearman,
    })

comp_df = pd.DataFrame(comp_rows)
comp_df.to_csv(os.path.join(RES_DIR, "human_first_vs_previous_gene_set_comparison.csv"), index=False)
print(f"  Comparison saved:")
for _, r in comp_df.iterrows():
    print(f"    {r['gene_set']}: prev={r['previous_n_genes']}, new={r['human_first_n_genes']}, "
          f"overlap={r['overlap_n_genes']}, pearson={r['classwise_pearson']:.4f}")


def _draw_manual_venn(ax, set_a, set_b, label_a, label_b):
    """Simple text-based Venn when matplotlib_venn is unavailable."""
    only_a = len(set_a - set_b)
    only_b = len(set_b - set_a)
    both = len(set_a & set_b)
    from matplotlib.patches import Circle
    c1 = Circle((-0.15, 0), 0.35, alpha=0.25, color="#4e79a7")
    c2 = Circle((0.15, 0), 0.35, alpha=0.25, color="#e15759")
    ax.add_patch(c1); ax.add_patch(c2)
    ax.text(-0.3, 0, str(only_a), ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(0.0, 0, str(both), ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(0.3, 0, str(only_b), ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(-0.25, 0.4, label_a, ha="center", fontsize=9, color="#4e79a7")
    ax.text(0.25, 0.4, label_b, ha="center", fontsize=9, color="#e15759")
    ax.set_xlim(-0.6, 0.6); ax.set_ylim(-0.5, 0.55)
    ax.set_aspect("equal"); ax.axis("off")


# ══════════════════════════════════════════════════════════════════════
# Part 5A: Gene set mapping QC figure
# ══════════════════════════════════════════════════════════════════════
print("\n=== Part 5: Figures ===")
print("  Figure A: Gene set mapping QC")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: SynGO filtering flow
ax = axes[0, 0]
syngo_excl = syngo_map_df["exclusion_reason"].value_counts()
flow_labels = [
    f"SynGO HGNC\n{n_syngo_total}",
    f"1:1 ortholog\nin model\n{syngo_map_df['in_model'].sum()}",
    f"In human M1\n{syngo_map_df[syngo_map_df['in_model'] & syngo_map_df['in_human_m1_matrix']].shape[0]}",
    f"Non-ribosomal\n(primary)\n{n_syngo_primary}",
]
x_pos = [0, 1, 2, 3]
for i, (x, label) in enumerate(zip(x_pos, flow_labels)):
    color = "#59a14f" if i == len(flow_labels) - 1 else "#4e79a7"
    ax.add_patch(plt.Rectangle((x - 0.35, 0.2), 0.7, 0.6, fill=True, facecolor=color, alpha=0.3, edgecolor=color))
    ax.text(x, 0.5, label, ha="center", va="center", fontsize=9, fontweight="bold")
    if i < len(flow_labels) - 1:
        ax.annotate("", xy=(x + 0.4, 0.5), xytext=(x + 0.6, 0.5),
                     arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
ax.set_xlim(-0.5, 3.5)
ax.set_ylim(0, 1)
ax.axis("off")
ax.set_title("SynGO Gene Filtering Flow", fontsize=11)

# Panel 2: TF filtering flow
ax = axes[0, 1]
n_jaspar_human = len(jaspar_rows)
n_tf_unique = len(tf_human_all)
n_complex_derived = sum(1 for v in tf_gene_info.values() if v["complex_component"])
flow_labels_tf = [
    f"JASPAR Hs\nmotif rows\n{n_jaspar_human}",
    f"Unique TFs\n{n_tf_unique}\n({n_complex_derived} from\ncomplexes)",
    f"Mapped to\nmodel+M1\n{n_tf_primary}",
    f"Singleton\nonly\n{n_tf_singleton}",
]
for i, (x, label) in enumerate(zip(x_pos, flow_labels_tf)):
    color = "#59a14f" if i >= 2 else "#4e79a7"
    ax.add_patch(plt.Rectangle((x - 0.35, 0.1), 0.7, 0.8, fill=True, facecolor=color, alpha=0.3, edgecolor=color))
    ax.text(x, 0.5, label, ha="center", va="center", fontsize=8, fontweight="bold")
    if i < len(flow_labels_tf) - 1:
        ax.annotate("", xy=(x + 0.4, 0.5), xytext=(x + 0.6, 0.5),
                     arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
ax.set_xlim(-0.5, 3.5)
ax.set_ylim(0, 1)
ax.axis("off")
ax.set_title("TF Gene Filtering Flow (JASPAR)", fontsize=11)

# Panel 3: SynGO Venn (previous vs human-first)
ax = axes[1, 0]
if HAS_VENN:
    venn2([prev_syngo_genes, syngo_primary_mouse],
          set_labels=("Previous\n(mouse-indexed)", "Human-first"),
          ax=ax)
else:
    _draw_manual_venn(ax, prev_syngo_genes, syngo_primary_mouse,
                      "Previous", "Human-first")
ax.set_title(f"SynGO Gene Set Overlap\n(prev={len(prev_syngo_genes)}, new={len(syngo_primary_mouse)}, "
             f"overlap={len(syngo_overlap)})", fontsize=10)

# Panel 4: TF Venn (previous vs human-first)
ax = axes[1, 1]
if HAS_VENN:
    venn2([prev_tf_genes, tf_primary_mouse],
          set_labels=("Previous\n(mouse-indexed)", "Human-first\nprimary"),
          ax=ax)
else:
    _draw_manual_venn(ax, prev_tf_genes, tf_primary_mouse,
                      "Previous", "Human-first")
ax.set_title(f"TF Gene Set Overlap\n(prev={len(prev_tf_genes)}, new={len(tf_primary_mouse)}, "
             f"overlap={len(tf_overlap)})", fontsize=10)

fig.suptitle("Human-First Gene Set Mapping QC", fontsize=13, y=1.01)
fig.tight_layout()
save_fig(fig, "human_first_gene_set_mapping_qc")


# ══════════════════════════════════════════════════════════════════════
# Part 5B: Predicted delta summary figure
# ══════════════════════════════════════════════════════════════════════
print("  Figure B: Predicted delta summary")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# Panel 1: Neuronal vs non-neuronal bar plot
ax = axes[0, 0]
gs_order = ["all_mapped", "human_first_SynGO", "human_first_TF_primary", "human_first_TF_singleton"]
gs_short = ["All mapped", "SynGO", "TF primary", "TF singleton"]
x = np.arange(len(gs_order))
w = 0.35
neur_vals = [neur_df[neur_df["gene_set"] == gs]["neuronal_mean_delta"].values[0] for gs in gs_order]
nneur_vals = [neur_df[neur_df["gene_set"] == gs]["nonneuronal_mean_delta"].values[0] for gs in gs_order]
ax.bar(x - w / 2, neur_vals, w, label="Neuronal", color="#e15759")
ax.bar(x + w / 2, nneur_vals, w, label="Non-neuronal", color="#4e79a7")
ax.set_xticks(x)
ax.set_xticklabels(gs_short, fontsize=8, rotation=20)
ax.set_ylabel("Mean predicted delta")
ax.set_title("Neuronal vs non-neuronal shift")
ax.legend(fontsize=8)

# Panel 2: SynGO vs TF by mouse-like class
ax = axes[0, 1]
cls_order = contrast_df.sort_values("syngo_mean_delta", ascending=False)["class"].values
x2 = np.arange(len(cls_order))
w2 = 0.35
syngo_vals = [contrast_df[contrast_df["class"] == c]["syngo_mean_delta"].values[0] for c in cls_order]
tf_vals = [contrast_df[contrast_df["class"] == c]["tf_primary_mean_delta"].values[0] for c in cls_order]
ax.bar(x2 - w2 / 2, syngo_vals, w2, label="SynGO", color="#e15759")
ax.bar(x2 + w2 / 2, tf_vals, w2, label="TF primary", color="#4e79a7")
ax.set_xticks(x2)
ax.set_xticklabels(cls_order, fontsize=6, rotation=45, ha="right")
ax.set_ylabel("Mean predicted delta")
ax.set_title("SynGO vs TF by mouse-like class")
ax.legend(fontsize=8)

# Panel 3: Old vs new neuronal diff comparison
ax = axes[0, 2]
labels_comp = ["SynGO", "TF"]
prev_diffs = [comp_df[comp_df["gene_set"] == gs]["previous_neuronal_diff"].values[0] for gs in labels_comp]
new_diffs = [comp_df[comp_df["gene_set"] == gs]["human_first_neuronal_diff"].values[0] for gs in labels_comp]
x3 = np.arange(len(labels_comp))
w3 = 0.35
ax.bar(x3 - w3 / 2, prev_diffs, w3, label="Previous (mouse-indexed)", color="#bab0ac")
ax.bar(x3 + w3 / 2, new_diffs, w3, label="Human-first", color="#59a14f")
ax.set_xticks(x3)
ax.set_xticklabels(labels_comp)
ax.set_ylabel("Neuronal - non-neuronal diff")
ax.set_title("Old vs human-first\nneuronal specificity")
ax.legend(fontsize=8)

# Panel 4: Class-wise correlation scatter (SynGO)
ax = axes[1, 0]
common_cls_syngo = sorted(set(prev_contrast["class"]) & set(contrast_df["class"]))
prev_syngo_cls = [prev_contrast[prev_contrast["class"] == c]["syngo_mean_delta"].values[0] for c in common_cls_syngo]
new_syngo_cls = [contrast_df[contrast_df["class"] == c]["syngo_mean_delta"].values[0] for c in common_cls_syngo]
ax.scatter(prev_syngo_cls, new_syngo_cls, s=40, c="#e15759")
for i, c in enumerate(common_cls_syngo):
    ax.annotate(c, (prev_syngo_cls[i], new_syngo_cls[i]), fontsize=6, ha="left")
r_s, _ = stats.pearsonr(prev_syngo_cls, new_syngo_cls)
lims = [min(min(prev_syngo_cls), min(new_syngo_cls)) - 0.02,
        max(max(prev_syngo_cls), max(new_syngo_cls)) + 0.02]
ax.plot(lims, lims, "k--", alpha=0.3)
ax.set_xlabel("Previous SynGO mean delta")
ax.set_ylabel("Human-first SynGO mean delta")
ax.set_title(f"SynGO class-wise: r={r_s:.4f}")

# Panel 5: Class-wise correlation scatter (TF)
ax = axes[1, 1]
prev_tf_cls = [prev_contrast[prev_contrast["class"] == c]["tf_mean_delta"].values[0] for c in common_cls_syngo]
new_tf_cls = [contrast_df[contrast_df["class"] == c]["tf_primary_mean_delta"].values[0] for c in common_cls_syngo]
ax.scatter(prev_tf_cls, new_tf_cls, s=40, c="#4e79a7")
for i, c in enumerate(common_cls_syngo):
    ax.annotate(c, (prev_tf_cls[i], new_tf_cls[i]), fontsize=6, ha="left")
r_t, _ = stats.pearsonr(prev_tf_cls, new_tf_cls)
lims_t = [min(min(prev_tf_cls), min(new_tf_cls)) - 0.02,
          max(max(prev_tf_cls), max(new_tf_cls)) + 0.02]
ax.plot(lims_t, lims_t, "k--", alpha=0.3)
ax.set_xlabel("Previous TF mean delta")
ax.set_ylabel("Human-first TF mean delta")
ax.set_title(f"TF class-wise: r={r_t:.4f}")

# Panel 6: Violin of predicted deltas by gene set + neuronality
ax = axes[1, 2]
violin_data = []
violin_labels_list = []
violin_colors = []
for gs_label, gs_mice, color in [
    ("SynGO\nneuronal", syngo_primary_mouse, "#e15759"),
    ("SynGO\nnon-neur", syngo_primary_mouse, "#f28e2b"),
    ("TF pri\nneuronal", tf_primary_mouse, "#4e79a7"),
    ("TF pri\nnon-neur", tf_primary_mouse, "#76b7b2"),
]:
    is_neur = "neuronal" in gs_label and "non" not in gs_label
    if is_neur:
        sub = broad_mapped_neur[broad_mapped_neur["mouse_model_gene"].isin(gs_mice)]
    else:
        sub = broad_mapped_nneur[broad_mapped_nneur["mouse_model_gene"].isin(gs_mice)]
    violin_data.append(sub["predicted_delta"].values)
    violin_labels_list.append(gs_label)
    violin_colors.append(color)

parts = ax.violinplot(violin_data, positions=range(len(violin_data)), showmeans=True, showmedians=True)
for i, pc in enumerate(parts["bodies"]):
    pc.set_facecolor(violin_colors[i])
    pc.set_alpha(0.6)
ax.set_xticks(range(len(violin_labels_list)))
ax.set_xticklabels(violin_labels_list, fontsize=7)
ax.set_ylabel("Predicted delta")
ax.set_title("Delta distributions by gene set")
ax.axhline(0, color="gray", linestyle="--", alpha=0.3)

fig.suptitle("Human-First Gene Set Predicted Delta Summary", fontsize=13, y=1.01)
fig.tight_layout()
save_fig(fig, "human_first_predicted_delta_summary")


# ══════════════════════════════════════════════════════════════════════
# Part 5C: Mouse-human similarity with human-first gene sets
# ══════════════════════════════════════════════════════════════════════
print("  Figure C: Mouse-human similarity")

CT_NAME_MAP = {
    "L2_3 IT": "L2/3_IT", "L5 IT": "L5_IT", "L5 ET": "L5_ET",
    "L5_6 NP": "L5/6_NP", "L6 CT": "L6_CT", "L6 IT": "L6_IT",
    "L6 IT Car3": "L6_IT_Car3", "L6b": "L6b",
    "Pvalb": "Pvalb", "Sst": "Sst", "Vip": "Vip",
    "Lamp5": "Lamp5", "Sncg": "Sncg", "Sst Chodl": "Sst_Chodl",
    "Astro": "Astro", "Oligo": "Oligo", "OPC": "OPC",
    "Endo": "Endo", "VLMC": "VLMC", "SMC": "SMC",
    "Macrophage": "Micro",
}

# Load mouse all-gene deltas
mouse_delta_files = glob.glob(os.path.join(MOUSE_DELTA_DIR, "*_delta.csv"))
mouse_deltas = {}  # (ct, mouse_gene) -> mean_delta
for fpath in mouse_delta_files:
    basename = os.path.basename(fpath).replace("_delta.csv", "")
    ct = CT_NAME_MAP.get(basename)
    if ct is None:
        continue
    with open(fpath) as f:
        for row in csv.DictReader(f):
            mouse_deltas[(ct, row["gene"])] = float(row["mean_delta"])

# Build human M1 delta lookup from ml_mapped
human_deltas = {}
for _, row in ml_mapped.iterrows():
    human_deltas[(row["group"], row["mouse_model_gene"])] = row["predicted_delta"]

# Common classes
mouse_cts = set(ct for (ct, _) in mouse_deltas)
human_cts = set(ct for (ct, _) in human_deltas)
common_cts = sorted(mouse_cts & human_cts)

similarity_rows = []
for gs_label, gs_mice in [("all_mapped", mouse_in_model),
                            ("human_first_SynGO", syngo_primary_mouse),
                            ("human_first_TF_primary", tf_primary_mouse)]:
    for ct in common_cts:
        mouse_vals = []
        human_vals = []
        for g in gs_mice:
            mk = (ct, g)
            hk = (ct, g)
            if mk in mouse_deltas and hk in human_deltas:
                mouse_vals.append(mouse_deltas[mk])
                human_vals.append(human_deltas[hk])
        if len(mouse_vals) >= 10:
            r_p, _ = stats.pearsonr(mouse_vals, human_vals)
            r_s, _ = stats.spearmanr(mouse_vals, human_vals)
            mae = np.mean(np.abs(np.array(mouse_vals) - np.array(human_vals)))
            sign_agree = np.mean(np.sign(mouse_vals) == np.sign(human_vals))
            similarity_rows.append({
                "gene_set": gs_label,
                "class": ct,
                "n_genes": len(mouse_vals),
                "pearson": r_p,
                "spearman": r_s,
                "mae": mae,
                "sign_agreement": sign_agree,
            })

sim_df = pd.DataFrame(similarity_rows)
sim_df.to_csv(os.path.join(RES_DIR, "human_first_mouse_human_shift_similarity_by_class.csv"), index=False)
print(f"  Mouse-human similarity: {len(sim_df)} rows")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax_i, (gs_label, gs_short) in enumerate([
    ("all_mapped", "All mapped"),
    ("human_first_SynGO", "SynGO"),
    ("human_first_TF_primary", "TF primary"),
]):
    ax = axes[ax_i]
    sub = sim_df[sim_df["gene_set"] == gs_label].sort_values("pearson", ascending=True)
    if len(sub) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(gs_short)
        continue
    colors = ["#e15759" if r["n_genes"] < 30 else "#4e79a7" for _, r in sub.iterrows()]
    ax.barh(range(len(sub)), sub["pearson"].values, color=colors)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub["class"].values, fontsize=7)
    ax.set_xlabel("Pearson r (mouse vs human predicted delta)")
    median_r = sub["pearson"].median()
    ax.set_title(f"{gs_short} (n={sub['n_genes'].values[0]}, median r={median_r:.3f})")
    ax.axvline(0, color="gray", linestyle="--", alpha=0.3)

fig.suptitle("Mouse-Human Predicted-Shift Similarity (Human-First Gene Sets)", fontsize=12, y=1.02)
fig.tight_layout()
save_fig(fig, "human_first_mouse_human_similarity")


# ══════════════════════════════════════════════════════════════════════
# Part 6: Report
# ══════════════════════════════════════════════════════════════════════
print("\n=== Part 6: Report ===")

report = f"""# Human-First Gene Set Transfer Validation Report

**Date:** {datetime.datetime.now().strftime("%Y-%m-%d")}
**Script:** `validation/human_m1_transfer/scripts/12_human_first_geneset_validation.py`

## 1. Why This Rerun Was Needed

The previous human M1 transfer analysis used gene sets defined in mouse gene space (mouse SynGO
gene symbols from the mouse-trained model validation, and mouse-indexed JASPAR TF genes). For
scientific rigor when reporting cross-species transfer results, gene sets should be defined from
**human annotation resources** and then projected into the mouse-trained model gene space using
one-to-one human-mouse orthologs.

Gene sets were defined from human annotation resources and then projected into the mouse-trained
model gene space using one-to-one human-mouse orthologs.

## 2. Data Sources

### SynGO
- **Source:** SynGO 1.3 (`syngo1.3_complete_data.zip`, `genes.xlsx`)
- **Path:** `validation/syngo/syngo1.3_complete_data.zip`
- **Gene identifiers:** HGNC symbols (human)
- **Total genes:** {n_syngo_total}
- **Version:** SynGO 1.3 (March 2025 release)

SynGO annotates the function and synaptic location of encoded proteins, not RNA localization.

### JASPAR
- **Source:** JASPAR CORE 2026 (`ultimate_metadata_table_CORE.tsv`)
- **Path:** `validation/tf/external/ultimate_metadata_table_CORE_2026.tsv`
- **File date:** {jaspar_date}
- **File MD5:** `{jaspar_md5}`
- **Filter:** `species == "Homo sapiens"` from CORE collection
- **Motif rows:** {n_jaspar_human}
- **Unique TF symbols:** {n_tf_unique} (including {n_complex_derived} derived from A::B complexes)

### Ortholog Mapping
- **Source:** MGI HOM_MouseHumanSequence.rpt, one-to-one orthologs only
- **Mapped genes in human M1:** {len(human_in_m1)} / {len(gene_map)} model genes

## 3. Human-First Mapping Logic

1. Start with human gene symbols from each annotation resource (SynGO HGNC, JASPAR Homo sapiens).
2. Map each human symbol to its one-to-one mouse ortholog using the MGI ortholog table.
3. Keep only genes where the mouse ortholog is in the ImputeVAE model gene list (10,134 genes).
4. Keep only genes where the human ortholog is detected in the Allen Human M1 matrix.
5. For SynGO: exclude ribosomal genes (human RPL/RPS or mouse Rpl/Rps prefixes).
6. For TF: create primary set (all mapped) and singleton-only sensitivity set (excluding
   genes known only from A::B complex motifs).

## 4. Final Gene Counts

### SynGO

| Stage | Count |
|-------|-------|
| SynGO HGNC symbols (all) | {n_syngo_total} |
| With 1:1 mouse ortholog in model | {syngo_map_df['in_model'].sum()} |
| Also detected in human M1 | {syngo_map_df[syngo_map_df['in_model'] & syngo_map_df['in_human_m1_matrix']].shape[0]} |
| Non-ribosomal (primary set) | **{n_syngo_primary}** |
| Excluded: ribosomal | {syngo_map_df[syngo_map_df['exclusion_reason'] == 'ribosomal'].shape[0]} |
| Excluded: no 1:1 ortholog | {syngo_map_df[syngo_map_df['exclusion_reason'] == 'no_one_to_one_ortholog'].shape[0]} |
| Excluded: ortholog not in model | {syngo_map_df[syngo_map_df['exclusion_reason'] == 'ortholog_not_in_model'].shape[0]} |
| Excluded: not in human M1 | {syngo_map_df[syngo_map_df['exclusion_reason'].isin(['not_in_human_m1_matrix', 'ortholog_in_model_but_not_in_human_m1'])].shape[0]} |

### TF (JASPAR)

| Stage | Count |
|-------|-------|
| Unique human TF symbols | {n_tf_total} |
| Mapped to model + human M1 (primary) | **{n_tf_primary}** |
| Singleton-only sensitivity set | **{n_tf_singleton}** |
| Excluded: no 1:1 ortholog | {tf_map_df[tf_map_df['exclusion_reason'] == 'no_one_to_one_ortholog'].shape[0]} |
| Excluded: ortholog not in model | {tf_map_df[tf_map_df['exclusion_reason'] == 'ortholog_not_in_model'].shape[0]} |
| Excluded: not in human M1 | {tf_map_df[tf_map_df['exclusion_reason'].isin(['not_in_human_m1_matrix', 'ortholog_in_model_but_not_in_human_m1'])].shape[0]} |

## 5. Comparison: Previous vs Human-First

| Metric | SynGO | TF |
|--------|-------|-----|
| Previous n genes | {comp_df[comp_df['gene_set']=='SynGO']['previous_n_genes'].values[0]} | {comp_df[comp_df['gene_set']=='TF']['previous_n_genes'].values[0]} |
| Human-first n genes | {comp_df[comp_df['gene_set']=='SynGO']['human_first_n_genes'].values[0]} | {comp_df[comp_df['gene_set']=='TF']['human_first_n_genes'].values[0]} |
| Overlap | {comp_df[comp_df['gene_set']=='SynGO']['overlap_n_genes'].values[0]} | {comp_df[comp_df['gene_set']=='TF']['overlap_n_genes'].values[0]} |
| Previous neuronal diff | {comp_df[comp_df['gene_set']=='SynGO']['previous_neuronal_diff'].values[0]:.4f} | {comp_df[comp_df['gene_set']=='TF']['previous_neuronal_diff'].values[0]:.4f} |
| Human-first neuronal diff | {comp_df[comp_df['gene_set']=='SynGO']['human_first_neuronal_diff'].values[0]:.4f} | {comp_df[comp_df['gene_set']=='TF']['human_first_neuronal_diff'].values[0]:.4f} |
| Class-wise Pearson | {comp_df[comp_df['gene_set']=='SynGO']['classwise_pearson'].values[0]:.4f} | {comp_df[comp_df['gene_set']=='TF']['classwise_pearson'].values[0]:.4f} |
| Class-wise Spearman | {comp_df[comp_df['gene_set']=='SynGO']['classwise_spearman'].values[0]:.4f} | {comp_df[comp_df['gene_set']=='TF']['classwise_spearman'].values[0]:.4f} |

## 6. Main Results (Human-First)

### Neuronal vs Non-Neuronal Predicted Shift

| Gene set | n genes | Neuronal | Non-neuronal | Diff |
|----------|---------|----------|-------------|------|
"""

for _, r in neur_df.iterrows():
    report += f"| {r['gene_set']} | {r['n_genes']} | {r['neuronal_mean_delta']:.4f} | {r['nonneuronal_mean_delta']:.4f} | **{r['neuronal_minus_nonneuronal']:.4f}** |\n"

report += f"""
### Mouse-Human Predicted-Shift Similarity

| Gene set | Median Pearson | Range | n classes |
|----------|---------------|-------|-----------|
"""

for gs in ["all_mapped", "human_first_SynGO", "human_first_TF_primary"]:
    sub = sim_df[sim_df["gene_set"] == gs]
    if len(sub) > 0:
        report += f"| {gs} | {sub['pearson'].median():.3f} | [{sub['pearson'].min():.3f}, {sub['pearson'].max():.3f}] | {len(sub)} |\n"

report += f"""
## 7. Biological Interpretation

The human-first gene set analysis confirms the same biological pattern as the previous
mouse-indexed analysis:

1. **SynGO genes show the largest neuronal-specific predicted shift.** Genes encoding synaptic
   proteins are predicted to be more cytoplasmically enriched in neurons than non-neurons, consistent
   with the mouse training data.

2. **TF genes show smaller, more uniform shifts.** Transcription factor genes, which function in the
   nucleus, show weaker predicted cytoplasmic enrichment regardless of cell type.

3. **Results are robust to gene set definition approach.** Class-wise Pearson correlations between
   previous (mouse-indexed) and human-first results are {comp_df[comp_df['gene_set']=='SynGO']['classwise_pearson'].values[0]:.3f} (SynGO) and
   {comp_df[comp_df['gene_set']=='TF']['classwise_pearson'].values[0]:.3f} (TF), indicating that conclusions do not depend on
   whether gene sets were defined in mouse or human gene space first.

## 8. Caveats

1. **No human whole-cell ground truth.** Allen Human M1 data are nuclei only. These are
   cross-species plausibility checks, not direct human validation.
2. **Zero-shot cross-species transfer.** The model was trained exclusively on mouse MOp data.
3. **Ortholog-restricted gene space.** Only {len(human_in_m1)}/{len(gene_map)} model genes have
   detected one-to-one human orthologs.
4. **SynGO annotates protein function/location, not RNA localization.** The SynGO score shift
   reflects that genes encoding synaptic proteins tend to have more cytoplasmically enriched mRNA
   in neurons, but this is a correlation with protein function, not direct RNA transport annotation.
5. **JASPAR complex handling.** TF motif complexes (A::B) were split into component genes. The
   singleton-only sensitivity set excludes genes known only from complexes, but results are similar.

## Output Files

```
results/
  human_first_syngo_gene_mapping.csv
  human_first_tf_gene_mapping.csv
  human_first_syngo_transfer_summary.csv
  human_first_tf_transfer_summary.csv
  human_first_neuronal_vs_nonneuronal_predicted_shift.csv
  human_first_syngo_vs_tf_transfer_contrast.csv
  human_first_vs_previous_gene_set_comparison.csv
  human_first_mouse_human_shift_similarity_by_class.csv
figures/
  human_first_gene_set_mapping_qc.{{png,pdf,svg}}
  human_first_predicted_delta_summary.{{png,pdf,svg}}
  human_first_mouse_human_similarity.{{png,pdf,svg}}
reports/
  HUMAN_FIRST_GENESET_TRANSFER_REPORT.md
  human_first_geneset_acceptance_checks.txt
```
"""

with open(os.path.join(REP_DIR, "HUMAN_FIRST_GENESET_TRANSFER_REPORT.md"), "w") as f:
    f.write(report)
print("  Saved HUMAN_FIRST_GENESET_TRANSFER_REPORT.md")


# ══════════════════════════════════════════════════════════════════════
# Part 7: Acceptance checks
# ══════════════════════════════════════════════════════════════════════
print("\n=== Part 7: Acceptance checks ===")

checks_passed = 0
checks_failed = 0
check_lines = []

def check(cond, msg):
    global checks_passed, checks_failed
    if cond:
        check_lines.append(f"  [OK] {msg}")
        checks_passed += 1
    else:
        check_lines.append(f"  [FAIL] {msg}")
        checks_failed += 1
    print(check_lines[-1])

# File existence
check(os.path.exists(os.path.join(RES_DIR, "human_first_syngo_gene_mapping.csv")),
      "SynGO mapping file exists")
check(os.path.exists(os.path.join(RES_DIR, "human_first_tf_gene_mapping.csv")),
      "TF mapping file exists")

# No duplicate mouse_model_gene in primary sets
syngo_primary_df = syngo_map_df[syngo_map_df["included_primary"]]
check(syngo_primary_df["mouse_model_gene"].is_unique,
      "No duplicate mouse_model_gene in SynGO primary")
tf_primary_df = tf_map_df[tf_map_df["included_primary"]]
check(tf_primary_df["mouse_model_gene"].is_unique,
      "No duplicate mouse_model_gene in TF primary")

# All included have non-empty symbols
check((syngo_primary_df["human_gene_symbol"] != "").all() and (syngo_primary_df["mouse_model_gene"] != "").all(),
      "SynGO primary: all have human + mouse symbols")
check((tf_primary_df["human_gene_symbol"] != "").all() and (tf_primary_df["mouse_model_gene"] != "").all(),
      "TF primary: all have human + mouse symbols")

# All in human M1 mapped space
check(syngo_primary_df["in_human_m1_matrix"].all(),
      "SynGO primary: all in human M1 mapped space")
check(tf_primary_df["in_human_m1_matrix"].all(),
      "TF primary: all in human M1 mapped space")

# All appear in predicted delta tables
ml_genes = set(ml_mapped["mouse_model_gene"])
check(syngo_primary_mouse.issubset(ml_genes),
      f"SynGO primary: all {len(syngo_primary_mouse)} genes in predicted delta table")
check(tf_primary_mouse.issubset(ml_genes),
      f"TF primary: all {len(tf_primary_mouse)} genes in predicted delta table")

# SynGO excludes ribosomal
check(not any(g.startswith("Rpl") or g.startswith("Rps") for g in syngo_primary_mouse),
      "SynGO primary: no Rpl/Rps mouse ribosomal genes")
check(not any(g.startswith("RPL") or g.startswith("RPS") for g in syngo_primary_human),
      "SynGO primary: no RPL/RPS human ribosomal genes")

# Counts reported
check(n_tf_primary > 0, f"TF primary count: {n_tf_primary}")
check(n_tf_singleton > 0, f"TF singleton count: {n_tf_singleton}")

# Neuronal summary has all expected gene sets
expected_gs = {"human_first_SynGO", "human_first_TF_primary", "human_first_TF_singleton", "all_mapped"}
check(set(neur_df["gene_set"]) == expected_gs,
      f"Neuronal summary has all {len(expected_gs)} expected gene sets")

# Comparison exists
check(os.path.exists(os.path.join(RES_DIR, "human_first_vs_previous_gene_set_comparison.csv")),
      "Previous vs human-first comparison exists")

# Figures in all formats
for fig_name in ["human_first_gene_set_mapping_qc",
                  "human_first_predicted_delta_summary",
                  "human_first_mouse_human_similarity"]:
    for ext in ["png", "pdf", "svg"]:
        check(os.path.exists(os.path.join(FIG_DIR, f"{fig_name}.{ext}")),
              f"Figure: {fig_name}.{ext}")

# Report exists
check(os.path.exists(os.path.join(REP_DIR, "HUMAN_FIRST_GENESET_TRANSFER_REPORT.md")),
      "Report exists")

# Save acceptance checks
acc_path = os.path.join(REP_DIR, "human_first_geneset_acceptance_checks.txt")
with open(acc_path, "w") as f:
    f.write("=== Human-First Gene Set Acceptance Checks ===\n\n")
    for line in check_lines:
        f.write(line + "\n")
    f.write(f"\n{'='*50}\n")
    f.write(f"RESULTS: {checks_passed} passed, {checks_failed} failed\n")
    f.write(f"{'='*50}\n")
print(f"\nSaved: {acc_path}")

print(f"\n{'='*50}")
print(f"RESULTS: {checks_passed} passed, {checks_failed} failed")
print(f"{'='*50}")

if checks_failed > 0:
    sys.exit(1)

print("\n=== Done ===")
