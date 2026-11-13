#!/usr/bin/env python3
"""Part B+C: Characterize Allen Human M1 metadata and build cell-type mapping."""

import csv
import json
import os
from collections import Counter

ROOT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE"
DATA_DIR = os.path.join(ROOT, "validation/human_m1_transfer/external/allen_human_m1_10x")
PROC_DIR = os.path.join(ROOT, "validation/human_m1_transfer/processed")
REPORT_DIR = os.path.join(ROOT, "validation/human_m1_transfer/reports")

os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Load metadata
print("Loading metadata.csv...")
with open(os.path.join(DATA_DIR, "metadata.csv")) as f:
    reader = csv.DictReader(f)
    meta_cols = reader.fieldnames
    rows = list(reader)

print(f"Total nuclei: {len(rows)}")
print(f"Columns ({len(meta_cols)}): {meta_cols[:15]}...")

# --- Part B: Characterize ---

# Extract key fields
class_counts = Counter(r["class_label"] for r in rows)
subclass_counts = Counter(r["subclass_label"] for r in rows)
cluster_counts = Counter(r["cluster_label"] for r in rows)
donor_counts = Counter(r.get("external_donor_name_label", r.get("donor_sex_label", "unknown")) for r in rows)

# Find donor column
donor_col = None
for c in meta_cols:
    if "donor" in c.lower() and "label" in c.lower():
        donor_col = c
        break
if donor_col is None:
    # Try without _label
    for c in meta_cols:
        if "donor" in c.lower() and "sex" not in c.lower() and "color" not in c.lower() and "order" not in c.lower():
            donor_col = c
            break

if donor_col:
    donor_counts = Counter(r[donor_col] for r in rows)
    print(f"Donor column: {donor_col}")
else:
    # Check all columns for potential donor info
    print("Available columns:", meta_cols)
    donor_counts = Counter()

sex_counts = Counter(r.get("donor_sex_label", "unknown") for r in rows)
region_counts = Counter(r.get("region_label", "unknown") for r in rows)
layer_counts = Counter(r.get("cortical_layer_label", "unknown") for r in rows)

# Save metadata summary
summary_path = os.path.join(REPORT_DIR, "allen_human_m1_metadata_summary.csv")
with open(summary_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "value"])
    w.writerow(["total_nuclei", len(rows)])
    w.writerow(["n_columns", len(meta_cols)])
    w.writerow(["columns", ";".join(meta_cols)])
    w.writerow(["n_classes", len(class_counts)])
    w.writerow(["classes", ";".join(sorted(class_counts.keys()))])
    w.writerow(["n_subclasses", len(subclass_counts)])
    w.writerow(["subclasses", ";".join(sorted(subclass_counts.keys()))])
    w.writerow(["n_clusters", len(cluster_counts)])
    w.writerow(["n_donors", len(donor_counts)])
    w.writerow(["donor_col_used", donor_col or "none_found"])
    w.writerow(["donors", ";".join(sorted(donor_counts.keys()))])
    w.writerow(["n_sexes", len(sex_counts)])
    w.writerow(["sexes", ";".join(f"{k}:{v}" for k, v in sex_counts.most_common())])
    w.writerow(["n_regions", len(region_counts)])
    w.writerow(["regions", ";".join(f"{k}:{v}" for k, v in region_counts.most_common())])
    w.writerow(["n_layers", len(layer_counts)])
    w.writerow(["layers", ";".join(f"{k}:{v}" for k, v in layer_counts.most_common())])
print(f"Saved: {summary_path}")

# Cell type counts
ct_path = os.path.join(REPORT_DIR, "allen_human_m1_celltype_counts.csv")
with open(ct_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["cell_type", "class", "subclass", "n_nuclei"])
    for ct, n in cluster_counts.most_common():
        # Find class/subclass for this cluster
        example = next(r for r in rows if r["cluster_label"] == ct)
        w.writerow([ct, example["class_label"], example["subclass_label"], n])
print(f"Saved: {ct_path}")

# Donor counts
donor_path = os.path.join(REPORT_DIR, "allen_human_m1_donor_counts.csv")
with open(donor_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["donor", "n_nuclei"])
    for d, n in donor_counts.most_common():
        w.writerow([d, n])
print(f"Saved: {donor_path}")

# --- Part C: Cell-type harmonization ---
print("\n=== Cell-type harmonization ===")

# Build broad class mapping from Allen class/subclass
BROAD_MAP = {
    "Glutamatergic": "excitatory_neuron",
    "GABAergic": "inhibitory_neuron",
    "Non-Neuronal": None,  # need subclass
}

SUBCLASS_BROAD_MAP = {
    "Astro": "astrocyte",
    "Astrocyte": "astrocyte",
    "Oligo": "oligodendrocyte",
    "Oligodendrocyte": "oligodendrocyte",
    "OPC": "opc",
    "Endo": "endothelial",
    "Endothelial": "endothelial",
    "Micro-PVM": "microglia_macrophage",
    "Microglia": "microglia_macrophage",
    "VLMC": "vlmc_pericyte_smc",
    "Pericyte": "vlmc_pericyte_smc",
    "SMC": "vlmc_pericyte_smc",
}

# Mouse-like fine class mapping from subclass
FINE_MAP = {
    # Excitatory
    "L2/3 IT": "L2/3_IT",
    "L4/5 IT": "L4/5_IT",
    "L5 IT": "L5_IT",
    "L5 ET": "L5_ET",
    "L5/6 NP": "L5/6_NP",
    "L6 IT": "L6_IT",
    "L6 IT Car3": "L6_IT_Car3",
    "L6 CT": "L6_CT",
    "L6b": "L6b",
    # Inhibitory
    "Lamp5": "Lamp5",
    "Sncg": "Sncg",
    "Vip": "Vip",
    "Sst": "Sst",
    "Sst Chodl": "Sst_Chodl",
    "Pvalb": "Pvalb",
    "Chandelier": "Pvalb",  # chandelier are Pvalb subtype
    # Non-neuronal
    "Astro": "Astro",
    "Oligo": "Oligo",
    "OPC": "OPC",
    "Endo": "Endo",
    "Micro-PVM": "Micro",
    "VLMC": "VLMC",
    "Pericyte": "VLMC",  # group with pericytes
    "SMC": "SMC",
}

# Build mapping for each unique cluster
mapping_rows = []
for ct, n in cluster_counts.most_common():
    example = next(r for r in rows if r["cluster_label"] == ct)
    cls = example["class_label"]
    subcls = example["subclass_label"]

    # Broad class
    broad = BROAD_MAP.get(cls)
    if broad is None:
        broad = SUBCLASS_BROAD_MAP.get(subcls, "unknown")

    # Fine mouse-like class
    fine = FINE_MAP.get(subcls, "broad_only")

    # Confidence
    if fine != "broad_only":
        confidence = "high"
    elif broad != "unknown":
        confidence = "medium"
    else:
        confidence = "low"

    notes = ""
    if subcls == "Chandelier":
        notes = "Chandelier cells mapped to Pvalb (biological subtype)"

    mapping_rows.append({
        "human_cell_type_label": ct,
        "original_annotation_level": "cluster",
        "allen_class": cls,
        "allen_subclass": subcls,
        "mapped_broad_class": broad,
        "mapped_mouse_like_class": fine,
        "n_nuclei": n,
        "mapping_confidence": confidence,
        "notes": notes,
    })

# Save mapping
map_path = os.path.join(PROC_DIR, "human_m1_celltype_mapping.csv")
map_fields = ["human_cell_type_label", "original_annotation_level", "allen_class",
              "allen_subclass", "mapped_broad_class", "mapped_mouse_like_class",
              "n_nuclei", "mapping_confidence", "notes"]
with open(map_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=map_fields)
    w.writeheader()
    w.writerows(mapping_rows)
print(f"Saved: {map_path}")

# Mapping summary
map_summary_path = os.path.join(REPORT_DIR, "celltype_mapping_summary.csv")
with open(map_summary_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=map_fields)
    w.writeheader()
    w.writerows(mapping_rows)
print(f"Saved: {map_summary_path}")

# Print summaries
print(f"\nBroad class distribution:")
broad_totals = Counter()
for r in mapping_rows:
    broad_totals[r["mapped_broad_class"]] += r["n_nuclei"]
for cls, n in broad_totals.most_common():
    print(f"  {cls}: {n:,}")

print(f"\nMouse-like class distribution:")
fine_totals = Counter()
for r in mapping_rows:
    fine_totals[r["mapped_mouse_like_class"]] += r["n_nuclei"]
for cls, n in fine_totals.most_common():
    print(f"  {cls}: {n:,}")

print(f"\nMapping confidence:")
conf_counts = Counter(r["mapping_confidence"] for r in mapping_rows)
for c, n in conf_counts.most_common():
    print(f"  {c}: {n} clusters")
