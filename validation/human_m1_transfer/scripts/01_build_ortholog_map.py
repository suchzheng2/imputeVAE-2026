#!/usr/bin/env python3
"""Part A: Build human-to-mouse one-to-one ortholog map via MGI homology report."""

import csv
import os
import sys
from collections import Counter, defaultdict

ROOT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE"
OUTDIR = os.path.join(ROOT, "validation/human_m1_transfer/orthologs")
REPORT_DIR = os.path.join(ROOT, "validation/human_m1_transfer/reports")
GENE_LIST = os.path.join(ROOT, "data/training_gene_list_v3.csv")
MGI_FILE = "/tmp/HOM_MouseHumanSequence.rpt"

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# 1. Load mouse model gene list
with open(GENE_LIST) as f:
    mouse_genes = [line.strip() for line in f if line.strip()]
mouse_gene_set = set(mouse_genes)
mouse_gene_index = {g: i for i, g in enumerate(mouse_genes)}
print(f"Mouse model genes: {len(mouse_genes)}")

# 2. Parse MGI homology report
# Format: rows grouped by "DB Class Key" — each key has one mouse + one human entry (for 1:1)
print("Parsing MGI homology report...")

groups = defaultdict(list)
with open(MGI_FILE, "r") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        key = row["DB Class Key"]
        organism = row["Common Organism Name"]
        symbol = row["Symbol"]
        entrez = row["EntrezGene ID"]
        groups[key].append({
            "organism": organism,
            "symbol": symbol,
            "entrez": entrez,
            "mgi_id": row.get("Mouse MGI ID", ""),
            "hgnc_id": row.get("HGNC ID", ""),
        })

print(f"Total homology groups: {len(groups)}")

# 3. Extract human-mouse pairs
raw_rows = []
for key, members in groups.items():
    mice = [m for m in members if m["organism"] == "mouse, laboratory"]
    humans = [m for m in members if m["organism"] == "human"]
    if not mice or not humans:
        continue

    is_one2one = len(mice) == 1 and len(humans) == 1
    orth_type = "ortholog_one2one" if is_one2one else f"ortholog_{len(humans)}2{len(mice)}"

    for h in humans:
        for m in mice:
            raw_rows.append({
                "human_gene_symbol": h["symbol"],
                "human_ensembl_gene_id": h.get("hgnc_id", ""),
                "mouse_gene_symbol": m["symbol"],
                "mouse_ensembl_gene_id": m.get("mgi_id", ""),
                "orthology_type": orth_type,
                "source": "MGI_HOM_MouseHumanSequence",
                "confidence_score": "1" if is_one2one else "0",
                "is_one_to_one": "1" if is_one2one else "0",
                "in_mouse_model_gene_list": "1" if m["symbol"] in mouse_gene_set else "0",
                "db_class_key": key,
            })

print(f"Total human-mouse ortholog pairs: {len(raw_rows)}")

# Save raw orthologs
raw_path = os.path.join(OUTDIR, "human_mouse_orthologs_raw.csv")
raw_fields = ["human_gene_symbol", "human_ensembl_gene_id", "mouse_gene_symbol",
              "mouse_ensembl_gene_id", "orthology_type", "source", "confidence_score",
              "is_one_to_one", "in_mouse_model_gene_list"]
with open(raw_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=raw_fields)
    w.writeheader()
    for r in raw_rows:
        w.writerow({k: r[k] for k in raw_fields})
print(f"Saved raw orthologs: {raw_path}")

# 4. Filter to one-to-one orthologs
one2one = [r for r in raw_rows if r["is_one_to_one"] == "1"]
print(f"One-to-one orthologs: {len(one2one)}")

o2o_path = os.path.join(OUTDIR, "human_to_mouse_one_to_one_orthologs.csv")
with open(o2o_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=raw_fields)
    w.writeheader()
    for r in one2one:
        w.writerow({k: r[k] for k in raw_fields})
print(f"Saved one-to-one orthologs: {o2o_path}")

# 5. Model-overlap: one-to-one AND in mouse model gene list, no duplicates
model_candidates = [r for r in one2one if r["in_mouse_model_gene_list"] == "1"]
print(f"One-to-one in model gene list (before dedup): {len(model_candidates)}")

# Remove ambiguous: if multiple human genes map to same mouse gene, or vice versa
human_to_mouse = defaultdict(list)
mouse_to_human = defaultdict(list)
for r in model_candidates:
    h, m = r["human_gene_symbol"], r["mouse_gene_symbol"]
    human_to_mouse[h].append(m)
    mouse_to_human[m].append(h)

ambig_human = {h for h, ms in human_to_mouse.items() if len(set(ms)) > 1}
ambig_mouse = {m for m, hs in mouse_to_human.items() if len(set(hs)) > 1}
print(f"Ambiguous human symbols (>1 mouse target): {len(ambig_human)}")
print(f"Ambiguous mouse symbols (>1 human source): {len(ambig_mouse)}")

# Keep only unambiguous
model_overlap = []
seen_human = set()
seen_mouse = set()
for r in model_candidates:
    h, m = r["human_gene_symbol"], r["mouse_gene_symbol"]
    if h in ambig_human or m in ambig_mouse:
        continue
    if h in seen_human or m in seen_mouse:
        continue
    seen_human.add(h)
    seen_mouse.add(m)
    model_overlap.append({
        "human_gene_symbol": h,
        "mouse_model_gene": m,
        "model_gene_index": mouse_gene_index[m],
        "human_ensembl_gene_id": r["human_ensembl_gene_id"],
        "mouse_ensembl_gene_id": r["mouse_ensembl_gene_id"],
        "mapping_source": "MGI_one2one",
        "mapping_status": "mapped",
    })

model_overlap.sort(key=lambda x: x["model_gene_index"])

overlap_path = os.path.join(OUTDIR, "human_to_mouse_model_overlap.csv")
overlap_fields = ["human_gene_symbol", "mouse_model_gene", "model_gene_index",
                  "human_ensembl_gene_id", "mouse_ensembl_gene_id",
                  "mapping_source", "mapping_status"]
with open(overlap_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=overlap_fields)
    w.writeheader()
    w.writerows(model_overlap)
print(f"Saved model overlap: {overlap_path} ({len(model_overlap)} genes)")

# 6. Summary stats
orth_type_counts = Counter(r["orthology_type"] for r in raw_rows)
summary = [
    ("n_mouse_model_genes", len(mouse_genes)),
    ("n_raw_human_mouse_ortholog_rows", len(raw_rows)),
    ("n_one_to_one_ortholog_pairs", len(one2one)),
    ("n_mouse_model_genes_with_human_one_to_one_ortholog", len(model_overlap)),
    ("n_ambiguous_human_removed", len(ambig_human)),
    ("n_ambiguous_mouse_removed", len(ambig_mouse)),
    ("pct_model_genes_mapped", f"{100*len(model_overlap)/len(mouse_genes):.1f}"),
    ("orthology_type_counts", str(dict(orth_type_counts))),
]

summary_path = os.path.join(REPORT_DIR, "ortholog_mapping_summary.csv")
with open(summary_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "value"])
    for k, v in summary:
        w.writerow([k, v])
print(f"\nSaved summary: {summary_path}")

print("\n=== Ortholog Mapping Summary ===")
for k, v in summary:
    print(f"  {k}: {v}")

# Quick sanity: check known orthologs
known_pairs = [("MALAT1", "Malat1"), ("ACTB", "Actb"), ("RPL3", "Rpl3"), ("GAPDH", "Gapdh")]
print("\nSanity check - known ortholog pairs:")
overlap_dict = {r["human_gene_symbol"]: r["mouse_model_gene"] for r in model_overlap}
for human, mouse in known_pairs:
    if human in overlap_dict:
        mapped = overlap_dict[human]
        status = "OK" if mapped == mouse else f"MISMATCH (got {mapped})"
        print(f"  {human} -> {mapped}: {status}")
    else:
        print(f"  {human}: NOT FOUND in overlap")
