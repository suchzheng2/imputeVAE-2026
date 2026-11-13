#!/usr/bin/env python3
"""Part D: Prepare model-ready human nuclei matrix in mouse model gene order.

Memory-safe: uses pandas usecols to read ONLY the ~8,400 ortholog columns
from the 50K-column CSV. Processes in chunks.
"""

import csv
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy import sparse
from collections import Counter

ROOT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE"
DATA_DIR = os.path.join(ROOT, "validation/human_m1_transfer/external/allen_human_m1_10x")
ORTH_DIR = os.path.join(ROOT, "validation/human_m1_transfer/orthologs")
PROC_DIR = os.path.join(ROOT, "validation/human_m1_transfer/processed")
REPORT_DIR = os.path.join(ROOT, "validation/human_m1_transfer/reports")
GENE_LIST = os.path.join(ROOT, "data/training_gene_list_v3.csv")

os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# 1. Load mouse model gene list
with open(GENE_LIST) as f:
    mouse_genes = [line.strip() for line in f if line.strip()]
n_model_genes = len(mouse_genes)
print(f"Mouse model genes: {n_model_genes}")

# 2. Load ortholog mapping
ortholog_path = os.path.join(ORTH_DIR, "human_to_mouse_model_overlap.csv")
human_to_mouse = {}  # human_symbol -> (mouse_symbol, model_index)
with open(ortholog_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        human_to_mouse[row["human_gene_symbol"]] = (
            row["mouse_model_gene"],
            int(row["model_gene_index"]),
        )
print(f"Ortholog mappings loaded: {len(human_to_mouse)}")

# 3. Read matrix header to find which columns to load
print("Reading matrix.csv header...")
matrix_path = os.path.join(DATA_DIR, "matrix.csv")

with open(matrix_path) as f:
    header = f.readline().strip().split(",")

sample_col = header[0]
gene_names = header[1:]
n_matrix_genes = len(gene_names)
print(f"Matrix columns: {len(header)} (1 sample + {n_matrix_genes} genes)")

# Find which human genes in the matrix have orthologs
cols_to_load = [sample_col]  # always load sample name
human_col_to_model_idx = {}  # column_name -> model_gene_index
for gene in gene_names:
    if gene in human_to_mouse:
        cols_to_load.append(gene)
        _, model_idx = human_to_mouse[gene]
        human_col_to_model_idx[gene] = model_idx

n_mapped = len(human_col_to_model_idx)
print(f"Human genes in matrix with model ortholog: {n_mapped}")
print(f"Columns to load: {len(cols_to_load)} (of {len(header)} total)")

# 4. Read matrix in chunks using pandas usecols
print(f"\nReading matrix.csv with usecols (only {len(cols_to_load)} columns)...")
t0 = time.time()

chunk_size = 10000
sparse_rows = []
sparse_cols = []
sparse_vals = []
sample_names = []
n_genes_detected = []
total_counts_list = []
row_offset = 0

for chunk in pd.read_csv(matrix_path, usecols=cols_to_load, chunksize=chunk_size,
                          dtype={sample_col: str}, low_memory=True):
    # Extract sample names
    samples = chunk[sample_col].tolist()
    sample_names.extend(samples)

    # Get gene columns only (exclude sample_col)
    gene_cols = [c for c in chunk.columns if c != sample_col]
    gene_data = chunk[gene_cols].values.astype(np.float32)

    # Model indices for these columns
    model_indices = np.array([human_col_to_model_idx[c] for c in gene_cols], dtype=np.int32)

    # Find non-zeros
    nz_rows, nz_cols = np.nonzero(gene_data)
    nz_vals = gene_data[nz_rows, nz_cols]

    # Adjust row indices to global offset
    sparse_rows.append(nz_rows + row_offset)
    sparse_cols.append(model_indices[nz_cols])
    sparse_vals.append(nz_vals)

    # Per-nucleus stats (on mapped genes only)
    n_genes_detected.extend((gene_data > 0).sum(axis=1).tolist())
    total_counts_list.extend(gene_data.sum(axis=1).tolist())

    row_offset += len(samples)
    elapsed = time.time() - t0
    print(f"  {row_offset:,} rows ({elapsed:.0f}s, ~{row_offset*len(header)*4/1e9:.1f}GB saved)")

    del chunk, gene_data  # free memory

elapsed = time.time() - t0
print(f"Done reading: {row_offset:,} rows in {elapsed:.0f}s")

# 5. Build sparse matrix
print("Building sparse matrix...")
all_rows = np.concatenate(sparse_rows)
all_cols = np.concatenate(sparse_cols)
all_vals = np.concatenate(sparse_vals)
del sparse_rows, sparse_cols, sparse_vals  # free memory

raw_sparse = sparse.csr_matrix(
    (all_vals, (all_rows, all_cols)),
    shape=(row_offset, n_model_genes),
    dtype=np.float32,
)
del all_rows, all_cols, all_vals

print(f"Sparse matrix shape: {raw_sparse.shape}")
print(f"Non-zeros: {raw_sparse.nnz:,}")
print(f"Density: {raw_sparse.nnz / (raw_sparse.shape[0] * raw_sparse.shape[1]):.4f}")

# 6. Check expression scale
sample_data = raw_sparse[:100].toarray()
nonzero_vals = sample_data[sample_data > 0]
max_val = nonzero_vals.max()
min_nonzero = nonzero_vals.min()
has_decimals = np.any(nonzero_vals != nonzero_vals.astype(int))

print(f"\nExpression scale check:")
print(f"  Max value (100 rows): {max_val}")
print(f"  Min non-zero: {min_nonzero}")
print(f"  Has decimals: {has_decimals}")

if not has_decimals and min_nonzero >= 1.0:
    scale = "raw_counts"
    print("  -> RAW COUNTS confirmed")
else:
    scale = "possibly_normalized"
    print("  -> Check README for normalization details")

# 7. Save raw counts sparse
raw_npz_path = os.path.join(PROC_DIR, "human_m1_nuclei_raw_counts.npz")
sparse.save_npz(raw_npz_path, raw_sparse)
print(f"\nSaved raw counts: {raw_npz_path}")

# 8. Apply log1p normalization (matching model training)
print("Applying log1p normalization...")
log1p_sparse = raw_sparse.copy()
log1p_sparse.data = np.log1p(log1p_sparse.data)

npz_path = os.path.join(PROC_DIR, "human_m1_nuclei_mouse_model_order.npz")
sparse.save_npz(npz_path, log1p_sparse)
print(f"Saved log1p matrix: {npz_path}")
del log1p_sparse

# 9. Available gene mask
gene_mask = np.zeros(n_model_genes, dtype=bool)
mapped_model_indices = set(human_col_to_model_idx.values())
for idx in mapped_model_indices:
    gene_mask[idx] = True
mask_path = os.path.join(PROC_DIR, "human_m1_available_gene_mask.npy")
np.save(mask_path, gene_mask)
print(f"Saved gene mask: {mask_path} ({gene_mask.sum()} of {n_model_genes} True)")

# 10. Load metadata for obs
print("Loading metadata for obs file...")
meta_path = os.path.join(DATA_DIR, "metadata.csv")
meta_df = pd.read_csv(meta_path, dtype=str)
meta_by_sample = dict(zip(meta_df["sample_name"], meta_df.to_dict("records")))

# Load cell-type mapping
ct_map_path = os.path.join(PROC_DIR, "human_m1_celltype_mapping.csv")
ct_mapping = {}
with open(ct_map_path) as f:
    for row in csv.DictReader(f):
        ct_mapping[row["human_cell_type_label"]] = (
            row["mapped_broad_class"], row["mapped_mouse_like_class"])

# Save obs
obs_path = os.path.join(PROC_DIR, "human_m1_nuclei_obs.csv")
with open(obs_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["nucleus_id", "donor", "specimen_type", "original_cell_type",
                "original_class", "original_subclass", "mapped_broad_class",
                "mapped_mouse_like_class", "n_genes_detected", "total_counts"])
    for i, sn in enumerate(sample_names):
        meta = meta_by_sample.get(sn, {})
        ct = meta.get("cluster_label", "unknown")
        cls = meta.get("class_label", "unknown")
        subcls = meta.get("subclass_label", "unknown")
        donor = meta.get("external_donor_name_label", "unknown")
        specimen = meta.get("specimen_type", "unknown")
        broad, fine = ct_mapping.get(ct, ("unknown", "unknown"))
        w.writerow([sn, donor, specimen, ct, cls, subcls, broad, fine,
                    n_genes_detected[i], f"{total_counts_list[i]:.0f}"])
print(f"Saved obs: {obs_path}")

# 11. Gene mapping for model order
gene_map_path = os.path.join(PROC_DIR, "human_m1_gene_mapping_for_model_order.csv")
model_idx_to_human = {}
for human, (mouse, midx) in human_to_mouse.items():
    if human in human_col_to_model_idx:
        model_idx_to_human[midx] = human

with open(gene_map_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model_gene_index", "mouse_model_gene", "human_gene_symbol",
                "human_gene_present_in_matrix", "one_to_one_ortholog", "filled_value_status"])
    for i, mg in enumerate(mouse_genes):
        human_sym = model_idx_to_human.get(i, "")
        if i in mapped_model_indices:
            w.writerow([i, mg, human_sym, True, True, "mapped"])
        elif any(m == mg for _, (m, _) in human_to_mouse.items()):
            w.writerow([i, mg, "", False, True, "missing_human_gene"])
        else:
            w.writerow([i, mg, "", False, False, "no_ortholog"])
print(f"Saved gene mapping: {gene_map_path}")

# 12. QC report
n_genes_arr = np.array(n_genes_detected)
total_counts_arr = np.array(total_counts_list)
frac_zero = 1.0 - (raw_sparse.nnz / (raw_sparse.shape[0] * raw_sparse.shape[1]))

broad_counts = Counter()
fine_counts = Counter()
for sn in sample_names:
    meta = meta_by_sample.get(sn, {})
    ct = meta.get("cluster_label", "unknown")
    broad, fine = ct_mapping.get(ct, ("unknown", "unknown"))
    broad_counts[broad] += 1
    fine_counts[fine] += 1

qc_path = os.path.join(REPORT_DIR, "human_m1_input_qc.csv")
with open(qc_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "value"])
    w.writerow(["n_nuclei", row_offset])
    w.writerow(["n_model_genes", n_model_genes])
    w.writerow(["n_transfer_mapped_genes", n_mapped])
    w.writerow(["percent_model_genes_mapped", f"{100*n_mapped/n_model_genes:.1f}"])
    w.writerow(["expression_scale_detected", scale])
    w.writerow(["normalization_applied", "log1p"])
    w.writerow(["median_genes_detected_per_nucleus", f"{np.median(n_genes_arr):.0f}"])
    w.writerow(["mean_genes_detected_per_nucleus", f"{np.mean(n_genes_arr):.0f}"])
    w.writerow(["median_total_counts_per_nucleus", f"{np.median(total_counts_arr):.0f}"])
    w.writerow(["mean_total_counts_per_nucleus", f"{np.mean(total_counts_arr):.0f}"])
    w.writerow(["fraction_zero_in_model_matrix", f"{frac_zero:.4f}"])
    w.writerow(["sparse_matrix_nnz", raw_sparse.nnz])
    w.writerow(["---", "---"])
    w.writerow(["broad_class", "n_nuclei"])
    for cls, n in broad_counts.most_common():
        w.writerow([f"  {cls}", n])
    w.writerow(["---", "---"])
    w.writerow(["mouse_like_class", "n_nuclei"])
    for cls, n in fine_counts.most_common():
        w.writerow([f"  {cls}", n])
print(f"Saved QC: {qc_path}")

print(f"\n{'='*50}")
print(f"FINAL SUMMARY")
print(f"{'='*50}")
print(f"  n_nuclei: {row_offset:,}")
print(f"  n_model_genes: {n_model_genes}")
print(f"  n_transfer_mapped_genes: {n_mapped}")
print(f"  pct_mapped: {100*n_mapped/n_model_genes:.1f}%")
print(f"  expression_scale: {scale}")
print(f"  normalization: log1p(raw_counts)")
print(f"  median_genes_detected: {np.median(n_genes_arr):.0f}")
print(f"  median_total_counts: {np.median(total_counts_arr):.0f}")
print(f"  fraction_zero: {frac_zero:.4f}")
print(f"  sparse_nnz: {raw_sparse.nnz:,}")
print(f"\nBroad class counts:")
for cls, n in broad_counts.most_common():
    print(f"    {cls}: {n:,}")
print(f"\nAll outputs saved to:")
print(f"  {PROC_DIR}/")
print(f"  {REPORT_DIR}/")
