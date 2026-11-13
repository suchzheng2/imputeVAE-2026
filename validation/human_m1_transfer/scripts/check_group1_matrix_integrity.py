#!/usr/bin/env python3
"""Check integrity of Task Group 1 outputs before inference."""

import os
import sys
import csv
import numpy as np
from scipy import sparse

ROOT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE"
PROC = os.path.join(ROOT, "validation/human_m1_transfer/processed")

errors = []

# 1. log1p matrix
print("=== log1p matrix ===")
mat = sparse.load_npz(os.path.join(PROC, "human_m1_nuclei_mouse_model_order.npz"))
print(f"  shape: {mat.shape}")
print(f"  dtype: {mat.dtype}")
print(f"  nnz: {mat.nnz:,}")
assert mat.shape[1] == 10134, f"Expected 10134 cols, got {mat.shape[1]}"

# 2. raw counts matrix
print("=== raw counts matrix ===")
raw = sparse.load_npz(os.path.join(PROC, "human_m1_nuclei_raw_counts.npz"))
print(f"  shape: {raw.shape}")
print(f"  nnz: {raw.nnz:,}")
assert raw.shape == mat.shape, "Shape mismatch"

# 3. log1p correspondence check (sample)
print("=== log1p correspondence check ===")
row_sample = [0, 100, 1000]
for r in row_sample:
    raw_row = raw[r].toarray().flatten()
    log_row = mat[r].toarray().flatten()
    nz = raw_row > 0
    if nz.any():
        expected = np.log1p(raw_row[nz])
        actual = log_row[nz]
        maxdiff = np.max(np.abs(expected - actual))
        print(f"  row {r}: max |log1p(raw) - mat| = {maxdiff:.2e} (n_nz={nz.sum()})")
        assert maxdiff < 1e-4, f"log1p mismatch at row {r}"

# 4. gene mask
print("=== gene mask ===")
mask = np.load(os.path.join(PROC, "human_m1_available_gene_mask.npy"))
print(f"  shape: {mask.shape}, sum: {mask.sum()}")
assert mask.shape[0] == 10134

# 5. obs
print("=== obs ===")
with open(os.path.join(PROC, "human_m1_nuclei_obs.csv")) as f:
    obs_rows = sum(1 for _ in f) - 1  # minus header
print(f"  rows: {obs_rows}")
assert obs_rows == mat.shape[0], f"Obs rows {obs_rows} != matrix rows {mat.shape[0]}"

# 6. first few nonzero values
print("=== sample values (row 0) ===")
row0 = mat[0].toarray().flatten()
nz_idx = np.nonzero(row0)[0][:5]
for idx in nz_idx:
    print(f"  gene[{idx}] = {row0[idx]:.4f} (raw={raw[0, idx]:.0f})")

print("\n=== ALL CHECKS PASSED ===")
print(f"  Matrix: {mat.shape[0]} nuclei x {mat.shape[1]} genes")
print(f"  Mapped genes: {mask.sum()}")
print(f"  Obs rows: {obs_rows}")
