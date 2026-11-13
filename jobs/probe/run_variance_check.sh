#!/bin/bash
#SBATCH --job-name=var_check
#SBATCH --output=jobs/logs/slurm-%j_var_check.out
#SBATCH --error=jobs/logs/slurm-%j_var_check.err
#SBATCH --partition=pi_gerstein
#SBATCH --mem=200G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00

source activate impute2

python3 << 'PYEOF'
import json, numpy as np, sys, os, gc
from scipy import sparse as sp
sys.path.insert(0, os.getcwd())
from imputeVAE.data import load_raw_data, filter_by_protocol
from types import SimpleNamespace

def sparse_var_stats(X, name):
    """Compute variance stats without densifying."""
    n_cells, n_genes = X.shape
    if sp.issparse(X):
        mean_pc = np.asarray(X.mean(axis=1)).flatten()
        sq_mean_pc = np.asarray(X.multiply(X).mean(axis=1)).flatten()
        per_cell_var = sq_mean_pc - mean_pc**2
        mean_pg = np.asarray(X.mean(axis=0)).flatten()
        sq_mean_pg = np.asarray(X.multiply(X).mean(axis=0)).flatten()
        per_gene_var = sq_mean_pg - mean_pg**2
    else:
        per_cell_var = np.var(X, axis=1)
        per_gene_var = np.var(X, axis=0)
    print(f"{name}: n={n_cells} cells, {n_genes} genes")
    print(f"  Per-cell var:  mean={per_cell_var.mean():.6f}, median={np.median(per_cell_var):.6f}")
    print(f"  Per-gene var:  mean={per_gene_var.mean():.6f}, median={np.median(per_gene_var):.6f}")
    return per_cell_var.mean(), per_gene_var.mean()

def dense_var_stats(mat, name):
    per_cell_var = np.var(mat, axis=1)
    per_gene_var = np.var(mat, axis=0)
    print(f"{name}: n={mat.shape[0]} cells, {mat.shape[1]} genes")
    print(f"  Per-cell var:  mean={per_cell_var.mean():.6f}, median={np.median(per_cell_var):.6f}")
    print(f"  Per-gene var:  mean={per_gene_var.mean():.6f}, median={np.median(per_gene_var):.6f}")
    return per_cell_var.mean(), per_gene_var.mean()

# ---- STEP 1: Raw cell data ----
print("=== STEP 1: RAW CELL DATA ===")
args_cell = SimpleNamespace(
    gene_list="data/training_gene_list_v3.csv",
    raw_data_normalization="log1p", use_raw_data=True,
    detection_rate_threshold=1.0, exclude_mitocarta=False, subset=False,
    protocol="v3", batch="c3A"
)
cell_adata = load_raw_data("10X_cells_v2_v3_AIBS/", args_cell)
cell_adata = filter_by_protocol(cell_adata, args_cell, batch_name="c3A")
rc = sparse_var_stats(cell_adata.layers["log1p"], "raw_cell")
del cell_adata; gc.collect()
print(f"Memory freed after cell data\n")

# ---- STEP 2: Raw nuclei data ----
print("=== STEP 2: RAW NUCLEI DATA ===")
args_nuc = SimpleNamespace(
    gene_list="data/training_gene_list_v3.csv",
    raw_data_normalization="log1p", use_raw_data=True,
    detection_rate_threshold=1.0, exclude_mitocarta=False, subset=False,
    protocol="v3", batch="n3A"
)
nuc_adata = load_raw_data("10X_nuclei_v2_v3_AIBS/", args_nuc)
nuc_adata = filter_by_protocol(nuc_adata, args_nuc, batch_name="n3A")
rn = sparse_var_stats(nuc_adata.layers["log1p"], "raw_nuclei")
del nuc_adata; gc.collect()
print(f"Memory freed after nuclei data\n")

# Save raw stats for reuse
print(f"\n=== RAW VARIANCE SUMMARY ===")
print(f"raw_cell:   per-cell={rc[0]:.6f}  per-gene={rc[1]:.6f}")
print(f"raw_nuclei: per-cell={rn[0]:.6f}  per-gene={rn[1]:.6f}")

# ---- STEP 3: Normloss model ----
run = "run/run_2026-05-08-00-23-35_3604390"
dump = f"{run}/epoch_dump/train_epoch_599.json"
print(f"\n=== STEP 3: NORMLOSS MODEL ===")
with open(dump) as f:
    data = json.load(f)

dec_cell, dec_nuc, cross_dec = [], [], []
for ct, batches in data.items():
    for bid, bd in batches.items():
        if "cell_decoded" in bd:
            dec_cell.append(np.array(bd["cell_decoded"], dtype=np.float32))
        if "nuclei_decoded" in bd:
            dec_nuc.append(np.array(bd["nuclei_decoded"], dtype=np.float32))
        if "cross_decoded" in bd:
            cross_dec.append(np.array(bd["cross_decoded"], dtype=np.float32))
del data; gc.collect()

dc = dense_var_stats(np.concatenate(dec_cell), "decoded_cell"); del dec_cell; gc.collect()
dn = dense_var_stats(np.concatenate(dec_nuc), "decoded_nuclei"); del dec_nuc; gc.collect()
xd = dense_var_stats(np.concatenate(cross_dec), "cross_dec"); del cross_dec; gc.collect()

print(f"\n=== VARIANCE RATIOS (normloss) ===")
print(f"decoded_cell / raw_cell:   per-cell={dc[0]/rc[0]:.4f}  per-gene={dc[1]/rc[1]:.4f}")
print(f"decoded_nuc / raw_nuc:     per-cell={dn[0]/rn[0]:.4f}  per-gene={dn[1]/rn[1]:.4f}")
print(f"cross_dec / raw_cell:      per-cell={xd[0]/rc[0]:.4f}  per-gene={xd[1]/rc[1]:.4f}")
print(f"cross_dec / raw_nuc:       per-cell={xd[0]/rn[0]:.4f}  per-gene={xd[1]/rn[1]:.4f}")

# ---- STEP 4: Old best cond model ----
old_run = "run/run_2026-04-02-02-02-35_3345839"
old_dump = f"{old_run}/epoch_dump/train_epoch_599.json"
if os.path.exists(old_dump):
    print(f"\n=== STEP 4: OLD BEST COND MODEL ===")
    with open(old_dump) as f:
        data2 = json.load(f)
    dec_cell2, dec_nuc2, cross_dec2 = [], [], []
    for ct, batches in data2.items():
        for bid, bd in batches.items():
            if "cell_decoded" in bd:
                dec_cell2.append(np.array(bd["cell_decoded"], dtype=np.float32))
            if "nuclei_decoded" in bd:
                dec_nuc2.append(np.array(bd["nuclei_decoded"], dtype=np.float32))
            if "cross_decoded" in bd:
                cross_dec2.append(np.array(bd["cross_decoded"], dtype=np.float32))
    del data2; gc.collect()
    dc2 = dense_var_stats(np.concatenate(dec_cell2), "decoded_cell"); del dec_cell2; gc.collect()
    dn2 = dense_var_stats(np.concatenate(dec_nuc2), "decoded_nuclei"); del dec_nuc2; gc.collect()
    xd2 = dense_var_stats(np.concatenate(cross_dec2), "cross_dec"); del cross_dec2; gc.collect()
    print(f"\n=== VARIANCE RATIOS (old cond) ===")
    print(f"decoded_cell / raw_cell:   per-cell={dc2[0]/rc[0]:.4f}  per-gene={dc2[1]/rc[1]:.4f}")
    print(f"decoded_nuc / raw_nuc:     per-cell={dn2[0]/rn[0]:.4f}  per-gene={dn2[1]/rn[1]:.4f}")
    print(f"cross_dec / raw_cell:      per-cell={xd2[0]/rc[0]:.4f}  per-gene={xd2[1]/rc[1]:.4f}")
    print(f"cross_dec / raw_nuc:       per-cell={xd2[0]/rn[0]:.4f}  per-gene={xd2[1]/rn[1]:.4f}")
else:
    print(f"\nOld cond model dump not found at {old_dump}")

PYEOF
