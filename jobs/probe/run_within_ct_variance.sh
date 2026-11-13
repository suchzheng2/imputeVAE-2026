#!/bin/bash
#SBATCH --job-name=ct_var
#SBATCH --output=jobs/logs/slurm-%j_ct_var.out
#SBATCH --error=jobs/logs/slurm-%j_ct_var.err
#SBATCH --partition=pi_gerstein
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00

source activate impute2

python3 << 'PYEOF'
import json, numpy as np, gc
from collections import defaultdict

def load_by_celltype(dump_path):
    """Load epoch dump and group decoded outputs by cell type."""
    with open(dump_path) as f:
        data = json.load(f)

    cell_by_ct = defaultdict(list)
    nuc_by_ct = defaultdict(list)
    xd_by_ct = defaultdict(list)

    for ct, batches in data.items():
        for bid, bd in batches.items():
            n = len(bd.get("cell_type", []))
            if "cell_decoded" in bd:
                cell_by_ct[ct].append(np.array(bd["cell_decoded"], dtype=np.float32))
            if "nuclei_decoded" in bd:
                nuc_by_ct[ct].append(np.array(bd["nuclei_decoded"], dtype=np.float32))
            if "cross_decoded" in bd:
                xd_by_ct[ct].append(np.array(bd["cross_decoded"], dtype=np.float32))

    # Concatenate per cell type
    for ct in cell_by_ct:
        cell_by_ct[ct] = np.concatenate(cell_by_ct[ct], axis=0)
    for ct in nuc_by_ct:
        nuc_by_ct[ct] = np.concatenate(nuc_by_ct[ct], axis=0)
    for ct in xd_by_ct:
        xd_by_ct[ct] = np.concatenate(xd_by_ct[ct], axis=0)

    del data; gc.collect()
    return dict(cell_by_ct), dict(nuc_by_ct), dict(xd_by_ct)

def within_ct_variance(data_by_ct, name):
    """Compute mean within-cell-type per-gene variance."""
    ct_vars = {}
    for ct, mat in data_by_ct.items():
        if mat.shape[0] < 5:
            continue
        per_gene_var = np.var(mat, axis=0)  # variance across cells within this CT
        ct_vars[ct] = per_gene_var.mean()
    overall = np.mean(list(ct_vars.values()))
    print(f"\n{name}: mean within-CT per-gene var = {overall:.6f}", flush=True)
    for ct in sorted(ct_vars.keys()):
        print(f"  {ct:15s}: {ct_vars[ct]:.6f} (n={data_by_ct[ct].shape[0]})", flush=True)
    return ct_vars, overall

# === NORMLOSS MODEL ===
print("=== NORMLOSS MODEL (e599) ===", flush=True)
cell599, nuc599, xd599 = load_by_celltype("run/run_2026-05-08-00-23-35_3604390/epoch_dump/train_epoch_599.json")
nl_cell_vars, nl_cell_overall = within_ct_variance(cell599, "decoded_cell")
nl_nuc_vars, nl_nuc_overall = within_ct_variance(nuc599, "decoded_nuc")
nl_xd_vars, nl_xd_overall = within_ct_variance(xd599, "cross_dec")
del cell599, nuc599, xd599; gc.collect()

print("\n=== NORMLOSS MODEL (e0, baseline) ===", flush=True)
cell0, nuc0, xd0 = load_by_celltype("run/run_2026-05-08-00-23-35_3604390/epoch_dump/train_epoch_0.json")
bl_cell_vars, bl_cell_overall = within_ct_variance(cell0, "decoded_cell_e0 (baseline)")
bl_nuc_vars, bl_nuc_overall = within_ct_variance(nuc0, "decoded_nuc_e0 (baseline)")
bl_xd_vars, bl_xd_overall = within_ct_variance(xd0, "cross_dec_e0 (baseline)")
del cell0, nuc0, xd0; gc.collect()

# === OLD COND MODEL ===
import os
old_run = "run/run_2026-04-02-02-02-35_3345839"
if os.path.exists(f"{old_run}/epoch_dump/train_epoch_599.json"):
    print("\n=== OLD COND MODEL (e599) ===", flush=True)
    old_cell, old_nuc, old_xd = load_by_celltype(f"{old_run}/epoch_dump/train_epoch_599.json")
    old_cell_vars, old_cell_overall = within_ct_variance(old_cell, "decoded_cell")
    old_nuc_vars, old_nuc_overall = within_ct_variance(old_nuc, "decoded_nuc")
    old_xd_vars, old_xd_overall = within_ct_variance(old_xd, "cross_dec")
    del old_cell, old_nuc, old_xd; gc.collect()

# === SUMMARY TABLE ===
print("\n" + "="*70, flush=True)
print("WITHIN-CELL-TYPE VARIANCE SUMMARY", flush=True)
print("="*70, flush=True)
print(f"{'Source':<35} {'Mean within-CT var':>20}", flush=True)
print(f"{'Normloss baseline cell (e0)':.<35} {bl_cell_overall:>20.6f}", flush=True)
print(f"{'Normloss baseline nuc (e0)':.<35} {bl_nuc_overall:>20.6f}", flush=True)
print(f"{'Normloss decoded cell (e599)':.<35} {nl_cell_overall:>20.6f}", flush=True)
print(f"{'Normloss decoded nuc (e599)':.<35} {nl_nuc_overall:>20.6f}", flush=True)
print(f"{'Normloss cross_dec (e599)':.<35} {nl_xd_overall:>20.6f}", flush=True)
if os.path.exists(f"{old_run}/epoch_dump/train_epoch_599.json"):
    print(f"{'Old cond decoded cell (e599)':.<35} {old_cell_overall:>20.6f}", flush=True)
    print(f"{'Old cond decoded nuc (e599)':.<35} {old_nuc_overall:>20.6f}", flush=True)
    print(f"{'Old cond cross_dec (e599)':.<35} {old_xd_overall:>20.6f}", flush=True)

print(f"\nVARIANCE RATIOS (within-CT):", flush=True)
print(f"Normloss cross_dec / baseline cell:  {nl_xd_overall/bl_cell_overall:.4f}", flush=True)
print(f"Normloss dec_cell / baseline cell:   {nl_cell_overall/bl_cell_overall:.4f}", flush=True)
print(f"Normloss dec_nuc / baseline nuc:     {nl_nuc_overall/bl_nuc_overall:.4f}", flush=True)
if os.path.exists(f"{old_run}/epoch_dump/train_epoch_599.json"):
    print(f"Old cond cross_dec / baseline cell:  {old_xd_overall/bl_cell_overall:.4f}", flush=True)
    print(f"Old cond dec_cell / baseline cell:   {old_cell_overall/bl_cell_overall:.4f}", flush=True)

# Per-CT comparison for cross_dec
print(f"\nPER-CELL-TYPE cross_dec var ratio (e599/baseline_cell):", flush=True)
print(f"{'Cell Type':15s} {'Normloss':>10} {'Old cond':>10}", flush=True)
for ct in sorted(nl_xd_vars.keys()):
    nl_ratio = nl_xd_vars[ct] / bl_cell_vars.get(ct, 1)
    old_ratio = old_xd_vars.get(ct, 0) / bl_cell_vars.get(ct, 1) if os.path.exists(f"{old_run}/epoch_dump/train_epoch_599.json") else 0
    print(f"{ct:15s} {nl_ratio:10.4f} {old_ratio:10.4f}", flush=True)

PYEOF
