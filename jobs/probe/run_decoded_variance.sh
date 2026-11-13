#!/bin/bash
#SBATCH --job-name=dec_var
#SBATCH --output=jobs/logs/slurm-%j_dec_var.out
#SBATCH --error=jobs/logs/slurm-%j_dec_var.err
#SBATCH --partition=pi_gerstein
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00

source activate impute2

python3 << 'PYEOF'
import json, numpy as np, os, gc

def dense_var_stats(mat, name):
    per_cell_var = np.var(mat, axis=1)
    per_gene_var = np.var(mat, axis=0)
    print(f"{name}: n={mat.shape[0]} cells, {mat.shape[1]} genes", flush=True)
    print(f"  Per-cell var:  mean={per_cell_var.mean():.6f}, median={np.median(per_cell_var):.6f}", flush=True)
    print(f"  Per-gene var:  mean={per_gene_var.mean():.6f}, median={np.median(per_gene_var):.6f}", flush=True)
    return per_cell_var.mean(), per_gene_var.mean()

def process_dump(run, label):
    dump = f"{run}/epoch_dump/train_epoch_599.json"
    print(f"\n=== {label}: {run} ===", flush=True)
    print(f"Loading {dump}...", flush=True)
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
    return dc, dn, xd

# Normloss model
dc, dn, xd = process_dump("run/run_2026-05-08-00-23-35_3604390", "NORMLOSS")

# Old best cond model
old_run = "run/run_2026-04-02-02-02-35_3345839"
if os.path.exists(f"{old_run}/epoch_dump/train_epoch_599.json"):
    dc2, dn2, xd2 = process_dump(old_run, "OLD BEST COND")
else:
    print(f"\nOld cond model dump not found", flush=True)
    dc2, dn2, xd2 = None, None, None

print(f"\n=== SUMMARY (absolute values) ===", flush=True)
print(f"{'Source':<25} {'Per-cell var':>15} {'Per-gene var':>15}", flush=True)
print(f"{'Normloss dec_cell':<25} {dc[0]:>15.6f} {dc[1]:>15.6f}", flush=True)
print(f"{'Normloss dec_nuc':<25} {dn[0]:>15.6f} {dn[1]:>15.6f}", flush=True)
print(f"{'Normloss cross_dec':<25} {xd[0]:>15.6f} {xd[1]:>15.6f}", flush=True)
if dc2:
    print(f"{'Old cond dec_cell':<25} {dc2[0]:>15.6f} {dc2[1]:>15.6f}", flush=True)
    print(f"{'Old cond dec_nuc':<25} {dn2[0]:>15.6f} {dn2[1]:>15.6f}", flush=True)
    print(f"{'Old cond cross_dec':<25} {xd2[0]:>15.6f} {xd2[1]:>15.6f}", flush=True)

PYEOF
