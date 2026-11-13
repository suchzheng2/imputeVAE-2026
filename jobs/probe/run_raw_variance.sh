#!/bin/bash
#SBATCH --job-name=raw_var
#SBATCH --output=jobs/logs/slurm-%j_raw_var.out
#SBATCH --error=jobs/logs/slurm-%j_raw_var.err
#SBATCH --partition=pi_gerstein
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00

source activate impute2

python3 << 'PYEOF'
import json, numpy as np, gc

def var_stats(mat, name):
    per_cell_var = np.var(mat, axis=1)
    per_gene_var = np.var(mat, axis=0)
    print(f"{name}: n={mat.shape[0]} cells, {mat.shape[1]} genes", flush=True)
    print(f"  Per-cell var:  mean={per_cell_var.mean():.6f}, median={np.median(per_cell_var):.6f}", flush=True)
    print(f"  Per-gene var:  mean={per_gene_var.mean():.6f}, median={np.median(per_gene_var):.6f}", flush=True)
    return per_cell_var.mean(), per_gene_var.mean()

# Use epoch 0 to get stage-1 decoded output as proxy for raw data
# The stage-1 model has ~0.99 correlation to raw, so decoded_cell at epoch 0 ≈ raw cell
dump0 = "run/run_2026-05-08-00-23-35_3604390/epoch_dump/train_epoch_0.json"
print(f"Loading epoch 0 dump for raw proxy...", flush=True)
with open(dump0) as f:
    data = json.load(f)

dec_cell_e0, dec_nuc_e0, xd_e0 = [], [], []
for ct, batches in data.items():
    for bid, bd in batches.items():
        if "cell_decoded" in bd:
            dec_cell_e0.append(np.array(bd["cell_decoded"], dtype=np.float32))
        if "nuclei_decoded" in bd:
            dec_nuc_e0.append(np.array(bd["nuclei_decoded"], dtype=np.float32))
        if "cross_decoded" in bd:
            xd_e0.append(np.array(bd["cross_decoded"], dtype=np.float32))
del data; gc.collect()

print("\n=== EPOCH 0 (stage 1 output, pre-stage-2) ===", flush=True)
rc0 = var_stats(np.concatenate(dec_cell_e0), "decoded_cell_e0 (proxy for raw cell)")
rn0 = var_stats(np.concatenate(dec_nuc_e0), "decoded_nuc_e0 (proxy for raw nuclei)")
xd0 = var_stats(np.concatenate(xd_e0), "cross_dec_e0")
del dec_cell_e0, dec_nuc_e0, xd_e0; gc.collect()

# Epoch 599 values from decoded_variance job
print(f"\n=== VARIANCE RATIOS (normloss, e599 / e0) ===", flush=True)
nl_dc = (0.714495, 0.172320)
nl_dn = (0.290817, 0.045089)
nl_xd = (0.670053, 0.132045)

print(f"decoded_cell e599/e0:   per-cell={nl_dc[0]/rc0[0]:.4f}  per-gene={nl_dc[1]/rc0[1]:.4f}", flush=True)
print(f"decoded_nuc e599/e0:    per-cell={nl_dn[0]/rn0[0]:.4f}  per-gene={nl_dn[1]/rn0[1]:.4f}", flush=True)
print(f"cross_dec e599/e0_cell: per-cell={nl_xd[0]/rc0[0]:.4f}  per-gene={nl_xd[1]/rc0[1]:.4f}", flush=True)

# Old cond model
old_dc = (0.705431, 0.167550)
old_dn = (0.288072, 0.043662)
old_xd = (0.639512, 0.120789)

print(f"\n=== VARIANCE RATIOS (old cond, e599 / e0) ===", flush=True)
print(f"decoded_cell e599/e0:   per-cell={old_dc[0]/rc0[0]:.4f}  per-gene={old_dc[1]/rc0[1]:.4f}", flush=True)
print(f"decoded_nuc e599/e0:    per-cell={old_dn[0]/rn0[0]:.4f}  per-gene={old_dn[1]/rn0[1]:.4f}", flush=True)
print(f"cross_dec e599/e0_cell: per-cell={old_xd[0]/rc0[0]:.4f}  per-gene={old_xd[1]/rc0[1]:.4f}", flush=True)

# Also check old cond epoch 0
import os
old_dump0 = "run/run_2026-04-02-02-02-35_3345839/epoch_dump/train_epoch_0.json"
if os.path.exists(old_dump0):
    print(f"\n=== OLD COND EPOCH 0 ===", flush=True)
    with open(old_dump0) as f:
        data2 = json.load(f)
    dec_cell2, dec_nuc2 = [], []
    for ct, batches in data2.items():
        for bid, bd in batches.items():
            if "cell_decoded" in bd:
                dec_cell2.append(np.array(bd["cell_decoded"], dtype=np.float32))
            if "nuclei_decoded" in bd:
                dec_nuc2.append(np.array(bd["nuclei_decoded"], dtype=np.float32))
    del data2; gc.collect()
    rc0_old = var_stats(np.concatenate(dec_cell2), "old_decoded_cell_e0")
    rn0_old = var_stats(np.concatenate(dec_nuc2), "old_decoded_nuc_e0")
    print(f"\nOld cond ratios (e599/own e0):", flush=True)
    print(f"decoded_cell: per-cell={old_dc[0]/rc0_old[0]:.4f}  per-gene={old_dc[1]/rc0_old[1]:.4f}", flush=True)
    print(f"decoded_nuc:  per-cell={old_dn[0]/rn0_old[0]:.4f}  per-gene={old_dn[1]/rn0_old[1]:.4f}", flush=True)
    print(f"cross_dec:    per-cell={old_xd[0]/rc0_old[0]:.4f}  per-gene={old_xd[1]/rc0_old[1]:.4f}", flush=True)

PYEOF
