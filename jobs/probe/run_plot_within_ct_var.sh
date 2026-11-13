#!/bin/bash
#SBATCH --job-name=plot_ctvar
#SBATCH --output=jobs/logs/slurm-%j_plot_ctvar.out
#SBATCH --error=jobs/logs/slurm-%j_plot_ctvar.err
#SBATCH --partition=pi_gerstein
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00

source activate impute2

python3 << 'PYEOF'
import json, numpy as np, gc, os
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_by_celltype(dump_path):
    with open(dump_path) as f:
        data = json.load(f)
    cell_by_ct = defaultdict(list)
    nuc_by_ct = defaultdict(list)
    xd_by_ct = defaultdict(list)
    for ct, batches in data.items():
        for bid, bd in batches.items():
            if "cell_decoded" in bd:
                cell_by_ct[ct].append(np.array(bd["cell_decoded"], dtype=np.float32))
            if "nuclei_decoded" in bd:
                nuc_by_ct[ct].append(np.array(bd["nuclei_decoded"], dtype=np.float32))
            if "cross_decoded" in bd:
                xd_by_ct[ct].append(np.array(bd["cross_decoded"], dtype=np.float32))
    for d in [cell_by_ct, nuc_by_ct, xd_by_ct]:
        for ct in d:
            d[ct] = np.concatenate(d[ct], axis=0)
    del data; gc.collect()
    return dict(cell_by_ct), dict(nuc_by_ct), dict(xd_by_ct)

def within_ct_per_gene_var(data_by_ct):
    result = {}
    for ct, mat in data_by_ct.items():
        if mat.shape[0] < 5:
            continue
        result[ct] = np.var(mat, axis=0).mean()
    return result

# Load data
print("Loading normloss e0 (baseline)...", flush=True)
bl_cell, bl_nuc, bl_xd = load_by_celltype("run/run_2026-05-08-00-23-35_3604390/epoch_dump/train_epoch_0.json")
bl_cell_v = within_ct_per_gene_var(bl_cell)
bl_nuc_v = within_ct_per_gene_var(bl_nuc)
del bl_cell, bl_nuc, bl_xd; gc.collect()

print("Loading normloss e599...", flush=True)
nl_cell, nl_nuc, nl_xd = load_by_celltype("run/run_2026-05-08-00-23-35_3604390/epoch_dump/train_epoch_599.json")
nl_cell_v = within_ct_per_gene_var(nl_cell)
nl_nuc_v = within_ct_per_gene_var(nl_nuc)
nl_xd_v = within_ct_per_gene_var(nl_xd)
del nl_cell, nl_nuc, nl_xd; gc.collect()

print("Loading old cond e599...", flush=True)
old_cell, old_nuc, old_xd = load_by_celltype("run/run_2026-04-02-02-02-35_3345839/epoch_dump/train_epoch_599.json")
old_cell_v = within_ct_per_gene_var(old_cell)
old_nuc_v = within_ct_per_gene_var(old_nuc)
old_xd_v = within_ct_per_gene_var(old_xd)
del old_cell, old_nuc, old_xd; gc.collect()

# Sort cell types by baseline cell variance (descending)
cts = sorted(bl_cell_v.keys(), key=lambda c: bl_cell_v[c], reverse=True)

outdir = "notes/experiments/s1_normalized_loss_probe"
os.makedirs(outdir, exist_ok=True)

# ============================================================
# PLOT 1: Absolute within-CT per-gene variance by cell type
# ============================================================
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(cts))
w = 0.15

bars = [
    (bl_cell_v, "Baseline cell (S1 dec, e0)", "#2196F3"),
    (bl_nuc_v, "Baseline nuc (S1 dec, e0)", "#90CAF9"),
    (nl_cell_v, "Normloss dec_cell (e599)", "#4CAF50"),
    (nl_nuc_v, "Normloss dec_nuc (e599)", "#A5D6A7"),
    (nl_xd_v, "Normloss cross_dec (e599)", "#FF9800"),
    (old_xd_v, "Old cond cross_dec (e599)", "#F44336"),
]

for i, (vals, label, color) in enumerate(bars):
    heights = [vals.get(ct, 0) for ct in cts]
    ax.bar(x + (i - 2.5)*w, heights, w, label=label, color=color, edgecolor='white', linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(cts, rotation=45, ha='right', fontsize=9)
ax.set_ylabel("Mean within-CT per-gene variance")
ax.set_title("Within-Cell-Type Variance: Absolute Values")
ax.legend(fontsize=7, loc='upper right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{outdir}/within_ct_variance_absolute.png", dpi=150)
plt.close()
print(f"Saved {outdir}/within_ct_variance_absolute.png", flush=True)

# ============================================================
# PLOT 2: Variance ratio (cross_dec / baseline cell) per CT
# ============================================================
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(cts))
w = 0.35

nl_ratios = [nl_xd_v.get(ct, 0) / bl_cell_v[ct] for ct in cts]
old_ratios = [old_xd_v.get(ct, 0) / bl_cell_v[ct] for ct in cts]

ax.bar(x - w/2, nl_ratios, w, label="Normloss cross_dec", color="#FF9800", edgecolor='white')
ax.bar(x + w/2, old_ratios, w, label="Old cond cross_dec", color="#F44336", edgecolor='white')
ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label="Perfect retention")
ax.axhline(y=np.mean(nl_ratios), color="#FF9800", linestyle=':', linewidth=1, alpha=0.7)
ax.axhline(y=np.mean(old_ratios), color="#F44336", linestyle=':', linewidth=1, alpha=0.7)

ax.set_xticks(x)
ax.set_xticklabels(cts, rotation=45, ha='right', fontsize=9)
ax.set_ylabel("Variance ratio (cross_dec / baseline cell)")
ax.set_title("Within-Cell-Type Variance Retention in Cross-Imputation")
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, max(max(nl_ratios), max(old_ratios)) * 1.15)

# Add mean annotations
ax.text(len(cts)-1, np.mean(nl_ratios)+0.02, f"mean={np.mean(nl_ratios):.3f}", color="#FF9800", fontsize=8, ha='right')
ax.text(len(cts)-1, np.mean(old_ratios)-0.04, f"mean={np.mean(old_ratios):.3f}", color="#F44336", fontsize=8, ha='right')

plt.tight_layout()
plt.savefig(f"{outdir}/within_ct_variance_ratio.png", dpi=150)
plt.close()
print(f"Saved {outdir}/within_ct_variance_ratio.png", flush=True)

# ============================================================
# PLOT 3: Stacked comparison — baseline vs cross_dec per CT
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left: normloss
for i, ct in enumerate(cts):
    ax1.scatter(bl_cell_v[ct], nl_xd_v.get(ct, 0), s=60, zorder=3)
    ax1.annotate(ct, (bl_cell_v[ct], nl_xd_v.get(ct, 0)), fontsize=7, ha='left', va='bottom')

lim = max(max(bl_cell_v.values()), max(nl_xd_v.values())) * 1.1
ax1.plot([0, lim], [0, lim], 'k--', alpha=0.4, label='y=x (perfect)')
ax1.set_xlabel("Baseline cell within-CT var")
ax1.set_ylabel("Cross-dec within-CT var")
ax1.set_title("Normloss: cross_dec vs baseline")
ax1.legend(fontsize=8)
ax1.set_xlim(0, lim)
ax1.set_ylim(0, lim)
ax1.grid(alpha=0.3)

# Right: old cond
for i, ct in enumerate(cts):
    ax2.scatter(bl_cell_v[ct], old_xd_v.get(ct, 0), s=60, zorder=3, color='#F44336')
    ax2.annotate(ct, (bl_cell_v[ct], old_xd_v.get(ct, 0)), fontsize=7, ha='left', va='bottom')

ax2.plot([0, lim], [0, lim], 'k--', alpha=0.4, label='y=x (perfect)')
ax2.set_xlabel("Baseline cell within-CT var")
ax2.set_ylabel("Cross-dec within-CT var")
ax2.set_title("Old cond: cross_dec vs baseline")
ax2.legend(fontsize=8)
ax2.set_xlim(0, lim)
ax2.set_ylim(0, lim)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{outdir}/within_ct_variance_scatter.png", dpi=150)
plt.close()
print(f"Saved {outdir}/within_ct_variance_scatter.png", flush=True)

print("\nDone!", flush=True)
PYEOF
