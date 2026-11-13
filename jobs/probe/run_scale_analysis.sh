#!/bin/bash
#SBATCH --job-name=scale_ana
#SBATCH --output=jobs/logs/slurm-%j_scale_ana.out
#SBATCH --error=jobs/logs/slurm-%j_scale_ana.err
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

outdir = "notes/experiments/s1_normalized_loss_probe"

# Load baseline (e0) and trained (e599)
print("Loading e0...", flush=True)
bl_cell, bl_nuc, bl_xd = load_by_celltype("run/run_2026-05-08-00-23-35_3604390/epoch_dump/train_epoch_0.json")
print("Loading e599...", flush=True)
nl_cell, nl_nuc, nl_xd = load_by_celltype("run/run_2026-05-08-00-23-35_3604390/epoch_dump/train_epoch_599.json")

# ============================================================
# ANALYSIS 1: Global scale stats
# ============================================================
print("\n=== GLOBAL SCALE STATS ===", flush=True)
def scale_stats(data_by_ct, name):
    all_data = np.concatenate(list(data_by_ct.values()), axis=0)
    zeros = (np.abs(all_data) < 0.01).mean()
    near_zero = (np.abs(all_data) < 0.1).mean()
    print(f"{name}: shape={all_data.shape}", flush=True)
    print(f"  mean={all_data.mean():.4f}, std={all_data.std():.4f}, min={all_data.min():.4f}, max={all_data.max():.4f}", flush=True)
    print(f"  |x|<0.01: {zeros:.4f} ({zeros*100:.1f}%), |x|<0.1: {near_zero:.4f} ({near_zero*100:.1f}%)", flush=True)
    print(f"  median={np.median(all_data):.4f}, p25={np.percentile(all_data, 25):.4f}, p75={np.percentile(all_data, 75):.4f}", flush=True)
    return all_data

print("\n--- Baseline (e0, stage 1 output) ---", flush=True)
bl_c = scale_stats(bl_cell, "baseline cell")
bl_n = scale_stats(bl_nuc, "baseline nuc")
bl_x = scale_stats(bl_xd, "baseline cross_dec")

print("\n--- Normloss (e599) ---", flush=True)
nl_c = scale_stats(nl_cell, "normloss cell")
nl_n = scale_stats(nl_nuc, "normloss nuc")
nl_x = scale_stats(nl_xd, "normloss cross_dec")

# ============================================================
# ANALYSIS 2: Per-cell-type zero/nonzero and scale
# ============================================================
print("\n=== PER-CT ZERO FRACTION AND SCALE ===", flush=True)
cts = sorted(bl_cell.keys())

print(f"\n{'CT':15s} {'BL cell %<0.1':>14} {'BL nuc %<0.1':>14} {'NL cell %<0.1':>14} {'NL nuc %<0.1':>14} {'NL xd %<0.1':>14}", flush=True)
for ct in cts:
    bc_z = (np.abs(bl_cell[ct]) < 0.1).mean()
    bn_z = (np.abs(bl_nuc[ct]) < 0.1).mean()
    nc_z = (np.abs(nl_cell[ct]) < 0.1).mean()
    nn_z = (np.abs(nl_nuc[ct]) < 0.1).mean()
    nx_z = (np.abs(nl_xd[ct]) < 0.1).mean()
    print(f"{ct:15s} {bc_z*100:13.1f}% {bn_z*100:13.1f}% {nc_z*100:13.1f}% {nn_z*100:13.1f}% {nx_z*100:13.1f}%", flush=True)

print(f"\n{'CT':15s} {'BL cell mean':>14} {'BL nuc mean':>14} {'NL cell mean':>14} {'NL nuc mean':>14} {'NL xd mean':>14} {'NL cell std':>14} {'NL nuc std':>14} {'NL xd std':>14}", flush=True)
for ct in cts:
    print(f"{ct:15s} {bl_cell[ct].mean():14.4f} {bl_nuc[ct].mean():14.4f} {nl_cell[ct].mean():14.4f} {nl_nuc[ct].mean():14.4f} {nl_xd[ct].mean():14.4f} {nl_cell[ct].std():14.4f} {nl_nuc[ct].std():14.4f} {nl_xd[ct].std():14.4f}", flush=True)

# ============================================================
# ANALYSIS 3: Per-cell std (across genes) — the actual per-cell variation
# ============================================================
print("\n=== PER-CELL STD (across genes, within CT) ===", flush=True)
print(f"{'CT':15s} {'n':>5} {'BL cell':>10} {'BL nuc':>10} {'NL cell':>10} {'NL nuc':>10} {'NL xd':>10} {'xd/cell':>10}", flush=True)
for ct in cts:
    n = bl_cell[ct].shape[0]
    bc_std = np.std(bl_cell[ct], axis=1).mean()
    bn_std = np.std(bl_nuc[ct], axis=1).mean()
    nc_std = np.std(nl_cell[ct], axis=1).mean()
    nn_std = np.std(nl_nuc[ct], axis=1).mean()
    nx_std = np.std(nl_xd[ct], axis=1).mean()
    print(f"{ct:15s} {n:5d} {bc_std:10.4f} {bn_std:10.4f} {nc_std:10.4f} {nn_std:10.4f} {nx_std:10.4f} {nx_std/nc_std:10.4f}", flush=True)

# ============================================================
# ANALYSIS 4: Pick one cell type, look at a single gene's distribution
# ============================================================
print("\n=== SINGLE CELL TYPE DEEP DIVE: L2/3 IT ===", flush=True)
ct = "L2/3 IT"
bc = bl_cell[ct]  # (964, 10134)
nc = nl_cell[ct]
nn = nl_nuc[ct]
nx = nl_xd[ct]

# Per-gene stats: pick top 10 most variable genes in baseline cell
gene_var = np.var(bc, axis=0)
top_genes = np.argsort(gene_var)[-10:][::-1]

print(f"\nTop 10 most variable genes in baseline cell for {ct}:", flush=True)
print(f"{'Gene idx':>10} {'BL cell mean':>14} {'BL cell std':>14} {'NL cell mean':>14} {'NL cell std':>14} {'NL nuc mean':>14} {'NL nuc std':>14} {'NL xd mean':>14} {'NL xd std':>14}", flush=True)
for g in top_genes:
    print(f"{g:10d} {bc[:, g].mean():14.4f} {bc[:, g].std():14.4f} {nc[:, g].mean():14.4f} {nc[:, g].std():14.4f} {nn[:, g].mean():14.4f} {nn[:, g].std():14.4f} {nx[:, g].mean():14.4f} {nx[:, g].std():14.4f}", flush=True)

# Nonzero stats for these genes
print(f"\nFraction nonzero (>0.1) for same genes:", flush=True)
print(f"{'Gene idx':>10} {'BL cell':>10} {'NL cell':>10} {'NL nuc':>10} {'NL xd':>10}", flush=True)
for g in top_genes:
    print(f"{g:10d} {(bc[:, g] > 0.1).mean():10.3f} {(nc[:, g] > 0.1).mean():10.3f} {(nn[:, g] > 0.1).mean():10.3f} {(nx[:, g] > 0.1).mean():10.3f}", flush=True)

# ============================================================
# PLOT 1: Distribution of per-cell means for L2/3 IT
# ============================================================
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
for ax, (data, label, color) in zip(axes, [
    (bc, "Baseline cell", "#2196F3"),
    (nn, "Normloss nuc (e599)", "#A5D6A7"),
    (nx, "Normloss cross_dec", "#FF9800"),
    (nc, "Normloss cell (e599)", "#4CAF50"),
]):
    cell_means = data.mean(axis=1)
    ax.hist(cell_means, bins=50, color=color, alpha=0.7, edgecolor='white')
    ax.set_title(f"{label}\nmean={cell_means.mean():.3f}, std={cell_means.std():.3f}", fontsize=9)
    ax.set_xlabel("Per-cell mean expression")
    ax.axvline(cell_means.mean(), color='black', linestyle='--', linewidth=0.8)
axes[0].set_ylabel("Count")
plt.suptitle(f"L2/3 IT: Distribution of Per-Cell Mean Expression (n={bc.shape[0]})", fontsize=11)
plt.tight_layout()
plt.savefig(f"{outdir}/scale_percell_mean_L23IT.png", dpi=150)
plt.close()
print(f"\nSaved {outdir}/scale_percell_mean_L23IT.png", flush=True)

# ============================================================
# PLOT 2: Distribution of per-cell std for L2/3 IT
# ============================================================
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
for ax, (data, label, color) in zip(axes, [
    (bc, "Baseline cell", "#2196F3"),
    (nn, "Normloss nuc (e599)", "#A5D6A7"),
    (nx, "Normloss cross_dec", "#FF9800"),
    (nc, "Normloss cell (e599)", "#4CAF50"),
]):
    cell_stds = data.std(axis=1)
    ax.hist(cell_stds, bins=50, color=color, alpha=0.7, edgecolor='white')
    ax.set_title(f"{label}\nmean={cell_stds.mean():.3f}, std={cell_stds.std():.3f}", fontsize=9)
    ax.set_xlabel("Per-cell std expression")
    ax.axvline(cell_stds.mean(), color='black', linestyle='--', linewidth=0.8)
axes[0].set_ylabel("Count")
plt.suptitle(f"L2/3 IT: Distribution of Per-Cell Std (n={bc.shape[0]})", fontsize=11)
plt.tight_layout()
plt.savefig(f"{outdir}/scale_percell_std_L23IT.png", dpi=150)
plt.close()
print(f"\nSaved {outdir}/scale_percell_std_L23IT.png", flush=True)

# ============================================================
# PLOT 3: Histogram of expression values for one highly variable gene
# ============================================================
g = top_genes[0]
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
for ax, (data, label, color) in zip(axes, [
    (bc[:, g], "Baseline cell", "#2196F3"),
    (nn[:, g], "Normloss nuc (e599)", "#A5D6A7"),
    (nx[:, g], "Normloss cross_dec", "#FF9800"),
    (nc[:, g], "Normloss cell (e599)", "#4CAF50"),
]):
    ax.hist(data, bins=50, color=color, alpha=0.7, edgecolor='white')
    ax.set_title(f"{label}\nmean={data.mean():.3f}, std={data.std():.3f}", fontsize=9)
    ax.set_xlabel("Expression")
axes[0].set_ylabel("Count")
plt.suptitle(f"L2/3 IT, Gene {g}: Expression Distribution (n={bc.shape[0]})", fontsize=11)
plt.tight_layout()
plt.savefig(f"{outdir}/scale_gene_dist_L23IT_gene{g}.png", dpi=150)
plt.close()
print(f"Saved {outdir}/scale_gene_dist_L23IT_gene{g}.png", flush=True)

# ============================================================
# PLOT 4: Scatter — for a random subset of cells, cross_dec value vs baseline cell value (one gene)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, g_idx in zip(axes, top_genes[:3]):
    ax.scatter(bc[:, g_idx], nx[:, g_idx], s=5, alpha=0.3, color="#FF9800")
    ax.set_xlabel(f"Baseline cell (gene {g_idx})")
    ax.set_ylabel(f"Cross_dec (gene {g_idx})")
    lim = max(bc[:, g_idx].max(), nx[:, g_idx].max()) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.3)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    r = np.corrcoef(bc[:, g_idx], nx[:, g_idx])[0,1]
    ax.set_title(f"Gene {g_idx}: r={r:.3f}")
plt.suptitle(f"L2/3 IT: Baseline cell vs Cross_dec (per cell, top 3 variable genes)")
plt.tight_layout()
plt.savefig(f"{outdir}/scale_scatter_cell_vs_xd_L23IT.png", dpi=150)
plt.close()
print(f"Saved {outdir}/scale_scatter_cell_vs_xd_L23IT.png", flush=True)

print("\nDone!", flush=True)
PYEOF
