#!/bin/bash
#SBATCH --job-name=lat_var
#SBATCH --output=jobs/logs/slurm-%j_lat_var.out
#SBATCH --error=jobs/logs/slurm-%j_lat_var.err
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

def load_latent_by_ct(dump_path):
    with open(dump_path) as f:
        data = json.load(f)
    cell_mu = defaultdict(list)
    nuc_mu = defaultdict(list)
    cell_logvar = defaultdict(list)
    nuc_logvar = defaultdict(list)
    for ct, batches in data.items():
        for bid, bd in batches.items():
            if "cell_mu" in bd:
                cell_mu[ct].append(np.array(bd["cell_mu"], dtype=np.float32))
            if "nuclei_mu" in bd:
                nuc_mu[ct].append(np.array(bd["nuclei_mu"], dtype=np.float32))
            if "cell_log_var" in bd:
                cell_logvar[ct].append(np.array(bd["cell_log_var"], dtype=np.float32))
            if "nuclei_log_var" in bd:
                nuc_logvar[ct].append(np.array(bd["nuclei_log_var"], dtype=np.float32))
    for d in [cell_mu, nuc_mu, cell_logvar, nuc_logvar]:
        for ct in d:
            d[ct] = np.concatenate(d[ct], axis=0)
    del data; gc.collect()
    return dict(cell_mu), dict(nuc_mu), dict(cell_logvar), dict(nuc_logvar)

def within_ct_latent_var(mu_by_ct, name):
    """Per-dim variance of mu within each cell type, then mean across dims."""
    ct_vars = {}
    for ct, mat in mu_by_ct.items():
        if mat.shape[0] < 5:
            continue
        # variance of mu across cells within CT, per latent dim, then mean across dims
        per_dim_var = np.var(mat, axis=0)  # shape: (32,)
        ct_vars[ct] = per_dim_var.mean()
    overall = np.mean(list(ct_vars.values()))
    print(f"\n{name}: mean within-CT per-dim mu variance = {overall:.6f}", flush=True)
    for ct in sorted(ct_vars.keys()):
        print(f"  {ct:15s}: {ct_vars[ct]:.6f} (n={mu_by_ct[ct].shape[0]})", flush=True)
    return ct_vars, overall

def mean_logvar_by_ct(logvar_by_ct, name):
    """Mean log_var within each cell type (indicates posterior width)."""
    ct_means = {}
    for ct, mat in logvar_by_ct.items():
        if mat.shape[0] < 5:
            continue
        ct_means[ct] = mat.mean()
    overall = np.mean(list(ct_means.values()))
    print(f"\n{name}: mean log_var = {overall:.4f} (exp={np.exp(overall):.4f})", flush=True)
    for ct in sorted(ct_means.keys()):
        print(f"  {ct:15s}: mean_logvar={ct_means[ct]:.4f}, exp={np.exp(ct_means[ct]):.4f}", flush=True)
    return ct_means, overall

# Load normloss e0 (baseline)
print("=== NORMLOSS E0 (baseline, stage 1 output) ===", flush=True)
cm0, nm0, clv0, nlv0 = load_latent_by_ct("run/run_2026-05-08-00-23-35_3604390/epoch_dump/train_epoch_0.json")
bl_cell_var, bl_cell_overall = within_ct_latent_var(cm0, "cell_mu e0")
bl_nuc_var, bl_nuc_overall = within_ct_latent_var(nm0, "nuc_mu e0")
bl_cell_lv, _ = mean_logvar_by_ct(clv0, "cell_log_var e0")
bl_nuc_lv, _ = mean_logvar_by_ct(nlv0, "nuc_log_var e0")
del cm0, nm0, clv0, nlv0; gc.collect()

# Load normloss e599
print("\n=== NORMLOSS E599 ===", flush=True)
cm599, nm599, clv599, nlv599 = load_latent_by_ct("run/run_2026-05-08-00-23-35_3604390/epoch_dump/train_epoch_599.json")
nl_cell_var, nl_cell_overall = within_ct_latent_var(cm599, "cell_mu e599")
nl_nuc_var, nl_nuc_overall = within_ct_latent_var(nm599, "nuc_mu e599")
nl_cell_lv, _ = mean_logvar_by_ct(clv599, "cell_log_var e599")
nl_nuc_lv, _ = mean_logvar_by_ct(nlv599, "nuc_log_var e599")
del cm599, nm599, clv599, nlv599; gc.collect()

# Load old cond e599
print("\n=== OLD COND E599 ===", flush=True)
old_cm, old_nm, old_clv, old_nlv = load_latent_by_ct("run/run_2026-04-02-02-02-35_3345839/epoch_dump/train_epoch_599.json")
old_cell_var, old_cell_overall = within_ct_latent_var(old_cm, "cell_mu e599")
old_nuc_var, old_nuc_overall = within_ct_latent_var(old_nm, "nuc_mu e599")
old_cell_lv, _ = mean_logvar_by_ct(old_clv, "cell_log_var e599")
old_nuc_lv, _ = mean_logvar_by_ct(old_nlv, "nuc_log_var e599")
del old_cm, old_nm, old_clv, old_nlv; gc.collect()

# Summary
print("\n" + "="*70, flush=True)
print("LATENT WITHIN-CT VARIANCE SUMMARY", flush=True)
print("="*70, flush=True)
print(f"{'Source':<40} {'cell mu var':>12} {'nuc mu var':>12} {'ratio nuc/cell':>15}", flush=True)
print(f"{'Normloss e0 (baseline)':<40} {bl_cell_overall:>12.6f} {bl_nuc_overall:>12.6f} {bl_nuc_overall/bl_cell_overall:>15.4f}", flush=True)
print(f"{'Normloss e599':<40} {nl_cell_overall:>12.6f} {nl_nuc_overall:>12.6f} {nl_nuc_overall/nl_cell_overall:>15.4f}", flush=True)
print(f"{'Old cond e599':<40} {old_cell_overall:>12.6f} {old_nuc_overall:>12.6f} {old_nuc_overall/old_cell_overall:>15.4f}", flush=True)

# Per-CT comparison
cts = sorted(bl_cell_var.keys())
print(f"\nPER-CT latent mu variance:", flush=True)
print(f"{'CT':15s} {'BL cell':>10} {'BL nuc':>10} {'NL cell':>10} {'NL nuc':>10} {'NL nuc/cell':>12} {'Old nuc/cell':>12}", flush=True)
for ct in cts:
    bc = bl_cell_var.get(ct, 0)
    bn = bl_nuc_var.get(ct, 0)
    nc = nl_cell_var.get(ct, 0)
    nn = nl_nuc_var.get(ct, 0)
    nl_ratio = nn / nc if nc > 0 else 0
    oc = old_cell_var.get(ct, 0)
    on = old_nuc_var.get(ct, 0)
    old_ratio = on / oc if oc > 0 else 0
    print(f"{ct:15s} {bc:10.6f} {bn:10.6f} {nc:10.6f} {nn:10.6f} {nl_ratio:12.4f} {old_ratio:12.4f}", flush=True)

# ============================================================
# PLOTS
# ============================================================
outdir = "notes/experiments/s1_normalized_loss_probe"
os.makedirs(outdir, exist_ok=True)

# Plot 1: Latent within-CT variance bar chart (cell vs nuclei, baseline vs e599)
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(cts))
w = 0.18

bars = [
    (bl_cell_var, "Baseline cell mu (e0)", "#2196F3"),
    (bl_nuc_var, "Baseline nuc mu (e0)", "#90CAF9"),
    (nl_cell_var, "Normloss cell mu (e599)", "#4CAF50"),
    (nl_nuc_var, "Normloss nuc mu (e599)", "#A5D6A7"),
    (old_nuc_var, "Old cond nuc mu (e599)", "#F44336"),
]
for i, (vals, label, color) in enumerate(bars):
    heights = [vals.get(ct, 0) for ct in cts]
    ax.bar(x + (i - 2)*w, heights, w, label=label, color=color, edgecolor='white', linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(cts, rotation=45, ha='right', fontsize=9)
ax.set_ylabel("Mean within-CT per-dim mu variance")
ax.set_title("Latent Space: Within-Cell-Type Variance of mu")
ax.legend(fontsize=7, loc='upper right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{outdir}/latent_within_ct_variance.png", dpi=150)
plt.close()
print(f"\nSaved {outdir}/latent_within_ct_variance.png", flush=True)

# Plot 2: Latent nuc/cell ratio vs decoded cross_dec/cell ratio
# This shows whether decoder amplifies or reduces the latent gap
# Load decoded cross_dec ratios from previous results (hardcoded from job output)
dec_xd_ratios = {
    'Astro': 0.3322, 'Endo': 3.2130, 'L2/3 IT': 0.3206, 'L5 ET': 0.2850,
    'L5 IT': 0.4892, 'L5/6 NP': 0.4693, 'L6 CT': 0.4284, 'L6 IT': 0.6274,
    'L6b': 0.4504, 'Lamp5': 0.3385, 'Macrophage': 0.0363, 'OPC': 1.0914,
    'Oligo': 0.3053, 'Pvalb': 0.3681, 'Sncg': 0.2720, 'Sst': 0.3469, 'Vip': 0.2656
}

fig, ax = plt.subplots(figsize=(8, 8))
lat_ratios = []
dec_ratios = []
labels = []
for ct in cts:
    nc = nl_cell_var.get(ct, 0)
    nn = nl_nuc_var.get(ct, 0)
    if nc > 0 and ct in dec_xd_ratios:
        lr = nn / nc
        dr = dec_xd_ratios[ct]
        lat_ratios.append(lr)
        dec_ratios.append(dr)
        labels.append(ct)
        ax.scatter(lr, dr, s=60, zorder=3)
        ax.annotate(ct, (lr, dr), fontsize=7, ha='left', va='bottom')

lim = max(max(lat_ratios), max(dec_ratios)) * 1.1
ax.plot([0, lim], [0, lim], 'k--', alpha=0.4, label='y=x (decoder preserves)')
ax.set_xlabel("Latent: nuc mu var / cell mu var (within-CT)")
ax.set_ylabel("Decoded: cross_dec var / baseline cell var (within-CT)")
ax.set_title("Variance Loss: Latent vs Decoded (normloss)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.set_xlim(0, min(lim, 4))
ax.set_ylim(0, min(lim, 4))
plt.tight_layout()
plt.savefig(f"{outdir}/latent_vs_decoded_variance_ratio.png", dpi=150)
plt.close()
print(f"Saved {outdir}/latent_vs_decoded_variance_ratio.png", flush=True)

# Plot 3: Mean log_var comparison (posterior width)
fig, ax = plt.subplots(figsize=(14, 5))
x = np.arange(len(cts))
w = 0.2
bars_lv = [
    (nl_cell_lv, "Normloss cell log_var", "#4CAF50"),
    (nl_nuc_lv, "Normloss nuc log_var", "#A5D6A7"),
    (old_cell_lv, "Old cond cell log_var", "#F44336"),
    (old_nuc_lv, "Old cond nuc log_var", "#FFCDD2"),
]
for i, (vals, label, color) in enumerate(bars_lv):
    heights = [vals.get(ct, 0) for ct in cts]
    ax.bar(x + (i - 1.5)*w, heights, w, label=label, color=color, edgecolor='white', linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(cts, rotation=45, ha='right', fontsize=9)
ax.set_ylabel("Mean log_var")
ax.set_title("Posterior Width (log_var) by Cell Type and Modality")
ax.legend(fontsize=7)
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{outdir}/latent_logvar_by_ct.png", dpi=150)
plt.close()
print(f"Saved {outdir}/latent_logvar_by_ct.png", flush=True)

print("\nDone!", flush=True)
PYEOF
