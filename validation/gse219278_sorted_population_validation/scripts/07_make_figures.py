#!/usr/bin/env python3
"""Step 13: Generate figures for GSE219278 sorted-population validation.

Figures:
  F1. Profile scatter: GSE log2(TPM+1) vs M1 imputed log1p per population
  F2. Reference matching heatmap (3x3 correlation matrix)
  F3. Delta-direction scatter: GSE pop difference vs M1 imputed difference
  F4. Gene-set shift comparison (SynGO vs JASPAR TF vs all genes)
  F5. Donor robustness: bar chart of LOO correlation stability
  F6. Population delta summary: grouped bar chart
"""
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, json

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")
OUT_TAB = os.path.join(BASE, "outputs", "tables")
OUT_FIG = os.path.join(BASE, "outputs", "figures")
os.makedirs(OUT_FIG, exist_ok=True)

# Style
plt.rcParams.update({
    "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
})

# Colors
POP_COLORS = {
    "Neurons": "#8B2252",
    "OligoLineage": "#4682B4",
    "OtherGlia": "#5F9EA0",
}

# Load data
harm = pd.read_csv(os.path.join(PROC, "harmonized_gene_profiles.csv.gz"))
populations = ["Neurons", "OligoLineage", "OtherGlia"]

# =====================================================================
# F1: Profile scatter per population
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for ax, pop in zip(axes, populations):
    gse = harm[f"gse_{pop}_mean_log2_tpm_p1"].values
    m1_imp = harm[f"m1_{pop}_imputed_log1p"].values
    m1_raw = harm[f"m1_{pop}_raw_log1p"].values

    ax.scatter(gse, m1_raw, s=1, alpha=0.15, c="gray", label="Raw nuclei", rasterized=True)
    ax.scatter(gse, m1_imp, s=1, alpha=0.15, c=POP_COLORS[pop], label="Imputed", rasterized=True)

    r_imp, _ = stats.pearsonr(gse, m1_imp)
    r_raw, _ = stats.pearsonr(gse, m1_raw)

    ax.set_xlabel("GSE219278 log2(TPM+1)")
    ax.set_ylabel("Human M1 log1p")
    ax.set_title(f"{pop}")
    ax.text(0.05, 0.95, f"Imputed r={r_imp:.3f}\nRaw r={r_raw:.3f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    ax.legend(loc="lower right", fontsize=8, markerscale=5)

fig.suptitle("GSE219278 Sorted-Population vs ImputeVAE Human M1 Profiles", fontsize=13)
fig.tight_layout()
for ext in ["svg", "png", "pdf"]:
    fig.savefig(os.path.join(OUT_FIG, f"F1_profile_scatter.{ext}"), dpi=200, bbox_inches="tight")
plt.close()
print("F1 saved.")

# =====================================================================
# F2: Reference matching heatmap
# =====================================================================
corr_imp = pd.read_csv(os.path.join(OUT_TAB, "reference_matching_imputed.csv"), index_col=0)
corr_raw = pd.read_csv(os.path.join(OUT_TAB, "reference_matching_raw.csv"), index_col=0)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, corr_df, title in [(axes[0], corr_imp, "Imputed vs GSE"),
                             (axes[1], corr_raw, "Raw Nuclei vs GSE")]:
    mat = corr_df.values
    im = ax.imshow(mat, cmap="RdYlBu_r", vmin=0.6, vmax=1.0, aspect="equal")
    ax.set_xticks(range(3))
    ax.set_xticklabels(populations, rotation=45, ha="right")
    ax.set_yticks(range(3))
    ax.set_yticklabels(populations)
    ax.set_xlabel("Human M1")
    ax.set_ylabel("GSE219278")
    ax.set_title(title)
    for i in range(3):
        for j in range(3):
            color = "white" if mat[i, j] < 0.85 else "black"
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", color=color, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")

fig.suptitle("Reference Matching: Population Cross-Correlation", fontsize=13)
fig.tight_layout()
for ext in ["svg", "png", "pdf"]:
    fig.savefig(os.path.join(OUT_FIG, f"F2_reference_matching.{ext}"), dpi=200, bbox_inches="tight")
plt.close()
print("F2 saved.")

# =====================================================================
# F3: Delta-direction scatter (population differences)
# =====================================================================
comparisons = [
    ("Neurons", "OligoLineage"),
    ("Neurons", "OtherGlia"),
    ("OligoLineage", "OtherGlia"),
]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for ax, (pop_a, pop_b) in zip(axes, comparisons):
    gse_diff = harm[f"gse_{pop_a}_mean_log2_tpm_p1"].values - harm[f"gse_{pop_b}_mean_log2_tpm_p1"].values
    m1_diff = harm[f"m1_{pop_a}_imputed_log1p"].values - harm[f"m1_{pop_b}_imputed_log1p"].values

    ax.scatter(gse_diff, m1_diff, s=1, alpha=0.15, c="#333333", rasterized=True)
    r, _ = stats.pearsonr(gse_diff, m1_diff)
    rho, _ = stats.spearmanr(gse_diff, m1_diff)
    sign_agree = np.mean(np.sign(gse_diff) == np.sign(m1_diff))

    # Add diagonal
    lims = [min(gse_diff.min(), m1_diff.min()), max(gse_diff.max(), m1_diff.max())]
    ax.plot(lims, lims, "r--", alpha=0.5, linewidth=0.8)
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)

    ax.set_xlabel(f"GSE diff: {pop_a} - {pop_b}")
    ax.set_ylabel(f"M1 imputed diff")
    ax.set_title(f"{pop_a} vs {pop_b}")
    ax.text(0.05, 0.95, f"r={r:.3f}\nrho={rho:.3f}\nsign={sign_agree:.1%}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

fig.suptitle("Delta-Direction: GSE Population Differences vs M1 Imputed Differences", fontsize=12)
fig.tight_layout()
for ext in ["svg", "png", "pdf"]:
    fig.savefig(os.path.join(OUT_FIG, f"F3_delta_direction.{ext}"), dpi=200, bbox_inches="tight")
plt.close()
print("F3 saved.")

# =====================================================================
# F4: Gene-set shift comparison
# =====================================================================
fig, ax = plt.subplots(figsize=(8, 5))

gene_sets = {
    "All genes": np.ones(len(harm), dtype=bool),
    "SynGO": harm["is_syngo"].values,
    "JASPAR TF": harm["is_jaspar_tf"].values,
}

x_pos = np.arange(len(populations))
width = 0.25
offsets = [-width, 0, width]

for (gs_name, mask), offset in zip(gene_sets.items(), offsets):
    subset = harm[mask]
    means = [subset[f"m1_{pop}_predicted_delta"].mean() for pop in populations]
    sems = [subset[f"m1_{pop}_predicted_delta"].sem() for pop in populations]
    bars = ax.bar(x_pos + offset, means, width, yerr=sems, label=f"{gs_name} (n={mask.sum()})",
                  capsize=3, alpha=0.85)

ax.set_xticks(x_pos)
ax.set_xticklabels(populations)
ax.set_ylabel("Mean predicted delta (imputed - raw)")
ax.set_title("Predicted Shift by Gene Set and Population")
ax.legend(fontsize=9)
ax.axhline(0, color="gray", linewidth=0.5)
fig.tight_layout()
for ext in ["svg", "png", "pdf"]:
    fig.savefig(os.path.join(OUT_FIG, f"F4_geneset_shifts.{ext}"), dpi=200, bbox_inches="tight")
plt.close()
print("F4 saved.")

# =====================================================================
# F5: Donor robustness
# =====================================================================
loo_file = os.path.join(OUT_TAB, "donor_robustness_loo.csv")
if os.path.exists(loo_file):
    loo_df = pd.read_csv(loo_file)
    fig, ax = plt.subplots(figsize=(8, 5))

    for pop in populations:
        sub = loo_df[loo_df["group"] == pop]
        ax.bar(sub["excluded_donor"] + f"\n({pop[:3]})", sub["pearson_loo"],
               color=POP_COLORS[pop], alpha=0.7, label=pop if pop == populations[0] else "")

    # Add full-data reference lines
    for pop in populations:
        sub = loo_df[loo_df["group"] == pop]
        if len(sub) > 0:
            ax.axhline(sub["pearson_full"].iloc[0], color=POP_COLORS[pop],
                       linestyle="--", alpha=0.5, linewidth=1)

    ax.set_ylabel("Pearson r (GSE vs M1 imputed)")
    ax.set_title("Donor Leave-One-Out Stability")
    ax.set_xlabel("Excluded M1 donor")
    # Custom legend
    from matplotlib.lines import Line2D
    handles = [plt.Rectangle((0,0), 1, 1, fc=POP_COLORS[p], alpha=0.7) for p in populations]
    handles.append(Line2D([0], [0], linestyle="--", color="gray"))
    labels = populations + ["Full data"]
    ax.legend(handles, labels, fontsize=9)
    fig.tight_layout()
    for ext in ["svg", "png", "pdf"]:
        fig.savefig(os.path.join(OUT_FIG, f"F5_donor_robustness.{ext}"), dpi=200, bbox_inches="tight")
    plt.close()
    print("F5 saved.")
else:
    print("F5 SKIPPED: donor_robustness_loo.csv not found.")

# =====================================================================
# F6: Population delta summary (grouped bar)
# =====================================================================
pb_file = os.path.join(OUT_TAB, "human_m1_pseudobulk_summary.csv")
if os.path.exists(pb_file):
    pb_df = pd.read_csv(pb_file)
    fig, ax = plt.subplots(figsize=(7, 5))

    x_pos = np.arange(len(pb_df))
    colors = [POP_COLORS[g] for g in pb_df["group"]]
    ax.bar(x_pos, pb_df["mean_predicted_delta"], color=colors, alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(pb_df["group"])
    ax.set_ylabel("Mean predicted delta")
    ax.set_title("Population-Level Predicted Shifts")
    ax.axhline(0, color="gray", linewidth=0.5)

    for i, row in pb_df.iterrows():
        ax.text(i, row["mean_predicted_delta"] + 0.002,
                f"n={row['n_nuclei']:,}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    for ext in ["svg", "png", "pdf"]:
        fig.savefig(os.path.join(OUT_FIG, f"F6_population_deltas.{ext}"), dpi=200, bbox_inches="tight")
    plt.close()
    print("F6 saved.")
else:
    print("F6 SKIPPED: pseudobulk_summary.csv not found.")

print("\nAll figures done.")
