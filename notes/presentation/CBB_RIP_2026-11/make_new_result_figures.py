"""Talk figures for the two new RIP-talk results (2026-10-28), from recorded result tables (no recomputation).

1. ripfig_trap_h2.png   Spearman similarity of per-type pseudobulk (raw nucleus -> imputed) to external RiboTag/TRAP
                        ribosome-bound profiles, with the real-cell ceiling; 17 populations, post L5-audit
                        (Scnn1a and Rbp4 excluded). Data: figE2_H2_all_datasets_data.csv; method tie from figE5 data.
2. ripfig_l5it_variance.png  L5 IT within-type per-gene variance: distribution (cell / nucleus / imputed) and
                        real-vs-imputed rank agreement (Spearman 0.92). Data: V1_focus_distribution_and_rank_data.csv.
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, gaussian_kde

ROOT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE"
OUT = os.path.join(ROOT, "notes/presentation/CBB_RIP_2026-11/figures"); os.makedirs(OUT, exist_ok=True)
C_NUC, C_IMP, C_CELL, INK, MUTED, GRID = "#898781", "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e1e0d9"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})

# ------------------------------------------------------------------ 1. TRAP / RiboTag H2
h2 = pd.read_csv(f"{ROOT}/validation/trap_ribotag_external_validation/outputs/extension/figures/figE2_H2_all_datasets_data.csv")
h2 = h2[~h2["population"].str.contains("Scnn1a|Rbp4")].copy()   # L5 mapping audit 2026-09-24/25
order = {"excitatory": 0, "inhibitory": 1, "glia": 2}
h2["cls"] = h2["cell_class"].map(order)
h2 = h2.sort_values(["cls", "spearman_imputed"], ascending=[True, False]).reset_index(drop=True)
pop_label = {"CamK2a_exc": "CamK2a excitatory", "Glt25d2_L5b": "Glt25d2 L5b", "Colgalt2_L5b": "Colgalt2 L5b (M1)", "Gprin3_L5b": "Gprin3 L5b (M1)",
             "S100a10_L5a": "S100a10 L5a", "Htr3a_IN": "Htr3a interneurons", "PV": "PV", "Pvalb": "Pvalb", "SST": "SST", "Sst": "Sst",
             "VIP": "VIP", "Vip": "Vip", "Aldh1l1_astro": "Aldh1l1 astrocytes", "GFAP_astro": "GFAP astrocytes", "Oligo_TRAP": "Oligodendrocytes"}
h2["name"] = [f"{pop_label.get(p, p)}  ->  {m.replace('POOL_', 'pool: ')}   ({d})" for p, m, d in zip(h2["population"], h2["model_col"], h2["dataset"])]

fig, ax = plt.subplots(figsize=(12.5, 7.2))
y = np.arange(len(h2))[::-1]
for yi, r in zip(y, h2.itertuples()):
    ax.plot([r.spearman_nucleus, r.spearman_imputed], [yi, yi], color="#c3c2b7", lw=2, zorder=1)
    ax.scatter(r.spearman_nucleus, yi, s=70, color=C_NUC, zorder=3, edgecolor="white", linewidth=1.2)
    ax.scatter(r.spearman_imputed, yi, s=90, color=C_IMP, zorder=4, edgecolor="white", linewidth=1.2)
    ax.plot([r.spearman_realcell, r.spearman_realcell], [yi - 0.32, yi + 0.32], color=C_CELL, lw=3, zorder=5)
# class separators and labels
bounds = h2.groupby("cls").apply(lambda d: (d.index.min(), d.index.max()))
for cls, (lo, hi) in bounds.items():
    ytop, ybot = y[lo] + 0.5, y[hi] - 0.5
    if hi < len(h2) - 1: ax.axhline(ybot, color=GRID, lw=1)
    ax.text(0.835, (ytop + ybot) / 2, {0: "excitatory", 1: "inhibitory", 2: "glia"}[cls], rotation=270, va="center", ha="center",
            color=MUTED, fontsize=11, clip_on=False)
ax.set_yticks(y); ax.set_yticklabels(h2["name"], fontsize=11, color=INK)
ax.set_xlim(0.2, 0.82); ax.spines["left"].set_visible(False); ax.set_xlabel("Spearman correlation of the cell-type pseudobulk with the external ribosome-bound profile")
ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.scatter([], [], s=70, color=C_NUC, label="raw nuclei"); ax.scatter([], [], s=90, color=C_IMP, label="ImputeVAE imputed")
ax.plot([], [], color=C_CELL, lw=3, label="real whole cells (ceiling)")
ax.legend(loc="lower right", frameon=False, fontsize=11)
med_gap = h2["gap_closed_frac"].median()
fig.suptitle("Imputation moves every population to the real-cell ceiling against independent RiboTag / TRAP profiles",
             x=0.01, ha="left", fontsize=14, color=INK)
ax.set_title(f"17 populations, 9 datasets, {int(h2['n_genes'].min()):,}-{int(h2['n_genes'].max()):,} genes; median gap closed {med_gap:.2f}; all 95 % CIs exclude zero",
             fontsize=11.5, loc="left", color=MUTED)
fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(f"{OUT}/ripfig_trap_h2.png", dpi=200); plt.close(fig)

# method tie (for the speaker note / optional inset): class means by method
m = pd.read_csv(f"{ROOT}/validation/trap_ribotag_external_validation/outputs/extension/figures/figE5_H2_method_comparison_data.csv")
m = m[~m["population"].str.contains("Scnn1a|Rbp4")]
tie = m.groupby(["cell_class", "method"])["spearman_imputed"].mean().unstack().round(3)
tie.to_csv(f"{OUT}/ripfig_trap_h2_method_tie.csv"); print(tie)

# ------------------------------------------------------------------ 2. L5 IT variance
v = pd.read_csv(f"{ROOT}/notes/experiments/variance_zero_nonzero_fullmodel/variance_rank/figures/V1_focus_distribution_and_rank_data.csv")
v = v[(v["cell_type"] == "L5 IT") & (v["var_cell_all"] > 0) & (v["var_nuc_all"] > 0) & (v["var_imp_all"] > 0)]
lc, ln, li = np.log10(v["var_cell_all"]), np.log10(v["var_nuc_all"]), np.log10(v["var_imp_all"])
rho = spearmanr(lc, li).correlation
rho_well = spearmanr(lc[v["det_cell"] > 0.1], li[v["det_cell"] > 0.1]).correlation
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axes[0]; grid = np.linspace(-4.5, 1, 400)
for arr, col, lab in [(lc, C_CELL, "real cells"), (ln, C_NUC, "raw nuclei"), (li, C_IMP, "imputed from nuclei")]:
    ax.plot(grid, gaussian_kde(arr)(grid), color=col, lw=2.6, label=f"{lab}  (median {10**np.median(arr):.2f})")
    ax.axvline(np.median(arr), color=col, lw=1, ls=":")
ax.set_xlabel("log10 within-type variance per gene (log1p scale)"); ax.set_ylabel("density of genes")
ax.set_title(f"Shifted and stretched, not collapsed\nL5 IT, {len(v):,} genes; 29,721 cells, 11,599 nuclei", loc="left", fontsize=13)
ax.legend(frameon=False, fontsize=11, loc="upper left"); ax.set_xlim(-4.5, 1)
ax = axes[1]
hb = ax.hexbin(lc, li, gridsize=55, cmap="Blues", mincnt=1, linewidths=0.2)
lim = [-3.8, 0.6]; ax.plot(lim, lim, color=MUTED, lw=1, ls="--", label="identity")
ax.set_xlim(-1.8, 0.6); ax.set_ylim(-3.8, 0.6)
ax.set_xlabel("log10 variance, real cells"); ax.set_ylabel("log10 variance, imputed")
ax.set_title(f"The same genes are the high-variance genes\nSpearman {rho:.2f} (well-detected {rho_well:.2f}); nucleus vs cell 0.72", loc="left", fontsize=13)
ax.legend(frameon=False, loc="lower right", fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT}/ripfig_l5it_variance.png", dpi=200); plt.close(fig)
print("rho", round(rho, 3), "well-detected", round(rho_well, 3), "n", len(v))
print("wrote", OUT)
