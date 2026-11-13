#!/usr/bin/env python3
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE/notes/experiments/variance_zero_nonzero_fullmodel/latent_usage"
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "verm": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9", "grey": "#7F7F7F"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#E5E5E5", "grid.linewidth": 0.6, "axes.axisbelow": True, "pdf.fonttype": 42})
D = pd.read_csv(f"{B}/per_dimension.csv"); D = D[D.source == "nuclei"].sort_values("var_total", ascending=False).reset_index(drop=True)
A = pd.read_csv(f"{B}/ablation_per_dimension.csv").set_index("dim"); CU = pd.read_csv(f"{B}/ablation_cumulative.csv"); CT = pd.read_csv(f"{B}/per_cell_type.csv")
fig, axes = plt.subplots(2, 2, figsize=(14, 8.2), gridspec_kw={"hspace": 0.42, "wspace": 0.25})
ax = axes[0, 0]; x = np.arange(len(D)); w = 0.28
ax.bar(x - w, D.var_between_ct.clip(lower=1e-5), w, color=C["orange"], label="between cell types (identity)", edgecolor="white")
ax.bar(x, D.var_within_ct.clip(lower=1e-5), w, color=C["blue"], label="within cell type", edgecolor="white")
ax.bar(x + w, D.mean_posterior_var, w, color=C["grey"], label="posterior noise exp(logvar); 1.0 = prior", edgecolor="white")
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(D.dim, fontsize=7); ax.set_xlabel("latent dimension (sorted by total variance of mu)"); ax.set_ylabel("variance (log scale)"); ax.legend(fontsize=7.5, loc="upper right")
ax.axhline(0.01, color=C["verm"], ls="--", lw=1); ax.text(len(D) - 0.5, 0.012, "active-unit threshold 0.01", ha="right", fontsize=7, color=C["verm"])
ax.set_title(f"6 of 32 latent dimensions are active; the other 26 sit at the prior (nuclei, n = 40,166)", fontsize=9.5)
ax = axes[0, 1]
ax.bar(x, D.kl_to_std_normal.clip(lower=1e-4), color=[C["verm"] if r > 0.5 else C["green"] for r in D.r2_libsize_within_ct], edgecolor="white")
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(D.dim, fontsize=7); ax.set_xlabel("latent dimension (same order)"); ax.set_ylabel("KL to N(0,1) per dimension, nats (log scale)")
top = D.iloc[0:6]
for i, r in top.iterrows(): ax.text(i, r.kl_to_std_normal * 1.25, f"{r.kl_to_std_normal:.2f}", ha="center", fontsize=7.5)
ax.set_title("Information per dimension (red = library size, R2 with log library size within cell type = %.2f)" % D.r2_libsize_within_ct.max(), fontsize=9.5)
ax = axes[1, 0]; t8 = A.sort_values("share_retained_when_only_dim", ascending=False).head(8); xx = np.arange(len(t8))
ax.bar(xx - 0.2, t8.share_retained_when_only_dim, 0.38, color=C["blue"], label="keep ONLY this dimension (others frozen at cell-type mean)", edgecolor="white")
ax.bar(xx + 0.2, t8.share_lost_when_frozen, 0.38, color=C["sky"], label="variance lost when this dimension is frozen", edgecolor="white")
for i, v in enumerate(t8.share_retained_when_only_dim): ax.text(i - 0.2, v + 0.015, f"{v:.2f}", ha="center", fontsize=7.5)
ax.set_xticks(xx); ax.set_xticklabels([f"dim {d}" for d in t8.index]); ax.set_ylim(0, 1.1); ax.set_ylabel("share of imputed within-cell-type variance"); ax.legend(fontsize=7.5, loc="upper right")
ax.set_title("Ablation on the imputed output: one dimension carries 95% of within-cell-type variance", fontsize=9.5)
ax = axes[1, 1]; P = CT.pivot(index="cell_type", columns="source", values="latent_participation_ratio").dropna(); L = CT.pivot(index="cell_type", columns="source", values="libsize_share_of_latent_var").loc[P.index]
n = CT[CT.source == "nuclei"].set_index("cell_type").n.loc[P.index]; o = n.sort_values().index; y = np.arange(len(o))
ax.barh(y, L.loc[o, "nuclei"], color=C["verm"], alpha=0.85, edgecolor="white", label="share of within-cell-type latent variance that is library size")
ax2x = P.loc[o, "nuclei"]
for i, c in enumerate(o): ax.text(1.02, i, f"{ax2x[c]:.1f} dims", va="center", fontsize=7.5)
ax.set_yticks(y); ax.set_yticklabels([f"{c} (n={n[c]:,})" for c in o], fontsize=7.5); ax.set_xlim(0, 1.18); ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_xlabel("orange bars = share of within-cell-type latent variance that is library size; text = effective latent dimensions")
ax.set_title("Per cell type (nuclei): 1.1 to 2.4 effective latent dimensions", fontsize=9.5)
fig.savefig(f"{B}/L1_latent_usage.png", dpi=200, bbox_inches="tight"); fig.savefig(f"{B}/L1_latent_usage.pdf", bbox_inches="tight"); print("saved")
