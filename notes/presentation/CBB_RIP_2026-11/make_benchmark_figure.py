"""Benchmark slide figure (2026-10-28): split-aware test partition, two groups by label use at inference.
Numbers: notes/benchmark_review_2026-10-09.md §5, §5a, §5b and the label-free re-decode table."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE/notes/presentation/CBB_RIP_2026-11/figures/ripfig_benchmark.png"
C_OURS, C_OTHER, C_CEIL, INK, MUTED, GRID = "#2a78d6", "#898781", "#eb6834", "#0b0b0b", "#52514e", "#e1e0d9"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
# (label, gene corr, energy distance, colour)
label_using = [("Cell-type mean of real cells (ceiling)", 0.981, 25.1, C_CEIL), ("scGen", 0.975, 15.9, C_OTHER),
               ("ImputeVAE, label at inference", 0.970, 2.6, C_OURS), ("scButterfly", 0.966, 12.2, C_OTHER),
               ("scANVI", 0.962, 16.4, C_OTHER), ("scVI, per-type library", 0.958, 16.6, C_OTHER)]
label_free = [("ImputeVAE, predicted type", 0.962, 2.6, C_OURS), ("ImputeVAE, no labels anywhere", 0.921, 5.6, C_OURS),
              ("ImputeVAE, no covariate (July)", 0.900, 6.8, C_OURS), ("scANVI, label-free", 0.865, 9.0, C_OTHER),
              ("scVI, label-free", 0.857, 9.2, C_OTHER), ("raw nuclei", 0.760, 128.6, C_OTHER)]
rows = [("HDR: cell-type label used at inference",)] + label_using + [("HDR: no label at inference",)] + label_free
fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw={"width_ratios": [1.25, 1]}, sharey=True)
y = np.arange(len(rows))[::-1]
for ax, col, xlabel, title in [(axes[0], 1, "gene correlation of per-type means (test, median gene)", "Cell-type-level accuracy"),
                               (axes[1], 2, "energy distance to real cells (log; floor 0.62 = real vs real)", "Population realism within type")]:
    for yi, r in zip(y, rows):
        if len(r) == 1: continue
        v = r[col]
        ax.plot([ax.get_xlim()[0] if False else 0, v], [yi, yi], color=GRID, lw=1, zorder=1) if col == 1 else None
        ax.scatter(v, yi, s=110, color=r[3], zorder=3, edgecolor="white", linewidth=1.2)
        ax.text(v, yi + 0.33, f"{v:.3f}" if col == 1 else f"{v:.1f}", ha="center", fontsize=10.5, color=INK)
    ax.set_title(title, loc="left", fontsize=13); ax.set_xlabel(xlabel, fontsize=11.5)
    ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True); ax.spines["left"].set_visible(False)
axes[0].set_xlim(0.72, 1.0); axes[1].set_xscale("log"); axes[1].set_xlim(1.5, 200); axes[1].axvline(0.62, color=MUTED, lw=1, ls=":")
axes[0].set_yticks(y); labs = axes[0].set_yticklabels([r[0].replace("HDR: ", "") for r in rows], fontsize=11.5, color=INK)
for lab, r in zip(labs, rows):
    if len(r) == 1: lab.set_fontweight("bold"); lab.set_color(MUTED)
sep = y[len(label_using) + 1]
for ax in axes: ax.axhline(sep + 0.5, color=GRID, lw=1)
axes[0].scatter([], [], s=110, color=C_OURS, label="ImputeVAE (covariate model)"); axes[0].scatter([], [], s=110, color=C_OTHER, label="baselines")
axes[0].scatter([], [], s=110, color=C_CEIL, label="ceiling"); fig.legend(frameon=False, fontsize=11, loc="upper right", ncol=3, bbox_to_anchor=(0.99, 0.955))
for ax in axes: ax.set_ylim(-0.7, len(rows) - 0.3)
fig.suptitle("Split-aware benchmark: best translator without labels; with labels, the type mean is the ceiling", x=0.01, ha="left", fontsize=14, color=INK)
fig.text(0.01, 0.01, "8,034 held-out nuclei, 10,134 genes, same training data for every method; count models decoded at the per-type cell library; 3 seeds where available",
         fontsize=10, color=MUTED)
fig.tight_layout(rect=(0, 0.05, 1, 0.95)); fig.savefig(OUT, dpi=200); print("wrote", OUT)
