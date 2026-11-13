"""Benchmark slide, label-free group only, no-covariate (July) ImputeVAE (2026-10-28).
Split-aware test partition; numbers from notes/benchmark_review_2026-10-09.md §5a and the label-free re-decode table."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE/notes/presentation/CBB_RIP_2026-11/figures/ripfig_benchmark_labelfree.png"
C_OURS, C_OTHER, INK, MUTED, GRID = "#2a78d6", "#898781", "#0b0b0b", "#52514e", "#e1e0d9"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
# label, gene corr, sd, energy distance, sd, colour
rows = [("ImputeVAE", 0.900, 0.002, 6.8, 0.7, C_OURS),
        ("scANVI (label-free decode)", 0.865, 0.002, 9.0, 0.4, C_OTHER),
        ("scVI (label-free decode)", 0.857, 0.007, 9.2, 0.4, C_OTHER),
        ("Raw nuclei", 0.760, 0.0, 128.6, 0.0, C_OTHER)]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.3, 1]}, sharey=True)
y = np.arange(len(rows))[::-1]
for ax, (vi, si), xlabel, title, fmt in [(axes[0], (1, 2), "median gene correlation of per-type means (held-out test)", "Cell-type-level accuracy", "{:.3f}"),
                                         (axes[1], (3, 4), "energy distance to real cells, within type (log; floor 0.62)", "Population realism", "{:.1f}")]:
    for yi, r in zip(y, rows):
        v, sd = r[vi], r[si]
        if sd: ax.plot([v - 2 * sd, v + 2 * sd], [yi, yi], color=r[5], lw=2.5, zorder=2)
        ax.scatter(v, yi, s=170, color=r[5], zorder=3, edgecolor="white", linewidth=1.5)
        ax.text(v, yi + 0.3, fmt.format(v) + (f" ± {sd:.3f}" if sd and vi == 1 else ""), ha="center", fontsize=12, color=INK)
    ax.set_title(title, loc="left", fontsize=14); ax.set_xlabel(xlabel, fontsize=12)
    ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True); ax.spines["left"].set_visible(False)
axes[0].set_xlim(0.70, 0.95); axes[1].set_xscale("log"); axes[1].set_xlim(4, 200)
axes[0].set_yticks(y); axes[0].set_yticklabels([r[0] for r in rows], fontsize=13, color=INK); axes[0].set_ylim(-0.6, len(rows) - 0.3)
fig.suptitle("No cell-type label at inference: ImputeVAE leads on accuracy and on population realism", x=0.01, ha="left", fontsize=15, color=INK)
fig.text(0.01, 0.015, "Split-aware benchmark: 8,034 held-out nuclei, 10,134 genes, same training data for every method; scVI / scANVI decoded label-free (own library x one global ratio); mean ± 2 sd over 3 seeds.\n"
         "Methods that need the cell-type label at inference (CT-mean, scGen, scANVI with a per-type library) reach 0.96-0.98 and are shown on a separate slide.", fontsize=9.5, color=MUTED)
fig.tight_layout(rect=(0, 0.09, 1, 0.93)); fig.savefig(OUT, dpi=200); print("wrote", OUT)
