#!/usr/bin/env python3
"""Extension figures E1-E5 (PNG + PDF + _data.csv), Okabe-Ito palette, one axis per panel."""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
TASK = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); EXT = f"{TASK}/outputs/extension"; TAB = f"{EXT}/tables"; FIG = f"{EXT}/figures"
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "verm": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9", "grey": "#7F7F7F"}
KIND = {"nucleus": C["grey"], "imputed": C["blue"], "cell": C["orange"]}; CLASSCOL = {"excitatory": C["green"], "inhibitory": C["purple"], "glia": C["verm"]}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#E5E5E5", "grid.linewidth": 0.6, "axes.axisbelow": True, "pdf.fonttype": 42})
def save(fig, name, data=None):
    fig.savefig(f"{FIG}/{name}.png", dpi=200, bbox_inches="tight"); fig.savefig(f"{FIG}/{name}.pdf", bbox_inches="tight"); plt.close(fig)
    if data is not None: data.to_csv(f"{FIG}/{name}_data.csv", index=False)
    print("saved", name)

# ---------------- E1 dataset overview (table graphic)
rows = [
 ("GSE210071", 2022, "cultured cortical cells (triptolide 3 h)", "cytosol / nucleus / whole cell, FPKM", "mixed culture", "3 paired cultures", "compartment (H1)"),
 ("GSE121199", 2019, "nucleus accumbens", "whole-cell FACS, nuclear FACS, RiboTag", "D1, D2 MSN", "5-8 per group (male)", "compartment (H1) + triangulation"),
 ("GSE263082", 2024, "cortex + hippocampus", "RiboTag IP RNA-seq", "Pvalb, Sst", "8 + 7 (4 batches)", "RiboTag abundance (H2, H3)"),
 ("GSE77243", 2016, "visual cortex", "RiboTag IP RNA-seq", "Pvalb, Sst, Vip", "3 each", "RiboTag abundance (H2, H3)"),
 ("GSE133291", 2019, "neocortex", "RiboTag IP RNA-seq", "CamK2a exc, Scnn1a L4, PV, SST, VIP", "4 each", "RiboTag abundance (H2, H3) [new]"),
 ("GSE172484", 2022, "motor cortex (M1)", "bacTRAP IP RNA-seq", "Gprin3 lower L5b, Colgalt2 upper L5b", "4 + 5 (2 batches)", "RiboTag abundance (H2, H3) [new]"),
 ("GSE202172", 2022, "cortex", "bacTRAP IP RNA-seq", "S100a10 L5a", "4", "RiboTag abundance (H2, H3) [new]"),
 ("GSE141337", 2019, "cerebral cortex", "bacTRAP IP RNA-seq", "Glt25d2 L5b", "3", "RiboTag abundance (H2, H3) [new]"),
 ("GSE142028", 2019, "cerebral cortex", "TRAP IP RNA-seq", "Rbp4 L5", "2", "RiboTag abundance (H2, H3) [new]"),
 ("GSE141874", 2019, "cerebral cortex", "TRAP IP RNA-seq", "Htr3a interneurons", "3", "RiboTag abundance (H2, H3) [new]"),
 ("GSE139684", 2019, "prefrontal cortex", "bacTRAP IP RNA-seq", "Aldh1l1 astrocytes", "3 controls", "RiboTag abundance (H2, H3) [new]"),
 ("GSE99791", 2017, "motor + visual cortex", "RiboTag IP RNA-seq", "GFAP astrocytes (4 mo)", "3 + 3", "RiboTag abundance (H2, H3) [new]"),
 ("GSE211162", 2023, "frontal cortex", "TRAP IP RNA-seq (RPKM)", "mature oligodendrocytes", "7", "RiboTag abundance (H2, H3) [new]"),
 ("GSE174303", 2021, "hippocampus CA1", "TRAP IP + input, cell body vs neuropil", "CA1 pyramidal", "4 per group (WT)", "enrichment (property mismatch)"),
 ("GSE13379", 2008, "cortex (bacTRAP atlas)", "Affymetrix microarray, IP + unbound", "9 cortical populations", "3 IP + 1 unbound", "legacy platform comparison"),
]
ov = pd.DataFrame(rows, columns=["accession", "year", "tissue", "assay", "cell types", "replicates used", "role"])
fig, ax = plt.subplots(figsize=(15, 5.6)); ax.axis("off")
tcol = {"compartment": "#E6F4EF", "RiboTag": "#FDF1DC", "enrichment": "#FBE6DC", "legacy": "#EEEEEE"}
tb = ax.table(cellText=ov.values, colLabels=ov.columns, cellLoc="left", loc="center", colWidths=[0.07, 0.04, 0.2, 0.22, 0.21, 0.11, 0.2])
tb.auto_set_font_size(False); tb.set_fontsize(7.5); tb.scale(1, 1.45)
for (r, c), cell in tb.get_celld().items():
    cell.set_edgecolor("#CCCCCC")
    if r == 0: cell.set_facecolor("#333333"); cell.get_text().set_color("white"); cell.get_text().set_weight("bold")
    else:
        role = ov.iloc[r - 1]["role"]; cell.set_facecolor(next(v for k, v in tcol.items() if role.startswith(k)))
ax.set_title("Datasets used in the external validation (green = direct compartment, amber = ribosome-associated abundance, red = enrichment-type, grey = legacy platform)", fontsize=9)
save(fig, "figE1_dataset_overview", ov)

# ---------------- E2 H2 across all datasets x cell types
h2 = pd.read_csv(f"{TAB}/extension_H2_profile_improvement.tsv", sep="\t")
multi = h2.groupby(["dataset", "population"]).batch.transform("nunique") > 1
h2s = h2[(~multi) | (h2.batch == "META")].copy()
order = {"excitatory": 0, "inhibitory": 1, "glia": 2}; h2s["o"] = h2s.cell_class.map(order); h2s = h2s.sort_values(["o", "model_col", "dataset"]).reset_index(drop=True)
h2s["label"] = h2s.dataset + " " + h2s.population + " -> " + h2s.model_col
fig, axes = plt.subplots(1, 2, figsize=(14, 0.36 * len(h2s) + 1.6), gridspec_kw={"width_ratios": [1.5, 1]}, sharey=True)
y = np.arange(len(h2s)); hgt = 0.26
for j, (k, col) in enumerate([("nucleus", "spearman_nucleus"), ("imputed", "spearman_imputed"), ("cell", "spearman_realcell")]):
    axes[0].barh(y + (j - 1) * hgt, h2s[col], hgt, color=KIND[k], edgecolor="white", label={"nucleus": "raw nucleus", "imputed": "ImputeVAE imputed", "cell": "real whole cell (ceiling)"}[k])
axes[0].set_yticks(y); axes[0].set_yticklabels(h2s.label, fontsize=7.5); axes[0].invert_yaxis(); axes[0].set_xlabel("Spearman r with external ribosome-associated profile"); axes[0].legend(fontsize=7.5, loc="lower right"); axes[0].set_title("H2: profile similarity (wild-type baseline IP samples)")
for t, cl in zip(axes[0].get_yticklabels(), h2s.cell_class): t.set_color("black")
for i, cl in enumerate(h2s.cell_class): axes[0].plot([-0.015], [i], marker="s", ms=6, color=CLASSCOL[cl], clip_on=False)
d = h2s.delta_similarity_spearman
axes[1].errorbar(d, y, xerr=[d - h2s.gene_boot_ci_lo, h2s.gene_boot_ci_hi - d], fmt="o", color=C["blue"], ms=5, capsize=2, label="gene-bootstrap 95% CI")
axes[1].axvline(0, color="k", lw=0.8); axes[1].set_xlabel("improvement = r(imputed) - r(nucleus)"); axes[1].set_title("Improvement with uncertainty"); axes[1].legend(fontsize=7.5, loc="lower right")
axes[1].legend(handles=[Line2D([0], [0], marker="o", color=C["blue"], ls="", label="gene-bootstrap 95% CI")] + [Line2D([0], [0], marker="s", color=v, ls="", label=f"{k} (row marker)") for k, v in CLASSCOL.items()], fontsize=7.5, loc="lower left")
save(fig, "figE2_H2_all_datasets", h2s)

# ---------------- E3 H3 rho incl. platform effect
h3 = pd.read_csv(f"{TAB}/extension_H3_translational_support.tsv", sep="\t")
multi = h3.groupby(["dataset", "population"]).batch.transform("nunique") > 1
h3s = h3[(~multi) | (h3.batch == "META")].copy(); h3s["platform"] = "RNA-seq"
sec = pd.read_csv(f"{TASK}/outputs/tables/concordance_summary_secondary.tsv", sep="\t")
arr = sec[(sec.dataset == "GSE13379") & sec.contrast.str.startswith("TRAP_abundance") & (sec.method == "ImputeVAE")]
arr = pd.DataFrame({"dataset": "GSE13379", "population": arr.contrast.str.replace("TRAP_abundance_", ""), "model_col": arr.model_col, "spearman_ribotag_vs_pred_delta": arr.spearman,
                    "partial_spearman_pred_given_nucexpr": arr.partial_spearman_given_nucexpr, "platform": "microarray (2008)", "cell_class": "mixed"})
allh = pd.concat([h3s, arr], ignore_index=True).sort_values(["platform", "spearman_ribotag_vs_pred_delta"], ascending=[False, False]).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(9, 0.3 * len(allh) + 1.5)); y = np.arange(len(allh))
for plat, col in (("RNA-seq", C["blue"]), ("microarray (2008)", C["grey"])):
    mk = allh.platform == plat
    ax.scatter(allh.spearman_ribotag_vs_pred_delta[mk], y[mk], s=45, color=col, label=f"{plat}: rho(abundance, predicted delta)", zorder=3)
    ax.scatter(allh.partial_spearman_pred_given_nucexpr[mk], y[mk], s=30, facecolors="white", edgecolors=col, label=f"{plat}: partial rho given nucleus expression", zorder=3)
ax.set_yticks(y); ax.set_yticklabels(allh.dataset + " " + allh.population.astype(str) + " -> " + allh.model_col, fontsize=7.5); ax.invert_yaxis(); ax.set_xlim(0, 0.9)
ax.set_xlabel("Spearman rho"); ax.legend(fontsize=7, loc="upper right"); ax.set_title("H3: ribosome-associated abundance vs ImputeVAE predicted delta; platform effect")
save(fig, "figE3_H3_platform_effect", allh)

# ---------------- E4 specificity heatmaps
sp = pd.read_csv(f"{TAB}/extension_specificity.tsv", sep="\t")
for centered in (False, True):
    s = sp[sp.centered == centered]; cts = [c for c in s.model_ct.unique()]; exts = list(s.external.unique())
    fig, axes = plt.subplots(1, 3, figsize=(19, 0.3 * len(exts) + 2.6), sharey=True)
    v = s.spearman.abs().max()
    for ax, kind in zip(axes, ("nucleus", "imputed", "cell")):
        M = s[s.kind == kind].pivot(index="external", columns="model_ct", values="spearman").loc[exts, cts]
        im = ax.imshow(M.values, cmap="RdBu_r" if centered else "Blues", vmin=-v if centered else M.values.min(), vmax=v if centered else M.values.max(), aspect="auto")
        ax.set_xticks(range(len(cts))); ax.set_xticklabels(cts, rotation=90, fontsize=7); ax.set_yticks(range(len(exts))); ax.set_yticklabels([e.replace("|", " ") for e in exts], fontsize=7); ax.grid(False)
        for i, e in enumerate(exts):
            row = s[(s.kind == kind) & (s.external == e)]; mc = row.matched_ct.iloc[0]; j = cts.index(mc)
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="black", lw=1.8))
            cand = [k for k, c in enumerate(cts) if (not c.startswith("POOL")) or c == mc]   # same candidate set as the rank statistic
            jb = cand[int(np.argmax(M.values[i][cand]))]; ax.plot(jb, i, marker="*", color=C["orange"], ms=7, mec="black", mew=0.4)
        rk = s[(s.kind == kind) & s.is_matched].matched_rank
        ax.set_title(f"{ {'nucleus': 'raw nucleus', 'imputed': 'ImputeVAE imputed', 'cell': 'real whole cell'}[kind]}: matched CT ranked 1st in {(rk == 1).sum()}/{len(rk)}, median rank {rk.median():.0f}", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    fig.suptitle(("Gene-centered " if centered else "Raw ") + "Spearman between external populations (rows) and model cell types (columns). Box = matched cell type, star = best-matching column.", fontsize=9)
    save(fig, f"figE4_specificity_{'centered' if centered else 'raw'}", s)

# ---------------- E5 method comparison
mc = pd.read_csv(f"{TAB}/extension_H2_method_comparison.tsv", sep="\t")
mc["label"] = mc.dataset + " " + mc.population; piv = mc.pivot_table(index=["cell_class", "label"], columns="method", values="spearman_imputed")
nuc = mc[mc.method == "ImputeVAE"].set_index("label").spearman_nucleus; cell = mc[mc.method == "ImputeVAE"].set_index("label").spearman_realcell
piv = piv.reindex([k for cl in ("excitatory", "inhibitory", "glia") for k in piv.index if k[0] == cl]); labels = [k[1] for k in piv.index]
meth = [m for m in ["ImputeVAE", "cVAE_20pct", "CT-mean", "Linear_reg", "scVI", "scANVI", "scGen", "scButterfly"] if m in piv.columns]
mcol = dict(zip(meth, [C["blue"], C["sky"], C["orange"], C["green"], C["verm"], C["purple"], "#999933", "#000000"])); mmk = dict(zip(meth, ["o", "o", "D", "s", "^", "v", "<", ">"]))
fig, ax = plt.subplots(figsize=(10, 0.36 * len(labels) + 1.8)); y = np.arange(len(labels))
ax.scatter(nuc.loc[labels], y, marker="|", s=160, color=C["grey"], label="raw nucleus", zorder=2); ax.scatter(cell.loc[labels], y, marker="|", s=160, color="black", label="real whole cell (ceiling)", zorder=2)
for m_ in meth: ax.scatter(piv[m_].values, y, s=36 if m_ != "ImputeVAE" else 60, marker=mmk[m_], color=mcol[m_], label=m_, zorder=3 if m_ != "ImputeVAE" else 4, alpha=0.9)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.5); ax.invert_yaxis(); ax.set_xlabel("Spearman r(imputed profile, external ribosome-associated profile)"); ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5)); ax.set_title("H2 by method (same nuclei, same genes). CT-mean equals the real-cell profile by construction.")
save(fig, "figE5_H2_method_comparison", mc)
