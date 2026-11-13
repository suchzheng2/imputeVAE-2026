#!/usr/bin/env python3
"""Phase 7: figures (PDF + PNG, each with a plotted-data CSV). Okabe-Ito palette, one axis per panel."""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy import stats

TASK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = f"{TASK}/outputs/tables"; QC = f"{TASK}/outputs/qc"; FIG = f"{TASK}/outputs/figures"; PB = f"{TASK}/outputs/model_pseudobulk"
os.makedirs(FIG, exist_ok=True)
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "verm": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9", "yellow": "#F0E442", "grey": "#7F7F7F", "black": "#000000"}
KIND = {"nucleus": C["grey"], "imputed": C["blue"], "cell": C["orange"]}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#E5E5E5", "grid.linewidth": 0.6, "axes.axisbelow": True, "pdf.fonttype": 42})


def save(fig, name, data=None):
    fig.savefig(f"{FIG}/{name}.png", dpi=200, bbox_inches="tight"); fig.savefig(f"{FIG}/{name}.pdf", bbox_inches="tight"); plt.close(fig)
    if data is not None:
        data.to_csv(f"{FIG}/{name}_data.csv", index=False)
    print("saved", name)


def scatter_panel(ax, x, y, title, xlabel, ylabel, rho=None, ci=None, auroc=None, ptype="compartment"):
    ax.hexbin(x, y, gridsize=60, cmap="Blues", mincnt=1, linewidths=0.2)
    ax.axhline(0, color=C["grey"], lw=0.8); ax.axvline(np.median(x), color=C["grey"], lw=0.8, ls="--")
    lo, hi = np.quantile(x, [0.01, 0.99]); bins = np.linspace(lo, hi, 21); idx = np.digitize(x, bins)
    bx = [x[idx == i].mean() for i in range(1, 21) if (idx == i).sum() >= 20]; by = [np.median(y[idx == i]) for i in range(1, 21) if (idx == i).sum() >= 20]
    ax.plot(bx, by, color=C["verm"], lw=2, label="binned median")
    txt = f"Spearman rho = {rho:+.3f}" + (f" [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "") + (f"\nAUROC = {auroc:.3f}" if auroc == auroc and auroc is not None else "")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", fontsize=8, bbox=dict(fc="white", ec="none", alpha=0.8))
    ax.set_title(title + ("" if ptype == "compartment" else "  [enrichment-type contrast]"), fontsize=9); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)


def get_row(summ, **kw):
    m = summ
    for k, v in kw.items():
        m = m[m[k] == v]
    return m.iloc[0] if len(m) else None


# ---------------- Figure 1: design schematic
fig, ax = plt.subplots(figsize=(10, 4.2)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 4.2)
ax.add_patch(FancyBboxPatch((0.2, 1.4), 2.4, 1.6, boxstyle="round,pad=0.05", fc="#DCE9F5", ec=C["blue"]))
ax.text(1.4, 2.6, "ImputeVAE (frozen, MOp-trained)", ha="center", fontsize=9, weight="bold")
ax.text(1.4, 2.15, "nucleus -> imputed whole cell\npredicted delta = imputed - nucleus\n(per cell type, log1p pseudobulk)", ha="center", fontsize=7.5)
boxes = [("GSE210071\ncultured cortical cells\ncytosol / nucleus / whole cell\n(triptolide 3 h)", "COMPARTMENT\n(direct)", C["green"]),
         ("GSE121199\nNAc D1/D2 MSNs\nwhole-cell / nuclear FACS", "COMPARTMENT\n(direct, unmatched CT)", C["green"]),
         ("GSE263082 + GSE77243\nPvalb / Sst RiboTag (IP only)", "RIBOSOME-ASSOCIATED\nabundance (H2, H3)", C["orange"]),
         ("GSE174303 CA1 TRAP/input\nGSE13379 bacTRAP IP/UB", "RIBOSOME-ASSOCIATED\nenrichment (property mismatch)", C["verm"])]
for i, (lab, typ, col) in enumerate(boxes):
    x0 = 3.3 + i * 1.7
    ax.add_patch(FancyBboxPatch((x0, 1.3), 1.55, 1.8, boxstyle="round,pad=0.04", fc="white", ec=col, lw=1.5))
    ax.text(x0 + 0.775, 2.55, lab, ha="center", va="center", fontsize=6.6)
    ax.text(x0 + 0.775, 3.5, typ, ha="center", va="center", fontsize=7, color=col, weight="bold")
    ax.annotate("", xy=(x0, 2.2), xytext=(2.6, 2.2), arrowprops=dict(arrowstyle="->", color=C["grey"], lw=0.8))
ax.text(5, 0.7, "Tests: H1 rank concordance of predicted delta with external log2(compartment ratio); H2 profile similarity nucleus vs imputed vs RiboTag;\n"
        "H3 RiboTag abundance across predicted-delta deciles; H4 nuclear ncRNA controls; H5 abundance-matched nulls and gene-class filters.\n"
        "Ground-truth (real cell - nucleus) delta reported beside every model result.", ha="center", fontsize=7.2)
save(fig, "fig1_design_schematic")

# ---------------- Figure 2: QC
fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
p = pd.read_csv(f"{QC}/GSE210071_qc_pca.csv", index_col=0)
for (comp, trt), g in p.groupby(["compartment", "treatment"]):
    axes[0].scatter(g.PC1, g.PC2, s=40, marker="o" if "unstim" in trt else "^", color={"cytosol": C["blue"], "nucleus": C["orange"], "whole_cell": C["green"]}[comp], label=f"{comp} {'Ctl' if 'unstim' in trt else 'Bic'}")
axes[0].set_title("GSE210071 PCA (log2 FPKM)"); axes[0].set_xlabel(f"PC1 ({p.var_PC1.iloc[0]*100:.0f}%)"); axes[0].set_ylabel(f"PC2 ({p.var_PC2.iloc[0]*100:.0f}%)"); axes[0].legend(fontsize=6, ncol=2)
q = pd.read_csv(f"{QC}/GSE121199_qc_primary.tsv", sep="\t")
for grp, g in q.groupby("group"):
    a, s = grp.split("_"); axes[1].scatter(g.MDS1, g.MDS2, s=40, marker="o" if s == "D1" else "s", color={"wc": C["green"], "nuclear": C["orange"], "ribo": C["purple"]}[a], label=grp)
axes[1].set_title("GSE121199 MDS (logCPM, male)"); axes[1].set_xlabel("MDS1"); axes[1].set_ylabel("MDS2"); axes[1].legend(fontsize=6, ncol=2)
rows = []
for b in ["batch1", "batch2", "batch3", "batch4"]:
    d = pd.read_csv(f"{QC}/GSE263082_{b}_qc.tsv", sep="\t"); d["batch"] = b; rows.append(d)
d = pd.concat(rows); d["x"] = np.arange(len(d))
axes[2].bar(d.x, d.lib_size / 1e6, color=[C["blue"] if c == "Pvalb" else C["orange"] for c in d.cell_type], edgecolor="white", width=0.85)
for i, r in d.iterrows():
    if r.outlier_flag: axes[2].text(r.x, r.lib_size / 1e6 + 0.3, "*", ha="center", fontsize=9)
axes[2].set_xticks(d.x); axes[2].set_xticklabels([f"{b[-1]}:{t[:3]}" for b, t in zip(d.batch, d.treatment)], rotation=90, fontsize=5.5)
axes[2].set_title("GSE263082 library size (Mreads; blue Pvalb, orange Sst; * flagged)"); axes[2].set_ylabel("million reads")
q4 = pd.read_csv(f"{QC}/GSE174303_qc_WT.tsv", sep="\t")
for grp, g in q4.groupby("group"):
    reg, a = grp.split("_"); axes[3].scatter(g.MDS1, g.MDS2, s=40, marker="o" if reg == "CB" else "^", color=C["purple"] if a == "TRAP" else C["grey"], label=grp)
axes[3].set_title("GSE174303 MDS (WT)"); axes[3].set_xlabel("MDS1"); axes[3].set_ylabel("MDS2"); axes[3].legend(fontsize=6)
save(fig, "fig2_qc", pd.concat([p.reset_index().assign(dataset="GSE210071"), q.assign(dataset="GSE121199"), d.assign(dataset="GSE263082"), q4.assign(dataset="GSE174303")]))

# ---------------- Figure 3/4: H1 scatter
summ = pd.read_csv(f"{TAB}/concordance_summary_GSE210071_GSE121199.tsv", sep="\t")
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, (yc, ttl) in zip(axes[:2], [("cyto_minus_nuc_log2fc", "cytosol - nucleus"), ("whole_minus_nuc_log2fc", "whole cell - nucleus")]):
    m = pd.read_csv(f"{TAB}/merged_GSE210071_Ctl_{yc}.tsv", sep="\t", index_col=0)
    r = get_row(summ, dataset="GSE210071", condition="Ctl", contrast=yc, method="ImputeVAE", model_col="ALL", filter="none")
    scatter_panel(ax, m["pred_ALL"].values, m.external_log2fc.values, f"GSE210071 Ctl: {ttl}", "ImputeVAE predicted delta (all nuclei, log1p)", f"external log2 FPKM ratio ({ttl})", r.spearman, (r.spearman_ci_lo, r.spearman_ci_hi), r.auroc_external_label)
m = pd.read_csv(f"{TAB}/merged_GSE210071_Ctl_cyto_minus_nuc_log2fc.tsv", sep="\t", index_col=0)
r = get_row(summ, dataset="GSE210071", condition="Ctl", contrast="cyto_minus_nuc_log2fc", method="GroundTruth", model_col="ALL", filter="none")
scatter_panel(axes[2], m["gt_ALL"].values, m.external_log2fc.values, "GSE210071 Ctl: ground-truth delta (real cell - nucleus)", "MOp ground-truth delta", "external log2 FPKM ratio (cytosol - nucleus)", r.spearman, (r.spearman_ci_lo, r.spearman_ci_hi), r.auroc_external_label)
axes[0].legend(fontsize=7, loc="lower right")
save(fig, "fig3_gse210071_concordance", m.reset_index()[["gene", "pred_ALL", "gt_ALL", "external_log2fc", "external_fdr"]])

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, (yc, ttl, pt) in zip(axes, [("wc_vs_nuc_avg_log2fc", "whole cell - nuclear", "compartment"), ("ribo_vs_nuc_avg_log2fc", "RiboTag - nuclear", "compartment"), ("ribo_vs_wc_avg_log2fc", "RiboTag - whole cell", "enrichment")]):
    m = pd.read_csv(f"{TAB}/merged_GSE121199_primary_{yc}.tsv", sep="\t", index_col=0)
    r = get_row(summ, dataset="GSE121199", condition="primary", contrast=yc, method="ImputeVAE", model_col="POOL_neuronal", filter="none")
    scatter_panel(ax, m["pred_POOL_neuronal"].values, m.external_log2fc.values, f"GSE121199 male MSN: {ttl}", "ImputeVAE predicted delta (neuron-pooled)", f"limma log2FC ({ttl})", r.spearman, (r.spearman_ci_lo, r.spearman_ci_hi), r.auroc_external_label, pt)
save(fig, "fig4_gse121199_concordance", m.reset_index()[["gene", "pred_POOL_neuronal", "gt_POOL_neuronal", "external_log2fc", "external_fdr"]])

# ---------------- Figure 5: H2 profile improvement
h2 = pd.read_csv(f"{TAB}/profile_improvement_H2.tsv", sep="\t"); h2 = h2[(h2["filter"] == "none") & (h2.model_split == "all")]
h2["label"] = h2.dataset.str.replace("GSE263082", "GSE263082").str.cat(h2.cell_type, sep=" ").str.cat(h2.batch, sep=" / ")
fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [2, 1]})
x = np.arange(len(h2)); w = 0.27
for j, k in enumerate(["nucleus", "imputed", "cell"]):
    axes[0].bar(x + (j - 1) * w, h2[f"spearman_{'realcell' if k == 'cell' else k}"], w, color=KIND[k], label={"nucleus": "raw nucleus", "imputed": "ImputeVAE imputed", "cell": "real whole cell (ceiling)"}[k], edgecolor="white")
axes[0].set_xticks(x); axes[0].set_xticklabels(h2.label, rotation=45, ha="right", fontsize=7); axes[0].set_ylabel("Spearman r with external RiboTag profile"); axes[0].set_ylim(0, 0.9); axes[0].legend(fontsize=7, loc="upper left"); axes[0].set_title("H2: profile similarity to RiboTag (baseline WT)")
axes[1].errorbar(h2.delta_similarity_spearman, x, xerr=[h2.delta_similarity_spearman - h2.replicate_boot_ci_lo.fillna(h2.gene_boot_ci_lo), h2.replicate_boot_ci_hi.fillna(h2.gene_boot_ci_hi) - h2.delta_similarity_spearman], fmt="o", color=C["blue"], ms=5, capsize=2, label="replicate bootstrap 95% CI")
axes[1].errorbar(h2.delta_similarity_spearman, x + 0.25, xerr=[h2.delta_similarity_spearman - h2.gene_boot_ci_lo, h2.gene_boot_ci_hi - h2.delta_similarity_spearman], fmt="s", color=C["sky"], ms=4, capsize=2, label="gene bootstrap 95% CI")
axes[1].axvline(0, color=C["grey"]); axes[1].set_yticks(x); axes[1].set_yticklabels(h2.label, fontsize=7); axes[1].set_xlabel("delta similarity = r(imputed) - r(nucleus)"); axes[1].legend(fontsize=7); axes[1].set_title("Improvement with uncertainty")
save(fig, "fig5_profile_improvement", h2)

# ---------------- Figure 6: H3 decile curves + enrichment
h3 = pd.read_csv(f"{TAB}/translational_support_H3.tsv", sep="\t")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for _, r in h3.iterrows():
    if r.batch not in ("META", "greenberg"): continue
    curve = json.loads(r.decile_curve); col = C["blue"] if r.cell_type == "Pvalb" else C["orange"]; ls = "-" if r.dataset == "GSE263082" else "--"
    axes[0].plot(range(1, 11), curve, color=col, ls=ls, lw=2, marker="o", ms=4, label=f"{r.dataset} {r.cell_type} (rho={r.spearman_ribotag_vs_pred_delta:.2f})")
axes[0].set_xlabel("predicted-delta decile (1 = most nuclear, 10 = most cytoplasmic)"); axes[0].set_ylabel("mean RiboTag log2 expression"); axes[0].legend(fontsize=7); axes[0].set_title("H3: RiboTag abundance across predicted-delta deciles")
sub = h3[h3.batch.isin(["META", "greenberg"])].reset_index(drop=True); x = np.arange(len(sub))
axes[1].bar(x - 0.2, sub.top10pct_frac_in_top20_ribotag, 0.38, color=C["blue"], label="top-10% predicted-positive genes", edgecolor="white")
axes[1].bar(x + 0.2, sub.top10pct_matched_null_mean, 0.38, color=C["grey"], label="abundance-matched null", edgecolor="white")
axes[1].axhline(sub.top10pct_baseline_frac.mean(), color=C["verm"], lw=1, ls="--", label="baseline (20%)")
axes[1].set_xticks(x); axes[1].set_xticklabels(sub.dataset + " " + sub.cell_type, fontsize=7); axes[1].set_ylabel("fraction in top-20% RiboTag-expressed"); axes[1].legend(fontsize=7); axes[1].set_title("Enrichment of predicted-positive genes (matched p < 0.001)")
save(fig, "fig6_h3_translational_support", h3)

# ---------------- Figure 7: showcase genes
genes = ["Malat1", "Xist", "Meg3", "Snhg11", "Neat1", "Kcnq1ot1", "Snap25", "Camk2n1", "Tmsb4x", "Actb", "Pvalb", "Sst"]
cg = pd.read_csv(f"{TAB}/concordance_genes_GSE210071_GSE121199.tsv", sep="\t"); cg = cg[cg.status == "control"]
panels = [("GSE210071", "Ctl", "cyto_minus_nuc_log2fc", "GSE210071 cytosol - nucleus"), ("GSE210071", "Ctl", "whole_minus_nuc_log2fc", "GSE210071 whole - nucleus"),
          ("GSE121199", "primary", "wc_vs_nuc_avg_log2fc", "GSE121199 whole cell - nuclear"), ("GSE121199", "primary", "ribo_vs_nuc_avg_log2fc", "GSE121199 RiboTag - nuclear")]
pred_all = pd.read_csv(f"{PB}/predicted_delta_by_ct_all.csv", index_col=0)
fig, axes = plt.subplots(1, len(panels) + 1, figsize=(15, 4.2), sharey=True); rows = []
y = np.arange(len(genes))
pa = [pred_all.loc[g, "ALL"] if g in pred_all.index else np.nan for g in genes]
axes[0].barh(y, pa, color=[C["blue"] if v == v and v >= 0 else C["verm"] for v in pa]); axes[0].set_yticks(y); axes[0].set_yticklabels(genes); axes[0].axvline(0, color="k", lw=0.8); axes[0].set_title("ImputeVAE predicted delta (ALL)", fontsize=8); axes[0].invert_yaxis()
for g_, v in zip(genes, pa):
    if v != v: axes[0].text(0.02, genes.index(g_), "absent from model", fontsize=7, va="center")
for ax, (ds, cond, yc, ttl) in zip(axes[1:], panels):
    sub = cg[(cg.dataset == ds) & (cg.condition == cond) & (cg.contrast == yc)].drop_duplicates("gene").set_index("gene")
    vals = [sub.loc[g, "external_log2fc"] if g in sub.index and sub.loc[g, "external_log2fc"] == sub.loc[g, "external_log2fc"] else np.nan for g in genes]
    ax.barh(y, vals, color=[C["blue"] if v == v and v >= 0 else C["verm"] for v in vals]); ax.axvline(0, color="k", lw=0.8); ax.set_title(ttl, fontsize=8)
    for g_, v in zip(genes, vals):
        if v != v: ax.text(0.02, genes.index(g_), "absent", fontsize=7, va="center", color=C["grey"])
    rows += [dict(gene=g_, panel=ttl, external_log2fc=v, predicted_delta=pv) for g_, v, pv in zip(genes, vals, pa)]
fig.suptitle("Showcase genes: blue = away from nucleus, red = toward nucleus. Meg3 and Snhg11 are cell-context dependent; Neat1/Kcnq1ot1 absent from the model gene set.", fontsize=8)
save(fig, "fig7_showcase_genes", pd.DataFrame(rows))

# ---------------- Figure 8: forest plot
sec = pd.read_csv(f"{TAB}/concordance_summary_secondary.tsv", sep="\t")
fp = []
for _, r in summ[(summ.method == "ImputeVAE")].iterrows():
    if r.model_col not in ("ALL", "POOL_neuronal"): continue
    if r.dataset == "GSE210071" and r.model_col != "ALL": continue
    if r.dataset == "GSE121199" and r.model_col != "POOL_neuronal": continue
    if r["filter"] not in ("none", "excl_all_above"): continue
    typ = "enrichment" if "ribo_vs_wc" in r.contrast else ("triangulation" if "ribo_vs_nuc" in r.contrast else "compartment")
    fp.append(dict(label=f"{r.dataset} {r.condition} {r.contrast.replace('_log2fc','')} [{r['filter']}]", rho=r.spearman, lo=r.spearman_ci_lo, hi=r.spearman_ci_hi, typ=typ, primary=r["filter"] == "none"))
for _, r in sec[sec.method == "ImputeVAE"].iterrows():
    fp.append(dict(label=f"{r.dataset} {r.contrast} ({r.model_col})", rho=r.spearman, lo=r.get("spearman_ci_lo", np.nan), hi=r.get("spearman_ci_hi", np.nan), typ="enrichment" if "enrich" in str(r.contrast_type) or "IP_over" in r.contrast or "trap_vs" in r.contrast else "abundance", primary=True))
h3m = h3[h3.batch.isin(["META", "greenberg"])]
for _, r in h3m.iterrows():
    fp.append(dict(label=f"{r.dataset} {r.cell_type} RiboTag abundance vs pred delta", rho=r.spearman_ribotag_vs_pred_delta, lo=np.nan, hi=np.nan, typ="abundance", primary=True))
fp = pd.DataFrame(fp); order = {"compartment": 0, "triangulation": 1, "abundance": 2, "enrichment": 3}; fp["o"] = fp.typ.map(order); fp = fp.sort_values(["o", "rho"], ascending=[True, False]).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(9, 0.28 * len(fp) + 1.5))
cols = {"compartment": C["green"], "triangulation": C["purple"], "abundance": C["orange"], "enrichment": C["verm"]}
for i, r in fp.iterrows():
    ax.errorbar(r.rho, i, xerr=[[r.rho - r.lo if r.lo == r.lo else 0], [r.hi - r.rho if r.hi == r.hi else 0]], fmt="o" if r.primary else "s", color=cols[r.typ], ms=5 if r.primary else 3.5, capsize=2)
ax.set_yticks(range(len(fp))); ax.set_yticklabels(fp.label, fontsize=6.5); ax.invert_yaxis(); ax.axvline(0, color="k", lw=0.8); ax.set_xlabel("Spearman rho (ImputeVAE predicted delta vs external measure), 95% bootstrap CI where available")
from matplotlib.lines import Line2D
ax.legend([Line2D([0], [0], marker="o", color=cols[k], ls="") for k in cols], ["compartment-type (direct)", "RiboTag vs nucleus (assay triangulation)", "RiboTag abundance (IP only)", "enrichment-type (IP/input, IP/UB; property mismatch)"], fontsize=7, loc="lower right")
ax.set_title("Summary across datasets and contrasts (circles = unfiltered; squares = all H5 filters applied)", fontsize=9)
save(fig, "fig8_forest_sensitivity", fp)
