#!/usr/bin/env python3
"""Variance-rank check requested by Prashant (Sep 2026 meeting).

Question: when imputation lowers within-cell-type variance, does it keep the *ranking* of genes by
variance (same genes high-variance in real cells and in imputed cells, at a lower scale = denoising),
or does it flatten every gene to a similar low variance (imputation "cheating" with near-constant values)?

Inputs: notes/experiments/variance_zero_nonzero_fullmodel/tables/per_ct_gene_variance.csv.gz (job 10910494)
        notes/experiments/variance_zero_nonzero_fullmodel/tables/libsize_r2_per_ct_gene.csv.gz
Outputs: notes/experiments/variance_zero_nonzero_fullmodel/variance_rank/{tables,figures}
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

BASE = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE/notes/experiments/variance_zero_nonzero_fullmodel"
OUT = f"{BASE}/variance_rank"; TAB, FIG = f"{OUT}/tables", f"{OUT}/figures"
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
# Okabe-Ito (colourblind-safe); same assignment as the other figures in this experiment
SRC = {"cell": "#E69F00", "imputed": "#0072B2", "nucleus": "#7F7F7F"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#E5E5E5", "grid.linewidth": 0.6, "axes.axisbelow": True, "pdf.fonttype": 42})
CT_ORDER = ['L2/3 IT', 'L5 IT', 'L5 ET', 'L5/6 NP', 'L6 CT', 'L6 IT', 'L6b', 'Lamp5', 'Pvalb', 'Sncg', 'Sst', 'Vip',
            'Astro', 'Oligo', 'OPC', 'Macrophage', 'Endo']
FOCUS = ["L5 IT", "Oligo"]
DET_WELL = 0.10   # "well-detected" gene: detected in >=10% of real cells of that type

T = pd.read_csv(f"{BASE}/tables/per_ct_gene_variance.csv.gz")
R = pd.read_csv(f"{BASE}/tables/libsize_r2_per_ct_gene.csv.gz")
T = T.merge(R, on=["cell_type", "gene"], how="left")
T = T[(T.det_cell > 0.01) & (T.var_cell_all > 1e-6) & (T.var_imp_all > 0)].copy()
# library-size-free variance on both sides (R2 of the gene on log library size within type, from the diagnostic)
T["var_cell_libfree"] = T.var_cell_all * (1 - T.r2_lib_cell.clip(0, 1))
T["var_imp_libfree"] = T.var_imp_all * (1 - T.r2_lib_imp.clip(0, 1))
# biological excess of real-cell variance (Poisson + library size removed), from the diagnostic
T["var_cell_bio"] = T.bio_excess_cell_B.clip(lower=0)
T["bin"] = pd.cut(T.det_cell, [0, 0.1, 0.3, 0.7, 1.01], labels=["<10%", "10-30%", "30-70%", ">70%"], right=False)


def sp(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(stats.spearmanr(a[ok], b[ok])[0]) if ok.sum() > 10 else np.nan


def logq(v):
    v = np.log10(v[np.isfinite(v) & (v > 0)])
    return dict(median=np.median(v), iqr=np.subtract(*np.percentile(v, [75, 25])), p5=np.percentile(v, 5), p95=np.percentile(v, 95))


def ct_stats(g, label):
    row = dict(cell_type=label, n_genes=len(g), n_nuclei=int(g.n_nuclei.iloc[0]), n_cells=int(g.n_cells.iloc[0]))
    row["spearman_var_cell_vs_imp"] = sp(g.var_cell_all.values, g.var_imp_all.values)
    row["spearman_var_nuc_vs_imp"] = sp(g.var_nuc_all.values, g.var_imp_all.values)
    row["spearman_var_cell_vs_nuc"] = sp(g.var_cell_all.values, g.var_nuc_all.values)
    row["spearman_var_cellbio_vs_imp"] = sp(g.var_cell_bio.values, g.var_imp_all.values)
    row["spearman_var_cell_vs_imp_libfree"] = sp(g.var_cell_libfree.values, g.var_imp_libfree.values)
    w = g[g.det_cell >= DET_WELL]
    row["n_genes_well_detected"] = len(w)
    row["spearman_var_cell_vs_imp_well_detected"] = sp(w.var_cell_all.values, w.var_imp_all.values)
    row["spearman_var_cell_vs_imp_libfree_well_detected"] = sp(w.var_cell_libfree.values, w.var_imp_libfree.values)
    # mean-adjusted: rank correlation of residuals after regressing log var on log mean (variance tracks mean on log1p scale)
    def resid(v, m):
        lv, lm = np.log10(v), np.log10(np.clip(m, 1e-4, None)); ok = np.isfinite(lv) & np.isfinite(lm)
        r = np.full(len(v), np.nan); p = np.polyfit(lm[ok], lv[ok], 2); r[ok] = lv[ok] - np.polyval(p, lm[ok]); return r
    row["spearman_var_cell_vs_imp_mean_adjusted"] = sp(resid(g.var_cell_all.values, g.mean_cell.values), resid(g.var_imp_all.values, g.mean_imp.values))
    # dynamic range / distribution shape on log10 variance
    for k, v in (("cell", g.var_cell_all), ("imputed", g.var_imp_all), ("nucleus", g.var_nuc_all)):
        q = logq(v.values)
        for kk, vv in q.items(): row[f"log10var_{k}_{kk}"] = vv
    row["iqr_ratio_imputed_over_cell"] = row["log10var_imputed_iqr"] / row["log10var_cell_iqr"]
    lc, li = np.log10(g.var_cell_all.values), np.log10(g.var_imp_all.values); ok = np.isfinite(lc) & np.isfinite(li)
    row["slope_log10var_imp_on_cell"] = float(np.polyfit(lc[ok], li[ok], 1)[0])
    row["median_ratio_imp_over_cell"] = float(np.median(g.var_imp_all / g.var_cell_all))
    row["frac_imp_var_below_1e-3"] = float((g.var_imp_all < 1e-3).mean()); row["frac_cell_var_below_1e-3"] = float((g.var_cell_all < 1e-3).mean())
    # top-gene overlap: are the 200 most variable real-cell genes also among the most variable imputed genes?
    top = g.nlargest(200, "var_cell_all").gene; topi = g.nlargest(200, "var_imp_all").gene
    row["top200_overlap"] = len(set(top) & set(topi)) / 200
    return row


rows = [ct_stats(g, ct) for ct, g in T.groupby("cell_type")]
rows.append(ct_stats(T, "ALL (pooled pairs)"))
S = pd.DataFrame(rows).set_index("cell_type").reindex(CT_ORDER + ["ALL (pooled pairs)"])
S.to_csv(f"{TAB}/variance_rank_by_cell_type.csv")
B = pd.DataFrame([dict(ct_stats(g, f"{ct}|{b}"), cell_type=ct, bin=b) for (ct, b), g in T.groupby(["cell_type", "bin"], observed=True) if len(g) > 30])
B.to_csv(f"{TAB}/variance_rank_by_cell_type_and_detection_bin.csv", index=False)
pd.set_option("display.width", 250)
cols = ["n_genes", "spearman_var_cell_vs_imp", "spearman_var_cell_vs_imp_well_detected", "spearman_var_cellbio_vs_imp", "spearman_var_cell_vs_imp_libfree",
        "spearman_var_cell_vs_imp_mean_adjusted", "spearman_var_cell_vs_nuc", "spearman_var_nuc_vs_imp", "iqr_ratio_imputed_over_cell", "slope_log10var_imp_on_cell", "median_ratio_imp_over_cell", "top200_overlap"]
print(S[cols].round(3).to_string())


def save(fig, name, data=None):
    fig.savefig(f"{FIG}/{name}.png", dpi=200, bbox_inches="tight"); fig.savefig(f"{FIG}/{name}.pdf", bbox_inches="tight"); plt.close(fig)
    if data is not None: data.to_csv(f"{FIG}/{name}_data.csv", index=False)
    print("saved", name)


# ---------------- V1: focus cell types (L5 IT, Oligo): distributions + rank scatter
fig, axes = plt.subplots(len(FOCUS), 3, figsize=(11, 3.4 * len(FOCUS)))
data_rows = []
for r, ct in enumerate(FOCUS):
    g = T[T.cell_type == ct]; s = S.loc[ct]
    ax = axes[r, 0]
    xs = np.linspace(-5, 0.7, 300)
    for k, v in (("cell", g.var_cell_all), ("nucleus", g.var_nuc_all), ("imputed", g.var_imp_all)):
        lv = np.log10(v[v > 0]); kde = stats.gaussian_kde(lv)
        ax.plot(xs, kde(xs), color=SRC[k], lw=2, label=f"{k} (median {10**np.median(lv):.3g})")
        ax.axvline(np.median(lv), color=SRC[k], lw=1, ls=":")
    ax.set_xlabel("log10 within-type per-gene variance (log1p scale)"); ax.set_ylabel("density"); ax.legend(frameon=False, fontsize=7.5)
    ax.set_title(f"{ct}: distribution of per-gene variances\n(n={len(g)} genes, {int(g.n_nuclei.iloc[0])} nuclei, {int(g.n_cells.iloc[0])} cells)", fontsize=9)
    for c, (xk, yk, xl, yl, tag) in enumerate([("var_cell_all", "var_imp_all", "real cell", "imputed", "spearman_var_cell_vs_imp"),
                                                ("var_cell_bio", "var_imp_all", "real cell (Poisson + lib size removed)", "imputed", "spearman_var_cellbio_vs_imp")], start=1):
        ax = axes[r, c]; x = g[xk].values; y = g[yk].values; ok = (x > 0) & (y > 0)
        lx, ly = np.log10(x[ok]), np.log10(y[ok])
        hb = ax.hexbin(lx, ly, gridsize=45, mincnt=1, cmap="Blues", linewidths=0.2)
        lo, hi = np.percentile(lx, 0.5), lx.max(); ax.plot([lo, hi], [lo, hi], color="#7F7F7F", lw=1, ls="--", label="identity")
        p = np.polyfit(lx, ly, 1); ax.plot([lo, hi], np.polyval(p, [lo, hi]), color=SRC["imputed"], lw=1.5, label=f"fit slope {p[0]:.2f}")
        ax.set_xlim(lo - 0.2, hi + 0.2); ax.set_ylim(np.percentile(ly, 0.5) - 0.2, ly.max() + 0.2)
        ax.set_xlabel(f"log10 variance, {xl}"); ax.set_ylabel(f"log10 variance, {yl}")
        ax.set_title(f"{ct}: Spearman = {s[tag]:.3f}" + (f" (well-detected {s['spearman_var_cell_vs_imp_well_detected']:.3f})" if c == 1 else ""), fontsize=9)
        ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    data_rows.append(g[["cell_type", "gene", "det_cell", "det_nuc", "mean_cell", "mean_imp", "var_cell_all", "var_nuc_all", "var_imp_all", "var_cell_bio", "var_cell_libfree", "var_imp_libfree"]])
fig.tight_layout(); save(fig, "V1_focus_distribution_and_rank", pd.concat(data_rows))

# ---------------- V2: all cell types, Spearman of per-gene variance (cell vs imputed), with cell-vs-nucleus as reference
fig, ax = plt.subplots(figsize=(8.5, 3.6))
Sx = S.loc[CT_ORDER]; x = np.arange(len(Sx))
ax.bar(x - 0.2, Sx.spearman_var_cell_vs_imp, 0.4, color=SRC["imputed"], label="real cell vs imputed")
ax.bar(x + 0.2, Sx.spearman_var_cell_vs_nuc, 0.4, color=SRC["nucleus"], label="real cell vs raw nucleus (reference)")
ax.scatter(x - 0.2, Sx.spearman_var_cellbio_vs_imp, color="black", s=14, zorder=3, label="real cell (Poisson + lib size removed) vs imputed")
for i, ct in enumerate(CT_ORDER): ax.text(i, -0.06, f"n={int(Sx.loc[ct, 'n_nuclei'])}", ha="center", va="top", fontsize=6, color="#555", rotation=90)
ax.set_xticks(x); ax.set_xticklabels(CT_ORDER, rotation=45, ha="right"); ax.set_ylabel("Spearman of per-gene\nwithin-type variance"); ax.set_ylim(-0.3, 1.0)
ax.axhline(0, color="#7F7F7F", lw=0.8); ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.42))
ax.set_title("Do the same genes vary most in real and imputed cells? (all genes detected in >1% of cells)", fontsize=9)
save(fig, "V2_spearman_by_cell_type", Sx.reset_index()[["cell_type", "n_genes", "n_nuclei", "spearman_var_cell_vs_imp", "spearman_var_cell_vs_nuc", "spearman_var_cellbio_vs_imp", "spearman_var_cell_vs_imp_well_detected"]])

# ---------------- V3: per-type distributions (ECDF of log10 variance) for all 17 types
fig, axes = plt.subplots(3, 6, figsize=(15, 7), sharex=True, sharey=True); axes = axes.ravel()
for i, ct in enumerate(CT_ORDER):
    ax = axes[i]; g = T[T.cell_type == ct]
    for k, v in (("cell", g.var_cell_all), ("nucleus", g.var_nuc_all), ("imputed", g.var_imp_all)):
        lv = np.sort(np.log10(v[v > 0])); ax.plot(lv, np.linspace(0, 1, len(lv)), color=SRC[k], lw=1.6, label=k)
    ax.set_title(f"{ct}  rho={S.loc[ct, 'spearman_var_cell_vs_imp']:.2f}  IQR ratio={S.loc[ct, 'iqr_ratio_imputed_over_cell']:.2f}", fontsize=7.5)
    ax.set_xlim(-5, 0.7)
axes[-1].axis("off"); axes[0].legend(frameon=False, fontsize=7); fig.supxlabel("log10 within-type per-gene variance"); fig.supylabel("ECDF")
fig.tight_layout(); save(fig, "V3_ecdf_all_cell_types")

# ---------------- V4: Spearman by detection bin, focus types + pooled
fig, ax = plt.subplots(figsize=(6, 3.2))
for ct, col in (("L5 IT", "#0072B2"), ("Oligo", "#009E73"), ("L2/3 IT", "#CC79A7"), ("Pvalb", "#D55E00")):
    b = B[B.cell_type == ct].set_index("bin").reindex(["<10%", "10-30%", "30-70%", ">70%"])
    ax.plot(range(4), b.spearman_var_cell_vs_imp, marker="o", color=col, lw=1.8, label=ct)
ax.set_xticks(range(4)); ax.set_xticklabels(["<10%", "10-30%", "30-70%", ">70%"]); ax.set_xlabel("gene detection rate in real cells of that type")
ax.set_ylabel("Spearman(var real cell, var imputed)"); ax.axhline(0, color="#7F7F7F", lw=0.8); ax.legend(frameon=False, fontsize=8); ax.set_ylim(-0.3, 1)
save(fig, "V4_spearman_by_detection_bin", B[["cell_type", "bin", "n_genes", "spearman_var_cell_vs_imp", "spearman_var_cellbio_vs_imp"]])
print("done")
