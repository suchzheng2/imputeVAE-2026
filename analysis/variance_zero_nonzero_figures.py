#!/usr/bin/env python3
"""Figures + summary tables for the zero/nonzero variance diagnostic (full-data cVAE)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

BASE = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE/notes/experiments/variance_zero_nonzero_fullmodel"
TAB, FIG = f"{BASE}/tables", f"{BASE}/figures"; os.makedirs(FIG, exist_ok=True)
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "verm": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9", "grey": "#7F7F7F"}
SRC = {"nucleus": C["grey"], "imputed": C["blue"], "cell": C["orange"]}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#E5E5E5", "grid.linewidth": 0.6, "axes.axisbelow": True, "pdf.fonttype": 42})
BINS = [(0, 0.1, "<10%"), (0.1, 0.3, "10-30%"), (0.3, 0.7, "30-70%"), (0.7, 1.01, ">70%")]
NEURONAL = ['L2/3 IT', 'L5 ET', 'L5 IT', 'L5/6 NP', 'L6 CT', 'L6 IT', 'L6b', 'Lamp5', 'Pvalb', 'Sncg', 'Sst', 'Vip']


def save(fig, name, data=None):
    fig.savefig(f"{FIG}/{name}.png", dpi=200, bbox_inches="tight"); fig.savefig(f"{FIG}/{name}.pdf", bbox_inches="tight"); plt.close(fig)
    if data is not None: data.to_csv(f"{FIG}/{name}_data.csv", index=False)
    print("saved", name)


T = pd.read_csv(f"{TAB}/per_ct_gene_variance.csv.gz")
T = T[(T.var_cell_all > 1e-4) & (T.det_cell > 0.01)].copy()
T["bin"] = pd.cut(T.det_nuc, [b[0] for b in BINS] + [1.01], labels=[b[2] for b in BINS], right=False)
# law of total variance components
T["imp_within_nz"] = T.det_nuc * T.var_imp_at_nuc_nz.fillna(0); T["imp_within_zero"] = (1 - T.det_nuc) * T.var_imp_at_nuc_zero.fillna(0)
T["imp_mixing"] = (T.var_imp_all - T.imp_within_nz - T.imp_within_zero).clip(lower=0)
T["nuc_within_nz"] = T.det_nuc * T.var_nuc_nz.fillna(0); T["nuc_mixing"] = (T.var_nuc_all - T.nuc_within_nz).clip(lower=0)
T["cell_within_nz"] = T.det_cell * T.var_cell_nz.fillna(0); T["cell_mixing"] = (T.var_cell_all - T.cell_within_nz).clip(lower=0)
T["cell_poisson"] = np.minimum(T.var_cell_nullA, T.var_cell_all); T["cell_libsize"] = (np.minimum(T.var_cell_nullB, T.var_cell_all) - T.cell_poisson).clip(lower=0)
T["cell_bio"] = (T.var_cell_all - T.cell_poisson - T.cell_libsize).clip(lower=0)
T["ratio_vs_total"] = T.var_imp_all / T.var_cell_all
T["ratio_vs_excessA"] = T.var_imp_all / T.bio_excess_cell_A.where(T.bio_excess_cell_A > 1e-4)
T["ratio_vs_excessB"] = T.var_imp_all / T.bio_excess_cell_B.where(T.bio_excess_cell_B > 1e-4)

# ---------------- summary tables
def summ(g):
    return pd.Series({"n_pairs": len(g), "median_var_cell": g.var_cell_all.median(), "median_var_nuc": g.var_nuc_all.median(), "median_var_imp": g.var_imp_all.median(),
                      "median_ratio_imp_over_cell_total": g.ratio_vs_total.median(),
                      "median_frac_cell_var_poisson": (g.cell_poisson / g.var_cell_all).median(), "median_frac_cell_var_libsize": (g.cell_libsize / g.var_cell_all).median(),
                      "median_frac_cell_var_bio": (g.cell_bio / g.var_cell_all).median(),
                      "sum_frac_poisson": g.cell_poisson.sum() / g.var_cell_all.sum(), "sum_frac_libsize": g.cell_libsize.sum() / g.var_cell_all.sum(), "sum_frac_bio": g.cell_bio.sum() / g.var_cell_all.sum(),
                      "sum_ratio_imp_over_total": g.var_imp_all.sum() / g.var_cell_all.sum(), "sum_ratio_imp_over_excessA": g.var_imp_all.sum() / g.bio_excess_cell_A.sum(), "sum_ratio_imp_over_excessB": g.var_imp_all.sum() / g.bio_excess_cell_B.sum(),
                      "median_ratio_imp_over_excessA": g.ratio_vs_excessA.median(), "median_ratio_imp_over_excessB": g.ratio_vs_excessB.median(),
                      "median_varratio_nz_imp_over_nuc": g.var_ratio_nz_imp_over_nuc.median(), "median_varratio_nz_imp_over_cellnz": g.var_ratio_nz_imp_over_cellnz.median(),
                      "median_var_nuc_nz": g.var_nuc_nz.median(), "median_var_nuc_nz_nullB": g.var_nuc_nz_nullB.median(), "median_var_cell_nz": g.var_cell_nz.median(), "median_var_cell_nz_nullB": g.var_cell_nz_nullB.median(),
                      "median_r_paired_nz": g.r_paired_nz.median(), "median_r_paired_all": g.r_paired_all.median(), "median_slope_paired_nz": g.slope_paired_nz.median(),
                      "median_mean_imp_at_zero": g.mean_imp_at_nuc_zero.median(), "median_mean_imp_at_nz": g.mean_imp_at_nuc_nz.median(),
                      "median_gap_nz_minus_zero": (g.mean_imp_at_nuc_nz - g.mean_imp_at_nuc_zero).median(), "median_nuc_nz_mean": g.mean_nuc_nz.median(),
                      "share_imp_var_within_nz": g.imp_within_nz.sum() / g.var_imp_all.sum(), "share_imp_var_within_zero": g.imp_within_zero.sum() / g.var_imp_all.sum(), "share_imp_var_mixing": g.imp_mixing.sum() / g.var_imp_all.sum(),
                      "share_nuc_var_mixing": g.nuc_mixing.sum() / g.var_nuc_all.sum(), "share_cell_var_mixing": g.cell_mixing.sum() / g.var_cell_all.sum(),
                      "median_det_nuc": g.det_nuc.median(), "median_det_cell": g.det_cell.median(), "median_det_cell_nullB": g.det_cell_nullB.median(), "median_det_nuc_nullB": g.det_nuc_nullB.median()})
by_bin = T.groupby("bin").apply(summ); overall = summ(T).to_frame("ALL").T; by_ct = T.groupby("cell_type").apply(summ)
pd.concat([by_bin, overall]).to_csv(f"{TAB}/summary_by_detection_bin.csv"); by_ct.to_csv(f"{TAB}/summary_by_cell_type.csv")
pd.set_option("display.width", 250); print(pd.concat([by_bin, overall]).T.round(3).to_string())

# ---------------- F1: law of total variance by source and detection bin
fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=False); rows = []
for ax, (lo, hi, lab) in zip(axes, BINS):
    g = T[T.bin == lab]
    comp = {"nucleus": [g.nuc_within_nz.mean(), 0, g.nuc_mixing.mean()], "imputed": [g.imp_within_nz.mean(), g.imp_within_zero.mean(), g.imp_mixing.mean()], "real cell": [g.cell_within_nz.mean(), 0, g.cell_mixing.mean()]}
    left = np.zeros(3); ys = np.arange(3)
    for j, (cn, col) in enumerate((("spread among nonzero entries", C["green"]), ("spread among zero entries (imputed only)", C["sky"]), ("zero vs nonzero mixing", C["purple"]))):
        vals = np.array([comp[k][j] for k in comp]); ax.barh(ys, vals, left=left, color=col, edgecolor="white", label=cn if lab == "<10%" else None); left += vals
    ax.set_yticks(ys); ax.set_yticklabels(list(comp)); ax.invert_yaxis(); ax.set_title(f"genes detected in {lab} of nuclei\n(n = {len(g):,} cell type x gene pairs)", fontsize=8.5); ax.set_xlabel("mean within-cell-type variance (log1p)")
    for k in comp: rows.append(dict(bin=lab, source=k, within_nonzero=comp[k][0], within_zero=comp[k][1], mixing=comp[k][2]))
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, fontsize=8, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12), frameon=False); fig.suptitle("Where within-cell-type variance comes from: zero/nonzero mixing vs spread among expressed entries (cells partitioned by their own zeros)", fontsize=10, y=1.06)
save(fig, "F1_variance_sources_by_detection", pd.DataFrame(rows))

# ---------------- F2: technical vs biological in real cells, imputed as marker
fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.2), gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.55}); rows = []
labs = [b[2] for b in BINS] + ["all genes"]; ys = np.arange(len(labs))
for i, lab in enumerate(labs):
    g = T if lab == "all genes" else T[T.bin == lab]; tot = g.var_cell_all.sum()
    parts = [g.cell_poisson.sum() / tot, g.cell_libsize.sum() / tot, g.cell_bio.sum() / tot]; left = 0
    for v, col, nm in zip(parts, (C["grey"], C["sky"], C["green"]), ("Poisson sampling noise", "library-size variation", "biological excess")):
        axes[0].barh(i, v, left=left, color=col, edgecolor="white", label=nm if i == 0 else None); left += v
    imp = g.var_imp_all.sum() / tot; axes[0].plot([imp, imp], [i - 0.42, i + 0.42], color=C["blue"], lw=3, label="imputed variance (same scale)" if i == 0 else None)
    axes[0].text(1.01, i, f"imputed = {imp:.2f} of total\n= {g.var_imp_all.sum() / g.cell_bio.sum():.2f} of biological", va="center", fontsize=7.5)
    rows.append(dict(bin=lab, poisson=parts[0], libsize=parts[1], biological=parts[2], imputed_over_total=imp))
axes[0].set_yticks(ys); axes[0].set_yticklabels(labs); axes[0].invert_yaxis(); axes[0].set_xlim(0, 1.3); axes[0].set_xticks([0, 0.25, 0.5, 0.75, 1]); axes[0].set_xlabel("share of real whole-cell within-cell-type variance"); axes[0].legend(fontsize=7.5, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.16), frameon=False)
axes[0].set_title("Real-cell variance split into technical and biological parts\n(Poisson simulation; rows = gene detection rate in nuclei)", fontsize=9)
r = pd.DataFrame({"vs total\nvariance": T.ratio_vs_total, "vs total minus\nPoisson noise": T.ratio_vs_excessA, "vs biological\nexcess": T.ratio_vs_excessB})
bp = axes[1].boxplot([r[c].dropna().clip(upper=5) for c in r], vert=False, showfliers=False, patch_artist=True, widths=0.6)
for p_ in bp["boxes"]: p_.set_facecolor(C["blue"]); p_.set_alpha(0.6)
axes[1].set_yticks([1, 2, 3]); axes[1].set_yticklabels(r.columns, fontsize=8); axes[1].axvline(1, color=C["verm"], lw=1.5, ls="--"); axes[1].set_xlabel("imputed variance / reference variance, per cell type x gene"); axes[1].set_title("Variance ratio depends on the reference", fontsize=9); axes[1].invert_yaxis()
for j, c in enumerate(r): axes[1].text(r[c].median(), j + 1.42, f"median {r[c].median():.2f}", ha="center", fontsize=7.5)
save(fig, "F2_technical_vs_biological", pd.DataFrame(rows))

# ---------------- F3: per-pair scatter imputed var vs references
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, (col, ttl) in zip(axes, (("var_cell_all", "real-cell total variance"), ("bio_excess_cell_A", "real-cell variance minus Poisson noise"), ("bio_excess_cell_B", "real-cell biological excess"))):
    g = T[(T[col] > 1e-4) & (T.var_imp_all > 1e-5)]
    ax.hexbin(np.log10(g[col]), np.log10(g.var_imp_all), gridsize=70, cmap="Blues", mincnt=1, bins="log", linewidths=0)
    lim = [-4, 1]; ax.plot(lim, lim, color=C["verm"], lw=1.5, ls="--", label="imputed = reference"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(f"log10 {ttl}"); ax.set_ylabel("log10 imputed variance"); rho = stats.spearmanr(g[col], g.var_imp_all)[0]
    ax.set_title(f"{ttl}\nSpearman across pairs = {rho:.2f}, median ratio = {(g.var_imp_all / g[col]).median():.2f}", fontsize=9)
axes[0].legend(fontsize=7, loc="upper left")
save(fig, "F3_imputed_vs_reference_variance", T[["cell_type", "gene", "var_imp_all", "var_cell_all", "bio_excess_cell_A", "bio_excess_cell_B"]].sample(20000, random_state=0))

# ---------------- F4: nonzero partition
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8)); labs = [b[2] for b in BINS]
def boxes(ax, cols, names, colors, ttl, xlabel, ref=None, clip=3):
    w = 0.8 / len(cols)
    for j, (c, nm, colr) in enumerate(zip(cols, names, colors)):
        data = [T.loc[T.bin == lab, c].dropna().clip(upper=clip) for lab in labs]
        bp = ax.boxplot(data, positions=np.arange(len(labs)) + (j - (len(cols) - 1) / 2) * w, widths=w * 0.9, showfliers=False, patch_artist=True)
        for p_ in bp["boxes"]: p_.set_facecolor(colr); p_.set_alpha(0.75)
        for m_ in bp["medians"]: m_.set_color("black")
        ax.plot([], [], color=colr, lw=6, alpha=0.75, label=nm)
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs); ax.set_xlabel("gene detection rate in nuclei (within cell type)"); ax.set_ylabel(xlabel); ax.set_title(ttl, fontsize=9); ax.legend(fontsize=7)
    if ref is not None: ax.axhline(ref, color=C["verm"], lw=1.2, ls="--")
boxes(axes[0], ["var_nuc_nz", "var_nuc_nz_nullB", "var_imp_at_nuc_nz", "var_cell_nz"], ["nucleus (observed)", "nucleus technical null", "imputed (same entries)", "real cell nonzero"], [C["grey"], "#C9C9C9", C["blue"], C["orange"]], "Variance among entries where the nucleus is nonzero", "variance (log1p)", clip=1.5)
boxes(axes[1], ["var_ratio_nz_imp_over_nuc", "var_ratio_nz_imp_over_cellnz"], ["imputed / nucleus", "imputed / real-cell nonzero"], [C["blue"], C["orange"]], "Variance ratio on nonzero entries", "ratio", ref=1)
boxes(axes[2], ["r_paired_nz", "r_paired_all"], ["nonzero entries only", "all entries (zeros included)"], [C["green"], C["purple"]], "Per-nucleus information carried through:\ncorrelation of nucleus input with its imputed value", "Pearson r within cell type x gene", ref=0, clip=1)
save(fig, "F4_nonzero_entries", T.groupby("bin")[["var_nuc_nz", "var_nuc_nz_nullB", "var_imp_at_nuc_nz", "var_cell_nz", "var_ratio_nz_imp_over_nuc", "var_ratio_nz_imp_over_cellnz", "r_paired_nz", "r_paired_all"]].median().reset_index())

# ---------------- F5: zero partition + detection
fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
T["gap"] = T.mean_imp_at_nuc_nz - T.mean_imp_at_nuc_zero
boxes(axes[0], ["mean_imp_at_nuc_zero", "mean_imp_at_nuc_nz", "mean_cell"], ["imputed where nucleus = 0", "imputed where nucleus > 0", "real-cell mean (all cells)"], [C["sky"], C["blue"], C["orange"]], "What the model predicts where the nucleus read zero", "mean value (log1p)", clip=6)
boxes(axes[1], ["var_imp_at_nuc_zero", "var_imp_at_nuc_nz"], ["imputed where nucleus = 0", "imputed where nucleus > 0"], [C["sky"], C["blue"]], "Imputed spread within each partition", "variance (log1p)", clip=0.6)
g = T.sample(min(len(T), 60000), random_state=1)
axes[2].hexbin(g.det_cell_nullB, g.det_cell, gridsize=60, cmap="Oranges", mincnt=1, bins="log", linewidths=0); axes[2].plot([0, 1], [0, 1], color=C["verm"], ls="--", lw=1.5)
axes[2].set_xlabel("detection rate expected from sampling\n(Poisson + library size)"); axes[2].set_ylabel("observed detection rate in real cells"); axes[2].set_title(f"Are real-cell zeros explained by sampling?\nmedian observed {T.det_cell.median():.2f} vs expected {T.det_cell_nullB.median():.2f}", fontsize=9)
save(fig, "F5_zero_entries", T.groupby("bin")[["mean_imp_at_nuc_zero", "mean_imp_at_nuc_nz", "mean_cell", "var_imp_at_nuc_zero", "var_imp_at_nuc_nz", "det_cell", "det_cell_nullB", "det_nuc", "det_nuc_nullB"]].median().reset_index())

# ---------------- F6: example genes
E = pd.read_csv(f"{TAB}/example_gene_points.csv.gz")
for ct in E.cell_type.unique():
    sub = E[E.cell_type == ct]; gl = sub.drop_duplicates("gene")[["gene", "det_bin"]].values.tolist()
    fig, axes = plt.subplots(2, len(gl), figsize=(3.6 * len(gl), 6.4)); rng = np.random.default_rng(0)
    for j, (gname, db) in enumerate(gl):
        p = sub[(sub.gene == gname) & (sub.kind == "nucleus_vs_imputed")]; c = sub[(sub.gene == gname) & (sub.kind == "real_cell")]
        z = p.nucleus_log1p == 0; ax = axes[0, j]
        ax.scatter(p.nucleus_log1p[z] + rng.normal(0, 0.03, z.sum()), p.imputed[z], s=3, alpha=0.25, color=C["sky"], label="nucleus = 0")
        ax.scatter(p.nucleus_log1p[~z] + rng.normal(0, 0.03, (~z).sum()), p.imputed[~z], s=3, alpha=0.25, color=C["blue"], label="nucleus > 0")
        r_ = np.corrcoef(p.nucleus_log1p[~z], p.imputed[~z])[0, 1] if (~z).sum() > 20 else np.nan
        ax.set_title(f"{gname} ({ct}; detected in {100 * (~z).mean():.0f}% of nuclei)\nr among nonzero = {r_:.2f}", fontsize=8.5); ax.set_xlabel("original nucleus log1p(count)"); ax.set_ylabel("imputed value")
        if j == 0: ax.legend(fontsize=7, markerscale=3)
        ax = axes[1, j]; hi = max(p.imputed.max(), c.cell_log1p.max(), p.nucleus_log1p.max()); bins = np.linspace(0, hi, 40)
        ax.hist(p.nucleus_log1p, bins=bins, density=True, histtype="step", lw=2, color=SRC["nucleus"], label="nucleus"); ax.hist(p.imputed, bins=bins, density=True, histtype="step", lw=2, color=SRC["imputed"], label="imputed"); ax.hist(c.cell_log1p, bins=bins, density=True, histtype="step", lw=2, color=SRC["cell"], label="real cell")
        ax.set_xlabel("log1p expression"); ax.set_ylabel("density"); ax.set_ylim(0, np.percentile(np.histogram(p.imputed, bins=bins, density=True)[0], 100) * 1.15)
        if j == 0: ax.legend(fontsize=7)
    fig.suptitle(f"Example genes in {ct}, one per detection bin (highest biological excess variance in the bin)", fontsize=10, y=1.01)
    save(fig, f"F6_example_genes_{ct.replace(' ', '_')}")

# ---------------- F7: gene-gene structure
G = pd.read_csv(f"{TAB}/genegene_structure.csv")
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8)); x = np.arange(len(G))
axes[0].bar(x - 0.2, G.spearman_nuc_vs_cell, 0.38, color=SRC["nucleus"], label="raw nuclei vs real cells", edgecolor="white"); axes[0].bar(x + 0.2, G.spearman_imp_vs_cell, 0.38, color=SRC["imputed"], label="imputed vs real cells", edgecolor="white")
axes[0].set_xticks(x); axes[0].set_xticklabels(G.cell_type, rotation=45, ha="right"); axes[0].set_ylabel("Spearman between gene-gene correlation matrices"); axes[0].legend(fontsize=7); axes[0].set_title("Within-cell-type co-variation structure\n(300 genes; library-size axis removed)", fontsize=9)
for ax, col, nm, colr in ((axes[1], "r_nuc", "raw nuclei", "Greys"), (axes[2], "r_imp", "imputed", "Blues")):
    P = pd.read_csv(f"{TAB}/genegene_corr_pairs_L5_IT.csv.gz"); ax.hexbin(P.r_cell, P[col], gridsize=60, cmap=colr, mincnt=1, bins="log", linewidths=0); ax.plot([-0.5, 1], [-0.5, 1], color=C["verm"], ls="--", lw=1.2)
    ax.set_xlabel("gene-gene correlation in real L5 IT cells"); ax.set_ylabel(f"gene-gene correlation in {nm}"); ax.set_title(f"L5 IT: {nm} (Spearman {stats.spearmanr(P.r_cell, P[col])[0]:.2f})", fontsize=9)
save(fig, "F7_genegene_structure", G)

# ---------------- F8: per cell type
by_ct = by_ct.sort_values("n_pairs"); ct_n = T.groupby("cell_type").n_nuclei.first().reindex(by_ct.index)
by_ct = by_ct.loc[ct_n.sort_values().index]
fig, ax = plt.subplots(figsize=(9, 4.2)); x = np.arange(len(by_ct))
ax.bar(x - 0.27, by_ct.sum_ratio_imp_over_total, 0.26, color=C["grey"], label="imputed / total real-cell variance", edgecolor="white"); ax.bar(x, by_ct.sum_ratio_imp_over_excessA, 0.26, color=C["sky"], label="imputed / (total - Poisson)", edgecolor="white"); ax.bar(x + 0.27, by_ct.sum_ratio_imp_over_excessB, 0.26, color=C["green"], label="imputed / biological excess", edgecolor="white")
ax.axhline(1, color=C["verm"], ls="--", lw=1.2); ax.set_xticks(x); ax.set_xticklabels([f"{c}\n(n={int(ct_n[c]):,})" for c in by_ct.index], rotation=45, ha="right", fontsize=7.5); ax.set_ylabel("variance ratio (sums over genes)"); ax.legend(fontsize=7.5); ax.set_title("Variance retention per cell type, ordered by number of nuclei", fontsize=9.5)
save(fig, "F8_per_cell_type", by_ct.reset_index())

# ---------------- F9: library-size axis and effective dimensionality (add-on tables)
if os.path.exists(f"{TAB}/libsize_r2_per_ct_gene.csv.gz"):
    R = pd.read_csv(f"{TAB}/libsize_r2_per_ct_gene.csv.gz").merge(T[["cell_type", "gene", "var_imp_all", "var_cell_all", "var_nuc_all", "bio_excess_cell_A", "bio_excess_cell_B", "cell_poisson"]], on=["cell_type", "gene"])
    D = pd.read_csv(f"{TAB}/dimensionality.csv")
    sh = {"nucleus": (R.r2_lib_nuc * R.var_nuc_all).sum() / R.var_nuc_all.sum(), "imputed": (R.r2_lib_imp * R.var_imp_all).sum() / R.var_imp_all.sum(), "real cell": (R.r2_lib_cell * R.var_cell_all).sum() / R.var_cell_all.sum()}
    imp_nolib = (R.var_imp_all * (1 - R.r2_lib_imp.fillna(0))).sum()
    ladder = pd.DataFrame({"comparison": ["imputed total /\nreal-cell total", "imputed total /\nreal-cell minus\nPoisson noise", "imputed minus\nlibrary-size axis /\nreal-cell biological\nexcess"],
                           "ratio": [R.var_imp_all.sum() / R.var_cell_all.sum(), R.var_imp_all.sum() / R.bio_excess_cell_A.sum(), imp_nolib / R.bio_excess_cell_B.sum()]})
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4), gridspec_kw={"wspace": 0.75})
    axes[0].bar(list(sh), list(sh.values()), color=[SRC["nucleus"], SRC["imputed"], SRC["cell"]], edgecolor="white", width=0.6)
    for i, v in enumerate(sh.values()): axes[0].text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=9)
    axes[0].set_ylim(0, 1); axes[0].set_ylabel("share of within-cell-type variance explained\nby per-cell library size (variance-weighted R2)"); axes[0].set_title("Most imputed variance is the library-size axis", fontsize=9.5)
    y = np.arange(len(D))
    for col, nm, colr in (("pr_nuc", "raw nuclei", SRC["nucleus"]), ("pr_cell", "real cells", SRC["cell"]), ("pr_imp", "imputed", SRC["imputed"])):
        axes[1].scatter(D[col], y, s=55, color=colr, label=nm, zorder=3, edgecolor="white", linewidth=1)
    axes[1].set_xscale("log"); axes[1].set_yticks(y); axes[1].set_yticklabels(D.cell_type); axes[1].invert_yaxis(); axes[1].set_xlabel("effective number of dimensions (participation ratio, log scale)\n300 most biologically variable genes per cell type"); axes[1].legend(fontsize=7.5, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.22), frameon=False); axes[1].set_title("Imputed within-cell-type variation is ~1-dimensional", fontsize=9.5)
    axes[2].barh(ladder.comparison, ladder.ratio, color=[C["grey"], C["sky"], C["green"]], edgecolor="white", height=0.6)
    for i, v in enumerate(ladder.ratio): axes[2].text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)
    axes[2].set_xlim(0, 0.5); axes[2].invert_yaxis(); axes[2].set_xlabel("variance retained (sum over genes and cell types)"); axes[2].set_title("Like-for-like variance retention", fontsize=9.5)
    save(fig, "F9_libsize_axis_and_dimensionality", pd.concat([pd.DataFrame([sh]).assign(table="libsize_share"), D.assign(table="dimensionality"), ladder.assign(table="ladder")]))
