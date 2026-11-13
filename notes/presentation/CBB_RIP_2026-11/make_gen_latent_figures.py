"""Talk figures for the generalisation and latent-finding slides (2026-10-28).

ripfig_generalisation.png : left, unseen-chemistry (10x v2) ladder of gene corr (libsize review §6.4a);
                            right, Bakken 2018 pooled predicted vs empirical delta (residual decode, arm A quantile map),
                            recomputed from the saved imputation and the processed h5ads, with ceiling and baseline.
ripfig_latent_finding.png : composite of recorded panels: L1 (6 of 32 dims active), F9 (0.82 of imputed variance is
                            library size; ~1-D), plus the latent-width control table (review §6.7).
"""
import os, json, numpy as np, pandas as pd, anndata as ad
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE"
OUT = f"{ROOT}/notes/presentation/CBB_RIP_2026-11/figures"
C_NUC, C_IMP, C_CELL, INK, MUTED, GRID = "#898781", "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e1e0d9"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})

# ---------------------------------------------------------------- generalisation
fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), gridspec_kw={"width_ratios": [1, 1.1]})
# left: v2 ladder (gene corr, v2 nuclei -> v2 cells; libsize review §6.4a + 2026-10-27 update)
rows = [("old model (no covariate), July", 0.593, C_NUC),
        ("covariate model, one chemistry-wide scale", 0.843, C_IMP),
        ("+ predicted type (reference classifier, 84 %)", 0.903, C_IMP),
        ("+ cell-type label at inference", 0.926, C_CELL)]
ax = axes[0]; y = np.arange(len(rows))[::-1]
for yi, (lab, v, c) in zip(y, rows):
    ax.barh(yi, v, color=c, height=0.55); ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=12, color=INK)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=11.5); ax.set_xlim(0, 1.0)
ax.set_xlabel("gene correlation, v2 nuclei -> v2 cells (unseen chemistry)")
ax.set_title("Unseen chemistry: the transformation transfers once\nthe output scale and the type are supplied label-free", loc="left", fontsize=13)
ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True); ax.spines["left"].set_visible(False)
ax.text(0.0, -1.05, "same v3-trained model; label-free rows use no cell-type label at inference", fontsize=10, color=MUTED, transform=ax.transData, va="top")

# right: Bakken pooled delta scatter
ev = json.load(open(f"{ROOT}/run/external_eval/bakken2018/eval_armA_quantile_map_target.json"))
nuc = ad.read_h5ad(ev["nuclei"]); cell = ad.read_h5ad(ev["cells"])
present = nuc.var["present"].values.astype(bool)
Xn = np.log1p(np.asarray(nuc.X, dtype=np.float32)); Xc = np.log1p(np.asarray(cell.X, dtype=np.float32))
res = np.load(f"{ROOT}/run/external_eval/bakken2018/imputed_residual_armA_quantile_map_target.npy")
ct_n = nuc.obs[ev["label_col"]].astype(str).values; ct_c = cell.obs[ev["label_col"]].astype(str).values
E, P = [], []
for ct in sorted(set(ct_n) & set(ct_c)):
    mn, mc = ct_n == ct, ct_c == ct
    if mn.sum() < 5 or mc.sum() < 5: continue
    E.append((Xc[mc].mean(0) - Xn[mn].mean(0))[present]); P.append((res[mn].mean(0) - Xn[mn].mean(0))[present])
emp, pred = np.concatenate(E), np.concatenate(P)
r = np.corrcoef(emp, pred)[0, 1]
rel = json.load(open(f"{ROOT}/run/external_eval/bakken2018/delta_reliability_exon_intron.json"))
ceil = rel["pooled_full_sample_reliability_spearman_brown"]
ax = axes[1]
ax.hexbin(emp, pred, gridsize=70, cmap="Blues", mincnt=1, linewidths=0.2, extent=(-1.5, 3.5, -1.5, 3.5))
lim = [-1.5, 3.5]; ax.set_xlim(lim); ax.set_ylim(lim)
ax.plot(lim, lim, color=MUTED, lw=1, ls="--"); ax.axhline(0, color=GRID, lw=1); ax.axvline(0, color=GRID, lw=1)
ax.set_xlabel("empirical delta: SMART-seq cell - nucleus (log1p mean)"); ax.set_ylabel("predicted delta: model shift on the nucleus")
ax.set_title(f"Matched hold-out: Bakken 2018 SMART-seq, 11 clusters\n{len(emp):,} cluster x gene pairs; r = {r:.2f}; ceiling {ceil:.2f}; sign 0.94 = ceiling",
             loc="left", fontsize=13)
ax.text(0.03, 0.97, "baseline 'predicted type + its mean training shift':\nr 0.56 (same as the model)", transform=ax.transAxes, va="top", fontsize=11, color=MUTED)
fig.tight_layout(); fig.savefig(f"{OUT}/ripfig_generalisation.png", dpi=200); plt.close(fig)
print("bakken pooled r", round(float(r), 3), "ceiling", round(ceil, 3))

# ---------------------------------------------------------------- latent finding (from recorded tables)
pdim = pd.read_csv(f"{ROOT}/notes/experiments/variance_zero_nonzero_fullmodel/latent_usage/per_dimension.csv")
pdim = pdim[pdim["source"] == "nuclei"].sort_values("var_total", ascending=False).reset_index(drop=True)
fig = plt.figure(figsize=(14, 7.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.55], width_ratios=[1.6, 1], wspace=0.35, hspace=0.45)
ax = fig.add_subplot(gs[0, 0]); x = np.arange(len(pdim)); w = 0.4
ax.bar(x - w / 2, pdim["var_between_ct"], w, color=C_CELL, label="between cell types (identity)")
ax.bar(x + w / 2, pdim["var_within_ct"], w, color=C_IMP, label="within cell type")
lib = pdim["r2_libsize_within_ct"].values.argmax()
ax.annotate("library-size dimension\n(R2 with log library 0.88)", xy=(lib + w / 2, pdim["var_within_ct"][lib]), xytext=(lib + 6, 2.0),
            fontsize=11, color=INK, arrowprops=dict(arrowstyle="->", color=MUTED))
ax.axhline(0.01, color=MUTED, lw=1, ls="--"); ax.text(len(pdim) - 0.5, 0.012, "active threshold", ha="right", fontsize=10, color=MUTED)
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(pdim["dim"], fontsize=9); ax.set_xlabel("latent dimension, sorted by total variance")
ax.set_ylabel("variance of the posterior mean (log)"); ax.legend(frameon=False, fontsize=11)
n_active = int((pdim["var_total"] > 0.01).sum())
ax.set_title(f"{n_active} of 32 latent dimensions are used; the rest sit at the prior (40,166 nuclei)", loc="left", fontsize=13)
ax = fig.add_subplot(gs[0, 1])
vals = [("raw nuclei", 0.06, C_NUC), ("imputed", 0.82, C_IMP), ("real cells", 0.24, C_CELL)]
ax.bar([v[0] for v in vals], [v[1] for v in vals], color=[v[2] for v in vals], width=0.6)
for i, v in enumerate(vals): ax.text(i, v[1] + 0.02, f"{v[1]:.2f}", ha="center", fontsize=12)
ax.set_ylim(0, 1); ax.set_ylabel("share of within-type variance\nexplained by library size")
ax.set_title("Imputed within-type variation is one axis:\nlibrary size (one dim carries 95 %)", loc="left", fontsize=13)
ax = fig.add_subplot(gs[1, :]); ax.axis("off")
tbl = [["latent width", "active dims", "gene corr (per-type decode)", "classification of imputed cells", "types lost"],
       ["32 (reported)", "4", "0.970", "0.967", "none"],
       ["8", "2", "0.957", "0.935", "L5 ET, Sncg merged"],
       ["4", "1", "0.910", "0.557", "all but 5 large types"]]
t = ax.table(cellText=tbl[1:], colLabels=tbl[0], loc="center", cellLoc="center", colWidths=[0.15, 0.12, 0.22, 0.25, 0.26])
t.auto_set_font_size(False); t.set_fontsize(12); t.scale(1, 1.6)
for (ri, ci), c in t.get_celld().items():
    c.set_edgecolor(GRID)
    if ri == 0: c.set_text_props(weight="bold"); c.set_facecolor("#eef3fa")
ax.set_title("Latent-width control (covariate model): 2 identity dimensions still place 15 of 17 types; capacity is not the limit", fontsize=12.5, loc="left", color=INK, pad=2)
fig.suptitle("A nucleus determines its type-level cytoplasm plus its RNA content: 2 to 4 identity dims, one library-size axis", x=0.01, ha="left", fontsize=14, color=INK)
fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.04); fig.savefig(f"{OUT}/ripfig_latent_finding.png", dpi=170); plt.close(fig)
print("wrote", OUT)
