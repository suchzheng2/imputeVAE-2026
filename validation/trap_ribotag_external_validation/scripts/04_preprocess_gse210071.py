#!/usr/bin/env python3
"""Phase 4A: GSE210071 (cultured cortical cells, cytosol / nucleus / whole cell, FPKM).

Design: 3 cultures (S1,S2,S4) x 3 compartments (C,N,W) x 2 stimulation (Ctl,Bic). ALL treated with
triptolide 3 h. Primary = Ctl; Bic = sensitivity. Compartments are paired within culture.

Pre-declared choices:
  * value = log2(FPKM + 0.1). eps=0.1 sits between the 5th (0.005) and median (0.27) of nonzero FPKM
    and caps log-ratios of near-zero genes at ~|log2(0.1)|.
  * expressed = mean FPKM >= 1 in at least one compartment of the condition (genes below are not tested).
  * effect per gene = mean over cultures of paired difference (C - N, W - N); SE from the 3 paired
    differences; paired t with df=2; BH FDR within each contrast.
Outputs: outputs/normalized/GSE210071_log2fpkm.tsv.gz, outputs/tables/GSE210071_contrasts_{Ctl,Bic}.tsv,
         outputs/qc/GSE210071_qc_*.{png,csv}
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

TASK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPS = 0.1
MIN_FPKM = 1.0

meta = pd.read_csv(f"{TASK}/outputs/metadata/GSE210071_samples.tsv", sep="\t")
meta = meta[meta.platform == "GPL17021"]
X = pd.read_csv(f"{TASK}/outputs/harmonized/GSE210071_symbol_matrix.tsv.gz", sep="\t", index_col=0)
X = X[meta.column.tolist()]
L = np.log2(X + EPS)
L.to_csv(f"{TASK}/outputs/normalized/GSE210071_log2fpkm.tsv.gz", sep="\t")

# ---------------- QC
qc = pd.DataFrame({"column": X.columns,
                   "detected_genes_fpkm_gt1": (X > 1).sum().values,
                   "detected_genes_fpkm_gt0": (X > 0).sum().values,
                   "total_fpkm": X.sum().values.round(0)}).merge(meta[["column", "compartment", "treatment", "batch"]], on="column")
qc.to_csv(f"{TASK}/outputs/qc/GSE210071_qc_detected.csv", index=False)
expressed_any = X.index[(X > MIN_FPKM).sum(axis=1) >= 3]
Lx = L.loc[expressed_any]
cor = Lx.corr(method="spearman")
cor.to_csv(f"{TASK}/outputs/qc/GSE210071_qc_sample_spearman.csv")
fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(cor, cmap="viridis", vmin=cor.values.min(), ax=ax, square=True, cbar_kws={"label": "Spearman"})
ax.set_title("GSE210071 sample correlation (log2 FPKM, expressed genes)")
plt.tight_layout(); plt.savefig(f"{TASK}/outputs/qc/GSE210071_qc_sample_spearman.png", dpi=150); plt.close()
# PCA
Z = Lx.sub(Lx.mean(axis=1), axis=0).T
U, S, Vt = np.linalg.svd(Z.values, full_matrices=False)
pcs = pd.DataFrame(U[:, :2] * S[:2], index=Z.index, columns=["PC1", "PC2"]).join(meta.set_index("column")[["compartment", "treatment", "batch"]])
pcs["var_PC1"] = S[0]**2 / (S**2).sum(); pcs["var_PC2"] = S[1]**2 / (S**2).sum()
pcs.to_csv(f"{TASK}/outputs/qc/GSE210071_qc_pca.csv")
fig, ax = plt.subplots(figsize=(6, 5))
mk = {"triptolide 3h + unstimulated": "o", "triptolide 3h + Bicuculline-1hr": "^"}
for (comp, trt), g in pcs.groupby(["compartment", "treatment"]):
    ax.scatter(g.PC1, g.PC2, label=f"{comp} / {'Ctl' if 'unstim' in trt else 'Bic'}", marker=mk[trt], s=70)
ax.set_xlabel(f"PC1 ({pcs.var_PC1.iloc[0]*100:.0f}%)"); ax.set_ylabel(f"PC2 ({pcs.var_PC2.iloc[0]*100:.0f}%)")
ax.legend(fontsize=8); ax.set_title("GSE210071 PCA"); plt.tight_layout()
plt.savefig(f"{TASK}/outputs/qc/GSE210071_qc_pca.png", dpi=150); plt.close()

# ---------------- contrasts
def contrast(cond):
    cols = {c: [f"{s}_{c}_{cond}" for s in ("S1", "S2", "S4")] for c in ("C", "N", "W")}
    expr = X.index[(X[cols["C"]].mean(axis=1) >= MIN_FPKM) | (X[cols["N"]].mean(axis=1) >= MIN_FPKM) | (X[cols["W"]].mean(axis=1) >= MIN_FPKM)]
    out = pd.DataFrame(index=expr)
    for name, a in (("cyto_minus_nuc", "C"), ("whole_minus_nuc", "W"), ("whole_minus_cyto", "C")):
        b = "N" if name != "whole_minus_cyto" else "C"
        aa = "W" if name == "whole_minus_cyto" else a
        d = L.loc[expr, cols[aa]].values - L.loc[expr, cols[b]].values   # paired by culture
        m = d.mean(axis=1); se = d.std(axis=1, ddof=1) / np.sqrt(d.shape[1])
        t = np.where(se > 0, m / np.where(se > 0, se, 1), 0.0)
        p = 2 * stats.t.sf(np.abs(t), df=d.shape[1] - 1)
        fdr = stats.false_discovery_control(p) if hasattr(stats, "false_discovery_control") else bh(p)
        out[f"{name}_log2fc"] = m; out[f"{name}_se"] = se; out[f"{name}_t"] = t; out[f"{name}_p"] = p; out[f"{name}_fdr"] = fdr
        for i, s in enumerate(("S1", "S2", "S4")):
            out[f"{name}_{s}"] = d[:, i]
    for c in ("C", "N", "W"):
        out[f"mean_log2fpkm_{c}"] = L.loc[expr, cols[c]].mean(axis=1)
        out[f"mean_fpkm_{c}"] = X.loc[expr, cols[c]].mean(axis=1)
    out.index.name = "gene"
    return out


def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p); r = np.empty(n); r[o] = np.arange(1, n + 1)
    q = np.minimum.accumulate((p * n / r)[o][::-1])[::-1]; out = np.empty(n); out[o] = np.clip(q, 0, 1); return out


for cond in ("Ctl", "Bic"):
    t = contrast(cond)
    t.to_csv(f"{TASK}/outputs/tables/GSE210071_contrasts_{cond}.tsv", sep="\t")
    sig = (t["cyto_minus_nuc_fdr"] < 0.1).sum()
    print(f"{cond}: {len(t)} expressed genes; C-N median log2FC {t.cyto_minus_nuc_log2fc.median():+.3f}, "
          f"W-N median {t.whole_minus_nuc_log2fc.median():+.3f}; C-N FDR<0.1: {sig}; "
          f"replicate consistency (Spearman S1 vs S2 C-N): {stats.spearmanr(t.cyto_minus_nuc_S1, t.cyto_minus_nuc_S2)[0]:.3f}")
    for g in ["Xist", "Meg3", "Snhg11", "Snap25", "Camk2n1", "Tmsb4x", "Sst", "Actb"]:
        if g in t.index:
            r = t.loc[g]; print(f"   {g:8s} C-N {r.cyto_minus_nuc_log2fc:+.2f}  W-N {r.whole_minus_nuc_log2fc:+.2f}  meanFPKM_N {r.mean_fpkm_N:.1f}")
print("QC:", qc.to_string(index=False))
