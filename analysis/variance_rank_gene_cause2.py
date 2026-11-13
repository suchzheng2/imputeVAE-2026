#!/usr/bin/env python3
"""Is the Gm49164 / Actb module in real L5 IT cells a 'cytoplasmic content' axis of the whole-cell preparation?

Per real cell, two whole-cell state scores that a nucleus cannot have:
  cyto_score = mean z of the 300 most cell-enriched genes (largest mean_cell - mean_nucleus, detected >20% in cells)
               minus mean z of the 300 most nucleus-enriched genes
  ribo_frac  = share of counts from Rpl*/Rps* genes
Then, for the example genes and for the Gm49164 partner score: correlation with the two scores, overall and within
sequencing library; variance of the scores explained by library. Same scores in nuclei for reference.
Output: variance_rank/tables/gene_cause2_L5IT.csv
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats
sys.path.insert(0, "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE/analysis")
from variance_rank_gene_cause import load_ct, anova_r2, resid, onehot, PROJECT, TAB

GENES = ["Gm49164", "Gm15810", "Actb", "Gfra1", "Pcp4", "Rorb", "Calm2", "Malat1"]


def main():
    gene_list = pd.read_csv(f"{PROJECT}/data/training_gene_list_v3.csv", header=None)[0].tolist()
    common = pd.read_csv(f"{PROJECT}/data/common_genes.csv")["common_genes"].tolist()
    cts = pd.read_csv(f"{PROJECT}/data/common_cell_types.csv")["common_cell_types"].tolist()
    Lc, names, lib_c, ng_c, _ = load_ct("10X_cells_v2_v3_AIBS", "c3A", cts, gene_list, common)
    Ln, _, lib_n, ng_n, _ = load_ct("10X_nuclei_v2_v3_AIBS", "n3A", cts, gene_list, common)
    det_c = (Lc > 0).mean(0); det_n = (Ln > 0).mean(0)
    delta = Lc.mean(0) - Ln.mean(0)
    ok = (det_c > 0.2) & (det_n > 0.2)
    cell_enr = np.argsort(-np.where(ok, delta, -np.inf))[:300]; nuc_enr = np.argsort(np.where(ok, delta, np.inf))[:300]
    ribo = np.array([n.startswith(("Rpl", "Rps")) and not n.startswith("Rpsa") for n in names])
    print(f"cell-enriched genes e.g. {names[cell_enr[:8]].tolist()}; nucleus-enriched e.g. {names[nuc_enr[:8]].tolist()}; ribo genes {ribo.sum()}", flush=True)

    def scores(L):
        Z = (L - L.mean(0)) / (L.std(0) + 1e-8)
        cyto = Z[:, cell_enr].mean(1) - Z[:, nuc_enr].mean(1)
        cnt = np.expm1(L); rf = cnt[:, ribo].sum(1) / cnt.sum(1)
        return cyto, rf, np.log(cnt.sum(1))
    cyto_c, rf_c, ll_c = scores(Lc); cyto_n, rf_n, ll_n = scores(Ln)
    # Gm49164 partner score in cells (same definition as before)
    j = int(np.where(names == "Gm49164")[0][0]); Zc = (Lc - Lc.mean(0)) / (Lc.std(0) + 1e-8)
    r = (Zc * Zc[:, [j]]).mean(0); r[j] = 0; r[det_c <= 0.05] = 0; top = np.argsort(-np.abs(r))[:30]
    psc = (Zc[:, top] * np.sign(r[top])).mean(1)
    rows = [dict(gene="[cyto_score]", source="cell", r2_library=anova_r2(cyto_c, lib_c), r_loglib=stats.pearsonr(cyto_c, ll_c)[0], r_ribo_frac=stats.pearsonr(cyto_c, rf_c)[0]),
            dict(gene="[cyto_score]", source="nucleus", r2_library=anova_r2(cyto_n, lib_n), r_loglib=stats.pearsonr(cyto_n, ll_n)[0], r_ribo_frac=stats.pearsonr(cyto_n, rf_n)[0]),
            dict(gene="[ribo_frac]", source="cell", r2_library=anova_r2(rf_c, lib_c), r_loglib=stats.pearsonr(rf_c, ll_c)[0]),
            dict(gene="[Gm49164 partner score]", source="cell", r2_library=anova_r2(psc, lib_c), r_cyto=stats.pearsonr(psc, cyto_c)[0],
                 r_cyto_within_lib=stats.pearsonr(resid(psc, onehot(lib_c)), resid(cyto_c, onehot(lib_c)))[0], r_ribo_frac=stats.pearsonr(psc, rf_c)[0])]
    for g in GENES:
        k = int(np.where(names == g)[0][0])
        for src, L, lib, cy, rf, ll in (("cell", Lc, lib_c, cyto_c, rf_c, ll_c), ("nucleus", Ln, lib_n, cyto_n, rf_n, ll_n)):
            y = L[:, k]; oh = onehot(lib)
            rows.append(dict(gene=g, source=src, det=float((y > 0).mean()), var=float(y.var()),
                             r_cyto=stats.pearsonr(y, cy)[0], r_cyto_within_lib=stats.pearsonr(resid(y, oh), resid(cy, oh))[0],
                             r_cyto_within_lib_loglib=stats.pearsonr(resid(y, oh + [ll]), resid(cy, oh + [ll]))[0],
                             r_ribo_frac=stats.pearsonr(y, rf)[0], r2_cyto=stats.pearsonr(y, cy)[0] ** 2, r2_library=anova_r2(y, lib)))
    R = pd.DataFrame(rows); R.to_csv(f"{TAB}/gene_cause2_L5IT.csv", index=False)
    pd.set_option("display.width", 250); print(R.round(3).to_string())


if __name__ == "__main__":
    main()
