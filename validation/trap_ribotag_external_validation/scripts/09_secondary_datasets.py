#!/usr/bin/env python3
"""Phase 4D/4E + Phase 5 for the secondary (enrichment-type) datasets.

GSE174303 (CA1 TRAP vs input, WT, from 09a limma tables): predicted delta vs TRAP-input log2FC within CB and ML,
  and vs the ML-CB difference in TRAP enrichment (exploratory dendritic axis). Expected weak/negative (property mismatch,
  see notes/experiments/trap_validation: IP/Input measures translational engagement, not compartment).
GSE13379 (Doyle 2008 bacTRAP, Affymetrix GCRMA linear -> log2): per cortical population, TRAP abundance (mean of 3 reps,
  gene = median over probes) vs predicted delta of the matched MOp cell type (abundance test, replicates the GSE77243 design),
  and IP/UB log2 ratio (n=1 unbound) vs predicted delta (descriptive only).
Population -> model cell type map: aldh1l1 astrocytes -> Astro; etv1 L5a -> L5 IT; glt25d2 L5b -> L5 ET; ntsr1 L6 -> L6 CT;
  cmtm5 oligodendrocytes -> Oligo; olig2 mixed oligodendroglia -> Oligo (+OPC); cck -> POOL_inhibitory (Cck+ cells span Vip/Sncg);
  pnoc -> POOL_inhibitory; cort -> Sst (Cort+ cells are largely Sst interneurons).
All comparisons also report the ground-truth delta. Matched-null permutation within nucleus-expression deciles.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
c07 = import_module("07_primary_concordance")  # reuse evaluate(), tables

PROJECT = c07.PROJECT; TASK = c07.TASK; TAB = c07.TAB; PB = c07.PB
pred, gt, nuc = c07.pred, c07.gt, c07.nuc
SEED = 42
POP_MAP = {"aldh1l1": ("Astro", "Astro"), "etv1": ("L5 IT", "L5 IT"), "glt25d2": ("L5 ET", "L5 ET"), "ntsr1": ("L6 CT", "L6 CT"),
           "cmtm5": ("Oligo", "Oligo"), "olig2": ("Oligo", "Oligo"), "cck": ("POOL_inhibitory", "Vip"), "pnoc": ("POOL_inhibitory", "Vip"), "cort": ("Sst", "Sst")}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n_perm", type=int, default=1000); a = ap.parse_args()
    rng = np.random.default_rng(SEED); rows = []; gene_rows = []
    # ---------------- GSE174303
    t = pd.read_csv(f"{TAB}/GSE174303_contrasts_WT.tsv", sep="\t", index_col=0)
    common = t.index.intersection(pred.index)
    for yc, ycol_mean in (("trap_vs_input_CB", "mean_logcpm_CB_input"), ("trap_vs_input_ML", "mean_logcpm_ML_input"), ("trap_enrich_ML_minus_CB", "mean_logcpm_CB_input"), ("input_ML_vs_CB", "mean_logcpm_CB_input")):
        merged = pd.DataFrame({"external_log2fc": t.loc[common, f"{yc}_log2fc"], "external_fdr": t.loc[common, f"{yc}_fdr"], "external_mean_expr": t.loc[common, ycol_mean]})
        for mc in ("POOL_excitatory", "L5 IT", "ALL"):
            merged[f"pred_{mc}"] = pred.loc[common, mc]; merged[f"gt_{mc}"] = gt.loc[common, mc]
            for kind in ("pred", "gt"):
                res = c07.evaluate(merged, f"{kind}_{mc}", "external_log2fc", merged.external_fdr.values, merged.external_mean_expr.values, nuc.loc[common, "L5 IT"].values,
                                   a.n_perm if (kind == "pred" and mc == "POOL_excitatory") else 200, rng, f"{kind}|{mc}")
                res.update(dataset="GSE174303", condition="WT", contrast=yc, method="ImputeVAE" if kind == "pred" else "GroundTruth", model_col=mc, contrast_type="enrichment(TRAP/input)" if "trap" in yc else "input ML-CB", filter="none")
                rows.append(res)
        merged.index.name = "gene"; merged.to_csv(f"{TAB}/merged_GSE174303_WT_{yc}.tsv", sep="\t")
        print(f"GSE174303 {yc:26s} n={len(common)}  ImputeVAE(POOL_exc) rho={rows[-6]['spearman']:+.3f}  GT rho={rows[-5]['spearman']:+.3f}")
    # ---------------- GSE13379
    meta = pd.read_csv(f"{TASK}/outputs/metadata/GSE13379_samples.tsv", sep="\t")
    probe = pd.read_csv(f"{TASK}/outputs/harmonized/GSE13379_probe_matrix.tsv.gz", sep="\t", index_col=0)
    p2s = pd.read_csv(f"{TASK}/outputs/gene_mapping/GSE13379_probe_to_symbol.tsv", sep="\t")
    p2s = p2s[p2s.status == "mapped"].set_index("probe")["symbol"]
    L = np.log2(probe.clip(lower=1))
    L = L.loc[L.index.intersection(p2s.index)]; L.index = p2s.loc[L.index].values
    G = L.groupby(level=0).median()   # gene-level: median over probes of log2 GCRMA
    G.to_csv(f"{TASK}/outputs/normalized/GSE13379_gene_log2gcrma.tsv.gz", sep="\t")
    meta["driver"] = meta.cell_type.str.extract(r"\((\w+) /")[0]
    prim = meta[meta.include_primary == True]
    for drv, (mc, nuc_ct) in POP_MAP.items():
        sub = prim[prim.driver == drv]
        trap = sub[sub.compartment == "ribosome_bound"].column.tolist(); ub = sub[sub.compartment != "ribosome_bound"].column.tolist()
        if len(trap) < 2:
            continue
        common = G.index.intersection(pred.index)
        e_trap = G.loc[common, trap].mean(axis=1); expressed = common[e_trap > np.log2(64)]   # GCRMA background ~ 2^6
        merged = pd.DataFrame({"external_log2fc": e_trap.loc[expressed], "external_fdr": np.zeros(len(expressed)), "external_mean_expr": e_trap.loc[expressed]})
        merged[f"pred_{mc}"] = pred.loc[expressed, mc]; merged[f"gt_{mc}"] = gt.loc[expressed, mc]
        nx = nuc.loc[expressed, nuc_ct].values
        for kind in ("pred", "gt"):
            x = merged[f"{kind}_{mc}"].values; y = merged.external_log2fc.values
            rho = stats.spearmanr(x, y)[0]; null = c07.matched_perm(x, y, nx, a.n_perm if kind == "pred" else 200, rng)
            rk = stats.rankdata
            partial = stats.pearsonr(rk(y) - np.polyval(np.polyfit(rk(nx), rk(y), 1), rk(nx)), rk(x) - np.polyval(np.polyfit(rk(nx), rk(x), 1), rk(nx)))[0]
            rep = [float(stats.spearmanr(x, G.loc[expressed, c])[0]) for c in trap]
            row = dict(dataset="GSE13379", condition="cortex", contrast=f"TRAP_abundance_{drv}", contrast_type="abundance(IP only)", method="ImputeVAE" if kind == "pred" else "GroundTruth",
                       model_col=mc, n_genes=len(expressed), spearman=float(rho), spearman_ci_lo=np.nan, spearman_ci_hi=np.nan, matched_null_mean=float(null.mean()), matched_null_sd=float(null.std()),
                       matched_perm_p=float((np.sum(null >= rho) + 1) / (len(null) + 1)), rho_pred_vs_nucexpr=float(stats.spearmanr(x, nx)[0]), rho_ext_vs_nucexpr=float(stats.spearmanr(y, nx)[0]),
                       partial_spearman_given_nucexpr=float(partial), replicate_spearman=json.dumps(rep), n_trap_reps=len(trap), n_unbound=len(ub), filter="none")
            rows.append(row)
        print(f"GSE13379 {drv:8s} -> {mc:16s} n={len(expressed)}  TRAP-abundance rho: ImputeVAE {rows[-2]['spearman']:+.3f} (null {rows[-2]['matched_null_mean']:+.3f}, partial {rows[-2]['partial_spearman_given_nucexpr']:+.3f})  GT {rows[-1]['spearman']:+.3f}")
        if ub:
            ipub = e_trap.loc[expressed] - G.loc[expressed, ub].mean(axis=1)
            for kind in ("pred", "gt"):
                x = merged[f"{kind}_{mc}"].values; rho = stats.spearmanr(x, ipub.values)[0]
                rows.append(dict(dataset="GSE13379", condition="cortex", contrast=f"IP_over_UB_{drv}", contrast_type="enrichment(IP/UB, n_UB=%d)" % len(ub), method="ImputeVAE" if kind == "pred" else "GroundTruth",
                                 model_col=mc, n_genes=len(expressed), spearman=float(rho), filter="none", n_trap_reps=len(trap), n_unbound=len(ub)))
            print(f"          IP/UB rho: ImputeVAE {rows[-2]['spearman']:+.3f}  GT {rows[-1]['spearman']:+.3f}")
        merged.index.name = "gene"; merged["ip_over_ub"] = ipub if ub else np.nan; merged.to_csv(f"{TAB}/merged_GSE13379_{drv}.tsv", sep="\t")
    pd.DataFrame(rows).to_csv(f"{TAB}/concordance_summary_secondary.tsv", sep="\t", index=False)
    print("wrote concordance_summary_secondary.tsv")


if __name__ == "__main__":
    main()
