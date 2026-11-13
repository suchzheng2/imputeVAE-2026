#!/usr/bin/env python3
"""Phase 5 / H1, H4, H5: predicted delta vs external compartment contrasts.

Datasets/contrasts (external effect = log2 ratio, positive = enriched away from the nucleus):
  GSE210071 Ctl : cyto_minus_nuc, whole_minus_nuc            (paired cultures, triptolide-treated)
  GSE210071 Bic : same (sensitivity)
  GSE121199     : wc_vs_nuc_avg, wc_vs_nuc_D1, wc_vs_nuc_D2, ribo_vs_nuc_avg, ribo_vs_wc_avg (enrichment-type)
Model columns (from outputs/model_pseudobulk/predicted_delta_by_ct_all.csv): ALL, POOL_all17, POOL_neuronal,
  POOL_inhibitory, POOL_nonneuronal. Ground truth from gt_delta_by_ct_all.csv. Other methods (global) from
  compartment_classification/predicted_deltas_global.csv.
Per contrast x model column: n, Spearman/Pearson + bootstrap CI, Theil-Sen slope + CI, sign agreement,
  AUROC (external label), top-decile enrichment (Fisher), abundance-matched permutation p (1000, seed 42),
  replicate-level robustness, sensitivity filters, top concordant/discordant genes, control-gene table.
Usage: python 07_primary_concordance.py [--n_perm 1000] [--datasets GSE210071,GSE121199]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extval_utils import bh

PROJECT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE"
TASK = f"{PROJECT}/validation/trap_ribotag_external_validation"
PB = f"{TASK}/outputs/model_pseudobulk"
TAB = f"{TASK}/outputs/tables"
SEED = 42
CONTROLS = ["Malat1", "Xist", "Meg3", "Snhg11", "Neat1", "Kcnq1ot1", "Snap25", "Camk2n1", "Tmsb4x", "Pvalb", "Sst", "Actb", "Rpl3", "Fos", "Arc"]
MODEL_COLS = ["ALL", "POOL_all17", "POOL_neuronal", "POOL_inhibitory", "POOL_nonneuronal"]
NUC_COL_FOR = {"ALL": "ALL", "POOL_all17": "ALL", "POOL_neuronal": "ALL", "POOL_inhibitory": "Pvalb", "POOL_nonneuronal": "Astro"}

pred = pd.read_csv(f"{PB}/predicted_delta_by_ct_all.csv", index_col=0)
gt = pd.read_csv(f"{PB}/gt_delta_by_ct_all.csv", index_col=0)
nuc = pd.read_csv(f"{PB}/nucleus_pseudobulk_by_ct_all.csv", index_col=0)
others = pd.read_csv(f"{PROJECT}/compartment_classification/predicted_deltas_global.csv", index_col=0)
classes = pd.read_csv(f"{PROJECT}/compartment_classification/gene_classes.csv", index_col=0)["gene_class"]
ieg = [l.split("\t")[0] for l in open(f"{PROJECT}/notes/experiments/dissociation_audit/dissociation_gene_list.txt") if l.strip()]
nuc45 = [l.strip() for l in open(f"{PROJECT}/notes/experiments/nuclear_lncrna_analysis/proposed_nuclear_lncrna_list.txt") if l.strip()]


def theil(x, y):
    if len(x) > 3000:  # subsample for speed (Theil-Sen is O(n^2)), deterministic
        idx = np.random.default_rng(SEED).choice(len(x), 3000, replace=False); x, y = x[idx], y[idx]
    s, i, lo, hi = stats.theilslopes(y, x, 0.95)
    return s, lo, hi


def boot_corr(x, y, n_boot, rng, method):
    n = len(x); f = stats.spearmanr if method == "spearman" else stats.pearsonr
    b = np.array([f(x[i], y[i])[0] for i in (rng.integers(0, n, n) for _ in range(n_boot))])
    return float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))


def matched_perm(x, y, expr, n_perm, rng):
    """Permute x within deciles of expr; return null Spearman array."""
    bins = pd.qcut(pd.Series(expr).rank(method="first"), 10, labels=False).values
    null = np.empty(n_perm); xp = x.copy()
    for k in range(n_perm):
        for b in range(10):
            m = bins == b; xp[m] = rng.permutation(x[m])
        null[k] = stats.spearmanr(xp, y)[0]
    return null


def evaluate(df, xcol, ycol, ext_fdr, ext_mean, nuc_expr, n_perm, rng, label):
    """df: merged per-gene table with columns xcol (predicted), ycol (external log2fc)."""
    x = df[xcol].values; y = df[ycol].values; n = len(df)
    rho, p_rho = stats.spearmanr(x, y); r, p_r = stats.pearsonr(x, y)
    nb = 1000 if n_perm >= 1000 else 100          # full bootstrap only for final (1000-perm) runs
    lo, hi = boot_corr(x, y, nb, rng, "spearman"); rlo, rhi = boot_corr(x, y, nb, rng, "pearson")
    slope, slo, shi = theil(x, y) if n_perm >= 1000 else (np.nan, np.nan, np.nan)
    rel = (np.abs(y) > 0.5) & (ext_fdr < 0.1)
    conf = np.abs(x) > 0.25
    sign_rel = float((np.sign(x[rel]) == np.sign(y[rel])).mean()) if rel.sum() else np.nan
    sign_conf = float((np.sign(x[conf]) == np.sign(y[conf])).mean()) if conf.sum() else np.nan
    sign_both = float((np.sign(x[rel & conf]) == np.sign(y[rel & conf])).mean()) if (rel & conf).sum() else np.nan
    # median-centered sign agreement: model deltas are on un-normalized log1p counts (positive for ~97% of
    # genes because whole cells carry more UMIs), external log-ratios are library-size normalized (centered ~0)
    xc, yc = x - np.median(x), y - np.median(y)
    sign_centered_all = float((np.sign(xc) == np.sign(yc)).mean())
    sign_centered_rel = float((np.sign(xc[rel]) == np.sign(yc[rel])).mean()) if rel.sum() else np.nan
    # AUROC: external cyto-enriched vs nuclear-enriched (reliable), score = predicted delta
    lab = np.where(y > 0.5, 1, np.where(y < -0.5, 0, -1)); use = (lab >= 0) & (ext_fdr < 0.1)
    auroc = float(roc_auc_score(lab[use], x[use])) if use.sum() and len(set(lab[use])) == 2 else np.nan
    # top-decile enrichment
    q90, q10 = np.quantile(x, 0.9), np.quantile(x, 0.1)
    def fisher(sel, target):
        a = (sel & target).sum(); b = (sel & ~target).sum(); c = (~sel & target).sum(); d = (~sel & ~target).sum()
        orr, p = stats.fisher_exact([[a, b], [c, d]], alternative="greater"); return float(orr), float(p), int(a), int(sel.sum())
    ext_cyto = (y > 0.5) & (ext_fdr < 0.1); ext_nuc = (y < -0.5) & (ext_fdr < 0.1)
    or_pos, p_pos, k_pos, n_pos = fisher(x >= q90, ext_cyto)
    or_neg, p_neg, k_neg, n_neg = fisher(x <= q10, ext_nuc)
    null = matched_perm(x, y, nuc_expr, n_perm, rng)
    p_perm = float((np.sum(null >= rho) + 1) / (n_perm + 1))
    res = dict(label=label, n_genes=int(n), spearman=float(rho), spearman_ci_lo=lo, spearman_ci_hi=hi, spearman_p=float(p_rho),
               pearson=float(r), pearson_ci_lo=rlo, pearson_ci_hi=rhi, theilsen_slope=float(slope), slope_ci_lo=float(slo), slope_ci_hi=float(shi),
               n_external_reliable=int(rel.sum()), sign_agree_external_reliable=sign_rel, n_pred_confident=int(conf.sum()),
               sign_agree_pred_confident=sign_conf, sign_agree_both=sign_both, sign_agree_centered_all=sign_centered_all, sign_agree_centered_external_reliable=sign_centered_rel, median_pred=float(np.median(x)), median_external=float(np.median(y)), auroc_external_label=auroc, n_auroc=int(use.sum()),
               top10pct_pos_OR=or_pos, top10pct_pos_p=p_pos, top10pct_pos_hits=k_pos, top10pct_pos_n=n_pos,
               bottom10pct_neg_OR=or_neg, bottom10pct_neg_p=p_neg, bottom10pct_neg_hits=k_neg, bottom10pct_neg_n=n_neg,
               matched_null_mean=float(null.mean()), matched_null_sd=float(null.std()), matched_perm_p=p_perm, n_perm=int(n_perm),
               rho_pred_vs_nucexpr=float(stats.spearmanr(x, nuc_expr)[0]), rho_ext_vs_extmean=float(stats.spearmanr(y, ext_mean)[0]))
    return res


def run_contrast(dataset, tag, ext, ycol, fdr_col, mean_col, rep_cols, n_perm, out_rows, gene_rows):
    """ext: external table indexed by gene. rep_cols: per-replicate effect columns (for replicate robustness)."""
    rng = np.random.default_rng(SEED)
    common = ext.index.intersection(pred.index)
    e = ext.loc[common]
    merged = pd.DataFrame({"external_log2fc": e[ycol], "external_fdr": e[fdr_col], "external_mean_expr": e[mean_col]}, index=common)
    for c in MODEL_COLS:
        merged[f"pred_{c}"] = pred.loc[common, c]; merged[f"gt_{c}"] = gt.loc[common, c]
    for c in ["CT-mean", "Linear_reg", "scVI", "scANVI", "scGen", "scButterfly", "Raw_nuclei"]:
        merged[f"other_{c}"] = others.reindex(common)[c]
    merged["nuc_expr_ALL"] = nuc.loc[common, "ALL"]; merged["gene_class"] = classes.reindex(common).fillna("unknown")
    merged.index.name = "gene"
    merged.to_csv(f"{TAB}/merged_{dataset}_{tag}_{ycol}.tsv", sep="\t")
    print(f"\n=== {dataset} {tag} {ycol}: {len(common)} genes ===")
    for c in MODEL_COLS:
        nuc_expr = nuc.loc[common, NUC_COL_FOR[c]].values
        for kind, col in (("ImputeVAE", f"pred_{c}"), ("GroundTruth", f"gt_{c}")):
            nperm = n_perm if kind == "ImputeVAE" else max(200, n_perm // 5)
            res = evaluate(merged, col, "external_log2fc", merged.external_fdr.values, merged.external_mean_expr.values, nuc_expr, nperm, rng, f"{kind}|{c}|all")
            res.update(dataset=dataset, condition=tag, contrast=ycol, method=kind, model_col=c, filter="none", primary=(c in ("ALL", "POOL_neuronal")))
            out_rows.append(res)
            if kind == "ImputeVAE" and c in ("ALL", "POOL_neuronal"):
                print(f"  {kind:11s} {c:16s} rho={res['spearman']:+.3f} [{res['spearman_ci_lo']:+.3f},{res['spearman_ci_hi']:+.3f}]  "
                      f"AUROC={res['auroc_external_label']:.3f}  sign(rel)={res['sign_agree_external_reliable']:.3f} centered={res['sign_agree_centered_external_reliable']:.3f}  perm_p={res['matched_perm_p']:.4f}  null={res['matched_null_mean']:+.3f}")
        if c == "ALL":
            for m in ["CT-mean", "Linear_reg", "scVI", "scANVI", "scGen", "scButterfly"]:
                sub = merged.dropna(subset=[f"other_{m}"])
                res = evaluate(sub, f"other_{m}", "external_log2fc", sub.external_fdr.values, sub.external_mean_expr.values, nuc.loc[sub.index, "ALL"].values, 200, rng, f"{m}|ALL|all")
                res.update(dataset=dataset, condition=tag, contrast=ycol, method=m, model_col="ALL", filter="none", primary=False); out_rows.append(res)
    # ---- H5 sensitivity filters (ImputeVAE primary columns only)
    filters = {
        "excl_low_expr_q25": merged.external_mean_expr >= merged.external_mean_expr.quantile(0.25),
        "excl_mito": ~merged.index.str.startswith("mt-"),
        "excl_ribosomal": ~(merged.index.str.match(r"^Rp[ls]\d") & ~merged.index.str.startswith("Rps6k")),
        "excl_ieg_module": ~merged.index.isin(ieg),
        "excl_nuclear_ncrna_class": merged.gene_class != "nuclear_ncrna",
        "coding_polya_only": merged.gene_class == "coding_polya",
        "excl_all_above": (merged.external_mean_expr >= merged.external_mean_expr.quantile(0.25)) & ~merged.index.str.startswith("mt-")
                          & ~merged.index.str.match(r"^Rp[ls]\d") & ~merged.index.isin(ieg) & (merged.gene_class != "nuclear_ncrna"),
    }
    for fname, mask in filters.items():
        sub = merged[np.asarray(mask, dtype=bool)]
        for c in ("ALL", "POOL_neuronal"):
            res = evaluate(sub, f"pred_{c}", "external_log2fc", sub.external_fdr.values, sub.external_mean_expr.values, nuc.loc[sub.index, NUC_COL_FOR[c]].values, 200, rng, f"ImputeVAE|{c}|{fname}")
            res.update(dataset=dataset, condition=tag, contrast=ycol, method="ImputeVAE", model_col=c, filter=fname, primary=False); out_rows.append(res)
    # ---- replicate-level robustness (external replicates) for the primary column
    pc = "ALL" if dataset == "GSE210071" else "POOL_neuronal"
    rep = {}
    for rc in rep_cols:
        if rc in e.columns:
            rep[rc] = float(stats.spearmanr(merged[f"pred_{pc}"], e[rc])[0])
    # ---- gene-level rows: top concordant / discordant + controls
    z = merged[[f"pred_{pc}", "external_log2fc"]].apply(stats.zscore)
    merged["concordance_score"] = z.iloc[:, 0] * z.iloc[:, 1]
    top = merged.sort_values("concordance_score", ascending=False)
    for status, block in (("top_concordant", top.head(20)), ("top_discordant", top.tail(20))):
        for g, r in block.iterrows():
            gene_rows.append(dict(dataset=dataset, condition=tag, contrast=ycol, gene=g, status=status, predicted_delta=r[f"pred_{pc}"], gt_delta=r[f"gt_{pc}"],
                                  external_log2fc=r.external_log2fc, external_fdr=r.external_fdr, external_mean_expr=r.external_mean_expr, gene_class=r.gene_class))
    for g in CONTROLS + nuc45:
        if g in merged.index:
            r = merged.loc[g]
            gene_rows.append(dict(dataset=dataset, condition=tag, contrast=ycol, gene=g, status="control" if g in CONTROLS else "nuc45_set",
                                  predicted_delta=r[f"pred_{pc}"], gt_delta=r[f"gt_{pc}"], external_log2fc=r.external_log2fc, external_fdr=r.external_fdr,
                                  external_mean_expr=r.external_mean_expr, gene_class=r.gene_class,
                                  concordant=bool(np.sign(r[f"pred_{pc}"]) == np.sign(r.external_log2fc))))
        elif g in CONTROLS:
            gene_rows.append(dict(dataset=dataset, condition=tag, contrast=ycol, gene=g, status="control", concordant=np.nan,
                                  notes="absent in model gene set" if g not in pred.index else "absent/filtered in external dataset"))
    # nuc45 set-level summary
    s45 = merged.index.intersection(nuc45)
    set_row = dict(dataset=dataset, condition=tag, contrast=ycol, n_nuc45_present=len(s45),
                   frac_nuc45_external_negative=float((merged.loc[s45, "external_log2fc"] < 0).mean()) if len(s45) else np.nan,
                   mean_external_nuc45=float(merged.loc[s45, "external_log2fc"].mean()) if len(s45) else np.nan,
                   mean_external_all=float(merged.external_log2fc.mean()),
                   frac_nuc45_pred_negative=float((merged.loc[s45, f"pred_{pc}"] < 0).mean()) if len(s45) else np.nan,
                   mannwhitney_p=float(stats.mannwhitneyu(merged.loc[s45, "external_log2fc"], merged.drop(s45).external_log2fc, alternative="less")[1]) if len(s45) >= 3 else np.nan,
                   replicate_spearman=json.dumps(rep))
    return set_row


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n_perm", type=int, default=1000); ap.add_argument("--datasets", default="GSE210071,GSE121199")
    a = ap.parse_args(); ds = a.datasets.split(",")
    out_rows, gene_rows, set_rows = [], [], []
    if "GSE210071" in ds:
        for cond in ("Ctl", "Bic"):
            t = pd.read_csv(f"{TAB}/GSE210071_contrasts_{cond}.tsv", sep="\t", index_col=0)
            for yc in ("cyto_minus_nuc", "whole_minus_nuc"):
                set_rows.append(run_contrast("GSE210071", cond, t, f"{yc}_log2fc", f"{yc}_fdr", "mean_log2fpkm_N",
                                             [f"{yc}_S1", f"{yc}_S2", f"{yc}_S4"], a.n_perm if cond == "Ctl" else 200, out_rows, gene_rows))
    if "GSE121199" in ds and os.path.exists(f"{TAB}/GSE121199_contrasts_primary.tsv"):
        t = pd.read_csv(f"{TAB}/GSE121199_contrasts_primary.tsv", sep="\t", index_col=0)
        t["mean_logcpm_nuclear"] = t[["mean_logcpm_nuclear_D1", "mean_logcpm_nuclear_D2"]].mean(axis=1)
        for yc in ("wc_vs_nuc_avg", "wc_vs_nuc_D1", "wc_vs_nuc_D2", "ribo_vs_nuc_avg", "ribo_vs_wc_avg"):
            set_rows.append(run_contrast("GSE121199", "primary", t, f"{yc}_log2fc", f"{yc}_fdr", "mean_logcpm_nuclear", [],
                                         a.n_perm if yc in ("wc_vs_nuc_avg", "ribo_vs_nuc_avg") else 200, out_rows, gene_rows))
        for sens in ("sens_with_D1_1_2", "sens_no_outliers"):
            p = f"{TAB}/GSE121199_contrasts_{sens}.tsv"
            if os.path.exists(p):
                t = pd.read_csv(p, sep="\t", index_col=0); t["mean_logcpm_nuclear"] = t[["mean_logcpm_nuclear_D1", "mean_logcpm_nuclear_D2"]].mean(axis=1)
                set_rows.append(run_contrast("GSE121199", sens, t, "wc_vs_nuc_avg_log2fc", "wc_vs_nuc_avg_fdr", "mean_logcpm_nuclear", [], 200, out_rows, gene_rows))
    tagf = "_".join(ds)
    pd.DataFrame(out_rows).to_csv(f"{TAB}/concordance_summary_{tagf}.tsv", sep="\t", index=False)
    pd.DataFrame(gene_rows).to_csv(f"{TAB}/concordance_genes_{tagf}.tsv", sep="\t", index=False)
    pd.DataFrame(set_rows).to_csv(f"{TAB}/concordance_nuc45_{tagf}.tsv", sep="\t", index=False)
    print("\nwrote concordance_summary/genes/nuc45 tables for", ds)


if __name__ == "__main__":
    main()
