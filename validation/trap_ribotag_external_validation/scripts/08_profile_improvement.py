#!/usr/bin/env python3
"""Phase 5 / H2 + H3 on GSE263082 (Pvalb, Sst RiboTag IP-only) and, on the same pipeline, GSE77243
(Greenberg RiboTag PV/SST, standard housing; used in notes/experiments/trap_validation) for cross-lab comparison.

H2  delta_similarity = cor(imputed_ct, external_ct) - cor(nucleus_ct, external_ct); ceiling = cor(real_cell_ct, external_ct).
    Uncertainty: paired gene bootstrap (1000) and external replicate bootstrap (1000, resampling replicates within batch).
    Sensitivity: within nucleus-expression deciles; excluding ribosomal/mito; validation-split model pseudobulks.
    Specificity: 2x2 cross-cell-type correlation matrices; differential test cor(predDelta_Pv - predDelta_Sst, RiboTag Pv-Sst log2FC).
H3  Spearman(RiboTag abundance, predicted delta) per batch and meta; enrichment of top 5/10/20 % predicted-positive genes among
    top-20 % RiboTag-expressed genes with abundance-matched permutation (nucleus-expression deciles, 1000, seed 42); decile curve.
Gene universe per comparison: genes in the model set AND present in the external table with mean logCPM > 0 (CPM > 1).
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

PROJECT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE"
TASK = f"{PROJECT}/validation/trap_ribotag_external_validation"
PB = f"{TASK}/outputs/model_pseudobulk"; TAB = f"{TASK}/outputs/tables"; NORM = f"{TASK}/outputs/normalized"
SEED = 42
CTS = ["Pvalb", "Sst"]


def load_model(tag):
    d = {k: pd.read_csv(f"{PB}/{k}_pseudobulk_by_ct_{tag}.csv", index_col=0) for k in ("nucleus", "imputed", "cell")}
    d["pred"] = pd.read_csv(f"{PB}/predicted_delta_by_ct_{tag}.csv", index_col=0)
    d["gt"] = pd.read_csv(f"{PB}/gt_delta_by_ct_{tag}.csv", index_col=0)
    return d


def external_gse263082():
    """Returns dict ct -> {batch -> (mean Series, replicate DataFrame)} of baseline logCPM."""
    meta = pd.read_csv(f"{TASK}/outputs/metadata/GSE263082_samples.tsv", sep="\t")
    meta["key"] = meta.batch.str.lower() + ":" + meta.column.astype(str)
    out = {ct: {} for ct in CTS}
    for b in ["batch1", "batch2", "batch3", "batch4"]:
        lc = pd.read_csv(f"{NORM}/GSE263082_{b}_logcpm_tmm.tsv.gz", sep="\t", index_col=0)
        for ct in CTS:
            cols = meta[(meta.batch.str.lower() == b) & (meta.cell_type == ct) & (meta.include_primary == True)].key.tolist()
            cols = [c for c in cols if c in lc.columns]
            if len(cols) >= 2:
                out[ct][b] = lc[cols]
    return out


def external_gse77243():
    x = pd.read_excel(f"{PROJECT}/data/external/furlanis2019/GSE77243_Greenberg_SupplementaryTable1.xlsx", sheet_name="allgenes.csv")
    x = x.set_index("Genenames"); x = x[~x.index.duplicated()]
    out = {}
    for ct, pre in (("Pvalb", "Pv"), ("Sst", "Sst")):
        reps = [c for c in x.columns if re.match(rf"^{pre}_Std_B\d+$", c)]
        out[ct] = {"greenberg": np.log2(x[reps].astype(float) + 1)}
    return out


def profile_stats(ext_mean, m, ct, genes, rng, n_boot):
    n = m["nucleus"].loc[genes, ct].values; i = m["imputed"].loc[genes, ct].values; c = m["cell"].loc[genes, ct].values; e = ext_mean.loc[genes].values
    def cors(idx):
        return dict(nuc=stats.spearmanr(n[idx], e[idx])[0], imp=stats.spearmanr(i[idx], e[idx])[0], cell=stats.spearmanr(c[idx], e[idx])[0],
                    nuc_p=stats.pearsonr(n[idx], e[idx])[0], imp_p=stats.pearsonr(i[idx], e[idx])[0], cell_p=stats.pearsonr(c[idx], e[idx])[0])
    full = cors(np.arange(len(genes)))
    boots = np.array([[d["imp"] - d["nuc"], d["imp_p"] - d["nuc_p"]] for d in (cors(rng.integers(0, len(genes), len(genes))) for _ in range(n_boot))])
    return dict(n_genes=len(genes), spearman_nucleus=full["nuc"], spearman_imputed=full["imp"], spearman_realcell=full["cell"],
                delta_similarity_spearman=full["imp"] - full["nuc"], gene_boot_ci_lo=float(np.percentile(boots[:, 0], 2.5)), gene_boot_ci_hi=float(np.percentile(boots[:, 0], 97.5)),
                pearson_nucleus=full["nuc_p"], pearson_imputed=full["imp_p"], pearson_realcell=full["cell_p"], delta_similarity_pearson=full["imp_p"] - full["nuc_p"],
                gene_boot_pearson_ci_lo=float(np.percentile(boots[:, 1], 2.5)), gene_boot_pearson_ci_hi=float(np.percentile(boots[:, 1], 97.5)),
                gap_closed_frac=(full["imp"] - full["nuc"]) / (full["cell"] - full["nuc"]) if full["cell"] != full["nuc"] else np.nan)


def replicate_boot(rep_df, m, ct, genes, rng, n_boot):
    n = m["nucleus"].loc[genes, ct].values; i = m["imputed"].loc[genes, ct].values; R = rep_df.loc[genes].values; k = R.shape[1]
    ds = []
    for _ in range(n_boot):
        e = R[:, rng.integers(0, k, k)].mean(axis=1)
        ds.append(stats.spearmanr(i, e)[0] - stats.spearmanr(n, e)[0])
    ds = np.array(ds); return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5)), float(np.mean(ds))


def h3(ext_mean, m, ct, genes, rng, n_perm):
    e = ext_mean.loc[genes].values; p = m["pred"].loc[genes, ct].values; g = m["gt"].loc[genes, ct].values; nx = m["nucleus"].loc[genes, ct].values
    rho_p = stats.spearmanr(e, p)[0]; rho_g = stats.spearmanr(e, g)[0]
    bins = pd.qcut(pd.Series(nx).rank(method="first"), 10, labels=False).values
    top_ext = e >= np.quantile(e, 0.8)
    res = dict(n_genes=len(genes), spearman_ribotag_vs_pred_delta=rho_p, spearman_ribotag_vs_gt_delta=rho_g,
               spearman_ribotag_vs_nucleus_expr=stats.spearmanr(e, nx)[0])
    # partial: does predicted delta explain RiboTag beyond nucleus expression? (Spearman partial via residual ranks)
    rk = lambda v: stats.rankdata(v)
    re_, rp_, rn_ = rk(e), rk(p), rk(nx)
    res["partial_spearman_pred_given_nucexpr"] = float(stats.pearsonr(re_ - np.polyval(np.polyfit(rn_, re_, 1), rn_), rp_ - np.polyval(np.polyfit(rn_, rp_, 1), rn_))[0])
    for q in (0.95, 0.90, 0.80):
        sel = p >= np.quantile(p, q)
        obs = (sel & top_ext).sum() / sel.sum()
        null = np.empty(n_perm); pp = p.copy()
        for k in range(n_perm):
            for b in range(10):
                mm = bins == b; pp[mm] = rng.permutation(p[mm])
            s2 = pp >= np.quantile(pp, q); null[k] = (s2 & top_ext).sum() / s2.sum()
        res[f"top{int(round((1-q)*100))}pct_frac_in_top20_ribotag"] = float(obs); res[f"top{int(round((1-q)*100))}pct_matched_null_mean"] = float(null.mean())
        res[f"top{int(round((1-q)*100))}pct_matched_p"] = float((np.sum(null >= obs) + 1) / (n_perm + 1)); res[f"top{int(round((1-q)*100))}pct_baseline_frac"] = float(top_ext.mean())
    dec = pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False).values
    res["decile_curve"] = json.dumps([float(np.mean(e[dec == d])) for d in range(10)])
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n_perm", type=int, default=1000); a = ap.parse_args()
    rng = np.random.default_rng(SEED); nb = a.n_perm
    ext = {"GSE263082": external_gse263082(), "GSE77243": external_gse77243()}
    rows, h3rows, spec_rows = [], [], []
    for tag in ("all", "valid"):
        m = load_model(tag)
        for ds, per_ct in ext.items():
            for ct in CTS:
                batches = per_ct[ct]
                # per-batch and meta profiles
                means = {b: df.mean(axis=1) for b, df in batches.items()}
                common_all = None
                for b, mu in means.items():
                    genes = mu.index[mu > 0].intersection(m["nucleus"].index)
                    common_all = genes if common_all is None else common_all.intersection(genes)
                    st = profile_stats(mu, m, ct, genes, rng, nb); lo, hi, mean_rep = replicate_boot(batches[b], m, ct, genes, rng, nb)
                    st.update(dataset=ds, cell_type=ct, batch=b, n_replicates=batches[b].shape[1], model_split=tag, filter="none",
                              replicate_boot_ci_lo=lo, replicate_boot_ci_hi=hi, replicate_boot_mean=mean_rep); rows.append(st)
                    if tag == "all":
                        h = h3(mu, m, ct, genes, rng, nb); h.update(dataset=ds, cell_type=ct, batch=b); h3rows.append(h)
                        # sensitivity filters on H2
                        for fname, mask in (("excl_ribo_mito", ~(genes.str.match(r"^Rp[ls]\d") | genes.str.startswith("mt-"))),):
                            st2 = profile_stats(mu, m, ct, genes[mask], rng, 200); st2.update(dataset=ds, cell_type=ct, batch=b, n_replicates=batches[b].shape[1], model_split=tag, filter=fname); rows.append(st2)
                        # within nucleus-expression deciles
                        nx = m["nucleus"].loc[genes, ct]; dec = pd.qcut(nx.rank(method="first"), 5, labels=False)
                        for d in range(5):
                            gg = genes[dec.values == d]; st3 = profile_stats(mu, m, ct, gg, rng, 100)
                            st3.update(dataset=ds, cell_type=ct, batch=b, n_replicates=batches[b].shape[1], model_split=tag, filter=f"nucexpr_quintile_{d+1}"); rows.append(st3)
                if len(means) > 1:
                    meta_mu = pd.concat([mu.loc[common_all] for mu in means.values()], axis=1).mean(axis=1)
                    st = profile_stats(meta_mu, m, ct, common_all, rng, nb)
                    st.update(dataset=ds, cell_type=ct, batch="META", n_replicates=sum(df.shape[1] for df in batches.values()), model_split=tag, filter="none"); rows.append(st)
                    if tag == "all":
                        h = h3(meta_mu, m, ct, common_all, rng, nb); h.update(dataset=ds, cell_type=ct, batch="META"); h3rows.append(h)
            # ---- specificity: cross-cell-type matrix + differential (batches with both CTs or META)
            if tag == "all":
                prof = {}
                for ct in CTS:
                    means = {b: df.mean(axis=1) for b, df in per_ct[ct].items()}
                    prof[ct] = pd.concat([mu for mu in means.values()], axis=1).mean(axis=1) if len(means) > 1 else list(means.values())[0]
                genes = prof["Pvalb"].index.intersection(prof["Sst"].index).intersection(m["nucleus"].index)
                genes = genes[(prof["Pvalb"].loc[genes] > 0) & (prof["Sst"].loc[genes] > 0)]
                for kind in ("nucleus", "imputed", "cell"):
                    for mct in CTS:
                        for ect in CTS:
                            spec_rows.append(dict(dataset=ds, kind=kind, model_ct=mct, external_ct=ect, n_genes=len(genes),
                                                  spearman=stats.spearmanr(m[kind].loc[genes, mct], prof[ect].loc[genes])[0]))
                ext_diff = prof["Pvalb"].loc[genes] - prof["Sst"].loc[genes]
                for kind, M in (("pred_delta", m["pred"]), ("gt_delta", m["gt"]), ("nucleus", m["nucleus"]), ("imputed", m["imputed"]), ("cell", m["cell"])):
                    spec_rows.append(dict(dataset=ds, kind=f"diff_{kind}", model_ct="Pvalb-Sst", external_ct="Pvalb-Sst", n_genes=len(genes),
                                          spearman=stats.spearmanr(M.loc[genes, "Pvalb"] - M.loc[genes, "Sst"], ext_diff)[0]))
                if ds == "GSE263082" and os.path.exists(f"{TAB}/GSE263082_combined_baseline_Pvalb_vs_Sst.tsv"):
                    cb = pd.read_csv(f"{TAB}/GSE263082_combined_baseline_Pvalb_vs_Sst.tsv", sep="\t", index_col=0)
                    g2 = cb.index.intersection(m["pred"].index)
                    for kind, M in (("pred_delta", m["pred"]), ("gt_delta", m["gt"]), ("imputed", m["imputed"]), ("nucleus", m["nucleus"])):
                        spec_rows.append(dict(dataset=ds, kind=f"diff_{kind}_vs_limma_batchadj", model_ct="Pvalb-Sst", external_ct="Pvalb-Sst(limma)", n_genes=len(g2),
                                              spearman=stats.spearmanr(M.loc[g2, "Pvalb"] - M.loc[g2, "Sst"], cb.loc[g2, "Pvalb_vs_Sst_log2fc"])[0]))
    pd.DataFrame(rows).to_csv(f"{TAB}/profile_improvement_H2.tsv", sep="\t", index=False)
    pd.DataFrame(h3rows).to_csv(f"{TAB}/translational_support_H3.tsv", sep="\t", index=False)
    pd.DataFrame(spec_rows).to_csv(f"{TAB}/celltype_specificity.tsv", sep="\t", index=False)
    r = pd.DataFrame(rows); r = r[(r["filter"] == "none") & (r.model_split == "all")]
    print(r[["dataset", "cell_type", "batch", "n_replicates", "n_genes", "spearman_nucleus", "spearman_imputed", "spearman_realcell", "delta_similarity_spearman", "gene_boot_ci_lo", "gene_boot_ci_hi", "replicate_boot_ci_lo", "replicate_boot_ci_hi"]].round(4).to_string(index=False))
    h = pd.DataFrame(h3rows); print(h[["dataset", "cell_type", "batch", "n_genes", "spearman_ribotag_vs_pred_delta", "spearman_ribotag_vs_gt_delta", "spearman_ribotag_vs_nucleus_expr", "partial_spearman_pred_given_nucexpr", "top10pct_frac_in_top20_ribotag", "top10pct_matched_null_mean", "top10pct_matched_p"]].round(4).to_string(index=False))
    s = pd.DataFrame(spec_rows); print(s.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
