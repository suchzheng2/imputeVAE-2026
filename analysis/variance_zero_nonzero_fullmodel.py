#!/usr/bin/env python3
"""Zero / nonzero variance diagnostic on the full-data cVAE (run_2026-07-18-19-40-51_570984).

Question (Prashant, 2026-09): before adding a variance loss term, is the within-cell-type variance that
the imputed data is missing biological or technical?

Per (cell type, gene), on log1p(counts) for 40,166 nuclei, their paired imputed values, and 71,117 real cells:
  A. all-entry variances and the imputed/real ratio (the existing 0.158 metric)
  B. partition by whether the ORIGINAL NUCLEUS entry is zero or nonzero:
       nonzero entries: var(nucleus), var(imputed), ratio, paired Pearson/Spearman/slope nucleus->imputed
       zero entries   : mean and var of imputed values (nucleus var is 0 by definition)
     law of total variance: var = p*var_nz + (1-p)*var_z + p(1-p)*(mean_nz-mean_z)^2   (within-nonzero,
     within-zero, zero/nonzero mixing) for nuclei, imputed, and real cells (cells partitioned by their own zeros)
  C. technical null by simulation on the counts scale, evaluated on log1p:
       null A: x ~ Poisson(mu_g)                 sampling noise only
       null B: x ~ Poisson(s_i * mu_g)           sampling noise + library size (s_i = lib_i / mean lib in CT)
     biological excess = observed var - null var (floored at 0); also expected detection rate under null B
  D. gene-gene correlation structure within CT (library-size axis regressed out): does imputed covariance
     resemble real-cell covariance better than raw nuclei do?
Outputs -> notes/experiments/variance_zero_nonzero_fullmodel/tables/
"""
import os
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse, stats

PROJECT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE"
DATA_ROOT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi"
OUT = f"{PROJECT}/notes/experiments/variance_zero_nonzero_fullmodel/tables"
os.makedirs(OUT, exist_ok=True)
MIN_N = 20
RNG = np.random.default_rng(42)


def load_counts(data_dir, batch, gene_list, common_genes, cts):
    a = ad.read_h5ad(f"{DATA_ROOT}/{data_dir}/data.h5ad")
    a.obs["mb"] = a.obs["batch_id"].apply(lambda x: x.split(".")[0])
    rm = a.obs["mb"].str.endswith(batch) & a.obs["cell_type"].isin(cts)
    cm = a.var["gene_names"].isin(common_genes) & a.var["gene_names"].isin(gene_list)
    a = a[rm.values, cm.values].copy(); a = a[:, a.var["gene_names"].argsort()].copy()
    X = (a.X.toarray() if sparse.issparse(a.X) else np.asarray(a.X)).astype(np.float32)
    return X, a.obs["cell_type"].astype(str).values, a.var["gene_names"].values.tolist()


def sim_null_var(mu, s, n_rep):
    """log1p-scale variance and detection under Poisson nulls. mu: (G,), s: (n,)"""
    n = len(s); vA = np.zeros_like(mu); vB = np.zeros_like(mu); vBnz = np.zeros_like(mu); cntnz = np.zeros_like(mu)
    for _ in range(n_rep):
        xa = np.log1p(RNG.poisson(np.broadcast_to(mu, (n, len(mu)))).astype(np.float32)); vA += xa.var(axis=0); del xa
        xb = RNG.poisson(np.outer(s, mu)).astype(np.float32); lb = np.log1p(xb); vB += lb.var(axis=0)
        nzm = xb > 0; k = nzm.sum(axis=0)
        m1 = np.where(k > 0, (lb * nzm).sum(axis=0) / np.maximum(k, 1), 0); m2 = np.where(k > 0, ((lb ** 2) * nzm).sum(axis=0) / np.maximum(k, 1), 0)
        ok = k >= MIN_N; vBnz += np.where(ok, m2 - m1 ** 2, 0); cntnz += ok; del xb, lb, nzm
    detB = 1 - np.exp(-np.outer(s, mu)).mean(axis=0)
    return vA / n_rep, vB / n_rep, np.where(cntnz > 0, vBnz / np.maximum(cntnz, 1), np.nan), detB


def part_stats(L, nz):
    """L: (n,G) values; nz: (n,G) bool mask. Returns p, mean/var within nz and within zero."""
    n = L.shape[0]; k = nz.sum(axis=0).astype(np.float64); kz = n - k
    s1 = (L * nz).sum(axis=0, dtype=np.float64); s2 = ((L.astype(np.float64) ** 2) * nz).sum(axis=0)
    t1 = L.sum(axis=0, dtype=np.float64); t2 = (L.astype(np.float64) ** 2).sum(axis=0)
    mnz = np.where(k > 0, s1 / np.maximum(k, 1), np.nan); vnz = np.where(k >= MIN_N, s2 / np.maximum(k, 1) - mnz ** 2, np.nan)
    mz = np.where(kz > 0, (t1 - s1) / np.maximum(kz, 1), np.nan); vz = np.where(kz >= MIN_N, (t2 - s2) / np.maximum(kz, 1) - mz ** 2, np.nan)
    return k / n, mnz, vnz, mz, vz, k, kz


def paired_corr(A, B, mask):
    """column-wise Pearson between A and B over mask entries; also slope of B on A."""
    k = mask.sum(axis=0).astype(np.float64); kk = np.maximum(k, 1)
    a = np.where(mask, A, 0).astype(np.float64); b = np.where(mask, B, 0).astype(np.float64)
    ma, mb = a.sum(0) / kk, b.sum(0) / kk
    cov = (a * b).sum(0) / kk - ma * mb; va = (a * a).sum(0) / kk - ma ** 2; vb = (b * b).sum(0) / kk - mb ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        r = cov / np.sqrt(va * vb); slope = cov / va
    bad = (k < MIN_N) | (va <= 1e-12) | (vb <= 1e-12); r[bad] = np.nan; slope[bad] = np.nan
    return r, slope


def main():
    global OUT
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--imputed", default=f"{PROJECT}/run/cvae_full_eval/imputed_cell_from_nucleus.npy",
                    help="exported imputed matrix (analysis/export_cvae_imputed.py); default = paper model")
    ap.add_argument("--obs", default=None, help="nucleus_obs.csv next to --imputed (default: same folder)")
    ap.add_argument("--out", default=OUT, help="output folder for the tables (default: paper-model folder)")
    cli = ap.parse_args()
    OUT = cli.out; os.makedirs(OUT, exist_ok=True)
    obs_path = cli.obs or os.path.join(os.path.dirname(cli.imputed), "nucleus_obs.csv")
    print(f"imputed: {cli.imputed}\nobs: {obs_path}\nout: {OUT}", flush=True)

    gene_list = pd.read_csv(f"{PROJECT}/data/training_gene_list_v3.csv", header=None)[0].tolist()
    common_genes = pd.read_csv(f"{PROJECT}/data/common_genes.csv")["common_genes"].tolist()
    cts = pd.read_csv(f"{PROJECT}/data/common_cell_types.csv")["common_cell_types"].tolist()
    Nc, nct, genes = load_counts("10X_nuclei_v2_v3_AIBS", "n3A", gene_list, common_genes, cts)
    Cc, cct, genes2 = load_counts("10X_cells_v2_v3_AIBS", "c3A", gene_list, common_genes, cts); assert genes == genes2
    I = np.load(cli.imputed)
    ict = pd.read_csv(obs_path)["cell_type"].astype(str).values
    assert I.shape == Nc.shape and (ict == nct).mean() > 0.999
    genes = np.array(genes); rows = []; examples = []; corr_rows = []
    for ct in cts:
        mn, mc = nct == ct, cct == ct
        if mn.sum() < MIN_N or mc.sum() < MIN_N:
            continue
        Xn, Xc, Xi = Nc[mn], Cc[mc], I[mn]; Ln, Lc = np.log1p(Xn), np.log1p(Xc)
        print(f"{ct}: {mn.sum()} nuclei, {mc.sum()} cells", flush=True)
        nzn, nzc = Xn > 0, Xc > 0
        p_n, mnz_n, vnz_n, _, _, k_n, kz_n = part_stats(Ln, nzn)
        p_i, mnz_i, vnz_i, mz_i, vz_i, _, _ = part_stats(Xi, nzn)          # imputed, partitioned by NUCLEUS zeros
        p_c, mnz_c, vnz_c, _, _, k_c, _ = part_stats(Lc, nzc)
        r_nz, slope_nz = paired_corr(Ln, Xi, nzn); r_all, slope_all = paired_corr(Ln, Xi, np.ones_like(nzn))
        # spearman on nonzero entries: rank within column over nz entries (approx via ranks of masked arrays, only for well-detected genes)
        n_rep = 10 if mc.sum() < 500 else (4 if mc.sum() < 5000 else 2)
        s_c = Xc.sum(axis=1); s_c = s_c / s_c.mean(); s_n = Xn.sum(axis=1); s_n = s_n / s_n.mean()
        vA_c, vB_c, vBnz_c, detB_c = sim_null_var(Xc.mean(axis=0), s_c, n_rep)
        n_rep_n = 10 if mn.sum() < 500 else (4 if mn.sum() < 5000 else 2)
        vA_n, vB_n, vBnz_n, detB_n = sim_null_var(Xn.mean(axis=0), s_n, n_rep_n)
        df = pd.DataFrame({"cell_type": ct, "gene": genes, "n_nuclei": mn.sum(), "n_cells": mc.sum(),
                           "det_nuc": p_n, "det_cell": p_c, "det_cell_nullB": detB_c, "det_nuc_nullB": detB_n,
                           "mean_nuc": Ln.mean(0), "mean_imp": Xi.mean(0), "mean_cell": Lc.mean(0),
                           "var_nuc_all": Ln.var(0), "var_imp_all": Xi.var(0), "var_cell_all": Lc.var(0),
                           "var_nuc_nz": vnz_n, "mean_nuc_nz": mnz_n, "var_imp_at_nuc_nz": vnz_i, "mean_imp_at_nuc_nz": mnz_i,
                           "var_imp_at_nuc_zero": vz_i, "mean_imp_at_nuc_zero": mz_i, "n_nuc_nz": k_n, "n_nuc_zero": kz_n,
                           "var_cell_nz": vnz_c, "mean_cell_nz": mnz_c, "n_cell_nz": k_c,
                           "r_paired_nz": r_nz, "slope_paired_nz": slope_nz, "r_paired_all": r_all, "slope_paired_all": slope_all,
                           "cv_libsize_cell": s_c.std(), "cv_libsize_nuc": s_n.std(),
                           "var_cell_nullA": vA_c, "var_cell_nullB": vB_c, "var_cell_nz_nullB": vBnz_c,
                           "var_nuc_nullA": vA_n, "var_nuc_nullB": vB_n, "var_nuc_nz_nullB": vBnz_n})
        rows.append(df)
        # ---- D. gene-gene correlation structure (library-size axis regressed out), big CTs only
        if mn.sum() >= 900 and mc.sum() >= 300:
            excess = np.maximum(df.var_cell_all.values - df.var_cell_nullB.values, 0)
            cand = np.where((p_c > 0.3) & (p_n > 0.1))[0]; top = cand[np.argsort(-excess[cand])[:300]]
            def resid(L):
                z = L[:, top] - L[:, top].mean(0); axis = L.mean(1, keepdims=True); axis = axis - axis.mean()
                beta = (z * axis).sum(0) / (axis ** 2).sum(); return z - axis * beta
            sub_c = RNG.choice(mc.sum(), min(5000, mc.sum()), replace=False); sub_n = RNG.choice(mn.sum(), min(5000, mn.sum()), replace=False)
            Rc = np.corrcoef(resid(Lc[sub_c]).T); Rn = np.corrcoef(resid(Ln[sub_n]).T); Ri = np.corrcoef(resid(Xi[sub_n]).T)
            iu = np.triu_indices(len(top), 1)
            corr_rows.append(dict(cell_type=ct, n_genes=len(top), spearman_nuc_vs_cell=stats.spearmanr(Rn[iu], Rc[iu])[0], spearman_imp_vs_cell=stats.spearmanr(Ri[iu], Rc[iu])[0],
                                  mean_abs_r_cell=np.abs(Rc[iu]).mean(), mean_abs_r_nuc=np.abs(Rn[iu]).mean(), mean_abs_r_imp=np.abs(Ri[iu]).mean()))
            if ct in ("L5 IT", "Sst"):
                pick = RNG.choice(len(iu[0]), 20000, replace=False)
                pd.DataFrame({"r_cell": Rc[iu][pick], "r_nuc": Rn[iu][pick], "r_imp": Ri[iu][pick]}).to_csv(f"{OUT}/genegene_corr_pairs_{ct.replace(' ', '_')}.csv.gz", index=False)
        # ---- example genes for scatter panels (L5 IT and Sst): pick by detection bin, highest biological excess
        if ct in ("L5 IT", "Sst"):
            excess = np.maximum(df.var_cell_all.values - df.var_cell_nullB.values, 0)
            for lo, hi in ((0.7, 1.01), (0.3, 0.7), (0.1, 0.3), (0.02, 0.1)):
                cand = np.where((p_n >= lo) & (p_n < hi))[0]; g = cand[np.argsort(-excess[cand])[0]]
                idx = RNG.choice(mn.sum(), min(4000, mn.sum()), replace=False); idc = RNG.choice(mc.sum(), min(4000, mc.sum()), replace=False)
                examples.append(pd.DataFrame({"cell_type": ct, "gene": genes[g], "det_bin": f"{lo}-{hi}", "kind": "nucleus_vs_imputed", "nucleus_log1p": Ln[idx, g], "imputed": Xi[idx, g], "cell_log1p": np.nan}))
                examples.append(pd.DataFrame({"cell_type": ct, "gene": genes[g], "det_bin": f"{lo}-{hi}", "kind": "real_cell", "nucleus_log1p": np.nan, "imputed": np.nan, "cell_log1p": Lc[idc, g]}))
    T = pd.concat(rows, ignore_index=True)
    T["var_ratio_all"] = T.var_imp_all / T.var_cell_all
    for nm in ("A", "B"):
        T[f"bio_excess_cell_{nm}"] = (T.var_cell_all - T[f"var_cell_null{nm}"]).clip(lower=0); T[f"frac_bio_cell_{nm}"] = T[f"bio_excess_cell_{nm}"] / T.var_cell_all
        T[f"bio_excess_nuc_{nm}"] = (T.var_nuc_all - T[f"var_nuc_null{nm}"]).clip(lower=0); T[f"frac_bio_nuc_{nm}"] = T[f"bio_excess_nuc_{nm}"] / T.var_nuc_all
    T["var_ratio_nz_imp_over_nuc"] = T.var_imp_at_nuc_nz / T.var_nuc_nz
    T["var_ratio_nz_imp_over_cellnz"] = T.var_imp_at_nuc_nz / T.var_cell_nz
    T.to_csv(f"{OUT}/per_ct_gene_variance.csv.gz", index=False)
    pd.concat(examples).to_csv(f"{OUT}/example_gene_points.csv.gz", index=False)
    pd.DataFrame(corr_rows).to_csv(f"{OUT}/genegene_structure.csv", index=False)
    print("wrote tables; rows:", len(T)); print(pd.DataFrame(corr_rows).round(3).to_string())
    ok = T[(T.var_cell_all > 1e-4)]
    print("median var_ratio_all:", round(float(ok.var_ratio_all.median()), 4), " median frac_bio_cell_B:", round(float(ok.frac_bio_cell_B.median()), 4), " A:", round(float(ok.frac_bio_cell_A.median()), 4))


if __name__ == "__main__":
    main()
