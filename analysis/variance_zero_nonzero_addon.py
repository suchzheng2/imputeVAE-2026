#!/usr/bin/env python3
"""Add-on to the zero/nonzero variance diagnostic:
 (a) share of within-CT variance explained by per-cell library size (log total counts), per (CT, gene), for nuclei,
     imputed values (regressed on the INPUT nucleus library size) and real cells;
 (b) effective dimensionality of within-CT variation (participation ratio, top-PC shares) on the 300 most
     biologically variable genes per CT;
 (c) alignment: share of imputed / nucleus variance lying in the top-10 real-cell PC subspace (same genes).
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from variance_zero_nonzero_fullmodel import load_counts, PROJECT, MIN_N
from variance_zero_nonzero_fullmodel import OUT as _DEFAULT_OUT
# 2026-10-23: parametrised like the main script. Env IMPUTED / OUT_DIR override the paper-run defaults
# (OUT_DIR must already hold per_ct_gene_variance.csv.gz written by variance_zero_nonzero_fullmodel.py --out).
OUT = os.environ.get("OUT_DIR", _DEFAULT_OUT)
IMPUTED = os.environ.get("IMPUTED", f"{PROJECT}/run/cvae_full_eval/imputed_cell_from_nucleus.npy")
RNG = np.random.default_rng(7)

gene_list = pd.read_csv(f"{PROJECT}/data/training_gene_list_v3.csv", header=None)[0].tolist()
common_genes = pd.read_csv(f"{PROJECT}/data/common_genes.csv")["common_genes"].tolist()
cts = pd.read_csv(f"{PROJECT}/data/common_cell_types.csv")["common_cell_types"].tolist()
Nc, nct, genes = load_counts("10X_nuclei_v2_v3_AIBS", "n3A", gene_list, common_genes, cts)
Cc, cct, _ = load_counts("10X_cells_v2_v3_AIBS", "c3A", gene_list, common_genes, cts)
I = np.load(IMPUTED); print("imputed:", IMPUTED, I.shape, "out:", OUT, flush=True)
T = pd.read_csv(f"{OUT}/per_ct_gene_variance.csv.gz"); genes = np.array(genes)

def r2_on(x, L):
    x = (x - x.mean()) / x.std(); Z = L - L.mean(0); v = Z.var(0); cov = (Z * x[:, None]).mean(0)
    with np.errstate(divide="ignore", invalid="ignore"): return np.where(v > 1e-10, cov ** 2 / v, np.nan)

def pca_stats(Z, k=10):
    Z = Z - Z.mean(0); U, S, Vt = np.linalg.svd(Z, full_matrices=False); ev = S ** 2 / (S ** 2).sum()
    return ev, Vt[:k]

rows, dims = [], []
for ct in cts:
    mn, mc = nct == ct, cct == ct
    if mn.sum() < MIN_N or mc.sum() < MIN_N: continue
    Xn, Xc, Xi = Nc[mn], Cc[mc], I[mn]; Ln, Lc = np.log1p(Xn), np.log1p(Xc)
    ln_lib, lc_lib = np.log(Xn.sum(1)), np.log(Xc.sum(1))
    rows.append(pd.DataFrame({"cell_type": ct, "gene": genes, "r2_lib_nuc": r2_on(ln_lib, Ln), "r2_lib_imp": r2_on(ln_lib, Xi), "r2_lib_cell": r2_on(lc_lib, Lc)}))
    if mn.sum() >= 900 and mc.sum() >= 300:
        t = T[T.cell_type == ct].set_index("gene").loc[genes]; excess = (t.var_cell_all - t.var_cell_nullB).clip(lower=0).values
        cand = np.where((t.det_cell.values > 0.3) & (t.det_nuc.values > 0.1))[0]; top = cand[np.argsort(-excess[cand])[:300]]
        sc = RNG.choice(mc.sum(), min(5000, mc.sum()), replace=False); sn = RNG.choice(mn.sum(), min(5000, mn.sum()), replace=False)
        ev_c, V_c = pca_stats(Lc[sc][:, top]); ev_n, _ = pca_stats(Ln[sn][:, top]); ev_i, _ = pca_stats(Xi[sn][:, top])
        def in_sub(L):
            Z = L - L.mean(0); return float(((Z @ V_c.T) ** 2).sum() / (Z ** 2).sum())
        dims.append(dict(cell_type=ct, pr_cell=1 / (ev_c ** 2).sum(), pr_nuc=1 / (ev_n ** 2).sum(), pr_imp=1 / (ev_i ** 2).sum(),
                         pc1_cell=ev_c[0], pc1_nuc=ev_n[0], pc1_imp=ev_i[0], top5_cell=ev_c[:5].sum(), top5_nuc=ev_n[:5].sum(), top5_imp=ev_i[:5].sum(),
                         share_in_cell_top10_cell=in_sub(Lc[sc][:, top]), share_in_cell_top10_nuc=in_sub(Ln[sn][:, top]), share_in_cell_top10_imp=in_sub(Xi[sn][:, top])))
    print(ct, flush=True)
R = pd.concat(rows); R.to_csv(f"{OUT}/libsize_r2_per_ct_gene.csv.gz", index=False); D = pd.DataFrame(dims); D.to_csv(f"{OUT}/dimensionality.csv", index=False)
M = R.merge(T[["cell_type", "gene", "var_imp_all", "var_cell_all", "var_nuc_all", "det_nuc", "det_cell", "bio_excess_cell_A", "bio_excess_cell_B"]], on=["cell_type", "gene"])
M = M[(M.var_cell_all > 1e-4) & (M.det_cell > 0.01)]
print("variance-weighted share explained by library size: nuclei %.3f  imputed %.3f  cells %.3f" % ((M.r2_lib_nuc * M.var_nuc_all).sum() / M.var_nuc_all.sum(), (M.r2_lib_imp * M.var_imp_all).sum() / M.var_imp_all.sum(), (M.r2_lib_cell * M.var_cell_all).sum() / M.var_cell_all.sum()))
imp_nolib = (M.var_imp_all * (1 - M.r2_lib_imp.fillna(0))).sum()
print("imputed variance with library-size axis removed / real-cell biological excess (B): %.3f ; imputed total / excess A: %.3f" % (imp_nolib / M.bio_excess_cell_B.sum(), M.var_imp_all.sum() / M.bio_excess_cell_A.sum()))
print(D.round(3).to_string())
