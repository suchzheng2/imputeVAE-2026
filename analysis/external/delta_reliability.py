"""Split-half reliability of the empirical nucleus->cell delta in a matched external dataset: the ceiling for any
predicted-vs-empirical delta correlation. Random halves of nuclei and of cells per cluster -> delta_A, delta_B;
Pearson/Spearman across present genes, pooled over clusters and per cluster; also the Spearman-Brown corrected
full-sample reliability. Usage: python analysis/external/delta_reliability.py --nuclei N.h5ad --cells C.h5ad [--label_col cluster_id] [--n_splits 50]"""
import sys, json, argparse, numpy as np, pandas as pd, anndata as ad
from scipy import stats
sys.path.insert(0, "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE")
from analysis.external.eval_matched_external import SIGN_THRESHOLD

ap = argparse.ArgumentParser(); ap.add_argument("--nuclei", required=True); ap.add_argument("--cells", required=True)
ap.add_argument("--label_col", default="cluster_id"); ap.add_argument("--n_splits", type=int, default=50); ap.add_argument("--out", required=True)
a = ap.parse_args()
nuc, cell = ad.read_h5ad(a.nuclei), ad.read_h5ad(a.cells)
present = nuc.var["present"].values.astype(bool)
Xn, Xc = np.log1p(np.asarray(nuc.X, dtype=np.float32))[:, present], np.log1p(np.asarray(cell.X, dtype=np.float32))[:, present]
cn, cc = nuc.obs[a.label_col].astype(str).values, cell.obs[a.label_col].astype(str).values
cts = [c for c in sorted(set(cn) & set(cc)) if (cn == c).sum() >= 6 and (cc == c).sum() >= 6]
rng = np.random.default_rng(0)
pooled_r, pooled_rho, pooled_sign, per = [], [], [], {c: [] for c in cts}
for s in range(a.n_splits):
    dA, dB = [], []
    for c in cts:
        n_i, c_i = rng.permutation(np.where(cn == c)[0]), rng.permutation(np.where(cc == c)[0])
        hn, hc = len(n_i) // 2, len(c_i) // 2
        da = Xc[c_i[:hc]].mean(0) - Xn[n_i[:hn]].mean(0); db = Xc[c_i[hc:]].mean(0) - Xn[n_i[hn:]].mean(0)
        per[c].append(np.corrcoef(da, db)[0, 1]); dA.append(da); dB.append(db)
    A, B = np.concatenate(dA), np.concatenate(dB)
    pooled_r.append(np.corrcoef(A, B)[0, 1]); pooled_rho.append(stats.spearmanr(A, B)[0])
    big = np.abs(A) > SIGN_THRESHOLD; pooled_sign.append((np.sign(A[big]) == np.sign(B[big])).mean())
sb = lambda r: 2 * r / (1 + r)
res = {"n_splits": a.n_splits, "clusters": cts, "sign_threshold": SIGN_THRESHOLD,
       "pooled_split_half_pearson": float(np.mean(pooled_r)), "pooled_split_half_spearman": float(np.mean(pooled_rho)),
       "pooled_split_half_sign_agreement": float(np.mean(pooled_sign)),
       "pooled_full_sample_reliability_spearman_brown": float(sb(np.mean(pooled_r))),
       "per_cluster_split_half_pearson": {c: float(np.mean(v)) for c, v in per.items()},
       "per_cluster_full_sample_reliability_spearman_brown": {c: float(sb(np.mean(v))) for c, v in per.items()},
       "per_cluster_median_full_sample_reliability": float(np.median([sb(np.mean(v)) for v in per.values()]))}
json.dump(res, open(a.out, "w"), indent=2); print(json.dumps(res, indent=1))
