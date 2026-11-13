"""Per-gene distribution agreement within one cell type, as a replacement for the variance ratio (2026-11-13).

For every gene g and cell type ct, with real cells R, raw nuclei N and an imputed decode I (all log1p counts):
  W1(I, R)   1-D Wasserstein distance between the imputed and the real per-cell distributions (location + shape)
  W1(N, R)   the distance the imputation had to travel (nuclei vs real cells)
  floor      W1 between two random halves of the real cells (what "identical" looks like with finite cells)
  closed     1 - (W1(I, R) - floor) / (W1(N, R) - floor): 1 = indistinguishable from real cells, 0 = no better than
             the raw nuclei, < 0 = worse than nuclei.  Clipped to [-1, 1] for summaries.
Also the variance ratio (var I / var R) on the same genes for comparison, and the KS statistic.
Series: old model (no covariate), covariate per-cell library (qmap), the same + Poisson sampling.
Genes are stratified by real-cell detection rate (<10 %, 10-30 %, 30-70 %, >70 %).
Usage: python analysis/benchmark/distribution_closure_ct.py --ct "L5 IT"
"""
import argparse, os, sys, time
import numpy as np, pandas as pd
from scipy.stats import wasserstein_distance, ks_2samp
sys.path.insert(0, os.path.dirname(__file__)); from data_split import load_partition, load_gene_names

BASE = "run/benchmark/test"
SERIES = [("old model (no covariate)", "cvae_seed1", False), ("covariate, per-cell library", "cvae_libsize_armA_qmap", False),
          ("covariate, per-cell library + Poisson", "cvae_libsize_armA_qmap", True)]


def w1_cols(A, B):
    return np.array([wasserstein_distance(A[:, j], B[:, j]) for j in range(A.shape[1])])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ct", default="L5 IT"); ap.add_argument("--out", default="notes/experiments/libsize_covariate/armA/tables")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    genes = load_gene_names()
    Xc, oc = load_partition("cell", "test", "log1p"); Xn, on = load_partition("nuc", "test", "log1p")
    R = Xc[oc.cell_type.values == a.ct]; N = Xn[on.cell_type.values == a.ct]
    rng = np.random.default_rng(0); perm = rng.permutation(len(R)); h = len(R) // 2
    floor = w1_cols(R[perm[:h]], R[perm[h:]]); w_nr = w1_cols(N, R); print(f"floor / nuclei done {time.time()-t0:.0f}s", flush=True)
    det = (R > 0.05).mean(0); vr = R.var(0)
    strata = pd.cut(det, [-1, 0.1, 0.3, 0.7, 1.01], labels=["<10%", "10-30%", "30-70%", ">70%"])
    per_gene = pd.DataFrame({"gene": genes, "detection_real": det, "var_real": vr, "stratum": strata, "W1_nuc_real": w_nr, "W1_floor": floor})
    summ = []
    for name, d, poisson in SERIES:
        im = np.load(f"{BASE}/{d}/imputed_eval.npy", mmap_mode="r"); ob = pd.read_csv(f"{BASE}/{d}/nuc_eval_obs.csv")
        I = np.asarray(im[ob.cell_type.values == a.ct])
        if poisson: I = np.log1p(rng.poisson(np.expm1(np.clip(I, 0, None))))
        w = w1_cols(I, R); ks = np.array([ks_2samp(I[:, j], R[:, j]).statistic for j in range(I.shape[1])])
        closed = np.clip(1 - (w - floor) / np.maximum(w_nr - floor, 1e-6), -1, 1)
        vratio = I.var(0) / np.maximum(vr, 1e-9)
        tag = name.split(" (")[0].replace(", ", "_").replace(" + ", "_").replace(" ", "")
        per_gene[f"W1_{tag}"] = w; per_gene[f"closed_{tag}"] = closed; per_gene[f"KS_{tag}"] = ks; per_gene[f"varratio_{tag}"] = vratio
        ok = (w_nr - floor) > 0.05  # genes where nuclei and cells actually differ; otherwise "closed" is undefined
        for s in list(strata.categories) + ["all"]:
            m = ok & ((strata == s) if s != "all" else True)
            summ.append({"series": name, "stratum": s, "n_genes": int(m.sum()), "median closed": float(np.median(closed[m])),
                         "frac closed > 0.8": float((closed[m] > 0.8).mean()), "frac worse than nuclei": float((closed[m] < 0).mean()),
                         "median W1/floor": float(np.median(w[m] / np.maximum(floor[m], 1e-6))), "median KS": float(np.median(ks[m])),
                         "median var ratio": float(np.median(vratio[m]))})
        print(f"{name}: done {time.time()-t0:.0f}s", flush=True)
    tagct = a.ct.replace(" ", "").replace("/", "")
    per_gene.to_csv(f"{a.out}/distribution_closure_{tagct}_per_gene.csv.gz", index=False)
    S = pd.DataFrame(summ); S.to_csv(f"{a.out}/distribution_closure_{tagct}_summary.csv", index=False)
    pd.set_option("display.width", 250); print(S.round(3).to_string(index=False))
    # the genes the variance ratio punishes although the distribution is matched
    pg = per_gene[ok]; worst = pg[(pg["closed_covariate_per-celllibrary_Poisson"] > 0.9) & (pg["varratio_covariate_per-celllibrary_Poisson"] < 0.6)]
    print(f"\ngenes with closure > 0.9 but variance ratio < 0.6 (Poisson decode): {len(worst)} of {ok.sum()}; e.g.",
          ", ".join(worst.sort_values("var_real", ascending=False).gene.head(12)))


if __name__ == "__main__":
    main()
