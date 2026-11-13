"""Per-gene within-type variance distribution for one cell type, real vs imputed decodes (re-created 2026-11-13).

Reproduces notes/experiments/libsize_covariate/armA/figures/fig_variance_distribution_L5IT.png and the table
armA/tables/variance_distribution_L5IT_test.csv (originally made 2026-11-06 from a scratch script). Asked by Prashant
(meeting 2026-11-06): the distribution over genes, not the mean ratio, for a large type with and without the library
covariate. Sources are the split-aware benchmark decodes under run/benchmark/test/<method>/imputed_eval.npy.

Usage: python analysis/benchmark/variance_distribution_ct.py --ct "L5 IT" --out notes/experiments/libsize_covariate/armA
"""
import argparse, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr, gaussian_kde
sys.path.insert(0, os.path.dirname(__file__)); from data_split import load_partition

BASE = "run/benchmark/test"
SOURCES = [("imputed, old model (no covariate)", "cvae_seed1", "#9ecae1"),
           ("imputed, covariate, fixed per-type library (ctmed)", "cvae_libsize_armA", "#d62728"),
           ("imputed, covariate, quantile map (label)", "cvae_libsize_armA_qmap", "#1f77b4"),
           ("imputed, covariate, label-free predicted type", "cvae_libsize_armA_knntypeq", "#2ca02c")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ct", default="L5 IT"); ap.add_argument("--out", default="notes/experiments/libsize_covariate/armA")
    ap.add_argument("--partition", default="test"); a = ap.parse_args()
    Xc, oc = load_partition("cell", a.partition, "log1p"); Xn, on = load_partition("nuc", a.partition, "log1p")
    Rc, _ = load_partition("cell", a.partition, "raw")
    mc = oc.cell_type.values == a.ct
    vc = Xc[mc].var(0); vn = Xn[on.cell_type.values == a.ct].var(0)
    # Prashant 2026-11-06 ("numerator and denominator"): real-cell variance after regressing log library size out of
    # every gene within the type = the depth-free part of the observed variance, to set against the fixed-library decode
    ls = np.log(Rc[mc].sum(1)); lsc = (ls - ls.mean())[:, None]
    Y = Xc[mc]; beta = (lsc * (Y - Y.mean(0))).sum(0) / (lsc ** 2).sum()
    vres = (Y - Y.mean(0) - lsc * beta).var(0)
    keep = vc > 0
    rows, series = [], [("real cells", vc, "k", "-"), ("real cells, library size regressed out", vres, "k", ":"),
                        ("raw nuclei", vn, "gray", "--")]
    for name, d, col in SOURCES:
        im = np.load(f"{BASE}/{d}/imputed_eval.npy", mmap_mode="r"); ob = pd.read_csv(f"{BASE}/{d}/nuc_eval_obs.csv")
        series.append((name, np.asarray(im[ob.cell_type.values == a.ct]).var(0), col, "-"))
    # Poisson-sampled decode: the model outputs expected log1p counts; sampling counts from expm1 of the quantile-mapped
    # decode adds the measurement noise that real cells carry (52 % of their within-type variance, diagnostic 2026-09-18)
    rng = np.random.default_rng(0)
    imq = np.load(f"{BASE}/cvae_libsize_armA_qmap/imputed_eval.npy", mmap_mode="r"); obq = pd.read_csv(f"{BASE}/cvae_libsize_armA_qmap/nuc_eval_obs.csv")
    lam = np.expm1(np.clip(np.asarray(imq[obq.cell_type.values == a.ct]), 0, None))
    series.append(("imputed, covariate, quantile map + Poisson sampling", np.log1p(rng.poisson(lam)).var(0), "#ff7f0e", "-"))
    for name, v, _, _ in series:
        r = v[keep] / vc[keep]
        rows.append({"source": name, "median var": float(np.median(v[keep])), "ratio to real (median of per-gene ratio)": float(np.median(r)),
                     "Spearman vs real": float(spearmanr(v[keep], vc[keep]).statistic), "IQR log10 var": float(np.subtract(*np.percentile(np.log10(v[keep] + 1e-12), [75, 25])))})
    tab = pd.DataFrame(rows); tag = a.ct.replace(" ", "").replace("/", "")
    os.makedirs(f"{a.out}/tables", exist_ok=True); os.makedirs(f"{a.out}/figures", exist_ok=True)
    tab.to_csv(f"{a.out}/tables/variance_distribution_{tag}_{a.partition}.csv", index=False); print(tab.round(4).to_string(index=False))
    fig, axes = plt.subplots(1, 4, figsize=(26, 5.5)); ax = axes[0]; grid = np.linspace(-6, 0.5, 400)
    for name, v, col, ls in series:
        lv = np.log10(v[keep] + 1e-12); ax.plot(grid, gaussian_kde(lv)(grid), color=col, ls=ls, lw=2, label=f"{name} (median {np.median(v[keep]):.3g})")
    ax.set_xlabel("log10 within-type per-gene variance (log1p scale)"); ax.set_ylabel("density"); ax.legend(fontsize=8); ax.set_title(f"{a.ct}, {a.partition} partition: {int(keep.sum())} genes")
    fixed, qmap, pois = series[4], series[5], series[7]
    for ax, (name, v, _, _), (xname, xv) in zip(axes[1:], [fixed, qmap, pois], [("real cells", vc), ("real cells", vc), ("real cells", vc)]):
        ax.hexbin(np.log10(xv[keep]), np.log10(v[keep] + 1e-12), gridsize=60, bins="log", mincnt=1); ax.plot([-6, 0.5], [-6, 0.5], "r--", lw=1, label="y = x")
        ax.set_xlim(-6, 0.5); ax.set_ylim(-6, 0.5); ax.set_xlabel(f"log10 var, {xname}"); ax.set_ylabel(f"log10 var, {name.replace('imputed, ', '')}")
        ax.set_title(f"Spearman {spearmanr(v[keep], xv[keep]).statistic:.2f}, median ratio {np.median(v[keep]/xv[keep]):.3g}"); ax.legend()
    tab["Spearman vs real (libsize regressed)"] = [float(spearmanr(v[keep], vres[keep]).statistic) for _, v, _, _ in series]
    tab["ratio to real (libsize regressed)"] = [float(np.median(v[keep] / vres[keep])) for _, v, _, _ in series]
    tab.to_csv(f"{a.out}/tables/variance_distribution_{tag}_{a.partition}.csv", index=False); print(tab.round(4).to_string(index=False))
    fig.tight_layout(); fig.savefig(f"{a.out}/figures/fig_variance_distribution_{tag}.png", dpi=120); print("wrote figure")


if __name__ == "__main__":
    main()
