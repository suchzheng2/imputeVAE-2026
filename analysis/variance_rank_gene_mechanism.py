#!/usr/bin/env python3
"""Why does imputed variance survive for some genes and collapse for others? Two direct tests on L5 IT example genes.

Test A (latent loading): imputed value of gene g regressed on the active latent dimensions of the full-data cVAE
    (17, 4, 16, 19, 13 from latent_usage_check; 19 = library size). Per-dimension and joint linear R2, plus a
    random-forest R2 on the five dims as a nonlinear ceiling. Tells which latent axis carries the gene's imputed spread.

Test B (information available in the nucleus): in real L5 IT cells, take the top-K genes most correlated with g
    (partners). Score each nucleus by the mean log1p of those partners (nuclear partner score). If the partner score
    varies across nuclei and predicts real-cell g well, the nucleus profile carries per-cell information about g,
    whether or not the model uses it. Compare: corr(partner score, g) in real cells (ceiling), corr(nuclear partner
    score, imputed g), and the spread of the partner score in nuclei vs cells.

Inputs as in variance_rank_gene_examples.py plus latent_usage/mu_nuclei.npy (aligned to the imputed matrix).
Output: variance_rank/tables/gene_mechanism_L5IT_{latent,partners}.csv, figures/V6_gene_mechanism_L5IT.{png,pdf}
"""
import argparse
import numpy as np
import pandas as pd
import anndata as ad
import h5py
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE"
DATA_ROOT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi"
BASE = f"{PROJECT}/notes/experiments/variance_zero_nonzero_fullmodel"
TAB, FIG = f"{BASE}/variance_rank/tables", f"{BASE}/variance_rank/figures"
ACTIVE = [17, 4, 16, 19, 13]
LIB_DIM = 19
GENES = ["Pcp4", "Rplp1", "Rorb", "Calm2", "Malat1", "Xist", "Gm49164", "Gfra1"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#E5E5E5", "grid.linewidth": 0.6, "axes.axisbelow": True, "pdf.fonttype": 42})


def gene_index(a, gene_list, common):
    cm = (a.var["gene_names"].isin(common) & a.var["gene_names"].isin(gene_list)).values
    idx = np.where(cm)[0]; names = a.var["gene_names"].values[idx]
    order = np.argsort(names, kind="stable")
    return idx[order], names[order]


def load_ct(data_dir, batch, ct, cts, gene_list, common):
    a = ad.read_h5ad(f"{DATA_ROOT}/{data_dir}/data.h5ad", backed="r")
    mb = a.obs["batch_id"].astype(str).str.split(".").str[0]
    rm = (mb.str.endswith(batch) & a.obs["cell_type"].isin(cts)).values
    kept = np.where(rm)[0]
    ct_pos = np.where(a.obs["cell_type"].astype(str).values[kept] == ct)[0]
    col_idx, names = gene_index(a, gene_list, common)
    a.file.close()
    rows = kept[ct_pos]
    X = np.zeros((len(rows), len(col_idx)), dtype=np.float32)
    with h5py.File(f"{DATA_ROOT}/{data_dir}/data.h5ad", "r") as f:
        D = f["X"]; step = 3000
        for lo in range(int(rows.min()), int(rows.max()) + 1, step):
            hi = min(lo + step, int(rows.max()) + 1)
            sel = np.where((rows >= lo) & (rows < hi))[0]
            if len(sel) == 0:
                continue
            block = D[lo:hi][:, col_idx]
            X[sel] = block[rows[sel] - lo]
    return np.log1p(X), ct_pos, names, len(kept)


def r2(y, yhat):
    return 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cell_type", default="L5 IT")
    p.add_argument("--top_k", type=int, default=30)
    args = p.parse_args(); ct = args.cell_type; K = args.top_k
    gene_list = pd.read_csv(f"{PROJECT}/data/training_gene_list_v3.csv", header=None)[0].tolist()
    common = pd.read_csv(f"{PROJECT}/data/common_genes.csv")["common_genes"].tolist()
    cts = pd.read_csv(f"{PROJECT}/data/common_cell_types.csv")["common_cell_types"].tolist()

    Ln, npos, names, n_kept = load_ct("10X_nuclei_v2_v3_AIBS", "n3A", ct, cts, gene_list, common)
    Lc, _, names_c, _ = load_ct("10X_cells_v2_v3_AIBS", "c3A", ct, cts, gene_list, common)
    assert (names == names_c).all()
    I = np.load(f"{PROJECT}/run/cvae_full_eval/imputed_cell_from_nucleus.npy", mmap_mode="r"); assert I.shape[0] == n_kept
    ict = pd.read_csv(f"{PROJECT}/run/cvae_full_eval/nucleus_obs.csv")["cell_type"].astype(str).values; assert (ict[npos] == ct).all()
    Li = np.asarray(I[npos], dtype=np.float32)
    MU = np.load(f"{BASE}/latent_usage/mu_nuclei.npy")[npos]
    print(f"{ct}: {Lc.shape[0]} cells, {Ln.shape[0]} nuclei, {len(names)} genes; mu {MU.shape}", flush=True)
    gi = {g: int(np.where(names == g)[0][0]) for g in GENES}
    log_lib_n = np.log(np.expm1(Ln).sum(1)); log_lib_c = np.log(np.expm1(Lc).sum(1))

    # ---------------- Test A: latent loading
    rows_a = []
    Z = MU[:, ACTIVE]
    for g in GENES:
        y = Li[:, gi[g]]
        row = dict(gene=g, var_imputed=float(y.var()))
        for d in ACTIVE:
            row[f"r2_dim{d}"] = float(r2(y, LinearRegression().fit(MU[:, [d]], y).predict(MU[:, [d]])))
        row["r2_active5_linear"] = float(r2(y, LinearRegression().fit(Z, y).predict(Z)))
        rf = RandomForestRegressor(n_estimators=150, min_samples_leaf=20, n_jobs=-1, random_state=0)
        row["r2_active5_rf_cv"] = float(r2(y, cross_val_predict(rf, Z, y, cv=5)))
        row["r2_loglib_imputed"] = float(r2(y, LinearRegression().fit(log_lib_n[:, None], y).predict(log_lib_n[:, None])))
        # variance left after removing the library-size dim linearly: what a non-library axis contributes
        resid = y - LinearRegression().fit(MU[:, [LIB_DIM]], y).predict(MU[:, [LIB_DIM]])
        nonlib = [d for d in ACTIVE if d != LIB_DIM]
        row["r2_nonlib4_on_libresid"] = float(r2(resid, LinearRegression().fit(MU[:, nonlib], resid).predict(MU[:, nonlib])))
        row["share_var_not_libsize"] = float(resid.var() / y.var())
        rows_a.append(row)
    A = pd.DataFrame(rows_a).set_index("gene"); A.to_csv(f"{TAB}/gene_mechanism_L5IT_latent.csv")
    pd.set_option("display.width", 250); print("\n=== Test A: imputed value vs active latent dims ==="); print(A.round(3).to_string())

    # ---------------- Test B: information available in the nucleus via co-expression partners
    det_n = (Ln > 0).mean(0); det_c = (Lc > 0).mean(0)
    keep = (det_c > 0.05)  # candidate partners: reasonably detected in cells
    Lc_std = (Lc - Lc.mean(0)) / (Lc.std(0) + 1e-8)
    rows_b = []; partners_out = {}
    for g in GENES:
        j = gi[g]
        r_c = (Lc_std * Lc_std[:, [j]]).mean(0); r_c[j] = 0; r_c[~keep] = 0
        top = np.argsort(-np.abs(r_c))[:K]; sign = np.sign(r_c[top])
        partners_out[g] = pd.DataFrame(dict(gene=g, partner=names[top], r_in_real_cells=r_c[top], det_cell=det_c[top], det_nuc=det_n[top]))
        # signed partner score = mean of standardised partner values (standardised within each source)
        def score(L):
            S = (L[:, top] - L[:, top].mean(0)) / (L[:, top].std(0) + 1e-8)
            return (S * sign).mean(1)
        sc_c, sc_n = score(Lc), score(Ln)
        # nucleus-side partner score should be built from partners actually detectable in nuclei
        top_n = top[det_n[top] > 0.2]; sign_n = sign[det_n[top] > 0.2]
        def score_n(L, idx, sg):
            if len(idx) == 0: return np.zeros(L.shape[0])
            S = (L[:, idx] - L[:, idx].mean(0)) / (L[:, idx].std(0) + 1e-8); return (S * sg).mean(1)
        sc_n2 = score_n(Ln, top_n, sign_n); sc_c2 = score_n(Lc, top_n, sign_n)
        row = dict(gene=g, det_cell=float(det_c[j]), det_nuc=float(det_n[j]), n_partners=K,
                   mean_abs_r_partners_real=float(np.abs(r_c[top]).mean()),
                   n_partners_det_nuc_gt20=int(len(top_n)),
                   # ceiling: how well partners predict g in real cells
                   spearman_partner_vs_g_realcell=float(stats.spearmanr(sc_c, Lc[:, j])[0]),
                   spearman_partnerNucDet_vs_g_realcell=float(stats.spearmanr(sc_c2, Lc[:, j])[0]) if len(top_n) else np.nan,
                   # does the nuclear partner score vary? (relative to its spread in cells, same gene set)
                   sd_partner_score_nuc_over_cell=float(sc_n2.std() / (sc_c2.std() + 1e-8)) if len(top_n) else np.nan,
                   # does the model use it?
                   spearman_partnerNuc_vs_imputed=float(stats.spearmanr(sc_n2, Li[:, j])[0]) if len(top_n) else np.nan,
                   spearman_partnerNuc_vs_ownNuc=float(stats.spearmanr(sc_n2, Ln[:, j])[0]) if len(top_n) else np.nan,
                   spearman_ownNuc_vs_imputed=float(stats.spearmanr(Ln[:, j], Li[:, j])[0]),
                   spearman_loglib_vs_imputed=float(stats.spearmanr(log_lib_n, Li[:, j])[0]),
                   spearman_partnerNuc_vs_loglib=float(stats.spearmanr(sc_n2, log_lib_n)[0]) if len(top_n) else np.nan,
                   # partial: partner score vs imputed after removing library size from both
                   )
        if len(top_n):
            rp = sc_n2 - np.polyval(np.polyfit(log_lib_n, sc_n2, 1), log_lib_n)
            ri = Li[:, j] - np.polyval(np.polyfit(log_lib_n, Li[:, j], 1), log_lib_n)
            row["spearman_partnerNuc_vs_imputed_libremoved"] = float(stats.spearmanr(rp, ri)[0])
            rc = sc_c2 - np.polyval(np.polyfit(log_lib_c, sc_c2, 1), log_lib_c)
            rg = Lc[:, j] - np.polyval(np.polyfit(log_lib_c, Lc[:, j], 1), log_lib_c)
            row["spearman_partner_vs_g_realcell_libremoved"] = float(stats.spearmanr(rc, rg)[0])
        rows_b.append(row)
    Bt = pd.DataFrame(rows_b).set_index("gene"); Bt.to_csv(f"{TAB}/gene_mechanism_L5IT_partners.csv")
    pd.concat(partners_out.values()).to_csv(f"{TAB}/gene_mechanism_L5IT_partner_lists.csv", index=False)
    print("\n=== Test B: co-expression partners (top %d in real cells) ===" % K); print(Bt.round(3).to_string())

    # ---------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    ax = axes[0]; x = np.arange(len(GENES)); w = 0.16
    for i, d in enumerate(ACTIVE):
        ax.bar(x + (i - 2) * w, A[f"r2_dim{d}"], w, label=f"dim {d}" + (" (library size)" if d == LIB_DIM else ""))
    ax.scatter(x, A["r2_active5_rf_cv"], color="black", s=18, zorder=3, label="5 dims, RF (cv)")
    ax.set_xticks(x); ax.set_xticklabels(GENES, rotation=30, ha="right"); ax.set_ylabel("R2 of imputed value"); ax.set_ylim(0, 1.05)
    ax.set_title("A. Which latent axis carries the imputed spread?", fontsize=9); ax.legend(frameon=False, fontsize=7, ncol=2)
    ax = axes[1]
    ax.bar(x - 0.3, Bt["spearman_partner_vs_g_realcell"], 0.2, color="#E69F00", label="partners -> g, real cells (ceiling)")
    ax.bar(x - 0.1, Bt["sd_partner_score_nuc_over_cell"], 0.2, color="#7F7F7F", label="sd(partner score) nucleus / cell")
    ax.bar(x + 0.1, Bt["spearman_partnerNuc_vs_imputed"], 0.2, color="#0072B2", label="nuclear partners -> imputed g")
    ax.bar(x + 0.3, Bt["spearman_partnerNuc_vs_imputed_libremoved"], 0.2, color="#56B4E9", label="same, library size removed")
    ax.axhline(0, color="#7F7F7F", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(GENES, rotation=30, ha="right"); ax.set_ylim(-0.4, 1.1)
    ax.set_title("B. Does the nucleus carry information about g, and does the model use it?", fontsize=9); ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(f"{FIG}/V6_gene_mechanism_L5IT.png", dpi=200, bbox_inches="tight"); fig.savefig(f"{FIG}/V6_gene_mechanism_L5IT.pdf", bbox_inches="tight")
    print("saved", f"{FIG}/V6_gene_mechanism_L5IT.png")


if __name__ == "__main__":
    main()
