#!/usr/bin/env python3
"""Cause of the variance collapse for Gm49164 and Gfra1 (L5 IT): is the real-cell variance technical, sampling, or biology?

For each gene (plus Pcp4 / Rorb / Actb as references), in real L5 IT cells and in raw L5 IT nuclei:
  - share of variance explained by sequencing library (batch_id prefix, 12 cell / 5 nucleus libraries), by n_genes, by log library size
  - correlation with Actb and Actg1 (multimapping / pseudogene hypothesis for Gm49164)
  - partner ceiling (top-30 co-expression partners) recomputed after residualising library, n_genes and log library size
  - Bernoulli share: variance of the detection indicator times squared nonzero mean, over total variance (sampling-dominated genes)
  - is the partner module coherent inside nuclei? mean pairwise correlation of the partners in cells vs in nuclei
Output: variance_rank/tables/gene_cause_L5IT.csv
"""
import numpy as np
import pandas as pd
import anndata as ad
import h5py
from scipy import stats

PROJECT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE"
DATA_ROOT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi"
BASE = f"{PROJECT}/notes/experiments/variance_zero_nonzero_fullmodel"
TAB = f"{BASE}/variance_rank/tables"
GENES = ["Gm49164", "Gfra1", "Gm15810", "Pcp4", "Rorb", "Calm2", "Actb"]
CT = "L5 IT"


def gene_index(a, gene_list, common):
    cm = (a.var["gene_names"].isin(common) & a.var["gene_names"].isin(gene_list)).values
    idx = np.where(cm)[0]; names = a.var["gene_names"].values[idx]
    order = np.argsort(names, kind="stable"); return idx[order], names[order]


def load_ct(data_dir, batch, cts, gene_list, common):
    a = ad.read_h5ad(f"{DATA_ROOT}/{data_dir}/data.h5ad", backed="r")
    bid = a.obs["batch_id"].astype(str); mb = bid.str.split(".").str[0]
    rm = (mb.str.endswith(batch) & a.obs["cell_type"].isin(cts)).values
    kept = np.where(rm)[0]; ctv = a.obs["cell_type"].astype(str).values[kept]
    ct_pos = np.where(ctv == CT)[0]; lib = mb.values[kept][ct_pos]; ngenes = a.obs["n_genes"].values[kept][ct_pos].astype(float)
    col_idx, names = gene_index(a, gene_list, common); a.file.close()
    rows = kept[ct_pos]; X = np.zeros((len(rows), len(col_idx)), dtype=np.float32)
    with h5py.File(f"{DATA_ROOT}/{data_dir}/data.h5ad", "r") as f:
        D = f["X"]; step = 3000
        for lo in range(int(rows.min()), int(rows.max()) + 1, step):
            hi = min(lo + step, int(rows.max()) + 1); sel = np.where((rows >= lo) & (rows < hi))[0]
            if len(sel): X[sel] = D[lo:hi][:, col_idx][rows[sel] - lo]
    return np.log1p(X), names, lib, ngenes, ct_pos


def anova_r2(y, groups):
    df = pd.DataFrame(dict(y=y, g=groups)); m = df.groupby("g")["y"].transform("mean")
    return float(1 - ((y - m.values) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def resid(y, covs):
    Xd = np.column_stack([np.ones(len(y))] + covs); beta, *_ = np.linalg.lstsq(Xd, y, rcond=None); return y - Xd @ beta


def onehot(groups):
    u = pd.get_dummies(pd.Series(groups), drop_first=True).values.astype(float); return [u[:, i] for i in range(u.shape[1])]


def main():
    gene_list = pd.read_csv(f"{PROJECT}/data/training_gene_list_v3.csv", header=None)[0].tolist()
    common = pd.read_csv(f"{PROJECT}/data/common_genes.csv")["common_genes"].tolist()
    cts = pd.read_csv(f"{PROJECT}/data/common_cell_types.csv")["common_cell_types"].tolist()
    Lc, names, lib_c, ng_c, _ = load_ct("10X_cells_v2_v3_AIBS", "c3A", cts, gene_list, common)
    Ln, names2, lib_n, ng_n, npos = load_ct("10X_nuclei_v2_v3_AIBS", "n3A", cts, gene_list, common); assert (names == names2).all()
    I = np.load(f"{PROJECT}/run/cvae_full_eval/imputed_cell_from_nucleus.npy", mmap_mode="r"); Li = np.asarray(I[npos], dtype=np.float32)
    print(f"cells {Lc.shape}, nuclei {Ln.shape}; cell libs {len(set(lib_c))}, nucleus libs {len(set(lib_n))}", flush=True)
    gi = {g: int(np.where(names == g)[0][0]) for g in GENES if g in names}
    det_c = (Lc > 0).mean(0); det_n = (Ln > 0).mean(0)
    Lc_std = (Lc - Lc.mean(0)) / (Lc.std(0) + 1e-8)
    loglib_c = np.log(np.expm1(Lc).sum(1)); loglib_n = np.log(np.expm1(Ln).sum(1))
    rows = []
    for g, j in gi.items():
        for src, L, lib, ng, loglib in (("cell", Lc, lib_c, ng_c, loglib_c), ("nucleus", Ln, lib_n, ng_n, loglib_n)):
            y = L[:, j]; row = dict(gene=g, source=src, det=float((y > 0).mean()), var=float(y.var()))
            row["r2_library"] = anova_r2(y, lib)
            row["r2_n_genes"] = float(stats.pearsonr(y, ng)[0] ** 2)
            row["r2_loglib"] = float(stats.pearsonr(y, loglib)[0] ** 2)
            row["r2_lib_ngenes_loglib"] = float(1 - resid(y, onehot(lib) + [ng, loglib]).var() / y.var())
            for ref in ("Actb", "Actg1", "Malat1"):
                k = int(np.where(names == ref)[0][0]); row[f"r_{ref}"] = float(stats.pearsonr(y, L[:, k])[0])
                row[f"r_{ref}_within_lib"] = float(stats.pearsonr(resid(y, onehot(lib)), resid(L[:, k], onehot(lib)))[0])
            # Bernoulli (detection) share of variance
            ind = (y > 0).astype(float); mnz = y[y > 0].mean() if (y > 0).any() else 0.0
            row["bernoulli_share"] = float(ind.var() * mnz ** 2 / y.var()) if y.var() > 0 else np.nan
            if src == "cell":
                r_c = (Lc_std * Lc_std[:, [j]]).mean(0); r_c[j] = 0; r_c[det_c <= 0.05] = 0
                top = np.argsort(-np.abs(r_c))[:30]; sign = np.sign(r_c[top])
                def score(L_, idx, sg):
                    S = (L_[:, idx] - L_[:, idx].mean(0)) / (L_[:, idx].std(0) + 1e-8); return (S * sg).mean(1)
                sc = score(Lc, top, sign)
                row["partner_ceiling"] = float(stats.spearmanr(sc, y)[0])
                covs = onehot(lib_c) + [ng_c, loglib_c]
                row["partner_ceiling_within_lib_qc"] = float(stats.spearmanr(resid(sc, covs), resid(y, covs))[0])
                row["partner_score_r2_library"] = anova_r2(sc, lib_c)
                # module coherence: mean pairwise |r| among partners in cells vs nuclei (partners detected >20% in nuclei)
                tn = top[det_n[top] > 0.2]
                def coh(L_, idx):
                    if len(idx) < 3: return np.nan
                    C = np.corrcoef(L_[:, idx].T); iu = np.triu_indices(len(idx), 1); return float(np.abs(C[iu]).mean())
                row["partner_coherence_cells"] = coh(Lc, tn); row["partner_coherence_nuclei"] = coh(Ln, tn)
                row["partner_coherence_nuclei_within_lib"] = coh(np.column_stack([resid(Ln[:, k], onehot(lib_n)) for k in tn]) if len(tn) >= 3 else Ln, np.arange(len(tn)) if len(tn) >= 3 else tn)
                # imputed: does it track library of the nucleus?
                row["imputed_r2_library"] = anova_r2(Li[:, j], lib_n); row["imputed_var"] = float(Li[:, j].var())
            rows.append(row)
    R = pd.DataFrame(rows); R.to_csv(f"{TAB}/gene_cause_L5IT.csv", index=False)
    pd.set_option("display.width", 300); print(R.round(3).to_string())


if __name__ == "__main__":
    main()
