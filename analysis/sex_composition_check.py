#!/usr/bin/env python3
"""Sex composition of the v3 cell and nucleus datasets from raw counts of Xist and Y-linked genes, per sequencing library.
Reads only the needed columns of the dense raw h5ad (one pass). Output: variance_rank/tables/sex_composition_by_library.csv"""
import h5py, numpy as np, pandas as pd
DATA_ROOT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi"
TAB = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE/notes/experiments/variance_zero_nonzero_fullmodel/variance_rank/tables"
GENES = ["Xist", "Tsix", "Ddx3y", "Eif2s3y", "Kdm5d", "Uty"]
rows = []
for d, suffix in (("10X_cells_v2_v3_AIBS", "c3A"), ("10X_nuclei_v2_v3_AIBS", "n3A")):
    with h5py.File(f"{DATA_ROOT}/{d}/data.h5ad", "r") as f:
        gn = np.array([x.decode() for x in f["var/gene_names"][:]]) if f["var/gene_names"].dtype.kind in "OS" else f["var/gene_names"][:]
        cols = {g: int(np.where(gn == g)[0][0]) for g in GENES if (gn == g).any()}
        print(d, "found:", list(cols), flush=True)
        cats = np.array([c.decode() for c in f["obs/batch_id/categories"][:]]); lib = pd.Series(cats[f["obs/batch_id/codes"][:]]).str.split(".").str[0].values
        ctc = np.array([c.decode() for c in f["obs/cell_type/categories"][:]]); ct = ctc[f["obs/cell_type/codes"][:]]
        D = f["X"]; n = D.shape[0]; ci = np.array(sorted(cols.values())); X = np.zeros((n, len(ci)), dtype=np.int64)
        for lo in range(0, n, 4000):
            X[lo:lo + 4000] = D[lo:lo + 4000][:, ci]
    names = [g for g in GENES if g in cols]; order = [list(ci).index(cols[g]) for g in names]; X = X[:, order]
    keep = pd.Series(lib).str.endswith(suffix).values & (ct != "Doublet")
    df = pd.DataFrame(X[keep], columns=names); df["library"] = lib[keep]
    for L, g in df.groupby("library"):
        r = dict(dataset=d, library=L, n=len(g))
        for nm in names: r[f"det_{nm}"] = float((g[nm] > 0).mean()); r[f"mean_{nm}"] = float(g[nm].mean())
        if "Xist" in names and "Ddx3y" in names:
            r["frac_female_like"] = float(((g.Xist > 0) & (g.Ddx3y == 0)).mean()); r["frac_male_like"] = float(((g.Xist == 0) & (g.Ddx3y > 0)).mean()); r["frac_both"] = float(((g.Xist > 0) & (g.Ddx3y > 0)).mean())
        rows.append(r)
R = pd.DataFrame(rows); R.to_csv(f"{TAB}/sex_composition_by_library.csv", index=False)
pd.set_option("display.width", 300); print(R.round(3).to_string())
