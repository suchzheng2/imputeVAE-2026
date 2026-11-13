"""Scoping pass over the 10x Xenium Prime 5K mouse brain coronal section (2026-11-13).

Answers the open items of notes/experiments/spatial_merfish_application/README.md before any modelling:
  (a) which cortical area / bregma level the section covers (layer and region markers per 10x graph cluster,
      cluster centroids in section coordinates), and
  (b) how the 10x clusters map onto our training types (marker-based coarse labels), i.e. what label transfer
      has to achieve.
Also records the gene-space overlap with the 10,134-gene training list for the cells actually present.
No scanpy (its import is broken in impute2 by a matplotlib clash); h5py + scipy only.

Inputs (scratch): cell_feature_matrix.h5, cells.parquet, analysis/clustering/gene_expression_graphclust/clusters.csv
Outputs: notes/experiments/spatial_merfish_application/scoping/ (tables only; no model is run).
"""
import argparse, json, os
import h5py, numpy as np, pandas as pd, scipy.sparse as sp

XEN = "/vast/palmer/scratch/gerstein/sz477/xenium_prime5k_mouse_brain"
OUT = "notes/experiments/spatial_merfish_application/scoping"
GENE_LIST = "data/training_gene_list_v3.csv"

# markers: cortical layers / our training types / non-cortical regions that fix the bregma level
MARKERS = {
    "L2/3 IT": ["Cux2", "Otof", "Rasgrf2"], "L4/5 IT": ["Rorb", "Rspo1", "Whrn"], "L5 IT": ["Deptor", "Etv1"],
    "L5 ET": ["Fezf2", "Bcl11b", "Pou3f1"], "L5/6 NP": ["Tshz2", "Nxph1"], "L6 CT": ["Syt6", "Foxp2"],
    "L6 IT": ["Osr1", "Car3"], "L6b": ["Ctgf", "Ccn2", "Cplx3"],
    "Pvalb": ["Pvalb"], "Sst": ["Sst"], "Vip": ["Vip"], "Lamp5": ["Lamp5", "Ndnf"], "Sncg": ["Sncg"],
    "Astro": ["Aqp4", "Gja1", "Slc1a3"], "Oligo": ["Mbp", "Mog", "Plp1"], "OPC": ["Pdgfra", "Cspg4"],
    "Micro": ["Cx3cr1", "P2ry12"], "Endo": ["Cldn5", "Flt1"], "Peri": ["Pdgfrb", "Kcnj8"],
    # region markers
    "Striatum": ["Ppp1r1b", "Drd1", "Drd2", "Penk"], "Hippocampus": ["Prox1", "Zbtb20", "Neurod6"],
    "Thalamus": ["Tcf7l2", "Prkcd"], "Hypothal": ["Hcrt", "Pomc", "Agrp"], "Piriform": ["Nrgn", "Tbr1"],
    "Olfactory": ["Eomes", "Sp8"], "Amygdala": ["Lypd1"],
}


def read_10x_h5(path):
    with h5py.File(path) as f:
        m = f["matrix"]
        X = sp.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=tuple(m["shape"][:])).T.tocsr()
        names = m["features"]["name"][:].astype(str)
        ftype = m["features"]["feature_type"][:].astype(str)
        bc = m["barcodes"][:].astype(str)
    return X, names, ftype, bc


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=OUT); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    X, names, ftype, bc = read_10x_h5(f"{XEN}/cell_feature_matrix.h5")
    keep = ftype == "Gene Expression"
    X, names = X[:, keep], names[keep]
    gi = {g: i for i, g in enumerate(names)}
    cells = pd.read_csv(f"{XEN}/cells.csv.gz").set_index("cell_id")  # no pyarrow in impute2
    cl = pd.read_csv(f"{XEN}/analysis/clustering/gene_expression_graphclust/clusters.csv").set_index("Barcode")
    obs = cells.reindex(bc)[["x_centroid", "y_centroid", "transcript_counts", "cell_area", "nucleus_area"]].copy()
    obs["cluster"] = cl.reindex(bc)["Cluster"].astype("Int64").astype(str).values
    print("cells x genes", X.shape, "clusters", obs.cluster.nunique(), flush=True)

    genes = set(pd.read_csv(GENE_LIST, header=None)[0].astype(str))  # 10,134 genes, no header row
    shared = [g for g in names if g in genes]
    tot = np.asarray(X.sum(1)).ravel()
    ok = tot >= 50
    X, obs = X[ok], obs[ok]
    Xs = X[:, [gi[g] for g in shared]]
    det = np.asarray((Xs > 0).sum(0)).ravel() / Xs.shape[0]
    stats = {"n_cells_ge50_transcripts": int(X.shape[0]), "panel_genes": int(len(names)),
             "shared_with_training_list": len(shared), "shared_detected_in_ge_5pct_cells": int((det >= 0.05).sum()),
             "median_transcripts_per_cell": float(np.median(tot[ok])),
             "median_shared_counts_per_cell": float(np.median(np.asarray(Xs.sum(1)).ravel())),
             "section_extent_um": {"x": [float(obs.x_centroid.min()), float(obs.x_centroid.max())],
                                   "y": [float(obs.y_centroid.min()), float(obs.y_centroid.max())]}}
    json.dump(stats, open(f"{a.out}/xenium_stats.json", "w"), indent=2); print(stats, flush=True)
    pd.Series(shared).to_csv(f"{a.out}/shared_genes.txt", index=False, header=False)

    # log1p CP100 (depth ~1k transcripts per cell) for marker means per cluster
    Xn = sp.diags(100.0 / tot[ok]) @ X
    Xn.data = np.log1p(Xn.data)
    present = {k: [g for g in v if g in gi] for k, v in MARKERS.items()}
    print("markers missing from panel:", {k: [g for g in v if g not in gi] for k, v in MARKERS.items() if any(g not in gi for g in v)}, flush=True)
    rows = []
    for c, idx in obs.groupby("cluster").indices.items():
        sub, so = Xn[idx], obs.iloc[idx]
        r = {"cluster": c, "n": len(idx), "x": float(so.x_centroid.median()), "y": float(so.y_centroid.median()),
             "x_sd": float(so.x_centroid.std()), "y_sd": float(so.y_centroid.std()),
             "transcripts": float(so.transcript_counts.median()), "nuc_frac_area": float((so.nucleus_area / so.cell_area).median())}
        for k, gs in present.items():
            r[k] = float(sub[:, [gi[g] for g in gs]].mean()) if gs else np.nan
        rows.append(r)
    tab = pd.DataFrame(rows).set_index("cluster")
    tab["top_marker_set"] = tab[list(present)].idxmax(1)
    tab = tab.sort_values("n", ascending=False)
    tab.round(3).to_csv(f"{a.out}/cluster_markers.csv")
    pd.set_option("display.width", 320); pd.set_option("display.max_columns", 50); pd.set_option("display.max_rows", 200)
    print(tab.round(2).to_string(), flush=True)

    reg = {g: float((X[:, gi[g]] > 0).mean()) for k in ["Striatum", "Hippocampus", "Thalamus", "Hypothal", "Olfactory", "L4/5 IT"]
           for g in present[k]}
    json.dump(reg, open(f"{a.out}/region_marker_detection.json", "w"), indent=2); print("region marker detection (frac cells > 0):", reg)
    obs.to_csv(f"{a.out}/cell_obs_with_cluster.csv.gz")


if __name__ == "__main__":
    main()
