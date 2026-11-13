"""
Ingest Ding 2020 (GSE132044) mouse cortex: NUCLEI ONLY (10x Chromium v2, DroNc-seq, sci-RNA-seq, Smart-seq2;
two experiments Cortex1/Cortex2 = two days, same tissue), plus the matched bulk RNA-seq of each experiment
(whole-tissue RNA, GSE132044_NIH3T3_cortex_TPM_bulk.tsv.gz). There are no matched whole cells in this dataset
(GEO series design), so it serves the nuclei-only application / chemistry-robustness test, with the bulk profile
as the only whole-cell reference.

Output: data/external/ding2020_GSE132044/processed/nuclei.h5ad in the training gene order (absent genes zero-filled,
var['present']); obs: experiment, method, lib_full (training genes), lib_downsampled. Smart-seq2 read counts
(~1 M per nucleus) are binomially thinned to libraries quantile-matched to the v3 training nuclei; UMI methods
are kept as they are (their depth is below the training depth; the model's log1p input range is still covered).
Bulk TPM is written to processed/bulk_tpm_training_genes.csv (Cortex1, Cortex2 columns).
"""
import os, sys, json, argparse
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad

ROOT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE"
sys.path.insert(0, ROOT)
from analysis.benchmark.data_split import load_partition
from analysis.external.ingest_bakken2018 import thin_to_library

D = os.path.join(ROOT, "data/external/ding2020_GSE132044")
GENE_LIST = os.path.join(ROOT, "data/training_gene_list_v3.csv")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--min_lib", type=int, default=500)
    args = ap.parse_args()
    out = os.path.join(D, "processed"); os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    genes = pd.read_csv(GENE_LIST, header=None)[0].astype(str).values
    M = sp.load_npz(os.path.join(D, "cortex_counts_genes_x_cells.npz")).tocsr()        # genes x cells
    cells = pd.read_csv(os.path.join(D, "GSE132044_cortex_mm10_cell.tsv.gz"), header=None)[0].astype(str)
    gsym = pd.read_csv(os.path.join(D, "GSE132044_cortex_mm10_gene.tsv.gz"), header=None)[0].astype(str).str.split("_", n=1).str[1]
    # duplicate symbols: keep the row with the largest total
    tot = np.asarray(M.sum(1)).ravel()
    order = np.argsort(-tot)
    first = {}
    for i in order:
        first.setdefault(gsym.iloc[i], i)
    idx = np.array([first.get(g, -1) for g in genes]); found = idx >= 0
    X = np.zeros((M.shape[1], len(genes)), dtype=np.int64)
    X[:, found] = M[idx[found]].toarray().T
    obs = pd.DataFrame({"experiment": cells.str.split(".").str[0].values, "method": cells.str.split(".").str[1].values}, index=cells.values)
    obs["lib_full_training_genes"] = X.sum(1)
    keep = obs["lib_full_training_genes"].values >= args.min_lib
    X, obs = X[keep], obs[keep].copy()
    tr_lib = np.sort(np.asarray(load_partition("nuc", "train", normalize="raw", mmap=True)[0]).sum(1).astype(float))
    ss2 = (obs["method"] == "Smart-seq2").values
    target = obs["lib_full_training_genes"].values.astype(float).copy()
    r = pd.Series(target[ss2]).rank(method="first").values
    target[ss2] = np.quantile(tr_lib, (r - 0.5) / ss2.sum())
    X_ds = X.copy(); X_ds[ss2] = thin_to_library(X[ss2], target[ss2], rng)
    obs["lib_downsampled"] = X_ds.sum(1); obs["downsampled"] = ss2
    a = ad.AnnData(X=X_ds.astype(np.float32), obs=obs, var=pd.DataFrame({"gene_names": genes, "present": found}, index=genes))
    a.layers["counts_full"] = X.astype(np.float32)
    a.uns["ingest"] = {"source": "GSE132044 cortex (nuclei only)", "min_lib": args.min_lib, "smartseq2_thinning": f"quantile-matched to v3 training nuclei (seed {args.seed})"}
    a.write_h5ad(os.path.join(out, "nuclei.h5ad"))
    bulk = pd.read_csv(os.path.join(D, "GSE132044_NIH3T3_cortex_TPM_bulk.tsv.gz"), sep="\t", index_col=0)
    bulk.index = bulk.index.str.split("_", n=1).str[1]
    bulk = bulk[["Cortex1", "Cortex2"]].groupby(level=0).sum().reindex(genes).fillna(0.0)
    bulk["present"] = found
    bulk.to_csv(os.path.join(out, "bulk_tpm_training_genes.csv"))
    summ = {"n_total": int(len(keep)), "n_kept": int(keep.sum()), "genes_present": int(found.sum()), "genes_zero_filled": int((~found).sum()),
            "per_group": obs.groupby(["method", "experiment"]).agg(n=("lib_full_training_genes", "size"), lib_full_median=("lib_full_training_genes", "median"),
                                                                   lib_ds_median=("lib_downsampled", "median")).reset_index().to_dict(orient="records")}
    json.dump(summ, open(os.path.join(out, "ingest_summary.json"), "w"), indent=2, default=float)
    print(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    main()
