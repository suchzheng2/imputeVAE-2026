"""
Ingest Bakken 2018 (GSE123454): 463 nuclei + 463 matched cells, mouse VISp, SMART-seq v4,
11 matched clusters (same cluster ids for nuclei and cells). Produces model-ready AnnData files
in the training gene order (data/training_gene_list_v3.csv; genes absent from the 2018 annotation
are zero-filled and flagged in var['present']).

Depth: SMART-seq libraries are ~1-2 M reads over the training genes vs 15 k (nuclei) / 54 k (cells)
UMIs in the 10x training data, and the model takes log1p(raw counts). Each nucleus/cell is therefore
binomially thinned to a library size quantile-matched to the training partition of the same modality
(nuclei -> training nuclei, cells -> training cells; benchmark cache). The un-thinned counts are kept
in layers['counts_full'].

Usage:
    python analysis/external/ingest_bakken2018.py [--counts exon_intron|exon] [--seed 0]
Outputs: data/external/bakken2018_GSE123454/processed/{nuclei,cells}_<counts>.h5ad + ingest_summary_<counts>.json
"""
import os, sys, json, argparse, re
import numpy as np
import pandas as pd
import anndata as ad

ROOT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE"
sys.path.insert(0, ROOT)
from analysis.benchmark.data_split import load_partition

D = os.path.join(ROOT, "data/external/bakken2018_GSE123454")
GENE_LIST = os.path.join(ROOT, "data/training_gene_list_v3.csv")


def norm_key(s):
    """Sample names in the count matrix and the GEO sheet differ by the sequencing-run suffix (_E1 vs _E2)
    and 42 cells are listed as 'A_and_B'. Normalise to the plate/well part."""
    s = s.split("_and_")[0]
    return re.sub(r"_E[12](_|$)", r"_", s).rstrip("_")


def training_library(modality):
    X = load_partition(modality, "train", normalize="raw", mmap=True)[0]
    return np.asarray(X).sum(axis=1).astype(np.float64)


def thin_to_library(counts, target_lib, rng):
    """Binomial thinning of each row to (approximately) target_lib total; rows already below target are kept."""
    out = np.zeros_like(counts, dtype=np.int64)
    lib = counts.sum(axis=1)
    for i in range(counts.shape[0]):
        if lib[i] <= target_lib[i] or lib[i] == 0:
            out[i] = counts[i]
        else:
            out[i] = rng.binomial(counts[i].astype(np.int64), target_lib[i] / lib[i])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", default="exon_intron", choices=["exon_intron", "exon"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out_dir = os.path.join(D, "processed")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    genes = pd.read_csv(GENE_LIST, header=None)[0].astype(str).values
    meta = pd.read_csv(os.path.join(D, "samples_metadata.csv"))
    meta = meta[meta["molecule"].isin(["Nuclei", "Cells"])].copy()
    meta["key"] = meta["Sample name"].map(norm_key)
    assert meta["key"].is_unique, "normalised metadata keys not unique"
    meta = meta.set_index("key")
    summary = {"counts": args.counts, "seed": args.seed, "n_training_genes": int(len(genes))}

    for mol, tag, train_mod in [("Nuclei", "nuclei", "nuc"), ("Cells", "cells", "cell")]:
        ex = pd.read_csv(os.path.join(D, f"GSE123454_{tag}_exon_counts.csv.gz"), index_col=0)
        if args.counts == "exon_intron":
            it = pd.read_csv(os.path.join(D, f"GSE123454_{tag}_intron_counts.csv.gz"), index_col=0)
            assert list(it.columns) == list(ex.columns) and list(it.index) == list(ex.index)
            M = ex + it
        else:
            M = ex
        keys = [norm_key(c) for c in M.columns]
        assert len(set(keys)) == len(keys)
        m = meta.loc[[k for k in keys if k in meta.index]]
        m = m[m["molecule"] == mol]
        keep = [c for c, k in zip(M.columns, keys) if k in m.index]
        unmatched = len(M.columns) - len(keep)
        M = M[keep]
        m = m.loc[[norm_key(c) for c in keep]]
        qc = (m["characteristics: sequencing_qc_pass_fail"] == "Pass").values
        M = M.loc[:, qc]
        m = m[qc]
        # gene space
        present = pd.Index(M.index)
        idx = present.get_indexer(genes)
        found = idx >= 0
        X_full = np.zeros((M.shape[1], len(genes)), dtype=np.int64)
        X_full[:, found] = M.values[idx[found]].T
        lib_full = X_full.sum(axis=1).astype(np.float64)
        # depth matching: quantile-match to training libraries of the same modality
        tr_lib = np.sort(training_library(train_mod))
        ranks = pd.Series(lib_full).rank(method="first").values  # 1..n
        q = (ranks - 0.5) / len(ranks)
        target = np.quantile(tr_lib, q)
        X_ds = thin_to_library(X_full, target, rng)
        lib_ds = X_ds.sum(axis=1)
        cl = m["characteristics: cell_cluster"].astype(str)
        obs = pd.DataFrame({
            "sample": M.columns.values,
            "cluster": cl.values,
            "cluster_id": cl.str.extract(r"^(cl\d+)_")[0].values,
            "cluster_label": cl.str.replace(r"_(Nuc|Cel)$", "", regex=True).str.split("_", n=2).str[2].values,
            "cell_class": m["characteristics: cell_class"].values,
            "donor": m["characteristics: donor_id"].values,
            "sex": m["characteristics: donor_sex"].values,
            "genotype": m["characteristics: donor_genotype"].values,
            "layer": m["characteristics: dissected_layer"].values,
            "gating": m["characteristics: facs_gating"].values,
            "lib_full_training_genes": lib_full,
            "lib_downsampled": lib_ds,
            "target_lib": target,
        }, index=M.columns.values)
        var = pd.DataFrame({"gene_names": genes, "present": found}, index=genes)
        a = ad.AnnData(X=X_ds.astype(np.float32), obs=obs, var=var)
        a.layers["counts_full"] = X_full.astype(np.float32)
        a.uns["ingest"] = {"source": "GSE123454", "counts": args.counts, "downsampling": f"binomial thinning quantile-matched to v3 training {train_mod} libraries (seed {args.seed})",
                           "genes_present": int(found.sum()), "genes_zero_filled": int((~found).sum())}
        path = os.path.join(out_dir, f"{tag}_{args.counts}.h5ad")
        a.write_h5ad(path)
        summary[tag] = {
            "n_matrix_columns": int(len(keys)), "n_unmatched_to_metadata": int(unmatched), "n_qc_fail_dropped": int((~qc).sum()),
            "n_kept": int(a.n_obs), "genes_present": int(found.sum()), "genes_zero_filled": int((~found).sum()),
            "lib_full_median": float(np.median(lib_full)), "lib_downsampled_median": float(np.median(lib_ds)),
            "training_lib_median": float(np.median(tr_lib)),
            "per_cluster": obs.groupby("cluster_id").size().to_dict(),
            "cluster_labels": obs.drop_duplicates("cluster_id").set_index("cluster_id")["cluster"].to_dict(),
        }
        print(f"{tag}: kept {a.n_obs} (unmatched {unmatched}, QC fail {(~qc).sum()}); genes present {found.sum()}; "
              f"lib median full {np.median(lib_full):.0f} -> downsampled {np.median(lib_ds):.0f} (training {np.median(tr_lib):.0f}) -> {path}")
    with open(os.path.join(out_dir, f"ingest_summary_{args.counts}.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
