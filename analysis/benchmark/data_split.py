"""
Shared data layer for the split-aware benchmark.

Design (see notes/benchmark_review_2026-10-09.md):

  * The train / valid / test partition is read from ``data/split_manifest.tsv``,
    which reproduces the exact ``balanced_split_by_cell_type`` split used to
    train the cVAE (stratified 60/20/20, random_state=42).  Every method in the
    benchmark therefore trains on the same observations the cVAE trained on and
    is scored on observations the cVAE never saw.  (The previous version of this
    module drew its *own* per-cell-type shuffle split, which did not match the
    cVAE split and let baseline "training" data overlap the cVAE's held-out set.)

  * The raw h5ad files are dense int64 (41 GB cells, 24 GB nuclei) and
    ``anndata.read_h5ad`` loads them whole.  That is why the August benchmark
    jobs were OOM-killed at 64 GB.  Here the matrices are read once with h5py in
    row blocks, restricted to the manifest rows and the Config-B gene columns,
    and cached as float32 raw counts under ``cache/benchmark/``.  All methods
    load from that cache (about 4.5 GB in total).

  * Methods are scored on the TEST partition by default.  The validation
    partition was used for epoch-level monitoring of the cVAE and for
    early-stopping of the baselines, so test is the only partition no model-fitting
    or model-monitoring step ever touched.  ``--eval_partition valid`` is kept as a
    secondary check.

Cache layout (``cache/benchmark/``):
    gene_names.csv                      one gene symbol per line, in model input order
    {cell,nuc}_{train,valid,test}.npy   float32 raw counts [n_obs, n_genes]
    {cell,nuc}_{train,valid,test}_obs.csv   obs_id, cell_type, donor, h5ad_row (same row order)
    cache_info.json                     provenance (manifest hash, counts, build date)
"""

import os
import sys
import json
import hashlib
import time
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi"
CELL_DIR = "10X_cells_v2_v3_AIBS"
NUC_DIR = "10X_nuclei_v2_v3_AIBS"
DEFAULT_GENE_LIST = os.path.join(PROJECT_ROOT, "data", "training_gene_list_v3.csv")
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "data", "split_manifest.tsv")
CACHE_DIR = os.environ.get("BENCH_CACHE_DIR", os.path.join(PROJECT_ROOT, "cache", "benchmark"))
SPLIT_SEED = 42
PARTITIONS = ("train", "valid", "test")
MODALITIES = {"cell": CELL_DIR, "nuc": NUC_DIR}
MANIFEST_MODALITY = {"cell": "cell", "nuc": "nuclei"}


# ---------------------------------------------------------------------------
# Gene list
# ---------------------------------------------------------------------------

def resolve_gene_order(var_gene_names, gene_list_path=DEFAULT_GENE_LIST):
    """Return (column_indices, sorted_gene_names) reproducing imputeVAE.data.load_raw_data:
    keep genes in common_genes.csv AND in the gene list, then sort by gene name."""
    common_genes = set(pd.read_csv(os.path.join(PROJECT_ROOT, "data", "common_genes.csv"),
                                   header=0)["common_genes"].tolist())
    gene_list = set(pd.read_csv(gene_list_path, header=None)[0].tolist())
    names = np.array([str(g) for g in var_gene_names], dtype=object)
    keep = np.array([(g in common_genes) and (g in gene_list) for g in names])
    idx = np.where(keep)[0]
    order = names[idx].argsort()
    idx = idx[order]
    return idx, names[idx]


# ---------------------------------------------------------------------------
# Cache building (h5py, row-blocked)
# ---------------------------------------------------------------------------

def _read_h5_strings(ds):
    vals = ds[:]
    if vals.dtype.kind in ("S", "O"):
        return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in vals], dtype=object)
    return vals.astype(str).astype(object)


def _read_categorical(group_or_ds):
    """Read an anndata obs column stored either as plain array or categorical group."""
    import h5py
    if isinstance(group_or_ds, h5py.Group):
        cats = _read_h5_strings(group_or_ds["categories"])
        codes = group_or_ds["codes"][:]
        return cats[codes]
    return _read_h5_strings(group_or_ds)


def build_cache(gene_list_path=DEFAULT_GENE_LIST, cache_dir=CACHE_DIR, block_rows=4096,
                manifest_path=MANIFEST_PATH):
    """Read both h5ad files block-wise and write the partitioned cache."""
    import h5py

    os.makedirs(cache_dir, exist_ok=True)
    manifest = pd.read_csv(manifest_path, sep="\t", dtype=str)
    assert set(manifest["partition"].unique()) == set(PARTITIONS), manifest["partition"].unique()
    with open(manifest_path, "rb") as fh:
        manifest_md5 = hashlib.md5(fh.read()).hexdigest()

    info = {
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "manifest": manifest_path,
        "manifest_md5": manifest_md5,
        "gene_list": gene_list_path,
        "split_seed": SPLIT_SEED,
        "split_by": "observation (not donor); stratified by cell_type; 60/20/20",
        "counts": {},
    }

    gene_names_ref = None
    for mod_key, data_dir in MODALITIES.items():
        path = os.path.join(DATA_ROOT, data_dir, "data.h5ad")
        print(f"\n[{mod_key}] opening {path}")
        with h5py.File(path, "r") as f:
            X = f["X"]
            n_obs_total, n_var_total = X.shape
            obs_index = _read_h5_strings(f["obs"]["_index"])
            var_names = _read_h5_strings(f["var"]["gene_names"])
            obs_ct = _read_categorical(f["obs"]["cell_type"])
            obs_batch = _read_categorical(f["obs"]["batch_id"])
            col_idx, gene_names = resolve_gene_order(var_names, gene_list_path)
            if gene_names_ref is None:
                gene_names_ref = gene_names
            else:
                assert np.array_equal(gene_names_ref, gene_names), "gene order differs between modalities"
            print(f"  X {X.shape} {X.dtype}; keeping {len(col_idx)} genes")

            row_of_id = {oid: i for i, oid in enumerate(obs_index)}
            sub = manifest[manifest["modality"] == MANIFEST_MODALITY[mod_key]]
            part_rows = {}
            for part in PARTITIONS:
                ids = sub.loc[sub["partition"] == part, "obs_id"].values
                rows = np.array(sorted(row_of_id[i] for i in ids), dtype=np.int64)
                part_rows[part] = rows
                # sanity: cell_type in manifest matches h5ad
                man_ct = sub.set_index("obs_id").loc[obs_index[rows], "cell_type"].values
                assert np.array_equal(man_ct.astype(str), obs_ct[rows].astype(str)), \
                    f"cell_type mismatch between manifest and h5ad for {mod_key}/{part}"
                print(f"  {part}: {len(rows)} rows")

            all_rows = np.concatenate([part_rows[p] for p in PARTITIONS])
            assert len(np.unique(all_rows)) == len(all_rows), "rows duplicated across partitions"
            needed = np.zeros(n_obs_total, dtype=bool)
            needed[all_rows] = True

            # Allocate outputs
            out = {p: np.empty((len(part_rows[p]), len(col_idx)), dtype=np.float32) for p in PARTITIONS}
            pos = {p: {r: k for k, r in enumerate(part_rows[p])} for p in PARTITIONS}
            row_part = np.full(n_obs_total, -1, dtype=np.int8)
            for pi, p in enumerate(PARTITIONS):
                row_part[part_rows[p]] = pi

            t0 = time.time()
            n_blocks = (n_obs_total + block_rows - 1) // block_rows
            for b in range(n_blocks):
                s, e = b * block_rows, min((b + 1) * block_rows, n_obs_total)
                if not needed[s:e].any():
                    continue
                block = X[s:e, :]                       # contiguous read
                local = np.where(needed[s:e])[0]
                sel = block[local][:, col_idx].astype(np.float32)
                for k, li in enumerate(local):
                    r = s + li
                    p = PARTITIONS[row_part[r]]
                    out[p][pos[p][r]] = sel[k]
                if b % 10 == 0:
                    print(f"  block {b+1}/{n_blocks}  ({time.time()-t0:.0f}s)", flush=True)
            print(f"  read done in {time.time()-t0:.0f}s")

            donors = np.array([str(x).split(".")[0].split("-")[0] for x in obs_batch], dtype=object)
            for p in PARTITIONS:
                rows = part_rows[p]
                np.save(os.path.join(cache_dir, f"{mod_key}_{p}.npy"), out[p])
                obs = pd.DataFrame({
                    "obs_id": obs_index[rows].astype(str),
                    "cell_type": obs_ct[rows].astype(str),
                    "donor": donors[rows].astype(str),
                    "h5ad_row": rows,
                })
                obs.to_csv(os.path.join(cache_dir, f"{mod_key}_{p}_obs.csv"), index=False)
                info["counts"][f"{mod_key}_{p}"] = int(len(rows))
                lib = out[p].sum(axis=1)
                print(f"  {p}: saved {out[p].shape}; median library {np.median(lib):.0f}; "
                      f"zero fraction {(out[p]==0).mean():.3f}")

    pd.Series(gene_names_ref).to_csv(os.path.join(cache_dir, "gene_names.csv"), index=False, header=False)
    info["n_genes"] = int(len(gene_names_ref))
    with open(os.path.join(cache_dir, "cache_info.json"), "w") as fh:
        json.dump(info, fh, indent=2)
    print(f"\nCache written to {cache_dir}")
    return info


# ---------------------------------------------------------------------------
# Loading from cache
# ---------------------------------------------------------------------------

def cache_exists(cache_dir=CACHE_DIR):
    need = ["gene_names.csv", "cache_info.json"] + \
           [f"{m}_{p}.npy" for m in MODALITIES for p in PARTITIONS] + \
           [f"{m}_{p}_obs.csv" for m in MODALITIES for p in PARTITIONS]
    return all(os.path.exists(os.path.join(cache_dir, f)) for f in need)


def load_partition(modality, partition, normalize="log1p", cache_dir=CACHE_DIR, mmap=False):
    """Return (X, obs) for one modality/partition.  normalize in {"raw", "log1p"}."""
    X = np.load(os.path.join(cache_dir, f"{modality}_{partition}.npy"),
                mmap_mode="r" if mmap else None)
    obs = pd.read_csv(os.path.join(cache_dir, f"{modality}_{partition}_obs.csv"), dtype={"obs_id": str})
    if normalize == "log1p":
        X = np.log1p(np.asarray(X, dtype=np.float32))
    elif normalize == "raw":
        X = np.asarray(X, dtype=np.float32)
    else:
        raise ValueError(normalize)
    return X, obs


def load_gene_names(cache_dir=CACHE_DIR):
    return pd.read_csv(os.path.join(cache_dir, "gene_names.csv"), header=None)[0].astype(str).values


def load_benchmark_data(eval_partition="test", return_raw_counts=False, cache_dir=CACHE_DIR,
                        include_eval_cells=True):
    """Load training data plus the evaluation partition.

    Returns dict with AnnData objects:
        cell_train, nuc_train          training partition (what every method may fit on)
        nuc_eval                       nuclei to translate and score
        cell_eval                      real cells of the eval partition (scoring reference only)
        gene_names, common_cts, split_info
    .X holds raw counts if return_raw_counts else log1p(raw counts).
    """
    import anndata as ad
    if not cache_exists(cache_dir):
        raise FileNotFoundError(
            f"Benchmark cache missing in {cache_dir}. Run: python -m analysis.benchmark.build_cache")
    assert eval_partition in ("valid", "test"), eval_partition
    norm = "raw" if return_raw_counts else "log1p"
    gene_names = load_gene_names(cache_dir)
    with open(os.path.join(cache_dir, "cache_info.json")) as fh:
        info = json.load(fh)

    def _adata(modality, partition):
        X, obs = load_partition(modality, partition, normalize=norm, cache_dir=cache_dir)
        obs.index = obs["obs_id"].astype(str).values
        a = ad.AnnData(X=X, obs=obs[["cell_type", "donor", "obs_id"]].copy())
        a.var_names = gene_names
        a.var["gene_names"] = gene_names
        return a

    result = {
        "cell_train": _adata("cell", "train"),
        "nuc_train": _adata("nuc", "train"),
        "nuc_eval": _adata("nuc", eval_partition),
        "gene_names": gene_names,
    }
    if include_eval_cells:
        result["cell_eval"] = _adata("cell", eval_partition)
    ct_eval_nuc = set(result["nuc_eval"].obs["cell_type"])
    ct_eval_cell = set(pd.read_csv(os.path.join(cache_dir, f"cell_{eval_partition}_obs.csv"))["cell_type"])
    common_cts = sorted(ct_eval_nuc & ct_eval_cell)
    result["common_cts"] = common_cts
    result["split_info"] = {
        "seed": SPLIT_SEED,
        "split_fractions": "60/20/20",
        "split_by": "observation (not donor), stratified by cell_type; identical to cVAE training split",
        "manifest": MANIFEST_PATH,
        "manifest_md5": info.get("manifest_md5"),
        "eval_partition": eval_partition,
        "gene_list": info.get("gene_list"),
        "n_genes": int(len(gene_names)),
        "n_common_cts": len(common_cts),
        "common_cts": common_cts,
        "cell_train_n": int(result["cell_train"].n_obs),
        "nuc_train_n": int(result["nuc_train"].n_obs),
        "nuc_eval_n": int(result["nuc_eval"].n_obs),
        "cell_eval_n": int(info["counts"][f"cell_{eval_partition}"]),
    }
    print(f"Train: {result['cell_train'].n_obs} cells, {result['nuc_train'].n_obs} nuclei | "
          f"eval ({eval_partition}): {result['nuc_eval'].n_obs} nuclei, "
          f"{result['split_info']['cell_eval_n']} cells | {len(common_cts)} common cell types")
    return result


def save_standard_outputs(output_dir, imputed_eval, nuc_eval_obs, gene_names, args_dict, split_info):
    """Save a method's prediction in the format expected by evaluate_all.py.

    imputed_eval must be log1p-scale expression [n_nuc_eval, n_genes] in the cache row order.
    """
    os.makedirs(output_dir, exist_ok=True)
    imputed_eval = np.asarray(imputed_eval, dtype=np.float32)
    assert imputed_eval.shape == (len(nuc_eval_obs), len(gene_names)), \
        f"imputed shape {imputed_eval.shape} != ({len(nuc_eval_obs)}, {len(gene_names)})"
    assert np.isfinite(imputed_eval).all(), "non-finite values in imputed output"
    np.save(os.path.join(output_dir, "imputed_eval.npy"), imputed_eval)
    nuc_eval_obs.to_csv(os.path.join(output_dir, "nuc_eval_obs.csv"), index=False)
    pd.Series(gene_names).to_csv(os.path.join(output_dir, "gene_names.csv"), index=False, header=False)
    with open(os.path.join(output_dir, "arguments.json"), "w") as f:
        json.dump(args_dict, f, indent=2, default=str)
    with open(os.path.join(output_dir, "split_info.json"), "w") as f:
        json.dump(split_info, f, indent=2, default=str)
    print(f"Outputs saved to {output_dir}/  (imputed {imputed_eval.shape}, "
          f"min {imputed_eval.min():.3f}, max {imputed_eval.max():.3f}, mean {imputed_eval.mean():.3f})")


def ct_median_library(X_raw, ct_labels, common_cts, fallback=None):
    """Median raw library size per cell type (used to put count-model decoders on the cell scale)."""
    lib = np.asarray(X_raw).sum(axis=1)
    out = {}
    for ct in common_cts:
        m = ct_labels == ct
        if m.sum() > 0:
            out[ct] = float(np.median(lib[m]))
    if fallback is None:
        fallback = float(np.median(lib))
    out["_global"] = fallback
    return out
