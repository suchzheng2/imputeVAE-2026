"""
Cross-decode the eval-partition nuclei through a trained ImputeVAE (conditional VAE) run.

The run was trained with balanced_split_by_cell_type(random_state=42) on the same
train partition the baselines use (see data/split_manifest.tsv), so the eval
partition here is held out from the cVAE as well.

Usage:
    python -m analysis.benchmark.predict_cvae \
        --run_dir run/run_2026-07-18-19-40-51_570984 \
        --eval_partition test --output_dir run/benchmark/test/cvae

Library-size covariate runs (--libsize_cond True, 2026-10-16) take --target_libsize:
    ct_median_cell  decode every nucleus of a type at the median TRAINING-cell library of that
                    type (uses the eval nucleus's cell-type label; within-type output variation
                    is then latent-driven only).  Default for libsize_cond models.
    quantile_map    map the nucleus's library quantile within its type (vs training nuclei) onto
                    the training-cell library distribution of that type (keeps per-cell size
                    variation on the cell scale; the imputation setting).
    source          the nucleus's own library (ablation; reproduces the no-covariate behaviour).
  Label-free scale modes (2026-10-23; no cell-type label at inference, the nuclei-only setting):
    global_quantile_map  nucleus's library quantile within ALL training nuclei mapped onto ALL
                         training cells (one pooled map per chemistry).
    global_shift         nucleus's own library + (median training-cell - median training-nucleus
                         library): a single cell/nucleus RNA-content ratio per chemistry.
    knn_cell_median      median library of the K=50 nearest TRAINING CELLS in the aligned latent
                         space (the model's own identity code supplies the expected whole-cell RNA
                         content; no label anywhere).
    knn_quantile_map     the nucleus's library quantile among its K nearest training nuclei mapped
                         onto its K nearest training cells (label-free analogue of quantile_map).
    knn_type_median      majority-vote cell type of the K=50 nearest TRAINING NUCLEI in latent space
                         (training labels only, none at inference) -> that type's median training-cell
                         library (= ct_median_cell with a predicted label).
    knn_type_quantile_map  same predicted type, then the per-type quantile map.
Per-type statistics come from <run_dir>/libsize_stats.json written at training time; the pooled
statistics are computed from the training partition of the benchmark cache (same partition the
model was trained on).

Library (batch) covariate runs (--library_cond True, 2026-11-06) take --target_library:
    unknown    (default) decode every nucleus as the UNKNOWN / marginal library (embedding row 0,
               trained by dropout) -- the library-agnostic "average cell library" output.
    mean_cell  decode at the mean embedding of all training CELL libraries.
    <name>     decode as one specific training library, e.g. "4-c3A" (a female library) -- lets
               the output be placed in a named batch (sex, run).
  The source library of each eval nucleus (its own meta_batch_id, via donor + modality) is always
  given to the encoder; nuclei from libraries not in <run_dir>/library_vocab.json get UNKNOWN.
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from analysis.benchmark.data_split import load_benchmark_data, save_standard_outputs
from imputeVAE.models_cond import VAE_JSD_Cond


def load_model(run_dir, device):
    with open(os.path.join(run_dir, "arguments.json")) as f:
        config = json.load(f)
    num_internal = config["num_internal"]
    if isinstance(num_internal, str):
        num_internal = [int(x) for x in num_internal.split(",")]
    model = VAE_JSD_Cond(
        input_size=config["num_feature"], num_internal=num_internal,
        num_latent=config["num_latent"], embed_dim=config.get("embed_dim", 8),
        scalar_cond=config.get("scalar_cond", False), cond_encoder=config.get("cond_encoder", False),
        libsize_cond=config.get("libsize_cond", False),
        library_cond=config.get("library_cond", False), library_capacity=config.get("library_capacity", 32),
        library_dim=config.get("library_dim", 4),
    )
    model_path = os.path.join(run_dir, "models", "model.pt")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval().to(device)
    return model, config, model_path


def load_library_vocab(run_dir):
    """<run_dir>/library_vocab.json written by cond_training.build_library_vocab (library_cond runs)."""
    with open(os.path.join(run_dir, "library_vocab.json")) as f:
        return json.load(f)


def library_ids_from_obs(vocab, obs, modality):
    """Vocabulary ids [N] for benchmark-cache obs (columns cell_type, donor) of one modality.

    The cache keeps the donor (= library prefix of meta_batch_id, e.g. "4" of "4-c3A"); the vocabulary
    maps (modality, donor) back to the id. Unseen libraries -> 0 (UNKNOWN).
    """
    lookup = {(vocab["modality"][n], vocab["donor"][n]): i for n, i in vocab["ids"].items()}
    donors = obs["donor"].astype(str).to_numpy()
    ids = np.array([lookup.get((modality, d), 0) for d in donors], dtype=np.int64)
    n_unseen = int((ids == 0).sum())
    if n_unseen:
        print(f"  library ids ({modality}): {n_unseen} of {len(ids)} observations from libraries outside the "
              f"training vocabulary -> UNKNOWN")
    return ids


def log_library_size_np(X):
    """log1p(total counts) per row from a log1p-normalised matrix (same definition as the model)."""
    return np.log1p(np.expm1(X.astype(np.float64)).sum(axis=1))


LABEL_FREE_MODES = ("source", "global_quantile_map", "global_shift", "knn_cell_median", "knn_quantile_map",
                    "knn_type_median", "knn_type_quantile_map")
KNN_K = 50


def encode_all(model, X, modality_id, device, batch_size=2048, library_id=None):
    """Posterior means [N, L] for a log1p matrix through the model's encoder (no grad).

    library_id: optional [N] vocabulary ids (library_cond models; the sample's own library)."""
    out = []
    with torch.no_grad():
        for s in range(0, X.shape[0], batch_size):
            b = torch.from_numpy(np.ascontiguousarray(X[s:s + batch_size], dtype=np.float32)).to(device)
            kw = {} if library_id is None else {"source_library_id": torch.from_numpy(library_id[s:s + batch_size]).to(device)}
            out.append(model.encode_mu(b, modality_id, **kw).cpu().numpy())
    return np.concatenate(out)


def knn_library_targets(mode, model, device, X, X_nuc_train, X_cell_train, k=KNN_K, libs=None):
    """Label-free decode libraries from the aligned latent: K nearest training cells (and nuclei).

    libs: optional {"eval", "nuc", "cell": [N] vocabulary ids} for library_cond models."""
    from sklearn.neighbors import NearestNeighbors
    libs = libs or {}
    t0 = time.time()
    mu_eval = encode_all(model, X, 1, device, library_id=libs.get("eval"))
    mu_cell = encode_all(model, X_cell_train, 0, device, library_id=libs.get("cell"))
    lib_cell = log_library_size_np(X_cell_train)
    nn_cell = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(mu_cell)
    idx_c = nn_cell.kneighbors(mu_eval, return_distance=False)
    cell_nb = lib_cell[idx_c]                                   # [N, k]
    info = {"knn_k": int(k), "latent_dim": int(mu_eval.shape[1]), "uses_cell_type_label": False,
            "libsize_stats": "K nearest training cells in latent space (benchmark cache)"}
    if mode == "knn_cell_median":
        target = np.median(cell_nb, axis=1)
    else:
        mu_nuc = encode_all(model, X_nuc_train, 1, device, library_id=libs.get("nuc"))
        lib_nuc = log_library_size_np(X_nuc_train)
        idx_n = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(mu_nuc).kneighbors(mu_eval, return_distance=False)
        nuc_nb = np.sort(lib_nuc[idx_n], axis=1)
        cell_nb = np.sort(cell_nb, axis=1)
        src = log_library_size_np(X)
        q = (nuc_nb < src[:, None]).mean(axis=1)                 # quantile among the K nearest training nuclei
        grid = (np.arange(k) + 0.5) / k
        target = np.array([np.interp(qi, grid, row) for qi, row in zip(q, cell_nb)])
    info["knn_time_s"] = round(time.time() - t0, 1)
    return target.astype(np.float32), info


def knn_predicted_types(model, device, X, X_nuc_train, ct_nuc_train, k=KNN_K, libs=None):
    """Majority-vote cell type over the K nearest training nuclei in the aligned latent (no inference label)."""
    from sklearn.neighbors import NearestNeighbors
    libs = libs or {}
    t0 = time.time()
    mu_eval = encode_all(model, X, 1, device, library_id=libs.get("eval"))
    mu_nuc = encode_all(model, X_nuc_train, 1, device, library_id=libs.get("nuc"))
    idx = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(mu_nuc).kneighbors(mu_eval, return_distance=False)
    ct_nuc_train = np.asarray(ct_nuc_train).astype(str)
    nb = ct_nuc_train[idx]
    pred = np.array([max(set(row), key=list(row).count) for row in nb])
    agree = (nb == pred[:, None]).mean(axis=1)
    return pred, {"knn_k": int(k), "latent_dim": int(mu_eval.shape[1]), "uses_cell_type_label": False,
                  "labels_used": "training nuclei only (kNN majority vote)",
                  "knn_vote_agreement_mean": float(agree.mean()), "knn_time_s": round(time.time() - t0, 1)}


def build_target_log_libsize(mode, run_dir, X, cell_types, X_nuc_train=None, X_cell_train=None,
                             model=None, device=None, ct_nuc_train=None, libs=None):
    """Per-nucleus log1p target library for the covariate model, or None for 'source'.

    X, X_nuc_train, X_cell_train are log1p-normalised matrices; the global and knn modes need the
    two training matrices and ignore cell_types; the knn modes also need the model.
    """
    if mode == "source":
        return None, {}
    if mode in ("knn_type_median", "knn_type_quantile_map"):
        assert model is not None and X_nuc_train is not None and ct_nuc_train is not None
        pred, pinfo = knn_predicted_types(model, device, X, X_nuc_train, ct_nuc_train, libs=libs)
        if cell_types is not None:
            pinfo["predicted_type_accuracy_vs_eval_label"] = float((pred == np.asarray(cell_types).astype(str)).mean())
        target, info = build_target_log_libsize(
            "ct_median_cell" if mode == "knn_type_median" else "quantile_map", run_dir, X, pred)
        info.update(pinfo)
        return target, info
    if mode in ("knn_cell_median", "knn_quantile_map"):
        assert model is not None and X_nuc_train is not None and X_cell_train is not None
        target, info = knn_library_targets(mode, model, device, X, X_nuc_train, X_cell_train, libs=libs)
        info.update({"target_log_libsize_mean": float(target.mean()), "target_log_libsize_std": float(target.std())})
        return target, info
    if mode in ("global_quantile_map", "global_shift"):
        assert X_nuc_train is not None and X_cell_train is not None, "global modes need the training partition"
        src = log_library_size_np(X)
        lib_nuc = log_library_size_np(X_nuc_train)
        lib_cell = log_library_size_np(X_cell_train)
        if mode == "global_shift":
            shift = float(np.median(lib_cell) - np.median(lib_nuc))
            target = src + shift
            extra = {"global_log_shift": shift, "global_ratio_cell_over_nucleus": float(np.exp(shift))}
        else:
            grid = np.linspace(0, 1, 101)
            q_nuc, q_cell = np.quantile(lib_nuc, grid), np.quantile(lib_cell, grid)
            q = np.interp(src, q_nuc, grid)
            target = np.interp(q, grid, q_cell)
            extra = {"global_median_cell": float(np.median(lib_cell)), "global_median_nuclei": float(np.median(lib_nuc))}
        info = {"libsize_stats": "pooled training partition (benchmark cache)", "uses_cell_type_label": False,
                "n_train_cell": int(len(lib_cell)), "n_train_nuclei": int(len(lib_nuc)),
                "target_log_libsize_mean": float(target.mean()), "target_log_libsize_std": float(target.std()), **extra}
        return target.astype(np.float32), info
    stats_path = os.path.join(run_dir, "libsize_stats.json")
    with open(stats_path) as f:
        stats = json.load(f)
    ct_median_cell = stats["ct_median_cell"]
    fallback = float(np.median(list(ct_median_cell.values())))
    cell_types = np.asarray(cell_types).astype(str)
    missing = sorted(set(cell_types) - set(ct_median_cell))
    if missing:
        print(f"WARNING: {len(missing)} eval cell types without training-cell library stats "
              f"(fallback = median of type medians {fallback:.3f}): {missing}")
    target = np.full(X.shape[0], fallback, dtype=np.float64)
    if mode == "ct_median_cell":
        for ct in np.unique(cell_types):
            if ct in ct_median_cell:
                target[cell_types == ct] = ct_median_cell[ct]
    elif mode == "quantile_map":
        grid = np.asarray(stats["quantile_grid"], dtype=np.float64)
        q_nuc, q_cell = stats["ct_quantiles_nuclei"], stats["ct_quantiles_cell"]
        src = log_library_size_np(X)
        for ct in np.unique(cell_types):
            m = cell_types == ct
            if ct in q_nuc and ct in q_cell:
                q = np.interp(src[m], np.asarray(q_nuc[ct]), grid)       # quantile within training nuclei of this type
                target[m] = np.interp(q, grid, np.asarray(q_cell[ct]))    # same quantile of training cells of this type
            elif ct in ct_median_cell:
                target[m] = ct_median_cell[ct]
    else:
        raise ValueError(mode)
    info = {"libsize_stats": stats_path, "libsize_mean": stats["mean"], "libsize_std": stats["std"],
            "uses_cell_type_label": True,
            "target_log_libsize_mean": float(target.mean()), "target_log_libsize_std": float(target.std())}
    return target.astype(np.float32), info


def apply_run_exclusions(data, config):
    """Mirror train.py --exclude_cell_types / --holdout_donors on the benchmark cache's TRAINING partition.

    The ablation runs (2026-11-04) drop a cell type or a nuclei donor from train/valid only, so the cache's
    train partition still holds those observations. Without this filter the label-free decodes would use
    them as kNN reference (the type vote named the held-out type for 95% of its test nuclei), which is a
    leak for an "unseen type / unseen animal" claim. Eval-partition data are never touched.
    """
    ex_ct = [c.strip() for c in (config.get("exclude_cell_types") or "").split(",") if c.strip()]
    ex_donor = [d.strip() for d in (config.get("holdout_donors") or "").split(",") if d.strip()]
    info = {}
    for key, modality in (("nuc_train", "nuclei"), ("cell_train", "cell")):
        a = data[key]
        keep = np.ones(a.n_obs, dtype=bool)
        if ex_ct:
            keep &= ~a.obs["cell_type"].astype(str).isin(ex_ct).to_numpy()
        mine = [d.split(":", 1)[1].strip() for d in ex_donor if d.split(":", 1)[0].strip() == modality]
        if mine:
            keep &= ~a.obs["donor"].astype(str).isin(mine).to_numpy()
        if not keep.all():
            data[key] = a[keep].copy()
            info[f"{key}_excluded"] = int((~keep).sum())
            print(f"[run exclusions] {key}: removed {int((~keep).sum())} of {a.n_obs} "
                  f"(exclude_cell_types={ex_ct or None}, holdout_donors={ex_donor or None}); {data[key].n_obs} remain")
    if info:
        info.update({"train_exclusions": {"exclude_cell_types": ex_ct or None, "holdout_donors": ex_donor or None},
                     "knn_reference": "benchmark-cache train partition minus the run's exclusions"})
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", default="run/run_2026-07-18-19-40-51_570984")
    p.add_argument("--eval_partition", default="test", choices=["valid", "test"])
    p.add_argument("--output_dir", default="run/benchmark/test/cvae")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--target_libsize", default="auto",
                   choices=["auto", "ct_median_cell", "quantile_map", "source", "global_quantile_map", "global_shift",
                            "knn_cell_median", "knn_quantile_map", "knn_type_median", "knn_type_quantile_map"],
                   help="libsize_cond models only; auto = ct_median_cell when the model has the covariate; "
                        "global_* and knn_* modes use no cell-type label")
    p.add_argument("--target_library", default="unknown",
                   help="library_cond models only: library the output is decoded as -- 'unknown' (marginal, "
                        "default), 'mean_cell' (mean embedding of the training cell libraries) or a library name "
                        "from <run_dir>/library_vocab.json such as '4-c3A'")
    p.add_argument("--training_job_id", default="7714771")
    p.add_argument("--training_time_s", type=float, default=44102.0,
                   help="wall time of the training job (sacct 7714771: 12:15:02)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, model_path = load_model(args.run_dir, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model {model_path} -> {device}; {n_params:,} parameters; libsize_cond={model.libsize_cond}")
    assert config.get("raw_data_normalization", "log1p") == "log1p", "benchmark assumes log1p inputs"
    assert config.get("gene_list", "").endswith("training_gene_list_v3.csv"), config.get("gene_list")

    data = load_benchmark_data(args.eval_partition, return_raw_counts=False, include_eval_cells=False)
    X = data["nuc_eval"].X.astype(np.float32)
    assert X.shape[1] == config["num_feature"], (X.shape, config["num_feature"])
    exclusion_info = apply_run_exclusions(data, config)

    # Library (batch) covariate: source library of every observation, and the decode-target library
    libs, library_info, target_lib_kw = None, {}, {}
    if model.library_cond:
        vocab = load_library_vocab(args.run_dir)
        libs = {"eval": library_ids_from_obs(vocab, data["nuc_eval"].obs, "nuclei"),
                "nuc": library_ids_from_obs(vocab, data["nuc_train"].obs, "nuclei"),
                "cell": library_ids_from_obs(vocab, data["cell_train"].obs, "cell")}
        cell_lib_ids = [i for n, i in vocab["ids"].items() if vocab["modality"][n] == "cell"]
        if args.target_library == "unknown":
            target_lib_kw = {}
        elif args.target_library == "mean_cell":
            target_lib_kw = {"target_library_emb": model.mean_library_emb(cell_lib_ids)}
        elif args.target_library in vocab["ids"]:
            target_lib_kw = {"target_library_id": int(vocab["ids"][args.target_library])}
        else:
            raise SystemExit(f"--target_library {args.target_library!r} is not 'unknown', 'mean_cell' or one of "
                             f"{vocab['names']}")
        library_info = {"library_cond": True, "target_library": args.target_library,
                        "n_libraries": len(vocab["names"]), "library_dim": int(model.library_dim),
                        "library_dropout_train": config.get("library_dropout"),
                        "eval_nuclei_unknown_library": int((libs["eval"] == 0).sum()),
                        "source_library": "each observation's own meta_batch_id via (modality, donor); unseen -> UNKNOWN"}
        print(f"library covariate: {len(vocab['names'])} training libraries; decode target = {args.target_library}")
    elif args.target_library != "unknown":
        raise SystemExit(f"--target_library needs a library_cond model; {args.run_dir} has none")

    target_mode = args.target_libsize
    if not model.libsize_cond:
        if target_mode not in ("auto", "source"):
            raise SystemExit(f"--target_libsize {target_mode} needs a libsize_cond model; {args.run_dir} has none")
        target_mode = "n/a (no covariate)"
        target, target_info = None, {}
    else:
        if target_mode == "auto":
            target_mode = "ct_median_cell"
        target, target_info = build_target_log_libsize(
            target_mode, args.run_dir, X, data["nuc_eval"].obs["cell_type"].values,
            X_nuc_train=data["nuc_train"].X, X_cell_train=data["cell_train"].X, model=model, device=device,
            ct_nuc_train=data["nuc_train"].obs["cell_type"].values, libs=libs)
        print(f"libsize target = {target_mode}; model mean/std = {float(model.libsize_mean):.4f}/{float(model.libsize_std):.4f}")

    t0 = time.time()
    imputed = np.zeros_like(X)
    with torch.no_grad():
        for s in range(0, X.shape[0], args.batch_size):
            e = min(s + args.batch_size, X.shape[0])
            batch = torch.from_numpy(X[s:e]).to(device)
            kw = {}
            if target is not None:
                kw["target_log_libsize"] = torch.from_numpy(target[s:e]).to(device)
            if libs is not None:
                kw["source_library_id"] = torch.from_numpy(libs["eval"][s:e]).to(device)
                kw.update(target_lib_kw)
            imputed[s:e] = model.decode_cross(batch, source_modality_id=1, target_modality_id=0, **kw).cpu().numpy()
    print(f"Cross-decoded {X.shape[0]} nuclei in {time.time()-t0:.1f}s")

    args_dict = {
        "method": "ImputeVAE", "run_dir": args.run_dir, "checkpoint": model_path,
        "training_job_id": args.training_job_id, "training_time_s": args.training_time_s,
        "n_params": int(n_params), "num_latent": config["num_latent"], "num_internal": config["num_internal"],
        "scalar_cond": config.get("scalar_cond"), "cond_encoder": config.get("cond_encoder"),
        "libsize_cond": bool(model.libsize_cond), "target_libsize": target_mode, **target_info, **exclusion_info,
        **library_info,
        "exclude_cell_types": config.get("exclude_cell_types"), "holdout_donors": config.get("holdout_donors"),
        "seed": config.get("seed"),
        "cell_jsd_weight": config.get("cell_jsd_weight"), "nuclei_jsd_weight": config.get("nuclei_jsd_weight"),
        "alignment_weight": config.get("alignment_weight"),
        "num_epochs": config.get("num_epochs"), "batch_size": config.get("batch_size"),
        "learning_rate": config.get("learning_rate"), "beta": config.get("beta"),
        "normalize_losses": config.get("normalize_losses"), "eval_partition": args.eval_partition,
        "decode": "deterministic (mu), source=nuclei(1) -> target=cell(0)",
    }
    save_standard_outputs(args.output_dir, imputed, data["nuc_eval"].obs[["obs_id", "cell_type", "donor"]],
                          data["gene_names"], args_dict, data["split_info"])
    print("Done.")


if __name__ == "__main__":
    main()
