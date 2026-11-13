"""
Cross-dataset validation: apply a v3-trained cVAE to a different batch/protocol
and evaluate against ground truth cells from that batch.

This tests generalization: model was trained on v3 (n3A -> c3A),
can it cross-decode v2 nuclei (n2A) into v2 cells (c2A)?

Usage:
    python analysis/cross_dataset_eval.py \
        --run_dir run/run_2026-07-18-19-40-51_570984 \
        --nuc_batch n2A \
        --cell_batch c2A \
        --output_dir run/cross_dataset_eval/v2_batch \
        --name cvae_v2
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import anndata as ad
from scipy import sparse, stats

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from imputeVAE.models_cond import VAE_JSD_Cond


DATA_ROOT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi"
DEFAULT_GENE_LIST = "data/training_gene_list_v3.csv"


def load_modality(data_dir, batch_suffix, gene_list_path):
    """Load one modality, filtered to a specific batch and gene list."""
    gene_list = pd.read_csv(gene_list_path, header=None)[0].tolist()
    common_genes = pd.read_csv("data/common_genes.csv", header=0)["common_genes"].tolist()
    common_ct = pd.read_csv("data/common_cell_types.csv", header=0)["common_cell_types"].tolist()

    adata = ad.read_h5ad(os.path.join(DATA_ROOT, data_dir, "data.h5ad"))
    if "meta_batch_id" not in adata.obs:
        adata.obs["meta_batch_id"] = adata.obs["batch_id"].apply(lambda x: x.split(".")[0])

    adata = adata[adata.obs["meta_batch_id"].str.endswith(batch_suffix)].copy()
    adata = adata[:, adata.var["gene_names"].isin(common_genes)].copy()
    adata = adata[adata.obs["cell_type"].isin(common_ct)].copy()
    adata = adata[:, adata.var["gene_names"].isin(gene_list)].copy()
    adata = adata[:, adata.var["gene_names"].argsort()].copy()

    X = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
    adata.X = np.log1p(X).astype(np.float32)
    return adata


def cross_decode(model, X_nuc, device, batch_size=512, target_log_libsize=None):
    """Encode as nuclei (1), decode as cell (0). target_log_libsize: per-nucleus [N] array (libsize_cond runs) or None."""
    n = X_nuc.shape[0]
    imputed = np.zeros_like(X_nuc)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = torch.from_numpy(X_nuc[start:end]).to(device)
            kw = {}
            if target_log_libsize is not None:
                kw["target_log_libsize"] = torch.from_numpy(target_log_libsize[start:end]).to(device)
            decoded = model.decode_cross(batch, source_modality_id=1, target_modality_id=0, **kw)
            imputed[start:end] = decoded.cpu().numpy()
    return imputed


def log_library_size_np(X):
    """log1p(total counts) per row of a log1p matrix (the model's own definition, over the training gene set)."""
    return np.log1p(np.expm1(X.astype(np.float64)).sum(axis=1))


def build_target_log_libsize(mode, run_dir, X_nuc, ct_nuc, X_cell, ct_cell, model=None, device=None,
                             type_predictor="latent_knn"):
    """Per-nucleus decode library for libsize_cond models (2026-10-23, cross-chemistry test of the covariate model).

    type_predictor (knn_type_* modes only): "latent_knn" = majority vote over the K=50 nearest v3 training nuclei in
    the model's latent (2026-10-23; 77 % accurate on v2); "pca_lr" = the depth-robust log-CPM / HVG / PCA / logistic
    regression reference classifier fitted on the v3 training nuclei (analysis/external/type_classifier.py,
    2026-10-26; replaces the vote off-chemistry). Both use training labels only. The prediction is returned in
    info["_predicted_types"] for the caller to save.

    source                 the nucleus's own library (no-covariate behaviour)
    ct_median_cell         per-type median library of the v3 TRAINING cells (<run_dir>/libsize_stats.json);
                           what the benchmark uses; ignores the chemistry of the target batch
    ct_median_target_cell  per-type median library of the target-batch cells (e.g. v2 cells): tests whether
                           the cross-chemistry drop is a library-scale shift (uses the cell-type label)
    quantile_map_target    nucleus's library quantile within target-batch nuclei of its type, mapped onto the
                           target-batch cells of that type (keeps per-cell size variation on the target scale)
    Label-free scale modes (2026-10-23; no cell-type label, the nuclei-only setting):
    global_quantile_map_target  pooled quantile map, target-batch nuclei -> target-batch cells (all types together)
    global_shift_target         nucleus's own library + (median target-batch cell - median target-batch nucleus)
    global_shift_train          nucleus's own library + the same shift computed on the v3 TRAINING partition
                                (benchmark cache): needs no target-batch cells at all; tests whether the
                                cell/nucleus RNA-content ratio transfers across chemistries
    knn_cell_median_target      median library of the K=50 nearest TARGET-batch cells in the aligned latent
                                (unlabelled target cells; the latent supplies the type-level RNA content)
    knn_quantile_map_target     nucleus's library quantile among its K nearest target-batch nuclei mapped onto
                                its K nearest target-batch cells
    knn_type_median_target      type predicted by kNN majority vote over v3 TRAINING nuclei (training labels only)
                                -> per-type median of the target-batch cells (uses target cell labels for the table)
    knn_type_train_shift_target predicted type -> v3 training-cell per-type median + one global scalar
                                (median target-batch cell - median training cell): no target label anywhere
    knn_train_shift_target      K nearest v3 TRAINING cells (benchmark cache) + one global scalar
                                (median target-batch cell - median training cell): the nuclei-only setting
                                for a new chemistry where only the chemistry's overall whole-cell RNA
                                content is known
    """
    if mode == "source":
        return None, {}
    if mode.startswith("knn_type_"):
        from analysis.benchmark.predict_cvae import knn_predicted_types
        from analysis.benchmark.data_split import load_partition
        assert model is not None
        Xn_tr, obs_tr = load_partition("nuc", "train", normalize="log1p")
        if type_predictor == "pca_lr":
            from analysis.external.type_classifier import pca_lr_predict
            Xn_va, obs_va = load_partition("nuc", "valid", normalize="log1p")
            present = np.ones(X_nuc.shape[1], dtype=bool)  # same training gene list, all genes present
            pred, info = pca_lr_predict(Xn_tr, obs_tr["cell_type"].values, X_nuc, present,
                                        X_valid=Xn_va, ct_valid=obs_va["cell_type"].values)
            del Xn_va
        elif type_predictor == "latent_knn":
            pred, info = knn_predicted_types(model, device, X_nuc, Xn_tr, obs_tr["cell_type"].values)
        else:
            raise ValueError(type_predictor)
        info["type_predictor"] = type_predictor
        info["predicted_type_accuracy_vs_target_label"] = float((pred == np.asarray(ct_nuc).astype(str)).mean())
        info["_predicted_types"] = np.asarray(pred).astype(str)
        if mode == "knn_type_median_target":
            target, tinfo = build_target_log_libsize("ct_median_target_cell", run_dir, X_nuc, pred, X_cell, ct_cell)
        elif mode == "knn_type_train_shift_target":
            target, tinfo = build_target_log_libsize("ct_median_cell", run_dir, X_nuc, pred, X_cell, ct_cell)
            Xc_tr = load_partition("cell", "train", normalize="log1p")[0]
            shift = float(np.median(log_library_size_np(X_cell)) - np.median(log_library_size_np(Xc_tr)))
            target = target + shift
            tinfo["global_log_shift"] = shift
        else:
            raise ValueError(mode)
        tinfo.update(info); tinfo["mode"] = mode; tinfo["uses_cell_type_label"] = False
        tinfo.update({"target_log_libsize_mean": float(target.mean()), "target_log_libsize_std": float(target.std())})
        return target.astype(np.float32), tinfo
    if mode.startswith("knn_"):
        from analysis.benchmark.predict_cvae import encode_all, knn_library_targets, log_library_size_np as _lib
        assert model is not None
        if mode == "knn_train_shift_target":
            from analysis.benchmark.data_split import load_partition
            X_cell_train = load_partition("cell", "train", normalize="log1p")[0]
            target, info = knn_library_targets("knn_cell_median", model, device, X_nuc, None, X_cell_train)
            shift = float(np.median(log_library_size_np(X_cell)) - np.median(log_library_size_np(X_cell_train)))
            target = target + shift
            info.update({"source": "K nearest v3 training cells + global chemistry shift", "global_log_shift": shift})
        else:
            target, info = knn_library_targets(mode.replace("_target", ""), model, device, X_nuc, X_nuc, X_cell)
            info["source"] = "K nearest target-batch cells (and nuclei) in latent space"
        info.update({"mode": mode, "target_log_libsize_mean": float(target.mean()),
                     "target_log_libsize_std": float(target.std())})
        return target.astype(np.float32), info
    ct_nuc = np.asarray(ct_nuc).astype(str)
    ct_cell = np.asarray(ct_cell).astype(str)
    target = np.zeros(X_nuc.shape[0], dtype=np.float64)
    info = {"mode": mode, "uses_cell_type_label": not mode.startswith("global_")}
    if mode.startswith("global_"):
        lib_nuc = log_library_size_np(X_nuc)
        if mode == "global_shift_train":
            from analysis.benchmark.data_split import load_partition
            tr_nuc = np.log1p(np.asarray(load_partition("nuc", "train", normalize="raw", mmap=True)[0]).sum(axis=1))
            tr_cell = np.log1p(np.asarray(load_partition("cell", "train", normalize="raw", mmap=True)[0]).sum(axis=1))
            shift = float(np.median(tr_cell) - np.median(tr_nuc))
            target = lib_nuc + shift
            info.update({"source": "v3 training partition medians (benchmark cache)", "global_log_shift": shift,
                         "train_median_cell": float(np.median(tr_cell)), "train_median_nuclei": float(np.median(tr_nuc))})
        else:
            lib_cell = log_library_size_np(X_cell)
            if mode == "global_shift_target":
                shift = float(np.median(lib_cell) - np.median(lib_nuc))
                target = lib_nuc + shift
                info.update({"source": "target-batch pooled medians", "global_log_shift": shift})
            elif mode == "global_quantile_map_target":
                grid = np.linspace(0, 1, 101)
                q = np.interp(lib_nuc, np.quantile(lib_nuc, grid), grid)
                target = np.interp(q, grid, np.quantile(lib_cell, grid))
                info["source"] = "target-batch pooled quantiles"
            else:
                raise ValueError(mode)
            info.update({"target_median_cell": float(np.median(lib_cell)), "target_median_nuclei": float(np.median(lib_nuc))})
        info["target_log_libsize_mean"] = float(target.mean())
        info["target_log_libsize_std"] = float(target.std())
        return target.astype(np.float32), info
    if mode == "ct_median_cell":
        with open(os.path.join(run_dir, "libsize_stats.json")) as f:
            stats = json.load(f)
        med = stats["ct_median_cell"]
        fallback = float(np.median(list(med.values())))
        target[:] = fallback
        for ct in np.unique(ct_nuc):
            if ct in med:
                target[ct_nuc == ct] = med[ct]
        info["source"] = "training-cell medians (libsize_stats.json)"
    else:
        lib_cell = log_library_size_np(X_cell)
        lib_nuc = log_library_size_np(X_nuc)
        fallback = float(np.median(lib_cell))
        target[:] = fallback
        grid = np.linspace(0, 1, 101)
        for ct in np.unique(ct_nuc):
            m = ct_nuc == ct
            mc = ct_cell == ct
            if mc.sum() == 0:
                continue
            if mode == "ct_median_target_cell":
                target[m] = float(np.median(lib_cell[mc]))
            elif mode == "quantile_map_target":
                q_nuc = np.quantile(lib_nuc[m], grid)
                q_cell = np.quantile(lib_cell[mc], grid)
                q = np.interp(lib_nuc[m], q_nuc, grid)
                target[m] = np.interp(q, grid, q_cell)
            else:
                raise ValueError(mode)
        info["source"] = "target-batch cells"
    info["target_log_libsize_mean"] = float(target.mean())
    info["target_log_libsize_std"] = float(target.std())
    return target.astype(np.float32), info


def compute_metrics(imputed, X_cell, ct_imputed, ct_cell, gene_names):
    """Compute key evaluation metrics."""
    common_cts = sorted(set(ct_imputed) & set(ct_cell))
    results = {"n_imputed": int(len(ct_imputed)), "n_cell": int(len(ct_cell)),
               "n_common_cts": len(common_cts), "common_cts": common_cts}

    # --- CT-level means ---
    imp_means = {}
    cell_means = {}
    for ct in common_cts:
        imp_means[ct] = imputed[ct_imputed == ct].mean(axis=0)
        cell_means[ct] = X_cell[ct_cell == ct].mean(axis=0)

    # Gene correlation (per-gene, across CTs)
    n_genes = imputed.shape[1]
    gene_corrs = []
    for g in range(n_genes):
        imp_vals = np.array([imp_means[ct][g] for ct in common_cts])
        cell_vals = np.array([cell_means[ct][g] for ct in common_cts])
        if np.std(imp_vals) > 0 and np.std(cell_vals) > 0:
            r = np.corrcoef(imp_vals, cell_vals)[0, 1]
            gene_corrs.append(r)
    results["gene_corr_median"] = float(np.median(gene_corrs))
    results["gene_corr_mean"] = float(np.mean(gene_corrs))
    results["gene_corr_n"] = len(gene_corrs)

    # R² per gene
    r2_vals = []
    for g in range(n_genes):
        imp_vals = np.array([imp_means[ct][g] for ct in common_cts])
        cell_vals = np.array([cell_means[ct][g] for ct in common_cts])
        ss_res = np.sum((cell_vals - imp_vals) ** 2)
        ss_tot = np.sum((cell_vals - cell_vals.mean()) ** 2)
        if ss_tot > 0:
            r2_vals.append(1 - ss_res / ss_tot)
    results["r2_median"] = float(np.median(r2_vals))
    results["r2_mean"] = float(np.mean(r2_vals))

    # Delta (shift) correlation
    # Ground truth delta: cell_mean - nuc_mean (but we don't have nuc ground truth for this batch
    # separately — we compare imputed vs real cell at CT level)
    # Overall correlation across all CT-gene pairs
    all_imp = np.concatenate([imp_means[ct] for ct in common_cts])
    all_cell = np.concatenate([cell_means[ct] for ct in common_cts])
    results["overall_corr"] = float(np.corrcoef(all_imp, all_cell)[0, 1])

    # --- Label-free metrics ---
    # QQ gene correlation (quantile-quantile per gene)
    n_quantiles = 500
    quantiles = np.linspace(0, 1, n_quantiles)
    qq_corrs = []
    for g in range(n_genes):
        q_imp = np.quantile(imputed[:, g], quantiles)
        q_cell = np.quantile(X_cell[:, g], quantiles)
        if np.std(q_imp) > 0 and np.std(q_cell) > 0:
            r = np.corrcoef(q_imp, q_cell)[0, 1]
            qq_corrs.append(r)
    results["qq_gene_corr_median"] = float(np.median(qq_corrs))
    results["qq_gene_corr_mean"] = float(np.mean(qq_corrs))

    # Wasserstein per gene (quantile-based)
    wass_vals = []
    for g in range(n_genes):
        q_imp = np.quantile(imputed[:, g], quantiles)
        q_cell = np.quantile(X_cell[:, g], quantiles)
        wass_vals.append(float(np.mean(np.abs(q_imp - q_cell))))
    results["wasserstein_median"] = float(np.median(wass_vals))
    results["wasserstein_mean"] = float(np.mean(wass_vals))

    # --- Within-CT variance ratio ---
    var_ratios = []
    for ct in common_cts:
        mask_imp = ct_imputed == ct
        mask_cell = ct_cell == ct
        if mask_imp.sum() >= 5 and mask_cell.sum() >= 5:
            var_imp = imputed[mask_imp].var(axis=0).mean()
            var_cell = X_cell[mask_cell].var(axis=0).mean()
            if var_cell > 0:
                var_ratios.append(float(var_imp / var_cell))
    results["variance_ratio_median"] = float(np.median(var_ratios)) if var_ratios else None
    results["variance_ratio_mean"] = float(np.mean(var_ratios)) if var_ratios else None

    # --- Per-cell correlation ---
    # For each imputed cell, correlate with its CT mean in ground truth
    per_cell_corrs = []
    for ct in common_cts:
        mask = ct_imputed == ct
        if mask.sum() < 1:
            continue
        ct_mean_cell = cell_means[ct]
        for i in np.where(mask)[0]:
            r = np.corrcoef(imputed[i], ct_mean_cell)[0, 1]
            per_cell_corrs.append(r)
    results["per_cell_corr_median"] = float(np.median(per_cell_corrs)) if per_cell_corrs else None

    # --- Shift score (mean absolute CT-mean difference) ---
    shifts = []
    for ct in common_cts:
        shift = np.abs(imp_means[ct] - cell_means[ct]).mean()
        shifts.append(float(shift))
    results["shift_score_mean"] = float(np.mean(shifts))

    # --- Per-CT breakdown ---
    ct_breakdown = {}
    for ct in common_cts:
        n_imp = int((ct_imputed == ct).sum())
        n_cell = int((ct_cell == ct).sum())
        # CT-level gene correlation
        r = np.corrcoef(imp_means[ct], cell_means[ct])[0, 1]
        ct_breakdown[ct] = {"n_imputed": n_imp, "n_cell": n_cell, "ct_gene_corr": float(r)}
    results["per_ct"] = ct_breakdown

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Training run directory with model")
    parser.add_argument("--nuc_batch", type=str, default="n2A",
                        help="Nuclei batch suffix to evaluate on (default: n2A)")
    parser.add_argument("--cell_batch", type=str, default="c2A",
                        help="Cell batch suffix for ground truth (default: c2A)")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--name", type=str, default="cross_dataset")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--target_libsize", default="auto",
                        choices=["auto", "source", "ct_median_cell", "ct_median_target_cell", "quantile_map_target",
                                 "global_quantile_map_target", "global_shift_target", "global_shift_train",
                                 "knn_cell_median_target", "knn_quantile_map_target", "knn_train_shift_target",
                                 "knn_type_median_target", "knn_type_train_shift_target"],
                        help="libsize_cond runs only; auto = ct_median_target_cell for covariate models; "
                             "global_* and knn_* modes use no cell-type label")
    parser.add_argument("--type_predictor", default="latent_knn", choices=["latent_knn", "pca_lr"],
                        help="knn_type_* modes: how the training type of each target nucleus is predicted "
                             "(latent kNN vote, or the depth-robust log-CPM/PCA/LR reference classifier)")
    args = parser.parse_args()

    output_dir = args.output_dir or f"run/cross_dataset_eval/{args.nuc_batch}"
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    with open(os.path.join(args.run_dir, "arguments.json")) as f:
        config = json.load(f)

    num_internal = config["num_internal"]
    if isinstance(num_internal, str):
        num_internal = [int(x) for x in num_internal.split(",")]
    model = VAE_JSD_Cond(
        input_size=config["num_feature"],
        num_internal=num_internal,
        num_latent=config["num_latent"],
        embed_dim=config.get("embed_dim", 8),
        scalar_cond=config.get("scalar_cond", False),
        cond_encoder=config.get("cond_encoder", False),
        libsize_cond=config.get("libsize_cond", False),
    )
    model_path = os.path.join(args.run_dir, "models", "model.pt")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded from {model_path} -> {device}")

    gene_list_path = config.get("gene_list", DEFAULT_GENE_LIST)

    # Load target batch data
    print(f"\nLoading nuclei ({args.nuc_batch})...")
    adata_nuc = load_modality("10X_nuclei_v2_v3_AIBS", args.nuc_batch, gene_list_path)
    print(f"  Nuclei: {adata_nuc.shape}")

    print(f"Loading cells ({args.cell_batch})...")
    adata_cell = load_modality("10X_cells_v2_v3_AIBS", args.cell_batch, gene_list_path)
    print(f"  Cells: {adata_cell.shape}")

    gene_names = list(adata_nuc.var["gene_names"].values)

    # Library target for covariate models
    X_cell_for_lib = adata_cell.X if not sparse.issparse(adata_cell.X) else adata_cell.X.toarray()
    target_mode = args.target_libsize
    if not model.libsize_cond:
        if target_mode not in ("auto", "source"):
            raise SystemExit(f"--target_libsize {target_mode} needs a libsize_cond model; {args.run_dir} has none")
        target_mode, target, target_info = "n/a (no covariate)", None, {}
    else:
        if target_mode == "auto":
            target_mode = "ct_median_target_cell"
        target, target_info = build_target_log_libsize(
            target_mode, args.run_dir, adata_nuc.X, adata_nuc.obs["cell_type"].values,
            X_cell_for_lib, adata_cell.obs["cell_type"].values, model=model, device=device,
            type_predictor=args.type_predictor)
        pred_types = target_info.pop("_predicted_types", None)
        if pred_types is not None:
            ct_true = adata_nuc.obs["cell_type"].astype(str).values
            pd.DataFrame({"obs_id": adata_nuc.obs_names, "cell_type": ct_true, "predicted_training_type": pred_types}).to_csv(
                os.path.join(output_dir, f"predicted_types_{args.name}.csv"), index=False)
            xtab = pd.crosstab(pd.Series(ct_true, name="cell_type"), pd.Series(pred_types, name="predicted_training_type"))
            xtab.to_csv(os.path.join(output_dir, f"predicted_type_crosstab_{args.name}.csv"))
            per_type_recall = {ct: float((pred_types[ct_true == ct] == ct).mean()) for ct in np.unique(ct_true)}
            target_info["predicted_type_recall_per_type"] = per_type_recall
        print(f"libsize target = {target_mode}: {target_info}")

    # Cross-decode
    print(f"\nCross-decoding {adata_nuc.shape[0]} nuclei -> cell ...")
    imputed = cross_decode(model, adata_nuc.X, device, args.batch_size, target_log_libsize=target)
    print(f"  Imputed: min={imputed.min():.3f}, max={imputed.max():.3f}, mean={imputed.mean():.3f}")

    # Save imputed
    np.save(os.path.join(output_dir, "imputed_cell_from_nucleus.npy"), imputed.astype(np.float32))
    adata_nuc.obs[["cell_type"]].to_csv(os.path.join(output_dir, "nucleus_obs.csv"))

    # Compute metrics
    X_cell = adata_cell.X if not sparse.issparse(adata_cell.X) else adata_cell.X.toarray()
    ct_imputed = adata_nuc.obs["cell_type"].values.astype(str)
    ct_cell = adata_cell.obs["cell_type"].values.astype(str)

    print("\nComputing metrics...")
    results = compute_metrics(imputed, X_cell, ct_imputed, ct_cell, gene_names)
    results["method"] = args.name
    results["nuc_batch"] = args.nuc_batch
    results["cell_batch"] = args.cell_batch
    results["run_dir"] = args.run_dir
    results["gene_list"] = gene_list_path
    results["libsize_cond"] = bool(model.libsize_cond)
    results["target_libsize"] = target_mode
    results["target_libsize_info"] = target_info
    results["type_predictor"] = args.type_predictor if target_mode.startswith("knn_type_") else None

    # Save results
    out_path = os.path.join(output_dir, f"eval_{args.name}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Cross-dataset evaluation: {args.nuc_batch} -> {args.cell_batch}")
    print(f"{'='*60}")
    print(f"  Gene corr (median):     {results['gene_corr_median']:.4f}")
    print(f"  R² (median):            {results['r2_median']:.4f}")
    print(f"  Overall corr:           {results['overall_corr']:.4f}")
    print(f"  QQ gene corr (median):  {results['qq_gene_corr_median']:.4f}")
    print(f"  Wasserstein (median):   {results['wasserstein_median']:.4f}")
    print(f"  Variance ratio (median):{results['variance_ratio_median']:.4f}")
    print(f"  Shift score (mean):     {results['shift_score_mean']:.4f}")
    print(f"  Per-cell corr (median): {results['per_cell_corr_median']:.4f}")

    print(f"\nPer cell type:")
    print(f"  {'CT':<18} {'n_imp':>6} {'n_cell':>7} {'gene_corr':>10}")
    print(f"  {'-'*43}")
    for ct in sorted(results["per_ct"].keys()):
        info = results["per_ct"][ct]
        print(f"  {ct:<18} {info['n_imputed']:>6} {info['n_cell']:>7} {info['ct_gene_corr']:>10.4f}")


if __name__ == "__main__":
    main()
