"""
Evaluate a trained cVAE on an INDEPENDENT matched nucleus/cell dataset (e.g. Bakken 2018, GSE123454):
nuclei -> imputed cells, scored against the real cells of the same matched cluster, plus the direct
test of the transformation: predicted delta (imputed - nucleus) vs empirical delta (cell - nucleus)
per matched cluster.

Inputs are the AnnData files from analysis/external/ingest_*.py (raw counts in .X, training gene order,
var['present'] marks genes that exist in the external annotation; absent genes are excluded from all metrics).
Labels: --label_col (default cluster_id) must be shared by nuclei and cells (the dataset's own matched clusters);
no mapping to the 17 training types is needed. The training type of each nucleus is additionally predicted
from the model's latent (kNN over training nuclei) and cross-tabulated against the external clusters.

Decode modes (--target_libsize; covariate models only, see analysis/cross_dataset_eval.py):
  source | ct_median_target_cell | quantile_map_target (use the external label)
  global_quantile_map_target | global_shift_target | global_shift_train | knn_cell_median_target |
  knn_quantile_map_target | knn_train_shift_target | knn_type_train_shift_target (no external label)
Baselines (always computed): raw nuclei; training CT-mean by predicted type; nucleus + training per-type delta
by predicted type.

Usage:
  python analysis/external/eval_matched_external.py --run_dir run/run_2026-10-16_libsize_armA_jsd0.1_seed1 \
      --nuclei data/external/bakken2018_GSE123454/processed/nuclei_exon_intron.h5ad \
      --cells  data/external/bakken2018_GSE123454/processed/cells_exon_intron.h5ad \
      --target_libsize knn_type_train_shift_target --name armA_knntype --output_dir run/external_eval/bakken2018
"""
import os, sys, json, argparse, time
import numpy as np
import pandas as pd
import anndata as ad
import torch
from scipy import stats

ROOT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE"
sys.path.insert(0, ROOT)
from analysis.cross_dataset_eval import cross_decode, build_target_log_libsize, compute_metrics, log_library_size_np
from analysis.benchmark.predict_cvae import load_model, knn_predicted_types
from analysis.benchmark.data_split import load_partition
from analysis.benchmark.evaluate_all import RIBOSOMAL_GENES, NUCLEAR_LNCRNA_GENES, CYTOPLASMIC_GENES

SIGN_THRESHOLD = 0.25   # log1p units; |empirical delta| above this counts for sign agreement


def delta_metrics(imputed, X_nuc, X_cell, ct_nuc, ct_cell, present, min_n=5):
    """Predicted delta (imputed - nucleus) vs empirical delta (cell - nucleus), per matched cluster and pooled."""
    out = {"per_cluster": {}}
    pooled_emp, pooled_pred = [], []
    cl_emp, cl_pred = [], []
    for ct in sorted(set(ct_nuc) & set(ct_cell)):
        mn, mc = ct_nuc == ct, ct_cell == ct
        if mn.sum() < min_n or mc.sum() < min_n:
            continue
        emp = (X_cell[mc].mean(0) - X_nuc[mn].mean(0))[present]
        pred = (imputed[mn].mean(0) - X_nuc[mn].mean(0))[present]
        big = np.abs(emp) > SIGN_THRESHOLD
        out["per_cluster"][ct] = {
            "n_nuc": int(mn.sum()), "n_cell": int(mc.sum()),
            "delta_pearson": float(np.corrcoef(emp, pred)[0, 1]),
            "delta_spearman": float(stats.spearmanr(emp, pred)[0]),
            "n_genes_abs_emp_gt_thr": int(big.sum()),
            "sign_agreement_abs_emp_gt_thr": float((np.sign(emp[big]) == np.sign(pred[big])).mean()) if big.sum() else np.nan,
            "emp_delta_sd": float(emp.std()), "pred_delta_sd": float(pred.std()),
            "slope_pred_on_emp": float(np.polyfit(emp, pred, 1)[0]),
        }
        pooled_emp.append(emp); pooled_pred.append(pred); cl_emp.append(emp); cl_pred.append(pred)
    if cl_emp:
        E, P = np.concatenate(pooled_emp), np.concatenate(pooled_pred)
        big = np.abs(E) > SIGN_THRESHOLD
        out["pooled_delta_pearson"] = float(np.corrcoef(E, P)[0, 1])
        out["pooled_delta_spearman"] = float(stats.spearmanr(E, P)[0])
        out["pooled_sign_agreement_abs_emp_gt_thr"] = float((np.sign(E[big]) == np.sign(P[big])).mean())
        out["pooled_n_genes_abs_emp_gt_thr"] = int(big.sum())
        # cluster-averaged (composition-free) delta
        e, p = np.mean(cl_emp, axis=0), np.mean(cl_pred, axis=0)
        out["cluster_mean_delta_pearson"] = float(np.corrcoef(e, p)[0, 1])
        out["cluster_mean_delta_spearman"] = float(stats.spearmanr(e, p)[0])
        pcs = [v["delta_pearson"] for v in out["per_cluster"].values()]
        out["per_cluster_delta_pearson_median"] = float(np.median(pcs))
        out["per_cluster_sign_agreement_median"] = float(np.nanmedian([v["sign_agreement_abs_emp_gt_thr"] for v in out["per_cluster"].values()]))
    out["sign_threshold"] = SIGN_THRESHOLD
    return out


def gene_class_deltas(imputed, X_nuc, X_cell, genes, present):
    """Mean delta of prespecified gene classes: empirical (cell - nucleus) and predicted (imputed - nucleus)."""
    gl = list(genes)
    pres = set(np.asarray(genes)[present])
    emp = X_cell.mean(0) - X_nuc.mean(0)
    pred = imputed.mean(0) - X_nuc.mean(0)
    out = {}
    for name, gs, sgn in [("ribosomal", RIBOSOMAL_GENES, 1), ("cytoplasmic", CYTOPLASMIC_GENES, 1), ("nuclear_lncrna", NUCLEAR_LNCRNA_GENES, -1)]:
        idx = [gl.index(g) for g in gs if g in pres]
        if not idx:
            out[name] = {"n_found": 0}; continue
        out[name] = {"n_found": len(idx), "expected_sign": sgn,
                     "empirical_mean_delta": float(emp[idx].mean()), "empirical_frac_expected_sign": float(((emp[idx] * sgn) > 0).mean()),
                     "predicted_mean_delta": float(pred[idx].mean()), "predicted_frac_expected_sign": float(((pred[idx] * sgn) > 0).mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--nuclei", required=True); ap.add_argument("--cells", required=True)
    ap.add_argument("--label_col", default="cluster_id")
    ap.add_argument("--target_libsize", default="auto")
    ap.add_argument("--name", required=True); ap.add_argument("--output_dir", required=True)
    ap.add_argument("--no_baselines", action="store_true")
    ap.add_argument("--type_predictor", default="latent_knn", choices=["latent_knn", "pca_lr"],
                    help="how the training type of each external nucleus is predicted (label-free): latent kNN vote or the depth-robust log-CPM/PCA/LR reference classifier")
    ap.add_argument("--batch_size", type=int, default=512)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, model_path = load_model(args.run_dir, device)
    print(f"model {model_path} -> {device}; libsize_cond={model.libsize_cond}")

    nuc, cell = ad.read_h5ad(args.nuclei), ad.read_h5ad(args.cells)
    genes = nuc.var["gene_names"].astype(str).values
    assert (cell.var["gene_names"].astype(str).values == genes).all()
    present = nuc.var["present"].values.astype(bool)
    X_nuc = np.log1p(np.asarray(nuc.X, dtype=np.float32)); X_cell = np.log1p(np.asarray(cell.X, dtype=np.float32))
    ct_nuc = nuc.obs[args.label_col].astype(str).values; ct_cell = cell.obs[args.label_col].astype(str).values
    print(f"nuclei {X_nuc.shape}, cells {X_cell.shape}, present genes {present.sum()}, clusters {len(set(ct_nuc) & set(ct_cell))}")

    # predicted training type from the latent (label-free)
    Xn_tr, obs_tr = load_partition("nuc", "train", normalize="log1p")
    if args.type_predictor == "pca_lr":
        from analysis.external.type_classifier import pca_lr_predict
        Xn_va, obs_va = load_partition("nuc", "valid", normalize="log1p")
        pred_type, pinfo = pca_lr_predict(Xn_tr, obs_tr["cell_type"].values, X_nuc, present, X_valid=Xn_va, ct_valid=obs_va["cell_type"].values)
        del Xn_va
    else:
        pred_type, pinfo = knn_predicted_types(model, device, X_nuc, Xn_tr, obs_tr["cell_type"].values)
    print(f"type predictor {args.type_predictor}: {json.dumps(pinfo)}")
    xtab = pd.crosstab(pd.Series(ct_nuc, name=args.label_col), pd.Series(pred_type, name="predicted_training_type"))
    xtab.to_csv(os.path.join(args.output_dir, f"predicted_type_crosstab_{args.name}.csv"))
    pd.DataFrame({"sample": nuc.obs_names, args.label_col: ct_nuc, "predicted_training_type": pred_type}).to_csv(
        os.path.join(args.output_dir, f"predicted_types_{args.name}.csv"), index=False)
    majority = xtab.idxmax(axis=1).to_dict(); purity = (xtab.max(axis=1) / xtab.sum(axis=1)).round(3).to_dict()

    # decode library
    mode = args.target_libsize
    if not model.libsize_cond:
        if mode not in ("auto", "source"):
            raise SystemExit(f"{mode} needs a libsize_cond model")
        mode, target, tinfo = "n/a (no covariate)", None, {}
    else:
        if mode == "auto":
            mode = "knn_type_train_shift_target"
        if mode.startswith("knn_type_") and args.type_predictor != "latent_knn":
            from analysis.benchmark.predict_cvae import build_target_log_libsize as bt_train
            sub = "ct_median_cell" if mode == "knn_type_train_shift_target" else "quantile_map"
            target, tinfo = bt_train(sub, args.run_dir, X_nuc, pred_type)
            tinfo.update({"mode": f"{args.type_predictor}_type -> {sub} (training libraries)", "uses_cell_type_label": False})
            mode = f"{args.type_predictor}_type_{sub}"
        else:
            target, tinfo = build_target_log_libsize(mode, args.run_dir, X_nuc, ct_nuc, X_cell, ct_cell, model=model, device=device)
    print(f"decode library: {mode} {json.dumps({k: v for k, v in tinfo.items() if not isinstance(v, (list, dict))})}")
    imputed = cross_decode(model, X_nuc, device, args.batch_size, target_log_libsize=target)
    np.save(os.path.join(args.output_dir, f"imputed_{args.name}.npy"), imputed.astype(np.float32))
    # The model's own delta applied to the observed nucleus: X_nuc + (cell decode - nucleus decode of the same latent).
    # Keeps the platform-specific profile of the input and uses the model only for the nucleus->cell shift.
    recon = np.zeros_like(X_nuc)
    with torch.no_grad():
        for s in range(0, X_nuc.shape[0], args.batch_size):
            b = torch.from_numpy(X_nuc[s:s + args.batch_size]).to(device)
            recon[s:s + args.batch_size] = model.decode_cross(b, 1, 1).cpu().numpy()
    residual = (X_nuc + imputed - recon).astype(np.float32)
    np.save(os.path.join(args.output_dir, f"imputed_residual_{args.name}.npy"), residual)

    def score(M, tag):
        r = compute_metrics(M[:, present], X_cell[:, present], ct_nuc, ct_cell, genes[present])
        r["delta"] = delta_metrics(M, X_nuc, X_cell, ct_nuc, ct_cell, present)
        r["gene_classes"] = gene_class_deltas(M, X_nuc, X_cell, genes, present)
        r["output_log_library_median"] = float(np.median(log_library_size_np(M)))
        r["tag"] = tag
        return r

    results = {"method": args.name, "type_predictor": args.type_predictor, "run_dir": args.run_dir, "nuclei": args.nuclei, "cells": args.cells, "label_col": args.label_col,
               "libsize_cond": bool(model.libsize_cond), "target_libsize": mode, "target_libsize_info": tinfo,
               "n_present_genes": int(present.sum()), "predicted_type_info": pinfo,
               "predicted_type_majority_per_cluster": majority, "predicted_type_purity_per_cluster": purity,
               "input_log_library_median": {"nuclei": float(np.median(log_library_size_np(X_nuc))), "cells": float(np.median(log_library_size_np(X_cell)))}}
    results["model"] = score(imputed, "model")
    results["model_delta_on_nucleus"] = score(residual, "model_delta_on_nucleus")
    results["nucleus_reconstruction_corr_median"] = float(np.median([np.corrcoef(X_nuc[i, present], recon[i, present])[0, 1] for i in range(X_nuc.shape[0])]))
    if not args.no_baselines:
        results["baseline_raw_nuclei"] = score(X_nuc, "raw_nuclei")
        Xc_tr, obs_c = load_partition("cell", "train", normalize="log1p")
        ct_tr_c = obs_c["cell_type"].astype(str).values; ct_tr_n = obs_tr["cell_type"].astype(str).values
        mean_c = {t: Xc_tr[ct_tr_c == t].mean(0) for t in np.unique(pred_type)}
        mean_n = {t: Xn_tr[ct_tr_n == t].mean(0) for t in np.unique(pred_type)}
        ctmean = np.stack([mean_c[t] for t in pred_type]).astype(np.float32)
        shifted = (X_nuc + np.stack([mean_c[t] - mean_n[t] for t in pred_type])).astype(np.float32)
        results["baseline_train_ctmean_predicted_type"] = score(ctmean, "train_ctmean_by_predicted_type")
        results["baseline_nucleus_plus_train_delta_predicted_type"] = score(shifted, "nucleus_plus_train_type_delta")
        del Xc_tr
    results["elapsed_s"] = round(time.time() - t0, 1)
    with open(os.path.join(args.output_dir, f"eval_{args.name}.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)

    keys = ["gene_corr_median", "r2_median", "overall_corr", "qq_gene_corr_median", "wasserstein_median", "variance_ratio_median", "per_cell_corr_median", "shift_score_mean"]
    print(f"\n== {args.name} ({mode}) ==")
    rows = [(k, results[k]) for k in results if isinstance(results[k], dict) and "tag" in results[k]]
    print(f"{'row':<40}" + "".join(f"{k[:14]:>15}" for k in keys) + f"{'dPear(pool)':>13}{'dPear(med)':>12}{'sign':>7}{'ribo':>7}{'lnc':>7}")
    for k, r in rows:
        d = r["delta"]; gc = r["gene_classes"]
        print(f"{r['tag']:<40}" + "".join(f"{(r[k] if r[k] is not None else np.nan):>15.3f}" for k in keys)
              + f"{d.get('pooled_delta_pearson', np.nan):>13.3f}{d.get('per_cluster_delta_pearson_median', np.nan):>12.3f}"
              + f"{d.get('pooled_sign_agreement_abs_emp_gt_thr', np.nan):>7.2f}{gc['ribosomal'].get('predicted_frac_expected_sign', np.nan):>7.2f}{gc['nuclear_lncrna'].get('predicted_frac_expected_sign', np.nan):>7.2f}")
    print("\npredicted training type per external cluster (majority, purity):")
    for c in sorted(majority, key=lambda s: int(s[2:]) if s[2:].isdigit() else s):
        print(f"  {c:<6} {majority[c]:<12} {purity[c]:.2f}")
    print(f"done in {results['elapsed_s']} s")


if __name__ == "__main__":
    main()
