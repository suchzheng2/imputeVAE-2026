"""
Nuclei-only application / chemistry-robustness test on Ding 2020 (GSE132044 cortex; 4 snRNA-seq chemistries x 2
experiments; no matched cells). Per group: label-free imputation (type predicted from the model's latent, decoded at
the training per-type cell library), predicted-type composition, pseudobulk of imputed vs raw nuclei against the
matched bulk RNA-seq of the same experiment (whole-tissue RNA = the only whole-cell reference; composition differs,
so the comparison is relative: does imputation move the nuclei pseudobulk toward the bulk?), and the agreement of
the predicted per-gene delta across chemistries.

Usage: python analysis/external/eval_ding2020.py --run_dir <covariate run> [--mode knn_type_quantile_map|knn_type_median|source] --name armA --output_dir run/external_eval/ding2020
"""
import os, sys, json, argparse, time
import numpy as np, pandas as pd, anndata as ad, torch
from scipy import stats
ROOT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE"; sys.path.insert(0, ROOT)
from analysis.cross_dataset_eval import cross_decode, log_library_size_np
from analysis.benchmark.predict_cvae import load_model, knn_predicted_types, build_target_log_libsize
from analysis.benchmark.data_split import load_partition
from analysis.benchmark.evaluate_all import RIBOSOMAL_GENES, NUCLEAR_LNCRNA_GENES, CYTOPLASMIC_GENES
from analysis.external.eval_matched_external import SIGN_THRESHOLD


def log_pseudobulk(X_log1p, present):
    """mean counts over cells -> CPM over present genes -> log1p."""
    m = np.expm1(np.clip(X_log1p.astype(np.float64), 0, None)).mean(0)[present]   # decoder output can be slightly negative
    return np.log1p(m / m.sum() * 1e6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True); ap.add_argument("--mode", default="knn_type_quantile_map")
    ap.add_argument("--name", required=True); ap.add_argument("--output_dir", required=True)
    ap.add_argument("--type_predictor", default="latent_knn", choices=["latent_knn", "pca_lr"])
    ap.add_argument("--nuclei", default=os.path.join(ROOT, "data/external/ding2020_GSE132044/processed/nuclei.h5ad"))
    ap.add_argument("--bulk", default=os.path.join(ROOT, "data/external/ding2020_GSE132044/processed/bulk_tpm_training_genes.csv"))
    args = ap.parse_args(); os.makedirs(args.output_dir, exist_ok=True); t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, _ = load_model(args.run_dir, device)
    nuc = ad.read_h5ad(args.nuclei); genes = nuc.var["gene_names"].astype(str).values; present = nuc.var["present"].values.astype(bool)
    X = np.log1p(np.asarray(nuc.X, dtype=np.float32)); grp = (nuc.obs["method"].astype(str) + "|" + nuc.obs["experiment"].astype(str)).values
    bulk = pd.read_csv(args.bulk, index_col=0)
    Xn_tr, obs_tr = load_partition("nuc", "train", normalize="log1p"); ct_tr = obs_tr["cell_type"].astype(str).values
    if args.type_predictor == "pca_lr":
        from analysis.external.type_classifier import pca_lr_predict
        Xn_va, obs_va = load_partition("nuc", "valid", normalize="log1p")
        pred, pinfo = pca_lr_predict(Xn_tr, ct_tr, X, present, X_valid=Xn_va, ct_valid=obs_va["cell_type"].values); del Xn_va
        pinfo["knn_vote_agreement_mean"] = pinfo["confidence_mean"]
    else:
        pred, pinfo = knn_predicted_types(model, device, X, Xn_tr, ct_tr)
    print(f"type predictor {args.type_predictor}: {json.dumps(pinfo)}")
    mode = args.mode
    if not model.libsize_cond:
        mode, target, tinfo = "n/a (no covariate)", None, {}
    elif mode == "source":
        target, tinfo = None, {}
    elif mode.startswith("knn_type_") and args.type_predictor != "latent_knn":
        sub = "ct_median_cell" if mode == "knn_type_median" else "quantile_map"
        target, tinfo = build_target_log_libsize(sub, args.run_dir, X, pred)
        mode = f"{args.type_predictor}_type_{sub}"
    else:
        target, tinfo = build_target_log_libsize(mode, args.run_dir, X, None, X_nuc_train=Xn_tr, model=model, device=device, ct_nuc_train=ct_tr)
    imputed = cross_decode(model, X, device, 512, target_log_libsize=target)
    recon = np.zeros_like(X)
    with torch.no_grad():
        for s in range(0, X.shape[0], 512):
            recon[s:s + 512] = model.decode_cross(torch.from_numpy(X[s:s + 512]).to(device), 1, 1).cpu().numpy()
    residual = X + imputed - recon
    # baseline: nucleus + training per-type delta of the predicted type (mean train cell - mean train nucleus)
    Xc_tr, obs_c = load_partition("cell", "train", normalize="log1p"); ct_c = obs_c["cell_type"].astype(str).values
    tdelta = {t: Xc_tr[ct_c == t].mean(0) - Xn_tr[ct_tr == t].mean(0) for t in np.unique(pred)}
    del Xc_tr
    shifted = X + np.stack([tdelta[t] for t in pred]).astype(np.float32)
    pd.DataFrame({"cell": nuc.obs_names, "group": grp, "predicted_training_type": pred, "lib": nuc.obs["lib_downsampled"].values}).to_csv(
        os.path.join(args.output_dir, f"predicted_types_{args.name}.csv"), index=False)
    comp = pd.crosstab(pd.Series(grp, name="group"), pd.Series(pred, name="predicted_type"), normalize="index").round(3)
    comp.to_csv(os.path.join(args.output_dir, f"predicted_type_composition_{args.name}.csv"))
    gl = list(genes); pres = set(genes[present])
    cls = {n: [gl.index(g) for g in gs if g in pres] for n, gs in [("ribosomal", RIBOSOMAL_GENES), ("cytoplasmic", CYTOPLASMIC_GENES), ("nuclear_lncrna", NUCLEAR_LNCRNA_GENES)]}
    res = {"method": args.name, "type_predictor": args.type_predictor, "run_dir": args.run_dir, "mode": mode, "target_info": {k: v for k, v in tinfo.items() if not isinstance(v, (list, dict))},
           "predicted_type_info": pinfo, "n_present_genes": int(present.sum()), "groups": {}}
    deltas = {}
    for g in sorted(set(grp)):
        m = grp == g; exp = g.split("|")[1]
        b = np.log1p(bulk.loc[genes[present], exp].values / bulk.loc[genes[present], exp].values.sum() * 1e6)
        pn, pi, pr, ps = log_pseudobulk(X[m], present), log_pseudobulk(imputed[m], present), log_pseudobulk(residual[m], present), log_pseudobulk(shifted[m], present)
        emp = b - pn; d_full = pi - pn; d_res = pr - pn; d_sh = ps - pn
        big = np.abs(emp) > SIGN_THRESHOLD
        dn_all = imputed[m].mean(0) - X[m].mean(0); deltas[g] = dn_all[present]
        r = {"n": int(m.sum()), "lib_median": float(np.median(nuc.obs["lib_downsampled"].values[m])),
             "predicted_type_vote_agreement": float(pinfo.get("knn_vote_agreement_mean", np.nan)),
             "composition_top": comp.loc[g].sort_values(ascending=False).head(6).to_dict(),
             "bulk_pearson": {"raw_nuclei": float(np.corrcoef(pn, b)[0, 1]), "imputed": float(np.corrcoef(pi, b)[0, 1]), "model_delta_on_nucleus": float(np.corrcoef(pr, b)[0, 1]), "nucleus_plus_train_type_delta": float(np.corrcoef(ps, b)[0, 1])},
             "bulk_spearman": {"raw_nuclei": float(stats.spearmanr(pn, b)[0]), "imputed": float(stats.spearmanr(pi, b)[0]), "model_delta_on_nucleus": float(stats.spearmanr(pr, b)[0]), "nucleus_plus_train_type_delta": float(stats.spearmanr(ps, b)[0])},
             "delta_vs_bulk": {"imputed_pearson": float(np.corrcoef(emp, d_full)[0, 1]), "imputed_spearman": float(stats.spearmanr(emp, d_full)[0]),
                               "imputed_sign_agreement": float((np.sign(emp[big]) == np.sign(d_full[big])).mean()),
                               "residual_pearson": float(np.corrcoef(emp, d_res)[0, 1]), "residual_spearman": float(stats.spearmanr(emp, d_res)[0]),
                               "residual_sign_agreement": float((np.sign(emp[big]) == np.sign(d_res[big])).mean()), "n_genes_abs_emp_gt_thr": int(big.sum()),
                               "train_delta_pearson": float(np.corrcoef(emp, d_sh)[0, 1]), "train_delta_spearman": float(stats.spearmanr(emp, d_sh)[0]),
                               "train_delta_sign_agreement": float((np.sign(emp[big]) == np.sign(d_sh[big])).mean())},
             "gene_classes": {n: {"n": len(ix), "predicted_mean_delta": float(dn_all[ix].mean()), "bulk_minus_nuclei_pseudobulk_mean": float((b - pn)[[list(np.where(present)[0]).index(i) for i in ix]].mean())} for n, ix in cls.items()}}
        res["groups"][g] = r
    keys = sorted(deltas); D = np.stack([deltas[k] for k in keys])
    res["predicted_delta_cross_group_pearson"] = pd.DataFrame(np.corrcoef(D), index=keys, columns=keys).round(3).to_dict()
    res["predicted_delta_cross_group_pearson_median_offdiag"] = float(np.median(np.corrcoef(D)[np.triu_indices(len(keys), 1)]))
    res["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(args.output_dir, f"eval_{args.name}.json"), "w"), indent=2, default=float)
    print(f"== {args.name} ({mode}); predicted-type vote agreement {pinfo['knn_vote_agreement_mean']:.3f} ==")
    print(f"{'group':<26}{'n':>6}{'lib':>8}{'r_nuc':>8}{'r_imp':>8}{'r_res':>8}{'r_trd':>8}{'rho_nuc':>9}{'rho_imp':>9}{'rho_res':>9}{'rho_trd':>9}{'dR_imp':>8}{'dR_res':>8}{'dR_trd':>8}{'sign_imp':>9}{'sign_res':>9}{'sign_trd':>9}{'ribo':>7}{'lnc':>7}")
    for g, r in res["groups"].items():
        bp, bs, dv, gc = r["bulk_pearson"], r["bulk_spearman"], r["delta_vs_bulk"], r["gene_classes"]
        print(f"{g:<26}{r['n']:>6}{r['lib_median']:>8.0f}{bp['raw_nuclei']:>8.3f}{bp['imputed']:>8.3f}{bp['model_delta_on_nucleus']:>8.3f}{bp['nucleus_plus_train_type_delta']:>8.3f}{bs['raw_nuclei']:>9.3f}{bs['imputed']:>9.3f}{bs['model_delta_on_nucleus']:>9.3f}{bs['nucleus_plus_train_type_delta']:>9.3f}"
              f"{dv['imputed_pearson']:>8.3f}{dv['residual_pearson']:>8.3f}{dv['train_delta_pearson']:>8.3f}{dv['imputed_sign_agreement']:>9.2f}{dv['residual_sign_agreement']:>9.2f}{dv['train_delta_sign_agreement']:>9.2f}{gc['ribosomal']['predicted_mean_delta']:>7.2f}{gc['nuclear_lncrna']['predicted_mean_delta']:>7.2f}")
    print(f"cross-chemistry agreement of predicted per-gene delta: median off-diagonal Pearson {res['predicted_delta_cross_group_pearson_median_offdiag']:.3f}")
    print("\npredicted-type composition (rows = group):"); print(comp.loc[:, comp.max(0) > 0.02].to_string())
    print(f"done in {res['elapsed_s']} s")


if __name__ == "__main__":
    main()
