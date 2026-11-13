"""
Per-gene linear regression baseline, fitted on TRAINING cell-type means.

    cell_mean[ct, g] ~ slope[g] * nuc_mean[ct, g] + intercept[g]      (one fit per gene, K cell types)
    imputed[n, g]    = slope[g] * nuc[n, g] + intercept[g]            (applied to each eval nucleus)

Two variants are written:
  linreg          plain OLS (as benchmarked before).  Slopes > 1 amplify per-nucleus noise.
  linreg_shrunk   cell-type-level prediction identical to linreg (slope*nuc_mean_train[ct] + intercept),
                  but the per-nucleus deviation from its training type mean is added at unit gain:
                      imputed = slope*nuc_mean_train[ct] + intercept + (nuc - nuc_mean_train[ct])
                  This removes the variance amplification of slopes > 1 and is the fair
                  "simple linear" competitor on per-cell metrics.

Usage:
    python -m analysis.benchmark.train_linreg --eval_partition test --output_root run/benchmark/test
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from analysis.benchmark.data_split import load_benchmark_data, save_standard_outputs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_partition", default="test", choices=["valid", "test"])
    p.add_argument("--output_root", default="run/benchmark/test")
    args = p.parse_args()

    print("=" * 60)
    print("Split-aware benchmark: per-gene linear regression")
    print("=" * 60)
    data = load_benchmark_data(args.eval_partition, return_raw_counts=False, include_eval_cells=False)
    gene_names = data["gene_names"]
    common_cts = data["common_cts"]
    n_genes = len(gene_names)

    X_cell_train = data["cell_train"].X
    X_nuc_train = data["nuc_train"].X
    ct_cell_train = data["cell_train"].obs["cell_type"].values.astype(str)
    ct_nuc_train = data["nuc_train"].obs["cell_type"].values.astype(str)
    X_nuc_eval = data["nuc_eval"].X
    ct_nuc_eval = data["nuc_eval"].obs["cell_type"].values.astype(str)
    nuc_eval_obs = data["nuc_eval"].obs[["obs_id", "cell_type", "donor"]].copy()

    t0 = time.time()
    cts = [ct for ct in common_cts if (ct_cell_train == ct).sum() > 0 and (ct_nuc_train == ct).sum() > 0]
    C = np.array([X_cell_train[ct_cell_train == ct].mean(axis=0) for ct in cts])   # [K, G]
    N = np.array([X_nuc_train[ct_nuc_train == ct].mean(axis=0) for ct in cts])     # [K, G]
    nuc_mean_train = {ct: N[k] for k, ct in enumerate(cts)}

    # Closed-form per-gene OLS (vectorised over genes)
    xm, ym = N.mean(axis=0), C.mean(axis=0)
    sxx = ((N - xm) ** 2).sum(axis=0)
    sxy = ((N - xm) * (C - ym)).sum(axis=0)
    slopes = np.where(sxx > 1e-10, sxy / np.maximum(sxx, 1e-10), 0.0)
    intercepts = ym - slopes * xm
    syy = ((C - ym) ** 2).sum(axis=0)
    r_values = np.where((sxx > 1e-10) & (syy > 1e-10), sxy / np.sqrt(np.maximum(sxx * syy, 1e-20)), 0.0)
    elapsed = time.time() - t0
    print(f"Fitted {n_genes} genes on {len(cts)} cell-type means in {elapsed:.1f}s; "
          f"median r {np.median(r_values):.3f}; median slope {np.median(slopes):.3f}; "
          f"slopes > 1: {(slopes > 1).mean():.1%}")

    # Plain OLS
    imputed = np.clip(X_nuc_eval * slopes[None, :] + intercepts[None, :], 0, None).astype(np.float32)

    # Shrunk variant: type-level prediction via training nucleus mean, per-nucleus deviation at unit gain
    imputed_shrunk = np.empty_like(X_nuc_eval)
    for ct in sorted(set(ct_nuc_eval)):
        m = ct_nuc_eval == ct
        base = nuc_mean_train.get(ct)
        if base is None:
            imputed_shrunk[m] = imputed[m]
            continue
        type_pred = slopes * base + intercepts
        imputed_shrunk[m] = type_pred[None, :] + (X_nuc_eval[m] - base[None, :])
    imputed_shrunk = np.clip(imputed_shrunk, 0, None).astype(np.float32)

    out = os.path.join(args.output_root, "linreg")
    os.makedirs(out, exist_ok=True)
    pd.DataFrame({"gene": gene_names, "slope": slopes, "intercept": intercepts, "r_value": r_values}) \
        .to_csv(os.path.join(out, "regression_coefficients.csv"), index=False)
    base_args = {
        "approach": "per-gene OLS on training cell-type means, applied per nucleus",
        "n_ct_anchor_points": len(cts), "n_genes": n_genes, "n_params": 2 * n_genes,
        "training_time_s": elapsed, "median_r_fit": float(np.median(r_values)),
        "checkpoint": "closed-form", "eval_partition": args.eval_partition,
    }
    save_standard_outputs(out, imputed, nuc_eval_obs, gene_names,
                          {"method": "Linear_regression", **base_args}, data["split_info"])
    save_standard_outputs(os.path.join(args.output_root, "linreg_shrunk"), imputed_shrunk, nuc_eval_obs,
                          gene_names, {"method": "Linear_regression_shrunk",
                                       "variant": "unit-gain per-nucleus deviation", **base_args},
                          data["split_info"])
    print("Done.")


if __name__ == "__main__":
    main()
