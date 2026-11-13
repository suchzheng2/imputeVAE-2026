"""
Trivial reference methods for the split-aware benchmark.

  ctmean      Each eval nucleus receives the mean log1p profile of TRAINING cells of its
              cell type.  Zero within-type variance by construction.  On cell-type-level
              metrics this is the sampling-noise CEILING (train-cell means vs eval-cell
              means), not a trivially perfect score as in the old all-data evaluation.
  raw_nuclei  No-op: the eval nucleus itself.  Lower bound.

Usage:
    python -m analysis.benchmark.train_ctmean --eval_partition test --output_root run/benchmark/test
"""

import os
import sys
import argparse
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from analysis.benchmark.data_split import load_benchmark_data, save_standard_outputs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_partition", default="test", choices=["valid", "test"])
    p.add_argument("--output_root", default="run/benchmark/test")
    args = p.parse_args()

    print("=" * 60)
    print("Split-aware benchmark: CT-mean and raw-nuclei references")
    print("=" * 60)
    data = load_benchmark_data(args.eval_partition, return_raw_counts=False, include_eval_cells=False)
    gene_names = data["gene_names"]
    common_cts = data["common_cts"]
    X_cell_train = data["cell_train"].X
    ct_cell_train = data["cell_train"].obs["cell_type"].values.astype(str)
    X_nuc_eval = data["nuc_eval"].X
    ct_nuc_eval = data["nuc_eval"].obs["cell_type"].values.astype(str)
    nuc_eval_obs = data["nuc_eval"].obs[["obs_id", "cell_type", "donor"]].copy()

    # ---- CT-mean ----
    t0 = time.time()
    ct_means = {ct: X_cell_train[ct_cell_train == ct].mean(axis=0) for ct in common_cts
                if (ct_cell_train == ct).sum() > 0}
    imputed = np.zeros_like(X_nuc_eval)
    missing = 0
    for i, ct in enumerate(ct_nuc_eval):
        if ct in ct_means:
            imputed[i] = ct_means[ct]
        else:
            missing += 1
    elapsed = time.time() - t0
    for ct in common_cts:
        m = ct_nuc_eval == ct
        if m.sum() > 1:
            assert np.allclose(imputed[m].var(axis=0), 0.0, atol=1e-6), f"CT-mean variance not ~0 for {ct}"
    print(f"CT-mean: {len(ct_means)} cell types, {missing} nuclei without a training cell type")
    save_standard_outputs(
        os.path.join(args.output_root, "ctmean"), imputed, nuc_eval_obs, gene_names,
        {"method": "CT_mean", "approach": "mean log1p profile of training cells per cell type",
         "n_params": 0, "training_time_s": elapsed, "checkpoint": "N/A",
         "n_training_cells": int(X_cell_train.shape[0]), "eval_partition": args.eval_partition},
        data["split_info"])

    # ---- raw nuclei ----
    save_standard_outputs(
        os.path.join(args.output_root, "raw_nuclei"), X_nuc_eval, nuc_eval_obs, gene_names,
        {"method": "raw_nuclei", "approach": "identity (no imputation)", "n_params": 0,
         "training_time_s": 0.0, "checkpoint": "N/A", "eval_partition": args.eval_partition},
        data["split_info"])
    print("Done.")


if __name__ == "__main__":
    main()
