"""
Cell-type mean baseline: assign each nucleus its cell-type mean cell expression.

This is the trivial floor baseline. Any method must beat this.
By definition: variance_ratio = 0, within_CT_spearman = 0.

Usage:
    python analysis/baseline_ctmean.py [--output_dir run/ctmean_baseline]
"""

import os
import sys
import numpy as np
import pandas as pd
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from analysis.evaluate_baseline import load_raw_data, evaluate, print_summary, DEFAULT_GENE_LIST

import json


def main():
    parser = argparse.ArgumentParser(description="Cell-type mean baseline")
    parser.add_argument("--output_dir", type=str, default="run/ctmean_baseline")
    parser.add_argument("--gene_list", type=str, default=DEFAULT_GENE_LIST)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading raw data...")
    adata_cell, adata_nuc = load_raw_data(args.gene_list)

    from scipy import sparse
    X_cell = adata_cell.X.toarray() if sparse.issparse(adata_cell.X) else adata_cell.X
    ct_cell = adata_cell.obs['cell_type'].values.astype(str)
    ct_nuc = adata_nuc.obs['cell_type'].values.astype(str)

    # Compute cell-type means from cell data
    ct_means = {}
    for ct in sorted(set(ct_cell)):
        ct_means[ct] = X_cell[ct_cell == ct].mean(axis=0)

    # Assign each nucleus its CT mean
    imputed = np.zeros_like(adata_nuc.X if not sparse.issparse(adata_nuc.X)
                            else adata_nuc.X.toarray(), dtype=np.float32)
    for i, ct in enumerate(ct_nuc):
        if ct in ct_means:
            imputed[i] = ct_means[ct]

    print(f"Imputed shape: {imputed.shape}")

    # Save
    np.save(os.path.join(args.output_dir, "imputed_cell_from_nucleus.npy"), imputed)
    obs = pd.DataFrame({'cell_type': ct_nuc}, index=adata_nuc.obs.index)
    obs.to_csv(os.path.join(args.output_dir, "nucleus_obs.csv"))

    # Evaluate
    results = evaluate(imputed, adata_cell, adata_nuc, ct_nuc, name="ctmean")
    print_summary(results)

    # Save results
    eval_dir = os.path.join(args.output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    serializable = {}
    for k, v in results.items():
        if isinstance(v, dict):
            serializable[k] = {kk: vv for kk, vv in v.items()
                              if not isinstance(vv, np.ndarray)}
        else:
            serializable[k] = v
    with open(os.path.join(eval_dir, "eval_ctmean.json"), 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {eval_dir}/eval_ctmean.json")


if __name__ == "__main__":
    main()
