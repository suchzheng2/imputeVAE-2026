"""
Export linear regression baseline imputed matrix as .npy for the eval pipeline.

Fits per-gene linear regression on cell-type means (nuclei -> cell),
applies to all individual nuclei, saves imputed .npy + obs CSV.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse, stats

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_raw_path(data_path):
    return os.path.join(
        "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi",
        data_path, "data.h5ad",
    )


def load_and_filter(data_path, batch, gene_list_path):
    adata = ad.read_h5ad(_resolve_raw_path(data_path))
    if "meta_batch_id" not in adata.obs:
        adata.obs["meta_batch_id"] = adata.obs["batch_id"].apply(lambda x: x.split(".")[0])

    # Filter rows and columns on the view BEFORE .copy() to avoid
    # materialising the full (193K x 26K) dense matrix (39 GiB as int64).
    batch_mask = adata.obs["meta_batch_id"].str.endswith(batch)
    common_genes = pd.read_csv(os.path.join(PROJECT, "data/common_genes.csv"), header=0)["common_genes"].tolist()
    common_ct = pd.read_csv(os.path.join(PROJECT, "data/common_cell_types.csv"), header=0)["common_cell_types"].tolist()
    gene_list = pd.read_csv(gene_list_path, header=None)[0].tolist()

    gene_mask = adata.var["gene_names"].isin(common_genes) & adata.var["gene_names"].isin(gene_list)
    ct_mask = adata.obs["cell_type"].isin(common_ct)
    adata = adata[batch_mask & ct_mask, gene_mask].copy()
    adata = adata[:, adata.var["gene_names"].argsort()].copy()

    X = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
    adata.X = np.log1p(X).astype(np.float32)
    print(f"  Loaded {data_path} batch={batch}: {adata.shape}")
    return adata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene_list", default="data/training_gene_list_v3.csv")
    parser.add_argument("--output_dir", default="run/linreg_baseline")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
    cell_adata = load_and_filter("10X_cells_v2_v3_AIBS/", "c3A", args.gene_list)
    nuc_adata = load_and_filter("10X_nuclei_v2_v3_AIBS/", "n3A", args.gene_list)

    X_cell = cell_adata.X if isinstance(cell_adata.X, np.ndarray) else cell_adata.X.toarray()
    X_nuc = nuc_adata.X if isinstance(nuc_adata.X, np.ndarray) else nuc_adata.X.toarray()
    ct_cell = cell_adata.obs["cell_type"].values
    ct_nuc = nuc_adata.obs["cell_type"].values
    gene_names = cell_adata.var["gene_names"].values
    n_genes = X_cell.shape[1]

    common_cts = sorted(set(ct_cell) & set(ct_nuc))
    print(f"Common cell types: {len(common_cts)}")

    # Compute cell-type means
    print("Computing cell-type means...")
    cell_means = {}
    nuc_means = {}
    for ct in common_cts:
        cell_means[ct] = X_cell[ct_cell == ct].mean(axis=0)
        nuc_means[ct] = X_nuc[ct_nuc == ct].mean(axis=0)

    cell_mean_matrix = np.array([cell_means[ct] for ct in common_cts])
    nuc_mean_matrix = np.array([nuc_means[ct] for ct in common_cts])

    # Fit per-gene linear regression on cell-type means
    print(f"Fitting per-gene linear regression on {len(common_cts)} cell types...")
    slopes = np.zeros(n_genes)
    intercepts = np.zeros(n_genes)

    for g in range(n_genes):
        x = nuc_mean_matrix[:, g]
        y = cell_mean_matrix[:, g]
        if np.std(x) > 1e-10:
            slope, intercept, _, _, _ = stats.linregress(x, y)
            slopes[g] = slope
            intercepts[g] = intercept
        else:
            slopes[g] = 0.0
            intercepts[g] = y.mean()

    # Apply to all nuclei
    print(f"Applying to {X_nuc.shape[0]} nuclei...")
    imputed = (X_nuc * slopes[np.newaxis, :] + intercepts[np.newaxis, :]).astype(np.float32)
    imputed = np.clip(imputed, 0, None)

    print(f"Imputed stats: min={imputed.min():.3f}, max={imputed.max():.3f}, "
          f"mean={imputed.mean():.3f}, nonzero={100*(imputed > 0).mean():.1f}%")

    # Save
    np.save(os.path.join(args.output_dir, "imputed_cell_from_nucleus.npy"), imputed)
    print(f"Saved imputed: {os.path.join(args.output_dir, 'imputed_cell_from_nucleus.npy')} {imputed.shape}")

    obs_df = nuc_adata.obs[["cell_type"]].copy()
    if "meta_batch_id" in nuc_adata.obs.columns:
        obs_df["meta_batch_id"] = nuc_adata.obs["meta_batch_id"]
    obs_df.to_csv(os.path.join(args.output_dir, "nucleus_obs.csv"))
    print(f"Saved obs: {os.path.join(args.output_dir, 'nucleus_obs.csv')}")

    # Save coefficients
    coeff_df = pd.DataFrame({"gene": gene_names, "slope": slopes, "intercept": intercepts})
    coeff_df.to_csv(os.path.join(args.output_dir, "regression_coefficients.csv"), index=False)
    print(f"Saved coefficients: {os.path.join(args.output_dir, 'regression_coefficients.csv')}")


if __name__ == "__main__":
    main()
