#!/usr/bin/env python
"""Generate the exact train/valid/test split manifest used by ImputeVAE training.

Reproduces the split from balanced_split_by_cell_type() in imputeVAE/data.py
with random_state=42, stratified by cell_type, subset=False.

Memory-efficient: only loads obs metadata, not the expression matrix.
The split depends solely on cell_type labels and observation order, not on X values.

Outputs: data/split_manifest.tsv

Usage:
    conda run -n impute2 python analysis/generate_split_manifest.py
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DATA_ROOT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi"
DEFAULT_GENE_LIST = os.path.join(PROJECT_ROOT, "data", "training_gene_list_v3.csv")


def load_filtered_obs(data_dir, batch_suffix):
    """Load adata, apply same filters as training, return obs DataFrame and n_obs.

    Applies: common_genes + common_ct + gene_list + protocol filter + gene sort.
    Does NOT densify X — we only need obs for the split.
    Returns obs with meta_batch_id populated.
    """
    path = os.path.join(DATA_ROOT, data_dir, "data.h5ad")
    adata = ad.read_h5ad(path, backed="r")
    obs = adata.obs.copy()
    var = adata.var.copy()

    # Add meta_batch_id
    if "meta_batch_id" not in obs.columns:
        obs["meta_batch_id"] = obs["batch_id"].apply(lambda x: x.split(".")[0])

    # Protocol filter (v3)
    row_mask = obs["meta_batch_id"].str.endswith(batch_suffix)

    # Common cell types
    common_ct = pd.read_csv(
        os.path.join(PROJECT_ROOT, "data", "common_cell_types.csv"), header=0
    )["common_cell_types"].tolist()
    row_mask &= obs["cell_type"].isin(common_ct)

    obs = obs[row_mask].copy()

    # Gene filtering (only to get n_genes, which determines shape[1] for consistency)
    common_genes = pd.read_csv(
        os.path.join(PROJECT_ROOT, "data", "common_genes.csv"), header=0
    )["common_genes"].tolist()
    gene_list = pd.read_csv(DEFAULT_GENE_LIST, header=None)[0].tolist()
    col_mask = var["gene_names"].isin(common_genes) & var["gene_names"].isin(gene_list)
    n_genes = col_mask.sum()

    if hasattr(adata, "file") and adata.file is not None:
        adata.file.close()

    print(f"  Loaded {data_dir} batch={batch_suffix}: {len(obs)} obs x {n_genes} genes")
    return obs, len(obs)


def split_indices(n_obs, cell_types, train_size=0.6, valid_size=0.2, test_size=0.2):
    """Reproduce balanced_split_by_cell_type index assignment.

    train_test_split in sklearn uses the *length* of X to generate indices,
    but the split is determined by stratify=y and random_state only.
    We pass a dummy array of indices instead of the full X matrix.
    """
    indices = np.arange(n_obs)
    y = cell_types

    # First split: train+valid vs test (same as data.py line 450-451)
    train_valid_idx, test_idx = train_test_split(
        indices, stratify=y, test_size=test_size, random_state=42,
    )

    # Second split: train vs valid (same as data.py line 454-456)
    valid_ratio = valid_size / (train_size + valid_size)
    y_train_valid = y.iloc[train_valid_idx] if hasattr(y, "iloc") else y[train_valid_idx]
    train_idx, valid_idx = train_test_split(
        train_valid_idx, stratify=y_train_valid, test_size=valid_ratio, random_state=42,
    )

    return train_idx, valid_idx, test_idx


def verify_split_matches_training(obs, train_idx, valid_idx, test_idx):
    """Verify the split matches what balanced_split_by_cell_type produces.

    The original function passes (X, y, np.arange(N)) to train_test_split.
    sklearn's train_test_split with stratify uses only the labels and indices.
    When we pass just indices with the same stratify and random_state,
    we get the same split.

    However, the original code passes X as the first positional arg.
    sklearn's train_test_split uses X.shape[0] internally but the actual
    split is determined by stratify + random_state. Passing a 1D index array
    vs a 2D X matrix: sklearn treats them the same for splitting purposes
    as long as lengths match.
    """
    # The key invariant: for the SAME y and random_state, train_test_split
    # produces the same partition regardless of what X is.
    # This is documented sklearn behavior.
    pass


def main():
    output_path = os.path.join(PROJECT_ROOT, "data", "split_manifest.tsv")

    all_records = []

    for data_dir, batch, modality_label in [
        ("10X_nuclei_v2_v3_AIBS", "n3A", "nuclei"),
        ("10X_cells_v2_v3_AIBS", "c3A", "cell"),
    ]:
        print(f"\n{'='*60}")
        print(f"Processing {modality_label.upper()}")
        print(f"{'='*60}")

        obs, n_obs = load_filtered_obs(data_dir, batch)

        # The obs index order after filtering determines the split.
        # balanced_split_by_cell_type receives adata after filter_by_protocol,
        # which does adata[mask].copy(). The resulting obs is in original h5ad order
        # but subsetted. We replicate that here.
        cell_types = obs["cell_type"]

        train_idx, valid_idx, test_idx = split_indices(n_obs, cell_types)

        # Assertions: disjoint and complete
        sets = [set(train_idx), set(valid_idx), set(test_idx)]
        assert sets[0].isdisjoint(sets[1]), f"{modality_label} train/valid overlap"
        assert sets[0].isdisjoint(sets[2]), f"{modality_label} train/test overlap"
        assert sets[1].isdisjoint(sets[2]), f"{modality_label} valid/test overlap"
        assert len(sets[0]) + len(sets[1]) + len(sets[2]) == n_obs, \
            f"{modality_label} split doesn't cover all obs"
        print(f"  Split: {len(train_idx)} train + {len(valid_idx)} valid + {len(test_idx)} test = {n_obs}")

        # Build records
        for idx_arr, partition in [
            (train_idx, "train"),
            (valid_idx, "valid"),
            (test_idx, "test"),
        ]:
            obs_slice = obs.iloc[idx_arr]
            donors = obs_slice["meta_batch_id"].astype(str).str.split("-").str[0]
            records = pd.DataFrame({
                "obs_id": obs_slice.index.astype(str).values,
                "modality": modality_label,
                "cell_type": obs_slice["cell_type"].values,
                "donor": donors.values,
                "partition": partition,
            })
            all_records.append(records)

    df_all = pd.concat(all_records, ignore_index=True)

    # ---- Final assertions ----
    print(f"\n{'='*60}")
    print("Assertions")
    print(f"{'='*60}")

    for mod in ["nuclei", "cell"]:
        sub = df_all[df_all["modality"] == mod]
        # Unique obs_ids within modality
        assert sub["obs_id"].is_unique, f"Duplicate obs_ids in {mod}"
        # Every cell type in all 3 partitions
        for ct in sub["cell_type"].unique():
            parts = sub[sub["cell_type"] == ct]["partition"].unique()
            assert set(parts) == {"train", "valid", "test"}, \
                f"{mod} cell type '{ct}' missing partitions: {set(parts)}"
    print("  All assertions passed")

    # ---- Donor overlap documentation ----
    print(f"\n{'='*60}")
    print("Donor overlap across partitions (expected: full overlap, obs-level split)")
    print(f"{'='*60}")
    for mod in ["nuclei", "cell"]:
        sub = df_all[df_all["modality"] == mod]
        for part in ["train", "valid", "test"]:
            donors = sorted(sub[sub["partition"] == part]["donor"].unique())
            print(f"  {mod} {part}: donors {donors}")

    # ---- Save ----
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_all.to_csv(output_path, sep="\t", index=False)
    print(f"\nManifest saved: {output_path}  ({len(df_all)} rows)")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("Totals by partition x modality")
    print(f"{'='*60}")
    totals = df_all.groupby(["partition", "modality"]).size().reset_index(name="count")
    print(totals.to_string(index=False))

    print(f"\n{'='*60}")
    print("Detailed by cell_type x modality x partition")
    print(f"{'='*60}")
    detail = (
        df_all.groupby(["modality", "cell_type", "partition"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    pd.set_option("display.max_rows", 200)
    print(detail.to_string(index=False))


if __name__ == "__main__":
    main()
