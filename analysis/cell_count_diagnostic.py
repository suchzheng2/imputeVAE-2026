"""Cell count diagnostic: how many cells per cell type per modality at each subset level.

Memory-efficient: loads only obs metadata, not expression matrices.

Usage:
    python analysis/cell_count_diagnostic.py [--subset_ratios 0.2,0.6,1.0]
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import anndata as ad

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


DATA_ROOT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi"
GENE_LIST = "data/training_gene_list_v3.csv"


def load_obs_only(data_path):
    """Load just the obs DataFrame from h5ad without reading X."""
    full_path = os.path.join(DATA_ROOT, data_path, "data.h5ad")
    adata = ad.read_h5ad(full_path, backed="r")
    obs = adata.obs.copy()
    if hasattr(adata, "file") and adata.file is not None:
        adata.file.close()
    return obs


def filter_obs_protocol_v3(obs, modality):
    """Filter obs to v3 protocol (same logic as filter_by_protocol with protocol=v3)."""
    if "meta_batch_id" not in obs.columns:
        obs = obs.copy()
        obs["meta_batch_id"] = obs["batch_id"].apply(lambda x: x.split('.')[0])

    modality_char = "c" if modality == "cell" else "n"
    sample_char = "A"
    target = f"{modality_char}3{sample_char}"
    return obs[obs["meta_batch_id"].str.endswith(target)].copy()


def simulate_subset_split(obs, subset_ratio, min_samples=50):
    """Simulate balanced_split_by_cell_type to get counts without loading X.

    Uses same random_state=42 and logic as data.py.
    """
    from sklearn.model_selection import train_test_split

    y = obs["cell_type"]

    if subset_ratio < 1.0:
        indices_by_celltype = {}
        for cell_type in y.unique():
            mask = y == cell_type
            cell_indices = np.where(mask)[0]
            subset_size = max(int(len(cell_indices) * subset_ratio), min(min_samples, len(cell_indices)))
            subset_indices = np.random.RandomState(42).choice(cell_indices, subset_size, replace=False)
            indices_by_celltype[cell_type] = subset_indices

        selected_indices = np.concatenate(list(indices_by_celltype.values()))
        obs_subset = obs.iloc[selected_indices]
        y_subset = y.iloc[selected_indices]
    else:
        obs_subset = obs
        y_subset = y

    # Split: 60/20/20, stratified, random_state=42
    idx = np.arange(len(obs_subset))
    _, _, _, _, train_valid_idx, test_idx = train_test_split(
        idx, y_subset, idx, stratify=y_subset, test_size=0.2, random_state=42
    )
    y_tv = y_subset.iloc[train_valid_idx]
    valid_ratio = 0.2 / 0.8
    _, _, _, _, train_idx, valid_idx = train_test_split(
        train_valid_idx, y_tv, train_valid_idx, stratify=y_tv, test_size=valid_ratio, random_state=42
    )

    counts = {}
    for split_name, split_idx in [("train", train_idx), ("valid", valid_idx), ("test", test_idx)]:
        split_obs = obs_subset.iloc[split_idx]
        for ct, cnt in split_obs["cell_type"].value_counts().items():
            counts[(ct, split_name)] = int(cnt)

    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset_ratios", type=str, default="0.2,1.0")
    args = parser.parse_args()

    ratios = [float(r) for r in args.subset_ratios.split(",")]

    # Load obs metadata only (very lightweight)
    print("Loading obs metadata (no expression data)...")
    obs_cell = load_obs_only("10X_cells_v2_v3_AIBS/")
    obs_nuc = load_obs_only("10X_nuclei_v2_v3_AIBS/")

    # Filter to v3 protocol
    obs_cell = filter_obs_protocol_v3(obs_cell, "cell")
    obs_nuc = filter_obs_protocol_v3(obs_nuc, "nuclei")

    print(f"After v3 filter: {len(obs_cell)} cells, {len(obs_nuc)} nuclei")

    all_rows = []
    for ratio in ratios:
        print(f"\n{'='*70}")
        print(f"Subset ratio: {ratio}")
        print(f"{'='*70}")

        cell_counts = simulate_subset_split(obs_cell, ratio)
        nuc_counts = simulate_subset_split(obs_nuc, ratio)

        all_cts = sorted(set(
            [k[0] for k in cell_counts.keys()] + [k[0] for k in nuc_counts.keys()]
        ))

        print(f"\n{'Cell Type':<20} {'C_train':>8} {'C_valid':>8} {'C_test':>8} "
              f"{'N_train':>8} {'N_valid':>8} {'N_test':>8}")
        print("-" * 80)

        for ct in all_cts:
            row = {"cell_type": ct, "subset_ratio": ratio}
            for split in ["train", "valid", "test"]:
                row[f"cell_{split}"] = cell_counts.get((ct, split), 0)
                row[f"nuclei_{split}"] = nuc_counts.get((ct, split), 0)
            all_rows.append(row)

            print(f"{ct:<20} {row['cell_train']:>8} {row['cell_valid']:>8} {row['cell_test']:>8} "
                  f"{row['nuclei_train']:>8} {row['nuclei_valid']:>8} {row['nuclei_test']:>8}")

        # Totals
        df_ratio = pd.DataFrame([r for r in all_rows if r["subset_ratio"] == ratio])
        totals = df_ratio[["cell_train", "cell_valid", "nuclei_train", "nuclei_valid"]].sum()
        print("-" * 80)
        print(f"{'TOTAL':<20} {int(totals['cell_train']):>8} {int(totals['cell_valid']):>8} "
              f"{'':>8} {int(totals['nuclei_train']):>8} {int(totals['nuclei_valid']):>8}")

        # Flag low-count cell types
        low_ct = df_ratio[(df_ratio["cell_train"] < 50) | (df_ratio["nuclei_train"] < 50)]
        if not low_ct.empty:
            print(f"\nWARNING: {len(low_ct)} cell types with <50 training samples:")
            for _, r in low_ct.iterrows():
                print(f"  {r['cell_type']}: cell={r['cell_train']}, nuclei={r['nuclei_train']}")

    # Save
    output_dir = "notes/experiments/cell_count_diagnostic"
    os.makedirs(output_dir, exist_ok=True)
    combined = pd.DataFrame(all_rows)
    combined.to_csv(os.path.join(output_dir, "cell_counts.csv"), index=False)
    print(f"\nSaved to {output_dir}/cell_counts.csv")


if __name__ == "__main__":
    main()
