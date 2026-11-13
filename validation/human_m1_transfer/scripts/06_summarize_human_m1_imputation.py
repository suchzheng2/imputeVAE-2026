#!/usr/bin/env python3
"""
Task Group 2, Part D: Pseudobulk summaries by cell class.

Compute per-group mean expression and predicted delta for broad and mouse-like classes.
"""

import os
import sys
import csv
import numpy as np
from scipy import sparse
from collections import defaultdict

ROOT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE"
INF_DIR = os.path.join(ROOT, "validation/human_m1_transfer/inference")
PROC_DIR = os.path.join(ROOT, "validation/human_m1_transfer/processed")
RESULTS_DIR = os.path.join(ROOT, "validation/human_m1_transfer/results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_data():
    """Load imputed, input, obs, and gene map."""
    print("Loading data...")
    imputed = np.load(os.path.join(INF_DIR, "human_m1_imputed_cell_log1p.npy"))
    inp_sparse = sparse.load_npz(os.path.join(INF_DIR, "human_m1_input_nucleus_log1p.npz"))
    inp = inp_sparse.toarray()
    del inp_sparse

    mask = np.load(os.path.join(PROC_DIR, "human_m1_available_gene_mask.npy"))

    with open(os.path.join(INF_DIR, "human_m1_inference_obs.csv")) as f:
        obs = list(csv.DictReader(f))

    with open(os.path.join(INF_DIR, "human_m1_inference_gene_map.csv")) as f:
        gene_map = list(csv.DictReader(f))

    print(f"  Imputed: {imputed.shape}, Input: {inp.shape}")
    print(f"  Obs: {len(obs)}, Gene map: {len(gene_map)}, Mapped: {mask.sum()}")
    return imputed, inp, mask, obs, gene_map


def compute_pseudobulk(imputed, inp, obs, gene_map, mask, group_key, donor_key=None):
    """Compute per-group, per-gene pseudobulk means and deltas.

    Returns list of dicts, one per (group, gene) pair for mapped genes only.
    """
    # Build group indices
    groups = defaultdict(list)
    for i, row in enumerate(obs):
        if donor_key:
            key = f"{row[donor_key]}|{row[group_key]}"
        else:
            key = row[group_key]
        groups[key].append(i)

    # Only mapped gene indices
    mapped_idx = np.where(mask)[0]
    # Build gene info lookup
    gene_info = {}
    for g in gene_map:
        idx = int(g["model_gene_index"])
        if mask[idx]:
            gene_info[idx] = g

    results = []
    for group_name, indices in sorted(groups.items()):
        idx_arr = np.array(indices)
        n_nuclei = len(idx_arr)
        imp_mean = imputed[idx_arr][:, mapped_idx].mean(axis=0)
        nuc_mean = inp[idx_arr][:, mapped_idx].mean(axis=0)
        delta = imp_mean - nuc_mean

        for j, gidx in enumerate(mapped_idx):
            gi = gene_info[gidx]
            row = {
                "group": group_name,
                "n_nuclei": n_nuclei,
                "model_gene_index": gidx,
                "mouse_model_gene": gi["mouse_model_gene"],
                "human_gene_symbol": gi["human_gene_symbol"],
                "nucleus_mean_log1p": f"{nuc_mean[j]:.6f}",
                "imputed_mean_log1p": f"{imp_mean[j]:.6f}",
                "predicted_delta": f"{delta[j]:.6f}",
                "filled_value_status": "mapped",
            }
            if donor_key:
                parts = group_name.split("|", 1)
                row["donor"] = parts[0]
                row["class"] = parts[1]
            results.append(row)

    return results


def save_long_csv(results, path, extra_fields=None):
    """Save long-format CSV."""
    if not results:
        return
    fields = list(results[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"  Saved: {path} ({len(results)} rows)")


def save_matrix_csvs(imputed, inp, obs, gene_map, mask, group_key, prefix):
    """Save matrix-style CSVs: nucleus mean, imputed mean, delta."""
    groups = defaultdict(list)
    for i, row in enumerate(obs):
        groups[row[group_key]].append(i)

    mapped_idx = np.where(mask)[0]
    gene_names = []
    for g in gene_map:
        idx = int(g["model_gene_index"])
        if mask[idx]:
            gene_names.append(g["mouse_model_gene"])

    group_names = sorted(groups.keys())

    # Precompute group means (vectorized) to avoid per-gene recomputation
    nuc_means = {}  # group_name -> array of shape (n_mapped,)
    imp_means = {}
    for gn in group_names:
        idx_arr = np.array(groups[gn])
        nuc_means[gn] = inp[idx_arr][:, mapped_idx].mean(axis=0)
        imp_means[gn] = imputed[idx_arr][:, mapped_idx].mean(axis=0)

    for suffix, means_dict in [("nucleus", nuc_means), ("imputed", imp_means)]:
        path = os.path.join(RESULTS_DIR, f"human_m1_{suffix}_{prefix}_mean_matrix.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["gene"] + group_names)
            for j, gname in enumerate(gene_names):
                row = [gname] + [f"{means_dict[gn][j]:.6f}" for gn in group_names]
                w.writerow(row)
        print(f"  Saved: {path}")

    # Delta
    path = os.path.join(RESULTS_DIR, f"human_m1_predicted_delta_{prefix}_matrix.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gene"] + group_names)
        for j, gname in enumerate(gene_names):
            row = [gname] + [f"{imp_means[gn][j] - nuc_means[gn][j]:.6f}" for gn in group_names]
            w.writerow(row)
    print(f"  Saved: {path}")


def main():
    imputed, inp, mask, obs, gene_map = load_data()

    # --- Long-format CSVs ---
    print("\n=== Broad class pseudobulk ===")
    broad_results = compute_pseudobulk(imputed, inp, obs, gene_map, mask, "mapped_broad_class")
    save_long_csv(broad_results,
                  os.path.join(RESULTS_DIR, "human_m1_broad_class_gene_means.csv"))

    print("\n=== Mouse-like class pseudobulk ===")
    fine_results = compute_pseudobulk(imputed, inp, obs, gene_map, mask, "mapped_mouse_like_class")
    save_long_csv(fine_results,
                  os.path.join(RESULTS_DIR, "human_m1_mouse_like_class_gene_means.csv"))

    print("\n=== Donor x broad class ===")
    donor_broad = compute_pseudobulk(imputed, inp, obs, gene_map, mask,
                                     "mapped_broad_class", donor_key="donor")
    save_long_csv(donor_broad,
                  os.path.join(RESULTS_DIR, "human_m1_donor_broad_class_gene_means.csv"))

    print("\n=== Donor x mouse-like class ===")
    donor_fine = compute_pseudobulk(imputed, inp, obs, gene_map, mask,
                                    "mapped_mouse_like_class", donor_key="donor")
    save_long_csv(donor_fine,
                  os.path.join(RESULTS_DIR, "human_m1_donor_mouse_like_class_gene_means.csv"))

    # --- Matrix-style CSVs ---
    print("\n=== Matrix-style CSVs ===")
    save_matrix_csvs(imputed, inp, obs, gene_map, mask, "mapped_broad_class", "broad_class")
    save_matrix_csvs(imputed, inp, obs, gene_map, mask, "mapped_mouse_like_class", "mouse_like_class")

    # --- Quick summary ---
    print("\n=== Quick delta summary by broad class ===")
    from collections import Counter
    broad_delta = defaultdict(list)
    for r in broad_results:
        broad_delta[r["group"]].append(float(r["predicted_delta"]))
    for cls in sorted(broad_delta.keys()):
        deltas = broad_delta[cls]
        print(f"  {cls}: mean_delta={np.mean(deltas):.4f}, median={np.median(deltas):.4f}, "
              f"pos%={100*np.mean([d > 0 for d in deltas]):.1f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
