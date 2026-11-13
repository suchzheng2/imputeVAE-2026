#!/usr/bin/env python3
"""
Cross-decoded expression heatmaps.

Generates heatmaps for cross_decoded (nuclei -> cell space) data alongside
raw cell and raw nuclei references, plus difference heatmaps.

Usage:
    python eval_cross_decoded_heatmaps.py --run_folder <name> --epochs 0 100 300 599
"""

import os
import sys
import gc
import json
import argparse
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from eval_utils import (
    find_run_folder, get_eval_dir, load_run_args,
    find_epoch_dump, load_common_filters, load_raw_data,
    stratified_subsample, compute_target_counts,
    MAX_CELLS_PER_TYPE, RANDOM_SEED,
    RAW_CELL_PATH, RAW_NUCLEI_PATH,
)
from eval_heatmaps import plot_expression_heatmap, plot_difference_heatmap
from scipy import sparse

import matplotlib
matplotlib.use('Agg')


def load_cross_decoded(train_fp, common_cell_types):
    """Load cross_decoded arrays from a joint epoch dump.

    Returns (cross_decoded, cell_types).
    """
    print(f"  Loading cross_decoded from: {train_fp}")
    with open(train_fp, 'r') as f:
        data = json.load(f)

    xdec_list, types_list = [], []
    for ct in sorted(data.keys()):
        if ct not in common_cell_types:
            continue
        for bid in data[ct]:
            bd = data[ct][bid]
            if 'cross_decoded' not in bd:
                raise KeyError(f"No 'cross_decoded' key in epoch dump for {ct}/{bid}")
            arr = np.array(bd['cross_decoded'], dtype=np.float32)
            xdec_list.append(arr)
            types_list.extend([ct] * arr.shape[0])

    del data
    gc.collect()

    xdec_arr = np.vstack(xdec_list)
    types_arr = np.array(types_list)
    print(f"    Cross-decoded shape: {xdec_arr.shape}, {len(np.unique(types_arr))} cell types")
    return xdec_arr, types_arr


def run_cross_decoded_heatmaps(run_folder, epoch,
                                raw_cell_adata=None, raw_nuclei_adata=None):
    """Generate cross-decoded heatmaps for one epoch."""
    eval_dir = get_eval_dir(run_folder)
    heatmap_dir = os.path.join(eval_dir, 'heatmaps_cross_decoded', f'epoch_{epoch}')
    os.makedirs(heatmap_dir, exist_ok=True)

    run_args = load_run_args(run_folder)
    use_log1p = run_args.get('raw_data_normalization', 'raw') == 'log1p'
    det_rate = float(run_args.get('detection_rate_threshold', 1.0))
    exclude_mito = run_args.get('exclude_mitocarta', True)
    norm_label = "log1p" if use_log1p else "raw"

    fp = find_epoch_dump(run_folder, epoch)
    if fp is None:
        print(f"  ERROR: epoch dump not found for epoch {epoch}")
        return {'status': 'failed', 'reason': f'epoch dump not found for epoch {epoch}'}

    common_genes, common_cell_types, mitocarta_genes = load_common_filters()
    protocol = run_args.get('protocol', None)
    batch = run_args.get('batch', None)

    # Load cross-decoded
    xdec, xdec_types = load_cross_decoded(fp, common_cell_types)

    # Load raw data
    if raw_cell_adata is None:
        print("  Loading raw cell data...")
        raw_cell_adata = load_raw_data(RAW_CELL_PATH, common_genes, common_cell_types,
                                        mitocarta_genes, det_rate, exclude_mito,
                                        protocol=protocol, batch=batch)
    if raw_nuclei_adata is None:
        print("  Loading raw nuclei data...")
        raw_nuclei_adata = load_raw_data(RAW_NUCLEI_PATH, common_genes, common_cell_types,
                                          mitocarta_genes, det_rate, exclude_mito,
                                          protocol=protocol, batch=batch)

    raw_cell_ct = raw_cell_adata.obs['cell_type'].values
    raw_nuclei_ct = raw_nuclei_adata.obs['cell_type'].values

    # Compute balanced target counts across all 3 sources
    tc = compute_target_counts([xdec_types, raw_cell_ct, raw_nuclei_ct], MAX_CELLS_PER_TYPE)
    print(f"  Target counts: {len(tc)} cell types, {sum(tc.values())} total")

    # Subsample
    xdec_sub, xdec_ct = stratified_subsample(xdec, xdec_types, seed=RANDOM_SEED, target_counts=tc)
    del xdec; gc.collect()

    raw_cell_X = raw_cell_adata.X
    if sparse.issparse(raw_cell_X):
        raw_cell_X = raw_cell_X.toarray()
    rc_sub, rc_types = stratified_subsample(raw_cell_X, raw_cell_ct, seed=RANDOM_SEED, target_counts=tc)
    rc_ref = np.log1p(rc_sub) if use_log1p else rc_sub.astype(np.float32)

    raw_nuclei_X = raw_nuclei_adata.X
    if sparse.issparse(raw_nuclei_X):
        raw_nuclei_X = raw_nuclei_X.toarray()
    rn_sub, rn_types = stratified_subsample(raw_nuclei_X, raw_nuclei_ct, seed=RANDOM_SEED, target_counts=tc)
    rn_ref = np.log1p(rn_sub) if use_log1p else rn_sub.astype(np.float32)

    print(f"  Shapes: cross_decoded={xdec_sub.shape}, raw_cell={rc_sub.shape}, raw_nuclei={rn_sub.shape}")

    # Common color scale across all 3
    all_vals = np.concatenate([xdec_sub.flatten(), rc_ref.flatten(), rn_ref.flatten()])
    vmin = float(np.percentile(all_vals, 1))
    vmax = float(np.percentile(all_vals, 99))
    del all_vals

    # --- Expression heatmaps ---
    plot_expression_heatmap(xdec_sub, xdec_ct, f"Cross-Decoded (nuclei→cell, epoch {epoch})",
                            os.path.join(heatmap_dir, "cross_decoded_heatmap.png"), vmin, vmax)
    plot_expression_heatmap(rc_ref, rc_types, f"Raw Cell ({norm_label})",
                            os.path.join(heatmap_dir, f"raw_cell_{norm_label}_heatmap.png"), vmin, vmax)
    plot_expression_heatmap(rn_ref, rn_types, f"Raw Nuclei ({norm_label})",
                            os.path.join(heatmap_dir, f"raw_nuclei_{norm_label}_heatmap.png"), vmin, vmax)

    # --- Difference heatmaps ---
    # cross_decoded vs raw_cell (the target we're trying to match)
    plot_difference_heatmap(xdec_sub, rc_ref, xdec_ct,
                            f"Cross-Decoded vs Raw Cell (epoch {epoch})",
                            os.path.join(heatmap_dir, f"diff_cross_decoded_vs_raw_cell_{norm_label}.png"),
                            "cross_decoded", f"raw_cell ({norm_label})")

    # cross_decoded vs raw_nuclei (the source modality)
    plot_difference_heatmap(xdec_sub, rn_ref, xdec_ct,
                            f"Cross-Decoded vs Raw Nuclei (epoch {epoch})",
                            os.path.join(heatmap_dir, f"diff_cross_decoded_vs_raw_nuclei_{norm_label}.png"),
                            "cross_decoded", f"raw_nuclei ({norm_label})")

    # raw_cell vs raw_nuclei (reference gap)
    plot_difference_heatmap(rc_ref, rn_ref, rc_types,
                            f"Raw Cell vs Raw Nuclei",
                            os.path.join(heatmap_dir, f"diff_raw_cell_vs_raw_nuclei_{norm_label}.png"),
                            f"raw_cell ({norm_label})", f"raw_nuclei ({norm_label})")

    del xdec_sub, rc_sub, rn_sub, rc_ref, rn_ref
    gc.collect()

    print(f"  Saved 6 heatmaps to {heatmap_dir}")
    return {'status': 'success', 'epoch': epoch, 'output_dir': heatmap_dir}


def main():
    parser = argparse.ArgumentParser(description="Cross-decoded expression heatmaps")
    parser.add_argument("--run_folder", required=True, help="Run folder name or path")
    parser.add_argument("--epochs", nargs='+', type=int, required=True,
                        help="Epochs to generate heatmaps for")
    args = parser.parse_args()

    run_folder = find_run_folder(args.run_folder)
    print(f"Run folder: {run_folder}")
    print(f"Epochs: {args.epochs}")

    # Load raw data once, reuse across epochs
    from eval_utils import load_common_filters, load_raw_data
    run_args = load_run_args(run_folder)
    common_genes, common_cell_types, mitocarta_genes = load_common_filters()
    det_rate = float(run_args.get('detection_rate_threshold', 1.0))
    exclude_mito = run_args.get('exclude_mitocarta', True)
    protocol = run_args.get('protocol', None)
    batch = run_args.get('batch', None)

    print("Loading raw cell data...")
    raw_cell_adata = load_raw_data(RAW_CELL_PATH, common_genes, common_cell_types,
                                    mitocarta_genes, det_rate, exclude_mito,
                                    protocol=protocol, batch=batch)
    print("Loading raw nuclei data...")
    raw_nuclei_adata = load_raw_data(RAW_NUCLEI_PATH, common_genes, common_cell_types,
                                      mitocarta_genes, det_rate, exclude_mito,
                                      protocol=protocol, batch=batch)

    for epoch in args.epochs:
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}")
        print(f"{'='*60}")
        result = run_cross_decoded_heatmaps(run_folder, epoch,
                                             raw_cell_adata=raw_cell_adata,
                                             raw_nuclei_adata=raw_nuclei_adata)
        print(f"  Result: {result}")

    print("\nDone!")


if __name__ == '__main__':
    main()
