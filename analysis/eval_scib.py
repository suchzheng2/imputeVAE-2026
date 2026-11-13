#!/usr/bin/env python3
"""
iLISI/cLISI Metrics module (SEPARATE ENV).

Standalone script that runs under `abc_env` (which has scib=1.1.5).
The orchestrator calls this via subprocess.

Modes:
  single       -- original single-modality scIB (backward compat)
  stage3_full  -- per-modality + cross-modality + modality-separability

Usage:
    conda run -n abc_env python eval_scib.py --run_folder <name> --epoch <N> \
        [--output_dir <path>] [--mode single|stage3_full]
"""

import os
import sys
import gc
import json
import argparse
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
from eval_utils import (
    load_run_args, infer_target_modality, resolve_batch_id_list,
    load_both_encoded_stage3,
)

try:
    import scib
    from scib.metrics import silhouette_batch
    from scib.metrics.lisi import lisi_graph
except ImportError:
    print("Error: scib not available. This script must run in the abc_env environment.")
    print("  conda run -n abc_env python eval_scib.py ...")
    sys.exit(1)


def evaluate_batch_effect(adata, batch_key, label_key, embed='X_original'):
    """Compute batch effect metrics: ASW_batch, iLISI, cLISI."""
    results = {}

    print("  Computing ASW Batch...")
    asw_batch = silhouette_batch(
        adata, batch_key=batch_key, label_key=label_key, embed=embed
    )
    results['ASW_batch'] = float(asw_batch)

    print("  Computing LISI...")
    lisi_result = lisi_graph(
        adata, batch_key=batch_key, label_key=label_key, type_='embed', use_rep=embed
    )
    results['iLISI_batch'] = float(lisi_result[0])
    results['cLISI_cell_type'] = float(lisi_result[1])

    return results


def plot_scib_results(results, output_dir):
    """Save a simple grouped bar chart for ASW_batch, iLISI, and cLISI."""
    metric_keys = ["ASW_batch", "iLISI_batch", "cLISI_cell_type"]
    metric_labels = ["ASW_batch", "iLISI", "cLISI"]

    encoded = results.get("encoded", {})
    decoded = results.get("decoded", {})
    if any("error" in section for section in (encoded, decoded)):
        print("  Skipping scib plot because one representation failed.")
        return

    encoded_vals = [encoded.get(key, np.nan) for key in metric_keys]
    decoded_vals = [decoded.get(key, np.nan) for key in metric_keys]

    x = np.arange(len(metric_labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, encoded_vals, width, label="encoded", color="#1f77b4")
    ax.bar(x + width / 2, decoded_vals, width, label="decoded", color="#ff7f0e")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    ax.set_title(f"scib Metrics, Epoch {results.get('epoch')}")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    for idx, val in enumerate(encoded_vals):
        if not np.isnan(val):
            ax.text(idx - width / 2, val, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    for idx, val in enumerate(decoded_vals):
        if not np.isnan(val):
            ax.text(idx + width / 2, val, f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    plot_path = os.path.join(output_dir, "scib_metrics.png")
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved to: {plot_path}")


def load_epoch_dump_for_scib(run_folder, epoch, modality=None):
    """Load epoch dump and build AnnData objects for encoded and decoded."""
    dump_dir = os.path.join(run_folder, 'epoch_dump')
    train_fp = None
    for pattern in [f'train_epoch_{epoch}.json', f'epoch_{epoch}_train.json', 'train.json']:
        fp = os.path.join(dump_dir, pattern)
        if os.path.isfile(fp):
            train_fp = fp
            break

    if train_fp is None:
        print(f"Error: epoch dump not found for epoch {epoch}")
        return None, None

    print(f"  Loading: {train_fp}")
    with open(train_fp, 'r') as f:
        data = json.load(f)

    encoded_list, decoded_list, cell_type_list, batch_list = [], [], [], []
    missing_batch_labels = False

    for ct in sorted(data.keys()):
        for bid in data[ct]:
            batch_data = data[ct][bid]
            encoded_key = 'encoded' if 'encoded' in batch_data else f'{modality}_encoded'
            decoded_key = 'decoded' if 'decoded' in batch_data else f'{modality}_decoded'
            encoded_list.extend(batch_data[encoded_key])
            decoded_list.extend(batch_data[decoded_key])
            n = len(batch_data[encoded_key])
            cell_type_list.extend([ct] * n)
            meta = resolve_batch_id_list(batch_data, modality=modality)
            if meta is None:
                missing_batch_labels = True
                batch_list.extend([None] * n)
            else:
                batch_list.extend(meta)

    if missing_batch_labels:
        raise RuntimeError(
            f"Batch labels are unavailable for modality '{modality}' in this joint dump. "
            "Current joint training output stores generic batch IDs only for the target modality."
        )

    del data; gc.collect()

    encoded_arr = np.array(encoded_list, dtype=np.float32)
    decoded_arr = np.array(decoded_list, dtype=np.float32)
    del encoded_list, decoded_list; gc.collect()

    # Build AnnData for encoded
    adata_enc = ad.AnnData(encoded_arr)
    adata_enc.obs['cell_type'] = pd.Series(cell_type_list, index=adata_enc.obs.index).astype('category')
    adata_enc.obs['meta_batch_id'] = pd.Series(batch_list, index=adata_enc.obs.index).astype('category')
    adata_enc.obsm['X_original'] = encoded_arr

    # Build AnnData for decoded
    adata_dec = ad.AnnData(decoded_arr)
    adata_dec.obs['cell_type'] = pd.Series(cell_type_list, index=adata_dec.obs.index).astype('category')
    adata_dec.obs['meta_batch_id'] = pd.Series(batch_list, index=adata_dec.obs.index).astype('category')
    adata_dec.obsm['X_original'] = decoded_arr

    return adata_enc, adata_dec


def _find_epoch_dump_path(run_folder, epoch):
    """Find epoch dump file path."""
    dump_dir = os.path.join(run_folder, 'epoch_dump')
    for pattern in [f'train_epoch_{epoch}.json', f'epoch_{epoch}_train.json', 'train.json']:
        fp = os.path.join(dump_dir, pattern)
        if os.path.isfile(fp):
            return fp
    return None


def _build_adata_from_arrays(arr, cell_types, batch_ids):
    """Build AnnData with cell_type, batch, and obsm['X_original']."""
    adata = ad.AnnData(arr)
    adata.obs['cell_type'] = pd.Series(cell_types, index=adata.obs.index).astype('category')
    adata.obs['meta_batch_id'] = pd.Series(batch_ids, index=adata.obs.index).astype('category')
    adata.obsm['X_original'] = arr
    return adata


def _run_scib_on_adata(adata, batch_key, label_key, label):
    """Run scIB metrics on an AnnData and return results dict."""
    print(f"  Evaluating {label} ({adata.shape})...")
    try:
        return evaluate_batch_effect(adata, batch_key=batch_key,
                                     label_key=label_key, embed='X_original')
    except Exception as e:
        print(f"    {label} failed: {e}")
        return {'error': str(e)}


def run_scib_analysis(run_folder, epoch, output_dir=None, modality=None):
    """Run scib metrics for a single epoch (original single-modality mode).

    Returns dict with results for the manifest.
    """
    if output_dir is None:
        output_dir = os.path.join(run_folder, 'evaluation', 'scib_metrics', f'epoch_{epoch}')
    os.makedirs(output_dir, exist_ok=True)

    adata_enc, adata_dec = load_epoch_dump_for_scib(run_folder, epoch, modality=modality)
    if adata_enc is None:
        return {'status': 'failed', 'reason': 'epoch dump not found'}

    results = {'epoch': epoch}

    # Encoded metrics
    print(f"  Evaluating encoded representation ({adata_enc.shape})...")
    try:
        enc_results = evaluate_batch_effect(
            adata_enc, batch_key='meta_batch_id', label_key='cell_type', embed='X_original'
        )
        results['encoded'] = enc_results
        print(f"    Encoded: {enc_results}")
    except Exception as e:
        print(f"    Encoded evaluation failed: {e}")
        results['encoded'] = {'error': str(e)}

    del adata_enc; gc.collect()

    # Decoded metrics
    print(f"  Evaluating decoded representation ({adata_dec.shape})...")
    try:
        dec_results = evaluate_batch_effect(
            adata_dec, batch_key='meta_batch_id', label_key='cell_type', embed='X_original'
        )
        results['decoded'] = dec_results
        print(f"    Decoded: {dec_results}")
    except Exception as e:
        print(f"    Decoded evaluation failed: {e}")
        results['decoded'] = {'error': str(e)}

    del adata_dec; gc.collect()

    # Save results
    results_path = os.path.join(output_dir, 'scib_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {results_path}")

    plot_scib_results(results, output_dir)

    results['status'] = 'success'
    return results


def run_scib_stage3_full(run_folder, epoch, output_dir=None):
    """Run full stage-3 scIB evaluation: per-modality + cross-modality + separability.

    Returns dict with all results.
    """
    if output_dir is None:
        output_dir = os.path.join(run_folder, 'evaluation', 'scib_metrics', f'epoch_{epoch}')
    os.makedirs(output_dir, exist_ok=True)

    train_fp = _find_epoch_dump_path(run_folder, epoch)
    if train_fp is None:
        return {'status': 'failed', 'reason': 'epoch dump not found'}

    # Use eval_utils to load common cell types
    from eval_utils import load_common_filters
    _, common_cell_types, _ = load_common_filters()

    (cell_enc, cell_types, cell_batch,
     nuclei_enc, nuclei_types, nuclei_batch) = load_both_encoded_stage3(
        train_fp, common_cell_types
    )

    results = {'epoch': epoch}

    # --- 1. Per-modality scIB ---
    print("\n=== Per-modality scIB: cell ===")
    cell_dir = os.path.join(output_dir, 'per_modality_cell')
    os.makedirs(cell_dir, exist_ok=True)
    adata_cell = _build_adata_from_arrays(cell_enc, cell_types, cell_batch)
    results['cell_encoded'] = _run_scib_on_adata(
        adata_cell, 'meta_batch_id', 'cell_type', 'cell encoded')
    del adata_cell; gc.collect()

    print("\n=== Per-modality scIB: nuclei ===")
    nuclei_dir = os.path.join(output_dir, 'per_modality_nuclei')
    os.makedirs(nuclei_dir, exist_ok=True)
    adata_nuclei = _build_adata_from_arrays(nuclei_enc, nuclei_types, nuclei_batch)
    results['nuclei_encoded'] = _run_scib_on_adata(
        adata_nuclei, 'meta_batch_id', 'cell_type', 'nuclei encoded')
    del adata_nuclei; gc.collect()

    # --- 2. Cross-modality mixing (cell+nuclei, batch=modality, label=cell_type) ---
    # Want: same cell types from different modalities should mix well
    print("\n=== Cross-modality mixing scIB ===")
    cross_dir = os.path.join(output_dir, 'cross_modality')
    os.makedirs(cross_dir, exist_ok=True)

    combined_enc = np.vstack([cell_enc, nuclei_enc])
    combined_types = np.concatenate([cell_types, nuclei_types])
    combined_modality = np.array(
        ['cell'] * len(cell_types) + ['nuclei'] * len(nuclei_types)
    )

    adata_cross = ad.AnnData(combined_enc)
    adata_cross.obs['cell_type'] = pd.Series(
        combined_types, index=adata_cross.obs.index).astype('category')
    adata_cross.obs['modality'] = pd.Series(
        combined_modality, index=adata_cross.obs.index).astype('category')
    adata_cross.obsm['X_original'] = combined_enc

    results['cross_modality_encoded'] = _run_scib_on_adata(
        adata_cross, 'modality', 'cell_type', 'cross-modality encoded')

    # --- 3. Modality separability (same data, measuring if modality is still separable) ---
    # Reuse same AnnData — same metrics but interpreted differently:
    #   high iLISI_batch = modalities are well-mixed (desired)
    #   low ASW_batch = modalities are not separable (desired)
    print("\n=== Modality separability scIB ===")
    sep_dir = os.path.join(output_dir, 'modality_separability')
    os.makedirs(sep_dir, exist_ok=True)
    # Already computed as cross_modality_encoded (same batch_key='modality')
    results['modality_separability_encoded'] = results['cross_modality_encoded']

    del adata_cross, combined_enc; gc.collect()

    # Now do decoded versions
    # Load decoded data
    print("\n=== Loading decoded for stage3 scIB ===")
    from eval_utils import load_both_decoded_stage3
    cell_dec, cell_dec_types, nuclei_dec, nuclei_dec_types = load_both_decoded_stage3(
        train_fp, common_cell_types
    )

    # Per-modality decoded
    adata_cell_dec = _build_adata_from_arrays(cell_dec, cell_dec_types, cell_batch)
    results['cell_decoded'] = _run_scib_on_adata(
        adata_cell_dec, 'meta_batch_id', 'cell_type', 'cell decoded')
    del adata_cell_dec; gc.collect()

    adata_nuclei_dec = _build_adata_from_arrays(nuclei_dec, nuclei_dec_types, nuclei_batch)
    results['nuclei_decoded'] = _run_scib_on_adata(
        adata_nuclei_dec, 'meta_batch_id', 'cell_type', 'nuclei decoded')
    del adata_nuclei_dec; gc.collect()

    # Cross-modality decoded
    combined_dec = np.vstack([cell_dec, nuclei_dec])
    combined_dec_types = np.concatenate([cell_dec_types, nuclei_dec_types])
    combined_dec_modality = np.array(
        ['cell'] * len(cell_dec_types) + ['nuclei'] * len(nuclei_dec_types)
    )
    adata_cross_dec = ad.AnnData(combined_dec)
    adata_cross_dec.obs['cell_type'] = pd.Series(
        combined_dec_types, index=adata_cross_dec.obs.index).astype('category')
    adata_cross_dec.obs['modality'] = pd.Series(
        combined_dec_modality, index=adata_cross_dec.obs.index).astype('category')
    adata_cross_dec.obsm['X_original'] = combined_dec

    results['cross_modality_decoded'] = _run_scib_on_adata(
        adata_cross_dec, 'modality', 'cell_type', 'cross-modality decoded')
    results['modality_separability_decoded'] = results['cross_modality_decoded']

    del adata_cross_dec, combined_dec; gc.collect()

    # Save results
    results_path = os.path.join(output_dir, 'scib_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to: {results_path}")

    results['status'] = 'success'
    return results


def main():
    parser = argparse.ArgumentParser(description="scib metrics (iLISI/cLISI)")
    parser.add_argument("--run_folder", required=True, help="Full path to run folder")
    parser.add_argument("--epoch", required=True, type=int, help="Epoch number")
    parser.add_argument("--output_dir", default=None, help="Output directory (default: auto)")
    parser.add_argument("--modality", default=None, choices=['cell', 'nuclei'],
                        help="Target modality for joint dumps (single mode only)")
    parser.add_argument("--mode", default="single", choices=['single', 'stage3_full'],
                        help="Evaluation mode: single (original) or stage3_full (all 3 families)")
    args = parser.parse_args()

    # For subprocess invocation, run_folder should be the full path
    run_folder = args.run_folder
    if not os.path.isdir(run_folder):
        # Try resolving as a name
        _PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        run_folder = os.path.join(_PROJECT_DIR, "run", args.run_folder)
    if not os.path.isdir(run_folder):
        print(f"Error: run folder not found: {args.run_folder}")
        sys.exit(1)

    print(f"Run folder: {run_folder}")
    print(f"Epoch: {args.epoch}")
    print(f"Mode: {args.mode}")

    if args.mode == 'stage3_full':
        results = run_scib_stage3_full(run_folder, args.epoch, args.output_dir)
    else:
        modality = args.modality or infer_target_modality(load_run_args(run_folder))
        results = run_scib_analysis(run_folder, args.epoch, args.output_dir, modality=modality)

    print(f"Done. Results: {json.dumps(results, indent=2, default=str)}")


if __name__ == '__main__':
    main()
