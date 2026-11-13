#!/usr/bin/env python3
"""
Expression & Error Heatmaps module.

Generates decoded expression heatmaps, difference heatmaps, and
fractional error analysis.

Usage:
    python eval_heatmaps.py --run_folder <name> --modality <cell|nuclei> --epoch <N>
"""

import os
import sys
import gc
import argparse
import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from eval_utils import (
    find_run_folder, get_eval_dir, load_run_args,
    find_epoch_dump, load_common_filters, load_raw_data,
    load_decoded_stage1, stratified_subsample, compute_target_counts,
    print_memory_usage, MAX_CELLS_PER_TYPE, RANDOM_SEED,
    RAW_CELL_PATH, RAW_NUCLEI_PATH,
)
from scipy import sparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def compute_fractional_error(decoded, raw):
    """Fractional error: nonzero raw -> (dec-raw)/raw, zero raw -> dec."""
    result = np.zeros_like(decoded, dtype=np.float32)
    nz = raw != 0
    result[nz] = (decoded[nz] - raw[nz]) / raw[nz]
    result[~nz] = decoded[~nz]
    return result


def plot_expression_heatmap(data, cell_types, title, output_path, vmin=None, vmax=None, cmap='viridis'):
    """Heatmap: genes (rows) x cells (columns), grouped by cell type."""
    data_T = data.T
    fig, ax = plt.subplots(figsize=(16, 10))
    if vmin is None: vmin = np.percentile(data_T, 1)
    if vmax is None: vmax = np.percentile(data_T, 99)
    im = ax.imshow(data_T, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_ylabel('Genes')
    ax.set_title(title, fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)

    unique_types, boundaries, centers = [], [0], []
    current, start = cell_types[0], 0
    for i, ct in enumerate(cell_types):
        if ct != current:
            boundaries.append(i); centers.append((start + i - 1) / 2)
            unique_types.append(current); current = ct; start = i
    boundaries.append(len(cell_types)); centers.append((start + len(cell_types) - 1) / 2)
    unique_types.append(current)
    for b in boundaries[1:-1]:
        ax.axvline(x=b - 0.5, color='white', linewidth=1.5, alpha=0.9)
    ax.set_xticks(centers)
    ax.set_xticklabels(unique_types, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Cell Type')

    stats_text = f"Shape: {data_T.shape[1]} cells x {data_T.shape[0]} genes\nMean: {data_T.mean():.3f}, Std: {data_T.std():.3f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    del data_T; gc.collect()


def plot_difference_heatmap(data1, data2, cell_types, title, output_path, label1="", label2=""):
    """Heatmap of data1 - data2."""
    diff_T = (data1 - data2).T
    fig, ax = plt.subplots(figsize=(16, 10))
    vmax = np.percentile(np.abs(diff_T), 99); vmin = -vmax
    im = ax.imshow(diff_T, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax.set_ylabel('Genes')
    ax.set_title(f"{title}\n({label1} - {label2})", fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8).set_label('Difference')

    unique_types, boundaries, centers = [], [0], []
    current, start = cell_types[0], 0
    for i, ct in enumerate(cell_types):
        if ct != current:
            boundaries.append(i); centers.append((start + i - 1) / 2)
            unique_types.append(current); current = ct; start = i
    boundaries.append(len(cell_types)); centers.append((start + len(cell_types) - 1) / 2)
    unique_types.append(current)
    for b in boundaries[1:-1]:
        ax.axvline(x=b - 0.5, color='black', linewidth=1.5, alpha=0.7)
    ax.set_xticks(centers)
    ax.set_xticklabels(unique_types, rotation=45, ha='right', fontsize=9)

    stats_text = f"Mean diff: {diff_T.mean():.3f}, Std: {diff_T.std():.3f}, Mean |diff|: {np.abs(diff_T).mean():.3f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    del diff_T; gc.collect()


def plot_fractional_error_heatmap(decoded, raw, cell_types, title, output_path):
    """Heatmap of fractional error."""
    frac = compute_fractional_error(decoded, raw)
    frac_T = frac.T

    nz_mask = raw != 0
    n_nz = nz_mask.sum()
    n_z = (~nz_mask).sum()
    total = frac.size

    frac_nz = frac[nz_mask]
    mean_frac = frac_nz.mean() if n_nz > 0 else 0
    mean_abs_frac = np.abs(frac_nz).mean() if n_nz > 0 else 0
    mean_halluc = frac[~nz_mask].mean() if n_z > 0 else 0
    del frac

    fig, ax = plt.subplots(figsize=(16, 10))
    vmax = np.percentile(np.abs(frac_T), 99); vmin = -vmax
    im = ax.imshow(frac_T, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax.set_ylabel('Genes')
    ax.set_title(title, fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8).set_label('Fractional Error')

    unique_types, boundaries, centers = [], [0], []
    current, start = cell_types[0], 0
    for i, ct in enumerate(cell_types):
        if ct != current:
            boundaries.append(i); centers.append((start + i - 1) / 2)
            unique_types.append(current); current = ct; start = i
    boundaries.append(len(cell_types)); centers.append((start + len(cell_types) - 1) / 2)
    unique_types.append(current)
    for b in boundaries[1:-1]:
        ax.axvline(x=b - 0.5, color='black', linewidth=1.5, alpha=0.7)
    ax.set_xticks(centers)
    ax.set_xticklabels(unique_types, rotation=45, ha='right', fontsize=9)

    stats_text = (f"Non-zero raw ({100*n_nz/total:.1f}%): mean frac err={mean_frac:.4f}, mean |frac err|={mean_abs_frac:.4f}\n"
                  f"Zero raw ({100*n_z/total:.1f}%): mean hallucinated={mean_halluc:.4f}")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=7, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), family='monospace')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    del frac_T; gc.collect()


def plot_fractional_error_distribution(decoded, raw, title, output_path):
    """Histogram of fractional error split by zero/nonzero raw."""
    frac = compute_fractional_error(decoded, raw)
    nz = raw != 0
    frac_nz = frac[nz].flatten()
    frac_z = frac[~nz].flatten()
    del frac

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    sample = np.random.choice(frac_nz, min(200000, len(frac_nz)), replace=False) if len(frac_nz) > 200000 else frac_nz
    clip = np.percentile(np.abs(sample), 99.5)
    ax1.hist(np.clip(sample, -clip, clip), bins=100, alpha=0.7, color='steelblue', edgecolor='none')
    ax1.axvline(x=0, color='red', linewidth=1.5, linestyle='--')
    ax1.set_xlabel('Fractional Error'); ax1.set_ylabel('Count')
    ax1.set_title(f'Non-zero Raw (n={len(frac_nz):,})')
    ax1.text(0.97, 0.97, f"Mean: {frac_nz.mean():.4f}\nMedian: {np.median(frac_nz):.4f}\nMean |err|: {np.abs(frac_nz).mean():.4f}",
             transform=ax1.transAxes, fontsize=8, va='top', ha='right',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9), family='monospace')

    ax2 = axes[1]
    if len(frac_z) > 0:
        sample_z = np.random.choice(frac_z, min(200000, len(frac_z)), replace=False) if len(frac_z) > 200000 else frac_z
        ax2.hist(sample_z, bins=100, alpha=0.7, color='coral', edgecolor='none')
        ax2.set_xlabel('Decoded Value (hallucinated)'); ax2.set_ylabel('Count')
        ax2.set_title(f'Zero Raw (n={len(frac_z):,})')
        ax2.text(0.97, 0.97, f"Mean: {frac_z.mean():.4f}\nMedian: {np.median(frac_z):.4f}",
                 transform=ax2.transAxes, fontsize=8, va='top', ha='right',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9), family='monospace')

    fig.suptitle(title, fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    gc.collect()


def run_heatmap_analysis_stage3(run_folder, epoch,
                                 raw_cell_adata=None, raw_nuclei_adata=None,
                                 output_dir=None):
    """Run 4-source heatmap analysis for a stage3_v2 epoch.

    Generates heatmaps for: raw_cell, raw_nuclei, decoded_cell, decoded_nuclei.
    Returns dict with summary info for the manifest.
    """
    from eval_utils import load_both_decoded_stage3

    eval_dir = get_eval_dir(run_folder)
    heatmap_dir = output_dir or os.path.join(eval_dir, 'heatmaps', f'epoch_{epoch}')
    os.makedirs(heatmap_dir, exist_ok=True)

    run_args = load_run_args(run_folder)
    use_log1p = run_args.get('raw_data_normalization', 'raw') == 'log1p'
    det_rate = float(run_args.get('detection_rate_threshold', 1.0))
    exclude_mito = run_args.get('exclude_mitocarta', True)
    norm_label = "log1p" if use_log1p else "raw"

    fp = find_epoch_dump(run_folder, epoch)
    if fp is None:
        return {'status': 'failed', 'reason': f'epoch dump not found for epoch {epoch}'}

    common_genes, common_cell_types, mitocarta_genes = load_common_filters()
    protocol = run_args.get('protocol', None)
    batch = run_args.get('batch', None)

    # Load both decoded modalities
    cell_dec, cell_dec_types, nuclei_dec, nuclei_dec_types = load_both_decoded_stage3(
        fp, common_cell_types
    )

    # Load raw data if needed
    if raw_cell_adata is None or raw_nuclei_adata is None:
        print("  Loading raw reference data...")
        raw_cell_adata = load_raw_data(RAW_CELL_PATH, common_genes, common_cell_types,
                                        mitocarta_genes, det_rate, exclude_mito,
                                        protocol=protocol, batch=batch)
        raw_nuclei_adata = load_raw_data(RAW_NUCLEI_PATH, common_genes, common_cell_types,
                                          mitocarta_genes, det_rate, exclude_mito,
                                          protocol=protocol, batch=batch)

    raw_cell_ct = raw_cell_adata.obs['cell_type'].values
    raw_nuclei_ct = raw_nuclei_adata.obs['cell_type'].values
    tc = compute_target_counts(
        [cell_dec_types, nuclei_dec_types, raw_cell_ct, raw_nuclei_ct],
        MAX_CELLS_PER_TYPE,
    )
    print(f"  4-source heatmap target counts: {len(tc)} cell types, {sum(tc.values())} total")

    # Subsample decoded
    dec_cell_sub, dec_cell_ct = stratified_subsample(cell_dec, cell_dec_types,
                                                       seed=RANDOM_SEED, target_counts=tc)
    dec_nuclei_sub, dec_nuclei_ct = stratified_subsample(nuclei_dec, nuclei_dec_types,
                                                           seed=RANDOM_SEED, target_counts=tc)
    del cell_dec, nuclei_dec; gc.collect()

    # Subsample raw
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

    print(f"  Shapes: dec_cell={dec_cell_sub.shape}, dec_nuclei={dec_nuclei_sub.shape}, "
          f"raw_cell={rc_sub.shape}, raw_nuclei={rn_sub.shape}")

    # Common color scale
    all_vals = np.concatenate([
        dec_cell_sub.flatten(), dec_nuclei_sub.flatten(),
        rc_ref.flatten(), rn_ref.flatten(),
    ])
    vmin = float(np.percentile(all_vals, 1))
    vmax = float(np.percentile(all_vals, 99))
    del all_vals

    # 4 expression heatmaps
    plot_expression_heatmap(dec_cell_sub, dec_cell_ct, f"Decoded Cell (epoch {epoch})",
                            os.path.join(heatmap_dir, "decoded_cell_heatmap.png"), vmin, vmax)
    plot_expression_heatmap(dec_nuclei_sub, dec_nuclei_ct, f"Decoded Nuclei (epoch {epoch})",
                            os.path.join(heatmap_dir, "decoded_nuclei_heatmap.png"), vmin, vmax)
    plot_expression_heatmap(rc_ref, rc_types, f"Raw Cell ({norm_label})",
                            os.path.join(heatmap_dir, f"raw_cell_{norm_label}_heatmap.png"), vmin, vmax)
    plot_expression_heatmap(rn_ref, rn_types, f"Raw Nuclei ({norm_label})",
                            os.path.join(heatmap_dir, f"raw_nuclei_{norm_label}_heatmap.png"), vmin, vmax)

    # 4 difference heatmaps (decoded_X vs raw_Y)
    pairs = [
        (dec_cell_sub, rc_ref, dec_cell_ct, "decoded_cell", f"raw_cell ({norm_label})"),
        (dec_cell_sub, rn_ref, dec_cell_ct, "decoded_cell", f"raw_nuclei ({norm_label})"),
        (dec_nuclei_sub, rc_ref, dec_nuclei_ct, "decoded_nuclei", f"raw_cell ({norm_label})"),
        (dec_nuclei_sub, rn_ref, dec_nuclei_ct, "decoded_nuclei", f"raw_nuclei ({norm_label})"),
    ]
    for dec, raw, ct, dec_label, raw_label in pairs:
        safe_raw = raw_label.replace(" ", "_").replace("(", "").replace(")", "")
        plot_difference_heatmap(
            dec, raw, ct,
            f"{dec_label} vs {raw_label}",
            os.path.join(heatmap_dir, f"diff_{dec_label}_vs_{safe_raw}.png"),
            dec_label, raw_label,
        )

    del dec_cell_sub, dec_nuclei_sub, rc_sub, rn_sub, rc_ref, rn_ref
    gc.collect()

    return {'status': 'success', 'epoch': epoch}


def run_heatmap_analysis(run_folder, modality, epoch,
                          raw_cell_adata=None, raw_nuclei_adata=None,
                          output_dir=None):
    """Run heatmap analysis for a single epoch.

    Returns dict with summary info for the manifest.
    """
    eval_dir = get_eval_dir(run_folder)
    heatmap_dir = output_dir or os.path.join(eval_dir, 'heatmaps', f'epoch_{epoch}')
    os.makedirs(heatmap_dir, exist_ok=True)

    run_args = load_run_args(run_folder)
    use_log1p = run_args.get('raw_data_normalization', 'raw') == 'log1p'
    det_rate = float(run_args.get('detection_rate_threshold', 1.0))
    exclude_mito = run_args.get('exclude_mitocarta', True)
    norm_label = "log1p" if use_log1p else "raw"

    fp = find_epoch_dump(run_folder, epoch)
    if fp is None:
        return {'status': 'failed', 'reason': f'epoch dump not found for epoch {epoch}'}

    common_genes, common_cell_types, mitocarta_genes = load_common_filters()

    # Load decoded
    decoded_arr, decoded_types = load_decoded_stage1(fp, common_cell_types, modality=modality)

    # Protocol filtering — match training data selection
    protocol = run_args.get('protocol', None)
    batch = run_args.get('batch', None)

    # Load raw data if needed
    if raw_cell_adata is None or raw_nuclei_adata is None:
        print("  Loading raw reference data...")
        raw_cell_adata = load_raw_data(RAW_CELL_PATH, common_genes, common_cell_types,
                                        mitocarta_genes, det_rate, exclude_mito,
                                        protocol=protocol, batch=batch)
        raw_nuclei_adata = load_raw_data(RAW_NUCLEI_PATH, common_genes, common_cell_types,
                                          mitocarta_genes, det_rate, exclude_mito,
                                          protocol=protocol, batch=batch)

    decoded_label = f'{modality}_decoded'

    # Compute target counts
    raw_cell_ct = raw_cell_adata.obs['cell_type'].values
    raw_nuclei_ct = raw_nuclei_adata.obs['cell_type'].values
    tc = compute_target_counts([decoded_types, raw_cell_ct, raw_nuclei_ct], MAX_CELLS_PER_TYPE)
    print(f"  Target counts ({len(tc)} cell types, {sum(tc.values())} total cells): {tc}")

    # Stratified subsample
    dec_sub, dec_types_sub = stratified_subsample(decoded_arr, decoded_types,
                                                   seed=RANDOM_SEED, target_counts=tc)
    del decoded_arr, decoded_types; gc.collect()

    raw_cell_X = raw_cell_adata.X
    rc_sub, rc_types = stratified_subsample(
        raw_cell_X.toarray() if sparse.issparse(raw_cell_X) else raw_cell_X,
        raw_cell_ct, seed=RANDOM_SEED, target_counts=tc)
    rc_ref = np.log1p(rc_sub) if use_log1p else rc_sub.astype(np.float32)

    raw_nuclei_X = raw_nuclei_adata.X
    rn_sub, rn_types = stratified_subsample(
        raw_nuclei_X.toarray() if sparse.issparse(raw_nuclei_X) else raw_nuclei_X,
        raw_nuclei_ct, seed=RANDOM_SEED, target_counts=tc)
    rn_ref = np.log1p(rn_sub) if use_log1p else rn_sub.astype(np.float32)

    print(f"  Shapes: decoded={dec_sub.shape}, raw_cell={rc_sub.shape}, raw_nuclei={rn_sub.shape}")

    # Common color scale
    vmin = np.percentile(np.concatenate([dec_sub.flatten(), rc_ref.flatten(), rn_ref.flatten()]), 1)
    vmax = np.percentile(np.concatenate([dec_sub.flatten(), rc_ref.flatten(), rn_ref.flatten()]), 99)

    modality_name = "Cell" if modality == 'cell' else "Nuclei"

    # Expression heatmaps
    plot_expression_heatmap(dec_sub, dec_types_sub, f"{modality_name} Decoded (epoch {epoch})",
                            os.path.join(heatmap_dir, "decoded_heatmap.png"), vmin, vmax)
    plot_expression_heatmap(rc_ref, rc_types, f"Raw Cell ({norm_label})",
                            os.path.join(heatmap_dir, f"raw_cell_{norm_label}_heatmap.png"), vmin, vmax)
    plot_expression_heatmap(rn_ref, rn_types, f"Raw Nuclei ({norm_label})",
                            os.path.join(heatmap_dir, f"raw_nuclei_{norm_label}_heatmap.png"), vmin, vmax)

    # Difference heatmaps
    plot_difference_heatmap(dec_sub, rc_ref, dec_types_sub,
                            f"Decoded vs Raw Cell ({norm_label})",
                            os.path.join(heatmap_dir, f"diff_decoded_vs_raw_cell_{norm_label}.png"),
                            decoded_label, f"raw_cell ({norm_label})")
    plot_difference_heatmap(dec_sub, rn_ref, dec_types_sub,
                            f"Decoded vs Raw Nuclei ({norm_label})",
                            os.path.join(heatmap_dir, f"diff_decoded_vs_raw_nuclei_{norm_label}.png"),
                            decoded_label, f"raw_nuclei ({norm_label})")

    # Fractional error
    if modality == 'cell':
        same_ref, same_label = rc_ref, "raw_cell"
        other_ref, other_label = rn_ref, "raw_nuclei"
    else:
        same_ref, same_label = rn_ref, "raw_nuclei"
        other_ref, other_label = rc_ref, "raw_cell"

    plot_fractional_error_heatmap(dec_sub, same_ref, dec_types_sub,
                                  f"Fractional Error: {decoded_label} vs {same_label} ({norm_label})",
                                  os.path.join(heatmap_dir, f"frac_error_vs_{same_label}_{norm_label}.png"))
    plot_fractional_error_heatmap(dec_sub, other_ref, dec_types_sub,
                                  f"Fractional Error: {decoded_label} vs {other_label} ({norm_label})",
                                  os.path.join(heatmap_dir, f"frac_error_vs_{other_label}_{norm_label}.png"))

    plot_fractional_error_distribution(dec_sub, same_ref,
                                       f"{decoded_label} vs {same_label} ({norm_label})",
                                       os.path.join(heatmap_dir, f"frac_error_dist_vs_{same_label}_{norm_label}.png"))
    plot_fractional_error_distribution(dec_sub, other_ref,
                                       f"{decoded_label} vs {other_label} ({norm_label})",
                                       os.path.join(heatmap_dir, f"frac_error_dist_vs_{other_label}_{norm_label}.png"))

    return {'status': 'success', 'epoch': epoch}


def main():
    parser = argparse.ArgumentParser(description="Expression & error heatmaps")
    parser.add_argument("--run_folder", required=True, help="Run folder name")
    parser.add_argument("--modality", required=True, choices=['cell', 'nuclei'])
    parser.add_argument("--epoch", required=True, type=int, help="Epoch to analyze")
    args = parser.parse_args()

    run_folder = find_run_folder(args.run_folder)
    print(f"Run folder: {run_folder}")
    print(f"Modality: {args.modality}, Epoch: {args.epoch}")
    print("Running heatmap analysis...")
    results = run_heatmap_analysis(run_folder, args.modality, args.epoch)
    print(f"Done. Results: {results}")


if __name__ == '__main__':
    main()
