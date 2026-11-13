#!/usr/bin/env python3
"""
Decoder Output Scale Tracking module.

For each epoch dump, computes statistics (mean, std, min, max, percentiles)
of decoded output values, both overall and per cell type.
Tracks how output magnitude changes over training.

Usage:
    python eval_decoder_scale.py --run_folder <name> [--epochs 0,50,100]
"""

import os
import sys
import gc
import json
import argparse
import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from eval_utils import (
    find_run_folder, get_eval_dir, list_epoch_dumps, find_epoch_dump,
    load_run_args, infer_target_modality,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def compute_scale_stats(train_fp, modality=None):
    """Compute scale statistics from a single epoch dump.

    Returns (overall_stats_dict, per_celltype_stats_dict).
    """
    with open(train_fp, 'r') as f:
        data = json.load(f)

    all_values = []
    ct_values = {}

    for ct in sorted(data.keys()):
        ct_vals = []
        for bid in data[ct]:
            batch_data = data[ct][bid]
            key = 'decoded' if 'decoded' in batch_data else f'{modality}_decoded'
            decoded = np.array(batch_data[key], dtype=np.float32)
            ct_vals.append(decoded)
            all_values.append(decoded)
        if ct_vals:
            ct_arr = np.concatenate([v.flatten() for v in ct_vals])
            ct_values[ct] = {
                'mean': float(np.mean(ct_arr)),
                'std': float(np.std(ct_arr)),
                'min': float(np.min(ct_arr)),
                'max': float(np.max(ct_arr)),
                'p5': float(np.percentile(ct_arr, 5)),
                'p25': float(np.percentile(ct_arr, 25)),
                'median': float(np.median(ct_arr)),
                'p75': float(np.percentile(ct_arr, 75)),
                'p95': float(np.percentile(ct_arr, 95)),
                'n_cells': sum(v.shape[0] for v in ct_vals),
            }

    del data
    gc.collect()

    all_arr = np.concatenate([v.flatten() for v in all_values])
    overall = {
        'mean': float(np.mean(all_arr)),
        'std': float(np.std(all_arr)),
        'min': float(np.min(all_arr)),
        'max': float(np.max(all_arr)),
        'p5': float(np.percentile(all_arr, 5)),
        'p25': float(np.percentile(all_arr, 25)),
        'median': float(np.median(all_arr)),
        'p75': float(np.percentile(all_arr, 75)),
        'p95': float(np.percentile(all_arr, 95)),
        'n_cells': sum(v.shape[0] for v in all_values),
    }

    del all_arr, all_values
    gc.collect()

    return overall, ct_values


def run_decoder_scale_analysis(run_folder, epochs=None, output_dir=None):
    """Run decoder scale analysis across epochs.

    Returns dict with summary info for the manifest.
    """
    if output_dir is None:
        output_dir = os.path.join(get_eval_dir(run_folder), 'decoder_scale')
    os.makedirs(output_dir, exist_ok=True)
    run_args = load_run_args(run_folder)
    modality = infer_target_modality(run_args)

    available_epochs = list_epoch_dumps(run_folder)
    if not available_epochs:
        print("  No epoch dumps found.")
        return {'status': 'skipped', 'reason': 'no epoch dumps'}

    if epochs is not None:
        available_epochs = [e for e in available_epochs if e in epochs]

    print(f"  Analyzing epochs: {available_epochs}")

    overall_rows = []
    ct_rows = []

    for epoch in available_epochs:
        fp = find_epoch_dump(run_folder, epoch)
        if fp is None:
            continue
        print(f"  Processing epoch {epoch}...")
        overall, ct_stats = compute_scale_stats(fp, modality=modality)
        overall['epoch'] = epoch
        overall_rows.append(overall)

        for ct, stats in ct_stats.items():
            row = dict(stats)
            row['epoch'] = epoch
            row['cell_type'] = ct
            ct_rows.append(row)

    if not overall_rows:
        return {'status': 'skipped', 'reason': 'no valid epochs'}

    # Save CSVs
    overall_df = pd.DataFrame(overall_rows)
    cols = ['epoch', 'mean', 'std', 'min', 'p5', 'p25', 'median', 'p75', 'p95', 'max', 'n_cells']
    overall_df = overall_df[cols]
    overall_df.to_csv(os.path.join(output_dir, 'decoder_scale_overall.csv'), index=False)

    ct_df = pd.DataFrame(ct_rows)
    ct_cols = ['epoch', 'cell_type', 'mean', 'std', 'min', 'p5', 'p25', 'median', 'p75', 'p95', 'max', 'n_cells']
    ct_df = ct_df[ct_cols]
    ct_df.to_csv(os.path.join(output_dir, 'decoder_scale_per_celltype.csv'), index=False)

    # Plot overall scale over epochs
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.plot(overall_df['epoch'], overall_df['mean'], 'b-o', markersize=4, label='Mean')
    ax.fill_between(overall_df['epoch'], overall_df['p25'], overall_df['p75'], alpha=0.2, label='IQR (p25-p75)')
    ax.fill_between(overall_df['epoch'], overall_df['p5'], overall_df['p95'], alpha=0.1, label='p5-p95')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Decoded Value')
    ax.set_title('Decoder Output Scale Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(overall_df['epoch'], overall_df['std'], 'r-o', markersize=4, label='Std')
    ax.plot(overall_df['epoch'], overall_df['max'], 'g-s', markersize=4, label='Max')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value')
    ax.set_title('Decoder Output Spread Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'decoder_scale_overall.png'), dpi=150)
    plt.close(fig)

    # Per-celltype scale plot
    cell_types = sorted(ct_df['cell_type'].unique())
    cmap = plt.cm.get_cmap('tab20', max(len(cell_types), 20))
    ct_colors = {ct: cmap(i) for i, ct in enumerate(cell_types)}

    fig, ax = plt.subplots(figsize=(14, 8))
    for ct in cell_types:
        mask = ct_df['cell_type'] == ct
        ax.plot(ct_df.loc[mask, 'epoch'], ct_df.loc[mask, 'mean'],
                linewidth=1.0, color=ct_colors[ct], alpha=0.8)
        # Label at last point
        last_row = ct_df.loc[mask].iloc[-1]
        ax.annotate(ct, xy=(last_row['epoch'], last_row['mean']),
                    xytext=(5, 0), textcoords='offset points',
                    fontsize=6.5, color=ct_colors[ct], fontweight='bold', va='center')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Decoded Value')
    ax.set_title('Mean Decoder Output by Cell Type Over Training')
    ax.grid(True, alpha=0.3)
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0], xlim[1] + (xlim[1] - xlim[0]) * 0.12)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'decoder_scale_per_celltype.png'), dpi=150)
    plt.close(fig)

    return {
        'status': 'success',
        'epochs_analyzed': available_epochs,
        'final_mean': float(overall_df.iloc[-1]['mean']),
        'final_std': float(overall_df.iloc[-1]['std']),
    }


def main():
    parser = argparse.ArgumentParser(description="Decoder scale analysis")
    parser.add_argument("--run_folder", required=True, help="Run folder name")
    parser.add_argument("--epochs", type=str, default=None,
                        help="Comma-separated epochs to analyze (default: all available)")
    args = parser.parse_args()

    run_folder = find_run_folder(args.run_folder)
    epochs = [int(e) for e in args.epochs.split(',')] if args.epochs else None

    print(f"Run folder: {run_folder}")
    print("Running decoder scale analysis...")
    results = run_decoder_scale_analysis(run_folder, epochs=epochs)
    print(f"Done. Results: {results}")


if __name__ == '__main__':
    main()
