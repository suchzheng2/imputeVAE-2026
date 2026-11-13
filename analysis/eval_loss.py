#!/usr/bin/env python3
"""
Loss Curve Analysis module.

Generates loss curve plots, per-celltype breakdowns, and train/valid overlays.
Reads only loss text files (no raw data needed).

Usage:
    python eval_loss.py --run_folder <name>
"""

import os
import sys
import argparse

# Reuse the existing generate_report.py logic
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from eval_utils import find_run_folder, get_eval_dir, load_run_args
from generate_report import (
    load_loss_csv, load_loss_tsv,
    plot_loss_curves, plot_per_celltype_loss, generate_summary,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run_loss_analysis(run_folder, output_dir=None):
    """Run loss curve analysis for a run folder.

    Returns dict with summary info for the manifest.
    """
    if output_dir is None:
        output_dir = os.path.join(get_eval_dir(run_folder), 'loss_curves')
    os.makedirs(output_dir, exist_ok=True)

    results = {'status': 'success', 'plots': []}

    # Training loss
    train_df = None
    for path in [
        os.path.join(run_folder, 'train_epoch_loss', 'loss.txt'),
        os.path.join(run_folder, 'loss.txt'),
    ]:
        if os.path.isfile(path):
            train_df = load_loss_csv(path)
            plot_loss_curves(train_df, output_dir, prefix="train_")
            results['train_epochs'] = len(train_df)
            print(f"  Training loss curves ({len(train_df)} epochs)")
            break

    # Validation loss
    valid_df = None
    for vname in ['valid_loss/valid_epoch_loss.txt', 'valid_loss/loss.txt']:
        vpath = os.path.join(run_folder, vname)
        if os.path.isfile(vpath):
            valid_df = load_loss_csv(vpath)
            plot_loss_curves(valid_df, output_dir, prefix="valid_")
            results['valid_epochs'] = len(valid_df)
            print(f"  Validation loss curves ({len(valid_df)} epochs)")
            break

    # Train vs valid overlay
    if train_df is not None and valid_df is not None:
        epoch_col_t = train_df.columns[0]
        epoch_col_v = valid_df.columns[0]
        common_cols = set(train_df.columns[1:]) & set(valid_df.columns[1:])
        for col in common_cols:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(train_df[epoch_col_t], train_df[col], linewidth=1.2, label=f'train_{col}')
            ax.plot(valid_df[epoch_col_v], valid_df[col], linewidth=1.2, label=f'valid_{col}', linestyle='--')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(col)
            ax.set_title(f'Train vs Valid: {col}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f'train_vs_valid_{col}.png'), dpi=150)
            plt.close(fig)
        print("  Train vs valid overlay plots")

    # Per-celltype loss
    train_loss_dir = os.path.join(run_folder, 'train_loss')
    if os.path.isdir(train_loss_dir):
        plot_per_celltype_loss(train_loss_dir, output_dir)
        print("  Per-celltype loss curves")

    # Text summary
    eval_dir = os.path.dirname(output_dir)
    generate_summary(run_folder, eval_dir, train_df, valid_df)

    return results


def main():
    parser = argparse.ArgumentParser(description="Loss curve analysis")
    parser.add_argument("--run_folder", required=True, help="Run folder name")
    args = parser.parse_args()

    run_folder = find_run_folder(args.run_folder)
    print(f"Run folder: {run_folder}")
    print("Running loss analysis...")
    results = run_loss_analysis(run_folder)
    print(f"Done. Results: {results}")


if __name__ == '__main__':
    main()
