#!/usr/bin/env python3
"""
Correlation & Shift Analysis module.

Computes 3x3 correlation matrices (raw_cell, raw_nuclei, decoded) per cell type,
shift scores, and multi-epoch shift trajectories.

Usage:
    python eval_correlation.py --run_folder <name> --modality <cell|nuclei> \
        [--epochs 0,50,100] [--cross_run <other_run>]
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
    list_epoch_dumps, find_epoch_dump,
    load_common_filters, load_raw_data, load_decoded_stage1,
    compute_celltype_mean, stratified_subsample_adata, get_raw_data_for_correlation,
    print_memory_usage, is_stage3_run, SUBSAMPLE_FRACTION, RANDOM_SEED,
    RAW_CELL_PATH, RAW_NUCLEI_PATH,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def _annotate_heatmap(ax, data, fmt='.3f', fontsize=10):
    """Manually annotate heatmap cells (workaround for seaborn 0.12.2 bug)."""
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data.iloc[i, j] if hasattr(data, 'iloc') else data[i, j]
            if np.isnan(val):
                continue
            color = 'white' if val > 0.7 else 'black'
            ax.text(j + 0.5, i + 0.5, f'{val:{fmt}}', ha='center', va='center',
                    fontsize=fontsize, color=color)


def compute_pairwise_corr(mean_dict, cell_type):
    """Compute pairwise Pearson correlations for one cell type."""
    sources = list(mean_dict.keys())
    n = len(sources)
    corr = np.full((n, n), np.nan)
    for i, s1 in enumerate(sources):
        for j, s2 in enumerate(sources):
            df1, df2 = mean_dict[s1], mean_dict[s2]
            if cell_type not in df1.index or cell_type not in df2.index:
                continue
            if df1.columns.dtype == 'object' and df2.columns.dtype == 'object':
                common = sorted(set(df1.columns) & set(df2.columns))
                if not common:
                    continue
                e1 = df1.loc[cell_type, common].values.astype(float)
                e2 = df2.loc[cell_type, common].values.astype(float)
            else:
                e1 = df1.loc[cell_type].values.astype(float)
                e2 = df2.loc[cell_type].values.astype(float)
            r, _ = stats.pearsonr(e1, e2)
            corr[i, j] = r
    return pd.DataFrame(corr, index=sources, columns=sources)


def run_correlation_analysis(mean_dict, output_dir, prefix=""):
    """Run full correlation analysis and generate plots."""
    os.makedirs(output_dir, exist_ok=True)

    all_cts = None
    for df in mean_dict.values():
        cts = set(df.index)
        all_cts = cts if all_cts is None else all_cts & cts
    all_cts = sorted(all_cts)
    print(f"  Common cell types: {len(all_cts)}")

    per_ct_corr = {}
    matrices = []
    for ct in all_cts:
        m = compute_pairwise_corr(mean_dict, ct)
        per_ct_corr[ct] = m
        matrices.append(m.values)

    avg_corr = pd.DataFrame(np.nanmean(matrices, axis=0),
                            index=list(mean_dict.keys()),
                            columns=list(mean_dict.keys()))

    avg_corr.to_csv(os.path.join(output_dir, f"{prefix}avg_correlation.csv"))

    # Average correlation heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(avg_corr, annot=False, cmap='RdYlBu_r',
                center=0.5, vmin=0, vmax=1, square=True, linewidths=0.5, ax=ax)
    _annotate_heatmap(ax, avg_corr, fmt='.3f', fontsize=10)
    ax.set_title(f'{prefix}Average Correlation Matrix', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}avg_correlation_heatmap.png"), dpi=200)
    plt.close(fig)

    # Per-cell-type grid
    n_cts = len(all_cts)
    n_cols = 4
    n_rows = (n_cts + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), squeeze=False)
    for idx, ct in enumerate(all_cts):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        m = per_ct_corr[ct]
        sns.heatmap(m, annot=False, cmap='RdYlBu_r', center=0.5,
                    vmin=0, vmax=1, square=True, linewidths=0.3, ax=ax,
                    cbar=False)
        _annotate_heatmap(ax, m, fmt='.2f', fontsize=7)
        ax.set_title(ct, fontsize=9, fontweight='bold')
        ax.tick_params(labelsize=6)
    for idx in range(n_cts, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].set_visible(False)
    fig.suptitle(f'{prefix}Per Cell Type Correlations', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}per_celltype_correlation_grid.png"), dpi=150)
    plt.close(fig)

    return per_ct_corr, avg_corr


def compute_shift_scores(per_ct_corr, decoded_label):
    """Compute shift = corr(decoded, raw_cell) - corr(decoded, raw_nuclei)."""
    results = []
    for ct, m in per_ct_corr.items():
        if decoded_label not in m.index:
            continue
        if 'raw_cell' not in m.index or 'raw_nuclei' not in m.index:
            continue
        d2c = m.loc[decoded_label, 'raw_cell']
        d2n = m.loc[decoded_label, 'raw_nuclei']
        results.append({
            'cell_type': ct,
            'shift': d2c - d2n,
            f'{decoded_label}_to_raw_cell': d2c,
            f'{decoded_label}_to_raw_nuclei': d2n,
            'raw_cell_to_raw_nuclei': m.loc['raw_cell', 'raw_nuclei'],
        })
    return pd.DataFrame(results)


def plot_shift_scores(shift_df, decoded_label, output_dir, prefix=""):
    """Plot shift score bar chart and scatter."""
    os.makedirs(output_dir, exist_ok=True)

    # Bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    df_s = shift_df.sort_values('shift', ascending=False)
    colors = ['green' if x > 0 else 'red' for x in df_s['shift']]
    ax.bar(range(len(df_s)), df_s['shift'], color=colors)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xticks(range(len(df_s)))
    ax.set_xticklabels(df_s['cell_type'], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Shift Score')
    ax.set_title(f'{prefix}{decoded_label} Shift Score\n'
                 f'shift = corr({decoded_label}, raw_cell) - corr({decoded_label}, raw_nuclei)\n'
                 f'(Positive = More Cell-like)', fontsize=11)
    n_pos = (df_s['shift'] > 0).sum()
    ax.text(0.98, 0.97,
            f"Cell-like: {n_pos}/{len(df_s)} ({100*n_pos/len(df_s):.1f}%)\nMean: {df_s['shift'].mean():.4f}",
            transform=ax.transAxes, fontsize=9, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}shift_scores_bar.png"), dpi=200)
    plt.close(fig)

    # Scatter
    fig, ax = plt.subplots(figsize=(8, 8))
    x = shift_df[f'{decoded_label}_to_raw_nuclei']
    y = shift_df[f'{decoded_label}_to_raw_cell']
    ax.scatter(x, y, c='steelblue', alpha=0.7, s=100, edgecolors='white', linewidth=0.5)
    for i, ct in enumerate(shift_df['cell_type']):
        ax.annotate(ct, (x.iloc[i], y.iloc[i]), fontsize=7, alpha=0.7)
    lims = [min(x.min(), y.min()) - 0.05, max(x.max(), y.max()) + 0.05]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='Equal similarity')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Corr with Raw Nuclei'); ax.set_ylabel('Corr with Raw Cell')
    ax.set_title(f'{prefix}{decoded_label} Similarity\n(Above diagonal = More cell-like)')
    ax.legend(); ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}shift_scatter.png"), dpi=200)
    plt.close(fig)

    shift_df.to_csv(os.path.join(output_dir, f"{prefix}shift_scores.csv"), index=False)


def plot_shift_over_epochs(all_shifts, decoded_label, output_dir):
    """Plot shift score trajectory across epochs."""
    os.makedirs(output_dir, exist_ok=True)

    if not all_shifts:
        return

    # Collect per cell type trajectories
    ct_trajectories = {}
    epochs_list = sorted(all_shifts.keys())
    for epoch in epochs_list:
        sdf = all_shifts[epoch]
        for _, row in sdf.iterrows():
            ct = row['cell_type']
            if ct not in ct_trajectories:
                ct_trajectories[ct] = {'epochs': [], 'shifts': []}
            ct_trajectories[ct]['epochs'].append(epoch)
            ct_trajectories[ct]['shifts'].append(row['shift'])

    # Mean shift over epochs
    mean_shifts = []
    for epoch in epochs_list:
        mean_shifts.append(all_shifts[epoch]['shift'].mean())

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Mean shift trajectory
    ax = axes[0]
    ax.plot(epochs_list, mean_shifts, 'b-o', markersize=5, linewidth=2)
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Shift Score')
    ax.set_title(f'Mean Shift Score Over Training\n'
                 f'(Positive = More Cell-like)')
    ax.grid(True, alpha=0.3)

    # Per cell type trajectories
    ax = axes[1]
    cmap = plt.cm.get_cmap('tab20', max(len(ct_trajectories), 20))
    for i, (ct, traj) in enumerate(sorted(ct_trajectories.items())):
        color = cmap(i)
        ax.plot(traj['epochs'], traj['shifts'], linewidth=1.0, color=color, alpha=0.8)
        if traj['epochs']:
            ax.annotate(ct, xy=(traj['epochs'][-1], traj['shifts'][-1]),
                        xytext=(5, 0), textcoords='offset points',
                        fontsize=6.5, color=color, fontweight='bold', va='center')
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Shift Score')
    ax.set_title('Shift Score by Cell Type Over Training')
    ax.grid(True, alpha=0.3)
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0], xlim[1] + (xlim[1] - xlim[0]) * 0.12)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'shift_over_epochs.png'), dpi=200)
    plt.close(fig)

    # Save CSV
    rows = []
    for epoch in epochs_list:
        sdf = all_shifts[epoch]
        for _, row in sdf.iterrows():
            rows.append({'epoch': epoch, **row.to_dict()})
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, 'shift_over_epochs.csv'), index=False)

    # Save mean shift CSV
    pd.DataFrame({'epoch': epochs_list, 'mean_shift': mean_shifts}).to_csv(
        os.path.join(output_dir, 'mean_shift_over_epochs.csv'), index=False)


def run_correlation_and_shift_stage3(run_folder, epochs=None,
                                      raw_cell_adata=None, raw_nuclei_adata=None,
                                      output_dir=None):
    """Run 4x4 correlation analysis for stage3_v2 runs.

    Uses both decoded modalities (decoded_cell, decoded_nuclei) plus raw_cell, raw_nuclei.
    Returns dict with summary info for the manifest.
    """
    from eval_utils import load_both_decoded_stage3

    eval_dir = get_eval_dir(run_folder)
    corr_base = output_dir or os.path.join(eval_dir, 'correlation')
    shift_dir = os.path.join(eval_dir, 'shift_over_epochs')

    run_args = load_run_args(run_folder)
    use_log1p = run_args.get('raw_data_normalization', 'raw') == 'log1p'
    det_rate = float(run_args.get('detection_rate_threshold', 1.0))
    exclude_mito = run_args.get('exclude_mitocarta', True)
    protocol = run_args.get('protocol', None)
    gene_list_path = run_args.get('gene_list', None)
    if gene_list_path and not os.path.isabs(gene_list_path):
        _project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        gene_list_path = os.path.join(_project_dir, gene_list_path)

    available_epochs = list_epoch_dumps(run_folder)
    if epochs is not None:
        available_epochs = [e for e in available_epochs if e in epochs]
    if not available_epochs:
        return {'status': 'skipped', 'reason': 'no epoch dumps'}

    common_genes, common_cell_types, mitocarta_genes = load_common_filters()

    if raw_cell_adata is None or raw_nuclei_adata is None:
        print("  Loading raw reference data...")
        raw_cell_adata = load_raw_data(RAW_CELL_PATH, common_genes, common_cell_types,
                                        mitocarta_genes, det_rate, exclude_mito,
                                        protocol=protocol, batch='c3A',
                                        gene_list=gene_list_path)
        raw_nuclei_adata = load_raw_data(RAW_NUCLEI_PATH, common_genes, common_cell_types,
                                          mitocarta_genes, det_rate, exclude_mito,
                                          protocol=protocol, batch='n3A',
                                          gene_list=gene_list_path)
        print_memory_usage("After raw data load")

    # Compute raw means (once)
    print("  Computing raw cell-type means...")
    raw_cell_sub = stratified_subsample_adata(raw_cell_adata, SUBSAMPLE_FRACTION, seed=RANDOM_SEED)
    raw_cell_X = get_raw_data_for_correlation(raw_cell_sub, use_log1p)
    raw_cell_mean = compute_celltype_mean(raw_cell_X, raw_cell_sub.obs['cell_type'].values,
                                           list(raw_cell_sub.var['gene_names']))
    del raw_cell_X, raw_cell_sub; gc.collect()

    raw_nuclei_sub = stratified_subsample_adata(raw_nuclei_adata, SUBSAMPLE_FRACTION, seed=RANDOM_SEED)
    raw_nuclei_X = get_raw_data_for_correlation(raw_nuclei_sub, use_log1p)
    raw_nuclei_mean = compute_celltype_mean(raw_nuclei_X, raw_nuclei_sub.obs['cell_type'].values,
                                             list(raw_nuclei_sub.var['gene_names']))
    del raw_nuclei_X, raw_nuclei_sub; gc.collect()

    raw_common_genes = sorted(set(raw_cell_mean.columns) & set(raw_nuclei_mean.columns))
    raw_cell_mean_c = raw_cell_mean[raw_common_genes]
    raw_nuclei_mean_c = raw_nuclei_mean[raw_common_genes]
    n_raw = len(raw_common_genes)

    all_shifts_cell = {}
    all_shifts_nuclei = {}

    for epoch in available_epochs:
        fp = find_epoch_dump(run_folder, epoch)
        if fp is None:
            continue

        print(f"\n  Epoch {epoch}:")
        epoch_dir = os.path.join(corr_base, f'epoch_{epoch}')
        os.makedirs(epoch_dir, exist_ok=True)

        cell_dec, cell_dec_types, nuclei_dec, nuclei_dec_types = load_both_decoded_stage3(
            fp, common_cell_types
        )

        cell_dec_mean = compute_celltype_mean(cell_dec, cell_dec_types)
        nuclei_dec_mean = compute_celltype_mean(nuclei_dec, nuclei_dec_types)
        del cell_dec, nuclei_dec; gc.collect()

        # Align decoded to raw gene order
        cell_dec_mean_c = cell_dec_mean.iloc[:, :min(cell_dec_mean.shape[1], n_raw)].copy()
        cell_dec_mean_c.columns = raw_common_genes[:cell_dec_mean_c.shape[1]]
        nuclei_dec_mean_c = nuclei_dec_mean.iloc[:, :min(nuclei_dec_mean.shape[1], n_raw)].copy()
        nuclei_dec_mean_c.columns = raw_common_genes[:nuclei_dec_mean_c.shape[1]]

        mean_dict = {
            'raw_cell': raw_cell_mean_c,
            'raw_nuclei': raw_nuclei_mean_c,
            'decoded_cell': cell_dec_mean_c,
            'decoded_nuclei': nuclei_dec_mean_c,
        }

        per_ct_corr, avg_corr = run_correlation_analysis(mean_dict, epoch_dir, prefix="4x4_")

        # Shift scores for both decoded modalities
        for decoded_label, all_shifts in [('decoded_cell', all_shifts_cell),
                                           ('decoded_nuclei', all_shifts_nuclei)]:
            shift_df = compute_shift_scores(per_ct_corr, decoded_label)
            if not shift_df.empty:
                plot_shift_scores(shift_df, decoded_label, epoch_dir, prefix=f"4x4_{decoded_label}_")
                all_shifts[epoch] = shift_df
                print(f"    {decoded_label} mean shift: {shift_df['shift'].mean():.4f}")

        del cell_dec_mean, nuclei_dec_mean, cell_dec_mean_c, nuclei_dec_mean_c
        gc.collect()

    # Shift over epochs
    if len(all_shifts_cell) > 1:
        print("\n  Generating decoded_cell shift trajectory...")
        plot_shift_over_epochs(all_shifts_cell, 'decoded_cell', shift_dir)
    if len(all_shifts_nuclei) > 1:
        print("\n  Generating decoded_nuclei shift trajectory...")
        plot_shift_over_epochs(all_shifts_nuclei, 'decoded_nuclei', shift_dir)

    final_cell_shift = float(list(all_shifts_cell.values())[-1]['shift'].mean()) if all_shifts_cell else None
    final_nuclei_shift = float(list(all_shifts_nuclei.values())[-1]['shift'].mean()) if all_shifts_nuclei else None

    return {
        'status': 'success',
        'epochs_analyzed': sorted(set(list(all_shifts_cell.keys()) + list(all_shifts_nuclei.keys()))),
        'final_decoded_cell_mean_shift': final_cell_shift,
        'final_decoded_nuclei_mean_shift': final_nuclei_shift,
    }


def run_correlation_and_shift(run_folder, modality, epochs=None,
                               raw_cell_adata=None, raw_nuclei_adata=None,
                               output_dir=None):
    """Run correlation analysis and shift scoring across epochs.

    If raw_cell_adata and raw_nuclei_adata are provided, they are reused.
    Otherwise they are loaded fresh (expensive).

    Returns dict with summary info for the manifest.
    """
    eval_dir = get_eval_dir(run_folder)
    corr_base = output_dir or os.path.join(eval_dir, 'correlation')
    shift_dir = os.path.join(eval_dir, 'shift_over_epochs')

    run_args = load_run_args(run_folder)
    use_log1p = run_args.get('raw_data_normalization', 'raw') == 'log1p'
    det_rate = float(run_args.get('detection_rate_threshold', 1.0))
    exclude_mito = run_args.get('exclude_mitocarta', True)
    protocol = run_args.get('protocol', None)
    gene_list_path = run_args.get('gene_list', None)
    if gene_list_path and not os.path.isabs(gene_list_path):
        _project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        gene_list_path = os.path.join(_project_dir, gene_list_path)

    available_epochs = list_epoch_dumps(run_folder)
    if epochs is not None:
        available_epochs = [e for e in available_epochs if e in epochs]
    if not available_epochs:
        return {'status': 'skipped', 'reason': 'no epoch dumps'}

    # Load reference data if not provided
    common_genes, common_cell_types, mitocarta_genes = load_common_filters()

    if raw_cell_adata is None or raw_nuclei_adata is None:
        print("  Loading raw reference data...")
        raw_cell_adata = load_raw_data(RAW_CELL_PATH, common_genes, common_cell_types,
                                        mitocarta_genes, det_rate, exclude_mito,
                                        protocol=protocol, batch='c3A',
                                        gene_list=gene_list_path)
        raw_nuclei_adata = load_raw_data(RAW_NUCLEI_PATH, common_genes, common_cell_types,
                                          mitocarta_genes, det_rate, exclude_mito,
                                          protocol=protocol, batch='n3A',
                                          gene_list=gene_list_path)
        print_memory_usage("After raw data load")

    # Compute raw means (once, reused across epochs)
    print("  Computing raw cell-type means...")
    raw_cell_sub = stratified_subsample_adata(raw_cell_adata, SUBSAMPLE_FRACTION, seed=RANDOM_SEED)
    raw_cell_X = get_raw_data_for_correlation(raw_cell_sub, use_log1p)
    raw_cell_mean = compute_celltype_mean(raw_cell_X, raw_cell_sub.obs['cell_type'].values,
                                           list(raw_cell_sub.var['gene_names']))
    del raw_cell_X, raw_cell_sub; gc.collect()

    raw_nuclei_sub = stratified_subsample_adata(raw_nuclei_adata, SUBSAMPLE_FRACTION, seed=RANDOM_SEED)
    raw_nuclei_X = get_raw_data_for_correlation(raw_nuclei_sub, use_log1p)
    raw_nuclei_mean = compute_celltype_mean(raw_nuclei_X, raw_nuclei_sub.obs['cell_type'].values,
                                             list(raw_nuclei_sub.var['gene_names']))
    del raw_nuclei_X, raw_nuclei_sub; gc.collect()

    # Gene alignment
    raw_common_genes = sorted(set(raw_cell_mean.columns) & set(raw_nuclei_mean.columns))
    raw_cell_mean_c = raw_cell_mean[raw_common_genes]
    raw_nuclei_mean_c = raw_nuclei_mean[raw_common_genes]
    n_raw = len(raw_common_genes)

    decoded_label = f'{modality}_decoded'
    all_shifts = {}

    for epoch in available_epochs:
        fp = find_epoch_dump(run_folder, epoch)
        if fp is None:
            continue

        print(f"\n  Epoch {epoch}:")
        epoch_dir = os.path.join(corr_base, f'epoch_{epoch}')
        os.makedirs(epoch_dir, exist_ok=True)

        decoded_arr, decoded_types = load_decoded_stage1(fp, common_cell_types, modality=modality)
        decoded_mean = compute_celltype_mean(decoded_arr, decoded_types)

        n_dec = decoded_mean.shape[1]
        n_use = min(n_dec, n_raw)
        decoded_mean_c = decoded_mean.iloc[:, :n_use].copy()
        decoded_mean_c.columns = raw_common_genes[:n_use]

        mean_dict = {
            'raw_cell': raw_cell_mean_c,
            'raw_nuclei': raw_nuclei_mean_c,
            decoded_label: decoded_mean_c,
        }

        per_ct_corr, avg_corr = run_correlation_analysis(mean_dict, epoch_dir, prefix="3x3_")
        shift_df = compute_shift_scores(per_ct_corr, decoded_label)
        plot_shift_scores(shift_df, decoded_label, epoch_dir, prefix="3x3_")

        all_shifts[epoch] = shift_df
        print(f"    Mean shift: {shift_df['shift'].mean():.4f}")

        del decoded_arr, decoded_types, decoded_mean, decoded_mean_c
        gc.collect()

    # Shift over epochs
    if len(all_shifts) > 1:
        print("\n  Generating shift trajectory...")
        plot_shift_over_epochs(all_shifts, decoded_label, shift_dir)

    return {
        'status': 'success',
        'epochs_analyzed': list(all_shifts.keys()),
        'final_mean_shift': float(list(all_shifts.values())[-1]['shift'].mean()) if all_shifts else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Correlation & shift analysis")
    parser.add_argument("--run_folder", required=True, help="Run folder name")
    parser.add_argument("--modality", required=True, choices=['cell', 'nuclei'])
    parser.add_argument("--epochs", type=str, default=None,
                        help="Comma-separated epochs (default: all)")
    args = parser.parse_args()

    run_folder = find_run_folder(args.run_folder)
    epochs = [int(e) for e in args.epochs.split(',')] if args.epochs else None

    print(f"Run folder: {run_folder}")
    print(f"Modality: {args.modality}")
    print("Running correlation & shift analysis...")
    results = run_correlation_and_shift(run_folder, args.modality, epochs=epochs)
    print(f"Done. Results: {results}")


if __name__ == '__main__':
    main()
