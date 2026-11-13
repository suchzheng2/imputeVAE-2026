#!/usr/bin/env python3
"""
Generate diagnostic analysis for a run folder, saved in-place.

Generates: loss curve plots, per-celltype analysis, train vs valid overlays,
and a text summary. All outputs go into the run folder itself.

Usage:
    python generate_report.py <run_folder_name>

Example:
    python generate_report.py run_2026-03-05-15-47-21_3630658

Output saved to:
    run/<run_folder_name>/analysis/
"""

import sys
import os
import glob
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_BASE_DIR = os.path.join(_PROJECT_DIR, "run")


def find_run_folder(run_name):
    """Find the run folder matching the given name."""
    exact = os.path.join(RUN_BASE_DIR, run_name)
    if os.path.isdir(exact):
        return exact
    matches = glob.glob(os.path.join(RUN_BASE_DIR, f"*{run_name}*"))
    dirs = [m for m in matches if os.path.isdir(m)]
    if len(dirs) == 1:
        return dirs[0]
    elif len(dirs) > 1:
        print(f"Multiple matches found: {dirs}")
        print(f"Using first match: {dirs[0]}")
        return dirs[0]
    else:
        print(f"Error: No run folder matching '{run_name}' found in {RUN_BASE_DIR}")
        sys.exit(1)


def load_loss_csv(path):
    """Load a CSV/comma-separated loss file, trimming runtime column if present."""
    with open(path, 'r') as f:
        header_line = f.readline().strip()
    columns = [c.strip() for c in header_line.split(',')]
    if 'runtime' in columns:
        idx = columns.index('runtime')
        columns = columns[:idx]
    df = pd.read_csv(path, usecols=range(len(columns)), names=columns, skiprows=1)
    return df


def load_loss_tsv(path):
    """Load a tab-separated loss file."""
    try:
        df = pd.read_csv(path, sep='\t')
        return df
    except Exception:
        return None


def plot_loss_curves(df, output_dir, prefix=""):
    """Generate individual + combined loss curves."""
    epoch_col = df.columns[0]
    loss_cols = [c for c in df.columns[1:]]

    for col in loss_cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df[epoch_col], df[col], marker='o', markersize=2, linewidth=1.2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(col)
        ax.set_title(f'{prefix}{col} vs Epoch')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f'{prefix}{col}.png'), dpi=150)
        plt.close(fig)

    if len(loss_cols) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        for col in loss_cols:
            ax.plot(df[epoch_col], df[col], marker='o', markersize=2, linewidth=1.2, label=col)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{prefix}All Loss Terms vs Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f'{prefix}all_losses.png'), dpi=150)
        plt.close(fig)


def plot_per_celltype_loss(train_loss_dir, output_dir):
    """Analyze per-cell-type loss from train_loss/ files to spot cell-type-specific issues."""
    files = sorted(glob.glob(os.path.join(train_loss_dir, '*.txt')))
    if not files:
        return

    celltype_data = {}

    for fpath in files:
        df = load_loss_tsv(fpath)
        if df is None or df.empty:
            continue
        basename = os.path.basename(fpath)
        try:
            epoch = int(basename.split('_')[1])
        except (IndexError, ValueError):
            continue

        for _, row in df.iterrows():
            ct = str(row.get('celltype', row.get('cell_type', 'unknown')))
            if ct not in celltype_data:
                celltype_data[ct] = {}
            metrics = {}
            for col in df.columns:
                if col in ('celltype', 'cell_type', 'batch', 'data_size'):
                    continue
                try:
                    metrics[col] = float(row[col])
                except (ValueError, TypeError):
                    continue
            if epoch not in celltype_data[ct]:
                celltype_data[ct][epoch] = metrics
            else:
                existing = celltype_data[ct][epoch]
                for k, v in metrics.items():
                    if k in existing:
                        existing[k] = (existing[k] + v) / 2
                    else:
                        existing[k] = v

    if not celltype_data:
        return

    ct_dir = os.path.join(output_dir, 'per_celltype')
    os.makedirs(ct_dir, exist_ok=True)

    all_metrics = set()
    for ct_epochs in celltype_data.values():
        for metrics in ct_epochs.values():
            all_metrics.update(metrics.keys())

    n_ct = len(celltype_data)
    cmap = plt.cm.get_cmap('tab20', max(n_ct, 20))
    ct_colors = {ct: cmap(i) for i, ct in enumerate(sorted(celltype_data.keys()))}

    for metric in all_metrics:
        # Overlaid plot with line labels
        fig, ax = plt.subplots(figsize=(14, 8))
        lines_info = []
        for ct in sorted(celltype_data.keys()):
            epochs_data = celltype_data[ct]
            epochs = sorted(epochs_data.keys())
            values = [epochs_data[e].get(metric, float('nan')) for e in epochs]
            color = ct_colors[ct]
            ax.plot(epochs, values, linewidth=1.0, color=color, alpha=0.8)
            for i in range(len(values) - 1, -1, -1):
                if not pd.isna(values[i]):
                    lines_info.append((epochs[i], values[i], ct, color))
                    break

        if lines_info:
            lines_info.sort(key=lambda x: x[1])
            max_epoch = max(e for e, _, _, _ in lines_info)
            for _, y_val, ct, color in lines_info:
                ax.annotate(ct, xy=(max_epoch, y_val),
                            xytext=(5, 0), textcoords='offset points',
                            fontsize=6.5, color=color, fontweight='bold',
                            va='center')

        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} by Cell Type')
        ax.grid(True, alpha=0.3)
        xlim = ax.get_xlim()
        ax.set_xlim(xlim[0], xlim[1] + (xlim[1] - xlim[0]) * 0.12)
        fig.tight_layout()
        fig.savefig(os.path.join(ct_dir, f'{metric}_by_celltype.png'), dpi=150)
        plt.close(fig)

        # Small multiples grid
        sorted_cts = sorted(celltype_data.keys())
        n_cols = 4
        n_rows = (n_ct + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows),
                                 squeeze=False, sharex=True, sharey=True)
        for idx, ct in enumerate(sorted_cts):
            r, c = divmod(idx, n_cols)
            ax = axes[r][c]
            epochs_data = celltype_data[ct]
            epochs = sorted(epochs_data.keys())
            values = [epochs_data[e].get(metric, float('nan')) for e in epochs]
            ax.plot(epochs, values, linewidth=1.0, color=ct_colors[ct])
            ax.set_title(ct, fontsize=9, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
        for idx in range(n_ct, n_rows * n_cols):
            r, c = divmod(idx, n_cols)
            axes[r][c].set_visible(False)
        fig.suptitle(f'{metric} by Cell Type (individual)', fontsize=12)
        fig.supxlabel('Epoch', fontsize=10)
        fig.supylabel(metric, fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(ct_dir, f'{metric}_by_celltype_grid.png'), dpi=150)
        plt.close(fig)

        # Interactive HTML (plotly)
        try:
            import plotly.graph_objects as pgo
            pfig = pgo.Figure()
            for ct in sorted(celltype_data.keys()):
                epochs_data = celltype_data[ct]
                epochs = sorted(epochs_data.keys())
                values = [epochs_data[e].get(metric, float('nan')) for e in epochs]
                pfig.add_trace(pgo.Scatter(
                    x=epochs, y=values, mode='lines', name=ct,
                    line=dict(width=1.5),
                    hovertemplate=f'{ct}<br>Epoch: %{{x}}<br>{metric}: %{{y:.6f}}<extra></extra>'
                ))
            pfig.update_layout(
                title=f'{metric} by Cell Type',
                xaxis_title='Epoch', yaxis_title=metric,
                hovermode='x unified',
                width=1000, height=600,
                legend=dict(font=dict(size=9))
            )
            pfig.write_html(os.path.join(ct_dir, f'{metric}_by_celltype.html'))
        except ImportError:
            pass


def generate_summary(run_folder, analysis_dir, train_df, valid_df):
    """Generate a text summary of the run."""
    summary_lines = []
    run_name = os.path.basename(run_folder)
    summary_lines.append(f"# Run Report: {run_name}")
    summary_lines.append(f"Run folder: {run_folder}")
    summary_lines.append("")

    args_path = os.path.join(run_folder, 'arguments.json')
    if os.path.isfile(args_path):
        with open(args_path, 'r') as f:
            args = json.load(f)
        summary_lines.append("## Training Arguments")
        for k, v in sorted(args.items()):
            summary_lines.append(f"  {k}: {v}")
        summary_lines.append("")

    if train_df is not None and not train_df.empty:
        summary_lines.append("## Training Loss Summary")
        summary_lines.append(f"  Total epochs: {len(train_df)}")
        epoch_col = train_df.columns[0]
        for col in train_df.columns[1:]:
            first_val = train_df[col].iloc[0]
            last_val = train_df[col].iloc[-1]
            min_val = train_df[col].min()
            min_epoch = train_df[epoch_col].iloc[train_df[col].idxmin()]
            summary_lines.append(f"  {col}:")
            summary_lines.append(f"    first={first_val:.6f}, last={last_val:.6f}, min={min_val:.6f} (epoch {min_epoch})")
            summary_lines.append(f"    delta={last_val - first_val:.6f} ({(last_val - first_val)/first_val*100:.2f}%)")
        summary_lines.append("")

    if valid_df is not None and not valid_df.empty:
        summary_lines.append("## Validation Loss Summary")
        summary_lines.append(f"  Total epochs: {len(valid_df)}")
        epoch_col = valid_df.columns[0]
        for col in valid_df.columns[1:]:
            first_val = valid_df[col].iloc[0]
            last_val = valid_df[col].iloc[-1]
            min_val = valid_df[col].min()
            min_epoch = valid_df[epoch_col].iloc[valid_df[col].idxmin()]
            summary_lines.append(f"  {col}:")
            summary_lines.append(f"    first={first_val:.6f}, last={last_val:.6f}, min={min_val:.6f} (epoch {min_epoch})")
            summary_lines.append(f"    delta={last_val - first_val:.6f} ({(last_val - first_val)/first_val*100:.2f}%)")
        summary_lines.append("")

    if train_df is not None and valid_df is not None and not train_df.empty and not valid_df.empty:
        summary_lines.append("## Train vs Validation Gap (last epoch)")
        common = set(train_df.columns[1:]) & set(valid_df.columns[1:])
        for col in sorted(common):
            t_last = train_df[col].iloc[-1]
            v_last = valid_df[col].iloc[-1]
            gap = v_last - t_last
            summary_lines.append(f"  {col}: train={t_last:.6f}, valid={v_last:.6f}, gap={gap:.6f}")
        summary_lines.append("")

    umap_dir = os.path.join(run_folder, 'umap')
    if os.path.isdir(umap_dir):
        umap_files = sorted(glob.glob(os.path.join(umap_dir, '*.png')))
        summary_lines.append(f"## UMAP Visualizations: {len(umap_files)} files")
        for f in umap_files:
            summary_lines.append(f"  {os.path.basename(f)}")
        summary_lines.append("")

    models_dir = os.path.join(run_folder, 'models')
    if os.path.isdir(models_dir):
        model_files = sorted(glob.glob(os.path.join(models_dir, '*.pt')))
        summary_lines.append(f"## Model Checkpoints: {len(model_files)} files")
        for f in model_files:
            size_mb = os.path.getsize(f) / (1024 * 1024)
            summary_lines.append(f"  {os.path.basename(f)} ({size_mb:.1f} MB)")
        summary_lines.append("")

    summary_lines.append("## Folder Sizes")
    for subdir in ['models', 'epoch_dump', 'umap', 'train_loss', 'valid_loss', 'train_epoch_loss']:
        subpath = os.path.join(run_folder, subdir)
        if os.path.isdir(subpath):
            total_size = 0
            for dirpath, _, filenames in os.walk(subpath):
                for f in filenames:
                    total_size += os.path.getsize(os.path.join(dirpath, f))
            size_mb = total_size / (1024 * 1024)
            summary_lines.append(f"  {subdir}: {size_mb:.1f} MB")
    summary_lines.append("")

    summary_path = os.path.join(analysis_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    return summary_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_report.py <run_folder_name>")
        sys.exit(1)

    run_name = sys.argv[1]
    run_folder = find_run_folder(run_name)
    print(f"Run folder: {run_folder}")

    # All analysis output goes into run_folder/analysis/
    analysis_dir = os.path.join(run_folder, 'analysis')
    loss_curve_dir = os.path.join(analysis_dir, 'loss_curves')
    os.makedirs(loss_curve_dir, exist_ok=True)
    print(f"Analysis dir: {analysis_dir}")

    # ---- 1. Plot training epoch loss ----
    train_df = None
    train_epoch_loss_path = os.path.join(run_folder, 'train_epoch_loss', 'loss.txt')
    root_loss = os.path.join(run_folder, 'loss.txt')

    if os.path.isfile(train_epoch_loss_path):
        train_df = load_loss_csv(train_epoch_loss_path)
        plot_loss_curves(train_df, loss_curve_dir, prefix="train_")
        print(f"Generated training loss curves ({len(train_df)} epochs)")
    elif os.path.isfile(root_loss):
        train_df = load_loss_csv(root_loss)
        plot_loss_curves(train_df, loss_curve_dir, prefix="train_")
        print(f"Generated training loss curves from root loss.txt ({len(train_df)} epochs)")

    # ---- 2. Plot validation loss ----
    valid_df = None
    for vname in ['valid_loss/valid_epoch_loss.txt', 'valid_loss/loss.txt']:
        vpath = os.path.join(run_folder, vname)
        if os.path.isfile(vpath):
            valid_df = load_loss_csv(vpath)
            plot_loss_curves(valid_df, loss_curve_dir, prefix="valid_")
            print(f"Generated validation loss curves ({len(valid_df)} epochs)")
            break

    # ---- 3. Train vs Valid overlay plots ----
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
            fig.savefig(os.path.join(loss_curve_dir, f'train_vs_valid_{col}.png'), dpi=150)
            plt.close(fig)
        print("Generated train vs valid overlay plots")

    # ---- 4. Per-celltype loss analysis ----
    train_loss_dir = os.path.join(run_folder, 'train_loss')
    if os.path.isdir(train_loss_dir):
        plot_per_celltype_loss(train_loss_dir, loss_curve_dir)
        print("Generated per-celltype loss curves")

    # ---- 5. Generate text summary ----
    summary_path = generate_summary(run_folder, analysis_dir, train_df, valid_df)
    print(f"Generated summary: {summary_path}")

    print("Done!")


if __name__ == '__main__':
    main()
