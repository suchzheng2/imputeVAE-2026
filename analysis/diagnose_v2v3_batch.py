#!/usr/bin/env python3
"""
Diagnostic script: Verify and visualize 10X-v2 vs 10X-v3 batch effects.

Checks whether the banding pattern in raw nuclei heatmaps is caused by
10X-v2 vs 10X-v3 protocol differences. Creates:
  1. Raw nuclei heatmap annotated with v2/v3 color bar
  2. Raw nuclei UMAP colored by v2/v3 protocol
  3. UMAP of encoded latent space at selected epochs colored by sample ID

Usage:
    python diagnose_v2v3_batch.py --run_path /path/to/run_folder [--epochs 200 300 500]
"""

import os
import sys
import gc
import argparse
import json
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from eval_utils import (
    load_common_filters, load_raw_data,
    print_memory_usage, MAX_CELLS_PER_TYPE, RANDOM_SEED,
    RAW_CELL_PATH, RAW_NUCLEI_PATH,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import umap as umap_lib


# ---- helpers ----

def extract_protocol(meta_batch_id):
    """Extract 10X protocol version from meta_batch_id.

    meta_batch_id examples: '1-n2A', '3-n3A', '1-c2A', '2-c3A'
    Returns '10X-v2' or '10X-v3'.
    """
    for part in str(meta_batch_id).split('-'):
        if len(part) >= 3 and part[0] in ('n', 'c') and part[1] in ('2', '3'):
            return f'10X-v{part[1]}'
    return 'unknown'


def add_protocol_to_adata(adata):
    """Add 'protocol' and 'meta_batch_id' columns to adata."""
    if 'meta_batch_id' not in adata.obs.columns:
        adata.obs['meta_batch_id'] = adata.obs['batch_id'].str.split('.').str[0]
    adata.obs['protocol'] = adata.obs['meta_batch_id'].apply(extract_protocol)
    return adata


def load_run_args(run_path):
    """Load arguments.json from a run folder."""
    args_path = os.path.join(run_path, 'arguments.json')
    if not os.path.isfile(args_path):
        return {}
    with open(args_path, 'r') as f:
        return json.load(f)


def find_epoch_dump(run_path, epoch):
    """Find the epoch dump file for a given epoch number."""
    dump_dir = os.path.join(run_path, 'epoch_dump')
    candidates = [
        os.path.join(dump_dir, f'train_epoch_{epoch}.json'),
        os.path.join(dump_dir, f'epoch_{epoch}_train.json'),
        os.path.join(dump_dir, f'train_{epoch}.json'),
    ]
    for fp in candidates:
        if os.path.isfile(fp):
            return fp
    return None


def list_epoch_dumps(run_path):
    """List available epoch dump files, returning sorted epoch numbers."""
    dump_dir = os.path.join(run_path, 'epoch_dump')
    if not os.path.isdir(dump_dir):
        return []
    epochs = []
    for f in os.listdir(dump_dir):
        if f.startswith('train_epoch_') and f.endswith('.json'):
            try:
                epoch = int(f.replace('train_epoch_', '').replace('.json', ''))
                epochs.append(epoch)
            except ValueError:
                pass
    return sorted(epochs)


def load_encoded_stage1(train_fp, common_cell_types):
    """Load encoded (latent) data from stage 1 JSON."""
    with open(train_fp, 'r') as f:
        data = json.load(f)

    encoded_list, types_list, batch_list = [], [], []
    for ct in sorted(data.keys()):
        if ct not in common_cell_types:
            continue
        for bid in data[ct]:
            batch_data = data[ct][bid]
            encoded_list.append(np.array(batch_data['encoded'], dtype=np.float32))
            n = len(batch_data['encoded'])
            types_list.extend([ct] * n)
            meta = batch_data.get('meta_batch_id_original', [ct] * n)
            batch_list.extend(meta)

    del data
    gc.collect()

    encoded_arr = np.vstack(encoded_list)
    types_arr = np.array(types_list)
    batch_arr = np.array(batch_list)
    return encoded_arr, types_arr, batch_arr


def stratified_subsample_with_obs(adata, max_per_type=30, seed=42, target_counts=None):
    """Subsample adata keeping cell_type proportions, return sorted by cell_type."""
    np.random.seed(seed)
    cell_types = adata.obs['cell_type'].values
    indices = []
    cts_to_use = sorted(target_counts.keys()) if target_counts else sorted(set(cell_types))
    for ct in cts_to_use:
        ct_idx = np.where(cell_types == ct)[0]
        if len(ct_idx) == 0:
            continue
        limit = target_counts[ct] if target_counts else max_per_type
        if len(ct_idx) > limit:
            chosen = np.random.choice(ct_idx, limit, replace=False)
        else:
            chosen = ct_idx
        indices.extend(sorted(chosen))
    indices = np.array(indices)
    sorted_order = np.argsort(cell_types[indices])
    indices = indices[sorted_order]
    return adata[indices].copy()


# ---- Heatmap with v2/v3 color annotation ----

def plot_heatmap_with_protocol_bar(data, cell_types, protocols, title, output_path,
                                    vmin=None, vmax=None, cmap='viridis'):
    """Expression heatmap with a color bar on top showing v2/v3 protocol."""
    data_T = data.T  # genes x cells
    if vmin is None:
        vmin = np.percentile(data_T, 1)
    if vmax is None:
        vmax = np.percentile(data_T, 99)

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 1, height_ratios=[0.5, 20], hspace=0.02)

    # Protocol color bar
    ax_bar = fig.add_subplot(gs[0])
    proto_num = np.array([0 if p == '10X-v2' else (1 if p == '10X-v3' else 2) for p in protocols])
    cmap_bar = matplotlib.colors.ListedColormap(['#e41a1c', '#377eb8', '#999999'])
    ax_bar.imshow(proto_num.reshape(1, -1), aspect='auto', cmap=cmap_bar,
                  vmin=-0.5, vmax=2.5, interpolation='nearest')
    ax_bar.set_yticks([0])
    ax_bar.set_yticklabels(['Protocol'], fontsize=9)
    ax_bar.set_xticks([])
    ax_bar.set_title(title, fontsize=13, fontweight='bold', pad=10)

    # Main heatmap
    ax_heat = fig.add_subplot(gs[1])
    im = ax_heat.imshow(data_T, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax_heat.set_ylabel('Genes')
    plt.colorbar(im, ax=ax_heat, shrink=0.8)

    # Cell type boundaries
    unique_types, boundaries, centers = [], [0], []
    current, start = cell_types[0], 0
    for i, ct in enumerate(cell_types):
        if ct != current:
            boundaries.append(i)
            centers.append((start + i - 1) / 2)
            unique_types.append(current)
            current = ct
            start = i
    boundaries.append(len(cell_types))
    centers.append((start + len(cell_types) - 1) / 2)
    unique_types.append(current)
    for b in boundaries[1:-1]:
        ax_heat.axvline(x=b - 0.5, color='white', linewidth=1.5, alpha=0.9)
        ax_bar.axvline(x=b - 0.5, color='black', linewidth=1.0, alpha=0.7)
    ax_heat.set_xticks(centers)
    ax_heat.set_xticklabels(unique_types, rotation=45, ha='right', fontsize=9)
    ax_heat.set_xlabel('Cell Type')

    # Stats + legend
    n_v2 = sum(1 for p in protocols if p == '10X-v2')
    n_v3 = sum(1 for p in protocols if p == '10X-v3')
    stats_text = (f"Shape: {data_T.shape[1]} cells x {data_T.shape[0]} genes\n"
                  f"10X-v2: {n_v2} cells, 10X-v3: {n_v3} cells\n"
                  f"Mean: {data_T.mean():.3f}, Std: {data_T.std():.3f}")
    ax_heat.text(0.02, 0.98, stats_text, transform=ax_heat.transAxes, fontsize=8, va='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    legend_elements = [Patch(facecolor='#e41a1c', label='10X-v2'),
                       Patch(facecolor='#377eb8', label='10X-v3')]
    ax_bar.legend(handles=legend_elements, loc='upper right', fontsize=8,
                  framealpha=0.9, ncol=2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")
    del data_T
    gc.collect()


# ---- UMAP plots ----

def plot_umap_panels(umap_coords, labels_dict, title_prefix, output_path, figsize=(28, 8)):
    """Create UMAP panels colored by different label sets."""
    n_panels = len(labels_dict)
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    for ax, (label_name, labels) in zip(axes, labels_dict.items()):
        unique_labels = sorted(set(labels))
        n_labels = len(unique_labels)

        if label_name == 'Protocol':
            color_map = {'10X-v2': '#e41a1c', '10X-v3': '#377eb8', 'unknown': '#999999'}
        elif n_labels <= 10:
            cmap = plt.cm.tab10
            color_map = {l: cmap(i / max(n_labels - 1, 1)) for i, l in enumerate(unique_labels)}
        elif n_labels <= 20:
            cmap = plt.cm.tab20
            color_map = {l: cmap(i / max(n_labels - 1, 1)) for i, l in enumerate(unique_labels)}
        else:
            cmap_colors = np.vstack([
                plt.cm.tab20(np.linspace(0, 1, 20)),
                plt.cm.tab20b(np.linspace(0, 1, min(20, n_labels - 20))),
            ])
            color_map = {l: cmap_colors[i % len(cmap_colors)] for i, l in enumerate(unique_labels)}

        for lbl in unique_labels:
            mask = np.array(labels) == lbl
            ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                       c=[color_map[lbl]], label=lbl, alpha=0.6, s=2)

        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title(f'{title_prefix} - {label_name} (n={n_labels})')

        if n_labels <= 25:
            ax.legend(markerscale=3, loc='upper right', fontsize=6, framealpha=0.8)
        else:
            ax.text(0.95, 0.95, f'{n_labels} groups',
                    transform=ax.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def compute_and_plot_umap(data, labels_dict, title_prefix, output_path,
                          n_pcs=50, max_cells=30000):
    """PCA -> UMAP -> plot."""
    from sklearn.decomposition import TruncatedSVD, PCA

    print(f"  Computing UMAP for '{title_prefix}' ({data.shape[0]} cells)...")

    if data.shape[0] > max_cells:
        idx = np.random.choice(data.shape[0], max_cells, replace=False)
        if sparse.issparse(data):
            data = data[idx]
        else:
            data = data[idx]
        labels_dict = {k: [v[i] for i in idx] for k, v in labels_dict.items()}

    n_components = min(n_pcs, data.shape[1] - 1)
    if sparse.issparse(data):
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        X_pca = svd.fit_transform(data)
    else:
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(data)

    reducer = umap_lib.UMAP(n_neighbors=15, min_dist=0.3, random_state=42, low_memory=True)
    umap_coords = reducer.fit_transform(X_pca)

    plot_umap_panels(umap_coords, labels_dict, title_prefix, output_path)
    return umap_coords


# ---- Main analysis ----

def run_v2v3_diagnosis(run_path, epochs, output_dir=None):
    """Run full v2/v3 batch effect diagnosis."""
    run_args = load_run_args(run_path)
    diag_dir = output_dir or os.path.join(run_path, 'evaluation', 'v2v3_diagnosis')
    os.makedirs(diag_dir, exist_ok=True)

    use_log1p = run_args.get('raw_data_normalization', 'raw') == 'log1p'
    det_rate = float(run_args.get('detection_rate_threshold', 1.0))
    exclude_mito = run_args.get('exclude_mitocarta', True)
    norm_label = "log1p" if use_log1p else "raw"

    common_genes, common_cell_types, mitocarta_genes = load_common_filters()

    # ---- Step 1: Load raw data and check v2/v3 distribution ----
    print("=" * 60)
    print("STEP 1: Loading raw data and checking v2/v3 distribution")
    print("=" * 60)

    raw_nuclei = load_raw_data(RAW_NUCLEI_PATH, common_genes, common_cell_types,
                                mitocarta_genes, det_rate, exclude_mito)
    raw_cell = load_raw_data(RAW_CELL_PATH, common_genes, common_cell_types,
                              mitocarta_genes, det_rate, exclude_mito)

    raw_nuclei = add_protocol_to_adata(raw_nuclei)
    raw_cell = add_protocol_to_adata(raw_cell)

    print(f"\n  Raw nuclei protocol distribution:")
    print(raw_nuclei.obs['protocol'].value_counts().to_string())
    print(f"\n  Raw nuclei meta_batch_id distribution:")
    print(raw_nuclei.obs['meta_batch_id'].value_counts().to_string())
    print(f"\n  Raw cell protocol distribution:")
    print(raw_cell.obs['protocol'].value_counts().to_string())

    # Save summary
    summary_path = os.path.join(diag_dir, 'v2v3_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("V2/V3 Batch Diagnosis Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Run folder: {run_path}\n")
        f.write(f"Training batch: {run_args.get('batch', 'unknown')}\n")
        f.write(f"Data path: {run_args.get('data_path', 'unknown')}\n\n")
        f.write("Raw Nuclei Protocol Distribution:\n")
        f.write(raw_nuclei.obs['protocol'].value_counts().to_string() + "\n\n")
        f.write("Raw Nuclei Meta Batch ID Distribution:\n")
        f.write(raw_nuclei.obs['meta_batch_id'].value_counts().to_string() + "\n\n")
        f.write("Raw Cell Protocol Distribution:\n")
        f.write(raw_cell.obs['protocol'].value_counts().to_string() + "\n\n")
        f.write("Raw Cell Meta Batch ID Distribution:\n")
        f.write(raw_cell.obs['meta_batch_id'].value_counts().to_string() + "\n\n")

    # ---- Step 2: Heatmaps with v2/v3 color bar ----
    print("\n" + "=" * 60)
    print("STEP 2: Creating heatmaps with v2/v3 protocol annotation")
    print("=" * 60)

    target_per_type = 50  # more cells per type to show the banding
    np.random.seed(RANDOM_SEED)

    for raw_adata, label in [(raw_nuclei, 'nuclei'), (raw_cell, 'cell')]:
        ct_arr = raw_adata.obs['cell_type'].values
        unique_cts = sorted(set(ct_arr))
        tc = {ct: min(int((ct_arr == ct).sum()), target_per_type) for ct in unique_cts}

        sub_adata = stratified_subsample_with_obs(raw_adata, target_counts=tc)
        X = sub_adata.X
        if sparse.issparse(X):
            X = X.toarray()
        if use_log1p:
            X = np.log1p(X).astype(np.float32)
        else:
            X = X.astype(np.float32)

        cell_types = sub_adata.obs['cell_type'].values
        protocols = sub_adata.obs['protocol'].values

        # Heatmap with original sample order (shows natural banding)
        plot_heatmap_with_protocol_bar(
            X, cell_types, protocols,
            f"Raw {label.title()} ({norm_label}) - Colored by 10X Protocol",
            os.path.join(diag_dir, f"raw_{label}_{norm_label}_heatmap_v2v3.png"),
        )

        # Heatmap sorted by protocol within each cell type
        sort_idx = np.lexsort((protocols, cell_types))
        plot_heatmap_with_protocol_bar(
            X[sort_idx], cell_types[sort_idx], protocols[sort_idx],
            f"Raw {label.title()} ({norm_label}) - Sorted by Protocol within Cell Type",
            os.path.join(diag_dir, f"raw_{label}_{norm_label}_heatmap_v2v3_sorted.png"),
        )

        del sub_adata, X
        gc.collect()

    # ---- Step 3: Raw data UMAP colored by v2/v3 ----
    print("\n" + "=" * 60)
    print("STEP 3: Raw data UMAP colored by v2/v3 protocol")
    print("=" * 60)

    for raw_adata, label in [(raw_nuclei, 'nuclei'), (raw_cell, 'cell')]:
        X = raw_adata.X
        if use_log1p:
            if 'log1p' in raw_adata.layers:
                X = raw_adata.layers['log1p']
            else:
                if sparse.issparse(X):
                    X = X.copy()
                    X.data = np.log1p(X.data)
                else:
                    X = np.log1p(X)

        labels = {
            'Cell Type': raw_adata.obs['cell_type'].values.tolist(),
            'Protocol': raw_adata.obs['protocol'].values.tolist(),
            'Meta Batch ID': raw_adata.obs['meta_batch_id'].values.tolist(),
        }

        compute_and_plot_umap(
            X, labels,
            f"Raw {label.title()} ({norm_label})",
            os.path.join(diag_dir, f"raw_{label}_{norm_label}_umap_v2v3.png"),
            max_cells=30000,
        )
        gc.collect()

    # Free raw data memory
    del raw_nuclei, raw_cell
    gc.collect()

    # ---- Step 4: Epoch UMAP with sample/batch coloring ----
    # Note: training uses single protocol (e.g. n3A = v3 only), so epoch
    # dumps contain only one protocol. We show sample-level (donor) coloring
    # and cell type coloring at a few epochs.
    print("\n" + "=" * 60)
    print("STEP 4: Epoch encoded UMAP with sample coloring")
    print("=" * 60)

    if not epochs:
        all_epochs = list_epoch_dumps(run_path)
        if len(all_epochs) >= 3:
            epochs = [all_epochs[0], all_epochs[len(all_epochs)//2], all_epochs[-1]]
        else:
            epochs = all_epochs
        print(f"  Auto-selected epochs: {epochs}")

    for epoch in epochs:
        fp = find_epoch_dump(run_path, epoch)
        if fp is None:
            print(f"  Epoch {epoch}: dump not found, skipping")
            continue

        print(f"\n  Processing epoch {epoch}...")
        encoded_arr, types_arr, batch_arr = load_encoded_stage1(fp, common_cell_types)

        protocol_arr = np.array([extract_protocol(b) for b in batch_arr])

        print(f"    Encoded shape: {encoded_arr.shape}")
        print(f"    Protocol distribution: {pd.Series(protocol_arr).value_counts().to_dict()}")
        print(f"    Batch distribution: {pd.Series(batch_arr).value_counts().to_dict()}")

        labels = {
            'Cell Type': types_arr.tolist(),
            'Sample (Donor)': batch_arr.tolist(),
            'Protocol': protocol_arr.tolist(),
        }

        # Encoded is already low-dim, go straight to UMAP
        print(f"    Computing UMAP on encoded space ({encoded_arr.shape[1]}D)...")
        n_cells = encoded_arr.shape[0]
        if n_cells > 30000:
            idx = np.random.choice(n_cells, 30000, replace=False)
            encoded_sub = encoded_arr[idx]
            labels_sub = {k: [v[i] for i in idx] for k, v in labels.items()}
        else:
            encoded_sub = encoded_arr
            labels_sub = labels

        reducer = umap_lib.UMAP(n_neighbors=15, min_dist=0.3, random_state=42)
        umap_coords = reducer.fit_transform(encoded_sub)

        plot_umap_panels(
            umap_coords, labels_sub,
            f"Encoded (epoch {epoch})",
            os.path.join(diag_dir, f"encoded_epoch{epoch}_umap_v2v3.png"),
        )

        del encoded_arr, types_arr, batch_arr, protocol_arr
        gc.collect()

    print("\n" + "=" * 60)
    print(f"DONE. All outputs saved to: {diag_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Diagnose v2/v3 batch effects")
    parser.add_argument("--run_path", required=True,
                        help="Full path to the run folder")
    parser.add_argument("--epochs", nargs='*', type=int, default=None,
                        help="Epochs to analyze (default: auto-detect)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: <run>/evaluation/v2v3_diagnosis)")
    args = parser.parse_args()

    if not os.path.isdir(args.run_path):
        print(f"Error: Run folder not found: {args.run_path}")
        sys.exit(1)

    print(f"Run folder: {args.run_path}")
    run_v2v3_diagnosis(args.run_path, args.epochs, args.output_dir)


if __name__ == '__main__':
    main()
