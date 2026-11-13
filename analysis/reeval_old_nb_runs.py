#!/usr/bin/env python3
"""
Re-evaluate Feb 19 NB stage 1 runs with correct normalization.

Original eval compared NB decoded (count-space) vs raw (log1p-space) → 0.619.
This script computes correlation in BOTH spaces:
  A) log1p(decoded) vs log1p(raw)  — comparable to MSE results
  B) decoded vs raw (count space)  — NB's native space

Uses the same methodology as the original stage-1-correlation-analysis.py:
  - Cell-type-level mean expression
  - Pearson correlation across all genes per cell type
  - Average across cell types
"""

import json
import os
import gc
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, sparse
from typing import Dict, Tuple, List
import warnings
import argparse

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--cell_run_path", type=str, required=True)
parser.add_argument("--nuclei_run_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--subsample_fraction", type=float, default=0.1)
args = parser.parse_args()

BASE_DATA_DIR = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi"
RAW_CELL_PATH = os.path.join(BASE_DATA_DIR, "10X_cells_v2_v3_AIBS", "data.h5ad")
RAW_NUCLEI_PATH = os.path.join(BASE_DATA_DIR, "10X_nuclei_v2_v3_AIBS", "data.h5ad")

RANDOM_SEED = 42


def print_memory_usage(label=""):
    import psutil
    mem_gb = psutil.Process(os.getpid()).memory_info().rss / 1024**3
    print(f"[MEMORY] {label}: {mem_gb:.2f} GB")


# =============================================================================
# Data Loading
# =============================================================================

def load_common_filters():
    base = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation"
    common_genes = set(pd.read_csv(f"{base}/common_genes.csv")['common_genes'])
    common_cell_types = set(pd.read_csv(f"{base}/common_cell_types.csv")['common_cell_types'])
    mitocarta_genes = set(pd.read_csv(f"{base}/mouse.miticarta.analysis/Mouse.MitoCarta.csv")['Symbol'])
    return common_genes, common_cell_types, mitocarta_genes


def load_raw_data(path, common_genes, common_cell_types, mitocarta_genes,
                  detection_rate_threshold=0.2):
    print(f"Loading: {path}")
    adata = ad.read_h5ad(path)
    adata = adata[adata.obs['cell_type'].isin(common_cell_types)].copy()
    gene_mask = (adata.var['gene_names'].isin(common_genes) &
                 ~adata.var['gene_names'].isin(mitocarta_genes))
    adata = adata[:, gene_mask].copy()
    if detection_rate_threshold < 1.0:
        det = (adata.X > 0).sum(axis=1) / adata.X.shape[1]
        if hasattr(det, 'A1'):
            det = det.A1
        adata = adata[det >= detection_rate_threshold].copy()
    adata = adata[:, adata.var['gene_names'].argsort()].copy()
    print(f"  Shape: {adata.shape}")
    return adata


def load_decoded_stage1(train_fp, common_cell_types):
    """Load decoded data from stage 1 JSON epoch dump."""
    print(f"Loading: {train_fp}")
    print_memory_usage("Before JSON load")
    with open(train_fp) as f:
        data = json.load(f)
    print_memory_usage("After JSON load")

    decoded_list, types_list = [], []
    for ct in sorted(data.keys()):
        if ct not in common_cell_types:
            continue
        ct_dec = []
        for bid in data[ct]:
            ct_dec.append(np.array(data[ct][bid]['decoded'], dtype=np.float32))
        if ct_dec:
            ct_dec = np.vstack(ct_dec)
            decoded_list.append(ct_dec)
            types_list.extend([ct] * ct_dec.shape[0])

    del data
    gc.collect()

    decoded = np.vstack(decoded_list)
    types = np.array(types_list)
    print(f"  Decoded shape: {decoded.shape}, cell types: {len(np.unique(types))}")
    print_memory_usage("After decode processing")
    return decoded, types


def compute_celltype_mean(data, cell_types, gene_names=None):
    unique_cts = np.unique(cell_types)
    means = []
    for ct in unique_cts:
        m = data[cell_types == ct].mean(axis=0)
        if hasattr(m, 'A1'):
            m = m.A1
        means.append(m)
    df = pd.DataFrame(means, index=unique_cts)
    if gene_names is not None:
        df.columns = gene_names
    return df


def compute_pairwise_corr(mean_dict, cell_type):
    """Pearson r between all source pairs for one cell type."""
    sources = list(mean_dict.keys())
    n = len(sources)
    mat = np.full((n, n), np.nan)
    for i, s1 in enumerate(sources):
        for j, s2 in enumerate(sources):
            df1, df2 = mean_dict[s1], mean_dict[s2]
            if cell_type not in df1.index or cell_type not in df2.index:
                continue
            if df1.columns.dtype == 'object' and df2.columns.dtype == 'object':
                cg = sorted(set(df1.columns) & set(df2.columns))
                if not cg:
                    continue
                v1 = df1.loc[cell_type, cg].values.astype(float)
                v2 = df2.loc[cell_type, cg].values.astype(float)
            else:
                v1 = df1.loc[cell_type].values.astype(float)
                v2 = df2.loc[cell_type].values.astype(float)
            mat[i, j] = stats.pearsonr(v1, v2)[0]
    return pd.DataFrame(mat, index=sources, columns=sources)


def compute_all_correlations(mean_dict):
    all_cts = None
    for df in mean_dict.values():
        s = set(df.index)
        all_cts = s if all_cts is None else all_cts & s
    all_cts = sorted(all_cts)
    print(f"  Cell types in all datasets: {len(all_cts)}: {all_cts}")

    per_ct, matrices = {}, []
    for ct in all_cts:
        m = compute_pairwise_corr(mean_dict, ct)
        per_ct[ct] = m
        matrices.append(m.values)

    avg = pd.DataFrame(np.nanmean(matrices, axis=0),
                       index=list(mean_dict.keys()),
                       columns=list(mean_dict.keys()))
    return per_ct, avg


def plot_heatmap(corr, title, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt='.3f', cmap='RdYlBu_r',
                center=0.5, vmin=0, vmax=1, square=True, linewidths=0.5, ax=ax)
    ax.set_title(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# =============================================================================
# Main
# =============================================================================

def main():
    os.makedirs(args.output_dir, exist_ok=True)

    # Load filters
    common_genes, common_cell_types, mitocarta_genes = load_common_filters()
    print(f"Common genes: {len(common_genes)}, cell types: {len(common_cell_types)}")

    # Load decoded data (NB model outputs — in count space)
    cell_dec, cell_dec_types = load_decoded_stage1(
        os.path.join(args.cell_run_path, "epoch_dump/train.json"), common_cell_types)
    nuclei_dec, nuclei_dec_types = load_decoded_stage1(
        os.path.join(args.nuclei_run_path, "epoch_dump/train.json"), common_cell_types)

    # Load raw reference data
    raw_cell_adata = load_raw_data(RAW_CELL_PATH, common_genes, common_cell_types, mitocarta_genes)
    raw_nuclei_adata = load_raw_data(RAW_NUCLEI_PATH, common_genes, common_cell_types, mitocarta_genes)

    # Subsample raw
    np.random.seed(RANDOM_SEED)
    for adata, label in [(raw_cell_adata, 'cell'), (raw_nuclei_adata, 'nuclei')]:
        n = adata.n_obs
        idx = np.random.choice(n, int(n * args.subsample_fraction), replace=False)
        if label == 'cell':
            raw_cell_adata = adata[idx].copy()
        else:
            raw_nuclei_adata = adata[idx].copy()

    print(f"Raw cell subsampled: {raw_cell_adata.shape}")
    print(f"Raw nuclei subsampled: {raw_nuclei_adata.shape}")

    # Gene names from raw data (sorted, filtered)
    raw_cell_genes = list(raw_cell_adata.var['gene_names'])
    raw_nuclei_genes = list(raw_nuclei_adata.var['gene_names'])
    raw_common_genes = sorted(set(raw_cell_genes) & set(raw_nuclei_genes))
    n_raw = len(raw_common_genes)
    print(f"Raw common genes: {n_raw}")

    # Decoded gene count
    n_cell_dec_genes = cell_dec.shape[1]
    n_nuclei_dec_genes = nuclei_dec.shape[1]
    print(f"Cell decoded genes: {n_cell_dec_genes}, Nuclei decoded genes: {n_nuclei_dec_genes}")

    # Check gene alignment
    gene_match = (n_cell_dec_genes == n_raw and n_nuclei_dec_genes == n_raw)
    if gene_match:
        print("Gene counts match — positional alignment is valid.")
    else:
        print(f"WARNING: Gene count mismatch! decoded ({n_cell_dec_genes}/{n_nuclei_dec_genes}) vs raw ({n_raw})")
        print("Positional alignment may be incorrect — results should be interpreted with caution.")

    # =========================================================================
    # Compute means and run both comparisons
    # =========================================================================

    results = {}

    for space_label, transform_decoded, transform_raw in [
        ("log1p", lambda x: np.log1p(np.clip(x, 0, None)), "log1p"),
        ("count", lambda x: np.clip(x, 0, None), "raw"),
    ]:
        print(f"\n{'='*80}")
        print(f"COMPARISON IN {space_label.upper()} SPACE")
        print(f"{'='*80}")

        # Transform decoded
        cell_dec_t = transform_decoded(cell_dec)
        nuclei_dec_t = transform_decoded(nuclei_dec)

        # Decoded means
        cell_dec_mean = compute_celltype_mean(cell_dec_t, cell_dec_types)
        nuclei_dec_mean = compute_celltype_mean(nuclei_dec_t, nuclei_dec_types)

        del cell_dec_t, nuclei_dec_t
        gc.collect()

        # Raw reference
        if transform_raw == "log1p":
            raw_cell_X = (raw_cell_adata.layers['log1p'] if 'log1p' in raw_cell_adata.layers
                          else np.log1p(raw_cell_adata.X))
            raw_nuclei_X = (raw_nuclei_adata.layers['log1p'] if 'log1p' in raw_nuclei_adata.layers
                            else np.log1p(raw_nuclei_adata.X))
        else:
            raw_cell_X = raw_cell_adata.X
            raw_nuclei_X = raw_nuclei_adata.X

        if hasattr(raw_cell_X, 'toarray'):
            raw_cell_X = raw_cell_X.toarray()
        if hasattr(raw_nuclei_X, 'toarray'):
            raw_nuclei_X = raw_nuclei_X.toarray()

        raw_cell_mean = compute_celltype_mean(
            raw_cell_X, raw_cell_adata.obs['cell_type'].values, gene_names=raw_cell_genes)
        raw_nuclei_mean = compute_celltype_mean(
            raw_nuclei_X, raw_nuclei_adata.obs['cell_type'].values, gene_names=raw_nuclei_genes)

        del raw_cell_X, raw_nuclei_X
        gc.collect()

        # Subset raw to common genes
        raw_cell_mean_c = raw_cell_mean[raw_common_genes]
        raw_nuclei_mean_c = raw_nuclei_mean[raw_common_genes]

        # Align decoded — assign gene names positionally
        n_use_cell = min(n_cell_dec_genes, n_raw)
        cell_dec_mean_c = cell_dec_mean.iloc[:, :n_use_cell].copy()
        cell_dec_mean_c.columns = raw_common_genes[:n_use_cell]

        n_use_nuc = min(n_nuclei_dec_genes, n_raw)
        nuclei_dec_mean_c = nuclei_dec_mean.iloc[:, :n_use_nuc].copy()
        nuclei_dec_mean_c.columns = raw_common_genes[:n_use_nuc]

        # Decoder scale stats
        print(f"\n  Decoder output scale ({space_label}):")
        print(f"    Cell decoded — mean: {cell_dec_mean.values.mean():.4f}, "
              f"std: {cell_dec_mean.values.std():.4f}, "
              f"min: {cell_dec_mean.values.min():.4f}, max: {cell_dec_mean.values.max():.4f}")
        print(f"    Nuclei decoded — mean: {nuclei_dec_mean.values.mean():.4f}, "
              f"std: {nuclei_dec_mean.values.std():.4f}, "
              f"min: {nuclei_dec_mean.values.min():.4f}, max: {nuclei_dec_mean.values.max():.4f}")
        print(f"    Raw cell — mean: {raw_cell_mean_c.values.mean():.4f}, "
              f"std: {raw_cell_mean_c.values.std():.4f}")
        print(f"    Raw nuclei — mean: {raw_nuclei_mean_c.values.mean():.4f}, "
              f"std: {raw_nuclei_mean_c.values.std():.4f}")

        # 4x4 correlation
        combined = {
            'raw_cell': raw_cell_mean_c,
            'raw_nuclei': raw_nuclei_mean_c,
            'cell_decoded': cell_dec_mean_c,
            'nuclei_decoded': nuclei_dec_mean_c,
        }

        per_ct, avg = compute_all_correlations(combined)

        # Print results
        print(f"\n  Average correlation matrix ({space_label} space):")
        print(avg.to_string())

        # Shift scores
        print(f"\n  Shift scores ({space_label} space):")
        for dec_label in ['cell_decoded', 'nuclei_decoded']:
            shifts = []
            for ct, m in per_ct.items():
                s = m.loc[dec_label, 'raw_cell'] - m.loc[dec_label, 'raw_nuclei']
                shifts.append(s)
            mean_shift = np.mean(shifts)
            to_cell = avg.loc[dec_label, 'raw_cell']
            to_nuclei = avg.loc[dec_label, 'raw_nuclei']
            print(f"    {dec_label}: corr_to_cell={to_cell:.4f}, corr_to_nuclei={to_nuclei:.4f}, "
                  f"mean_shift={mean_shift:.4f}")

        # Save
        avg.to_csv(os.path.join(args.output_dir, f"correlation_matrix_{space_label}.csv"))
        plot_heatmap(avg,
                     f"NB Stage 1 Correlation ({space_label} space)\n"
                     f"Cell: {os.path.basename(args.cell_run_path)}\n"
                     f"Nuclei: {os.path.basename(args.nuclei_run_path)}",
                     os.path.join(args.output_dir, f"correlation_heatmap_{space_label}.png"))

        # Per cell type detail
        per_ct_rows = []
        for ct, m in per_ct.items():
            row = {'cell_type': ct}
            for s1 in m.index:
                for s2 in m.columns:
                    row[f"{s1}_vs_{s2}"] = m.loc[s1, s2]
            per_ct_rows.append(row)
        pd.DataFrame(per_ct_rows).to_csv(
            os.path.join(args.output_dir, f"per_celltype_correlations_{space_label}.csv"), index=False)

        results[space_label] = avg

    # =========================================================================
    # Summary comparison
    # =========================================================================
    print("\n" + "="*80)
    print("SUMMARY: log1p vs count space comparison")
    print("="*80)

    for metric in ['cell_decoded', 'nuclei_decoded']:
        print(f"\n  {metric}:")
        for space in ['log1p', 'count']:
            avg = results[space]
            to_cell = avg.loc[metric, 'raw_cell']
            to_nuc = avg.loc[metric, 'raw_nuclei']
            print(f"    {space:6s} space: corr_to_cell={to_cell:.4f}, corr_to_nuclei={to_nuc:.4f}, "
                  f"shift={to_cell - to_nuc:.4f}")

    print(f"\n  Original eval (log1p raw vs count decoded): cell_decoded↔raw_cell = 0.619")
    print(f"  raw_cell↔raw_nuclei baseline: {results['log1p'].loc['raw_cell', 'raw_nuclei']:.4f}")

    # Save summary
    with open(os.path.join(args.output_dir, "summary.txt"), 'w') as f:
        f.write("NB Stage 1 Re-evaluation: log1p vs count space\n")
        f.write(f"Cell run: {args.cell_run_path}\n")
        f.write(f"Nuclei run: {args.nuclei_run_path}\n\n")
        for space in ['log1p', 'count']:
            f.write(f"\n=== {space.upper()} SPACE ===\n")
            f.write(results[space].to_string())
            f.write("\n")

    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
