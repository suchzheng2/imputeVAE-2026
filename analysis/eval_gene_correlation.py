#!/usr/bin/env python3
"""
Gene-Level Correlation Analysis module.

For each epoch, computes per-gene Pearson correlation between decoded and
raw_cell (using cell-type means across cell types), ranks genes by quality,
and identifies top-10 best and worst correlated genes.

Usage:
    python eval_gene_correlation.py --run_folder <name> --modality <cell|nuclei> \
        [--epochs 0,50,100]
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
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def compute_per_gene_correlation(decoded_mean, raw_mean, gene_names):
    """Compute per-gene Pearson correlation across cell types.

    Both inputs are DataFrames with cell types as rows and genes as columns.
    Returns a DataFrame with one row per gene.
    """
    common_cts = sorted(set(decoded_mean.index) & set(raw_mean.index))
    if not common_cts:
        return pd.DataFrame()

    results = []
    for g_idx, gene in enumerate(gene_names):
        if gene in raw_mean.columns:
            dec_vals = decoded_mean.iloc[:, g_idx].loc[common_cts].values.astype(float)
            raw_vals = raw_mean.loc[common_cts, gene].values.astype(float)

            if np.std(dec_vals) == 0 or np.std(raw_vals) == 0:
                r = 0.0
            else:
                r, _ = stats.pearsonr(dec_vals, raw_vals)

            results.append({
                'gene_name': gene,
                'corr': r,
                'mean_expr_decoded': float(np.mean(dec_vals)),
                'mean_expr_raw': float(np.mean(raw_vals)),
            })

    return pd.DataFrame(results)


def _run_gene_correlation_stage3(run_folder, available_epochs, gene_corr_base, use_log1p,
                                  common_cell_types, raw_cell_mean, raw_nuclei_mean,
                                  raw_common_genes, raw_cell_gene_names, raw_nuclei_gene_names):
    """Run gene-level correlation for stage3_v2 with both decoded modalities."""
    from eval_utils import load_both_decoded_stage3

    for epoch in available_epochs:
        fp = find_epoch_dump(run_folder, epoch)
        if fp is None:
            continue

        print(f"\n  Epoch {epoch} (stage3 4-source):")
        epoch_dir = os.path.join(gene_corr_base, f'epoch_{epoch}')
        os.makedirs(epoch_dir, exist_ok=True)

        cell_dec, cell_dec_types, nuclei_dec, nuclei_dec_types = load_both_decoded_stage3(
            fp, common_cell_types
        )

        for dec_arr, dec_types, dec_label in [
            (cell_dec, cell_dec_types, 'decoded_cell'),
            (nuclei_dec, nuclei_dec_types, 'decoded_nuclei'),
        ]:
            decoded_mean = compute_celltype_mean(dec_arr, dec_types)

            n_dec = decoded_mean.shape[1]
            n_raw = len(raw_common_genes)
            n_use = min(n_dec, n_raw)
            gene_names_used = raw_common_genes[:n_use]
            decoded_mean_c = decoded_mean.iloc[:, :n_use].copy()
            decoded_mean_c.columns = gene_names_used

            gene_corr_cell = compute_per_gene_correlation(decoded_mean_c, raw_cell_mean, gene_names_used)
            if not gene_corr_cell.empty:
                gene_corr_cell = gene_corr_cell.rename(columns={'corr': 'corr_vs_raw_cell',
                                                                  'mean_expr_raw': 'mean_expr_raw_cell'})

            gene_corr_nuclei = compute_per_gene_correlation(decoded_mean_c, raw_nuclei_mean, gene_names_used)
            if not gene_corr_nuclei.empty:
                gene_corr_nuclei = gene_corr_nuclei.rename(columns={'corr': 'corr_vs_raw_nuclei',
                                                                      'mean_expr_raw': 'mean_expr_raw_nuclei'})

            if not gene_corr_cell.empty and not gene_corr_nuclei.empty:
                gene_df = gene_corr_cell.merge(
                    gene_corr_nuclei[['gene_name', 'corr_vs_raw_nuclei', 'mean_expr_raw_nuclei']],
                    on='gene_name', how='outer'
                )
            elif not gene_corr_cell.empty:
                gene_df = gene_corr_cell
            else:
                gene_df = gene_corr_nuclei

            gene_df = gene_df.sort_values('corr_vs_raw_cell', ascending=False)
            gene_df.to_csv(os.path.join(epoch_dir, f'{dec_label}_gene_correlation.csv'), index=False)

            top10 = gene_df.head(10)
            bottom10 = gene_df.tail(10)
            with open(os.path.join(epoch_dir, f'{dec_label}_top_bottom_genes.txt'), 'w') as f:
                f.write(f"Top 10 best correlated genes - {dec_label} (epoch {epoch}):\n")
                f.write(top10.to_string(index=False))
                f.write(f"\n\nBottom 10 worst correlated genes - {dec_label} (epoch {epoch}):\n")
                f.write(bottom10.to_string(index=False))

            # Histogram
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            corr_vals = gene_df['corr_vs_raw_cell'].dropna()
            axes[0].hist(corr_vals, bins=50, alpha=0.7, color='steelblue', edgecolor='none')
            axes[0].axvline(x=corr_vals.median(), color='red', linestyle='--',
                            label=f'Median: {corr_vals.median():.3f}')
            axes[0].set_xlabel('Pearson Correlation')
            axes[0].set_ylabel('Number of Genes')
            axes[0].set_title(f'{dec_label} vs Raw Cell (epoch {epoch})')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            if 'corr_vs_raw_nuclei' in gene_df.columns:
                corr_vals_n = gene_df['corr_vs_raw_nuclei'].dropna()
                axes[1].hist(corr_vals_n, bins=50, alpha=0.7, color='coral', edgecolor='none')
                axes[1].axvline(x=corr_vals_n.median(), color='red', linestyle='--',
                                label=f'Median: {corr_vals_n.median():.3f}')
                axes[1].set_xlabel('Pearson Correlation')
                axes[1].set_ylabel('Number of Genes')
                axes[1].set_title(f'{dec_label} vs Raw Nuclei (epoch {epoch})')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

            fig.tight_layout()
            fig.savefig(os.path.join(epoch_dir, f'{dec_label}_gene_correlation_histogram.png'), dpi=150)
            plt.close(fig)

            print(f"    {dec_label}: {len(gene_df)} genes, median corr(cell): {corr_vals.median():.4f}")

            del decoded_mean, decoded_mean_c
            gc.collect()

        del cell_dec, nuclei_dec
        gc.collect()

    return {
        'status': 'success',
        'epochs_analyzed': available_epochs,
    }


def run_gene_correlation_analysis(run_folder, modality, epochs=None,
                                   raw_cell_adata=None, raw_nuclei_adata=None,
                                   output_dir=None):
    """Run gene-level correlation analysis across epochs.

    Returns dict with summary info for the manifest.
    """
    eval_dir = get_eval_dir(run_folder)
    gene_corr_base = output_dir or os.path.join(eval_dir, 'gene_correlation')

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

    # Compute raw means with gene names
    raw_cell_sub = stratified_subsample_adata(raw_cell_adata, SUBSAMPLE_FRACTION, seed=RANDOM_SEED)
    raw_cell_X = get_raw_data_for_correlation(raw_cell_sub, use_log1p)
    raw_cell_gene_names = list(raw_cell_sub.var['gene_names'])
    raw_cell_mean = compute_celltype_mean(raw_cell_X, raw_cell_sub.obs['cell_type'].values, raw_cell_gene_names)
    del raw_cell_X, raw_cell_sub; gc.collect()

    raw_nuclei_sub = stratified_subsample_adata(raw_nuclei_adata, SUBSAMPLE_FRACTION, seed=RANDOM_SEED)
    raw_nuclei_X = get_raw_data_for_correlation(raw_nuclei_sub, use_log1p)
    raw_nuclei_gene_names = list(raw_nuclei_sub.var['gene_names'])
    raw_nuclei_mean = compute_celltype_mean(raw_nuclei_X, raw_nuclei_sub.obs['cell_type'].values, raw_nuclei_gene_names)
    del raw_nuclei_X, raw_nuclei_sub; gc.collect()

    raw_common_genes = sorted(set(raw_cell_gene_names) & set(raw_nuclei_gene_names))

    is_stage3 = is_stage3_run(run_args)
    if is_stage3:
        return _run_gene_correlation_stage3(
            run_folder, available_epochs, gene_corr_base, use_log1p,
            common_cell_types, raw_cell_mean, raw_nuclei_mean,
            raw_common_genes, raw_cell_gene_names, raw_nuclei_gene_names,
        )

    for epoch in available_epochs:
        fp = find_epoch_dump(run_folder, epoch)
        if fp is None:
            continue

        print(f"\n  Epoch {epoch}:")
        epoch_dir = os.path.join(gene_corr_base, f'epoch_{epoch}')
        os.makedirs(epoch_dir, exist_ok=True)

        decoded_arr, decoded_types = load_decoded_stage1(fp, common_cell_types, modality=modality)
        decoded_mean = compute_celltype_mean(decoded_arr, decoded_types)

        n_dec = decoded_mean.shape[1]
        n_raw = len(raw_common_genes)
        n_use = min(n_dec, n_raw)
        gene_names_used = raw_common_genes[:n_use]
        decoded_mean_c = decoded_mean.iloc[:, :n_use].copy()
        decoded_mean_c.columns = gene_names_used

        # Per-gene correlation vs raw_cell
        gene_corr_cell = compute_per_gene_correlation(decoded_mean_c, raw_cell_mean, gene_names_used)
        if not gene_corr_cell.empty:
            gene_corr_cell = gene_corr_cell.rename(columns={'corr': 'corr_vs_raw_cell',
                                                              'mean_expr_raw': 'mean_expr_raw_cell'})

        # Per-gene correlation vs raw_nuclei
        gene_corr_nuclei = compute_per_gene_correlation(decoded_mean_c, raw_nuclei_mean, gene_names_used)
        if not gene_corr_nuclei.empty:
            gene_corr_nuclei = gene_corr_nuclei.rename(columns={'corr': 'corr_vs_raw_nuclei',
                                                                  'mean_expr_raw': 'mean_expr_raw_nuclei'})

        # Merge
        if not gene_corr_cell.empty and not gene_corr_nuclei.empty:
            gene_df = gene_corr_cell.merge(
                gene_corr_nuclei[['gene_name', 'corr_vs_raw_nuclei', 'mean_expr_raw_nuclei']],
                on='gene_name', how='outer'
            )
        elif not gene_corr_cell.empty:
            gene_df = gene_corr_cell
        else:
            gene_df = gene_corr_nuclei

        gene_df = gene_df.sort_values('corr_vs_raw_cell', ascending=False)
        gene_df.to_csv(os.path.join(epoch_dir, 'gene_correlation.csv'), index=False)

        # Top/bottom genes
        top10 = gene_df.head(10)
        bottom10 = gene_df.tail(10)

        with open(os.path.join(epoch_dir, 'top_bottom_genes.txt'), 'w') as f:
            f.write(f"Top 10 best correlated genes (epoch {epoch}):\n")
            f.write(top10.to_string(index=False))
            f.write(f"\n\nBottom 10 worst correlated genes (epoch {epoch}):\n")
            f.write(bottom10.to_string(index=False))

        # Histogram of per-gene correlations
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        corr_vals = gene_df['corr_vs_raw_cell'].dropna()
        ax.hist(corr_vals, bins=50, alpha=0.7, color='steelblue', edgecolor='none')
        ax.axvline(x=corr_vals.median(), color='red', linestyle='--', label=f'Median: {corr_vals.median():.3f}')
        ax.set_xlabel('Pearson Correlation')
        ax.set_ylabel('Number of Genes')
        ax.set_title(f'Gene Correlation vs Raw Cell (epoch {epoch})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        if 'corr_vs_raw_nuclei' in gene_df.columns:
            corr_vals_n = gene_df['corr_vs_raw_nuclei'].dropna()
            ax.hist(corr_vals_n, bins=50, alpha=0.7, color='coral', edgecolor='none')
            ax.axvline(x=corr_vals_n.median(), color='red', linestyle='--', label=f'Median: {corr_vals_n.median():.3f}')
            ax.set_xlabel('Pearson Correlation')
            ax.set_ylabel('Number of Genes')
            ax.set_title(f'Gene Correlation vs Raw Nuclei (epoch {epoch})')
            ax.legend()
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(epoch_dir, 'gene_correlation_histogram.png'), dpi=150)
        plt.close(fig)

        # Scatter: correlation vs mean expression
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(gene_df['mean_expr_decoded'], gene_df['corr_vs_raw_cell'],
                   alpha=0.3, s=10, c='steelblue')
        ax.set_xlabel('Mean Decoded Expression')
        ax.set_ylabel('Correlation vs Raw Cell')
        ax.set_title(f'Gene Correlation vs Mean Expression (epoch {epoch})')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(epoch_dir, 'gene_corr_vs_expression.png'), dpi=150)
        plt.close(fig)

        print(f"    {len(gene_df)} genes analyzed, median corr(cell): {corr_vals.median():.4f}")

        del decoded_arr, decoded_types, decoded_mean, decoded_mean_c
        gc.collect()

    return {
        'status': 'success',
        'epochs_analyzed': available_epochs,
    }


def main():
    parser = argparse.ArgumentParser(description="Gene-level correlation analysis")
    parser.add_argument("--run_folder", required=True, help="Run folder name")
    parser.add_argument("--modality", required=True, choices=['cell', 'nuclei'])
    parser.add_argument("--epochs", type=str, default=None,
                        help="Comma-separated epochs (default: all)")
    args = parser.parse_args()

    run_folder = find_run_folder(args.run_folder)
    epochs = [int(e) for e in args.epochs.split(',')] if args.epochs else None

    print(f"Run folder: {run_folder}")
    print("Running gene-level correlation analysis...")
    results = run_gene_correlation_analysis(run_folder, args.modality, epochs=epochs)
    print(f"Done. Results: {results}")


if __name__ == '__main__':
    main()
