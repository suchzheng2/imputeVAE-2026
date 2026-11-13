"""
Within-CT, nonzero-only, per-gene statistics.

For each (cell type, gene) pair:
  - Subset to cells where raw > 0
  - Compute mean, std, variance of raw vs decoded
  - Compute ratios (decoded / raw)

This removes both between-CT and between-gene confounding.
"""

import argparse
import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_DIR)

from analysis.eval.registry import get_model
from analysis.eval.data import load_eval_data
from analysis.eval.inference import load_model, run_5source_inference


def within_ct_nonzero_gene_stats(raw, decoded, ct_array, common_cts, gene_names,
                                  min_nonzero=10):
    """Per-gene, within-CT, nonzero-only stats.

    Parameters
    ----------
    raw, decoded : [N, G]
    ct_array : [N]
    min_nonzero : skip if fewer than this many nonzero cells

    Returns
    -------
    DataFrame with one row per (CT, gene) pair
    """
    rows = []
    n_genes = raw.shape[1]

    for ct in common_cts:
        mask = ct_array == ct
        r_ct = raw[mask]     # [N_ct, G]
        d_ct = decoded[mask]

        for g in range(n_genes):
            r_gene = r_ct[:, g]
            d_gene = d_ct[:, g]

            nz = r_gene > 0
            n_nz = nz.sum()
            n_total = len(r_gene)

            if n_nz < min_nonzero:
                continue

            r_nz = r_gene[nz]
            d_nz = d_gene[nz]

            rows.append({
                'ct': ct,
                'gene': gene_names[g] if gene_names is not None else g,
                'gene_idx': g,
                'n_cells': n_total,
                'n_nonzero': int(n_nz),
                'frac_nonzero': n_nz / n_total,
                'raw_mean': float(r_nz.mean()),
                'raw_std': float(r_nz.std()),
                'raw_var': float(r_nz.var()),
                'raw_min': float(r_nz.min()),
                'raw_max': float(r_nz.max()),
                'dec_mean': float(d_nz.mean()),
                'dec_std': float(d_nz.std()),
                'dec_var': float(d_nz.var()),
                'dec_min': float(d_nz.min()),
                'dec_max': float(d_nz.max()),
                'mean_ratio': float(d_nz.mean() / r_nz.mean()) if r_nz.mean() > 1e-10 else np.nan,
                'std_ratio': float(d_nz.std() / r_nz.std()) if r_nz.std() > 1e-10 else np.nan,
                'var_ratio': float(d_nz.var() / r_nz.var()) if r_nz.var() > 1e-10 else np.nan,
            })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="pct5v2_cond")
    parser.add_argument("--out", type=str,
                        default="notes/experiments/zero_nonzero_analysis/")
    parser.add_argument("--also-baseline", action="store_true", default=True,
                        help="Also compute stats for CT mean baseline")
    args = parser.parse_args()

    out_dir = os.path.join(_PROJECT_DIR, args.out, args.model)
    os.makedirs(out_dir, exist_ok=True)

    import gc

    model_config = get_model(args.model)
    data = load_eval_data(model_config['gene_list'], max_per_type=99999)
    model, device = load_model(model_config)

    print(f"Model: {args.model}")
    print(f"Cell: {data.X_cell.shape}, Nuclei: {data.X_nuc.shape}")

    # Run inference and keep only what we need
    print("\nRunning inference...")
    sources = run_5source_inference(model, model_config,
                                     data.X_cell, data.X_nuc, device)
    decoded_cell = sources['decoded_cell']
    cross_decoded = sources['cross_decoded']
    del sources, model
    gc.collect()

    # --- Cell reconstruction ---
    print("\nComputing within-CT nonzero stats: cell recon...", flush=True)
    df_recon = within_ct_nonzero_gene_stats(
        data.X_cell, decoded_cell,
        data.ct_cell, data.common_cts, data.gene_names
    )
    df_recon['source'] = 'cell_recon'
    del decoded_cell
    gc.collect()

    # --- CT mean baseline ---
    print("Computing within-CT nonzero stats: CT mean baseline...", flush=True)
    ct_mean_decoded = np.zeros_like(data.X_cell)
    for ct in data.common_cts:
        m = data.ct_cell == ct
        if m.sum() > 0:
            ct_mean_decoded[m] = data.X_cell[m].mean(axis=0)

    df_baseline = within_ct_nonzero_gene_stats(
        data.X_cell, ct_mean_decoded,
        data.ct_cell, data.common_cts, data.gene_names
    )
    df_baseline['source'] = 'ct_mean_baseline'
    del ct_mean_decoded
    gc.collect()

    # --- Cross-decoded (vs raw nuc) ---
    print("Computing within-CT nonzero stats: cross-decoded (vs raw nuc)...", flush=True)
    df_cross = within_ct_nonzero_gene_stats(
        data.X_nuc, cross_decoded,
        data.ct_nuc, data.common_cts, data.gene_names
    )
    df_cross['source'] = 'cross_decoded'
    del cross_decoded
    gc.collect()

    # --- Combine and save ---
    df_all = pd.concat([df_recon, df_baseline, df_cross], ignore_index=True)
    out_path = os.path.join(out_dir, "within_ct_nonzero_gene_stats.csv")
    df_all.to_csv(out_path, index=False)

    # --- Print summary ---
    print("\n" + "=" * 70)
    print("SUMMARY: median across all (CT, gene) pairs")
    print("=" * 70)
    for source in ['cell_recon', 'ct_mean_baseline', 'cross_decoded']:
        sub = df_all[df_all['source'] == source]
        print(f"\n  [{source}]  n_pairs = {len(sub)}")
        print(f"    mean_ratio:  median={sub['mean_ratio'].median():.4f}  "
              f"mean={sub['mean_ratio'].mean():.4f}")
        print(f"    std_ratio:   median={sub['std_ratio'].median():.4f}  "
              f"mean={sub['std_ratio'].mean():.4f}")
        print(f"    var_ratio:   median={sub['var_ratio'].median():.4f}  "
              f"mean={sub['var_ratio'].mean():.4f}")
        print(f"    raw_std:     median={sub['raw_std'].median():.4f}")
        print(f"    dec_std:     median={sub['dec_std'].median():.4f}")

    # --- Breakdown by sparsity ---
    print("\n" + "=" * 70)
    print("BY GENE SPARSITY (frac_nonzero buckets)")
    print("=" * 70)
    bins = [(0, 0.1, '<10%'), (0.1, 0.3, '10-30%'), (0.3, 0.7, '30-70%'), (0.7, 1.01, '>70%')]
    for source in ['cell_recon', 'ct_mean_baseline']:
        sub = df_all[df_all['source'] == source]
        print(f"\n  [{source}]")
        print(f"  {'bucket':>10s}  {'n':>6s}  {'mean_ratio':>10s}  {'std_ratio':>10s}  {'var_ratio':>10s}  {'raw_std':>8s}  {'dec_std':>8s}")
        for lo, hi, label in bins:
            b = sub[(sub['frac_nonzero'] >= lo) & (sub['frac_nonzero'] < hi)]
            if len(b) == 0:
                continue
            print(f"  {label:>10s}  {len(b):>6d}  {b['mean_ratio'].median():>10.4f}  "
                  f"{b['std_ratio'].median():>10.4f}  {b['var_ratio'].median():>10.4f}  "
                  f"{b['raw_std'].median():>8.4f}  {b['dec_std'].median():>8.4f}")

    # --- Per-CT summary (median across genes within each CT) ---
    print("\n" + "=" * 70)
    print("PER CELL-TYPE SUMMARY (median across genes)")
    print("=" * 70)
    for source in ['cell_recon', 'ct_mean_baseline', 'cross_decoded']:
        sub = df_all[df_all['source'] == source]
        print(f"\n  [{source}]")
        print(f"  {'CT':>12s}  {'n_genes':>7s}  {'mean_ratio':>10s}  {'std_ratio':>10s}  {'var_ratio':>10s}  {'raw_std':>8s}  {'dec_std':>8s}")
        for ct in sorted(sub['ct'].unique()):
            ct_sub = sub[sub['ct'] == ct]
            print(f"  {ct:>12s}  {len(ct_sub):>7d}  {ct_sub['mean_ratio'].median():>10.4f}  "
                  f"{ct_sub['std_ratio'].median():>10.4f}  {ct_sub['var_ratio'].median():>10.4f}  "
                  f"{ct_sub['raw_std'].median():>8.4f}  {ct_sub['dec_std'].median():>8.4f}")

    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
