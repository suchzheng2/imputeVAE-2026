"""
Per-gene Wilcoxon rank-sum test: real cell vs imputed (cross-decoded nuclei).

For each gene, tests H0: cell and imputed distributions are identical.
Reports: p-value, effect size (rank-biserial correlation), median difference.

Usage:
    python analysis/wilcoxon_gene_test.py --run_folder run/run_2026-05-20-00-46-21_2182434
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_model_and_cross_decode(run_folder):
    """Load model, data, and produce cross-decoded output."""
    import torch
    from imputeVAE.data import load_data_for_stage3

    args_path = os.path.join(run_folder, 'arguments.json')
    with open(args_path) as f:
        saved_args = json.load(f)

    # Build args namespace
    from argparse import Namespace
    args = Namespace(**saved_args)

    # Load data
    print("[data] Loading cell and nuclei data...")
    t0 = time.time()
    result = load_data_for_stage3(args)
    cell_loader = result['cell_loader']
    nuc_loader = result['nuc_loader']
    print(f"[data] Done in {time.time()-t0:.0f}s")

    # Determine model class
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(run_folder, 'models', 'model.pt')

    scalar_cond = saved_args.get('scalar_cond', False)
    cond_encoder = saved_args.get('cond_encoder', False)
    embed_dim = saved_args.get('embed_dim', 8)

    from imputeVAE.models_cond import VAE_JSD_Cond
    if scalar_cond:
        model = VAE_JSD_Cond(
            num_feature=saved_args['num_feature'],
            num_internal=saved_args['num_internal'],
            num_latent=saved_args['num_latent'],
            embed_dim=1,
            scalar_cond=True,
            cond_encoder=cond_encoder,
        )
    else:
        model = VAE_JSD_Cond(
            num_feature=saved_args['num_feature'],
            num_internal=saved_args['num_internal'],
            num_latent=saved_args['num_latent'],
            embed_dim=embed_dim,
            cond_encoder=cond_encoder,
        )

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # Collect raw cell data
    print("[inference] Collecting raw cell data...")
    X_cell_list = []
    for batch in cell_loader:
        X_cell_list.append(batch[0].numpy())
    X_cell = np.concatenate(X_cell_list, axis=0)

    # Cross-decode nuclei → cell
    print("[inference] Cross-decoding nuclei → cell...")
    imputed_list = []
    with torch.no_grad():
        for batch in nuc_loader:
            x_nuc = batch[0].to(device)
            # Encode with nuclei modality
            if cond_encoder:
                nuc_flag = torch.zeros(x_nuc.shape[0], 1, device=device)
                encoder_input = torch.cat([x_nuc, nuc_flag], dim=1)
            else:
                encoder_input = x_nuc
            mu, log_var = model.encode(encoder_input)
            # Decode with cell modality
            cross_decoded = model.decode_cross(mu, modality_idx=0)  # 0 = cell
            imputed_list.append(cross_decoded.cpu().numpy())
    X_imputed = np.concatenate(imputed_list, axis=0)

    # Get gene names
    gene_names = result.get('gene_names', None)
    if gene_names is None:
        gene_names = [f"gene_{i}" for i in range(X_cell.shape[1])]

    # Get cell types
    ct_nuc = None
    if 'nuc_cell_types' in result:
        ct_nuc = result['nuc_cell_types']

    print(f"[data] Cell: {X_cell.shape}, Imputed: {X_imputed.shape}")
    return X_cell, X_imputed, gene_names, ct_nuc


def run_wilcoxon_tests(X_cell, X_imputed, gene_names, ct_nuc=None, output_dir=None):
    """Run per-gene Wilcoxon rank-sum test."""
    n_genes = X_cell.shape[1]
    n_cell = X_cell.shape[0]
    n_nuc = X_imputed.shape[0]

    print(f"\n[test] Running Wilcoxon rank-sum on {n_genes} genes...")
    print(f"  Cells: {n_cell}, Imputed nuclei: {n_nuc}")
    t0 = time.time()

    results = []
    for g in range(n_genes):
        cell_vals = X_cell[:, g]
        imp_vals = X_imputed[:, g]

        # Wilcoxon rank-sum (Mann-Whitney U)
        try:
            stat, pval = stats.mannwhitneyu(cell_vals, imp_vals, alternative='two-sided')
            # Rank-biserial correlation (effect size): r = 1 - 2U/(n1*n2)
            r_effect = 1 - 2 * stat / (n_cell * n_nuc)
        except ValueError:
            stat, pval, r_effect = np.nan, 1.0, 0.0

        results.append({
            'gene': gene_names[g],
            'p_value': pval,
            'effect_size': r_effect,
            'abs_effect_size': abs(r_effect),
            'median_cell': np.median(cell_vals),
            'median_imputed': np.median(imp_vals),
            'median_diff': np.median(cell_vals) - np.median(imp_vals),
            'mean_cell': np.mean(cell_vals),
            'mean_imputed': np.mean(imp_vals),
        })

    df = pd.DataFrame(results)

    # BH correction
    from statsmodels.stats.multitest import multipletests
    _, df['p_adj'], _, _ = multipletests(df['p_value'].fillna(1), method='fdr_bh')

    df = df.sort_values('p_adj')
    elapsed = time.time() - t0
    print(f"[test] Done in {elapsed:.1f}s")

    # Summary
    sig_005 = (df['p_adj'] < 0.05).sum()
    sig_001 = (df['p_adj'] < 0.01).sum()
    nonsig = (df['p_adj'] >= 0.05).sum()
    print(f"\n{'='*60}")
    print(f"  Results summary ({n_genes} genes)")
    print(f"{'='*60}")
    print(f"  Significant (p_adj < 0.05): {sig_005} ({100*sig_005/n_genes:.1f}%)")
    print(f"  Significant (p_adj < 0.01): {sig_001} ({100*sig_001/n_genes:.1f}%)")
    print(f"  Non-significant (p_adj >= 0.05): {nonsig} ({100*nonsig/n_genes:.1f}%)")
    print(f"  Median |effect size|: {df['abs_effect_size'].median():.4f}")
    print(f"  Mean |effect size|: {df['abs_effect_size'].mean():.4f}")
    print(f"  Median |median_diff|: {df['median_diff'].abs().median():.4f}")

    # Top genes by effect size
    print(f"\n  Top 20 genes by |effect size|:")
    top = df.nlargest(20, 'abs_effect_size')
    for _, row in top.iterrows():
        print(f"    {row['gene']:20s}  effect={row['effect_size']:+.4f}  "
              f"p_adj={row['p_adj']:.2e}  med_cell={row['median_cell']:.3f}  "
              f"med_imp={row['median_imputed']:.3f}")

    # Bottom genes (most similar)
    print(f"\n  Top 20 most similar genes (smallest |effect size|):")
    bot = df.nsmallest(20, 'abs_effect_size')
    for _, row in bot.iterrows():
        print(f"    {row['gene']:20s}  effect={row['effect_size']:+.4f}  "
              f"p_adj={row['p_adj']:.2e}  med_cell={row['median_cell']:.3f}  "
              f"med_imp={row['median_imputed']:.3f}")

    # Per-CT analysis if available
    if ct_nuc is not None:
        print(f"\n  Per-CT summary (fraction of genes with p_adj < 0.05):")
        cts = sorted(set(ct_nuc))
        # This would be too slow per-CT per-gene, just report counts
        print(f"  (CT-level breakdown available but skipped for speed; "
              f"overall test uses pooled samples across CTs)")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, 'wilcoxon_per_gene.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n  Saved to {csv_path}")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_folder', required=True)
    parser.add_argument('--output_dir', default=None,
                        help='Output directory (default: notes/eval_results/<run_name>/wilcoxon/)')
    args = parser.parse_args()

    run_folder = args.run_folder
    if not os.path.isabs(run_folder):
        run_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), run_folder)

    run_name = os.path.basename(run_folder)
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'notes', 'eval_results', run_name, 'wilcoxon')

    X_cell, X_imputed, gene_names, ct_nuc = load_model_and_cross_decode(run_folder)
    run_wilcoxon_tests(X_cell, X_imputed, gene_names, ct_nuc, output_dir)


if __name__ == '__main__':
    main()
