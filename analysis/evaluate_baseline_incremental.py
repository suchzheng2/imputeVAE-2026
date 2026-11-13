"""
Incremental baseline evaluation — runs each metric group separately and saves
results after each step. This way partial results are available even if a
later step times out.

Usage:
    python analysis/evaluate_baseline_incremental.py \
        --imputed_npy run/scvi_baseline/imputed_cell_from_nucleus.npy \
        --obs_csv run/scvi_baseline/nucleus_obs.csv \
        --name scvi \
        --output_dir run/scvi_baseline/eval
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse, stats

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from analysis.eval.metrics.ct_level import (
    compute_celltype_means, gene_correlation, r_squared,
    shift_scores, classification, de_preservation, coexpression,
)
from analysis.eval.metrics.sample_level import (
    variance_ratio, within_ct_spearman, per_cell_correlation,
)
from analysis.evaluate_baseline import (
    load_raw_data, per_gene_correlation_allcells, wasserstein_per_gene,
    delta_vs_ground_truth, compartment_scores, ct_specific_delta,
    DEFAULT_GENE_LIST,
)


def save_results(results, output_dir, name):
    """Save current results dict to JSON (overwrites)."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"eval_{name}.json")
    serializable = {}
    for k, v in results.items():
        if isinstance(v, dict):
            serializable[k] = {kk: vv for kk, vv in v.items()
                              if not isinstance(vv, np.ndarray)}
        else:
            serializable[k] = v
    with open(out_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"  -> Saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imputed_npy", type=str, required=True)
    parser.add_argument("--obs_csv", type=str, default=None)
    parser.add_argument("--name", type=str, default="baseline")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--gene_list", type=str, default=DEFAULT_GENE_LIST)
    parser.add_argument("--norm", type=str, default="log1p",
                        help="Normalization for ground truth: log1p (default) or cpm")
    parser.add_argument("--skip", type=str, default="",
                        help="Comma-separated steps to skip: clf,de,coexp,sample,labelfree,bio")
    args = parser.parse_args()

    skip = set(args.skip.split(",")) if args.skip else set()
    out_dir = args.output_dir or os.path.dirname(args.imputed_npy)
    t0 = time.time()

    # Load data
    print(f"Loading imputed: {args.imputed_npy}")
    imputed = np.load(args.imputed_npy).astype(np.float32)
    print(f"  Shape: {imputed.shape}")

    if args.obs_csv:
        obs = pd.read_csv(args.obs_csv, index_col=0)
        ct_nuc_imputed = obs['cell_type'].values.astype(str)
    else:
        ct_nuc_imputed = None

    print(f"Loading raw data (norm={args.norm})...")
    adata_cell, adata_nuc = load_raw_data(args.gene_list, normalization=args.norm)

    if ct_nuc_imputed is None:
        ct_nuc_imputed = adata_nuc.obs['cell_type'].values.astype(str)

    X_cell = adata_cell.X if not sparse.issparse(adata_cell.X) else adata_cell.X.toarray()
    X_nuc = adata_nuc.X if not sparse.issparse(adata_nuc.X) else adata_nuc.X.toarray()
    ct_cell = adata_cell.obs['cell_type'].values.astype(str)
    ct_nuc = adata_nuc.obs['cell_type'].values.astype(str)
    common_cts = sorted(set(ct_cell) & set(ct_nuc) & set(ct_nuc_imputed))

    print(f"\nEvaluating '{args.name}': {imputed.shape[0]} imputed, "
          f"{X_cell.shape[0]} cells, {X_nuc.shape[0]} nuclei, "
          f"{len(common_cts)} common CTs")

    results = {'method': args.name}

    # Load existing results if present (for --skip merge)
    existing_path = os.path.join(out_dir, f"eval_{args.name}.json")
    if os.path.exists(existing_path):
        with open(existing_path) as f:
            existing = json.load(f)
        results.update(existing)
        print(f"  Loaded existing results from {existing_path}")

    # ===== STEP 1: Gene correlation + R² + shift (fast) =====
    print(f"\n[Step 1] Gene correlation, R², shift... ({time.time()-t0:.0f}s)")
    cell_means = compute_celltype_means(X_cell, ct_cell, common_cts)
    nuc_means = compute_celltype_means(X_nuc, ct_nuc, common_cts)
    imp_means = compute_celltype_means(imputed, ct_nuc_imputed, common_cts)

    gc_per_gene, gc_summary = gene_correlation(imp_means, cell_means, common_cts)
    results['gene_corr'] = gc_summary
    _, gc_nuc = gene_correlation(nuc_means, cell_means, common_cts)
    results['gene_corr_raw_nuc'] = gc_nuc

    _, r2_summary = r_squared(imp_means, cell_means, common_cts)
    results['r_squared'] = r2_summary

    _, shift_summary = shift_scores(imp_means, cell_means, nuc_means, common_cts)
    results['shift'] = shift_summary

    from analysis.evaluate_baseline import gene_mse
    results['gene_mse'] = gene_mse(imp_means, cell_means, common_cts)
    del results['gene_mse']['per_gene']

    save_results(results, out_dir, args.name)

    # ===== STEP 2: Classification (moderate) =====
    if 'clf' not in skip:
        print(f"\n[Step 2] Classification... ({time.time()-t0:.0f}s)")
        MAX_PER_CT = 2000
        def _subsample(X, cts):
            np.random.seed(42)
            idx = []
            for ct in sorted(set(cts)):
                ct_idx = np.where(cts == ct)[0]
                if len(ct_idx) > MAX_PER_CT:
                    ct_idx = np.random.choice(ct_idx, MAX_PER_CT, replace=False)
                idx.extend(ct_idx)
            return X[sorted(idx)], cts[sorted(idx)]

        X_cell_sub, ct_cell_sub = _subsample(X_cell, ct_cell)
        imp_sub, ct_imp_sub = _subsample(imputed, ct_nuc_imputed)
        nuc_sub, ct_nuc_sub = _subsample(X_nuc, ct_nuc)
        print(f"  Subsampled: {X_cell_sub.shape[0]} cells, {imp_sub.shape[0]} imputed, {nuc_sub.shape[0]} nuclei")

        clf_imp = classification(X_cell_sub, ct_cell_sub, imp_sub, ct_imp_sub, common_cts)
        clf_nuc = classification(X_cell_sub, ct_cell_sub, nuc_sub, ct_nuc_sub, common_cts)
        results['clf_imputed'] = {'accuracy': clf_imp['accuracy'], 'macro_f1': clf_imp['macro_f1']}
        results['clf_raw_nuc'] = {'accuracy': clf_nuc['accuracy'], 'macro_f1': clf_nuc['macro_f1']}
        save_results(results, out_dir, args.name)
    else:
        print("  [SKIP] Classification")

    # ===== STEP 3: DE preservation (slow — Wilcoxon on 10k genes) =====
    if 'de' not in skip:
        print(f"\n[Step 3] DE preservation... ({time.time()-t0:.0f}s)")
        if 'X_cell_sub' not in dir():
            MAX_PER_CT = 2000
            def _subsample(X, cts):
                np.random.seed(42)
                idx = []
                for ct in sorted(set(cts)):
                    ct_idx = np.where(cts == ct)[0]
                    if len(ct_idx) > MAX_PER_CT:
                        ct_idx = np.random.choice(ct_idx, MAX_PER_CT, replace=False)
                    idx.extend(ct_idx)
                return X[sorted(idx)], cts[sorted(idx)]
            X_cell_sub, ct_cell_sub = _subsample(X_cell, ct_cell)
            imp_sub, ct_imp_sub = _subsample(imputed, ct_nuc_imputed)
            nuc_sub, ct_nuc_sub = _subsample(X_nuc, ct_nuc)

        _, de_imp = de_preservation(X_cell_sub, ct_cell_sub, imp_sub, ct_imp_sub, common_cts)
        _, de_nuc = de_preservation(X_cell_sub, ct_cell_sub, nuc_sub, ct_nuc_sub, common_cts)
        results['de_imputed'] = de_imp
        results['de_raw_nuc'] = de_nuc
        save_results(results, out_dir, args.name)
    else:
        print("  [SKIP] DE preservation")

    # ===== STEP 4: Coexpression (moderate) =====
    if 'coexp' not in skip:
        print(f"\n[Step 4] Coexpression... ({time.time()-t0:.0f}s)")
        results['coexp_imputed'] = coexpression(X_cell, imputed)
        results['coexp_raw_nuc'] = coexpression(X_cell, X_nuc)
        save_results(results, out_dir, args.name)
    else:
        print("  [SKIP] Coexpression")

    # ===== STEP 5: Sample-level metrics =====
    if 'sample' not in skip:
        print(f"\n[Step 5] Sample-level metrics... ({time.time()-t0:.0f}s)")
        _, vr_summary = variance_ratio(imputed, ct_nuc_imputed, X_cell, ct_cell, common_cts)
        results['variance_ratio'] = vr_summary

        if imputed.shape[0] == X_nuc.shape[0]:
            _, sp_summary = within_ct_spearman(
                imputed, ct_nuc_imputed, X_nuc, ct_nuc, common_cts, n_genes_sample=1000
            )
            results['within_ct_spearman'] = sp_summary
        else:
            results['within_ct_spearman'] = {'overall_median': None, 'note': 'size_mismatch'}

        _, pc_summary = per_cell_correlation(imputed, ct_nuc_imputed, cell_means, common_cts)
        results['per_cell_corr'] = pc_summary
        save_results(results, out_dir, args.name)
    else:
        print("  [SKIP] Sample-level")

    # ===== STEP 6: Label-free metrics =====
    if 'labelfree' not in skip:
        print(f"\n[Step 6] Label-free metrics... ({time.time()-t0:.0f}s)")
        qq = per_gene_correlation_allcells(imputed, X_cell)
        results['qq_gene_corr'] = {'median': qq['median'], 'mean': qq['mean']}

        results['wasserstein'] = wasserstein_per_gene(imputed, X_cell)
        save_results(results, out_dir, args.name)
    else:
        print("  [SKIP] Label-free")

    # ===== STEP 7: Biological validation =====
    if 'bio' not in skip:
        print(f"\n[Step 7] Biological validation... ({time.time()-t0:.0f}s)")
        gene_names = (adata_nuc.var['gene_names'].values
                      if 'gene_names' in adata_nuc.var else adata_nuc.var_names)

        results['delta_gt'] = delta_vs_ground_truth(
            imputed, X_nuc, X_cell, ct_nuc, ct_cell, gene_names, common_cts
        )
        results['compartment'] = compartment_scores(imputed, X_nuc, gene_names)
        results['ct_specific'] = ct_specific_delta(
            imputed, X_nuc, ct_nuc, gene_names, common_cts
        )
        save_results(results, out_dir, args.name)
    else:
        print("  [SKIP] Biological")

    # ===== Done =====
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  {args.name} — COMPLETE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}")

    # Print summary table
    print(f"\n  {'Metric':<35s} {'Value':>8s}")
    print(f"  {'—'*50}")
    print(f"  {'Gene correlation (median)':<35s} {results['gene_corr']['median']:>8.3f}")
    print(f"  {'  (raw nuc baseline)':<35s} {results['gene_corr_raw_nuc']['median']:>8.3f}")
    print(f"  {'R-squared (median)':<35s} {results['r_squared']['median']:>8.3f}")
    print(f"  {'Shift score (mean)':<35s} {results['shift']['mean_shift']:>8.4f}")
    if 'clf_imputed' in results:
        print(f"  {'Classification acc':<35s} {results['clf_imputed']['accuracy']:>8.3f}")
        print(f"  {'  (raw nuc baseline)':<35s} {results['clf_raw_nuc']['accuracy']:>8.3f}")
    if 'de_imputed' in results:
        print(f"  {'DE overlap (top 100)':<35s} {results['de_imputed']['mean_overlap']:>8.3f}")
    if 'coexp_imputed' in results:
        print(f"  {'Coexpression (Spearman)':<35s} {results['coexp_imputed']['spearman_r']:>8.3f}")
    if 'variance_ratio' in results:
        print(f"  {'Variance ratio (median)':<35s} {results['variance_ratio']['median_ratio']:>8.3f}")
    if 'delta_gt' in results:
        print(f"  {'Delta corr (overall)':<35s} {results['delta_gt']['overall_corr']:>8.3f}")
    if 'compartment' in results:
        ribo = results['compartment'].get('ribosomal', {})
        print(f"  {'Ribosomal correct sign':<35s} {ribo.get('frac_correct_sign', 0):>7.0%}")


if __name__ == "__main__":
    main()
