"""
Cross-model comparison tables and report generation.
"""

import json
import os

import numpy as np
import pandas as pd


def format_summary_table(results_by_model):
    """Create a comparison table across models.

    Parameters
    ----------
    results_by_model : dict, model_name -> results dict (from runner)

    Returns
    -------
    pd.DataFrame with one row per model
    """
    rows = []
    for model_name, results in results_by_model.items():
        row = {'model': model_name}

        # CT-level
        ct = results.get('ct_level', {})
        row['gene_corr'] = ct.get('gene_corr', {}).get('median', '')
        row['gene_corr_raw_nuc'] = ct.get('gene_corr_raw_nuc', {}).get('median', '')
        row['r2'] = ct.get('r_squared', {}).get('median', '')
        row['shift'] = ct.get('shift', {}).get('mean_shift', '')
        row['clf_acc'] = ct.get('clf_imputed', {}).get('accuracy', '')
        row['clf_raw_nuc_acc'] = ct.get('clf_raw_nuc', {}).get('accuracy', '')
        row['clf_improvement'] = ct.get('clf_improvement', '')
        row['de_overlap'] = ct.get('de_imputed', {}).get('mean_overlap', '')
        row['de_raw_nuc'] = ct.get('de_raw_nuc', {}).get('mean_overlap', '')
        row['coexp'] = ct.get('coexp_imputed', {}).get('spearman_r', '')

        # Sample-level
        sl = results.get('sample_level', {})
        row['var_ratio'] = sl.get('variance_ratio', {}).get('median_ratio', '')
        row['within_ct_spearman'] = sl.get('within_ct_spearman', {}).get('overall_median', '')
        row['per_cell_corr'] = sl.get('per_cell_corr', {}).get('overall_median', '')

        # Latent
        lt = results.get('latent', {})
        row['z_abl_corr'] = lt.get('z_ablation', {}).get('mean_corr', '')
        row['z_between_frac'] = lt.get('z_variance_decomp', {}).get('mean_between_frac', '')

        rows.append(row)

    return pd.DataFrame(rows)


def save_results(results, model_name, output_dir):
    """Save evaluation results for a single model.

    Parameters
    ----------
    results : dict (from runner)
    model_name : str
    output_dir : str
    """
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Save each category
    for category in ['ct_level', 'sample_level', 'latent']:
        if category not in results:
            continue
        cat_data = results[category]
        cat_dir = os.path.join(model_dir, category)
        os.makedirs(cat_dir, exist_ok=True)

        # Separate DataFrames/arrays from scalars
        scalars = {}
        for key, val in cat_data.items():
            if key.startswith('_'):
                continue  # skip internal data
            if isinstance(val, pd.DataFrame):
                val.to_csv(os.path.join(cat_dir, f'{key}.csv'), index=False)
            elif isinstance(val, np.ndarray):
                np.save(os.path.join(cat_dir, f'{key}.npy'), val)
            elif isinstance(val, dict):
                # Check for nested non-serializable
                clean = {}
                for k, v in val.items():
                    if isinstance(v, (int, float, str, bool, type(None))):
                        clean[k] = v
                    elif isinstance(v, np.floating):
                        clean[k] = float(v)
                    elif isinstance(v, np.integer):
                        clean[k] = int(v)
                scalars[key] = clean
            else:
                scalars[key] = val

        if scalars:
            with open(os.path.join(cat_dir, 'summary.json'), 'w') as f:
                json.dump(scalars, f, indent=2)

    # Gene stratification (separate category)
    if 'gene_stratified' in results:
        gs = results['gene_stratified']
        gs_dir = os.path.join(model_dir, 'gene_stratified')
        os.makedirs(gs_dir, exist_ok=True)

        for key in ['by_expr_quintile', 'by_diff_quintile', 'by_bio_category', 'per_gene']:
            if key in gs and isinstance(gs[key], pd.DataFrame):
                gs[key].to_csv(os.path.join(gs_dir, f'{key}.csv'), index=False)

        if 'strata_df' in gs and isinstance(gs['strata_df'], pd.DataFrame):
            gs['strata_df'].to_csv(os.path.join(gs_dir, 'strata_assignments.csv'), index=False)


def print_summary(results, model_name):
    """Print formatted summary for a single model."""
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")

    ct = results.get('ct_level', {})
    if ct:
        gc = ct.get('gene_corr', {})
        print(f"  CT-level gene correlation:  {gc.get('median', 'N/A'):.4f}" if isinstance(gc.get('median'), float) else "")
        r2 = ct.get('r_squared', {})
        print(f"  CT-level R^2:               {r2.get('median', 'N/A'):.4f}" if isinstance(r2.get('median'), float) else "")
        sh = ct.get('shift', {})
        print(f"  Mean shift score:           {sh.get('mean_shift', 'N/A'):.4f}" if isinstance(sh.get('mean_shift'), float) else "")

        clf = ct.get('clf_imputed', {})
        clf_nuc = ct.get('clf_raw_nuc', {})
        if clf.get('accuracy') is not None:
            print(f"  Classification (imputed):   {clf['accuracy']:.4f}")
            print(f"  Classification (raw nuc):   {clf_nuc.get('accuracy', 0):.4f}")
            imp = ct.get('clf_improvement', 0)
            print(f"  Classification improvement: {imp*100:+.1f}%")

        de = ct.get('de_imputed', {})
        de_nuc = ct.get('de_raw_nuc', {})
        if de.get('mean_overlap') is not None:
            print(f"  DE overlap (imputed):       {de['mean_overlap']:.3f}")
            print(f"  DE overlap (raw nuc):       {de_nuc.get('mean_overlap', 0):.3f}")

        coexp = ct.get('coexp_imputed', {})
        if coexp.get('spearman_r') is not None:
            print(f"  Coexpression (imputed):     {coexp['spearman_r']:.4f}")

    sl = results.get('sample_level', {})
    if sl:
        vr = sl.get('variance_ratio', {})
        print(f"\n  Variance ratio (median):    {vr.get('median_ratio', 'N/A')}")
        sp = sl.get('within_ct_spearman', {})
        print(f"  Within-CT Spearman:         {sp.get('overall_median', 'N/A')}")
        pc = sl.get('per_cell_corr', {})
        print(f"  Per-cell corr to CT mean:   {pc.get('overall_median', 'N/A')}")

    lt = results.get('latent', {})
    if lt:
        za = lt.get('z_ablation', {})
        if za.get('mean_corr') is not None:
            print(f"\n  z ablation corr:            {za['mean_corr']:.4f}")
        zv = lt.get('z_variance_decomp', {})
        if zv.get('mean_between_frac') is not None:
            print(f"  z between-CT variance frac: {zv['mean_between_frac']:.4f}")
