#!/usr/bin/env python3
"""
SynGO sensitivity analysis: mean-shift and distribution metrics
across cell-type exclusion settings.

Reads existing tables:
  syngo_gene_celltype_deltas.csv     (mean-shift)
  syngo_distribution_gene_celltype.csv  (Wasserstein)
  syngo_celltype_summary.csv         (cell/nucleus counts, neuronal status)

Outputs:
  syngo_sensitivity_summary.csv
  syngo_sensitivity_exclude_l5et.csv
  syngo_distribution_sensitivity_summary.csv
  syngo_sensitivity_neuronal_vs_nonneuronal.png/pdf
  syngo_distribution_sensitivity_panel.png/pdf
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUTDIR = Path(__file__).parent
SEED = 42
N_BOOT = 1000
CI_LO, CI_HI = 2.5, 97.5

NEURONAL = ['L2/3 IT', 'L5 ET', 'L5 IT', 'L5/6 NP', 'L6 CT', 'L6 IT',
            'L6b', 'Lamp5', 'Pvalb', 'Sncg', 'Sst', 'Vip']
NON_NEURONAL = ['Astro', 'Endo', 'Macrophage', 'OPC', 'Oligo']

LOW_SAMPLE_THRESHOLD = 20  # n_validation < 20 in either modality

# --- settings ---
SETTINGS = {
    'all': {'exclude_cts': []},
    'exclude_l5et': {'exclude_cts': ['L5 ET']},
    'exclude_low_sample': {'exclude_cts': []},  # filled dynamically
    'exclude_l5et_and_low_sample': {'exclude_cts': ['L5 ET']},  # + low sample
}


def load_data():
    deltas = pd.read_csv(OUTDIR / 'syngo_gene_celltype_deltas.csv')
    distrib = pd.read_csv(OUTDIR / 'syngo_distribution_gene_celltype.csv')
    ct_summary = pd.read_csv(OUTDIR / 'syngo_celltype_summary.csv')
    return deltas, distrib, ct_summary


def identify_low_sample_cts(ct_summary):
    low = ct_summary[
        (ct_summary['n_cell'] < LOW_SAMPLE_THRESHOLD) |
        (ct_summary['n_nucleus'] < LOW_SAMPLE_THRESHOLD)
    ]['cell_type'].tolist()
    return low


def compute_meanshift_metrics(df):
    """Compute mean-shift quality metrics from gene x cell-type deltas."""
    emp = df['empirical_delta'].values
    pred = df['predicted_delta'].values
    n = len(df)
    n_genes = df['gene'].nunique()
    n_cts = df['cell_type'].nunique()

    pearson_r = np.corrcoef(emp, pred)[0, 1]
    spearman_r = stats.spearmanr(emp, pred).statistic
    ss_res = np.sum((pred - emp) ** 2)
    ss_tot = np.sum((emp - np.mean(emp)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    mae = np.mean(np.abs(pred - emp))
    rmse = np.sqrt(np.mean((pred - emp) ** 2))
    sign_agree = np.mean(np.sign(emp) == np.sign(pred))
    slope, intercept = np.polyfit(emp, pred, 1)

    return {
        'n_genes': n_genes,
        'n_cell_types': n_cts,
        'n_gene_celltype_pairs': n,
        'pearson': pearson_r,
        'spearman': spearman_r,
        'R2': r2,
        'MAE': mae,
        'RMSE': rmse,
        'sign_agreement': sign_agree,
        'slope': slope,
        'intercept': intercept,
    }


def compute_neuronal_nonneuronal(df, rng):
    """Compute neuronal vs non-neuronal mean shifts with bootstrap CIs."""
    neuro = df[df['neuronal_status'] == 'neuronal']
    non_neuro = df[df['neuronal_status'] == 'non-neuronal']

    neuro_cts = neuro['cell_type'].unique()
    non_neuro_cts = non_neuro['cell_type'].unique()

    # Per-CT mean empirical/predicted delta, then average across CTs (equal weight)
    ct_means = df.groupby('cell_type')[['empirical_delta', 'predicted_delta']].mean()
    ct_means['neuronal_status'] = ct_means.index.map(
        lambda x: 'neuronal' if x in NEURONAL else 'non-neuronal')

    neuro_ct = ct_means[ct_means['neuronal_status'] == 'neuronal']
    non_neuro_ct = ct_means[ct_means['neuronal_status'] == 'non-neuronal']

    emp_neuronal = neuro_ct['empirical_delta'].mean()
    emp_non_neuronal = non_neuro_ct['empirical_delta'].mean()
    emp_diff = emp_neuronal - emp_non_neuronal
    pred_neuronal = neuro_ct['predicted_delta'].mean()
    pred_non_neuronal = non_neuro_ct['predicted_delta'].mean()
    pred_diff = pred_neuronal - pred_non_neuronal

    # Bootstrap: resample cell types within groups
    boot_emp_diff = []
    boot_pred_diff = []
    for _ in range(N_BOOT):
        n_idx = rng.choice(len(neuro_ct), size=len(neuro_ct), replace=True)
        nn_idx = rng.choice(len(non_neuro_ct), size=len(non_neuro_ct), replace=True)
        n_emp = neuro_ct['empirical_delta'].values[n_idx].mean()
        nn_emp = non_neuro_ct['empirical_delta'].values[nn_idx].mean()
        n_pred = neuro_ct['predicted_delta'].values[n_idx].mean()
        nn_pred = non_neuro_ct['predicted_delta'].values[nn_idx].mean()
        boot_emp_diff.append(n_emp - nn_emp)
        boot_pred_diff.append(n_pred - nn_pred)

    boot_emp_diff = np.array(boot_emp_diff)
    boot_pred_diff = np.array(boot_pred_diff)

    return {
        'emp_neuronal': emp_neuronal,
        'emp_non_neuronal': emp_non_neuronal,
        'emp_diff': emp_diff,
        'pred_neuronal': pred_neuronal,
        'pred_non_neuronal': pred_non_neuronal,
        'pred_diff': pred_diff,
        'emp_diff_ci_lo': np.percentile(boot_emp_diff, CI_LO),
        'emp_diff_ci_hi': np.percentile(boot_emp_diff, CI_HI),
        'pred_diff_ci_lo': np.percentile(boot_pred_diff, CI_LO),
        'pred_diff_ci_hi': np.percentile(boot_pred_diff, CI_HI),
        'n_neuronal_cts': len(neuro_cts),
        'n_non_neuronal_cts': len(non_neuro_cts),
    }


def compute_distribution_metrics(distrib_df):
    """Compute Wasserstein distribution-level summary metrics."""
    n = len(distrib_df)
    n_genes = distrib_df['gene'].nunique()
    n_cts = distrib_df['cell_type'].nunique()

    W_raw = distrib_df['W_raw'].values
    W_imputed = distrib_df['W_imputed'].values
    abs_imp = distrib_df['absolute_improvement'].values

    neuro = distrib_df[distrib_df['neuronal_status'] == 'neuronal']
    non_neuro = distrib_df[distrib_df['neuronal_status'] == 'non-neuronal']

    return {
        'n_genes': n_genes,
        'n_cell_types': n_cts,
        'n_gene_celltype_pairs': n,
        'median_W_raw': np.median(W_raw),
        'median_W_imputed': np.median(W_imputed),
        'median_absolute_improvement': np.median(abs_imp),
        'mean_absolute_improvement': np.mean(abs_imp),
        'proportion_improved_W': np.mean(abs_imp > 0),
        'neuronal_median_abs_imp': np.median(neuro['absolute_improvement'].values),
        'neuronal_proportion_improved': np.mean(neuro['absolute_improvement'].values > 0),
        'non_neuronal_median_abs_imp': np.median(non_neuro['absolute_improvement'].values),
        'non_neuronal_proportion_improved': np.mean(non_neuro['absolute_improvement'].values > 0),
    }


def run_sensitivity():
    deltas, distrib, ct_summary = load_data()
    rng = np.random.default_rng(SEED)

    low_sample_cts = identify_low_sample_cts(ct_summary)
    print(f"Low-sample CTs (n < {LOW_SAMPLE_THRESHOLD}): {low_sample_cts}")

    # Verify low-sample CTs
    for ct in low_sample_cts:
        row = ct_summary[ct_summary['cell_type'] == ct].iloc[0]
        print(f"  {ct}: {int(row['n_cell'])} cells, {int(row['n_nucleus'])} nuclei")

    # Update dynamic settings
    SETTINGS['exclude_low_sample']['exclude_cts'] = low_sample_cts
    SETTINGS['exclude_l5et_and_low_sample']['exclude_cts'] = list(
        set(['L5 ET'] + low_sample_cts))

    # Mean-shift sensitivity
    meanshift_rows = []
    neuronal_rows = []

    for setting_name, config in SETTINGS.items():
        exclude = config['exclude_cts']
        df = deltas[~deltas['cell_type'].isin(exclude)]
        dist_df = distrib[~distrib['cell_type'].isin(exclude)]

        metrics = compute_meanshift_metrics(df)
        metrics['setting'] = setting_name
        metrics['excluded_cts'] = '; '.join(exclude) if exclude else 'none'

        neuro_metrics = compute_neuronal_nonneuronal(df, rng)
        neuro_metrics['setting'] = setting_name
        neuro_metrics['excluded_cts'] = metrics['excluded_cts']
        neuro_metrics.update({k: v for k, v in metrics.items()
                              if k not in neuro_metrics})

        meanshift_rows.append(metrics)
        neuronal_rows.append(neuro_metrics)

    # Save mean-shift sensitivity
    ms_df = pd.DataFrame(meanshift_rows)
    col_order = ['setting', 'excluded_cts', 'n_genes', 'n_cell_types',
                 'n_gene_celltype_pairs', 'pearson', 'spearman', 'R2',
                 'MAE', 'RMSE', 'sign_agreement', 'slope', 'intercept']
    ms_df = ms_df[col_order]
    ms_df.to_csv(OUTDIR / 'syngo_sensitivity_summary.csv', index=False)
    print("\nMean-shift sensitivity:")
    print(ms_df.to_string(index=False))

    # Save neuronal vs non-neuronal
    neuro_df = pd.DataFrame(neuronal_rows)
    neuro_col_order = ['setting', 'excluded_cts', 'n_neuronal_cts',
                       'n_non_neuronal_cts',
                       'emp_neuronal', 'emp_non_neuronal', 'emp_diff',
                       'emp_diff_ci_lo', 'emp_diff_ci_hi',
                       'pred_neuronal', 'pred_non_neuronal', 'pred_diff',
                       'pred_diff_ci_lo', 'pred_diff_ci_hi']
    neuro_df = neuro_df[neuro_col_order]
    neuro_df.to_csv(OUTDIR / 'syngo_sensitivity_exclude_l5et.csv', index=False)
    print("\nNeuronal vs non-neuronal:")
    print(neuro_df.to_string(index=False))

    # Distribution sensitivity
    dist_rows = []
    for setting_name, config in SETTINGS.items():
        exclude = config['exclude_cts']
        dist_df = distrib[~distrib['cell_type'].isin(exclude)]
        dmetrics = compute_distribution_metrics(dist_df)
        dmetrics['setting'] = setting_name
        dmetrics['excluded_cts'] = '; '.join(exclude) if exclude else 'none'
        dist_rows.append(dmetrics)

    dist_sens_df = pd.DataFrame(dist_rows)
    dist_col_order = ['setting', 'excluded_cts', 'n_genes', 'n_cell_types',
                      'n_gene_celltype_pairs',
                      'median_W_raw', 'median_W_imputed',
                      'median_absolute_improvement', 'mean_absolute_improvement',
                      'proportion_improved_W',
                      'neuronal_median_abs_imp', 'neuronal_proportion_improved',
                      'non_neuronal_median_abs_imp', 'non_neuronal_proportion_improved']
    dist_sens_df = dist_sens_df[dist_col_order]
    dist_sens_df.to_csv(OUTDIR / 'syngo_distribution_sensitivity_summary.csv',
                        index=False)
    print("\nDistribution sensitivity:")
    print(dist_sens_df.to_string(index=False))

    # --- Key interpretation ---
    print("\n" + "=" * 70)
    print("KEY QUESTION: Does neuronal > non-neuronal SynGO pattern survive")
    print("after removing L5 ET?")
    all_row = neuro_df[neuro_df['setting'] == 'all'].iloc[0]
    excl_row = neuro_df[neuro_df['setting'] == 'exclude_l5et'].iloc[0]
    print(f"  All CTs:        emp_diff = {all_row['emp_diff']:.4f}  "
          f"[{all_row['emp_diff_ci_lo']:.4f}, {all_row['emp_diff_ci_hi']:.4f}]")
    print(f"  Exclude L5 ET:  emp_diff = {excl_row['emp_diff']:.4f}  "
          f"[{excl_row['emp_diff_ci_lo']:.4f}, {excl_row['emp_diff_ci_hi']:.4f}]")
    print(f"  All CTs:        pred_diff = {all_row['pred_diff']:.4f}  "
          f"[{all_row['pred_diff_ci_lo']:.4f}, {all_row['pred_diff_ci_hi']:.4f}]")
    print(f"  Exclude L5 ET:  pred_diff = {excl_row['pred_diff']:.4f}  "
          f"[{excl_row['pred_diff_ci_lo']:.4f}, {excl_row['pred_diff_ci_hi']:.4f}]")
    survives = excl_row['emp_diff_ci_lo'] > 0
    print(f"  Survives (95% CI excludes 0): {survives}")
    print("=" * 70)

    # --- PLOTS ---
    plot_neuronal_sensitivity(neuro_df)
    plot_distribution_sensitivity(dist_sens_df, distrib)


def plot_neuronal_sensitivity(neuro_df):
    """Bar plot: neuronal vs non-neuronal mean shift across settings."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    settings = neuro_df['setting'].values
    labels = ['All CTs', 'Excl. L5 ET', 'Excl. low-n', 'Excl. both']
    x = np.arange(len(settings))
    width = 0.35

    for ax_idx, (prefix, title) in enumerate([
        ('emp', 'Empirical (ground truth)'),
        ('pred', 'Predicted (ImputeVAE)')
    ]):
        ax = axes[ax_idx]
        neuronal_vals = neuro_df[f'{prefix}_neuronal'].values
        nonneuronal_vals = neuro_df[f'{prefix}_non_neuronal'].values

        bars1 = ax.bar(x - width/2, neuronal_vals, width, label='Neuronal',
                       color='#2196F3', alpha=0.85)
        bars2 = ax.bar(x + width/2, nonneuronal_vals, width, label='Non-neuronal',
                       color='#FF9800', alpha=0.85)

        # Add diff annotations
        for i in range(len(settings)):
            diff = neuro_df[f'{prefix}_diff'].values[i]
            ci_lo = neuro_df[f'{prefix}_diff_ci_lo'].values[i]
            ci_hi = neuro_df[f'{prefix}_diff_ci_hi'].values[i]
            y_max = max(neuronal_vals[i], nonneuronal_vals[i])
            ax.annotate(f'diff={diff:.3f}\n[{ci_lo:.3f}, {ci_hi:.3f}]',
                        xy=(x[i], y_max + 0.02), ha='center', va='bottom',
                        fontsize=7.5, color='#333333')

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel('Mean SynGO delta (cell - nucleus)')
        ax.set_title(title)
        ax.legend(loc='upper right', fontsize=9)
        ax.set_ylim(0, max(neuronal_vals.max(), nonneuronal_vals.max()) + 0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('SynGO positive-control: neuronal vs non-neuronal mean shift\n'
                 'sensitivity to cell-type exclusion',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    for ext in ['png', 'pdf']:
        fig.savefig(OUTDIR / f'syngo_sensitivity_neuronal_vs_nonneuronal.{ext}',
                    dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved neuronal vs non-neuronal sensitivity plot.")


def plot_distribution_sensitivity(dist_sens_df, distrib_full):
    """Panel plot: Wasserstein distribution sensitivity."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    settings = dist_sens_df['setting'].values
    labels = ['All CTs', 'Excl. L5 ET', 'Excl. low-n', 'Excl. both']

    # Panel 1: Median absolute improvement
    ax = axes[0]
    vals = dist_sens_df['median_absolute_improvement'].values
    colors = ['#4CAF50' if v > 0 else '#F44336' for v in vals]
    ax.bar(range(len(vals)), vals, color=colors, alpha=0.85)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9, rotation=15)
    ax.set_ylabel('Median absolute improvement')
    ax.set_title('Wasserstein improvement\n(positive = imputation closer to cell)')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel 2: Proportion improved
    ax = axes[1]
    vals_all = dist_sens_df['proportion_improved_W'].values
    vals_n = dist_sens_df['neuronal_proportion_improved'].values
    vals_nn = dist_sens_df['non_neuronal_proportion_improved'].values
    x = np.arange(len(settings))
    w = 0.25
    ax.bar(x - w, vals_n, w, label='Neuronal', color='#2196F3', alpha=0.85)
    ax.bar(x, vals_all, w, label='All', color='#9E9E9E', alpha=0.85)
    ax.bar(x + w, vals_nn, w, label='Non-neuronal', color='#FF9800', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15)
    ax.set_ylabel('Proportion improved')
    ax.set_title('Fraction of gene-CT pairs improved')
    ax.axhline(0.5, color='gray', linewidth=0.5, linestyle='--')
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel 3: Neuronal vs non-neuronal median improvement
    ax = axes[2]
    ax.bar(x - w/2, dist_sens_df['neuronal_median_abs_imp'].values, w,
           label='Neuronal', color='#2196F3', alpha=0.85)
    ax.bar(x + w/2, dist_sens_df['non_neuronal_median_abs_imp'].values, w,
           label='Non-neuronal', color='#FF9800', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15)
    ax.set_ylabel('Median Wasserstein improvement')
    ax.set_title('Neuronal vs non-neuronal\nWasserstein improvement')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.suptitle('SynGO distribution-level sensitivity analysis',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    for ext in ['png', 'pdf']:
        fig.savefig(OUTDIR / f'syngo_distribution_sensitivity_panel.{ext}',
                    dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved distribution sensitivity panel.")


if __name__ == '__main__':
    run_sensitivity()
