#!/usr/bin/env python
"""
Generate Aim 1 biological-validation figure set for qualifying exam presentation.

Usage:
    python -u validation/aim1_biological_validation_figures/scripts/make_aim1_figures.py

Requires conda env: impute2
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats

# ============================================================
# Paths
# ============================================================
PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

SYNGO_DIR = os.path.join(PROJECT, "validation/syngo/syngo_brain_validation")
TF_DIR = os.path.join(PROJECT, "validation/tf/tf_brain_validation")
TF_DIAG_DIR = os.path.join(PROJECT, "validation/tf/jaspar_diagnostic")
TF_COMP_DIR = os.path.join(PROJECT, "validation/tf/syngo_vs_tf")
OUT_DIR = os.path.join(PROJECT, "validation/aim1_biological_validation_figures")

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Style constants
# ============================================================
COLOR_EMPIRICAL = '#3274A1'   # blue
COLOR_PREDICTED = '#E1812C'   # orange
COLOR_NEURONAL = '#8B2252'    # dark red/purple
COLOR_NON_NEURONAL = '#5F9EA0' # teal
COLOR_CAUTION = '#C55A5A'     # muted red
COLOR_LIGHT_GRAY = '#CCCCCC'

FONTSIZE_TITLE = 14
FONTSIZE_LABEL = 12
FONTSIZE_TICK = 10
FONTSIZE_ANNOT = 9

plt.rcParams.update({
    'font.size': FONTSIZE_TICK,
    'axes.titlesize': FONTSIZE_TITLE,
    'axes.labelsize': FONTSIZE_LABEL,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'svg.fonttype': 'none',  # editable text in SVG
})


def _save(fig, base_path):
    """Save figure in png, pdf, svg."""
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(f"{base_path}.{ext}", bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {os.path.basename(base_path)}.{{png,pdf,svg}}")


# ============================================================
# Figure A1.1: Validation schematic
# ============================================================
def make_figure_A1_1():
    print("Figure A1.1: Validation schematic")
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_aspect('equal')

    # --- Data flow boxes ---
    box_kw = dict(boxstyle='round,pad=0.4', linewidth=1.5)

    # Held-out nuclei
    ax.text(2, 6.8, 'Held-out validation\nnuclei (snRNA-seq)',
            ha='center', va='center', fontsize=11, fontweight='bold',
            bbox=dict(facecolor='#D6EAF8', edgecolor='#2E86C1', **box_kw))

    # ImputeVAE
    ax.text(6, 6.8, 'Frozen\nImputeVAE\n(cVAE)',
            ha='center', va='center', fontsize=11, fontweight='bold',
            bbox=dict(facecolor='#FCF3CF', edgecolor='#D4AC0D', **box_kw))

    # Imputed cell-like
    ax.text(10, 6.8, 'Imputed\nwhole-cell-like\nprofiles',
            ha='center', va='center', fontsize=11, fontweight='bold',
            bbox=dict(facecolor='#FADBD8', edgecolor='#E74C3C', **box_kw))

    # Held-out cells
    ax.text(10, 5.0, 'Held-out validation\nwhole cells (scRNA-seq)',
            ha='center', va='center', fontsize=11, fontweight='bold',
            bbox=dict(facecolor='#D5F5E3', edgecolor='#27AE60', **box_kw))

    # Arrows
    ax.annotate('', xy=(4.2, 6.8), xytext=(3.5, 6.8),
                arrowprops=dict(arrowstyle='->', lw=2, color='#555'))
    ax.annotate('', xy=(8.2, 6.8), xytext=(7.6, 6.8),
                arrowprops=dict(arrowstyle='->', lw=2, color='#555'))

    # Delta definitions
    ax.text(7, 5.5, 'empirical_delta = mean(raw cell) \u2013 mean(raw nucleus)',
            ha='center', fontsize=10, fontstyle='italic', color='#2E86C1')
    ax.text(7, 5.0, 'predicted_delta = mean(imputed cell) \u2013 mean(raw nucleus)',
            ha='center', fontsize=10, fontstyle='italic', color='#E74C3C')
    ax.text(7, 4.5, 'Population-level within matched cell types (unpaired)',
            ha='center', fontsize=9, color='#666')

    # --- Validation tiers ---
    tier_x = 1.5
    tier_w = 11
    tier_colors = ['#EBF5FB', '#FEF9E7', '#FDEDEC']
    tier_labels = [
        ('Tier 1: Same-dataset biological gene sets',
         'SynGO synaptic genes  |  JASPAR TF contrast set  |  Ribosomal/translation genes'),
        ('Tier 2: RNA-localization-aware checks',
         'Nuclear-retained lncRNAs  |  lncATLAS subcellular localization'),
        ('Tier 3: Orthogonal external references',
         'SINC/NanoSINC  |  TRAP/RiboTag  |  Patch-seq  |  Bulk/pseudobulk fractionation'),
    ]

    for i, (title, detail) in enumerate(tier_labels):
        y = 3.4 - i * 1.1
        rect = mpatches.FancyBboxPatch((tier_x, y - 0.35), tier_w, 0.85,
                                       boxstyle='round,pad=0.1',
                                       facecolor=tier_colors[i],
                                       edgecolor='#AAA', linewidth=1)
        ax.add_patch(rect)
        ax.text(tier_x + 0.3, y + 0.2, title, fontsize=10.5, fontweight='bold',
                va='center')
        ax.text(tier_x + 0.3, y - 0.1, detail, fontsize=9, va='center', color='#555')

    # Caution text
    ax.text(7, 0.4,
            'Caution: Gene-set annotations are not interchangeable ground truth.\n'
            'SynGO annotates protein function/location, not RNA localization.',
            ha='center', va='center', fontsize=9, fontstyle='italic',
            color=COLOR_CAUTION,
            bbox=dict(facecolor='#FFF5F5', edgecolor=COLOR_CAUTION,
                      boxstyle='round,pad=0.3', linewidth=1, alpha=0.9))

    _save(fig, os.path.join(OUT_DIR, "aim1_figure_A1_1_validation_schematic"))


# ============================================================
# Figure A1.2: SynGO delta scatter
# ============================================================
def make_figure_A1_2():
    print("Figure A1.2: SynGO delta scatter")
    delta_df = pd.read_csv(os.path.join(SYNGO_DIR, "syngo_gene_celltype_deltas.csv"))
    qual_df = pd.read_csv(os.path.join(SYNGO_DIR, "syngo_overall_quality.csv"))

    q = qual_df.iloc[0]

    fig, ax = plt.subplots(figsize=(7, 6.5))

    # Color by neuronal status
    neuronal = delta_df['neuronal_status'] == 'neuronal'
    ax.scatter(delta_df.loc[~neuronal, 'empirical_delta'],
               delta_df.loc[~neuronal, 'predicted_delta'],
               s=4, alpha=0.25, color=COLOR_NON_NEURONAL, label='Non-neuronal', rasterized=True)
    ax.scatter(delta_df.loc[neuronal, 'empirical_delta'],
               delta_df.loc[neuronal, 'predicted_delta'],
               s=4, alpha=0.25, color=COLOR_NEURONAL, label='Neuronal', rasterized=True)

    # Reference lines
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, '--', color='#999', lw=1, zorder=0, label='y = x')

    # Regression line
    slope, intercept = q['slope'], q['intercept']
    x_fit = np.array(lims)
    ax.plot(x_fit, slope * x_fit + intercept, '-', color=COLOR_PREDICTED, lw=1.5,
            alpha=0.8, label=f'Fit (slope={slope:.3f})')

    ax.set_xlabel('Empirical delta (raw cell \u2013 raw nucleus)', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('Predicted delta (imputed cell \u2013 raw nucleus)', fontsize=FONTSIZE_LABEL)
    ax.set_title('SynGO Mean-Shift Recovery', fontsize=FONTSIZE_TITLE, fontweight='bold')

    # Annotation box
    ann_text = (
        f"n genes = {int(q['n_genes'])}\n"
        f"n gene\u00d7CT pairs = {int(q['n_observations'])}\n"
        f"Pearson r = {q['pearson_r']:.4f}\n"
        f"Spearman \u03c1 = {q['spearman_r']:.4f}\n"
        f"R\u00b2 = {q['R2']:.4f}\n"
        f"Sign agreement = {q['sign_agreement']:.4f}\n"
        f"Slope = {q['slope']:.4f}"
    )
    ax.text(0.03, 0.97, ann_text, transform=ax.transAxes, fontsize=FONTSIZE_ANNOT,
            va='top', ha='left', family='monospace',
            bbox=dict(facecolor='white', edgecolor='#ccc', alpha=0.9, boxstyle='round,pad=0.4'))

    ax.legend(loc='lower right', fontsize=9, markerscale=3)
    ax.set_aspect('equal', adjustable='datalim')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    _save(fig, os.path.join(OUT_DIR, "aim1_figure_A1_2_syngo_delta_scatter"))

    # Save plotted data
    delta_df[['gene', 'cell_type', 'neuronal_status', 'empirical_delta', 'predicted_delta']].to_csv(
        os.path.join(OUT_DIR, "aim1_figure_A1_2_syngo_delta_scatter_data.csv"), index=False)
    print(f"  Data: aim1_figure_A1_2_syngo_delta_scatter_data.csv ({len(delta_df)} rows)")


# ============================================================
# Figure A1.3: Group shifts (neuronal/non-neuronal separate)
# ============================================================
def make_figure_A1_3():
    print("Figure A1.3: Group shifts (neuronal/non-neuronal)")
    syngo_nn = pd.read_csv(os.path.join(SYNGO_DIR, "syngo_neuronal_vs_nonneuronal.csv"))
    tf_nn = pd.read_csv(os.path.join(TF_DIR, "tf_neuronal_vs_nonneuronal.csv"))

    s = syngo_nn.iloc[0]
    t = tf_nn.iloc[0]

    # Data for grouped bar
    gene_sets = ['SynGO\n(n=1,275)', 'JASPAR TF\n(n=63)']
    categories = ['Emp.\nneuronal', 'Pred.\nneuronal', 'Emp.\nnon-neuronal', 'Pred.\nnon-neuronal']

    values = [
        [s['emp_neuronal'], s['pred_neuronal'], s['emp_non_neuronal'], s['pred_non_neuronal']],
        [t['emp_neuronal'], t['pred_neuronal'], t['emp_non_neuronal'], t['pred_non_neuronal']],
    ]

    # CIs from bootstrap
    syngo_sens = pd.read_csv(os.path.join(SYNGO_DIR, "syngo_sensitivity_exclude_l5et.csv"))
    tf_sens = pd.read_csv(os.path.join(TF_DIR, "tf_sensitivity_neuronal_vs_nonneuronal.csv"))
    s_all = syngo_sens[syngo_sens['setting'] == 'all'].iloc[0]
    t_all = tf_sens[tf_sens['setting'] == 'all'].iloc[0]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bar_colors = [COLOR_NEURONAL, COLOR_PREDICTED, COLOR_NON_NEURONAL, '#8FBC8F']
    bar_labels_legend = ['Empirical neuronal', 'Predicted neuronal',
                         'Empirical non-neuronal', 'Predicted non-neuronal']

    n_sets = len(gene_sets)
    n_cats = len(categories)
    bar_width = 0.17
    group_width = n_cats * bar_width + 0.08

    for i, gs in enumerate(gene_sets):
        base_x = i * (group_width + 0.5)
        for j in range(n_cats):
            x = base_x + j * (bar_width + 0.02)
            bar = ax.bar(x, values[i][j], width=bar_width, color=bar_colors[j],
                         edgecolor='white', linewidth=0.8)
            # Value label
            ax.text(x, values[i][j] + 0.02, f'{values[i][j]:.3f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

    # X ticks
    set_centers = []
    for i in range(n_sets):
        base_x = i * (group_width + 0.5)
        center = base_x + (n_cats - 1) * (bar_width + 0.02) / 2
        set_centers.append(center)

    ax.set_xticks(set_centers)
    ax.set_xticklabels(gene_sets, fontsize=FONTSIZE_LABEL)

    # Legend
    handles = [mpatches.Patch(facecolor=c, label=l) for c, l in zip(bar_colors, bar_labels_legend)]
    ax.legend(handles=handles, loc='upper right', fontsize=9, framealpha=0.9)

    ax.set_ylabel('Mean shift (log1p)', fontsize=FONTSIZE_LABEL)
    ax.set_title('Neuronal vs Non-Neuronal Shifts by Gene Set', fontsize=FONTSIZE_TITLE,
                 fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, max(max(v) for v in values) * 1.25)

    # Secondary annotation: diff values
    diff_text = (
        f"Neuronal \u2013 Non-neuronal diff:\n"
        f"  SynGO:  emp {s['emp_diff']:.4f}  pred {s['pred_diff']:.4f}\n"
        f"  TF:     emp {t['emp_diff']:.4f}  pred {t['pred_diff']:.4f}"
    )
    ax.text(0.02, 0.75, diff_text, transform=ax.transAxes, fontsize=FONTSIZE_ANNOT,
            va='top', family='monospace',
            bbox=dict(facecolor='#F9F9F9', edgecolor='#ccc', alpha=0.9,
                      boxstyle='round,pad=0.3'))

    _save(fig, os.path.join(OUT_DIR, "aim1_figure_A1_3_group_shifts_separate"))

    # Save data
    rows = []
    for i, gs_name in enumerate(['SynGO', 'JASPAR_TF']):
        for j, cat in enumerate(categories):
            rows.append({'gene_set': gs_name, 'category': cat.replace('\n', ' '),
                         'value': values[i][j]})
    pd.DataFrame(rows).to_csv(
        os.path.join(OUT_DIR, "aim1_figure_A1_3_group_shifts_separate_data.csv"), index=False)


# ============================================================
# Figure A1.4: Cell-type heterogeneity and failure modes
# ============================================================
def make_figure_A1_4():
    print("Figure A1.4: Cell-type failure modes")
    ct_df = pd.read_csv(os.path.join(SYNGO_DIR, "syngo_celltype_summary.csv"))

    # Sort: neuronal first (alphabetical), then non-neuronal
    ct_df['sort_key'] = ct_df['neuronal_status'].map({'neuronal': 0, 'non-neuronal': 1})
    ct_df = ct_df.sort_values(['sort_key', 'cell_type']).reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                    gridspec_kw={'width_ratios': [3, 1.2], 'wspace': 0.35})

    y_pos = np.arange(len(ct_df))
    h = 0.35

    # Main panel: empirical vs predicted mean shift
    bars_emp = ax1.barh(y_pos + h/2, ct_df['empirical_mean_shift'], height=h,
                        color=COLOR_EMPIRICAL, label='Empirical', alpha=0.85)
    bars_pred = ax1.barh(y_pos - h/2, ct_df['predicted_mean_shift'], height=h,
                         color=COLOR_PREDICTED, label='Predicted', alpha=0.85)

    # CIs
    for idx, row in ct_df.iterrows():
        i = ct_df.index.get_loc(idx) if idx in ct_df.index else idx
        if pd.notna(row.get('emp_ci_lo')) and pd.notna(row.get('emp_ci_hi')):
            ax1.errorbar(row['empirical_mean_shift'], y_pos[i] + h/2,
                         xerr=[[row['empirical_mean_shift'] - row['emp_ci_lo']],
                                [row['emp_ci_hi'] - row['empirical_mean_shift']]],
                         fmt='none', ecolor='#333', capsize=2, lw=1)
        if pd.notna(row.get('pred_ci_lo')) and pd.notna(row.get('pred_ci_hi')):
            ax1.errorbar(row['predicted_mean_shift'], y_pos[i] - h/2,
                         xerr=[[row['predicted_mean_shift'] - row['pred_ci_lo']],
                                [row['pred_ci_hi'] - row['predicted_mean_shift']]],
                         fmt='none', ecolor='#333', capsize=2, lw=1)

    # Labels with sample counts and highlights
    labels = []
    for _, row in ct_df.iterrows():
        label = row['cell_type']
        n_info = f"  (c={row['n_cell']}, n={row['n_nucleus']})"
        flag = ''
        if row['cell_type'] == 'L5 ET':
            flag = ' *'
        elif row.get('low_sample', False):
            flag = ' \u2020'
        labels.append(label + n_info + flag)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.set_xlabel('Mean shift (log1p)', fontsize=FONTSIZE_LABEL)
    ax1.set_title('SynGO Mean Shift by Cell Type', fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax1.legend(fontsize=10, loc='lower right')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.invert_yaxis()

    # Neuronal/non-neuronal shading
    for i, row in ct_df.iterrows():
        idx_pos = ct_df.index.get_loc(i) if i in ct_df.index else i
        if row['neuronal_status'] == 'neuronal':
            ax1.axhspan(idx_pos - 0.5, idx_pos + 0.5, color=COLOR_NEURONAL, alpha=0.04, zorder=0)

    # Error panel
    errors = ct_df['prediction_error'].values
    colors_err = [COLOR_CAUTION if abs(e) > 0.15 else '#888' for e in errors]
    ax2.barh(y_pos, errors, height=0.6, color=colors_err, alpha=0.7)
    ax2.axvline(0, color='#999', lw=0.8, ls='--')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    ax2.set_xlabel('Prediction error\n(predicted \u2013 empirical)', fontsize=10)
    ax2.set_title('Error', fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.invert_yaxis()

    # Annotations
    fig.text(0.5, 0.01,
             '* L5 ET: magnitude-calibration outlier (overpredicts shift)    '
             '\u2020 Low-sample caution (<20 validation nuclei or cells)',
             ha='center', fontsize=9, fontstyle='italic', color='#666')

    _save(fig, os.path.join(OUT_DIR, "aim1_figure_A1_4_celltype_failure_modes"))

    ct_df.to_csv(os.path.join(OUT_DIR, "aim1_figure_A1_4_celltype_failure_modes_data.csv"),
                 index=False)


# ============================================================
# Figure A1.5: L5 ET sensitivity
# ============================================================
def make_figure_A1_5():
    print("Figure A1.5: L5 ET sensitivity")
    syngo_sens = pd.read_csv(os.path.join(SYNGO_DIR, "syngo_sensitivity_exclude_l5et.csv"))
    tf_sens = pd.read_csv(os.path.join(TF_DIR, "tf_sensitivity_neuronal_vs_nonneuronal.csv"))

    # Extract all / exclude_l5et rows
    s_all = syngo_sens[syngo_sens['setting'] == 'all'].iloc[0]
    s_no = syngo_sens[syngo_sens['setting'] == 'exclude_l5et'].iloc[0]
    t_all = tf_sens[tf_sens['setting'] == 'all'].iloc[0]
    t_no = tf_sens[tf_sens['setting'] == 'exclude_l5et'].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)

    for ax_idx, (gs_name, s_a, s_n, n_genes) in enumerate([
        ('SynGO (n=1,275)', s_all, s_no, 1275),
        ('JASPAR TF (n=63)', t_all, t_no, 63),
    ]):
        ax = axes[ax_idx]
        categories = ['Emp.\nneuronal', 'Pred.\nneuronal',
                       'Emp.\nnon-neur.', 'Pred.\nnon-neur.']
        vals_all = [s_a['emp_neuronal'], s_a['pred_neuronal'],
                    s_a['emp_non_neuronal'], s_a['pred_non_neuronal']]
        vals_no = [s_n['emp_neuronal'], s_n['pred_neuronal'],
                   s_n['emp_non_neuronal'], s_n['pred_non_neuronal']]

        x = np.arange(len(categories))
        w = 0.35
        bars1 = ax.bar(x - w/2, vals_all, w, color='#4472C4', alpha=0.8, label='All CTs')
        bars2 = ax.bar(x + w/2, vals_no, w, color='#ED7D31', alpha=0.8, label='Excl. L5 ET')

        # Value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.3f}',
                        ha='center', va='bottom', fontsize=7.5, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_title(gs_name, fontsize=FONTSIZE_TITLE, fontweight='bold')
        ax.legend(fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Diff annotation
        diff_text = (
            f"Diff (neuro\u2013non-neuro):\n"
            f"  All:      emp {s_a['emp_diff']:.4f}  pred {s_a['pred_diff']:.4f}\n"
            f"  No L5ET: emp {s_n['emp_diff']:.4f}  pred {s_n['pred_diff']:.4f}"
        )
        ax.text(0.02, 0.98, diff_text, transform=ax.transAxes, fontsize=8,
                va='top', family='monospace',
                bbox=dict(facecolor='white', edgecolor='#ccc', alpha=0.9,
                          boxstyle='round,pad=0.3'))

    axes[0].set_ylabel('Mean shift (log1p)', fontsize=FONTSIZE_LABEL)

    fig.text(0.5, 0.01,
             'L5 ET contributes to overprediction, but does not create '
             'the main SynGO validation signal.',
             ha='center', fontsize=10, fontstyle='italic', color='#555')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, os.path.join(OUT_DIR, "aim1_figure_A1_5_l5et_sensitivity"))

    # Save data
    rows = []
    for gs, sa, sn in [('SynGO', s_all, s_no), ('JASPAR_TF', t_all, t_no)]:
        for setting, row in [('all', sa), ('exclude_l5et', sn)]:
            rows.append({
                'gene_set': gs, 'setting': setting,
                'emp_neuronal': row['emp_neuronal'],
                'pred_neuronal': row['pred_neuronal'],
                'emp_non_neuronal': row['emp_non_neuronal'],
                'pred_non_neuronal': row['pred_non_neuronal'],
                'emp_diff': row['emp_diff'],
                'pred_diff': row['pred_diff'],
            })
    pd.DataFrame(rows).to_csv(
        os.path.join(OUT_DIR, "aim1_figure_A1_5_l5et_sensitivity_data.csv"), index=False)


# ============================================================
# Figure A1.6: TF diagnostic waterfall
# ============================================================
def make_figure_A1_6():
    print("Figure A1.6: TF diagnostic waterfall")
    diag_df = pd.read_csv(os.path.join(TF_DIAG_DIR, "tf_diagnostic_summary.csv"))

    # Extract counts by stage name
    def get_count(stage_substr):
        matches = diag_df[diag_df['stage'].str.contains(stage_substr, case=False)]
        return int(matches.iloc[0]['count']) if len(matches) > 0 else 0

    motif_rows = get_count('Mouse CORE motif')
    unique_syms = get_count('Unique TF symbols after')
    in_model = get_count('TF symbols in model')
    missing_common = get_count('Missing TFs in common_genes')
    missing_absent = get_count('Missing TFs absent')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                    gridspec_kw={'width_ratios': [1.3, 1], 'wspace': 0.4})

    # --- Left: Waterfall ---
    labels = [
        f'JASPAR CORE\nmouse motif rows',
        f'Unique TF symbols\n(after :: split)',
        f'In model\n(10,134 genes)',
    ]
    values = [motif_rows, unique_syms, in_model]
    colors = ['#4472C4', '#ED7D31', '#70AD47']

    bars = ax1.bar(range(len(values)), values, color=colors, width=0.55,
                   edgecolor='white', linewidth=1.5)

    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 str(val), ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Drop annotations
    for i in range(len(values) - 1):
        drop = values[i] - values[i+1]
        pct = drop / values[i] * 100
        mid_x = i + 0.5
        mid_y = (values[i] + values[i+1]) / 2
        ax1.annotate(f'\u2013{drop} ({pct:.0f}%)',
                     xy=(mid_x, mid_y), fontsize=10, ha='center', va='center',
                     color='#666',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                               edgecolor='#ccc', alpha=0.9))

    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel('Count', fontsize=FONTSIZE_LABEL)
    ax1.set_title('JASPAR CORE Mouse TF Filtering', fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylim(0, max(values) * 1.2)

    # --- Right: Missing gene breakdown ---
    n_missing = unique_syms - in_model
    pie_vals = [missing_common, missing_absent, in_model]
    pie_labels = [
        f'In common_genes\nbut <5% detection\n(n={missing_common})',
        f'Absent from shared\ncell/nuclei space\n(n={missing_absent})',
        f'In model\n(n={in_model})',
    ]
    pie_colors = ['#F4D03F', '#E74C3C', '#70AD47']

    wedges, texts, autotexts = ax2.pie(
        pie_vals, labels=pie_labels, colors=pie_colors, autopct='%1.0f%%',
        startangle=140, textprops={'fontsize': 9},
        pctdistance=0.75, labeldistance=1.2)
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight('bold')

    ax2.set_title(f'Fate of {unique_syms} TF Symbols', fontsize=FONTSIZE_TITLE, fontweight='bold')

    # QC notes
    fig.text(0.5, 0.01,
             'QC: :: complex splitting correct | hyphens preserved | '
             'case-insensitive matching applied\n'
             'Label: "JASPAR CORE motif-supported TF contrast set (n=63)"',
             ha='center', fontsize=9, fontstyle='italic', color='#555')

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    _save(fig, os.path.join(OUT_DIR, "aim1_figure_A1_6_tf_diagnostic"))

    # Save data
    pd.DataFrame([
        {'stage': 'JASPAR CORE mouse motif rows', 'count': motif_rows},
        {'stage': 'Unique TF symbols (after :: split)', 'count': unique_syms},
        {'stage': 'In model (10,134 genes)', 'count': in_model},
        {'stage': 'Missing: in common_genes but <5% detection', 'count': missing_common},
        {'stage': 'Missing: absent from shared gene space', 'count': missing_absent},
    ]).to_csv(os.path.join(OUT_DIR, "aim1_figure_A1_6_tf_diagnostic_data.csv"), index=False)


# ============================================================
# Figure A1.7: SynGO subgroup validation
# ============================================================
def make_figure_A1_7():
    print("Figure A1.7: SynGO subgroups")
    sub_df = pd.read_csv(os.path.join(SYNGO_DIR, "syngo_subgroup_summary.csv"))

    # Filter to key subgroups
    target_subgroups = ['broad_synapse', 'presynaptic', 'postsynaptic',
                        'postsynaptic_density', 'presynaptic_active_zone',
                        'synaptic_vesicle']
    sub_df = sub_df[sub_df['subgroup'].isin(target_subgroups)].copy()

    # Reorder
    sub_df['order'] = sub_df['subgroup'].map({s: i for i, s in enumerate(target_subgroups)})
    sub_df = sub_df.sort_values('order').reset_index(drop=True)

    # Pretty labels
    pretty = {
        'broad_synapse': f'All SynGO\n(n={sub_df[sub_df.subgroup=="broad_synapse"].iloc[0]["n_genes"]})',
        'presynaptic': f'Presynaptic\n(n={sub_df[sub_df.subgroup=="presynaptic"].iloc[0]["n_genes"]})',
        'postsynaptic': f'Postsynaptic\n(n={sub_df[sub_df.subgroup=="postsynaptic"].iloc[0]["n_genes"]})',
        'postsynaptic_density': f'Postsynaptic\ndensity\n(n={sub_df[sub_df.subgroup=="postsynaptic_density"].iloc[0]["n_genes"]})',
        'presynaptic_active_zone': f'Presynaptic\nactive zone\n(n={sub_df[sub_df.subgroup=="presynaptic_active_zone"].iloc[0]["n_genes"]})',
        'synaptic_vesicle': f'Synaptic\nvesicle\n(n={sub_df[sub_df.subgroup=="synaptic_vesicle"].iloc[0]["n_genes"]})',
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: neuronal vs non-neuronal by subgroup
    x = np.arange(len(sub_df))
    w = 0.2

    ax1.bar(x - 1.5*w, sub_df['emp_neuronal'], w, color=COLOR_NEURONAL, alpha=0.8,
            label='Emp. neuronal')
    ax1.bar(x - 0.5*w, sub_df['pred_neuronal'], w, color=COLOR_PREDICTED, alpha=0.8,
            label='Pred. neuronal')
    ax1.bar(x + 0.5*w, sub_df['emp_non_neuronal'], w, color=COLOR_NON_NEURONAL, alpha=0.8,
            label='Emp. non-neuronal')
    ax1.bar(x + 1.5*w, sub_df['pred_non_neuronal'], w, color='#8FBC8F', alpha=0.8,
            label='Pred. non-neuronal')

    ax1.set_xticks(x)
    ax1.set_xticklabels([pretty[s] for s in sub_df['subgroup']], fontsize=8.5)
    ax1.set_ylabel('Mean shift (log1p)', fontsize=FONTSIZE_LABEL)
    ax1.set_title('SynGO Subgroup: Neuronal vs Non-Neuronal',
                  fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax1.legend(fontsize=8, loc='upper right', ncol=2)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel B: empirical vs predicted diff
    ax2.bar(x - w/2, sub_df['emp_neuro_diff'], w, color=COLOR_EMPIRICAL, label='Empirical diff')
    ax2.bar(x + w/2, sub_df['pred_neuro_diff'], w, color=COLOR_PREDICTED, label='Predicted diff')

    for i, row in sub_df.iterrows():
        idx = sub_df.index.get_loc(i)
        ax2.text(idx - w/2, row['emp_neuro_diff'] + 0.01, f"{row['emp_neuro_diff']:.3f}",
                 ha='center', va='bottom', fontsize=7.5)
        ax2.text(idx + w/2, row['pred_neuro_diff'] + 0.01, f"{row['pred_neuro_diff']:.3f}",
                 ha='center', va='bottom', fontsize=7.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels([pretty[s] for s in sub_df['subgroup']], fontsize=8.5)
    ax2.set_ylabel('Neuronal \u2013 Non-neuronal diff', fontsize=FONTSIZE_LABEL)
    ax2.set_title('SynGO Subgroup: Neuronal Enrichment',
                  fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.text(0.5, 0.01,
             'Caution: SynGO subgroups annotate protein function/location, not RNA localization.',
             ha='center', fontsize=9, fontstyle='italic', color=COLOR_CAUTION)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, os.path.join(OUT_DIR, "aim1_figure_A1_7_syngo_subgroups"))

    sub_df.to_csv(os.path.join(OUT_DIR, "aim1_figure_A1_7_syngo_subgroups_data.csv"), index=False)


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("Aim 1 Biological Validation Figure Set")
    print(f"Output: {OUT_DIR}")
    print("=" * 70)

    # Verify input files exist
    required = {
        'syngo_gene_celltype_deltas': os.path.join(SYNGO_DIR, "syngo_gene_celltype_deltas.csv"),
        'syngo_overall_quality': os.path.join(SYNGO_DIR, "syngo_overall_quality.csv"),
        'syngo_celltype_summary': os.path.join(SYNGO_DIR, "syngo_celltype_summary.csv"),
        'syngo_neuronal_vs_nonneuronal': os.path.join(SYNGO_DIR, "syngo_neuronal_vs_nonneuronal.csv"),
        'syngo_sensitivity_exclude_l5et': os.path.join(SYNGO_DIR, "syngo_sensitivity_exclude_l5et.csv"),
        'syngo_subgroup_summary': os.path.join(SYNGO_DIR, "syngo_subgroup_summary.csv"),
        'tf_gene_celltype_deltas': os.path.join(TF_DIR, "tf_gene_celltype_deltas.csv"),
        'tf_overall_quality': os.path.join(TF_DIR, "tf_overall_quality.csv"),
        'tf_celltype_summary': os.path.join(TF_DIR, "tf_celltype_summary.csv"),
        'tf_neuronal_vs_nonneuronal': os.path.join(TF_DIR, "tf_neuronal_vs_nonneuronal.csv"),
        'tf_sensitivity': os.path.join(TF_DIR, "tf_sensitivity_neuronal_vs_nonneuronal.csv"),
        'tf_diagnostic': os.path.join(TF_DIAG_DIR, "tf_diagnostic_summary.csv"),
    }

    missing = []
    for name, path in required.items():
        if not os.path.exists(path):
            missing.append(f"  MISSING: {name} -> {path}")
        else:
            print(f"  OK: {name}")

    if missing:
        print("\nMissing input files:")
        for m in missing:
            print(m)
        print("Continuing with available files...\n")
    else:
        print("  All input files found.\n")

    # Generate figures
    make_figure_A1_1()
    make_figure_A1_2()
    make_figure_A1_3()
    make_figure_A1_4()
    make_figure_A1_5()
    make_figure_A1_6()
    make_figure_A1_7()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    outputs = sorted([f for f in os.listdir(OUT_DIR) if not f.startswith('scripts') and f != 'README.md'])
    print(f"Generated {len(outputs)} output files:")
    for f in outputs:
        print(f"  {f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
