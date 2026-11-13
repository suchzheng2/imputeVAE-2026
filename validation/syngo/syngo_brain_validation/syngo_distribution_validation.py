#!/usr/bin/env python
"""
SynGO distribution-level biological validation for ImputeVAE.

Extends the mean-shift analysis by asking whether cross-decoding recovers the
full population distribution of SynGO-gene expression within each cell type,
not only cell-type means.

Uses exactly the same 1,275 non-ribosomal SynGO genes, validation observations,
gene ordering, cell-type definitions and subgroups as the completed mean-shift
analysis.

Usage:
    python -u validation/syngo/syngo_brain_validation/syngo_distribution_validation.py

Requires conda env: impute2
"""

import os
import sys
import io
import zipfile
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats, sparse
from scipy.stats import wasserstein_distance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import anndata as ad

warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

SEED = 42
rng = np.random.RandomState(SEED)

# ============================================================
# Paths — identical to syngo_brain_validation.py
# ============================================================
PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
DATA_ROOT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi"

GENE_LIST_PATH = os.path.join(PROJECT, "data/training_gene_list_v3.csv")
SPLIT_MANIFEST = os.path.join(PROJECT, "data/split_manifest.tsv")
COMMON_CT_PATH = os.path.join(PROJECT, "data/common_cell_types.csv")
COMMON_GENES_PATH = os.path.join(PROJECT, "data/common_genes.csv")
ORTHOLOG_PATH = os.path.join(PROJECT, "data/external/mouse_human_orthologs.csv")
SYNGO_ZIP = os.path.join(PROJECT, "validation/syngo/syngo1.3_complete_data.zip")
IMPUTED_NPY = os.path.join(PROJECT, "run/eval_v3_splitaware/cvae_imputed_valid.npy")
IMPUTED_OBS = os.path.join(PROJECT, "run/eval_v3_splitaware/valid_nuc_obs.csv")

# Existing mean-shift results for cross-check
MEAN_SHIFT_DELTAS = os.path.join(PROJECT,
    "validation/syngo/syngo_brain_validation/syngo_gene_celltype_deltas.csv")
MEAN_SHIFT_METADATA = os.path.join(PROJECT,
    "validation/syngo/syngo_brain_validation/syngo_gene_metadata.csv")
MEAN_SHIFT_FILTERING = os.path.join(PROJECT,
    "validation/syngo/syngo_brain_validation/syngo_filtering_counts.csv")

OUT_DIR = os.path.join(PROJECT, "validation/syngo/syngo_brain_validation")

NEURONAL = ['L2/3 IT', 'L5 ET', 'L5 IT', 'L5/6 NP', 'L6 CT', 'L6 IT', 'L6b',
            'Lamp5', 'Pvalb', 'Sncg', 'Sst', 'Vip']
NON_NEURONAL = ['Astro', 'Endo', 'Macrophage', 'OPC', 'Oligo']
LOW_SAMPLE_CTS = ['Endo', 'Macrophage', 'OPC']

SUBGROUPS = {
    'broad_synapse':          {'go_ids': ['GO:0045202']},
    'presynaptic':            {'go_ids': ['GO:0098793']},
    'postsynaptic':           {'go_ids': ['GO:0098794']},
    'postsynaptic_density':   {'go_ids': ['GO:0014069']},
    'presynaptic_active_zone':{'go_ids': ['GO:0048786']},
    'synaptic_vesicle':       {'go_ids': ['GO:0008021']},
}

N_BOOT = 1000

# Detection thresholds for sensitivity analysis (log1p scale)
# Prespecified before examining results: 0 (primary), 0.1, 0.5, 1.0
DETECTION_THRESHOLDS = [0.0, 0.1, 0.5, 1.0]

# Minimum denominator for normalized Wasserstein improvement
MIN_W_RAW_DENOM = 0.01  # prespecified

# Minimum observations per distribution for positive-expression analysis
MIN_POSITIVE_OBS = 10

# Example genes for supplementary figure — selected by subgroup membership
EXAMPLE_GENES = {
    'Syn1': 'presynaptic',       # Synapsin-1, presynaptic vesicle
    'Dlg4': 'postsynaptic_density',  # PSD-95
    'Sv2a': 'synaptic_vesicle',  # SV2A, synaptic vesicle glycoprotein
}


def ct_status(ct):
    if ct in NEURONAL: return 'neuronal'
    if ct in NON_NEURONAL: return 'non-neuronal'
    return 'unknown'


# ============================================================
# 1. Load data (identical to mean-shift analysis)
# ============================================================

def load_data():
    """Load all required matrices with validation-only subsetting."""
    print("=" * 70)
    print("Loading data")
    print("=" * 70)

    gene_list = pd.read_csv(GENE_LIST_PATH, header=None)[0].tolist()
    manifest = pd.read_csv(SPLIT_MANIFEST, sep='\t')
    common_ct = pd.read_csv(COMMON_CT_PATH)['common_cell_types'].tolist()
    common_genes = pd.read_csv(COMMON_GENES_PATH)['common_genes'].tolist()

    print(f"  Gene list: {len(gene_list)} genes")
    print(f"  Split manifest: {len(manifest)} entries")

    def get_indices(modality):
        mod = manifest[manifest.modality == modality].sort_values('obs_id').reset_index(drop=True)
        valid_mask = mod.partition == 'valid'
        return np.where(valid_mask)[0], mod[valid_mask].reset_index(drop=True)

    valid_nuc_idx, valid_nuc_meta = get_indices('nuclei')
    valid_cell_idx, valid_cell_meta = get_indices('cell')
    print(f"  Valid nuclei: {len(valid_nuc_idx)}, Valid cells: {len(valid_cell_idx)}")

    def load_h5ad(data_dir, batch):
        path = os.path.join(DATA_ROOT, data_dir, "data.h5ad")
        print(f"  Loading {path}...")
        adata = ad.read_h5ad(path)
        if 'meta_batch_id' not in adata.obs:
            adata.obs['meta_batch_id'] = adata.obs['batch_id'].apply(lambda x: x.split('.')[0])
        row_mask = (adata.obs['meta_batch_id'].str.endswith(batch) &
                    adata.obs['cell_type'].isin(common_ct))
        col_mask = (adata.var['gene_names'].isin(common_genes) &
                    adata.var['gene_names'].isin(gene_list))
        adata = adata[row_mask, col_mask].copy()
        adata = adata[:, adata.var['gene_names'].argsort()].copy()
        X = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
        adata.X = np.log1p(X).astype(np.float32)
        gene_names = adata.var['gene_names'].values.tolist()
        print(f"    Shape after filtering: {adata.shape}")
        return adata, gene_names

    cell_adata, gn_cell = load_h5ad("10X_cells_v2_v3_AIBS", "c3A")
    nuc_adata, gn_nuc = load_h5ad("10X_nuclei_v2_v3_AIBS", "n3A")
    assert gn_cell == gn_nuc, "Gene names mismatch"
    gene_names = gn_cell
    print(f"  Gene alignment verified: {len(gene_names)} genes")

    valid_cell_X = cell_adata.X[valid_cell_idx]
    valid_cell_ct = cell_adata.obs.iloc[valid_cell_idx]['cell_type'].values
    valid_nuc_X = nuc_adata.X[valid_nuc_idx]
    valid_nuc_ct = nuc_adata.obs.iloc[valid_nuc_idx]['cell_type'].values

    del cell_adata, nuc_adata

    print(f"  Loading imputed: {IMPUTED_NPY}")
    imputed_X = np.load(IMPUTED_NPY)
    imputed_obs = pd.read_csv(IMPUTED_OBS)
    imputed_ct = imputed_obs['cell_type'].values

    print(f"    Imputed shape: {imputed_X.shape}")
    assert imputed_X.shape[0] == valid_nuc_X.shape[0]
    assert imputed_X.shape[1] == len(gene_names)
    ct_match = (imputed_ct == valid_nuc_ct).mean()
    assert ct_match > 0.99, f"CT alignment only {ct_match*100:.1f}%"
    print(f"    CT alignment: {ct_match*100:.1f}%")

    # Range checks
    print(f"    Cell range: [{valid_cell_X.min():.2f}, {valid_cell_X.max():.2f}]")
    print(f"    Nuc range: [{valid_nuc_X.min():.2f}, {valid_nuc_X.max():.2f}]")
    print(f"    Imputed range: [{imputed_X.min():.2f}, {imputed_X.max():.2f}]")

    return (gene_names, gene_list,
            valid_cell_X, valid_cell_ct,
            valid_nuc_X, valid_nuc_ct,
            imputed_X, imputed_ct)


# ============================================================
# 2. Build SynGO gene sets (identical to mean-shift analysis)
# ============================================================

def build_syngo_sets(gene_names, gene_list):
    """Build SynGO gene sets and verify against mean-shift analysis."""
    print("\n" + "=" * 70)
    print("Building SynGO gene sets")
    print("=" * 70)

    zf = zipfile.ZipFile(SYNGO_ZIP)
    ontologies = pd.read_excel(io.BytesIO(zf.read('ontologies.xlsx')))

    term_genes = {}
    for _, row in ontologies.iterrows():
        val = str(row.get('hgnc_symbol', ''))
        if val and val != 'nan':
            term_genes[row['id']] = {g.strip() for g in val.split(',') if g.strip()}
        else:
            term_genes[row['id']] = set()

    ortho = pd.read_csv(ORTHOLOG_PATH)
    human_counts = ortho.groupby('human_sym').size()
    one_to_one = set(human_counts[human_counts == 1].index)
    ortho_1to1 = ortho[ortho.human_sym.isin(one_to_one)]
    h2m = dict(zip(ortho_1to1.human_sym, ortho_1to1.mouse_sym))

    gene_set_model = set(gene_names)
    def map_human_to_mouse(human_sym):
        m = h2m.get(human_sym)
        if m: return m, 'ortholog'
        cap = human_sym[0] + human_sym[1:].lower() if human_sym else None
        if cap and cap in gene_set_model:
            return cap, 'capitalization'
        return None, 'unmapped'

    broad_human = term_genes.get('GO:0045202', set())

    records = []
    for h in sorted(broad_human):
        mouse, method = map_human_to_mouse(h)
        in_model = mouse in gene_set_model if mouse else False
        is_ribo = (mouse or '').startswith('Rpl') or (mouse or '').startswith('Rps')
        memberships = {}
        for sg_name, sg_spec in SUBGROUPS.items():
            for go_id in sg_spec['go_ids']:
                if h in term_genes.get(go_id, set()):
                    memberships[sg_name] = True
                    break
        records.append({
            'human_symbol': h,
            'mouse_symbol': mouse,
            'mapping_method': method,
            'in_model': in_model,
            'is_ribosomal': is_ribo,
            **{f'sg_{k}': memberships.get(k, False) for k in SUBGROUPS},
        })

    meta = pd.DataFrame(records)
    primary = meta[(meta.in_model) & (~meta.is_ribosomal)].copy()
    primary_genes = sorted(set(primary.mouse_symbol))

    subgroup_genes = {}
    for sg_name in SUBGROUPS:
        sg_col = f'sg_{sg_name}'
        sg_set = set(primary[primary[sg_col]].mouse_symbol)
        subgroup_genes[sg_name] = sg_set

    # ---- Cross-check against mean-shift analysis ----
    print("  Cross-checking against mean-shift analysis...")

    ms_meta = pd.read_csv(MEAN_SHIFT_METADATA)
    ms_primary = ms_meta[(ms_meta.in_model) & (~ms_meta.is_ribosomal)]
    ms_genes = sorted(set(ms_primary.mouse_symbol))

    if set(primary_genes) != set(ms_genes):
        diff1 = set(primary_genes) - set(ms_genes)
        diff2 = set(ms_genes) - set(primary_genes)
        print(f"  FATAL: Gene set mismatch! "
              f"New-only: {len(diff1)}, Mean-shift-only: {len(diff2)}")
        sys.exit(1)

    ms_filtering = pd.read_csv(MEAN_SHIFT_FILTERING)
    ms_count = ms_filtering[ms_filtering.stage == 'non_ribosomal_primary']['count'].values[0]
    assert len(primary_genes) == ms_count, \
        f"Gene count mismatch: {len(primary_genes)} vs {ms_count}"

    ms_deltas = pd.read_csv(MEAN_SHIFT_DELTAS)
    ms_delta_genes = sorted(ms_deltas.gene.unique())
    assert ms_delta_genes == primary_genes, "Delta CSV gene set mismatch"

    print(f"  VERIFIED: {len(primary_genes)} genes match mean-shift analysis exactly")
    print(f"  Subgroups: {', '.join(f'{k}={len(v)}' for k, v in subgroup_genes.items())}")

    return primary_genes, subgroup_genes, meta


# ============================================================
# 3. Inference audit
# ============================================================

def inference_audit(imputed_X):
    """Document inference pipeline properties."""
    print("\n" + "=" * 70)
    print("Inference audit")
    print("=" * 70)

    n_obs, n_genes = imputed_X.shape
    exact_zeros = (imputed_X == 0.0).sum()
    near_zeros = (np.abs(imputed_X) < 1e-6).sum()
    total = n_obs * n_genes
    negative = (imputed_X < 0).sum()
    min_val = imputed_X.min()
    max_val = imputed_X.max()

    # Check if dense (almost no exact zeros)
    zero_frac = exact_zeros / total

    audit = {
        'n_observations': n_obs,
        'n_genes': n_genes,
        'exact_zeros': int(exact_zeros),
        'exact_zero_fraction': float(zero_frac),
        'near_zero_fraction': float(near_zeros / total),
        'negative_values': int(negative),
        'min_value': float(min_val),
        'max_value': float(max_val),
        'is_dense': bool(zero_frac < 0.01),
        'encoding': 'posterior_mean',
        'decoding': 'conditional_mean_deterministic',
        'n_draws': 1,
        'clipping': 'none',
        'final_activation': 'LeakyReLU(0.1)',
        'scale': 'log1p',
    }

    print(f"  Encoding: posterior mean (mu_latent, no sampling)")
    print(f"  Decoding: deterministic conditional mean (no sampling)")
    print(f"  Draws per nucleus: 1")
    print(f"  Final activation: LeakyReLU(0.1) — allows small negative values")
    print(f"  Clipping/thresholding: none")
    print(f"  Scale: log1p")
    print(f"  Exact zeros: {exact_zeros}/{total} ({zero_frac:.6f})")
    print(f"  Near zeros (<1e-6): {near_zeros}/{total} ({near_zeros/total:.6f})")
    print(f"  Negative values: {negative}/{total} ({negative/total:.6f})")
    print(f"  Range: [{min_val:.4f}, {max_val:.4f}]")
    print(f"  Dense output: {'YES' if audit['is_dense'] else 'NO'}")

    if audit['is_dense']:
        print("\n  NOTE: Deterministic mean decoding produces a dense output.")
        print("  Detection-rate analysis at threshold=0 will show near-100% detection")
        print("  in the imputed matrix. This is a known limitation of VAE mean decoding,")
        print("  not a model failure.")

    return audit


# ============================================================
# 4. Analysis 1: Wasserstein distributional recovery
# ============================================================

def analysis1_wasserstein(gene_names, syngo_genes,
                          valid_cell_X, valid_cell_ct,
                          valid_nuc_X, valid_nuc_ct,
                          imputed_X, imputed_ct):
    """Wasserstein-1 distance between distributions for each gene-CT pair."""
    print("\n" + "=" * 70)
    print("Analysis 1: Wasserstein distributional recovery")
    print("=" * 70)

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    common_cts = sorted(set(valid_cell_ct) & set(valid_nuc_ct))

    # Zero-fraction audit per matrix
    cell_zeros = (valid_cell_X == 0).sum()
    nuc_zeros = (valid_nuc_X == 0).sum()
    imp_zeros = (imputed_X == 0).sum()
    print(f"  Exact-zero fractions:")
    print(f"    Cells: {cell_zeros / valid_cell_X.size:.4f}")
    print(f"    Nuclei: {nuc_zeros / valid_nuc_X.size:.4f}")
    print(f"    Imputed: {imp_zeros / imputed_X.size:.4f}")

    records = []
    for ct in common_cts:
        cell_mask = valid_cell_ct == ct
        nuc_mask = valid_nuc_ct == ct
        imp_mask = imputed_ct == ct

        cell_ct = valid_cell_X[cell_mask]
        nuc_ct = valid_nuc_X[nuc_mask]
        imp_ct = imputed_X[imp_mask]

        n_cell = cell_mask.sum()
        n_nuc = nuc_mask.sum()

        for g in syngo_genes:
            gi = gene_to_idx[g]
            cell_vals = cell_ct[:, gi]
            nuc_vals = nuc_ct[:, gi]
            imp_vals = imp_ct[:, gi]

            w_raw = wasserstein_distance(nuc_vals, cell_vals)
            w_imp = wasserstein_distance(imp_vals, cell_vals)
            abs_imp = w_raw - w_imp

            # Normalized improvement
            if w_raw >= MIN_W_RAW_DENOM:
                norm_imp = 1.0 - w_imp / w_raw
                norm_eligible = True
            else:
                norm_imp = np.nan
                norm_eligible = False

            records.append({
                'gene': g,
                'cell_type': ct,
                'neuronal_status': ct_status(ct),
                'low_sample': ct in LOW_SAMPLE_CTS,
                'n_cell': int(n_cell),
                'n_nucleus': int(n_nuc),
                'W_raw': float(w_raw),
                'W_imputed': float(w_imp),
                'absolute_improvement': float(abs_imp),
                'normalized_improvement': float(norm_imp) if norm_eligible else np.nan,
                'norm_eligible': norm_eligible,
            })

        if common_cts.index(ct) % 5 == 0:
            print(f"    {ct} done ({n_cell} cells, {n_nuc} nuclei)")

    w_df = pd.DataFrame(records)
    print(f"  Total gene-CT pairs: {len(w_df)}")
    print(f"  Norm-eligible: {w_df.norm_eligible.sum()}/{len(w_df)}")

    # Summary
    improved = (w_df.absolute_improvement > 0).mean()
    med_abs = w_df.absolute_improvement.median()
    med_w_raw = w_df.W_raw.median()
    med_w_imp = w_df.W_imputed.median()
    elig = w_df[w_df.norm_eligible]
    med_norm = elig.normalized_improvement.median() if len(elig) > 0 else np.nan

    print(f"\n  Overall summary:")
    print(f"    Median W_raw: {med_w_raw:.4f}")
    print(f"    Median W_imputed: {med_w_imp:.4f}")
    print(f"    Median absolute improvement: {med_abs:.4f}")
    print(f"    Median normalized improvement (eligible): {med_norm:.4f}")
    print(f"    Proportion improved: {improved:.4f}")

    return w_df


# ============================================================
# 5. Analysis 2: Detection-rate recovery
# ============================================================

def analysis2_detection(gene_names, syngo_genes,
                        valid_cell_X, valid_cell_ct,
                        valid_nuc_X, valid_nuc_ct,
                        imputed_X, imputed_ct,
                        audit_info):
    """Detection-rate recovery analysis."""
    print("\n" + "=" * 70)
    print("Analysis 2: Detection-rate recovery")
    print("=" * 70)

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    common_cts = sorted(set(valid_cell_ct) & set(valid_nuc_ct))

    all_threshold_results = {}

    for thresh in DETECTION_THRESHOLDS:
        label = 'primary' if thresh == 0.0 else f'threshold_{thresh}'
        print(f"\n  Detection threshold: {thresh} ({label})")

        records = []
        for ct in common_cts:
            cell_mask = valid_cell_ct == ct
            nuc_mask = valid_nuc_ct == ct
            imp_mask = imputed_ct == ct

            cell_ct = valid_cell_X[cell_mask]
            nuc_ct = valid_nuc_X[nuc_mask]
            imp_ct = imputed_X[imp_mask]

            for g in syngo_genes:
                gi = gene_to_idx[g]
                cell_det = (cell_ct[:, gi] > thresh).mean()
                nuc_det = (nuc_ct[:, gi] > thresh).mean()
                imp_det = (imp_ct[:, gi] > thresh).mean()

                raw_err = abs(nuc_det - cell_det)
                imp_err = abs(imp_det - cell_det)
                reduction = raw_err - imp_err

                records.append({
                    'gene': g,
                    'cell_type': ct,
                    'neuronal_status': ct_status(ct),
                    'low_sample': ct in LOW_SAMPLE_CTS,
                    'threshold': thresh,
                    'cell_detection_rate': float(cell_det),
                    'nucleus_detection_rate': float(nuc_det),
                    'imputed_detection_rate': float(imp_det),
                    'raw_detection_error': float(raw_err),
                    'imputed_detection_error': float(imp_err),
                    'detection_error_reduction': float(reduction),
                })

        det_df = pd.DataFrame(records)

        improved = (det_df.detection_error_reduction > 0).mean()
        med_raw = det_df.raw_detection_error.median()
        med_imp = det_df.imputed_detection_error.median()
        med_red = det_df.detection_error_reduction.median()
        mean_imp_det = det_df.imputed_detection_rate.mean()

        print(f"    Mean imputed detection rate: {mean_imp_det:.4f}")
        print(f"    Median raw detection error: {med_raw:.4f}")
        print(f"    Median imputed detection error: {med_imp:.4f}")
        print(f"    Median error reduction: {med_red:.4f}")
        print(f"    Proportion improved: {improved:.4f}")

        if thresh == 0.0 and audit_info['is_dense']:
            print(f"\n    WARNING: Imputed matrix is dense — nearly all values > 0.")
            print(f"    At threshold=0, imputed detection rate is ~{mean_imp_det:.3f}.")
            print(f"    This makes detection-rate comparison at threshold=0 uninformative")
            print(f"    for the imputed matrix. See sensitivity thresholds for meaningful")
            print(f"    comparisons.")

        all_threshold_results[label] = det_df

    return all_threshold_results


# ============================================================
# 6. Analysis 3: Positive-expression distributions
# ============================================================

def analysis3_positive_expression(gene_names, syngo_genes,
                                  valid_cell_X, valid_cell_ct,
                                  valid_nuc_X, valid_nuc_ct,
                                  imputed_X, imputed_ct,
                                  audit_info):
    """Wasserstein distance on positive-expression subset."""
    print("\n" + "=" * 70)
    print("Analysis 3: Positive-expression distributions")
    print("=" * 70)

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    common_cts = sorted(set(valid_cell_ct) & set(valid_nuc_ct))

    is_dense = audit_info['is_dense']
    if is_dense:
        print("  Imputed output is dense — conditioning on >0 is not meaningful.")
        print("  Using prespecified thresholds for sensitivity analysis.")
        # Use non-zero thresholds only
        thresholds = [t for t in DETECTION_THRESHOLDS if t > 0]
    else:
        thresholds = DETECTION_THRESHOLDS

    all_results = {}
    for thresh in thresholds:
        label = f'threshold_{thresh}'
        print(f"\n  Positive-expression threshold: {thresh} ({label})")

        records = []
        n_excluded_insufficient = 0

        for ct in common_cts:
            cell_mask = valid_cell_ct == ct
            nuc_mask = valid_nuc_ct == ct
            imp_mask = imputed_ct == ct

            cell_ct = valid_cell_X[cell_mask]
            nuc_ct = valid_nuc_X[nuc_mask]
            imp_ct = imputed_X[imp_mask]

            for g in syngo_genes:
                gi = gene_to_idx[g]
                cell_pos = cell_ct[:, gi][cell_ct[:, gi] > thresh]
                nuc_pos = nuc_ct[:, gi][nuc_ct[:, gi] > thresh]
                imp_pos = imp_ct[:, gi][imp_ct[:, gi] > thresh]

                # Check minimum observations
                reason = None
                if len(cell_pos) < MIN_POSITIVE_OBS:
                    reason = f'cell<{MIN_POSITIVE_OBS}'
                elif len(nuc_pos) < MIN_POSITIVE_OBS:
                    reason = f'nuc<{MIN_POSITIVE_OBS}'
                elif len(imp_pos) < MIN_POSITIVE_OBS:
                    reason = f'imp<{MIN_POSITIVE_OBS}'

                if reason:
                    n_excluded_insufficient += 1
                    records.append({
                        'gene': g,
                        'cell_type': ct,
                        'neuronal_status': ct_status(ct),
                        'low_sample': ct in LOW_SAMPLE_CTS,
                        'threshold': thresh,
                        'eligible': False,
                        'exclusion_reason': reason,
                        'n_cell_positive': len(cell_pos),
                        'n_nuc_positive': len(nuc_pos),
                        'n_imp_positive': len(imp_pos),
                        'W_positive_raw': np.nan,
                        'W_positive_imputed': np.nan,
                        'positive_expression_improvement': np.nan,
                    })
                    continue

                w_raw = wasserstein_distance(nuc_pos, cell_pos)
                w_imp = wasserstein_distance(imp_pos, cell_pos)
                improvement = w_raw - w_imp

                records.append({
                    'gene': g,
                    'cell_type': ct,
                    'neuronal_status': ct_status(ct),
                    'low_sample': ct in LOW_SAMPLE_CTS,
                    'threshold': thresh,
                    'eligible': True,
                    'exclusion_reason': None,
                    'n_cell_positive': len(cell_pos),
                    'n_nuc_positive': len(nuc_pos),
                    'n_imp_positive': len(imp_pos),
                    'W_positive_raw': float(w_raw),
                    'W_positive_imputed': float(w_imp),
                    'positive_expression_improvement': float(improvement),
                })

        pos_df = pd.DataFrame(records)
        eligible = pos_df[pos_df.eligible]

        print(f"    Total pairs: {len(pos_df)}")
        print(f"    Eligible: {len(eligible)} ({len(eligible)/len(pos_df):.3f})")
        print(f"    Excluded (insufficient obs): {n_excluded_insufficient}")

        if len(eligible) > 0:
            med_imp = eligible.positive_expression_improvement.median()
            prop_imp = (eligible.positive_expression_improvement > 0).mean()
            print(f"    Median improvement: {med_imp:.4f}")
            print(f"    Proportion improved: {prop_imp:.4f}")

        all_results[label] = pos_df

    return all_results


# ============================================================
# 7. Per-cell-type and group summaries
# ============================================================

def summarize_by_celltype(w_df, det_results, pos_results):
    """Summarize all metrics by cell type."""
    print("\n" + "=" * 70)
    print("Per-cell-type summaries")
    print("=" * 70)

    common_cts = sorted(w_df.cell_type.unique())

    records = []
    for ct in common_cts:
        ct_w = w_df[w_df.cell_type == ct]

        # Wasserstein
        med_w_raw = ct_w.W_raw.median()
        med_w_imp = ct_w.W_imputed.median()
        med_abs = ct_w.absolute_improvement.median()
        prop_imp_w = (ct_w.absolute_improvement > 0).mean()
        elig = ct_w[ct_w.norm_eligible]
        med_norm = elig.normalized_improvement.median() if len(elig) > 0 else np.nan

        # Bootstrap CI on median absolute improvement (resample genes)
        boot_med = np.empty(N_BOOT)
        vals = ct_w.absolute_improvement.values
        for b in range(N_BOOT):
            idx = rng.choice(len(vals), len(vals), replace=True)
            boot_med[b] = np.median(vals[idx])
        ci_lo = np.percentile(boot_med, 2.5)
        ci_hi = np.percentile(boot_med, 97.5)

        rec = {
            'cell_type': ct,
            'neuronal_status': ct_status(ct),
            'low_sample': ct in LOW_SAMPLE_CTS,
            'n_cell': int(ct_w.n_cell.iloc[0]),
            'n_nucleus': int(ct_w.n_nucleus.iloc[0]),
            'n_genes': len(ct_w),
            'median_W_raw': float(med_w_raw),
            'median_W_imputed': float(med_w_imp),
            'median_absolute_improvement': float(med_abs),
            'abs_imp_ci_lo': float(ci_lo),
            'abs_imp_ci_hi': float(ci_hi),
            'proportion_improved_W': float(prop_imp_w),
            'median_normalized_improvement': float(med_norm) if not np.isnan(med_norm) else np.nan,
            'n_norm_eligible': int(len(elig)),
        }

        # Detection (primary threshold=0)
        if 'primary' in det_results:
            det0 = det_results['primary']
            ct_det = det0[det0.cell_type == ct]
            rec['median_raw_detection_error'] = float(ct_det.raw_detection_error.median())
            rec['median_imputed_detection_error'] = float(ct_det.imputed_detection_error.median())
            rec['median_detection_error_reduction'] = float(ct_det.detection_error_reduction.median())
            rec['proportion_improved_detection'] = float((ct_det.detection_error_reduction > 0).mean())

        # Best non-zero threshold for positive expression
        best_pos_key = None
        for k in sorted(pos_results.keys()):
            df = pos_results[k]
            elig_ct = df[(df.cell_type == ct) & (df.eligible)]
            if len(elig_ct) > 50:  # reasonable number of eligible genes
                best_pos_key = k
                break

        if best_pos_key:
            pdf = pos_results[best_pos_key]
            ct_pos = pdf[(pdf.cell_type == ct) & (pdf.eligible)]
            rec['positive_threshold_used'] = float(ct_pos.threshold.iloc[0]) if len(ct_pos) > 0 else np.nan
            rec['median_positive_improvement'] = float(ct_pos.positive_expression_improvement.median())
            rec['proportion_improved_positive'] = float((ct_pos.positive_expression_improvement > 0).mean())
            rec['n_eligible_positive'] = int(len(ct_pos))

        records.append(rec)
        print(f"  {ct:15s}: med_abs_imp={med_abs:.4f}, prop_imp={prop_imp_w:.3f}, "
              f"CI=[{ci_lo:.4f}, {ci_hi:.4f}]")

    ct_summary = pd.DataFrame(records)
    return ct_summary


def summarize_by_group(w_df, det_results, pos_results):
    """Summarize by neuronal/non-neuronal/low-sample groups."""
    print("\n" + "=" * 70)
    print("Group summaries")
    print("=" * 70)

    groups = {
        'all': w_df,
        'neuronal': w_df[w_df.neuronal_status == 'neuronal'],
        'non_neuronal': w_df[w_df.neuronal_status == 'non-neuronal'],
        'low_sample': w_df[w_df.low_sample],
        'excluding_low_sample': w_df[~w_df.low_sample],
    }

    records = []
    for gname, gdf in groups.items():
        if len(gdf) == 0:
            continue

        rec = {
            'group': gname,
            'n_gene_ct_pairs': len(gdf),
            'median_W_raw': float(gdf.W_raw.median()),
            'median_W_imputed': float(gdf.W_imputed.median()),
            'median_absolute_improvement': float(gdf.absolute_improvement.median()),
            'proportion_improved_W': float((gdf.absolute_improvement > 0).mean()),
            'mean_absolute_improvement': float(gdf.absolute_improvement.mean()),
        }

        elig = gdf[gdf.norm_eligible]
        rec['median_normalized_improvement'] = float(elig.normalized_improvement.median()) if len(elig) > 0 else np.nan

        # Detection
        if 'primary' in det_results:
            det0 = det_results['primary']
            # match by gene+ct
            cts_in_group = set(gdf.cell_type.unique())
            genes_in_group = set(gdf.gene.unique())
            det_g = det0[(det0.cell_type.isin(cts_in_group)) & (det0.gene.isin(genes_in_group))]
            rec['median_detection_error_reduction'] = float(det_g.detection_error_reduction.median())
            rec['proportion_improved_detection'] = float((det_g.detection_error_reduction > 0).mean())

        # Positive expression (first available threshold)
        for k in sorted(pos_results.keys()):
            pdf = pos_results[k]
            cts_in_group = set(gdf.cell_type.unique())
            genes_in_group = set(gdf.gene.unique())
            pos_g = pdf[(pdf.cell_type.isin(cts_in_group)) &
                        (pdf.gene.isin(genes_in_group)) &
                        (pdf.eligible)]
            if len(pos_g) > 100:
                rec['positive_threshold'] = float(pos_g.threshold.iloc[0])
                rec['median_positive_improvement'] = float(pos_g.positive_expression_improvement.median())
                rec['proportion_improved_positive'] = float((pos_g.positive_expression_improvement > 0).mean())
                break

        records.append(rec)
        print(f"  {gname:25s}: med_abs={rec['median_absolute_improvement']:.4f}, "
              f"prop_imp={rec['proportion_improved_W']:.3f}, n={len(gdf)}")

    group_df = pd.DataFrame(records)
    return group_df


# ============================================================
# 8. Analysis 4: Subgroup summaries
# ============================================================

def analysis4_subgroups(subgroup_genes, w_df, det_results, pos_results):
    """Repeat distributional summaries for SynGO subgroups."""
    print("\n" + "=" * 70)
    print("Analysis 4: SynGO subgroups")
    print("=" * 70)

    records = []
    for sg_name, sg_set in subgroup_genes.items():
        sg_w = w_df[w_df.gene.isin(sg_set)]
        n_genes = sg_w.gene.nunique()
        n_pairs = len(sg_w)

        if n_genes < 5:
            records.append({'subgroup': sg_name, 'n_genes': n_genes, 'status': 'skipped'})
            continue

        med_abs = sg_w.absolute_improvement.median()
        prop_imp_w = (sg_w.absolute_improvement > 0).mean()

        # Neuronal/non-neuronal
        neuro_w = sg_w[sg_w.neuronal_status == 'neuronal']
        non_w = sg_w[sg_w.neuronal_status == 'non-neuronal']
        neuro_med = neuro_w.absolute_improvement.median() if len(neuro_w) > 0 else np.nan
        non_med = non_w.absolute_improvement.median() if len(non_w) > 0 else np.nan

        rec = {
            'subgroup': sg_name,
            'n_genes': n_genes,
            'n_gene_ct_pairs': n_pairs,
            'status': 'computed',
            'median_absolute_improvement': float(med_abs),
            'proportion_improved_W': float(prop_imp_w),
            'neuronal_median_abs_imp': float(neuro_med) if not np.isnan(neuro_med) else np.nan,
            'non_neuronal_median_abs_imp': float(non_med) if not np.isnan(non_med) else np.nan,
        }

        # Detection
        if 'primary' in det_results:
            det0 = det_results['primary']
            sg_det = det0[det0.gene.isin(sg_set)]
            rec['median_detection_error_reduction'] = float(sg_det.detection_error_reduction.median())

        # Positive expression
        for k in sorted(pos_results.keys()):
            pdf = pos_results[k]
            sg_pos = pdf[(pdf.gene.isin(sg_set)) & (pdf.eligible)]
            if len(sg_pos) > 50:
                rec['median_positive_improvement'] = float(sg_pos.positive_expression_improvement.median())
                rec['positive_threshold'] = float(sg_pos.threshold.iloc[0])
                break

        records.append(rec)
        print(f"  {sg_name:25s}: n={n_genes}, med_abs={med_abs:.4f}, prop_imp={prop_imp_w:.3f}")

    sg_df = pd.DataFrame(records)
    return sg_df


# ============================================================
# 9. Analysis 5: Relationship to mean-shift accuracy
# ============================================================

def analysis5_mean_relationship(w_df, det_results, pos_results):
    """Compare distributional metrics to mean-shift accuracy."""
    print("\n" + "=" * 70)
    print("Analysis 5: Relationship to mean-shift accuracy")
    print("=" * 70)

    ms_deltas = pd.read_csv(MEAN_SHIFT_DELTAS)

    # Merge on gene + cell_type
    merged = w_df.merge(ms_deltas[['gene', 'cell_type', 'empirical_delta', 'predicted_delta']],
                        on=['gene', 'cell_type'], how='inner')
    merged['mean_shift_error'] = np.abs(merged.predicted_delta - merged.empirical_delta)

    assert len(merged) == len(w_df), \
        f"Merge mismatch: {len(merged)} vs {len(w_df)}"

    # Add detection
    if 'primary' in det_results:
        det0 = det_results['primary']
        merged = merged.merge(
            det0[['gene', 'cell_type', 'detection_error_reduction']],
            on=['gene', 'cell_type'], how='left')

    # Add positive expression (first available threshold)
    for k in sorted(pos_results.keys()):
        pdf = pos_results[k]
        elig = pdf[pdf.eligible][['gene', 'cell_type', 'positive_expression_improvement']]
        if len(elig) > 1000:
            merged = merged.merge(elig, on=['gene', 'cell_type'], how='left')
            break

    # Spearman correlations
    rho_w, _ = stats.spearmanr(merged.mean_shift_error, merged.absolute_improvement)
    print(f"  Spearman(mean_shift_error, abs_improvement): {rho_w:.4f}")

    if 'detection_error_reduction' in merged.columns:
        valid = merged.dropna(subset=['detection_error_reduction'])
        rho_d, _ = stats.spearmanr(valid.mean_shift_error, valid.detection_error_reduction)
        print(f"  Spearman(mean_shift_error, det_error_reduction): {rho_d:.4f}")

    if 'positive_expression_improvement' in merged.columns:
        valid = merged.dropna(subset=['positive_expression_improvement'])
        if len(valid) > 100:
            rho_p, _ = stats.spearmanr(valid.mean_shift_error, valid.positive_expression_improvement)
            print(f"  Spearman(mean_shift_error, pos_exp_improvement): {rho_p:.4f}")

    # Case analysis
    low_err = merged.mean_shift_error < merged.mean_shift_error.quantile(0.25)
    high_err = merged.mean_shift_error > merged.mean_shift_error.quantile(0.75)
    dist_imp = merged.absolute_improvement > 0

    case1 = (low_err & dist_imp).sum()
    case2_mask = low_err & ~dist_imp
    case2 = case2_mask.sum()
    case3 = (~low_err & dist_imp).sum()

    print(f"\n  Case analysis (n={len(merged)}):")
    print(f"    Good mean + dist improves: {case1} ({case1/len(merged):.3f})")
    print(f"    Good mean + dist worsens: {case2} ({case2/len(merged):.3f})")
    print(f"    Poor mean + dist improves: {case3} ({case3/len(merged):.3f})")

    return merged


# ============================================================
# 10. Sensitivity: Wasserstein denominator threshold
# ============================================================

def wasserstein_denominator_sensitivity(w_df):
    """Test sensitivity of normalized improvement to denominator threshold."""
    print("\n" + "=" * 70)
    print("Sensitivity: Wasserstein denominator thresholds")
    print("=" * 70)

    thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    records = []
    for t in thresholds:
        elig = w_df[w_df.W_raw >= t]
        norm = 1.0 - elig.W_imputed / elig.W_raw
        records.append({
            'min_W_raw_threshold': t,
            'n_eligible': len(elig),
            'frac_eligible': len(elig) / len(w_df),
            'median_normalized_improvement': float(norm.median()),
            'mean_normalized_improvement': float(norm.mean()),
        })
        print(f"  threshold={t:.3f}: n={len(elig)}, "
              f"median_norm={norm.median():.4f}")

    return pd.DataFrame(records)


# ============================================================
# 11. Main figure
# ============================================================

def create_main_figure(ct_summary, group_df, w_df, merged_df, det_results,
                       pos_results, audit_info):
    """Create 6-panel publication figure."""
    print("\n" + "=" * 70)
    print("Creating main figure")
    print("=" * 70)

    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # Sort cell types by median absolute improvement
    ct_sorted = ct_summary.sort_values('median_absolute_improvement')

    # --- Panel A: Paired W_raw and W_imputed by cell type ---
    ax_a = fig.add_subplot(gs[0, 0])
    y_pos = np.arange(len(ct_sorted))
    colors_raw = ['#2196F3' if s == 'neuronal' else '#4CAF50'
                  for s in ct_sorted.neuronal_status]
    colors_imp = ['#90CAF9' if s == 'neuronal' else '#A5D6A7'
                  for s in ct_sorted.neuronal_status]

    ax_a.barh(y_pos + 0.15, ct_sorted.median_W_raw, 0.3,
              color=colors_raw, label='W(nucleus, cell)', edgecolor='none')
    ax_a.barh(y_pos - 0.15, ct_sorted.median_W_imputed, 0.3,
              color=colors_imp, label='W(imputed, cell)', edgecolor='none')
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(ct_sorted.cell_type, fontsize=8)
    ax_a.set_xlabel('Median Wasserstein-1 distance', fontsize=10)
    ax_a.legend(fontsize=7, loc='lower right')
    ax_a.set_title('(A) Distributional distance to cells', fontsize=11, fontweight='bold')
    ax_a.invert_yaxis()

    # --- Panel B: Absolute improvement by cell type ---
    ax_b = fig.add_subplot(gs[0, 1])
    colors_b = []
    markers_b = []
    for _, r in ct_sorted.iterrows():
        if r.low_sample:
            colors_b.append('gray')
        elif r.neuronal_status == 'neuronal':
            colors_b.append('#2196F3')
        else:
            colors_b.append('#4CAF50')

    ax_b.barh(y_pos, ct_sorted.median_absolute_improvement, 0.6,
              color=colors_b, edgecolor='none')
    ax_b.errorbar(ct_sorted.median_absolute_improvement.values, y_pos,
                  xerr=[ct_sorted.median_absolute_improvement.values - ct_sorted.abs_imp_ci_lo.values,
                        ct_sorted.abs_imp_ci_hi.values - ct_sorted.median_absolute_improvement.values],
                  fmt='none', ecolor='black', elinewidth=0.8, capsize=2)
    ax_b.axvline(0, color='black', linewidth=0.5, linestyle='--')
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(ct_sorted.cell_type, fontsize=8)
    ax_b.set_xlabel('Median absolute Wasserstein improvement', fontsize=10)
    ax_b.set_title('(B) Distributional improvement', fontsize=11, fontweight='bold')
    ax_b.invert_yaxis()

    # Add neuronal/non-neuronal/low-sample legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2196F3', label='Neuronal'),
                       Patch(facecolor='#4CAF50', label='Non-neuronal'),
                       Patch(facecolor='gray', label='Low-sample')]
    ax_b.legend(handles=legend_elements, fontsize=7, loc='lower right')

    # --- Panel C: Detection-rate error by cell type ---
    ax_c = fig.add_subplot(gs[0, 2])
    # Use the best sensitivity threshold where imputed detection is informative
    best_thresh_key = None
    for k in sorted(det_results.keys()):
        df = det_results[k]
        mean_imp = df.imputed_detection_rate.mean()
        if mean_imp < 0.95:  # detection rate not saturated
            best_thresh_key = k
            break
    if best_thresh_key is None:
        best_thresh_key = sorted(det_results.keys())[-1]  # use highest threshold

    det_df = det_results[best_thresh_key]
    thresh_val = det_df.threshold.iloc[0]
    ct_det_summary = []
    for ct in ct_sorted.cell_type:
        ct_d = det_df[det_df.cell_type == ct]
        ct_det_summary.append({
            'cell_type': ct,
            'median_raw_error': ct_d.raw_detection_error.median(),
            'median_imp_error': ct_d.imputed_detection_error.median(),
        })
    ct_det = pd.DataFrame(ct_det_summary)

    y_pos_c = np.arange(len(ct_det))
    ax_c.barh(y_pos_c + 0.15, ct_det.median_raw_error, 0.3,
              color='#EF5350', label='Raw nucleus', edgecolor='none')
    ax_c.barh(y_pos_c - 0.15, ct_det.median_imp_error, 0.3,
              color='#66BB6A', label='Imputed', edgecolor='none')
    ax_c.set_yticks(y_pos_c)
    ax_c.set_yticklabels(ct_det.cell_type, fontsize=8)
    ax_c.set_xlabel('Median detection-rate error', fontsize=10)
    ax_c.set_title(f'(C) Detection error (thresh={thresh_val})', fontsize=11, fontweight='bold')
    ax_c.legend(fontsize=7, loc='lower right')
    ax_c.invert_yaxis()

    # --- Panel D: Positive-expression improvement ---
    ax_d = fig.add_subplot(gs[1, 0])
    best_pos_key = None
    for k in sorted(pos_results.keys()):
        pdf = pos_results[k]
        n_elig = pdf.eligible.sum()
        if n_elig > 5000:
            best_pos_key = k
            break
    if best_pos_key is None and len(pos_results) > 0:
        best_pos_key = sorted(pos_results.keys())[0]

    if best_pos_key:
        pos_df = pos_results[best_pos_key]
        pos_elig = pos_df[pos_df.eligible]
        thresh_p = pos_elig.threshold.iloc[0] if len(pos_elig) > 0 else 0

        ct_pos_summary = []
        for ct in ct_sorted.cell_type:
            ct_p = pos_elig[pos_elig.cell_type == ct]
            if len(ct_p) > 0:
                ct_pos_summary.append({
                    'cell_type': ct,
                    'median_improvement': ct_p.positive_expression_improvement.median(),
                    'n_eligible': len(ct_p),
                })
            else:
                ct_pos_summary.append({
                    'cell_type': ct,
                    'median_improvement': 0,
                    'n_eligible': 0,
                })

        ct_pos = pd.DataFrame(ct_pos_summary)
        y_pos_d = np.arange(len(ct_pos))
        colors_d = ['#2196F3' if ct_status(ct) == 'neuronal' else
                    ('#4CAF50' if ct_status(ct) == 'non-neuronal' else 'gray')
                    for ct in ct_pos.cell_type]
        ax_d.barh(y_pos_d, ct_pos.median_improvement, 0.6,
                  color=colors_d, edgecolor='none')
        ax_d.axvline(0, color='black', linewidth=0.5, linestyle='--')
        ax_d.set_yticks(y_pos_d)
        ax_d.set_yticklabels(ct_pos.cell_type, fontsize=8)
        ax_d.set_xlabel('Median Wasserstein improvement\n(positive expression only)', fontsize=10)
        dense_note = ' [threshold-dependent]' if audit_info['is_dense'] else ''
        ax_d.set_title(f'(D) Positive-expression{dense_note}\n(thresh={thresh_p})',
                       fontsize=11, fontweight='bold')
        ax_d.invert_yaxis()

    # --- Panel E: Group summaries ---
    ax_e = fig.add_subplot(gs[1, 1])
    grp_neuro = group_df[group_df.group == 'neuronal'].iloc[0] if 'neuronal' in group_df.group.values else None
    grp_non = group_df[group_df.group == 'non_neuronal'].iloc[0] if 'non_neuronal' in group_df.group.values else None

    if grp_neuro is not None and grp_non is not None:
        metrics = ['median_absolute_improvement', 'proportion_improved_W']
        metric_labels = ['Median W improvement', 'Proportion improved']

        # Add detection if available
        if 'median_detection_error_reduction' in group_df.columns:
            metrics.append('median_detection_error_reduction')
            metric_labels.append('Detection error\nreduction')
        if 'median_positive_improvement' in group_df.columns:
            metrics.append('median_positive_improvement')
            metric_labels.append('Positive-expr\nimprovement')

        x = np.arange(len(metrics))
        width = 0.35
        neuro_vals = [float(grp_neuro[m]) if pd.notna(grp_neuro.get(m)) else 0 for m in metrics]
        non_vals = [float(grp_non[m]) if pd.notna(grp_non.get(m)) else 0 for m in metrics]

        ax_e.bar(x - width/2, neuro_vals, width, label='Neuronal', color='#2196F3')
        ax_e.bar(x + width/2, non_vals, width, label='Non-neuronal', color='#4CAF50')
        ax_e.set_xticks(x)
        ax_e.set_xticklabels(metric_labels, fontsize=8, rotation=15, ha='right')
        ax_e.legend(fontsize=8)
        ax_e.set_title('(E) Neuronal vs non-neuronal', fontsize=11, fontweight='bold')
        ax_e.axhline(0, color='black', linewidth=0.5)

    # --- Panel F: Mean-shift error vs Wasserstein improvement ---
    ax_f = fig.add_subplot(gs[1, 2])
    if merged_df is not None and len(merged_df) > 0:
        rho, _ = stats.spearmanr(merged_df.mean_shift_error, merged_df.absolute_improvement)

        # Hexbin for density
        hb = ax_f.hexbin(merged_df.mean_shift_error, merged_df.absolute_improvement,
                         gridsize=40, cmap='Blues', mincnt=1)
        ax_f.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax_f.set_xlabel('Absolute mean-shift error', fontsize=10)
        ax_f.set_ylabel('Wasserstein absolute improvement', fontsize=10)
        ax_f.set_title(f'(F) Mean accuracy vs distribution\n(Spearman ρ = {rho:.3f})',
                       fontsize=11, fontweight='bold')
        plt.colorbar(hb, ax=ax_f, label='Count')

    for fmt in ['png', 'pdf']:
        path = os.path.join(OUT_DIR, f'syngo_distribution_validation.{fmt}')
        fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved main figure")


# ============================================================
# 12. Example gene figure
# ============================================================

def create_example_figure(gene_names, syngo_genes, subgroup_genes,
                          valid_cell_X, valid_cell_ct,
                          valid_nuc_X, valid_nuc_ct,
                          imputed_X, imputed_ct):
    """Create supplementary figure with example gene distributions."""
    print("\n" + "=" * 70)
    print("Creating example gene figure")
    print("=" * 70)

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    # Verify example genes exist and find substitutes if needed
    examples = {}
    for gene, subgroup in EXAMPLE_GENES.items():
        if gene in gene_to_idx and gene in syngo_genes:
            examples[gene] = subgroup
            print(f"  {gene} ({subgroup}): found")
        else:
            print(f"  {gene} not found, finding substitute from {subgroup}...")
            sg_col = f'sg_{subgroup}'
            meta = pd.read_csv(MEAN_SHIFT_METADATA)
            candidates = meta[(meta[sg_col] == True) & (meta.in_model) & (~meta.is_ribosomal)]
            for _, r in candidates.iterrows():
                m = r.mouse_symbol
                if m in gene_to_idx and m in syngo_genes:
                    examples[m] = subgroup
                    print(f"    Substitute: {m}")
                    break

    # Show in one neuronal (L5 IT, high n) and one non-neuronal (Oligo)
    show_cts = ['L5 IT', 'Oligo']

    n_genes = len(examples)
    n_cts = len(show_cts)
    fig, axes = plt.subplots(n_genes, n_cts, figsize=(5 * n_cts, 4 * n_genes))
    if n_genes == 1:
        axes = axes.reshape(1, -1)

    for gi, (gene, subgroup) in enumerate(examples.items()):
        gidx = gene_to_idx[gene]
        for ci, ct in enumerate(show_cts):
            ax = axes[gi, ci]

            cell_mask = valid_cell_ct == ct
            nuc_mask = valid_nuc_ct == ct
            imp_mask = imputed_ct == ct

            cell_vals = valid_cell_X[cell_mask, gidx]
            nuc_vals = valid_nuc_X[nuc_mask, gidx]
            imp_vals = imputed_X[imp_mask, gidx]

            # ECDF plot with zero-mass annotation
            def plot_ecdf(vals, label, color, ax):
                sorted_vals = np.sort(vals)
                ecdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
                ax.step(sorted_vals, ecdf, where='post', label=label,
                        color=color, linewidth=1.5)
                zero_frac = (vals == 0).mean()
                return zero_frac

            zf_cell = plot_ecdf(cell_vals, f'Cell (n={len(cell_vals)})', '#2196F3', ax)
            zf_nuc = plot_ecdf(nuc_vals, f'Nucleus (n={len(nuc_vals)})', '#FF9800', ax)
            zf_imp = plot_ecdf(imp_vals, f'Imputed (n={len(imp_vals)})', '#4CAF50', ax)

            # Mark zero mass
            zero_text = f'Zeros: cell={zf_cell:.2f}, nuc={zf_nuc:.2f}, imp={zf_imp:.2f}'
            ax.text(0.98, 0.05, zero_text, transform=ax.transAxes,
                    fontsize=7, ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

            # Wasserstein distances
            w_raw = wasserstein_distance(nuc_vals, cell_vals)
            w_imp = wasserstein_distance(imp_vals, cell_vals)
            w_text = f'W(nuc,cell)={w_raw:.3f}\nW(imp,cell)={w_imp:.3f}'
            ax.text(0.98, 0.95, w_text, transform=ax.transAxes,
                    fontsize=8, ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

            ax.set_xlabel('Expression (log1p)', fontsize=9)
            ax.set_ylabel('ECDF', fontsize=9)
            ax.set_title(f'{gene} ({subgroup}) — {ct}', fontsize=10, fontweight='bold')
            ax.legend(fontsize=7, loc='center right')

    fig.suptitle('Example SynGO gene distributions', fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()

    for fmt in ['png', 'pdf']:
        path = os.path.join(OUT_DIR, f'syngo_distribution_examples.{fmt}')
        fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved example figure with {len(examples)} genes x {len(show_cts)} CTs")


# ============================================================
# 13. Write README
# ============================================================

def write_readme(audit_info, ct_summary, group_df, sg_df, denom_sens):
    """Write DISTRIBUTION_README.md."""
    print("\n" + "=" * 70)
    print("Writing README")
    print("=" * 70)

    is_dense = audit_info['is_dense']

    lines = [
        "# SynGO Distribution-Level Biological Validation",
        "",
        "## Question",
        "",
        "For non-ribosomal genes encoding synaptic proteins, does cross-decoding",
        "make the expression distribution of validation nuclei more similar to the",
        "observed whole-cell distribution? Does the degree of distributional recovery",
        "differ across neuronal and non-neuronal cell types?",
        "",
        "## Relationship to Mean-Shift Analysis",
        "",
        "This analysis uses exactly the same 1,275 non-ribosomal SynGO genes,",
        "validation observations, gene ordering, cell-type definitions, and subgroups",
        "as the completed mean-shift analysis (syngo_brain_validation.py). Only the",
        "evaluation metrics differ.",
        "",
        "## Inference Audit",
        "",
        f"- **Encoding:** Posterior mean (mu_latent). No reparameterization sampling.",
        f"- **Decoding:** Deterministic conditional mean. Final activation: LeakyReLU(0.1).",
        f"- **Draws per nucleus:** 1 (single forward pass).",
        f"- **Clipping/thresholding:** None applied to saved predictions.",
        f"- **Scale:** log1p (same as input).",
        f"- **Dense output:** {'YES' if is_dense else 'NO'} "
        f"({audit_info['exact_zero_fraction']:.6f} exact zeros).",
        f"- **Negative values:** {audit_info['negative_values']} "
        f"({audit_info['negative_values'] / (audit_info['n_observations'] * audit_info['n_genes']):.6f}).",
        "",
        "### Implications for Distribution Recovery",
        "",
        "Deterministic mean decoding produces a single point estimate per nucleus,",
        "not a sample from the predictive distribution. The resulting imputed",
        "distribution is expected to be narrower than the real whole-cell distribution",
        "because:",
        "",
        "1. Per-cell biological variation is not sampled.",
        "2. The mean decoder output smooths over gene-specific noise.",
        "3. Nearly all imputed values are nonzero, even for genes with high dropout",
        "   in real cells.",
        "",
        "This means detection-rate analysis at threshold=0 is uninformative for the",
        "imputed matrix. Sensitivity analyses at higher thresholds are provided.",
        "",
        "## Distributional Metrics",
        "",
        "### Wasserstein-1 Distance",
        "",
        "For each SynGO gene g and cell type c:",
        "- W_raw[g,c] = Wasserstein-1(nucleus[g,c], cell[g,c])",
        "- W_imputed[g,c] = Wasserstein-1(imputed[g,c], cell[g,c])",
        "- absolute_improvement[g,c] = W_raw - W_imputed",
        "",
        "Positive = imputation moved distribution closer to cells.",
        "",
        "### Normalized Improvement",
        "",
        f"normalized_improvement = 1 - W_imputed / W_raw",
        f"",
        f"Minimum denominator: W_raw >= {MIN_W_RAW_DENOM}.",
        "",
        "### Detection-Rate Recovery",
        "",
        "detection_error_reduction = |nucleus_rate - cell_rate| - |imputed_rate - cell_rate|",
        "",
        f"Prespecified thresholds (log1p scale): {DETECTION_THRESHOLDS}",
        "",
        "### Positive-Expression Wasserstein",
        "",
        "Wasserstein-1 computed on observations exceeding the detection threshold.",
        f"Minimum {MIN_POSITIVE_OBS} positive observations required per distribution.",
        "",
    ]

    if is_dense:
        lines.extend([
            "**NOTE:** Because the decoder output is dense, positive-expression",
            "analysis uses threshold-dependent definitions. Results are labeled",
            "accordingly and should be interpreted as sensitivity analyses.",
            "",
        ])

    # Summary table
    lines.extend([
        "## Key Results",
        "",
    ])

    for _, r in group_df.iterrows():
        lines.append(f"### {r['group'].replace('_', ' ').title()}")
        lines.append(f"- Gene-CT pairs: {r['n_gene_ct_pairs']}")
        lines.append(f"- Median W_raw: {r['median_W_raw']:.4f}")
        lines.append(f"- Median W_imputed: {r['median_W_imputed']:.4f}")
        lines.append(f"- Median absolute improvement: {r['median_absolute_improvement']:.4f}")
        lines.append(f"- Proportion improved: {r['proportion_improved_W']:.3f}")
        lines.append("")

    lines.extend([
        "## Denominator Sensitivity",
        "",
    ])
    for _, r in denom_sens.iterrows():
        lines.append(f"- W_raw >= {r['min_W_raw_threshold']:.3f}: "
                     f"n={r['n_eligible']}, "
                     f"median_norm={r['median_normalized_improvement']:.4f}")
    lines.append("")

    lines.extend([
        "## Interpretation",
        "",
        "Cells and nuclei are not physically paired. These are marginal",
        "population-distribution comparisons within cell type.",
        "",
        "Deterministic mean decoding produces distributions that are narrower",
        "than real whole-cell distributions. Distribution recovery should be",
        "interpreted as a lower bound on what sampling-based decoding might achieve.",
        "",
        "## Output Files",
        "",
        "- `syngo_distribution_gene_celltype.csv` - All gene x CT distributional metrics",
        "- `syngo_distribution_by_celltype.csv` - Per-CT summaries with bootstrap CIs",
        "- `syngo_distribution_by_group.csv` - Group-level summaries",
        "- `syngo_distribution_by_subgroup.csv` - Subgroup summaries",
        "- `syngo_distribution_mean_relationship.csv` - Mean-shift vs distribution",
        "- `syngo_distribution_validation.png/pdf` - 6-panel main figure",
        "- `syngo_distribution_examples.png/pdf` - Example gene ECDFs",
        "- `DISTRIBUTION_README.md` - This file",
        "",
    ])

    path = os.path.join(OUT_DIR, "DISTRIBUTION_README.md")
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Saved {path}")


# ============================================================
# 14. Verify results
# ============================================================

def verify_results(w_df, ct_summary, group_df):
    """Independent verification of exported results."""
    print("\n" + "=" * 70)
    print("Verification")
    print("=" * 70)

    # 1. Check gene count
    n_genes = w_df.gene.nunique()
    assert n_genes == 1275, f"Expected 1275 genes, got {n_genes}"
    print(f"  Gene count: {n_genes} [OK]")

    # 2. Check CT count
    n_cts = w_df.cell_type.nunique()
    assert n_cts == 17, f"Expected 17 CTs, got {n_cts}"
    print(f"  Cell type count: {n_cts} [OK]")

    # 3. Verify CT summary matches gene-level data
    for _, row in ct_summary.iterrows():
        ct = row['cell_type']
        ct_data = w_df[w_df.cell_type == ct]
        recomputed_median = ct_data.absolute_improvement.median()
        diff = abs(recomputed_median - row['median_absolute_improvement'])
        assert diff < 1e-10, f"Median mismatch for {ct}: {diff}"
    print(f"  CT summary matches gene-level data [OK]")

    # 4. Verify group totals
    total_pairs = group_df[group_df.group == 'all']['n_gene_ct_pairs'].values[0]
    assert total_pairs == len(w_df), f"Group total {total_pairs} != {len(w_df)}"
    print(f"  Group totals consistent [OK]")

    # 5. Cross-check with mean-shift gene set
    ms_deltas = pd.read_csv(MEAN_SHIFT_DELTAS)
    ms_genes = sorted(ms_deltas.gene.unique())
    w_genes = sorted(w_df.gene.unique())
    assert ms_genes == w_genes, "Gene set mismatch with mean-shift analysis"
    print(f"  Gene set matches mean-shift analysis [OK]")

    print("\n  All verifications passed.")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("SynGO Distribution-Level Validation")
    print("=" * 70)
    print(f"  Output: {OUT_DIR}")
    print(f"  Seed: {SEED}")
    print(f"  Bootstrap: {N_BOOT}")

    # Load data
    (gene_names, gene_list,
     valid_cell_X, valid_cell_ct,
     valid_nuc_X, valid_nuc_ct,
     imputed_X, imputed_ct) = load_data()

    # Build and verify gene sets
    syngo_genes, subgroup_genes, meta = build_syngo_sets(gene_names, gene_list)

    # Inference audit
    audit_info = inference_audit(imputed_X)

    # Analysis 1: Wasserstein
    w_df = analysis1_wasserstein(gene_names, syngo_genes,
                                 valid_cell_X, valid_cell_ct,
                                 valid_nuc_X, valid_nuc_ct,
                                 imputed_X, imputed_ct)

    # Analysis 2: Detection rate
    det_results = analysis2_detection(gene_names, syngo_genes,
                                      valid_cell_X, valid_cell_ct,
                                      valid_nuc_X, valid_nuc_ct,
                                      imputed_X, imputed_ct,
                                      audit_info)

    # Analysis 3: Positive expression
    pos_results = analysis3_positive_expression(gene_names, syngo_genes,
                                                valid_cell_X, valid_cell_ct,
                                                valid_nuc_X, valid_nuc_ct,
                                                imputed_X, imputed_ct,
                                                audit_info)

    # Summaries
    ct_summary = summarize_by_celltype(w_df, det_results, pos_results)
    group_df = summarize_by_group(w_df, det_results, pos_results)

    # Analysis 4: Subgroups
    sg_df = analysis4_subgroups(subgroup_genes, w_df, det_results, pos_results)

    # Analysis 5: Relationship to mean shift
    merged_df = analysis5_mean_relationship(w_df, det_results, pos_results)

    # Denominator sensitivity
    denom_sens = wasserstein_denominator_sensitivity(w_df)

    # Save CSVs
    print("\n" + "=" * 70)
    print("Saving outputs")
    print("=" * 70)

    w_df.to_csv(os.path.join(OUT_DIR, "syngo_distribution_gene_celltype.csv"), index=False)
    ct_summary.to_csv(os.path.join(OUT_DIR, "syngo_distribution_by_celltype.csv"), index=False)
    group_df.to_csv(os.path.join(OUT_DIR, "syngo_distribution_by_group.csv"), index=False)
    sg_df.to_csv(os.path.join(OUT_DIR, "syngo_distribution_by_subgroup.csv"), index=False)
    merged_df.to_csv(os.path.join(OUT_DIR, "syngo_distribution_mean_relationship.csv"), index=False)
    print("  CSVs saved")

    # Figures
    create_main_figure(ct_summary, group_df, w_df, merged_df, det_results,
                       pos_results, audit_info)
    create_example_figure(gene_names, syngo_genes, subgroup_genes,
                          valid_cell_X, valid_cell_ct,
                          valid_nuc_X, valid_nuc_ct,
                          imputed_X, imputed_ct)

    # README
    write_readme(audit_info, ct_summary, group_df, sg_df, denom_sens)

    # Verify
    verify_results(w_df, ct_summary, group_df)

    # Terminal summary
    print("\n" + "=" * 70)
    print("TERMINAL SUMMARY")
    print("=" * 70)

    all_grp = group_df[group_df.group == 'all'].iloc[0]
    neuro_grp = group_df[group_df.group == 'neuronal'].iloc[0]
    non_grp = group_df[group_df.group == 'non_neuronal'].iloc[0]

    print(f"\n  SynGO genes: {w_df.gene.nunique()} non-ribosomal")
    print(f"  Gene-CT pairs: {len(w_df)}")
    print(f"\n  Overall Wasserstein:")
    print(f"    Median W_raw: {all_grp['median_W_raw']:.4f}")
    print(f"    Median W_imputed: {all_grp['median_W_imputed']:.4f}")
    print(f"    Median absolute improvement: {all_grp['median_absolute_improvement']:.4f}")
    print(f"    Proportion improved: {all_grp['proportion_improved_W']:.3f}")
    print(f"\n  Neuronal vs non-neuronal:")
    print(f"    Neuronal median abs improvement: {neuro_grp['median_absolute_improvement']:.4f}")
    print(f"    Non-neuronal median abs improvement: {non_grp['median_absolute_improvement']:.4f}")

    print(f"\n  Outputs: {OUT_DIR}/")
    for f in sorted(os.listdir(OUT_DIR)):
        if f.startswith('syngo_distribution') or f == 'DISTRIBUTION_README.md':
            print(f"    {f}")

    print("\n" + "=" * 70)
    print("Analysis complete.")
    print("=" * 70)


if __name__ == '__main__':
    main()
