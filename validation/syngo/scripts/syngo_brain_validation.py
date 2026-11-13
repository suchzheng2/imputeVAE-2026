#!/usr/bin/env python
"""
SynGO brain-specific biological consistency analysis for ImputeVAE.

Question: Do non-ribosomal genes encoding synaptic proteins exhibit larger
nucleus-to-cell expression shifts in neuronal than in non-neuronal populations,
and does ImputeVAE reproduce the magnitude and cell-type dependence of those shifts?

SynGO annotates protein function/localization, not RNA localization. This analysis
tests brain-specific biological consistency, not molecular validation of RNA
localization.

Usage:
    python -u validation/syngo/scripts/syngo_brain_validation.py

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
from sklearn.metrics import roc_auc_score
import anndata as ad

warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

SEED = 42
rng = np.random.RandomState(SEED)

# ============================================================
# Paths
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

OUT_DIR = os.path.join(PROJECT, "validation/syngo/syngo_brain_validation")

NEURONAL = ['L2/3 IT', 'L5 ET', 'L5 IT', 'L5/6 NP', 'L6 CT', 'L6 IT', 'L6b',
            'Lamp5', 'Pvalb', 'Sncg', 'Sst', 'Vip']
NON_NEURONAL = ['Astro', 'Endo', 'Macrophage', 'OPC', 'Oligo']

SUBGROUPS = {
    'broad_synapse':          {'go_ids': ['GO:0045202']},
    'presynaptic':            {'go_ids': ['GO:0098793']},
    'postsynaptic':           {'go_ids': ['GO:0098794']},
    'postsynaptic_density':   {'go_ids': ['GO:0014069']},
    'presynaptic_active_zone':{'go_ids': ['GO:0048786']},
    'synaptic_vesicle':       {'go_ids': ['GO:0008021']},
}

N_BOOT = 1000


# ============================================================
# 1. Load data
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
    print(f"  Common cell types: {len(common_ct)}")

    # --- Validation indices ---
    def get_indices(modality):
        mod = manifest[manifest.modality == modality].sort_values('obs_id').reset_index(drop=True)
        valid_mask = mod.partition == 'valid'
        return np.where(valid_mask)[0], mod[valid_mask].reset_index(drop=True)

    valid_nuc_idx, valid_nuc_meta = get_indices('nuclei')
    valid_cell_idx, valid_cell_meta = get_indices('cell')
    print(f"  Valid nuclei: {len(valid_nuc_idx)}, Valid cells: {len(valid_cell_idx)}")

    # --- Load raw h5ad and extract validation subsets ---
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

    # Free full matrices
    del cell_adata, nuc_adata

    # --- Load imputed ---
    print(f"  Loading imputed: {IMPUTED_NPY}")
    imputed_X = np.load(IMPUTED_NPY)
    imputed_obs = pd.read_csv(IMPUTED_OBS)
    imputed_ct = imputed_obs['cell_type'].values

    print(f"    Imputed shape: {imputed_X.shape}")
    assert imputed_X.shape[0] == valid_nuc_X.shape[0], \
        f"Row mismatch: imputed {imputed_X.shape[0]} vs nuc {valid_nuc_X.shape[0]}"
    assert imputed_X.shape[1] == len(gene_names), \
        f"Col mismatch: imputed {imputed_X.shape[1]} vs genes {len(gene_names)}"

    ct_match = (imputed_ct == valid_nuc_ct).mean()
    print(f"    CT alignment: {ct_match*100:.1f}%")
    assert ct_match > 0.99, f"CT alignment only {ct_match*100:.1f}%"

    # Normalization check
    print(f"    Cell range: [{valid_cell_X.min():.2f}, {valid_cell_X.max():.2f}]")
    print(f"    Nuc range: [{valid_nuc_X.min():.2f}, {valid_nuc_X.max():.2f}]")
    print(f"    Imputed range: [{imputed_X.min():.2f}, {imputed_X.max():.2f}]")

    return (gene_names, gene_list,
            valid_cell_X, valid_cell_ct, valid_cell_meta,
            valid_nuc_X, valid_nuc_ct, valid_nuc_meta,
            imputed_X, imputed_ct)


# ============================================================
# 2. Build SynGO gene sets
# ============================================================

def build_syngo_sets(gene_names, gene_list):
    """Build SynGO gene sets from ontologies.xlsx via ortholog mapping."""
    print("\n" + "=" * 70)
    print("Building SynGO gene sets")
    print("=" * 70)

    # Load SynGO
    zf = zipfile.ZipFile(SYNGO_ZIP)
    genes_df = pd.read_excel(io.BytesIO(zf.read('genes.xlsx')))
    ontologies = pd.read_excel(io.BytesIO(zf.read('ontologies.xlsx')))

    # Parse term -> genes from ontologies.xlsx
    term_genes = {}
    for _, row in ontologies.iterrows():
        val = str(row.get('hgnc_symbol', ''))
        if val and val != 'nan':
            term_genes[row['id']] = {g.strip() for g in val.split(',') if g.strip()}
        else:
            term_genes[row['id']] = set()

    # CC terms only
    cc_terms = set(ontologies[ontologies['domain'] == 'CC']['id'])
    print(f"  SynGO: {len(genes_df)} genes, {len(ontologies)} terms ({len(cc_terms)} CC)")

    # Ortholog mapping
    ortho = pd.read_csv(ORTHOLOG_PATH)
    human_counts = ortho.groupby('human_sym').size()
    one_to_one = set(human_counts[human_counts == 1].index)
    ortho_1to1 = ortho[ortho.human_sym.isin(one_to_one)]
    h2m = dict(zip(ortho_1to1.human_sym, ortho_1to1.mouse_sym))
    print(f"  Orthologs: {len(ortho)} total, {len(h2m)} one-to-one human->mouse")

    # Also capitalization fallback
    gene_set_model = set(gene_names)
    def map_human_to_mouse(human_sym):
        m = h2m.get(human_sym)
        if m: return m, 'ortholog'
        cap = human_sym[0] + human_sym[1:].lower() if human_sym else None
        if cap and cap in gene_set_model:
            return cap, 'capitalization'
        return None, 'unmapped'

    # Build broad synapse set (GO:0045202 includes descendants in ontologies.xlsx)
    broad_human = term_genes.get('GO:0045202', set())
    print(f"  Broad synapse (GO:0045202): {len(broad_human)} human genes")

    # Map and filter
    records = []
    for h in sorted(broad_human):
        mouse, method = map_human_to_mouse(h)
        in_model = mouse in gene_set_model if mouse else False
        is_ribo = (mouse or '').startswith('Rpl') or (mouse or '').startswith('Rps')

        # Subgroup memberships
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

    # Filtering flow
    flow = [
        ('human_syngo_CC_synapse', len(broad_human)),
        ('mapped_to_mouse', (meta.mapping_method != 'unmapped').sum()),
        ('mapped_via_ortholog', (meta.mapping_method == 'ortholog').sum()),
        ('mapped_via_capitalization', (meta.mapping_method == 'capitalization').sum()),
        ('in_model_10134', meta.in_model.sum()),
        ('ribosomal_Rpl_Rps', (meta.in_model & meta.is_ribosomal).sum()),
        ('non_ribosomal_primary', (meta.in_model & ~meta.is_ribosomal).sum()),
    ]
    flow_df = pd.DataFrame(flow, columns=['stage', 'count'])

    print("  Filtering flow:")
    for _, r in flow_df.iterrows():
        print(f"    {r['stage']}: {r['count']}")

    # Primary set
    primary = meta[(meta.in_model) & (~meta.is_ribosomal)].copy()
    primary_genes = set(primary.mouse_symbol)
    print(f"  Primary analysis set: {len(primary_genes)} non-ribosomal SynGO genes")

    # Subgroup gene lists
    subgroup_genes = {}
    for sg_name in SUBGROUPS:
        sg_col = f'sg_{sg_name}'
        sg_set = set(primary[primary[sg_col]].mouse_symbol)
        subgroup_genes[sg_name] = sg_set

    # Print subgroup counts
    print("  Subgroup gene counts:")
    for sg_name, sg_set in subgroup_genes.items():
        print(f"    {sg_name}: {len(sg_set)}")

    # Verify gene order
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    for g in primary_genes:
        assert g in gene_to_idx, f"SynGO gene {g} not found in gene_names"

    return primary_genes, subgroup_genes, meta, flow_df


# ============================================================
# 3. Compute deltas
# ============================================================

def compute_deltas(gene_names, syngo_genes,
                   valid_cell_X, valid_cell_ct,
                   valid_nuc_X, valid_nuc_ct,
                   imputed_X, imputed_ct):
    """Compute empirical and predicted deltas for SynGO genes."""
    print("\n" + "=" * 70)
    print("Computing deltas")
    print("=" * 70)

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    syngo_idx = sorted([gene_to_idx[g] for g in syngo_genes])
    syngo_names = [gene_names[i] for i in syngo_idx]

    cts_cell = set(valid_cell_ct)
    cts_nuc = set(valid_nuc_ct)
    common_cts = sorted(cts_cell & cts_nuc)

    print(f"  SynGO genes: {len(syngo_idx)}")
    print(f"  Common cell types: {len(common_cts)}")

    # Sample counts
    print("\n  Validation sample counts:")
    rare_cts = []
    for ct in common_cts:
        nc = (valid_cell_ct == ct).sum()
        nn = (valid_nuc_ct == ct).sum()
        flag = ' *' if nc < 20 or nn < 20 else ''
        if flag: rare_cts.append(ct)
        ns = ct_status(ct)
        print(f"    {ct:15s} ({ns:12s}): {nc:5d} cells, {nn:5d} nuclei{flag}")

    records = []
    for ct in common_cts:
        cell_mask = valid_cell_ct == ct
        nuc_mask = valid_nuc_ct == ct
        imp_mask = imputed_ct == ct

        cell_mean = valid_cell_X[cell_mask][:, syngo_idx].mean(axis=0)
        nuc_mean = valid_nuc_X[nuc_mask][:, syngo_idx].mean(axis=0)
        imp_mean = imputed_X[imp_mask][:, syngo_idx].mean(axis=0)

        for j, gi in enumerate(syngo_idx):
            g = gene_names[gi]
            records.append({
                'gene': g,
                'cell_type': ct,
                'neuronal_status': ct_status(ct),
                'empirical_delta': float(cell_mean[j] - nuc_mean[j]),
                'predicted_delta': float(imp_mean[j] - nuc_mean[j]),
                'cell_mean': float(cell_mean[j]),
                'nucleus_mean': float(nuc_mean[j]),
                'imputed_mean': float(imp_mean[j]),
                'n_cell': int(cell_mask.sum()),
                'n_nucleus': int(nuc_mask.sum()),
            })

    delta_df = pd.DataFrame(records)
    print(f"  Total gene-CT observations: {len(delta_df)}")

    return delta_df, common_cts, rare_cts


def ct_status(ct):
    if ct in NEURONAL: return 'neuronal'
    if ct in NON_NEURONAL: return 'non-neuronal'
    return 'unknown'


# ============================================================
# 4. Analysis 1: Overall imputation quality
# ============================================================

def analysis1_quality(delta_df, common_cts):
    """Gene-CT-level imputation quality metrics."""
    print("\n" + "=" * 70)
    print("Analysis 1: SynGO-gene imputation quality")
    print("=" * 70)

    de = delta_df.empirical_delta.values
    dp = delta_df.predicted_delta.values

    overall = compute_quality(de, dp, 'all')
    overall['n_genes'] = delta_df.gene.nunique()
    overall['n_cell_types'] = delta_df.cell_type.nunique()
    overall['n_observations'] = len(delta_df)

    print(f"  n_genes={overall['n_genes']}, n_CTs={overall['n_cell_types']}, "
          f"n_obs={overall['n_observations']}")
    print(f"  Pearson={overall['pearson_r']:.4f}, Spearman={overall['spearman_r']:.4f}")
    print(f"  R2={overall['R2']:.4f}, MAE={overall['MAE']:.4f}, RMSE={overall['RMSE']:.4f}")
    print(f"  Sign agreement={overall['sign_agreement']:.4f}")
    print(f"  Slope={overall['slope']:.4f}, Intercept={overall['intercept']:.4f}")

    # Sign breakdown
    pp = ((de > 0) & (dp > 0)).sum()
    nn = ((de < 0) & (dp < 0)).sum()
    pn = ((de > 0) & (dp < 0)).sum()
    np_ = ((de < 0) & (dp > 0)).sum()
    total = len(de)
    print(f"\n  Sign breakdown (n={total}):")
    print(f"    emp+/pred+: {pp} ({pp/total:.3f})")
    print(f"    emp-/pred-: {nn} ({nn/total:.3f})")
    print(f"    emp+/pred-: {pn} ({pn/total:.3f})")
    print(f"    emp-/pred+: {np_} ({np_/total:.3f})")

    overall['sign_pp'] = pp
    overall['sign_nn'] = nn
    overall['sign_pn'] = pn
    overall['sign_np'] = np_

    # Per cell type
    per_ct = []
    for ct in common_cts:
        ct_data = delta_df[delta_df.cell_type == ct]
        m = compute_quality(ct_data.empirical_delta.values,
                            ct_data.predicted_delta.values, ct)
        m['cell_type'] = ct
        m['neuronal_status'] = ct_status(ct)
        m['n_genes'] = ct_data.gene.nunique()
        per_ct.append(m)

    per_ct_df = pd.DataFrame(per_ct)
    overall_df = pd.DataFrame([overall])

    print("\n  Per-cell-type Pearson r:")
    for _, r in per_ct_df.iterrows():
        print(f"    {r['cell_type']:15s}: r={r['pearson_r']:.3f}, R2={r['R2']:.3f}, "
              f"MAE={r['MAE']:.3f}")

    return overall_df, per_ct_df


def compute_quality(de, dp, label):
    """Compute quality metrics between empirical and predicted deltas."""
    valid = ~(np.isnan(de) | np.isnan(dp))
    de, dp = de[valid], dp[valid]

    pr, _ = stats.pearsonr(de, dp)
    sr, _ = stats.spearmanr(de, dp)
    ss_res = np.sum((dp - de) ** 2)
    ss_tot = np.sum((de - de.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    mae = np.mean(np.abs(dp - de))
    rmse = np.sqrt(np.mean((dp - de) ** 2))
    sign = np.mean(np.sign(de) == np.sign(dp))

    slope, intercept, _, _, _ = stats.linregress(de, dp)

    return {
        'label': label,
        'pearson_r': float(pr),
        'spearman_r': float(sr),
        'R2': float(r2),
        'MAE': float(mae),
        'RMSE': float(rmse),
        'sign_agreement': float(sign),
        'slope': float(slope),
        'intercept': float(intercept),
        'n': len(de),
    }


# ============================================================
# 5. Analysis 2: Cell-type differences
# ============================================================

def analysis2_celltype(delta_df, common_cts, rare_cts,
                       valid_cell_X, valid_cell_ct,
                       valid_nuc_X, valid_nuc_ct,
                       imputed_X, imputed_ct,
                       gene_names, syngo_genes):
    """Cell-type mean SynGO shifts with bootstrap CIs."""
    print("\n" + "=" * 70)
    print("Analysis 2: Cell-type differences")
    print("=" * 70)

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    syngo_idx = np.array(sorted([gene_to_idx[g] for g in syngo_genes]))

    records = []
    for ct in common_cts:
        cell_mask = valid_cell_ct == ct
        nuc_mask = valid_nuc_ct == ct
        imp_mask = imputed_ct == ct

        cell_X_ct = valid_cell_X[cell_mask][:, syngo_idx]
        nuc_X_ct = valid_nuc_X[nuc_mask][:, syngo_idx]
        imp_X_ct = imputed_X[imp_mask][:, syngo_idx]

        nc = cell_mask.sum()
        nn = nuc_mask.sum()

        # Point estimate: mean across genes of (mean_cell - mean_nuc)
        emp_shift = (cell_X_ct.mean(axis=0) - nuc_X_ct.mean(axis=0)).mean()
        pred_shift = (imp_X_ct.mean(axis=0) - nuc_X_ct.mean(axis=0)).mean()

        # Bootstrap: resample observations within modality
        boot_emp = np.empty(N_BOOT)
        boot_pred = np.empty(N_BOOT)
        for b in range(N_BOOT):
            ci = rng.choice(nc, nc, replace=True)
            ni = rng.choice(nn, nn, replace=True)
            c_mean = cell_X_ct[ci].mean(axis=0)
            n_mean = nuc_X_ct[ni].mean(axis=0)
            i_mean = imp_X_ct[ni].mean(axis=0)  # paired with nuclei
            boot_emp[b] = (c_mean - n_mean).mean()
            boot_pred[b] = (i_mean - n_mean).mean()

        records.append({
            'cell_type': ct,
            'neuronal_status': ct_status(ct),
            'n_cell': nc,
            'n_nucleus': nn,
            'low_sample': ct in rare_cts,
            'empirical_mean_shift': float(emp_shift),
            'predicted_mean_shift': float(pred_shift),
            'prediction_error': float(pred_shift - emp_shift),
            'emp_ci_lo': float(np.percentile(boot_emp, 2.5)),
            'emp_ci_hi': float(np.percentile(boot_emp, 97.5)),
            'pred_ci_lo': float(np.percentile(boot_pred, 2.5)),
            'pred_ci_hi': float(np.percentile(boot_pred, 97.5)),
        })

    ct_df = pd.DataFrame(records)

    # Cross-cell-type metrics
    emp = ct_df.empirical_mean_shift.values
    pred = ct_df.predicted_mean_shift.values
    pr, _ = stats.pearsonr(emp, pred)
    sr, _ = stats.spearmanr(emp, pred)
    ss_res = np.sum((pred - emp) ** 2)
    ss_tot = np.sum((emp - emp.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    mae = np.mean(np.abs(pred - emp))
    slope, intercept, _, _, _ = stats.linregress(emp, pred)

    print(f"  Cross-CT (all {len(common_cts)}):")
    print(f"    Pearson={pr:.4f}, Spearman={sr:.4f}, R2={r2:.4f}")
    print(f"    MAE={mae:.4f}, slope={slope:.3f}, intercept={intercept:.4f}")

    ct_df.attrs['cross_ct_all'] = {
        'pearson_r': pr, 'spearman_r': sr, 'R2': r2,
        'MAE': mae, 'slope': slope, 'intercept': intercept, 'n': len(common_cts),
    }

    # Excluding rare
    robust = ct_df[~ct_df.low_sample]
    if len(robust) >= 3:
        e2 = robust.empirical_mean_shift.values
        p2 = robust.predicted_mean_shift.values
        pr2, _ = stats.pearsonr(e2, p2)
        sr2, _ = stats.spearmanr(e2, p2)
        ss2 = np.sum((p2 - e2) ** 2)
        st2 = np.sum((e2 - e2.mean()) ** 2)
        r2_2 = 1 - ss2 / st2 if st2 > 0 else np.nan
        mae2 = np.mean(np.abs(p2 - e2))
        sl2, int2, _, _, _ = stats.linregress(e2, p2)
        print(f"  Cross-CT (excl rare, n={len(robust)}):")
        print(f"    Pearson={pr2:.4f}, Spearman={sr2:.4f}, R2={r2_2:.4f}")
        ct_df.attrs['cross_ct_robust'] = {
            'pearson_r': pr2, 'spearman_r': sr2, 'R2': r2_2,
            'MAE': mae2, 'slope': sl2, 'intercept': int2, 'n': len(robust),
        }

    return ct_df


# ============================================================
# 6. Analysis 3: Neuronal vs non-neuronal
# ============================================================

def analysis3_neuro(ct_df, delta_df):
    """Neuronal vs non-neuronal difference with bootstrap CIs."""
    print("\n" + "=" * 70)
    print("Analysis 3: Neuronal vs non-neuronal")
    print("=" * 70)

    neuro = ct_df[ct_df.neuronal_status == 'neuronal']
    non_neuro = ct_df[ct_df.neuronal_status == 'non-neuronal']

    # Cell-type-level means (each CT weighted equally)
    emp_neuro = neuro.empirical_mean_shift.mean()
    emp_non = non_neuro.empirical_mean_shift.mean()
    pred_neuro = neuro.predicted_mean_shift.mean()
    pred_non = non_neuro.predicted_mean_shift.mean()

    print(f"  Cell-type-level (equal weight per CT):")
    print(f"    Empirical neuronal: {emp_neuro:.4f}")
    print(f"    Empirical non-neuronal: {emp_non:.4f}")
    print(f"    Empirical diff: {emp_neuro - emp_non:.4f}")
    print(f"    Predicted neuronal: {pred_neuro:.4f}")
    print(f"    Predicted non-neuronal: {pred_non:.4f}")
    print(f"    Predicted diff: {pred_neuro - pred_non:.4f}")

    # Bootstrap the neuronal-minus-non-neuronal difference
    # Resample cell types within group, then take mean of their mean shifts
    neuro_shifts_emp = neuro.empirical_mean_shift.values
    neuro_shifts_pred = neuro.predicted_mean_shift.values
    non_shifts_emp = non_neuro.empirical_mean_shift.values
    non_shifts_pred = non_neuro.predicted_mean_shift.values

    boot_diff_emp = np.empty(N_BOOT)
    boot_diff_pred = np.empty(N_BOOT)
    for b in range(N_BOOT):
        ni = rng.choice(len(neuro_shifts_emp), len(neuro_shifts_emp), replace=True)
        nni = rng.choice(len(non_shifts_emp), len(non_shifts_emp), replace=True)
        boot_diff_emp[b] = neuro_shifts_emp[ni].mean() - non_shifts_emp[nni].mean()
        boot_diff_pred[b] = neuro_shifts_pred[ni].mean() - non_shifts_pred[nni].mean()

    print(f"    Emp diff 95% CI: [{np.percentile(boot_diff_emp, 2.5):.4f}, "
          f"{np.percentile(boot_diff_emp, 97.5):.4f}]")
    print(f"    Pred diff 95% CI: [{np.percentile(boot_diff_pred, 2.5):.4f}, "
          f"{np.percentile(boot_diff_pred, 97.5):.4f}]")

    # Observation-weighted sensitivity
    neuro_delta = delta_df[delta_df.neuronal_status == 'neuronal']
    non_delta = delta_df[delta_df.neuronal_status == 'non-neuronal']
    emp_neuro_w = neuro_delta.empirical_delta.mean()
    emp_non_w = non_delta.empirical_delta.mean()
    pred_neuro_w = neuro_delta.predicted_delta.mean()
    pred_non_w = non_delta.predicted_delta.mean()

    print(f"\n  Observation-weighted (sensitivity):")
    print(f"    Empirical diff: {emp_neuro_w - emp_non_w:.4f}")
    print(f"    Predicted diff: {pred_neuro_w - pred_non_w:.4f}")

    result = pd.DataFrame([{
        'analysis': 'ct_level_equal_weight',
        'emp_neuronal': emp_neuro, 'emp_non_neuronal': emp_non,
        'emp_diff': emp_neuro - emp_non,
        'pred_neuronal': pred_neuro, 'pred_non_neuronal': pred_non,
        'pred_diff': pred_neuro - pred_non,
        'emp_diff_ci_lo': float(np.percentile(boot_diff_emp, 2.5)),
        'emp_diff_ci_hi': float(np.percentile(boot_diff_emp, 97.5)),
        'pred_diff_ci_lo': float(np.percentile(boot_diff_pred, 2.5)),
        'pred_diff_ci_hi': float(np.percentile(boot_diff_pred, 97.5)),
        'bootstrap_resampled': 'cell types within neuronal/non-neuronal groups',
        'n_boot': N_BOOT,
    }, {
        'analysis': 'observation_weighted',
        'emp_neuronal': emp_neuro_w, 'emp_non_neuronal': emp_non_w,
        'emp_diff': emp_neuro_w - emp_non_w,
        'pred_neuronal': pred_neuro_w, 'pred_non_neuronal': pred_non_w,
        'pred_diff': pred_neuro_w - pred_non_w,
        'bootstrap_resampled': 'N/A (sensitivity only)',
        'n_boot': 0,
    }])

    return result


# ============================================================
# 7. Analysis 4: Subgroups
# ============================================================

def analysis4_subgroups(subgroup_genes, delta_df, common_cts):
    """Repeat cell-type and neuro/non-neuro for SynGO subgroups."""
    print("\n" + "=" * 70)
    print("Analysis 4: Synaptic subgroups")
    print("=" * 70)

    records = []
    for sg_name, sg_set in subgroup_genes.items():
        sg_delta = delta_df[delta_df.gene.isin(sg_set)]
        n_genes = sg_delta.gene.nunique()

        if n_genes < 5:
            print(f"  {sg_name}: {n_genes} genes — skipped")
            records.append({
                'subgroup': sg_name, 'n_genes': n_genes,
                'status': 'skipped', 'reason': 'too few genes',
            })
            continue

        # Per-CT mean shifts
        ct_emp = {}
        ct_pred = {}
        for ct in common_cts:
            ct_data = sg_delta[sg_delta.cell_type == ct]
            ct_emp[ct] = ct_data.empirical_delta.mean()
            ct_pred[ct] = ct_data.predicted_delta.mean()

        emp_arr = np.array([ct_emp[ct] for ct in common_cts])
        pred_arr = np.array([ct_pred[ct] for ct in common_cts])

        pr, _ = stats.pearsonr(emp_arr, pred_arr)
        mae = np.mean(np.abs(pred_arr - emp_arr))

        # Neuronal vs non-neuronal
        neuro_emp = np.mean([ct_emp[ct] for ct in common_cts if ct in NEURONAL])
        non_emp = np.mean([ct_emp[ct] for ct in common_cts if ct in NON_NEURONAL])
        neuro_pred = np.mean([ct_pred[ct] for ct in common_cts if ct in NEURONAL])
        non_pred = np.mean([ct_pred[ct] for ct in common_cts if ct in NON_NEURONAL])

        records.append({
            'subgroup': sg_name,
            'n_genes': n_genes,
            'status': 'computed',
            'emp_neuro_diff': float(neuro_emp - non_emp),
            'pred_neuro_diff': float(neuro_pred - non_pred),
            'cross_ct_pearson': float(pr),
            'cross_ct_mae': float(mae),
            'emp_neuronal': float(neuro_emp),
            'emp_non_neuronal': float(non_emp),
            'pred_neuronal': float(neuro_pred),
            'pred_non_neuronal': float(non_pred),
        })

        print(f"  {sg_name} (n={n_genes}): r={pr:.3f}, MAE={mae:.4f}, "
              f"emp_diff={neuro_emp-non_emp:.4f}, pred_diff={neuro_pred-non_pred:.4f}")

    return pd.DataFrame(records)


# ============================================================
# 8. Figure
# ============================================================

def create_figure(delta_df, ct_df, neuro_df, sg_df):
    """Four-panel publication figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    print("\n" + "=" * 70)
    print("Creating figure")
    print("=" * 70)

    colors = {'neuronal': '#4477AA', 'non-neuronal': '#228833'}

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # --- Panel A: Hexbin scatter ---
    ax = axes[0, 0]
    de = delta_df.empirical_delta.values
    dp = delta_df.predicted_delta.values
    hb = ax.hexbin(de, dp, gridsize=60, cmap='Blues', mincnt=1)
    plt.colorbar(hb, ax=ax, label='Count')

    lims = [min(de.min(), dp.min()) - 0.1, max(de.max(), dp.max()) + 0.1]
    ax.plot(lims, lims, 'k--', alpha=0.4, linewidth=0.8)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Empirical delta (ground truth)', fontsize=10)
    ax.set_ylabel('Predicted delta (ImputeVAE)', fontsize=10)
    ax.set_title('(A) Gene-level imputation accuracy', fontsize=11, fontweight='bold')

    pr, _ = stats.pearsonr(de, dp)
    sr, _ = stats.spearmanr(de, dp)
    ss_r = np.sum((dp - de) ** 2)
    ss_t = np.sum((de - de.mean()) ** 2)
    r2 = 1 - ss_r / ss_t
    mae = np.mean(np.abs(dp - de))
    n = len(de)
    stats_text = (f"n = {n:,}\n"
                  f"Pearson r = {pr:.3f}\n"
                  f"Spearman \u03c1 = {sr:.3f}\n"
                  f"R\u00b2 = {r2:.3f}\n"
                  f"MAE = {mae:.4f}")
    ax.text(0.04, 0.96, stats_text, transform=ax.transAxes, fontsize=8,
            va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # --- Panel B: Cell-type mean shifts ---
    ax = axes[0, 1]
    ct_sorted = ct_df.sort_values('empirical_mean_shift', ascending=False)
    y = np.arange(len(ct_sorted))

    for i, (_, row) in enumerate(ct_sorted.iterrows()):
        c = colors.get(row['neuronal_status'], 'gray')
        alpha_pt = 0.4 if row['low_sample'] else 1.0

        # CIs
        ax.errorbar(row['empirical_mean_shift'], y[i],
                     xerr=[[row['empirical_mean_shift'] - row['emp_ci_lo']],
                            [row['emp_ci_hi'] - row['empirical_mean_shift']]],
                     color=c, alpha=0.2, fmt='none', capsize=2)
        ax.errorbar(row['predicted_mean_shift'], y[i],
                     xerr=[[row['predicted_mean_shift'] - row['pred_ci_lo']],
                            [row['pred_ci_hi'] - row['predicted_mean_shift']]],
                     color=c, alpha=0.12, fmt='none', capsize=2)

        ax.plot([row['empirical_mean_shift'], row['predicted_mean_shift']],
                [y[i], y[i]], color=c, alpha=0.3, linewidth=1)
        marker_e = 'D' if row['low_sample'] else 'o'
        marker_p = 'D' if row['low_sample'] else 's'
        ax.scatter(row['empirical_mean_shift'], y[i], color=c, marker=marker_e,
                   s=40, zorder=3, alpha=alpha_pt, edgecolors='white', linewidth=0.3)
        ax.scatter(row['predicted_mean_shift'], y[i], color=c, marker=marker_p,
                   s=40, zorder=3, alpha=alpha_pt * 0.8, edgecolors='white', linewidth=0.3)

    ax.set_yticks(y)
    ax.set_yticklabels(ct_sorted.cell_type.values, fontsize=8)
    ax.set_xlabel('Mean SynGO shift (cell minus nucleus)', fontsize=9)
    ax.set_title('(B) Cell-type mean SynGO shifts', fontsize=11, fontweight='bold')

    legend_el = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               label='Empirical (ground truth)', markersize=7),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
               label='Predicted (ImputeVAE)', markersize=7),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='gray',
               label='Low-sample (<20 obs)', markersize=6),
        Line2D([0], [0], color=colors['neuronal'], label='Neuronal', linewidth=2),
        Line2D([0], [0], color=colors['non-neuronal'], label='Non-neuronal', linewidth=2),
    ]
    ax.legend(handles=legend_el, fontsize=6.5, loc='lower right')

    # --- Panel C: Neuronal vs non-neuronal summary ---
    ax = axes[1, 0]
    primary = neuro_df[neuro_df.analysis == 'ct_level_equal_weight'].iloc[0]

    bar_w = 0.25
    x_pos = np.array([0, 1, 2.2])
    emp_vals = [primary['emp_neuronal'], primary['emp_non_neuronal'], primary['emp_diff']]
    pred_vals = [primary['pred_neuronal'], primary['pred_non_neuronal'], primary['pred_diff']]

    ax.bar(x_pos - bar_w/2, emp_vals, bar_w, color='#4477AA', alpha=0.8,
           label='Empirical')
    ax.bar(x_pos + bar_w/2, pred_vals, bar_w, color='#E8855E', alpha=0.8,
           label='Predicted')

    # CI on difference
    ax.errorbar(x_pos[2] - bar_w/2, emp_vals[2], fmt='none',
                yerr=[[emp_vals[2] - primary['emp_diff_ci_lo']],
                      [primary['emp_diff_ci_hi'] - emp_vals[2]]],
                color='#4477AA', capsize=4)
    ax.errorbar(x_pos[2] + bar_w/2, pred_vals[2], fmt='none',
                yerr=[[pred_vals[2] - primary['pred_diff_ci_lo']],
                      [primary['pred_diff_ci_hi'] - pred_vals[2]]],
                color='#E8855E', capsize=4)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Neuronal', 'Non-neuronal', 'Difference'], fontsize=9)
    ax.set_ylabel('Mean SynGO shift', fontsize=9)
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.legend(fontsize=8)
    ax.set_title('(C) Neuronal vs non-neuronal', fontsize=11, fontweight='bold')

    fig.text(0.27, 0.475, 'Error bars: 95% CI (bootstrap resampling cell types)',
             fontsize=7, fontstyle='italic', ha='center')

    # --- Panel D: Subgroup differences ---
    ax = axes[1, 1]
    sg_computed = sg_df[sg_df.status == 'computed'].sort_values('emp_neuro_diff')
    if len(sg_computed) > 0:
        y_sg = np.arange(len(sg_computed))
        bar_h = 0.35

        ax.barh(y_sg - bar_h/2, sg_computed.emp_neuro_diff.values,
                height=bar_h, color='#4477AA', alpha=0.8, label='Empirical')
        ax.barh(y_sg + bar_h/2, sg_computed.pred_neuro_diff.values,
                height=bar_h, color='#E8855E', alpha=0.8, label='Predicted')

        labels = [f"{r['subgroup'].replace('_', ' ')} (n={r['n_genes']})"
                  for _, r in sg_computed.iterrows()]
        ax.set_yticks(y_sg)
        ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
        ax.set_xlabel('Neuronal minus non-neuronal\nmean SynGO shift', fontsize=9)
        ax.legend(fontsize=8, loc='lower right')

    ax.set_title('(D) Subgroup neuronal enrichment', fontsize=11, fontweight='bold')

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(os.path.join(OUT_DIR, 'syngo_brain_validation.png'),
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, 'syngo_brain_validation.pdf'),
                bbox_inches='tight')
    plt.close(fig)
    print("  Figure saved")


# ============================================================
# 9. README
# ============================================================

def write_readme(gene_names, syngo_genes, common_cts, rare_cts,
                 flow_df, overall_df, ct_df, neuro_df, sg_df):
    """Write comprehensive README."""

    cross_all = ct_df.attrs.get('cross_ct_all', {})
    cross_robust = ct_df.attrs.get('cross_ct_robust', {})

    readme = f"""# SynGO Brain-Specific Biological Consistency Analysis

## Question

Do non-ribosomal genes encoding synaptic proteins exhibit larger nucleus-to-cell
expression shifts in neuronal than in non-neuronal populations, and does ImputeVAE
reproduce the magnitude and cell-type dependence of those shifts?

SynGO annotates the function and localization of the encoded proteins, not RNA
localization. This analysis tests brain-specific biological consistency.

## Input Files

| Input | Path | Notes |
|-------|------|-------|
| Gene list | `data/training_gene_list_v3.csv` | 10,134 genes (Config B) |
| Split manifest | `data/split_manifest.tsv` | Train/valid/test split |
| Common cell types | `data/common_cell_types.csv` | 17 shared cell types |
| Common genes | `data/common_genes.csv` | Shared across modalities |
| Ortholog mapping | `data/external/mouse_human_orthologs.csv` | Ensembl BioMart |
| SynGO archive | `validation/syngo/syngo1.3_complete_data.zip` | v1.3 |
| Raw cells | `10X_cells_v2_v3_AIBS/data.h5ad` (c3A batch) | log1p normalized |
| Raw nuclei | `10X_nuclei_v2_v3_AIBS/data.h5ad` (n3A batch) | log1p normalized |
| ImputeVAE predictions | `run/eval_v3_splitaware/cvae_imputed_valid.npy` | Validation only |

## Validation Partition

- Only validation observations are used for all metrics (20% partition)
- Training observations are never loaded for evaluation
- Test observations are never loaded
- Split defined in `data/split_manifest.tsv`, random_state=42

## Matrix Shapes and Alignment

- Validation cells: shape verified at runtime
- Validation nuclei: shape verified at runtime
- ImputeVAE imputed: (8033, 10134), validation nuclei only
- Gene order: all three matrices verified identical (sorted by gene name)
- Cell type alignment: imputed vs observed nuclei verified at 100%
- Cross-decoded nuclei use the same validation nuclear observations

## SynGO Hierarchy Construction

1. SynGO v1.3 ontologies.xlsx: each CC term row contains propagated gene lists
2. GO:0045202 ("synapse") row contains all descendant-term genes
3. No recursive traversal needed; the comma-separated gene list is pre-propagated

## Human-to-Mouse Mapping

- Primary: one-to-one orthologs from `data/external/mouse_human_orthologs.csv`
  (Ensembl BioMart, mouse_sym unique, filtered to one-to-one human->mouse)
- Fallback: capitalization heuristic (GENE -> Gene) for genes not in ortholog table,
  labeled explicitly in syngo_gene_metadata.csv
- {len(syngo_genes)} non-ribosomal genes in primary analysis

## Ribosomal Exclusion

Genes with mouse symbols starting with Rpl or Rps are excluded from the primary
set. They encode cytoplasmic ribosomal proteins expected to show large positive
shifts regardless of synaptic biology.

## Equations

For SynGO gene g and cell type c:
- empirical_delta[g,c] = mean(valid_cell[g,c]) - mean(valid_nuc[g,c])
- predicted_delta[g,c] = mean(imputed_valid_nuc[g,c]) - mean(valid_nuc[g,c])

Cell-type mean SynGO shift:
- empirical_mean_shift[c] = mean of empirical_delta[g,c] across SynGO genes g
- predicted_mean_shift[c] = mean of predicted_delta[g,c] across SynGO genes g

## Bootstrap Procedure

- {N_BOOT} replicates, seed={SEED}
- Cell-type shifts: resample whole observations within modality and cell type;
  cross-decoded and observed nuclei are paired when resampling
- Neuronal vs non-neuronal: resample cell types within group (12 neuronal, 5 non-neuronal)
- 95% CI from 2.5th and 97.5th percentiles

## Neuronal / Non-Neuronal Classification

Neuronal (12): {', '.join(NEURONAL)}
Non-neuronal (5): {', '.join(NON_NEURONAL)}

## Low-Sample Cell Types

Flagged (< 20 validation cells or nuclei): {', '.join(rare_cts) if rare_cts else 'none'}

These are retained in the figure (diamond markers) but excluded from a sensitivity
cross-cell-type summary.

## Cross-Cell-Type Agreement

All {len(common_cts)} CTs: Pearson r = {cross_all.get('pearson_r', 'N/A'):.4f}, R2 = {cross_all.get('R2', 'N/A'):.4f}
Excluding low-sample: Pearson r = {cross_robust.get('pearson_r', 'N/A'):.4f}, R2 = {cross_robust.get('R2', 'N/A'):.4f}

## Statistical Limitations

1. Gene-cell-type observations are not statistically independent (genes are
   correlated, cell types share donors). Pooled correlations overstate effective
   sample size.
2. Cell-type-level analyses use 17 (or fewer) data points. Standard errors are wide.
3. Bootstrap CIs for cell-type shifts resample observations but do not account for
   gene-gene correlation within cell types.
4. Neuronal vs non-neuronal bootstrap resamples cell types (n=12 neuronal,
   n=5 non-neuronal), which yields wide intervals.
5. Rare cell types (Endo: 4 nuclei) have very few observations; their point
   estimates are unstable.
6. This is a distributional comparison (unpaired cells and nuclei), not a
   paired-cell analysis.

## Interpretation

SynGO annotates synaptic protein function and localization, not subcellular
RNA localization. This analysis establishes neurobiological consistency rather
than direct molecular validation of RNA localization.

## Output Files

- `syngo_filtering_counts.csv` - Gene set filtering flow
- `syngo_gene_metadata.csv` - Per-gene ortholog/subgroup metadata
- `syngo_gene_celltype_deltas.csv` - All gene x cell type deltas
- `syngo_overall_quality.csv` - Pooled imputation quality metrics
- `syngo_quality_by_celltype.csv` - Per-CT imputation quality
- `syngo_celltype_summary.csv` - CT mean shifts with bootstrap CIs
- `syngo_neuronal_vs_nonneuronal.csv` - Neuro/non-neuro comparison
- `syngo_subgroup_summary.csv` - Subgroup analyses
- `syngo_brain_validation.png/pdf` - Four-panel figure
"""

    with open(os.path.join(OUT_DIR, 'README.md'), 'w') as f:
        f.write(readme)
    print("  README written")


# ============================================================
# 10. QC verification
# ============================================================

def verify_results(delta_df, overall_df, ct_df):
    """Independently recompute key metrics from exported delta CSV."""
    print("\n" + "=" * 70)
    print("QC verification")
    print("=" * 70)

    de = delta_df.empirical_delta.values
    dp = delta_df.predicted_delta.values

    pr, _ = stats.pearsonr(de, dp)
    reported = overall_df.iloc[0]['pearson_r']
    match = abs(pr - reported) < 1e-6
    print(f"  Pearson recomputed: {pr:.6f}, reported: {reported:.6f}, match: {match}")

    mae = np.mean(np.abs(dp - de))
    reported_mae = overall_df.iloc[0]['MAE']
    match_mae = abs(mae - reported_mae) < 1e-6
    print(f"  MAE recomputed: {mae:.6f}, reported: {reported_mae:.6f}, match: {match_mae}")

    # Verify CT means from delta_df
    for _, row in ct_df.iterrows():
        ct = row['cell_type']
        ct_data = delta_df[delta_df.cell_type == ct]
        emp_mean = ct_data.empirical_delta.mean()
        err = abs(emp_mean - row['empirical_mean_shift'])
        if err > 1e-4:
            print(f"  WARNING: {ct} empirical mean mismatch: {emp_mean:.4f} vs {row['empirical_mean_shift']:.4f}")

    print("  QC verification complete")


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load
    (gene_names, gene_list,
     valid_cell_X, valid_cell_ct, valid_cell_meta,
     valid_nuc_X, valid_nuc_ct, valid_nuc_meta,
     imputed_X, imputed_ct) = load_data()

    # SynGO sets
    syngo_genes, subgroup_genes, gene_meta, flow_df = build_syngo_sets(gene_names, gene_list)

    # Save gene metadata and filtering flow
    flow_df.to_csv(os.path.join(OUT_DIR, 'syngo_filtering_counts.csv'), index=False)
    gene_meta.to_csv(os.path.join(OUT_DIR, 'syngo_gene_metadata.csv'), index=False)

    # Compute deltas
    delta_df, common_cts, rare_cts = compute_deltas(
        gene_names, syngo_genes,
        valid_cell_X, valid_cell_ct,
        valid_nuc_X, valid_nuc_ct,
        imputed_X, imputed_ct)
    delta_df.to_csv(os.path.join(OUT_DIR, 'syngo_gene_celltype_deltas.csv'), index=False)

    # Analysis 1
    overall_df, per_ct_df = analysis1_quality(delta_df, common_cts)
    overall_df.to_csv(os.path.join(OUT_DIR, 'syngo_overall_quality.csv'), index=False)
    per_ct_df.to_csv(os.path.join(OUT_DIR, 'syngo_quality_by_celltype.csv'), index=False)

    # Analysis 2
    ct_df = analysis2_celltype(
        delta_df, common_cts, rare_cts,
        valid_cell_X, valid_cell_ct,
        valid_nuc_X, valid_nuc_ct,
        imputed_X, imputed_ct,
        gene_names, syngo_genes)
    ct_df.to_csv(os.path.join(OUT_DIR, 'syngo_celltype_summary.csv'), index=False)

    # Analysis 3
    neuro_df = analysis3_neuro(ct_df, delta_df)
    neuro_df.to_csv(os.path.join(OUT_DIR, 'syngo_neuronal_vs_nonneuronal.csv'), index=False)

    # Analysis 4
    sg_df = analysis4_subgroups(subgroup_genes, delta_df, common_cts)
    sg_df.to_csv(os.path.join(OUT_DIR, 'syngo_subgroup_summary.csv'), index=False)

    # Figure
    create_figure(delta_df, ct_df, neuro_df, sg_df)

    # QC
    verify_results(delta_df, overall_df, ct_df)

    # README
    write_readme(gene_names, syngo_genes, common_cts, rare_cts,
                 flow_df, overall_df, ct_df, neuro_df, sg_df)

    # Terminal summary
    print("\n" + "=" * 70)
    print("TERMINAL SUMMARY")
    print("=" * 70)
    print(f"\n  SynGO genes: {len(syngo_genes)} non-ribosomal")
    print(f"  Cell types: {len(common_cts)} ({len(rare_cts)} low-sample)")
    print(f"  Gene-CT observations: {len(delta_df)}")

    o = overall_df.iloc[0]
    print(f"\n  Overall quality:")
    print(f"    Pearson r = {o['pearson_r']:.4f}")
    print(f"    Spearman rho = {o['spearman_r']:.4f}")
    print(f"    R2 = {o['R2']:.4f}")
    print(f"    MAE = {o['MAE']:.4f}")
    print(f"    RMSE = {o['RMSE']:.4f}")
    print(f"    Sign agreement = {o['sign_agreement']:.4f}")

    cross = ct_df.attrs.get('cross_ct_all', {})
    print(f"\n  Cross-CT agreement:")
    print(f"    Pearson r = {cross.get('pearson_r', 'N/A')}")
    print(f"    R2 = {cross.get('R2', 'N/A')}")

    p = neuro_df[neuro_df.analysis == 'ct_level_equal_weight'].iloc[0]
    print(f"\n  Neuronal vs non-neuronal:")
    print(f"    Emp diff = {p['emp_diff']:.4f}")
    print(f"    Pred diff = {p['pred_diff']:.4f}")

    print(f"\n  Outputs: {OUT_DIR}/")
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"    {f}")

    print("\n" + "=" * 70)
    print("Analysis complete.")
    print("=" * 70)


if __name__ == '__main__':
    main()
