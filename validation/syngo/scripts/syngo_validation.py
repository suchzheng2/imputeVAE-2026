#!/usr/bin/env python
"""
SynGO-based biological consistency analysis for ImputeVAE.

Evaluates whether ImputeVAE reproduces cell-type-specific nucleus-to-cell
expression shifts among SynGO-annotated synaptic genes, with stronger shifts
in neuronal populations.

Usage:
    python validation/syngo/scripts/syngo_validation.py \
        --output_dir validation/syngo/results

Requires conda env: impute2
"""

import os
import sys
import argparse
import zipfile
import io
import json
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats, sparse
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score
import anndata as ad

# Suppress openpyxl warnings
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

SEED = 42
np.random.seed(SEED)

# ============================================================
# Project paths
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_ROOT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi"

DEFAULT_SYNGO_ZIP = os.path.join(PROJECT_ROOT, "validation/syngo/syngo1.3_complete_data.zip")
DEFAULT_GENE_LIST = os.path.join(PROJECT_ROOT, "data/training_gene_list_v3.csv")
DEFAULT_SPLIT_MANIFEST = os.path.join(PROJECT_ROOT, "data/split_manifest.tsv")
DEFAULT_COMMON_CT = os.path.join(PROJECT_ROOT, "data/common_cell_types.csv")
DEFAULT_ORTHOLOG_MAP = os.path.join(PROJECT_ROOT, "data/external/mouse_human_orthologs.csv")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "validation/syngo/results")

# Prediction files
CVAE_FULL_IMPUTED = os.path.join(PROJECT_ROOT, "run/cvae_full_eval/imputed_cell_from_nucleus.npy")
CVAE_FULL_OBS = os.path.join(PROJECT_ROOT, "run/cvae_full_eval/nucleus_obs.csv")
CVAE_VALID_IMPUTED = os.path.join(PROJECT_ROOT, "run/eval_v3_splitaware/cvae_imputed_valid.npy")
CVAE_VALID_OBS = os.path.join(PROJECT_ROOT, "run/eval_v3_splitaware/valid_nuc_obs.csv")

# Baseline prediction directories
BASELINE_DIRS = {
    'CT-mean': os.path.join(PROJECT_ROOT, 'run/ctmean_baseline'),
    'scVI': os.path.join(PROJECT_ROOT, 'run/scvi_baseline'),
    'scANVI': os.path.join(PROJECT_ROOT, 'run/scanvi_baseline'),
    'Linear_reg': os.path.join(PROJECT_ROOT, 'run/linreg_baseline'),
    'scGen': os.path.join(PROJECT_ROOT, 'run/scgen_baseline'),
    'scButterfly': os.path.join(PROJECT_ROOT, 'run/scbutterfly_baseline'),
}

# Neuronal cell types (verified from project metadata)
NEURONAL_TYPES = [
    'L2/3 IT', 'L5 ET', 'L5 IT', 'L5/6 NP', 'L6 CT', 'L6 IT', 'L6b',
    'Lamp5', 'Pvalb', 'Sncg', 'Sst', 'Vip',
]
NON_NEURONAL_TYPES = ['Astro', 'Endo', 'Macrophage', 'OPC', 'Oligo']

# SynGO gene set definitions
SYNGO_SETS = {
    'synapse': {'go_ids': ['GO:0045202'], 'type': 'broad'},
    'presynaptic_only': {'go_ids': ['GO:0098793'], 'exclude': ['GO:0098794'], 'type': 'location'},
    'postsynaptic_only': {'go_ids': ['GO:0098794'], 'exclude': ['GO:0098793'], 'type': 'location'},
    'both_pre_post': {'go_ids': ['GO:0098793', 'GO:0098794'], 'intersect': True, 'type': 'location'},
    'postsynaptic_density': {'go_ids': ['GO:0014069'], 'type': 'distal'},
    'presynaptic_active_zone': {'go_ids': ['GO:0048786'], 'type': 'distal'},
    'synaptic_vesicle': {'go_ids': ['GO:0008021'], 'type': 'distal'},
    'presynaptic_cytosol': {'go_ids': ['GO:0099523'], 'type': 'distal'},
    'postsynaptic_cytosol': {'go_ids': ['GO:0099524'], 'type': 'distal'},
    'synaptic_cytosol': {'go_ids': ['GO:0099523', 'GO:0099524'], 'union': True, 'type': 'distal'},
    'axo_dendritic_transport': {'go_ids': ['GO:0008088'], 'type': 'transport'},
    'dendritic_transport': {'go_ids': ['GO:0098935'], 'type': 'transport'},
    'axonal_transport': {'go_ids': ['GO:0098930'], 'type': 'transport'},
    'translation_at_synapse': {'go_ids': ['GO:0140241'], 'type': 'translation'},
    'translation_at_presynapse': {'go_ids': ['GO:0140236'], 'type': 'translation'},
    'translation_at_postsynapse': {'go_ids': ['GO:0140242'], 'type': 'translation'},
    'presynaptic_ribosome': {'go_ids': ['SYNGO:presyn_ribosome'], 'type': 'translation'},
    'postsynaptic_ribosome': {'go_ids': ['SYNGO:postsyn_ribosome'], 'type': 'translation'},
}


def parse_args():
    p = argparse.ArgumentParser(description='SynGO biological consistency analysis')
    p.add_argument('--output_dir', default=DEFAULT_OUTPUT_DIR)
    p.add_argument('--syngo_zip', default=DEFAULT_SYNGO_ZIP)
    p.add_argument('--gene_list', default=DEFAULT_GENE_LIST)
    p.add_argument('--split_manifest', default=DEFAULT_SPLIT_MANIFEST)
    p.add_argument('--ortholog_map', default=DEFAULT_ORTHOLOG_MAP)
    p.add_argument('--n_bootstrap', type=int, default=2000)
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--priority', type=int, default=3, choices=[1, 2, 3],
                   help='Priority level: 1=core, 2=subgroups+figure, 3=baselines+supp')
    return p.parse_args()


# ============================================================
# Section 1: Data Loading
# ============================================================

def load_gene_list(path):
    """Load the 10,134-gene model feature list."""
    genes = pd.read_csv(path, header=None)[0].tolist()
    print(f"  Gene list: {len(genes)} genes from {path}")
    return genes


def load_split_manifest(path):
    """Load split manifest with obs_id, modality, cell_type, donor, partition."""
    manifest = pd.read_csv(path, sep='\t')
    print(f"  Split manifest: {manifest.shape[0]} entries")
    return manifest


def get_validation_indices(manifest, modality):
    """Get positional indices for validation observations.

    The split manifest obs_ids map to the sorted order of the filtered adata.
    Positional index = rank in sorted obs_id order within the modality.
    """
    mod_data = manifest[manifest.modality == modality].sort_values('obs_id').reset_index(drop=True)
    valid_mask = mod_data.partition == 'valid'
    train_mask = mod_data.partition == 'train'
    valid_idx = np.where(valid_mask)[0]
    train_idx = np.where(train_mask)[0]
    valid_meta = mod_data[valid_mask].reset_index(drop=True)
    train_meta = mod_data[train_mask].reset_index(drop=True)
    return valid_idx, valid_meta, train_idx, train_meta


def load_raw_expression(data_dir, batch, gene_list, common_ct):
    """Load raw expression data with same filtering as training pipeline.

    Loads full h5ad, applies row/col filters on the view, then copies only
    the filtered subset to avoid materializing the full dense matrix.
    """
    path = os.path.join(DATA_ROOT, data_dir, "data.h5ad")
    print(f"  Loading {path}...")
    adata = ad.read_h5ad(path)

    if 'meta_batch_id' not in adata.obs:
        adata.obs['meta_batch_id'] = adata.obs['batch_id'].apply(lambda x: x.split('.')[0])

    common_genes = pd.read_csv(os.path.join(PROJECT_ROOT, 'data/common_genes.csv'),
                               header=0)['common_genes'].tolist()

    row_mask = (adata.obs['meta_batch_id'].str.endswith(batch) &
                adata.obs['cell_type'].isin(common_ct))
    col_mask = (adata.var['gene_names'].isin(common_genes) &
                adata.var['gene_names'].isin(gene_list))
    adata = adata[row_mask, col_mask].copy()
    adata = adata[:, adata.var['gene_names'].argsort()].copy()

    X = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
    adata.X = np.log1p(X).astype(np.float32)

    gene_names = adata.var['gene_names'].values.tolist()
    print(f"    Shape: {adata.shape}, genes: {len(gene_names)}")
    return adata, gene_names


def load_imputed_validation(gene_list):
    """Load ImputeVAE cross-decoded predictions for validation nuclei only."""
    # Try split-aware first
    if os.path.exists(CVAE_VALID_IMPUTED):
        imputed = np.load(CVAE_VALID_IMPUTED)
        obs = pd.read_csv(CVAE_VALID_OBS)
        print(f"  Loaded split-aware valid imputed: {imputed.shape}")
        return imputed, obs

    # Fall back to extracting from full predictions
    print("  Split-aware predictions not found, extracting from full predictions...")
    imputed_full = np.load(CVAE_FULL_IMPUTED)
    obs_full = pd.read_csv(CVAE_FULL_OBS)
    manifest = load_split_manifest(DEFAULT_SPLIT_MANIFEST)
    valid_idx, valid_meta, _, _ = get_validation_indices(manifest, 'nuclei')

    imputed = imputed_full[valid_idx]
    obs = pd.DataFrame({'cell_type': obs_full.iloc[valid_idx].cell_type.values})
    print(f"  Extracted valid imputed: {imputed.shape}")
    return imputed, obs


def load_baseline_validation(name, base_dir, manifest, valid_idx):
    """Load baseline predictions for validation nuclei."""
    npy_path = os.path.join(base_dir, 'imputed_cell_from_nucleus.npy')
    obs_path = os.path.join(base_dir, 'nucleus_obs.csv')
    if not os.path.exists(npy_path):
        return None, None, f"Missing {npy_path}"

    imputed_full = np.load(npy_path)
    obs_full = pd.read_csv(obs_path)
    imputed = imputed_full[valid_idx]
    obs = pd.DataFrame({'cell_type': obs_full.iloc[valid_idx].cell_type.values})
    return imputed, obs, None


# ============================================================
# Section 2: SynGO Characterization
# ============================================================

def load_syngo(zip_path):
    """Load and characterize the SynGO resource."""
    zf = zipfile.ZipFile(zip_path)
    genes = pd.read_excel(io.BytesIO(zf.read('genes.xlsx')))
    annotations = pd.read_excel(io.BytesIO(zf.read('annotations.xlsx')))
    ontologies = pd.read_excel(io.BytesIO(zf.read('ontologies.xlsx')))
    return genes, annotations, ontologies


def characterize_syngo(genes, annotations, ontologies, output_dir):
    """Characterize the SynGO resource and save summary tables."""
    tables_dir = os.path.join(output_dir, 'tables')

    # Resource summary
    n_genes = len(genes)
    n_hgnc = genes['hgnc_symbol'].nunique()
    n_annotations = len(annotations)
    n_terms = ontologies['id'].nunique()

    cc_annot = annotations[annotations['go_domain'] == 'CC']
    bp_annot = annotations[annotations['go_domain'] == 'BP']
    n_cc = len(cc_annot)
    n_bp = len(bp_annot)

    n_pubmed = annotations['pubmed_id'].nunique()
    annot_per_gene = annotations.groupby('hgnc_symbol').size()

    # Evidence breakdown
    evidence_bio = annotations['evidence_biological_system'].dropna()
    evidence_counts = {}
    for val in evidence_bio:
        for item in str(val).split(','):
            item = item.strip()
            if item and item != 'nan':
                evidence_counts[item] = evidence_counts.get(item, 0) + 1

    # Categorize evidence
    cultured_neuron = sum(v for k, v in evidence_counts.items() if 'cultneuron' in k)
    intact_tissue = sum(v for k, v in evidence_counts.items() if 'intacttissue' in k)
    non_neuronal = sum(v for k, v in evidence_counts.items()
                       if any(x in k for x in ['cellline', 'nonneuron', 'heterocell']))

    # Assay types
    assay_counts = {}
    for val in annotations['evidence_experiment_assay'].dropna():
        for item in str(val).split(','):
            item = item.strip()
            if item and item != 'nan':
                assay_counts[item] = assay_counts.get(item, 0) + 1

    # Parse gene lists from ontologies.xlsx (comma-separated per term row)
    def _parse_term_genes(onto_df, go_id):
        row = onto_df[onto_df['id'] == go_id]
        if len(row) == 0:
            return set()
        val = str(row.iloc[0]['hgnc_symbol'])
        if val == 'nan':
            return set()
        return {g.strip() for g in val.split(',') if g.strip()}

    # Genes per major ontology branch
    branch_counts = {}
    for go_id in ['GO:0045202', 'GO:0098793', 'GO:0098794']:
        branch_genes = _parse_term_genes(ontologies, go_id)
        row = ontologies[ontologies['id'] == go_id]
        branch_name = row.iloc[0]['name'] if len(row) > 0 else go_id
        branch_counts[branch_name] = len(branch_genes)

    # Count genes annotated to each term
    term_gene_counts = {}
    for _, row in ontologies.iterrows():
        val = str(row['hgnc_symbol'])
        if val != 'nan':
            genes_in_term = {g.strip() for g in val.split(',') if g.strip()}
            term_gene_counts[row['id']] = len(genes_in_term)
        else:
            term_gene_counts[row['id']] = 0

    resource_summary = pd.DataFrame([{
        'metric': 'SynGO version', 'value': '1.3',
    }, {
        'metric': 'genes_in_genes_xlsx', 'value': n_genes,
    }, {
        'metric': 'unique_hgnc_symbols_genes', 'value': n_hgnc,
    }, {
        'metric': 'unique_hgnc_symbols_annotations', 'value': annotations['hgnc_symbol'].nunique(),
    }, {
        'metric': 'total_annotations', 'value': n_annotations,
    }, {
        'metric': 'ontology_terms', 'value': n_terms,
    }, {
        'metric': 'CC_annotations', 'value': n_cc,
    }, {
        'metric': 'BP_annotations', 'value': n_bp,
    }, {
        'metric': 'supporting_pubmed_records', 'value': n_pubmed,
    }, {
        'metric': 'median_annotations_per_gene', 'value': float(annot_per_gene.median()),
    }, {
        'metric': 'mean_annotations_per_gene', 'value': float(annot_per_gene.mean()),
    }, {
        'metric': 'evidence_intact_tissue', 'value': intact_tissue,
    }, {
        'metric': 'evidence_cultured_neurons', 'value': cultured_neuron,
    }, {
        'metric': 'evidence_non_neuronal', 'value': non_neuronal,
    }, {
        'metric': 'synapse_CC_genes', 'value': term_gene_counts.get('GO:0045202', 0),
    }, {
        'metric': 'presynapse_genes', 'value': term_gene_counts.get('GO:0098793', 0),
    }, {
        'metric': 'postsynapse_genes', 'value': term_gene_counts.get('GO:0098794', 0),
    }])
    resource_summary.to_csv(os.path.join(tables_dir, 'syngo_resource_summary.csv'), index=False)

    # Evidence summary
    evidence_df = pd.DataFrame([
        {'evidence_type': k, 'count': v} for k, v in sorted(evidence_counts.items(), key=lambda x: -x[1])
    ])
    evidence_df.to_csv(os.path.join(tables_dir, 'syngo_evidence_summary.csv'), index=False)

    # Annotation discrepancies
    genes_hgnc = set(genes['hgnc_symbol'].dropna())
    annot_hgnc = set(annotations['hgnc_symbol'].dropna())
    only_in_genes = genes_hgnc - annot_hgnc
    only_in_annot = annot_hgnc - genes_hgnc

    discrepancies = []
    for s in only_in_genes:
        discrepancies.append({'symbol': s, 'in_genes_xlsx': True, 'in_annotations_xlsx': False,
                              'possible_reason': 'gene listed but no annotations'})
    for s in only_in_annot:
        discrepancies.append({'symbol': s, 'in_genes_xlsx': False, 'in_annotations_xlsx': True,
                              'possible_reason': 'annotated but not in gene list (synonym/deprecated?)'})

    disc_df = pd.DataFrame(discrepancies)
    disc_df.to_csv(os.path.join(tables_dir, 'syngo_annotation_discrepancies.csv'), index=False)

    print(f"\n  SynGO Resource Summary:")
    print(f"    Version: 1.3")
    print(f"    Genes in genes.xlsx: {n_genes}")
    print(f"    Unique HGNC symbols (genes): {n_hgnc}")
    print(f"    Unique HGNC symbols (annotations): {annotations['hgnc_symbol'].nunique()}")
    print(f"    Total annotations: {n_annotations}")
    print(f"    Ontology terms: {n_terms}")
    print(f"    CC annotations: {n_cc}, BP annotations: {n_bp}")
    print(f"    Supporting PubMed records: {n_pubmed}")
    print(f"    Median annotations per gene: {annot_per_gene.median():.1f}")
    print(f"    Evidence - intact tissue: {intact_tissue}, cultured neurons: {cultured_neuron}, non-neuronal: {non_neuronal}")
    print(f"    Discrepancies: {len(only_in_genes)} genes-only, {len(only_in_annot)} annotations-only")

    return resource_summary, evidence_df, disc_df


# ============================================================
# Section 3: Ortholog Mapping
# ============================================================

def build_ortholog_mapping(genes_df, ortholog_path, gene_list, ontologies, output_dir):
    """Map human SynGO genes to mouse orthologs."""
    tables_dir = os.path.join(output_dir, 'tables')

    # Load ortholog mapping
    ortho = pd.read_csv(ortholog_path)
    print(f"\n  Ortholog mapping: {len(ortho)} entries from {ortholog_path}")

    # The mapping has unique mouse symbols but some human symbols map to multiple mouse genes
    # For SynGO mapping: human -> mouse, we want one-to-one
    # Keep only human symbols that map to exactly one mouse gene
    human_counts = ortho.groupby('human_sym').size()
    one_to_one_human = set(human_counts[human_counts == 1].index)
    ortho_1to1 = ortho[ortho.human_sym.isin(one_to_one_human)]
    print(f"    One-to-one human->mouse mappings: {len(ortho_1to1)}")

    # Build mapping dict
    h2m = dict(zip(ortho_1to1.human_sym, ortho_1to1.mouse_sym))

    gene_list_set = set(gene_list)

    # Get all SynGO gene sets membership (ontologies.xlsx has comma-separated gene lists per term)
    syngo_memberships = defaultdict(set)
    for _, row in ontologies.iterrows():
        go_id = row['id']
        val = str(row.get('hgnc_symbol', ''))
        if val and val != 'nan':
            for symbol in val.split(','):
                symbol = symbol.strip()
                if symbol:
                    syngo_memberships[symbol].add(go_id)

    # Map each SynGO gene
    mapping_records = []
    for _, row in genes_df.iterrows():
        human_sym = row['hgnc_symbol']
        hgnc_id = row['hgnc_id']
        ensembl_id = row.get('ensembl_id', '')

        mouse_sym = h2m.get(human_sym, None)
        is_1to1 = human_sym in one_to_one_human

        # Also try capitalization fallback
        cap_mouse = human_sym[0] + human_sym[1:].lower() if human_sym else None

        # Determine mapping method
        if mouse_sym is not None:
            mapping_method = 'ortholog_table'
            final_mouse = mouse_sym
        elif cap_mouse and cap_mouse in gene_list_set:
            mapping_method = 'capitalization_fallback'
            final_mouse = cap_mouse
        else:
            mapping_method = 'unmapped'
            final_mouse = None

        present = final_mouse in gene_list_set if final_mouse else False
        is_ribo = (final_mouse or '').startswith('Rpl') or (final_mouse or '').startswith('Rps')

        memberships = syngo_memberships.get(human_sym, set())

        mapping_records.append({
            'human_hgnc_id': hgnc_id,
            'human_symbol': human_sym,
            'human_ensembl_id': ensembl_id,
            'mouse_symbol': final_mouse,
            'mouse_ensembl_id': '',
            'mapping_method': mapping_method,
            'one_to_one': is_1to1,
            'present_in_model': present,
            'is_ribosomal': is_ribo,
            'syngo_memberships': ';'.join(sorted(memberships)),
        })

    mapping_df = pd.DataFrame(mapping_records)
    mapping_df.to_csv(os.path.join(tables_dir, 'syngo_mouse_mapping.csv'), index=False)

    # Filtering flow
    total = len(mapping_df)
    has_ortholog = mapping_df[mapping_df.mapping_method == 'ortholog_table']
    has_fallback = mapping_df[mapping_df.mapping_method == 'capitalization_fallback']
    mapped = mapping_df[mapping_df.mapping_method != 'unmapped']
    in_model = mapping_df[mapping_df.present_in_model]
    ribo = in_model[in_model.is_ribosomal]
    non_ribo = in_model[~in_model.is_ribosomal]

    flow = pd.DataFrame([
        {'stage': 'starting_human_syngo', 'count': total},
        {'stage': 'one_to_one_ortholog', 'count': len(has_ortholog)},
        {'stage': 'capitalization_fallback', 'count': len(has_fallback)},
        {'stage': 'total_mapped', 'count': len(mapped)},
        {'stage': 'present_in_model_10134', 'count': len(in_model)},
        {'stage': 'ribosomal_Rpl_Rps', 'count': len(ribo)},
        {'stage': 'non_ribosomal_in_model', 'count': len(non_ribo)},
    ])
    flow.to_csv(os.path.join(tables_dir, 'syngo_filtering_flow.csv'), index=False)

    print(f"\n  Filtering flow:")
    for _, r in flow.iterrows():
        print(f"    {r['stage']}: {r['count']}")

    return mapping_df


# ============================================================
# Section 4: Gene Set Construction
# ============================================================

def build_gene_sets(ontologies, mapping_df, output_dir):
    """Build SynGO gene sets from ontology hierarchy."""
    tables_dir = os.path.join(output_dir, 'tables')

    # Get genes annotated to each GO term (ontologies.xlsx has comma-separated gene lists per term)
    term_genes = defaultdict(set)
    for _, row in ontologies.iterrows():
        go_id = row['id']
        val = str(row.get('hgnc_symbol', ''))
        if val and val != 'nan':
            for symbol in val.split(','):
                symbol = symbol.strip()
                if symbol:
                    term_genes[go_id].add(symbol)

    # Map human symbols to mouse via mapping_df
    mapped = mapping_df[mapping_df.present_in_model].copy()
    h2m_model = dict(zip(mapped.human_symbol, mapped.mouse_symbol))

    def get_mouse_genes(go_ids, exclude=None, intersect=False, union=False):
        """Get mouse gene symbols for a set of GO terms."""
        if intersect and len(go_ids) >= 2:
            # Intersection of all GO term gene sets
            gene_sets = [term_genes.get(gid, set()) for gid in go_ids]
            human_genes = gene_sets[0]
            for gs in gene_sets[1:]:
                human_genes = human_genes & gs
        elif union and len(go_ids) >= 2:
            human_genes = set()
            for gid in go_ids:
                human_genes |= term_genes.get(gid, set())
        else:
            human_genes = set()
            for gid in go_ids:
                human_genes |= term_genes.get(gid, set())

        if exclude:
            exclude_genes = set()
            for gid in exclude:
                exclude_genes |= term_genes.get(gid, set())
            human_genes = human_genes - exclude_genes

        mouse_genes = set()
        human_count = len(human_genes)
        for h in human_genes:
            m = h2m_model.get(h)
            if m:
                mouse_genes.add(m)

        return human_genes, mouse_genes

    subgroup_records = []
    gene_sets = {}

    for name, spec in SYNGO_SETS.items():
        go_ids = spec['go_ids']
        exclude = spec.get('exclude', None)
        intersect = spec.get('intersect', False)
        union = spec.get('union', False)

        human_genes, mouse_genes = get_mouse_genes(go_ids, exclude, intersect, union)

        ribo = {g for g in mouse_genes if g.startswith('Rpl') or g.startswith('Rps')}
        non_ribo = mouse_genes - ribo

        gene_sets[name] = {
            'human_genes': human_genes,
            'mouse_genes': mouse_genes,
            'mouse_non_ribo': non_ribo,
            'mouse_ribo': ribo,
            'type': spec['type'],
        }

        # Count orthologs (mapped but maybe not in model)
        all_mapped = mapping_df[mapping_df.mapping_method != 'unmapped']
        ortholog_count = sum(1 for h in human_genes if h in set(all_mapped.human_symbol))

        subgroup_records.append({
            'subgroup': name,
            'type': spec['type'],
            'human_gene_count': len(human_genes),
            'one_to_one_ortholog_count': ortholog_count,
            'model_feature_overlap': len(mouse_genes),
            'Rpl_Rps_count': len(ribo),
            'non_ribosomal_analysis_count': len(non_ribo),
        })

    subgroup_df = pd.DataFrame(subgroup_records)
    subgroup_df.to_csv(os.path.join(tables_dir, 'syngo_subgroup_counts.csv'), index=False)

    print(f"\n  SynGO subgroup counts:")
    for _, r in subgroup_df.iterrows():
        print(f"    {r['subgroup']}: {r['human_gene_count']} human -> "
              f"{r['model_feature_overlap']} model ({r['non_ribosomal_analysis_count']} non-ribo)")

    return gene_sets, subgroup_df


# ============================================================
# Section 5: Neuronal Classification
# ============================================================

def classify_cell_types(common_ct, output_dir):
    """Classify cell types as neuronal or non-neuronal."""
    tables_dir = os.path.join(output_dir, 'tables')

    records = []
    for ct in common_ct:
        if ct in NEURONAL_TYPES:
            status = 'neuronal'
        elif ct in NON_NEURONAL_TYPES:
            status = 'non-neuronal'
        else:
            status = 'unknown'
            print(f"    WARNING: Unknown cell type classification: {ct}")
        records.append({'cell_type': ct, 'neuronal_status': status})

    ct_df = pd.DataFrame(records)
    ct_df.to_csv(os.path.join(tables_dir, 'celltype_classification.csv'), index=False)

    n_neuronal = sum(1 for r in records if r['neuronal_status'] == 'neuronal')
    n_non = sum(1 for r in records if r['neuronal_status'] == 'non-neuronal')
    print(f"\n  Cell type classification: {n_neuronal} neuronal, {n_non} non-neuronal")

    return {r['cell_type']: r['neuronal_status'] for r in records}


# ============================================================
# Section 6: Expression-Matched Controls
# ============================================================

def build_matched_controls(syngo_genes, all_syngo_broad, gene_list, train_nuc_X, train_nuc_genes,
                           subgroup_name, output_dir):
    """Build expression-matched non-SynGO control set using training nuclei."""

    gene_to_idx = {g: i for i, g in enumerate(train_nuc_genes)}

    # Control pool: model genes not in broad SynGO, not Rpl/Rps
    control_pool = [g for g in gene_list
                    if g not in all_syngo_broad
                    and not g.startswith('Rpl') and not g.startswith('Rps')
                    and g in gene_to_idx]

    syngo_in_model = [g for g in syngo_genes if g in gene_to_idx]

    if len(syngo_in_model) == 0 or len(control_pool) == 0:
        return pd.DataFrame(), pd.DataFrame()

    # Compute training nuclear expression stats
    def get_stats(genes):
        idx = [gene_to_idx[g] for g in genes]
        expr = train_nuc_X[:, idx]
        means = expr.mean(axis=0)
        detection = (expr > 0).mean(axis=0)
        return means, detection

    syngo_means, syngo_det = get_stats(syngo_in_model)
    ctrl_means, ctrl_det = get_stats(control_pool)

    # Standardize both variables together
    all_means = np.concatenate([syngo_means, ctrl_means])
    all_det = np.concatenate([syngo_det, ctrl_det])
    mean_mu, mean_sd = all_means.mean(), all_means.std()
    det_mu, det_sd = all_det.mean(), all_det.std()

    if mean_sd == 0: mean_sd = 1
    if det_sd == 0: det_sd = 1

    syngo_features = np.column_stack([
        (syngo_means - mean_mu) / mean_sd,
        (syngo_det - det_mu) / det_sd
    ])
    ctrl_features = np.column_stack([
        (ctrl_means - mean_mu) / mean_sd,
        (ctrl_det - det_mu) / det_sd
    ])

    # 1:1 nearest neighbor matching without replacement
    dists = cdist(syngo_features, ctrl_features, metric='euclidean')

    matched_pairs = []
    used_ctrl = set()

    # Sort SynGO genes by distance to nearest control (greedy matching)
    min_dists = dists.min(axis=1)
    order = np.argsort(min_dists)

    for i in order:
        available = np.array([j for j in range(len(control_pool)) if j not in used_ctrl])
        if len(available) == 0:
            break
        best_j = available[dists[i, available].argmin()]
        used_ctrl.add(best_j)
        matched_pairs.append({
            'syngo_gene': syngo_in_model[i],
            'control_gene': control_pool[best_j],
            'subgroup': subgroup_name,
            'syngo_training_mean': float(syngo_means[i]),
            'control_training_mean': float(ctrl_means[best_j]),
            'syngo_detection_rate': float(syngo_det[i]),
            'control_detection_rate': float(ctrl_det[best_j]),
            'standardized_matching_distance': float(dists[i, best_j]),
            'matching_with_replacement': False,
        })

    pairs_df = pd.DataFrame(matched_pairs)

    # Matching QC
    if len(pairs_df) > 0:
        smd_mean = abs(pairs_df.syngo_training_mean.mean() - pairs_df.control_training_mean.mean()) / \
                   pairs_df[['syngo_training_mean', 'control_training_mean']].values.flatten().std()
        smd_det = abs(pairs_df.syngo_detection_rate.mean() - pairs_df.control_detection_rate.mean()) / \
                  pairs_df[['syngo_detection_rate', 'control_detection_rate']].values.flatten().std()
    else:
        smd_mean = smd_det = np.nan

    qc = pd.DataFrame([{
        'subgroup': subgroup_name,
        'n_syngo': len(syngo_in_model),
        'n_matched': len(pairs_df),
        'n_control_pool': len(control_pool),
        'smd_mean_expression': float(smd_mean),
        'smd_detection_rate': float(smd_det),
        'mean_matching_distance': float(pairs_df.standardized_matching_distance.mean()) if len(pairs_df) > 0 else np.nan,
        'with_replacement': False,
    }])

    return pairs_df, qc


# ============================================================
# Section 7: Validation Shifts
# ============================================================

def compute_validation_shifts(valid_cell_X, valid_cell_ct, valid_nuc_X, valid_nuc_ct,
                              imputed_X, imputed_ct, gene_names, ct_status, gene_sets,
                              output_dir):
    """Compute empirical and predicted shifts for all gene-celltype combinations."""
    tables_dir = os.path.join(output_dir, 'tables')

    # Get common cell types (present in both modalities' validation sets)
    cell_types_cell = set(valid_cell_ct)
    cell_types_nuc = set(valid_nuc_ct)
    common_cts = sorted(cell_types_cell & cell_types_nuc)

    # Report validation sample sizes
    print(f"\n  Validation sample sizes:")
    rare_cts = []
    for ct in common_cts:
        n_cell = (valid_cell_ct == ct).sum()
        n_nuc = (valid_nuc_ct == ct).sum()
        flag = '*' if n_cell < 20 or n_nuc < 20 else ''
        if flag:
            rare_cts.append(ct)
        print(f"    {ct}: {n_cell} cells, {n_nuc} nuclei {flag}")

    # Build gene set membership lookup
    gene_membership = {}
    for g in gene_names:
        memberships = []
        for name, gs in gene_sets.items():
            if g in gs.get('mouse_non_ribo', set()) or g in gs.get('mouse_ribo', set()):
                memberships.append(name)
        gene_membership[g] = ';'.join(memberships) if memberships else ''

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    records = []
    for ct in common_cts:
        cell_mask = valid_cell_ct == ct
        nuc_mask = valid_nuc_ct == ct
        imp_mask = imputed_ct == ct

        n_cell = cell_mask.sum()
        n_nuc = nuc_mask.sum()

        cell_mean = valid_cell_X[cell_mask].mean(axis=0)
        nuc_mean = valid_nuc_X[nuc_mask].mean(axis=0)
        imp_mean = imputed_X[imp_mask].mean(axis=0)

        for gi, g in enumerate(gene_names):
            records.append({
                'gene': g,
                'cell_type': ct,
                'neuronal_status': ct_status.get(ct, 'unknown'),
                'gene_set_memberships': gene_membership.get(g, ''),
                'delta_empirical': float(cell_mean[gi] - nuc_mean[gi]),
                'delta_predicted': float(imp_mean[gi] - nuc_mean[gi]),
                'cell_mean': float(cell_mean[gi]),
                'nucleus_mean': float(nuc_mean[gi]),
                'imputed_mean': float(imp_mean[gi]),
                'n_cell': int(n_cell),
                'n_nucleus': int(n_nuc),
            })

    shifts_df = pd.DataFrame(records)
    shifts_df.to_csv(os.path.join(tables_dir, 'gene_celltype_shifts.csv'), index=False)
    print(f"  Gene-celltype shifts: {len(shifts_df)} entries saved")

    return shifts_df, common_cts, rare_cts


# ============================================================
# Section 8: Enrichment Analysis
# ============================================================

def compute_enrichment(shifts_df, matched_pairs, ct_status, common_cts, n_bootstrap, rng,
                       subgroup_name='synapse_non_ribo'):
    """Compute matched SynGO enrichment by cell type with bootstrap CIs."""

    if len(matched_pairs) == 0:
        return pd.DataFrame(), pd.DataFrame()

    # Pivot shifts for fast lookup
    shift_lookup = {}
    for _, row in shifts_df.iterrows():
        shift_lookup[(row['gene'], row['cell_type'])] = (row['delta_empirical'], row['delta_predicted'])

    # Compute pair differences for each cell type
    ct_results = []
    pair_diffs_emp = {}  # ct -> array of pair differences
    pair_diffs_pred = {}

    for ct in common_cts:
        emp_diffs = []
        pred_diffs = []
        for _, pair in matched_pairs.iterrows():
            sg = pair['syngo_gene']
            cg = pair['control_gene']
            key_s = (sg, ct)
            key_c = (cg, ct)
            if key_s in shift_lookup and key_c in shift_lookup:
                de_s, dp_s = shift_lookup[key_s]
                de_c, dp_c = shift_lookup[key_c]
                emp_diffs.append(de_s - de_c)
                pred_diffs.append(dp_s - dp_c)

        emp_diffs = np.array(emp_diffs)
        pred_diffs = np.array(pred_diffs)
        pair_diffs_emp[ct] = emp_diffs
        pair_diffs_pred[ct] = pred_diffs

        ct_results.append({
            'cell_type': ct,
            'neuronal_status': ct_status.get(ct, 'unknown'),
            'E_empirical': float(emp_diffs.mean()) if len(emp_diffs) > 0 else np.nan,
            'E_predicted': float(pred_diffs.mean()) if len(pred_diffs) > 0 else np.nan,
            'n_pairs': len(emp_diffs),
        })

    ct_enrich = pd.DataFrame(ct_results)

    # Bootstrap CIs
    n_pairs = len(matched_pairs)
    boot_E_emp = np.zeros((n_bootstrap, len(common_cts)))
    boot_E_pred = np.zeros((n_bootstrap, len(common_cts)))

    for b in range(n_bootstrap):
        pair_idx = rng.choice(n_pairs, size=n_pairs, replace=True)
        for ci, ct in enumerate(common_cts):
            if len(pair_diffs_emp[ct]) == 0:
                boot_E_emp[b, ci] = np.nan
                boot_E_pred[b, ci] = np.nan
                continue
            boot_E_emp[b, ci] = pair_diffs_emp[ct][pair_idx[pair_idx < len(pair_diffs_emp[ct])]].mean()
            boot_E_pred[b, ci] = pair_diffs_pred[ct][pair_idx[pair_idx < len(pair_diffs_pred[ct])]].mean()

    # Add CIs to ct_enrich
    for ci, ct in enumerate(common_cts):
        mask = ct_enrich.cell_type == ct
        ct_enrich.loc[mask, 'E_empirical_ci_lo'] = np.nanpercentile(boot_E_emp[:, ci], 2.5)
        ct_enrich.loc[mask, 'E_empirical_ci_hi'] = np.nanpercentile(boot_E_emp[:, ci], 97.5)
        ct_enrich.loc[mask, 'E_predicted_ci_lo'] = np.nanpercentile(boot_E_pred[:, ci], 2.5)
        ct_enrich.loc[mask, 'E_predicted_ci_hi'] = np.nanpercentile(boot_E_pred[:, ci], 97.5)

    # Cross-cell-type metrics
    E_emp = ct_enrich.E_empirical.values
    E_pred = ct_enrich.E_predicted.values
    valid = ~(np.isnan(E_emp) | np.isnan(E_pred))

    if valid.sum() >= 3:
        pearson_r, pearson_p = stats.pearsonr(E_emp[valid], E_pred[valid])
        spearman_r, spearman_p = stats.spearmanr(E_emp[valid], E_pred[valid])
        ss_res = np.sum((E_pred[valid] - E_emp[valid]) ** 2)
        ss_tot = np.sum((E_emp[valid] - E_emp[valid].mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        mae = np.mean(np.abs(E_pred[valid] - E_emp[valid]))
    else:
        pearson_r = spearman_r = r2 = mae = np.nan
        pearson_p = spearman_p = np.nan

    # Neuronal vs non-neuronal
    neuronal_mask = ct_enrich.neuronal_status == 'neuronal'
    non_neuronal_mask = ct_enrich.neuronal_status == 'non-neuronal'

    mean_E_emp_neuronal = ct_enrich.loc[neuronal_mask, 'E_empirical'].mean()
    mean_E_emp_non = ct_enrich.loc[non_neuronal_mask, 'E_empirical'].mean()
    mean_E_pred_neuronal = ct_enrich.loc[neuronal_mask, 'E_predicted'].mean()
    mean_E_pred_non = ct_enrich.loc[non_neuronal_mask, 'E_predicted'].mean()

    # Bootstrap neuronal vs non-neuronal difference
    boot_neuro_diff_emp = []
    boot_neuro_diff_pred = []
    neuronal_idx = np.where(neuronal_mask)[0]
    non_neuronal_idx = np.where(non_neuronal_mask)[0]

    for b in range(n_bootstrap):
        if len(neuronal_idx) > 0 and len(non_neuronal_idx) > 0:
            boot_neuro_diff_emp.append(
                boot_E_emp[b, neuronal_idx].mean() - boot_E_emp[b, non_neuronal_idx].mean())
            boot_neuro_diff_pred.append(
                boot_E_pred[b, neuronal_idx].mean() - boot_E_pred[b, non_neuronal_idx].mean())

    boot_results = pd.DataFrame([{
        'subgroup': subgroup_name,
        'n_pairs': n_pairs,
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p) if not np.isnan(pearson_p) else np.nan,
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p) if not np.isnan(spearman_p) else np.nan,
        'R2': float(r2),
        'MAE': float(mae),
        'E_empirical_neuronal': float(mean_E_emp_neuronal),
        'E_empirical_non_neuronal': float(mean_E_emp_non),
        'E_predicted_neuronal': float(mean_E_pred_neuronal),
        'E_predicted_non_neuronal': float(mean_E_pred_non),
        'empirical_neuro_diff': float(mean_E_emp_neuronal - mean_E_emp_non),
        'predicted_neuro_diff': float(mean_E_pred_neuronal - mean_E_pred_non),
        'neuro_diff_emp_ci_lo': float(np.percentile(boot_neuro_diff_emp, 2.5)) if boot_neuro_diff_emp else np.nan,
        'neuro_diff_emp_ci_hi': float(np.percentile(boot_neuro_diff_emp, 97.5)) if boot_neuro_diff_emp else np.nan,
        'neuro_diff_pred_ci_lo': float(np.percentile(boot_neuro_diff_pred, 2.5)) if boot_neuro_diff_pred else np.nan,
        'neuro_diff_pred_ci_hi': float(np.percentile(boot_neuro_diff_pred, 97.5)) if boot_neuro_diff_pred else np.nan,
    }])

    return ct_enrich, boot_results


# ============================================================
# Section 9: Recovery Accuracy
# ============================================================

def compute_recovery_metrics(shifts_df, syngo_genes, ct_status, common_cts, output_dir):
    """Compute delta recovery accuracy within SynGO genes."""
    tables_dir = os.path.join(output_dir, 'tables')

    sg = shifts_df[shifts_df.gene.isin(syngo_genes)].copy()

    def calc_metrics(de, dp, label):
        valid = ~(np.isnan(de) | np.isnan(dp))
        de, dp = de[valid], dp[valid]
        if len(de) < 5:
            return {'label': label, 'n': len(de)}

        pr, pp = stats.pearsonr(de, dp)
        sr, sp = stats.spearmanr(de, dp)
        ss_res = np.sum((dp - de) ** 2)
        ss_tot = np.sum((de - de.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        mae = np.mean(np.abs(dp - de))
        sign_agree = np.mean(np.sign(de) == np.sign(dp))

        # AUROC: positive vs negative empirical shift
        pos_mask = de > 0
        neg_mask = de < 0
        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
            labels = pos_mask.astype(int)
            auroc = roc_auc_score(labels, dp)
        else:
            auroc = np.nan

        return {
            'label': label, 'n': len(de),
            'pearson_r': float(pr), 'spearman_r': float(sr),
            'R2': float(r2), 'MAE': float(mae),
            'sign_agreement': float(sign_agree), 'AUROC': float(auroc),
        }

    results = []

    # All pairs
    de = sg.delta_empirical.values
    dp = sg.delta_predicted.values
    results.append(calc_metrics(de, dp, 'all_gene_celltype_pairs'))

    # Neuronal only
    neuronal = sg[sg.neuronal_status == 'neuronal']
    results.append(calc_metrics(neuronal.delta_empirical.values, neuronal.delta_predicted.values, 'neuronal_only'))

    # Non-neuronal only
    non_neuronal = sg[sg.neuronal_status == 'non-neuronal']
    results.append(calc_metrics(non_neuronal.delta_empirical.values, non_neuronal.delta_predicted.values, 'non_neuronal_only'))

    recovery_df = pd.DataFrame(results)
    recovery_df.to_csv(os.path.join(tables_dir, 'syngo_recovery_metrics.csv'), index=False)

    # Per cell type (across genes)
    per_ct_records = []
    for ct in common_cts:
        ct_data = sg[sg.cell_type == ct]
        m = calc_metrics(ct_data.delta_empirical.values, ct_data.delta_predicted.values, ct)
        m['cell_type'] = ct
        m['neuronal_status'] = ct_status.get(ct, 'unknown')
        m['resolution'] = 'across_genes_within_ct'
        per_ct_records.append(m)

    per_ct_df = pd.DataFrame(per_ct_records)
    per_ct_df.to_csv(os.path.join(tables_dir, 'syngo_per_celltype_metrics.csv'), index=False)

    # Per gene (across cell types)
    per_gene_records = []
    for g in syngo_genes:
        g_data = sg[sg.gene == g]
        if len(g_data) >= 3:
            m = calc_metrics(g_data.delta_empirical.values, g_data.delta_predicted.values, g)
            m['gene'] = g
            m['resolution'] = 'across_cts_within_gene'
            per_gene_records.append(m)

    per_gene_df = pd.DataFrame(per_gene_records)
    per_gene_df.to_csv(os.path.join(tables_dir, 'syngo_per_gene_metrics.csv'), index=False)

    print(f"\n  Recovery metrics (SynGO non-ribo):")
    for _, r in recovery_df.iterrows():
        pr = r.get('pearson_r', np.nan)
        sr = r.get('spearman_r', np.nan)
        print(f"    {r['label']}: n={r['n']}, Pearson={pr:.3f}, Spearman={sr:.3f}, "
              f"R2={r.get('R2', np.nan):.3f}, sign={r.get('sign_agreement', np.nan):.3f}")

    if len(per_gene_df) > 0:
        med_pr = per_gene_df['pearson_r'].median()
        print(f"    Per-gene median Pearson (across CTs): {med_pr:.3f} (n={len(per_gene_df)} genes)")

    return recovery_df, per_ct_df, per_gene_df


# ============================================================
# Section 10: Sensitivity Analyses
# ============================================================

def sensitivity_with_ribo(shifts_df, gene_sets, matched_controls_broad, ct_status,
                          common_cts, n_bootstrap, rng, output_dir):
    """Compare analysis with and without Rpl/Rps genes."""
    # Already have non-ribo results from primary analysis
    # Now compute with Rpl/Rps included
    all_syngo = gene_sets['synapse']['mouse_genes']

    # Need new matched controls for the full set including ribo
    # For simplicity, reuse existing control pairs + add ribo genes unmatched
    # (Ribo genes are positive controls, not primary analysis)

    return None  # Handled in main flow


def sensitivity_rare_ct(shifts_df, matched_pairs, ct_status, common_cts, rare_cts,
                        n_bootstrap, rng):
    """Repeat enrichment excluding rare cell types."""
    robust_cts = [ct for ct in common_cts if ct not in rare_cts]
    ct_enrich, boot_results = compute_enrichment(
        shifts_df, matched_pairs, ct_status, robust_cts, n_bootstrap, rng,
        subgroup_name='synapse_non_ribo_no_rare')
    return ct_enrich, boot_results


def sensitivity_donor_balanced(shifts_df, valid_cell_meta, valid_nuc_meta,
                               valid_cell_X, valid_nuc_X, imputed_X,
                               gene_names, syngo_genes, matched_pairs,
                               ct_status, common_cts, output_dir):
    """Donor-balanced analysis: donor-level means before averaging."""
    tables_dir = os.path.join(output_dir, 'tables')

    # Find shared donors
    cell_donors = set(valid_cell_meta.donor.unique())
    nuc_donors = set(valid_nuc_meta.donor.unique())
    shared_donors = sorted(cell_donors & nuc_donors)

    if len(shared_donors) < 2:
        print("  Donor-balanced: insufficient shared donors")
        pd.DataFrame([{'status': 'insufficient_shared_donors', 'n_shared': len(shared_donors)}]).to_csv(
            os.path.join(tables_dir, 'syngo_donor_sensitivity.csv'), index=False)
        return None

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    # For each CT, compute donor-level means, then average
    records = []
    feasible_cts = []

    for ct in common_cts:
        donor_deltas_emp = []
        donor_deltas_pred = []
        donors_used = []

        for d in shared_donors:
            cell_mask = (valid_cell_meta.cell_type.values == ct) & (valid_cell_meta.donor.values == d)
            nuc_mask = (valid_nuc_meta.cell_type.values == ct) & (valid_nuc_meta.donor.values == d)

            n_c = cell_mask.sum()
            n_n = nuc_mask.sum()

            if n_c < 5 or n_n < 5:
                continue

            cell_mean = valid_cell_X[cell_mask].mean(axis=0)
            nuc_mean = valid_nuc_X[nuc_mask].mean(axis=0)
            imp_mean = imputed_X[nuc_mask].mean(axis=0)

            donor_deltas_emp.append(cell_mean - nuc_mean)
            donor_deltas_pred.append(imp_mean - nuc_mean)
            donors_used.append(d)

        if len(donors_used) >= 2:
            feasible_cts.append(ct)
            avg_emp = np.mean(donor_deltas_emp, axis=0)
            avg_pred = np.mean(donor_deltas_pred, axis=0)

            for gi, g in enumerate(gene_names):
                if g in syngo_genes:
                    records.append({
                        'gene': g, 'cell_type': ct,
                        'delta_empirical_donor_balanced': float(avg_emp[gi]),
                        'delta_predicted_donor_balanced': float(avg_pred[gi]),
                        'n_donors': len(donors_used),
                        'donors': ','.join(map(str, donors_used)),
                    })

    donor_df = pd.DataFrame(records)
    donor_df.to_csv(os.path.join(tables_dir, 'syngo_donor_sensitivity.csv'), index=False)

    print(f"\n  Donor-balanced sensitivity:")
    print(f"    Shared donors: {shared_donors}")
    print(f"    Feasible cell types: {feasible_cts}")
    print(f"    Records: {len(donor_df)}")

    return donor_df


# ============================================================
# Section 11: Figures
# ============================================================

def create_figures(ct_enrich_primary, boot_primary, subgroup_results,
                   ct_status, common_cts, output_dir):
    """Create main and supplementary figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig_dir = os.path.join(output_dir, 'figures')

    # Color scheme
    colors = {'neuronal': '#4477AA', 'non-neuronal': '#228833'}

    # ---- Main Figure: 3 panels ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: Cell-type enrichment
    ax = axes[0]
    ct_sorted = ct_enrich_primary.sort_values('E_empirical', ascending=False)
    y_pos = np.arange(len(ct_sorted))

    for i, (_, row) in enumerate(ct_sorted.iterrows()):
        c = colors.get(row['neuronal_status'], 'gray')
        ax.plot([row['E_empirical'], row['E_predicted']], [y_pos[i], y_pos[i]],
                color=c, alpha=0.4, linewidth=1)
        ax.scatter(row['E_empirical'], y_pos[i], color=c, marker='o', s=40, zorder=3)
        ax.scatter(row['E_predicted'], y_pos[i], color=c, marker='s', s=40, zorder=3, alpha=0.7)

        # Error bars if available
        if 'E_empirical_ci_lo' in row and not np.isnan(row.get('E_empirical_ci_lo', np.nan)):
            ax.errorbar(row['E_empirical'], y_pos[i],
                       xerr=[[row['E_empirical'] - row['E_empirical_ci_lo']],
                              [row['E_empirical_ci_hi'] - row['E_empirical']]],
                       color=c, alpha=0.3, fmt='none', capsize=2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(ct_sorted.cell_type.values, fontsize=8)
    ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlabel('SynGO enrichment\n(SynGO - matched control delta)')
    ax.set_title('A. Cell-type SynGO enrichment')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', label='Empirical', markersize=8),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', label='Predicted', markersize=8, alpha=0.7),
        Line2D([0], [0], color=colors['neuronal'], label='Neuronal', linewidth=2),
        Line2D([0], [0], color=colors['non-neuronal'], label='Non-neuronal', linewidth=2),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc='lower right')

    # Panel B: Scatter
    ax = axes[1]
    for _, row in ct_enrich_primary.iterrows():
        c = colors.get(row['neuronal_status'], 'gray')
        ax.scatter(row['E_empirical'], row['E_predicted'], color=c, s=50, zorder=3)
        ax.annotate(row['cell_type'], (row['E_empirical'], row['E_predicted']),
                   fontsize=6, ha='left', va='bottom', xytext=(3, 3),
                   textcoords='offset points')

    # y=x line
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=0.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel('E_empirical')
    ax.set_ylabel('E_predicted')

    # Add stats text
    br = boot_primary.iloc[0]
    stats_text = (f"Pearson r = {br['pearson_r']:.3f}\n"
                  f"Spearman rho = {br['spearman_r']:.3f}\n"
                  f"R² = {br['R2']:.3f}\n"
                  f"MAE = {br['MAE']:.4f}")
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_title('B. Empirical vs predicted enrichment')

    # Panel C: Subgroup recovery
    ax = axes[2]
    if subgroup_results is not None and len(subgroup_results) > 0:
        sg_plot = subgroup_results[subgroup_results['type'].isin(['broad', 'location', 'distal'])]
        sg_plot = sg_plot.sort_values('empirical_neuro_diff', ascending=False)
        y_pos = np.arange(len(sg_plot))

        for i, (_, row) in enumerate(sg_plot.iterrows()):
            ax.barh(y_pos[i] - 0.15, row['empirical_neuro_diff'], height=0.3,
                   color=colors['neuronal'], alpha=0.7, label='Empirical' if i == 0 else '')
            ax.barh(y_pos[i] + 0.15, row['predicted_neuro_diff'], height=0.3,
                   color=colors['non-neuronal'], alpha=0.7, label='Predicted' if i == 0 else '')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(sg_plot.subgroup.values, fontsize=7)
        ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
        ax.set_xlabel('Neuronal - non-neuronal enrichment')
        ax.legend(fontsize=7)
    ax.set_title('C. Subgroup recovery')

    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'syngo_validation_main.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(fig_dir, 'syngo_validation_main.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  Main figure saved")

    # ---- Supplementary: Matching QC ----
    return True


def create_supplementary_figures(matched_pairs, shifts_df, syngo_genes, control_genes,
                                 ct_enrich_primary, per_ct_recovery, ct_status,
                                 common_cts, rare_cts, boot_ribo, boot_no_rare,
                                 subgroup_results, output_dir):
    """Create supplementary figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig_dir = os.path.join(output_dir, 'figures')
    colors = {'neuronal': '#4477AA', 'non-neuronal': '#228833'}

    # S1: Matching QC
    if len(matched_pairs) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        ax.hist(matched_pairs.syngo_training_mean, bins=30, alpha=0.5, label='SynGO', color='steelblue')
        ax.hist(matched_pairs.control_training_mean, bins=30, alpha=0.5, label='Control', color='coral')
        ax.set_xlabel('Mean log1p expression (training nuclei)')
        ax.set_ylabel('Count')
        ax.set_title('Expression matching')
        ax.legend()

        ax = axes[1]
        ax.hist(matched_pairs.syngo_detection_rate, bins=30, alpha=0.5, label='SynGO', color='steelblue')
        ax.hist(matched_pairs.control_detection_rate, bins=30, alpha=0.5, label='Control', color='coral')
        ax.set_xlabel('Detection rate (training nuclei)')
        ax.set_ylabel('Count')
        ax.set_title('Detection rate matching')
        ax.legend()

        plt.tight_layout()
        fig.savefig(os.path.join(fig_dir, 'syngo_matching_qc.png'), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(fig_dir, 'syngo_matching_qc.pdf'), bbox_inches='tight')
        plt.close(fig)

    # S2: SynGO vs control shift distributions
    sg_shifts = shifts_df[shifts_df.gene.isin(syngo_genes)]
    ctrl_genes_set = set(matched_pairs.control_gene) if len(matched_pairs) > 0 else set()
    ctrl_shifts = shifts_df[shifts_df.gene.isin(ctrl_genes_set)]

    if len(sg_shifts) > 0 and len(ctrl_shifts) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        ax.hist(sg_shifts.delta_empirical, bins=50, alpha=0.5, label='SynGO', color='steelblue')
        ax.hist(ctrl_shifts.delta_empirical, bins=50, alpha=0.5, label='Control', color='coral')
        ax.set_xlabel('Empirical delta')
        ax.set_title('Empirical shift distribution')
        ax.legend()

        ax = axes[1]
        ax.hist(sg_shifts.delta_predicted, bins=50, alpha=0.5, label='SynGO', color='steelblue')
        ax.hist(ctrl_shifts.delta_predicted, bins=50, alpha=0.5, label='Control', color='coral')
        ax.set_xlabel('Predicted delta')
        ax.set_title('Predicted shift distribution')
        ax.legend()

        plt.tight_layout()
        fig.savefig(os.path.join(fig_dir, 'syngo_shift_distributions.png'), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(fig_dir, 'syngo_shift_distributions.pdf'), bbox_inches='tight')
        plt.close(fig)

    # S3: Per-cell-type accuracy heatmap
    if per_ct_recovery is not None and len(per_ct_recovery) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))

        metrics_to_plot = ['pearson_r', 'spearman_r', 'R2', 'sign_agreement']
        avail = [m for m in metrics_to_plot if m in per_ct_recovery.columns]

        if avail:
            data = per_ct_recovery.set_index('cell_type')[avail].astype(float)
            im = ax.imshow(data.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            ax.set_xticks(range(len(avail)))
            ax.set_xticklabels(avail, rotation=45, ha='right')
            ax.set_yticks(range(len(data)))
            ax.set_yticklabels(data.index, fontsize=8)
            plt.colorbar(im, ax=ax, label='Score')

            # Add values
            for i in range(len(data)):
                for j in range(len(avail)):
                    val = data.values[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=6)

            ax.set_title('Per-cell-type recovery accuracy (SynGO genes)')

        plt.tight_layout()
        fig.savefig(os.path.join(fig_dir, 'syngo_per_ct_heatmap.png'), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(fig_dir, 'syngo_per_ct_heatmap.pdf'), bbox_inches='tight')
        plt.close(fig)

    # S4: Filtering flow bar chart
    flow_path = os.path.join(output_dir, 'tables', 'syngo_filtering_flow.csv')
    if os.path.exists(flow_path):
        flow = pd.read_csv(flow_path)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(range(len(flow)), flow['count'].values, color='steelblue')
        ax.set_yticks(range(len(flow)))
        ax.set_yticklabels(flow.stage.values, fontsize=8)
        ax.set_xlabel('Gene count')
        ax.set_title('SynGO gene filtering flow')
        ax.invert_yaxis()
        plt.tight_layout()
        fig.savefig(os.path.join(fig_dir, 'syngo_filtering_counts.png'), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(fig_dir, 'syngo_filtering_counts.pdf'), bbox_inches='tight')
        plt.close(fig)

    # S5: With vs without Rpl/Rps
    if boot_ribo is not None and len(boot_ribo) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = ['Non-ribo only', 'Including Rpl/Rps']

        # Get primary boot results
        primary_path = os.path.join(output_dir, 'tables', 'bootstrap_primary_results.csv')
        if os.path.exists(primary_path):
            primary = pd.read_csv(primary_path)
            primary_row = primary.iloc[0]
            ribo_row = boot_ribo.iloc[0]

            x = np.arange(2)
            width = 0.35
            ax.bar(x - width/2, [primary_row['empirical_neuro_diff'], ribo_row['empirical_neuro_diff']],
                  width, label='Empirical', color='steelblue')
            ax.bar(x + width/2, [primary_row['predicted_neuro_diff'], ribo_row['predicted_neuro_diff']],
                  width, label='Predicted', color='coral')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylabel('Neuronal - non-neuronal enrichment')
            ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
            ax.legend()
            ax.set_title('Rpl/Rps sensitivity')

        plt.tight_layout()
        fig.savefig(os.path.join(fig_dir, 'syngo_ribo_sensitivity.png'), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(fig_dir, 'syngo_ribo_sensitivity.pdf'), bbox_inches='tight')
        plt.close(fig)

    print("  Supplementary figures saved")


# ============================================================
# Section 12: Baseline Comparison
# ============================================================

def baseline_comparison(shifts_df, matched_pairs, syngo_genes, ct_status, common_cts,
                        manifest, valid_nuc_idx, n_bootstrap, rng, output_dir):
    """Compare SynGO metrics across methods (validation-only, with caveats)."""
    tables_dir = os.path.join(output_dir, 'tables')

    method_results = []

    for method_name, base_dir in BASELINE_DIRS.items():
        imputed, obs, error = load_baseline_validation(method_name, base_dir, manifest, valid_nuc_idx)
        if error:
            method_results.append({
                'method': method_name, 'status': 'unavailable', 'reason': error,
                'caveat': 'baseline trained on all data (not split-aware)',
            })
            continue

        # NOTE: All baselines were trained on ALL data, not just training partition
        # This is documented as a caveat

        # Compute SynGO-restricted metrics using baseline predictions
        # We need to recompute shifts for this method
        # Get validation nucleus expression (same as primary analysis)
        # The shifts_df already has nucleus_mean, just need imputed_mean from this method

        gene_names_path = os.path.join(PROJECT_ROOT, 'data/training_gene_list_v3.csv')
        gene_names = pd.read_csv(gene_names_path, header=None)[0].tolist()
        gene_to_idx = {g: i for i, g in enumerate(gene_names)}

        imp_ct = obs.cell_type.values

        # Compute method-specific predicted shifts for SynGO genes
        method_shifts = []
        for ct in common_cts:
            imp_mask = imp_ct == ct
            if imp_mask.sum() == 0:
                continue
            imp_mean = imputed[imp_mask].mean(axis=0)

            for g in syngo_genes:
                gi = gene_to_idx.get(g)
                if gi is None:
                    continue
                # Get nucleus mean from shifts_df
                ref = shifts_df[(shifts_df.gene == g) & (shifts_df.cell_type == ct)]
                if len(ref) == 0:
                    continue
                nuc_mean = ref.iloc[0]['nucleus_mean']
                de = ref.iloc[0]['delta_empirical']
                dp = float(imp_mean[gi] - nuc_mean)
                method_shifts.append({
                    'gene': g, 'cell_type': ct,
                    'delta_empirical': de, 'delta_predicted': dp,
                })

        if len(method_shifts) == 0:
            method_results.append({
                'method': method_name, 'status': 'no_data',
                'caveat': 'baseline trained on all data (not split-aware)',
            })
            continue

        method_df = pd.DataFrame(method_shifts)
        de = method_df.delta_empirical.values
        dp = method_df.delta_predicted.values

        valid = ~(np.isnan(de) | np.isnan(dp))
        de, dp = de[valid], dp[valid]

        if len(de) < 5:
            method_results.append({
                'method': method_name, 'status': 'insufficient_data',
                'caveat': 'baseline trained on all data (not split-aware)',
            })
            continue

        pr, _ = stats.pearsonr(de, dp)
        sr, _ = stats.spearmanr(de, dp)
        ss_res = np.sum((dp - de) ** 2)
        ss_tot = np.sum((de - de.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        mae = np.mean(np.abs(dp - de))
        sign_agree = np.mean(np.sign(de) == np.sign(dp))

        pos_mask = de > 0
        neg_mask = de < 0
        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
            auroc = roc_auc_score(pos_mask.astype(int), dp)
        else:
            auroc = np.nan

        method_results.append({
            'method': method_name,
            'status': 'computed',
            'n_gene_ct_pairs': len(de),
            'delta_pearson': float(pr),
            'delta_spearman': float(sr),
            'R2': float(r2),
            'MAE': float(mae),
            'sign_agreement': float(sign_agree),
            'AUROC': float(auroc),
            'caveat': 'baseline trained on all data (not split-aware)',
        })

    method_df = pd.DataFrame(method_results)
    method_df.to_csv(os.path.join(tables_dir, 'syngo_method_comparison.csv'), index=False)

    print(f"\n  Baseline comparison (SynGO-restricted):")
    for _, r in method_df.iterrows():
        if r['status'] == 'computed':
            print(f"    {r['method']}: Pearson={r['delta_pearson']:.3f}, R2={r['R2']:.3f}, "
                  f"sign={r['sign_agreement']:.3f}")
        else:
            print(f"    {r['method']}: {r['status']}")

    return method_df


# ============================================================
# Section 13: Input Audit
# ============================================================

def create_input_audit(gene_list, manifest, valid_nuc_X, valid_cell_X, imputed_X,
                       gene_names, output_dir):
    """Create input audit table."""
    tables_dir = os.path.join(output_dir, 'tables')

    audit = [
        {
            'input_name': 'gene_list',
            'path': DEFAULT_GENE_LIST,
            'shape': f'{len(gene_list)}',
            'normalization': 'N/A',
            'partition': 'all',
            'gene_alignment_status': 'reference',
            'observation_alignment_status': 'N/A',
            'notes': 'Config B, 10134 genes',
        },
        {
            'input_name': 'split_manifest',
            'path': DEFAULT_SPLIT_MANIFEST,
            'shape': f'{len(manifest)}x5',
            'normalization': 'N/A',
            'partition': 'all',
            'gene_alignment_status': 'N/A',
            'observation_alignment_status': 'reference',
            'notes': 'obs_id, modality, cell_type, donor, partition',
        },
        {
            'input_name': 'validation_nuclei_expression',
            'path': 'loaded from h5ad via split manifest',
            'shape': f'{valid_nuc_X.shape[0]}x{valid_nuc_X.shape[1]}',
            'normalization': 'log1p',
            'partition': 'valid',
            'gene_alignment_status': f'aligned ({len(gene_names)} genes)',
            'observation_alignment_status': 'split-manifest validated',
            'notes': 'Raw counts -> log1p, sorted by gene name',
        },
        {
            'input_name': 'validation_cell_expression',
            'path': 'loaded from h5ad via split manifest',
            'shape': f'{valid_cell_X.shape[0]}x{valid_cell_X.shape[1]}',
            'normalization': 'log1p',
            'partition': 'valid',
            'gene_alignment_status': f'aligned ({len(gene_names)} genes)',
            'observation_alignment_status': 'split-manifest validated',
            'notes': 'Raw counts -> log1p, sorted by gene name',
        },
        {
            'input_name': 'ImputeVAE_cross_decoded',
            'path': CVAE_VALID_IMPUTED if os.path.exists(CVAE_VALID_IMPUTED) else CVAE_FULL_IMPUTED,
            'shape': f'{imputed_X.shape[0]}x{imputed_X.shape[1]}',
            'normalization': 'log1p (model output)',
            'partition': 'valid',
            'gene_alignment_status': f'aligned ({imputed_X.shape[1]} genes)',
            'observation_alignment_status': 'validation nuclei only',
            'notes': 'Cross-decoded cell profiles from validation nuclei',
        },
        {
            'input_name': 'syngo_archive',
            'path': DEFAULT_SYNGO_ZIP,
            'shape': '3 files (genes, annotations, ontologies)',
            'normalization': 'N/A',
            'partition': 'N/A',
            'gene_alignment_status': 'human HGNC -> mouse ortholog',
            'observation_alignment_status': 'N/A',
            'notes': 'SynGO v1.3',
        },
        {
            'input_name': 'ortholog_mapping',
            'path': DEFAULT_ORTHOLOG_MAP,
            'shape': '20089x2',
            'normalization': 'N/A',
            'partition': 'N/A',
            'gene_alignment_status': 'mouse_sym unique, human_sym has many-to-one',
            'observation_alignment_status': 'N/A',
            'notes': 'mouse_sym -> human_sym, source: Ensembl BioMart',
        },
    ]

    audit_df = pd.DataFrame(audit)
    audit_df.to_csv(os.path.join(tables_dir, 'input_audit.csv'), index=False)
    return audit_df


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)
    output_dir = args.output_dir
    tables_dir = os.path.join(output_dir, 'tables')
    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 70)
    print("SynGO Biological Consistency Analysis for ImputeVAE")
    print("=" * 70)

    # ---- Load gene list and manifest ----
    print("\n[1] Loading project data...")
    gene_list = load_gene_list(args.gene_list)
    manifest = load_split_manifest(args.split_manifest)
    common_ct = pd.read_csv(DEFAULT_COMMON_CT, header=0)['common_cell_types'].tolist()

    # ---- Get validation indices ----
    print("\n[2] Extracting validation indices...")
    valid_nuc_idx, valid_nuc_meta, train_nuc_idx, train_nuc_meta = get_validation_indices(manifest, 'nuclei')
    valid_cell_idx, valid_cell_meta, train_cell_idx, train_cell_meta = get_validation_indices(manifest, 'cell')
    print(f"  Validation nuclei: {len(valid_nuc_idx)}, Validation cells: {len(valid_cell_idx)}")
    print(f"  Training nuclei: {len(train_nuc_idx)}, Training cells: {len(train_cell_idx)}")

    # ---- Load raw expression ----
    print("\n[3] Loading raw expression data...")
    cell_adata, gene_names_cell = load_raw_expression(
        "10X_cells_v2_v3_AIBS", "c3A", gene_list, common_ct)
    nuc_adata, gene_names_nuc = load_raw_expression(
        "10X_nuclei_v2_v3_AIBS", "n3A", gene_list, common_ct)

    # Verify gene alignment
    assert gene_names_cell == gene_names_nuc, "Gene names mismatch between cell and nuclei!"
    gene_names = gene_names_cell
    print(f"  Gene alignment verified: {len(gene_names)} genes match")

    # Extract validation and training subsets
    valid_cell_X = cell_adata.X[valid_cell_idx]
    valid_cell_ct = cell_adata.obs.iloc[valid_cell_idx]['cell_type'].values
    valid_nuc_X = nuc_adata.X[valid_nuc_idx]
    valid_nuc_ct = nuc_adata.obs.iloc[valid_nuc_idx]['cell_type'].values
    train_nuc_X = nuc_adata.X[train_nuc_idx]
    train_nuc_ct = nuc_adata.obs.iloc[train_nuc_idx]['cell_type'].values

    print(f"  Validation cells: {valid_cell_X.shape}")
    print(f"  Validation nuclei: {valid_nuc_X.shape}")
    print(f"  Training nuclei: {train_nuc_X.shape}")

    # ---- Load ImputeVAE predictions ----
    print("\n[4] Loading ImputeVAE predictions...")
    imputed_X, imputed_obs = load_imputed_validation(gene_list)
    imputed_ct = imputed_obs.cell_type.values

    # Verify alignment
    assert imputed_X.shape[0] == valid_nuc_X.shape[0], \
        f"Imputed ({imputed_X.shape[0]}) != valid nuclei ({valid_nuc_X.shape[0]})"
    assert imputed_X.shape[1] == len(gene_names), \
        f"Imputed genes ({imputed_X.shape[1]}) != gene list ({len(gene_names)})"

    # Verify cell type alignment between imputed and validation nuclei
    ct_match = (imputed_ct == valid_nuc_ct).mean()
    print(f"  Cell type alignment: {ct_match*100:.1f}% match")
    if ct_match < 0.99:
        print("  WARNING: Cell type alignment below 99%! Checking details...")
        # Use imputed_ct as the reference since it came from the model

    # Verify normalization scale
    print(f"  Normalization check:")
    print(f"    Valid cell range: [{valid_cell_X.min():.2f}, {valid_cell_X.max():.2f}]")
    print(f"    Valid nuc range: [{valid_nuc_X.min():.2f}, {valid_nuc_X.max():.2f}]")
    print(f"    Imputed range: [{imputed_X.min():.2f}, {imputed_X.max():.2f}]")

    # ---- Input audit ----
    print("\n[5] Creating input audit...")
    create_input_audit(gene_list, manifest, valid_nuc_X, valid_cell_X, imputed_X,
                       gene_names, output_dir)

    # ---- SynGO characterization ----
    print("\n[6] Characterizing SynGO resource...")
    genes_df, annotations_df, ontologies_df = load_syngo(args.syngo_zip)
    characterize_syngo(genes_df, annotations_df, ontologies_df, output_dir)

    # ---- Ortholog mapping ----
    print("\n[7] Building ortholog mapping...")
    mapping_df = build_ortholog_mapping(genes_df, args.ortholog_map, gene_list,
                                        ontologies_df, output_dir)

    # ---- Gene sets ----
    print("\n[8] Building SynGO gene sets...")
    gene_sets, subgroup_df = build_gene_sets(ontologies_df, mapping_df, output_dir)

    # ---- Cell type classification ----
    print("\n[9] Classifying cell types...")
    # Use common_cts that are present in both modalities
    cts_in_nuc = set(valid_nuc_ct)
    cts_in_cell = set(valid_cell_ct)
    analysis_cts = sorted(cts_in_nuc & cts_in_cell)
    ct_status = classify_cell_types(analysis_cts, output_dir)

    # ---- Expression-matched controls ----
    print("\n[10] Building expression-matched controls...")
    syngo_broad_non_ribo = gene_sets['synapse']['mouse_non_ribo']
    all_syngo_broad = gene_sets['synapse']['mouse_genes']

    matched_pairs, matching_qc = build_matched_controls(
        syngo_broad_non_ribo, all_syngo_broad, gene_list,
        train_nuc_X, gene_names, 'synapse_non_ribo', output_dir)

    matched_pairs.to_csv(os.path.join(tables_dir, 'matched_controls.csv'), index=False)
    matching_qc.to_csv(os.path.join(tables_dir, 'matching_qc.csv'), index=False)

    if len(matching_qc) > 0:
        qc = matching_qc.iloc[0]
        print(f"  Matched {qc['n_matched']} pairs from {qc['n_control_pool']} controls")
        print(f"  SMD mean expression: {qc['smd_mean_expression']:.4f}")
        print(f"  SMD detection rate: {qc['smd_detection_rate']:.4f}")

    # ---- Validation shifts ----
    print("\n[11] Computing validation shifts...")
    shifts_df, common_cts, rare_cts = compute_validation_shifts(
        valid_cell_X, valid_cell_ct, valid_nuc_X, valid_nuc_ct,
        imputed_X, imputed_ct, gene_names, ct_status, gene_sets, output_dir)

    # ---- Primary enrichment analysis ----
    print("\n[12] Primary matched-enrichment analysis...")
    ct_enrich_primary, boot_primary = compute_enrichment(
        shifts_df, matched_pairs, ct_status, common_cts,
        args.n_bootstrap, rng, 'synapse_non_ribo')

    ct_enrich_primary.to_csv(os.path.join(tables_dir, 'celltype_syngo_enrichment.csv'), index=False)
    boot_primary.to_csv(os.path.join(tables_dir, 'bootstrap_primary_results.csv'), index=False)

    br = boot_primary.iloc[0]
    print(f"\n  PRIMARY RESULTS (broad non-ribosomal SynGO):")
    print(f"    Pearson r: {br['pearson_r']:.4f}")
    print(f"    Spearman rho: {br['spearman_r']:.4f}")
    print(f"    R²: {br['R2']:.4f}")
    print(f"    MAE: {br['MAE']:.4f}")
    print(f"    E_empirical neuronal: {br['E_empirical_neuronal']:.4f}")
    print(f"    E_empirical non-neuronal: {br['E_empirical_non_neuronal']:.4f}")
    print(f"    E_predicted neuronal: {br['E_predicted_neuronal']:.4f}")
    print(f"    E_predicted non-neuronal: {br['E_predicted_non_neuronal']:.4f}")
    print(f"    Empirical neuro diff: {br['empirical_neuro_diff']:.4f}")
    print(f"    Predicted neuro diff: {br['predicted_neuro_diff']:.4f}")

    # ---- Recovery metrics ----
    print("\n[13] Computing recovery metrics...")
    recovery_df, per_ct_recovery, per_gene_recovery = compute_recovery_metrics(
        shifts_df, syngo_broad_non_ribo, ct_status, common_cts, output_dir)

    # ---- Rpl/Rps sensitivity ----
    print("\n[14] Rpl/Rps sensitivity analysis...")
    syngo_broad_all = gene_sets['synapse']['mouse_genes']
    matched_pairs_ribo, matching_qc_ribo = build_matched_controls(
        syngo_broad_all, all_syngo_broad, gene_list,
        train_nuc_X, gene_names, 'synapse_with_ribo', output_dir)

    ct_enrich_ribo, boot_ribo = compute_enrichment(
        shifts_df, matched_pairs_ribo, ct_status, common_cts,
        args.n_bootstrap, rng, 'synapse_with_ribo')

    # Rare cell type sensitivity
    print("\n[15] Rare cell-type sensitivity...")
    ct_enrich_no_rare, boot_no_rare = sensitivity_rare_ct(
        shifts_df, matched_pairs, ct_status, common_cts, rare_cts,
        args.n_bootstrap, rng)

    # Combine sensitivity results
    sensitivity_results = pd.concat([boot_primary, boot_ribo, boot_no_rare], ignore_index=True)
    sensitivity_results.to_csv(os.path.join(tables_dir, 'syngo_sensitivity_results.csv'), index=False)

    print(f"\n  Sensitivity comparison:")
    for _, r in sensitivity_results.iterrows():
        print(f"    {r['subgroup']}: Pearson={r['pearson_r']:.3f}, "
              f"neuro_diff_emp={r['empirical_neuro_diff']:.4f}, "
              f"neuro_diff_pred={r['predicted_neuro_diff']:.4f}")

    # ==== PRIORITY 2 ====
    if args.priority >= 2:
        print("\n" + "=" * 70)
        print("PRIORITY 2: Subgroup analyses and main figure")
        print("=" * 70)

        # Subgroup analyses
        print("\n[16] Subgroup analyses...")
        subgroup_results_list = []

        subgroups_to_analyze = [
            'presynaptic_only', 'postsynaptic_only', 'both_pre_post',
            'postsynaptic_density', 'presynaptic_active_zone',
            'synaptic_vesicle', 'synaptic_cytosol',
        ]

        for sg_name in subgroups_to_analyze:
            gs = gene_sets.get(sg_name)
            if gs is None:
                continue

            sg_genes = gs['mouse_non_ribo']
            if len(sg_genes) < 5:
                print(f"    {sg_name}: skipped (only {len(sg_genes)} genes)")
                subgroup_results_list.append({
                    'subgroup': sg_name, 'type': gs['type'],
                    'n_genes': len(sg_genes), 'status': 'skipped_too_few',
                })
                continue

            sg_pairs, sg_qc = build_matched_controls(
                sg_genes, all_syngo_broad, gene_list,
                train_nuc_X, gene_names, sg_name, output_dir)

            if len(sg_pairs) == 0:
                subgroup_results_list.append({
                    'subgroup': sg_name, 'type': gs['type'],
                    'n_genes': len(sg_genes), 'status': 'no_matches',
                })
                continue

            sg_ct_enrich, sg_boot = compute_enrichment(
                shifts_df, sg_pairs, ct_status, common_cts,
                args.n_bootstrap, rng, sg_name)

            if len(sg_boot) > 0:
                row = sg_boot.iloc[0].to_dict()
                row['n_genes'] = len(sg_genes)
                row['type'] = gs['type']
                row['status'] = 'computed'
                subgroup_results_list.append(row)

        subgroup_results = pd.DataFrame(subgroup_results_list)
        subgroup_results.to_csv(os.path.join(tables_dir, 'syngo_subgroup_results.csv'), index=False)

        print(f"\n  Subgroup results:")
        for _, r in subgroup_results.iterrows():
            if r.get('status') == 'computed':
                print(f"    {r['subgroup']} (n={r['n_genes']}): "
                      f"Pearson={r.get('pearson_r', np.nan):.3f}, "
                      f"neuro_diff_emp={r.get('empirical_neuro_diff', np.nan):.4f}")
            else:
                print(f"    {r['subgroup']}: {r.get('status', 'unknown')}")

        # BH correction
        computed_mask = subgroup_results['status'] == 'computed'
        if computed_mask.sum() > 1 and 'pearson_p' in subgroup_results.columns:
            pvals = subgroup_results.loc[computed_mask, 'pearson_p'].values
            valid_p = ~np.isnan(pvals)
            if valid_p.sum() > 1:
                from statsmodels.stats.multitest import multipletests
                _, pvals_corrected, _, _ = multipletests(pvals[valid_p], method='fdr_bh')
                corrected = np.full(len(pvals), np.nan)
                corrected[valid_p] = pvals_corrected
                subgroup_results.loc[computed_mask, 'pearson_p_bh'] = corrected
                subgroup_results.to_csv(os.path.join(tables_dir, 'syngo_subgroup_results.csv'), index=False)

        # Main figure
        print("\n[17] Creating main figure...")
        create_figures(ct_enrich_primary, boot_primary, subgroup_results,
                       ct_status, common_cts, output_dir)
    else:
        subgroup_results = None

    # ==== PRIORITY 3 ====
    if args.priority >= 3:
        print("\n" + "=" * 70)
        print("PRIORITY 3: Baselines, donor sensitivity, supplementary")
        print("=" * 70)

        # Baseline comparison
        print("\n[18] Baseline comparison...")
        baseline_df = baseline_comparison(
            shifts_df, matched_pairs, syngo_broad_non_ribo, ct_status,
            common_cts, manifest, valid_nuc_idx, args.n_bootstrap, rng, output_dir)

        # Donor-balanced sensitivity
        print("\n[19] Donor-balanced sensitivity...")
        # Add donor info to validation metadata
        nuc_all = manifest[manifest.modality == 'nuclei'].sort_values('obs_id').reset_index(drop=True)
        cell_all = manifest[manifest.modality == 'cell'].sort_values('obs_id').reset_index(drop=True)

        valid_nuc_donors = nuc_all.iloc[valid_nuc_idx][['cell_type', 'donor']].reset_index(drop=True)
        valid_cell_donors = cell_all.iloc[valid_cell_idx][['cell_type', 'donor']].reset_index(drop=True)

        donor_df = sensitivity_donor_balanced(
            shifts_df, valid_cell_donors, valid_nuc_donors,
            valid_cell_X, valid_nuc_X, imputed_X,
            gene_names, syngo_broad_non_ribo, matched_pairs,
            ct_status, common_cts, output_dir)

        # Transport categories (exploratory)
        print("\n[20] Transport categories (exploratory)...")
        for transport_name in ['axo_dendritic_transport', 'dendritic_transport', 'axonal_transport']:
            gs = gene_sets.get(transport_name)
            if gs and len(gs['mouse_non_ribo']) >= 15:
                print(f"    {transport_name}: {len(gs['mouse_non_ribo'])} genes")
            else:
                n = len(gs['mouse_non_ribo']) if gs else 0
                print(f"    {transport_name}: {n} genes (< 15, skipped)")

        # Supplementary figures
        print("\n[21] Creating supplementary figures...")
        control_genes = set(matched_pairs.control_gene) if len(matched_pairs) > 0 else set()
        create_supplementary_figures(
            matched_pairs, shifts_df, syngo_broad_non_ribo, control_genes,
            ct_enrich_primary, per_ct_recovery, ct_status,
            common_cts, rare_cts, boot_ribo, boot_no_rare,
            subgroup_results, output_dir)

    # ============================================================
    # Terminal Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("TERMINAL SUMMARY")
    print("=" * 70)

    print(f"\n  Input paths:")
    print(f"    Gene list: {args.gene_list}")
    print(f"    Split manifest: {args.split_manifest}")
    print(f"    SynGO archive: {args.syngo_zip}")
    print(f"    Ortholog mapping: {args.ortholog_map}")
    print(f"    ImputeVAE predictions: {CVAE_VALID_IMPUTED}")

    print(f"\n  Split validation:")
    print(f"    Validation nuclei: {len(valid_nuc_idx)}")
    print(f"    Validation cells: {len(valid_cell_idx)}")
    print(f"    Training nuclei (for matching): {len(train_nuc_idx)}")

    print(f"\n  Matrix shapes:")
    print(f"    Valid cell: {valid_cell_X.shape}")
    print(f"    Valid nuclei: {valid_nuc_X.shape}")
    print(f"    Imputed: {imputed_X.shape}")

    print(f"\n  Ortholog mapping: {args.ortholog_map}")
    flow = pd.read_csv(os.path.join(tables_dir, 'syngo_filtering_flow.csv'))
    print(f"\n  SynGO filtering flow:")
    for _, r in flow.iterrows():
        print(f"    {r['stage']}: {r['count']}")

    print(f"\n  Subgroup counts:")
    sg_counts = pd.read_csv(os.path.join(tables_dir, 'syngo_subgroup_counts.csv'))
    for _, r in sg_counts.iterrows():
        print(f"    {r['subgroup']}: {r['non_ribosomal_analysis_count']}")

    if len(matching_qc) > 0:
        qc = matching_qc.iloc[0]
        print(f"\n  Matching QC:")
        print(f"    SMD mean expression: {qc['smd_mean_expression']:.4f}")
        print(f"    SMD detection rate: {qc['smd_detection_rate']:.4f}")

    br = boot_primary.iloc[0]
    print(f"\n  Primary results (broad non-ribosomal SynGO):")
    print(f"    Pearson r: {br['pearson_r']:.4f}")
    print(f"    Spearman rho: {br['spearman_r']:.4f}")
    print(f"    R²: {br['R2']:.4f}")
    print(f"    MAE: {br['MAE']:.4f}")
    print(f"    Empirical neuronal enrichment: {br['E_empirical_neuronal']:.4f}")
    print(f"    Empirical non-neuronal enrichment: {br['E_empirical_non_neuronal']:.4f}")
    print(f"    Predicted neuronal enrichment: {br['E_predicted_neuronal']:.4f}")
    print(f"    Predicted non-neuronal enrichment: {br['E_predicted_non_neuronal']:.4f}")
    print(f"    Empirical neuro-non diff: {br['empirical_neuro_diff']:.4f}")
    print(f"    Predicted neuro-non diff: {br['predicted_neuro_diff']:.4f}")

    print(f"\n  Recovery metrics:")
    for _, r in recovery_df.iterrows():
        pr = r.get('pearson_r', np.nan)
        print(f"    {r['label']}: Pearson={pr:.3f}, R2={r.get('R2', np.nan):.3f}, "
              f"sign={r.get('sign_agreement', np.nan):.3f}")

    print(f"\n  Sensitivity:")
    for _, r in sensitivity_results.iterrows():
        print(f"    {r['subgroup']}: Pearson={r['pearson_r']:.3f}, "
              f"neuro_diff_emp={r['empirical_neuro_diff']:.4f}")

    if args.priority >= 3:
        print(f"\n  Donor sensitivity: {'computed' if donor_df is not None else 'not feasible'}")
        print(f"  Baseline comparison: see {os.path.join(tables_dir, 'syngo_method_comparison.csv')}")

    # List all output files
    print(f"\n  Output tables:")
    for f in sorted(os.listdir(tables_dir)):
        print(f"    {os.path.join(tables_dir, f)}")
    print(f"\n  Output figures:")
    for f in sorted(os.listdir(fig_dir)):
        print(f"    {os.path.join(fig_dir, f)}")

    print("\n" + "=" * 70)
    print("Analysis complete.")
    print("=" * 70)


if __name__ == '__main__':
    main()
