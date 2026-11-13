#!/usr/bin/env python3
"""
Shared utilities for the unified evaluation pipeline.

Provides common functions for loading data, filtering, subsampling,
and computing cell-type means used across all analysis modules.
"""

import os
import sys
import gc
import json
import glob
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse

# =============================================================================
# Paths
# =============================================================================

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_BASE_DIR = os.path.join(_PROJECT_DIR, "run")
DATA_DIR = os.path.join(_PROJECT_DIR, "data")
BASE_DATA_DIR = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi"
RAW_CELL_PATH = "10X_cells_v2_v3_AIBS"
RAW_NUCLEI_PATH = "10X_nuclei_v2_v3_AIBS"
MITOCARTA_PATH = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/mouse.miticarta.analysis/Mouse.MitoCarta.csv"

# Analysis parameters
MAX_CELLS_PER_TYPE = 30
SUBSAMPLE_FRACTION = 0.1
RANDOM_SEED = 42


def find_run_folder(run_name):
    """Find the run folder matching the given name."""
    # Strip leading "run/" prefix if present
    if run_name.startswith("run/"):
        run_name = run_name[4:]
    exact = os.path.join(RUN_BASE_DIR, run_name)
    if os.path.isdir(exact):
        return exact
    matches = glob.glob(os.path.join(RUN_BASE_DIR, f"*{run_name}*"))
    dirs = [m for m in matches if os.path.isdir(m)]
    if len(dirs) == 1:
        return dirs[0]
    elif len(dirs) > 1:
        print(f"Multiple matches found: {dirs}")
        print(f"Using first match: {dirs[0]}")
        return dirs[0]
    else:
        print(f"Error: No run folder matching '{run_name}' found in {RUN_BASE_DIR}")
        sys.exit(1)


def load_run_args(run_folder):
    """Load arguments.json from a run folder."""
    args_path = os.path.join(run_folder, 'arguments.json')
    if not os.path.isfile(args_path):
        return {}
    with open(args_path, 'r') as f:
        return json.load(f)


def get_eval_dir(run_folder):
    """Get the evaluation output directory, creating if needed."""
    eval_dir = os.path.join(run_folder, 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    return eval_dir


def list_epoch_dumps(run_folder):
    """List available epoch dump files, returning sorted epoch numbers."""
    dump_dir = os.path.join(run_folder, 'epoch_dump')
    if not os.path.isdir(dump_dir):
        return []
    epochs = []
    for f in os.listdir(dump_dir):
        if f.startswith('train_epoch_') and f.endswith('.json'):
            try:
                epoch = int(f.replace('train_epoch_', '').replace('.json', ''))
                epochs.append(epoch)
            except ValueError:
                pass
    if epochs:
        return sorted(epochs)

    latest_joint_dump = os.path.join(dump_dir, 'train.json')
    if os.path.isfile(latest_joint_dump):
        for loss_path in [
            os.path.join(run_folder, 'train_epoch_loss', 'loss.txt'),
            os.path.join(run_folder, 'loss.txt'),
        ]:
            if not os.path.isfile(loss_path):
                continue
            try:
                df = pd.read_csv(loss_path)
                if not df.empty:
                    return [int(df.iloc[-1, 0])]
            except Exception:
                continue
        return [0]
    return sorted(epochs)


def find_epoch_dump(run_folder, epoch):
    """Find the epoch dump file for a given epoch number."""
    dump_dir = os.path.join(run_folder, 'epoch_dump')
    candidates = [
        os.path.join(dump_dir, f'train_epoch_{epoch}.json'),
        os.path.join(dump_dir, f'epoch_{epoch}_train.json'),
        os.path.join(dump_dir, f'train_{epoch}.json'),
    ]
    for fp in candidates:
        if os.path.isfile(fp):
            return fp
    latest_joint_dump = os.path.join(dump_dir, 'train.json')
    if os.path.isfile(latest_joint_dump):
        return latest_joint_dump
    return None


def infer_target_modality(run_args):
    data_path = str(run_args.get('data_path', '')).lower()
    if 'nuclei' in data_path:
        return 'nuclei'
    if 'cell' in data_path:
        return 'cell'
    return 'nuclei'


def _resolve_feature_key(batch_data, feature, modality=None):
    if feature in batch_data:
        return feature
    if modality is not None:
        modality_key = f'{modality}_{feature}'
        if modality_key in batch_data:
            return modality_key
    raise KeyError(f"Missing feature '{feature}' for modality '{modality}'")


def has_joint_modality_outputs(batch_data):
    """Return True when a batch dump stores both cell and nuclei outputs."""
    return (
        'cell_decoded' in batch_data and 'nuclei_decoded' in batch_data and
        'cell_encoded' in batch_data and 'nuclei_encoded' in batch_data
    )


def resolve_batch_id_list(batch_data, modality=None):
    """Resolve batch labels for a dump batch.

    Single-modality dumps store `meta_batch_id_original`.
    Joint dumps currently also store only `meta_batch_id_original` for the
    target modality batch. If the caller requests the non-target modality and
    no modality-specific batch labels are present, return `None` so evaluators
    can skip or fail clearly instead of using the wrong labels.
    """
    if modality is not None:
        modality_key = f'{modality}_meta_batch_id_original'
        if modality_key in batch_data:
            return batch_data[modality_key]

    if 'meta_batch_id_original' not in batch_data:
        return None

    if modality is None or not has_joint_modality_outputs(batch_data):
        return batch_data['meta_batch_id_original']

    decoded_key = f'{modality}_decoded'
    if decoded_key not in batch_data:
        return batch_data['meta_batch_id_original']

    # Joint dump with generic batch ids only: assume these belong to the
    # target modality batch, not the paired reference modality batch.
    target_modality = None
    batch_ids = batch_data['meta_batch_id_original']
    if batch_ids:
        sample = str(batch_ids[0])
        if '-n' in sample or sample.endswith('n3A') or sample.endswith('n2A'):
            target_modality = 'nuclei'
        elif '-c' in sample or sample.endswith('c3A') or sample.endswith('c2A'):
            target_modality = 'cell'

    if target_modality is None or modality == target_modality:
        return batch_data['meta_batch_id_original']

    return None


def load_common_filters():
    """Load common genes, cell types, and mitocarta genes."""
    cg_path = os.path.join(DATA_DIR, 'common_genes.csv')
    cct_path = os.path.join(DATA_DIR, 'common_cell_types.csv')

    if not os.path.isfile(cg_path) or not os.path.isfile(cct_path):
        print(f"Error: Cannot find common_genes.csv or common_cell_types.csv in {DATA_DIR}")
        sys.exit(1)

    common_genes = set(pd.read_csv(cg_path, header=0).iloc[:, 0].tolist())
    common_cell_types = set(pd.read_csv(cct_path, header=0).iloc[:, 0].tolist())

    mitocarta_genes = set()
    if os.path.isfile(MITOCARTA_PATH):
        mitocarta_genes = set(pd.read_csv(MITOCARTA_PATH, header=0)['Symbol'].tolist())

    return common_genes, common_cell_types, mitocarta_genes


def is_stage3_run(run_args):
    """Return True if the run is a stage 3 / joint training run."""
    return (run_args.get('fine_tune_stage3_v2', False)
            or run_args.get('fine_tune_cond_stage3', False))


def load_raw_data(data_path, common_genes, common_cell_types, mitocarta_genes,
                  detection_rate_threshold=1.0, exclude_mitocarta=True,
                  protocol=None, batch=None, gene_list=None):
    """Load and preprocess raw single-cell/nuclei data.

    Parameters
    ----------
    protocol : str or None
        If set ('v2', 'v3', or 'both'), filter cells by 10X protocol version.
        Uses the meta_batch_id suffix convention (e.g. n2A=v2, n3A=v3).
    batch : str or None
        Batch string (e.g. 'n3A', 'c3A') used to infer modality/sample chars
        when protocol filtering is applied.
    gene_list : str or None
        Path to a gene list CSV file.  If provided, raw data is filtered to
        exactly these genes (intersected with available genes), matching the
        training gene set.  Overrides detection_rate_threshold and
        exclude_mitocarta for gene selection (same logic as data.py).
    """
    raw_dp = os.path.join(BASE_DATA_DIR, data_path, "data.h5ad")
    print(f"Loading raw data from: {raw_dp}")
    adata = ad.read_h5ad(raw_dp)

    adata = adata[adata.obs['cell_type'].isin(common_cell_types)].copy()

    # Protocol filtering — match training data selection
    if protocol is not None and 'meta_batch_id' in adata.obs.columns:
        batch_str = (batch or '').strip()
        if batch_str and batch_str[0] in ('n', 'c'):
            modality_char = batch_str[0]
        else:
            # infer from data path
            modality_char = 'n' if 'nuclei' in data_path.lower() else 'c'
        sample_char = batch_str[-1] if batch_str and batch_str[-1].isalpha() else 'A'

        if protocol == 'both':
            v2_suffix = f'{modality_char}2{sample_char}'
            v3_suffix = f'{modality_char}3{sample_char}'
            mask = (adata.obs['meta_batch_id'].str.endswith(v2_suffix) |
                    adata.obs['meta_batch_id'].str.endswith(v3_suffix))
            adata = adata[mask].copy()
        elif protocol in ('v2', 'v3'):
            version = '2' if protocol == 'v2' else '3'
            target = f'{modality_char}{version}{sample_char}'
            adata = adata[adata.obs['meta_batch_id'].str.endswith(target)].copy()
        print(f"  Protocol filter ({protocol}): {adata.shape[0]} cells remaining")

    # Gene filtering — two paths, matching data.py logic:
    # 1. If gene_list provided: filter to common_genes first, then to gene_list
    # 2. Otherwise: filter by common_genes + detection_rate + mitocarta
    # First always filter to common_genes
    gene_mask = adata.var['gene_names'].isin(common_genes)
    adata = adata[:, gene_mask].copy()

    if gene_list is not None:
        # Load gene list and filter to exactly those genes
        gene_list_genes = set(pd.read_csv(gene_list, header=None)[0].tolist())
        adata = adata[:, adata.var['gene_names'].isin(gene_list_genes)].copy()
        print(f"  Gene list filter ({gene_list}): {adata.shape[1]} genes")
    else:
        if exclude_mitocarta and mitocarta_genes:
            adata = adata[:, ~adata.var['gene_names'].isin(mitocarta_genes)].copy()

        if detection_rate_threshold < 1.0:
            det_rates = (adata.X > 0).sum(axis=1) / adata.X.shape[1]
            if hasattr(det_rates, 'A1'):
                det_rates = det_rates.A1
            adata = adata[det_rates >= detection_rate_threshold].copy()

    adata = adata[:, adata.var['gene_names'].argsort()].copy()
    print(f"  Final shape: {adata.shape}, {len(adata.obs['cell_type'].unique())} cell types")
    return adata


def load_decoded_stage1(train_fp, common_cell_types, modality=None):
    """Load decoded data from single-modality or joint epoch dumps."""
    print(f"Loading decoded from: {train_fp}")
    with open(train_fp, 'r') as f:
        data = json.load(f)

    decoded_list, types_list = [], []
    for ct in sorted(data.keys()):
        if ct not in common_cell_types:
            continue
        ct_dec = []
        for bid in data[ct]:
            batch_data = data[ct][bid]
            key = _resolve_feature_key(batch_data, 'decoded', modality=modality)
            ct_dec.append(np.array(batch_data[key], dtype=np.float32))
        if ct_dec:
            ct_dec = np.vstack(ct_dec)
            decoded_list.append(ct_dec)
            types_list.extend([ct] * ct_dec.shape[0])

    del data
    gc.collect()

    decoded_arr = np.vstack(decoded_list)
    types_arr = np.array(types_list)
    print(f"  Decoded shape: {decoded_arr.shape}, {len(np.unique(types_arr))} cell types")
    return decoded_arr, types_arr


def load_encoded_stage1(train_fp, common_cell_types, modality=None):
    """Load encoded (latent) data from single-modality or joint epoch dumps."""
    with open(train_fp, 'r') as f:
        data = json.load(f)

    encoded_list, types_list, batch_list = [], [], []
    for ct in sorted(data.keys()):
        if ct not in common_cell_types:
            continue
        for bid in data[ct]:
            batch_data = data[ct][bid]
            key = _resolve_feature_key(batch_data, 'encoded', modality=modality)
            encoded_list.append(np.array(batch_data[key], dtype=np.float32))
            n = len(batch_data[key])
            types_list.extend([ct] * n)
            meta = batch_data.get('meta_batch_id_original', [ct] * n)
            batch_list.extend(meta)

    del data
    gc.collect()

    encoded_arr = np.vstack(encoded_list)
    types_arr = np.array(types_list)
    batch_arr = np.array(batch_list)
    return encoded_arr, types_arr, batch_arr


def compute_celltype_mean(data, cell_types, gene_names=None):
    """Compute mean expression per cell type."""
    unique_cts = np.unique(cell_types)
    means = []
    for ct in unique_cts:
        m = data[cell_types == ct].mean(axis=0)
        if hasattr(m, 'A1'):
            m = m.A1
        means.append(m)
    df = pd.DataFrame(means, index=unique_cts)
    if gene_names is not None:
        df.columns = gene_names
    return df


def stratified_subsample(data, cell_types, max_per_type=30, seed=42, target_counts=None):
    """Stratified subsample, return sorted by cell type."""
    np.random.seed(seed)
    indices = []
    cts_to_use = sorted(target_counts.keys()) if target_counts else sorted(set(cell_types))
    for ct in cts_to_use:
        ct_idx = np.where(cell_types == ct)[0]
        if len(ct_idx) == 0:
            continue
        limit = target_counts[ct] if target_counts else max_per_type
        if len(ct_idx) > limit:
            chosen = np.random.choice(ct_idx, limit, replace=False)
        else:
            chosen = ct_idx
        indices.extend(sorted(chosen))
    indices = np.array(indices)
    sorted_order = np.argsort(cell_types[indices])
    indices = indices[sorted_order]
    return data[indices], cell_types[indices]


def compute_target_counts(datasets_ct, max_per_type=30):
    """Compute target sample counts as min across all datasets per cell type."""
    common_cts = None
    for ct_arr in datasets_ct:
        cts = set(ct_arr)
        common_cts = cts if common_cts is None else common_cts & cts
    common_cts = sorted(common_cts)

    target = {}
    for ct in common_cts:
        counts = [int((ct_arr == ct).sum()) for ct_arr in datasets_ct]
        target[ct] = min(min(counts), max_per_type)

    return {ct: c for ct, c in target.items() if c > 0}


def stratified_subsample_adata(adata, fraction=0.1, min_per_type=2, seed=42):
    """Subsample an AnnData while keeping at least min_per_type cells per cell type."""
    rng = np.random.RandomState(seed)
    indices = []
    for ct in adata.obs['cell_type'].unique():
        ct_idx = np.where(adata.obs['cell_type'].values == ct)[0]
        n_sample = max(min_per_type, int(len(ct_idx) * fraction))
        n_sample = min(n_sample, len(ct_idx))
        indices.extend(rng.choice(ct_idx, n_sample, replace=False))
    return adata[indices]


def get_raw_data_for_correlation(adata, use_log1p=False):
    """Extract expression matrix from AnnData, optionally log1p-transformed."""
    if use_log1p:
        X = adata.layers['log1p'] if 'log1p' in adata.layers else np.log1p(adata.X)
    else:
        X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    return X


def load_both_decoded_stage3(train_fp, common_cell_types):
    """Load both cell and nuclei decoded arrays from a joint dump in one pass.

    Returns (cell_decoded, cell_types, nuclei_decoded, nuclei_types).
    """
    print(f"Loading both decoded from: {train_fp}")
    with open(train_fp, 'r') as f:
        data = json.load(f)

    cell_dec, cell_types = [], []
    nuclei_dec, nuclei_types = [], []
    for ct in sorted(data.keys()):
        if ct not in common_cell_types:
            continue
        for bid in data[ct]:
            bd = data[ct][bid]
            cell_key = _resolve_feature_key(bd, 'decoded', modality='cell')
            nuclei_key = _resolve_feature_key(bd, 'decoded', modality='nuclei')
            n = len(bd[cell_key])
            cell_dec.append(np.array(bd[cell_key], dtype=np.float32))
            cell_types.extend([ct] * n)
            nuclei_dec.append(np.array(bd[nuclei_key], dtype=np.float32))
            nuclei_types.extend([ct] * n)

    del data
    gc.collect()

    cell_arr = np.vstack(cell_dec) if cell_dec else np.empty((0, 0), dtype=np.float32)
    nuclei_arr = np.vstack(nuclei_dec) if nuclei_dec else np.empty((0, 0), dtype=np.float32)
    print(f"  Cell decoded: {cell_arr.shape}, Nuclei decoded: {nuclei_arr.shape}")
    return cell_arr, np.array(cell_types), nuclei_arr, np.array(nuclei_types)


def load_both_encoded_stage3(train_fp, common_cell_types):
    """Load both cell and nuclei encoded arrays from a joint dump in one pass.

    Returns (cell_encoded, cell_types, cell_batch, nuclei_encoded, nuclei_types, nuclei_batch, modality_labels).
    """
    print(f"Loading both encoded from: {train_fp}")
    with open(train_fp, 'r') as f:
        data = json.load(f)

    cell_enc, cell_types, cell_batch = [], [], []
    nuclei_enc, nuclei_types, nuclei_batch = [], [], []
    for ct in sorted(data.keys()):
        if ct not in common_cell_types:
            continue
        for bid in data[ct]:
            bd = data[ct][bid]
            cell_key = _resolve_feature_key(bd, 'encoded', modality='cell')
            nuclei_key = _resolve_feature_key(bd, 'encoded', modality='nuclei')
            n = len(bd[cell_key])

            cell_enc.append(np.array(bd[cell_key], dtype=np.float32))
            cell_types.extend([ct] * n)
            cell_meta = resolve_batch_id_list(bd, modality='cell')
            cell_batch.extend(cell_meta if cell_meta else [ct] * n)

            nuclei_enc.append(np.array(bd[nuclei_key], dtype=np.float32))
            nuclei_types.extend([ct] * n)
            nuclei_meta = resolve_batch_id_list(bd, modality='nuclei')
            nuclei_batch.extend(nuclei_meta if nuclei_meta else [ct] * n)

    del data
    gc.collect()

    cell_arr = np.vstack(cell_enc) if cell_enc else np.empty((0, 0), dtype=np.float32)
    nuclei_arr = np.vstack(nuclei_enc) if nuclei_enc else np.empty((0, 0), dtype=np.float32)
    return (
        cell_arr, np.array(cell_types), np.array(cell_batch),
        nuclei_arr, np.array(nuclei_types), np.array(nuclei_batch),
    )


def print_memory_usage(label=""):
    try:
        import psutil
        mem_gb = psutil.Process(os.getpid()).memory_info().rss / 1024**3
        print(f"[MEMORY] {label}: {mem_gb:.2f} GB")
    except ImportError:
        pass
