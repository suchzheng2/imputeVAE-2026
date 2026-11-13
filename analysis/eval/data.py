"""
Data loading for evaluation. Loads raw cell/nuclei data, filters, caches by gene list.

All functions return numpy arrays in log1p space (matching training normalization).
"""

import os
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_DATA_DIR = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi"
CELL_PATH = "10X_cells_v2_v3_AIBS"
NUC_PATH = "10X_nuclei_v2_v3_AIBS"
CELL_BATCH = "c3A"
NUC_BATCH = "n3A"

CELL_ID = 0
NUCLEI_ID = 1

# Cache: gene_list_path -> EvalData
_cache = {}


class EvalData:
    """Container for matched cell/nuclei evaluation data."""

    def __init__(self, X_cell, X_nuc, ct_cell, ct_nuc, gene_names, common_cts):
        self.X_cell = X_cell          # [N_cell, G] float32, log1p
        self.X_nuc = X_nuc            # [N_nuc, G] float32, log1p
        self.ct_cell = ct_cell        # [N_cell] str array
        self.ct_nuc = ct_nuc          # [N_nuc] str array
        self.gene_names = gene_names  # [G] str array
        self.common_cts = common_cts  # sorted list of shared cell types
        self.n_genes = X_cell.shape[1]

    def celltype_means(self, X, ct):
        """Compute per-cell-type mean expression. Returns dict: ct -> [G] array."""
        means = {}
        for c in self.common_cts:
            mask = ct == c
            if mask.sum() > 0:
                means[c] = X[mask].mean(axis=0)
        return means


def _load_h5ad(data_path, batch, gene_list_path, normalization='log1p'):
    """Load and filter a single h5ad file.

    Parameters
    ----------
    normalization : str
        'log1p' — log1p(counts)  (default, matches most models)
        'cpm'   — log1p(counts / library_size * 1e6)
    """
    h5ad_path = os.path.join(BASE_DATA_DIR, data_path, "data.h5ad")
    adata = ad.read_h5ad(h5ad_path)

    # Add meta_batch_id if missing
    if 'meta_batch_id' not in adata.obs:
        adata.obs['meta_batch_id'] = adata.obs['batch_id'].apply(lambda x: x.split('.')[0])

    # Filter to v3 protocol
    adata = adata[adata.obs['meta_batch_id'].str.endswith(batch)].copy()

    # Common genes
    common_genes = pd.read_csv(
        os.path.join(_PROJECT_DIR, 'data', 'common_genes.csv'), header=0
    )['common_genes'].tolist()
    adata = adata[:, adata.var['gene_names'].isin(common_genes)].copy()

    # Common cell types
    common_ct = pd.read_csv(
        os.path.join(_PROJECT_DIR, 'data', 'common_cell_types.csv'), header=0
    )['common_cell_types'].tolist()
    adata = adata[adata.obs['cell_type'].isin(common_ct)].copy()

    # Gene list filter
    gene_list = pd.read_csv(gene_list_path, header=None)[0].tolist()
    adata = adata[:, adata.var['gene_names'].isin(gene_list)].copy()

    # Sort by gene name
    adata = adata[:, adata.var['gene_names'].argsort()].copy()

    # Normalize
    X = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
    if normalization == 'cpm':
        lib_size = X.sum(axis=1, keepdims=True)
        lib_size = np.clip(lib_size, 1.0, None)
        X = X / lib_size * 1e6
    adata.X = np.log1p(X).astype(np.float32)

    return adata


def _stratified_subsample(adata, max_per_type=2000, seed=42):
    """Subsample to at most max_per_type cells per cell type."""
    rng = np.random.RandomState(seed)
    idx = []
    for ct in adata.obs['cell_type'].unique():
        ct_idx = np.where(adata.obs['cell_type'].values == ct)[0]
        if len(ct_idx) > max_per_type:
            ct_idx = rng.choice(ct_idx, max_per_type, replace=False)
        idx.extend(ct_idx)
    return adata[sorted(idx)].copy()


def load_eval_data(gene_list_path, max_per_type=2000, force_reload=False,
                   normalization='log1p'):
    """Load matched cell/nuclei data for evaluation. Cached by gene list + normalization.

    Parameters
    ----------
    gene_list_path : str
        Path to gene list CSV (determines which genes to evaluate on).
    max_per_type : int
        Maximum cells per cell type (subsampling for tractability).
    force_reload : bool
        If True, bypass cache.
    normalization : str
        'log1p' or 'cpm'. Must match the model's training normalization.

    Returns
    -------
    EvalData
    """
    cache_key = (gene_list_path, normalization)
    if not force_reload and cache_key in _cache:
        print(f"  [data] Using cached data for {os.path.basename(gene_list_path)} ({normalization})")
        return _cache[cache_key]

    print(f"  [data] Loading data for gene list: {os.path.basename(gene_list_path)} ({normalization})")

    cell_adata = _load_h5ad(CELL_PATH, CELL_BATCH, gene_list_path, normalization)
    nuc_adata = _load_h5ad(NUC_PATH, NUC_BATCH, gene_list_path, normalization)

    cell_adata = _stratified_subsample(cell_adata, max_per_type)
    nuc_adata = _stratified_subsample(nuc_adata, max_per_type)

    X_cell = cell_adata.X if not sparse.issparse(cell_adata.X) else cell_adata.X.toarray()
    X_nuc = nuc_adata.X if not sparse.issparse(nuc_adata.X) else nuc_adata.X.toarray()
    ct_cell = cell_adata.obs['cell_type'].values.astype(str)
    ct_nuc = nuc_adata.obs['cell_type'].values.astype(str)
    gene_names = cell_adata.var['gene_names'].values.astype(str)
    common_cts = sorted(set(ct_cell) & set(ct_nuc))

    print(f"  [data] Cell: {X_cell.shape}, Nuclei: {X_nuc.shape}, "
          f"CTs: {len(common_cts)}, Genes: {X_cell.shape[1]}")

    data = EvalData(X_cell, X_nuc, ct_cell, ct_nuc, gene_names, common_cts)
    _cache[cache_key] = data
    return data
