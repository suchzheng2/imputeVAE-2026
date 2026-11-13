import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from sklearn.model_selection import train_test_split
import anndata as ad

from .utils import device


def _infer_modality_from_data_path(data_path):
    data_path = (data_path or "").strip().strip("/").lower()
    if "nuclei" in data_path:
        return "nuclei"
    if "cell" in data_path:
        return "cell"
    return "unknown"


def _build_sample_ids(obs, data_path):
    dataset_name = (data_path or "").strip().strip("/") or "unknown_dataset"
    modality = _infer_modality_from_data_path(data_path)
    obs_names = pd.Index(obs.index).astype(str)

    if "meta_batch_id" in obs.columns:
        meta_batch = obs["meta_batch_id"].astype(str)
        return pd.Index(
            [
                f"{modality}:{dataset_name}:{batch_id}:{obs_name}"
                for batch_id, obs_name in zip(meta_batch, obs_names)
            ]
        )

    return pd.Index(
        [f"{modality}:{dataset_name}:{obs_name}" for obs_name in obs_names]
    )


def _resolve_raw_data_path(data_path):
    return os.path.join(
        "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi",
        data_path,
        "data.h5ad",
    )


def _ensure_obs_id_fields(adata, data_path):
    if "meta_batch_id" not in adata.obs:
        def extract_meta_batch_id(batch_id):
            return batch_id.split('.')[0]
        adata.obs['meta_batch_id'] = adata.obs['batch_id'].apply(extract_meta_batch_id)

    adata.obs["sample_id"] = _build_sample_ids(adata.obs, data_path).to_numpy()
    return adata


def export_sample_manifest(data_path, output_path=None):
    dataset_name = (data_path or "").strip().strip("/") or "unknown_dataset"
    raw_dp = _resolve_raw_data_path(data_path)
    adata = ad.read_h5ad(raw_dp, backed="r")
    obs = adata.obs.copy()
    if hasattr(adata, "file") and adata.file is not None:
        adata.file.close()

    obs = _ensure_obs_id_fields(ad.AnnData(obs=obs), data_path).obs

    manifest = pd.DataFrame({
        "sample_id": obs["sample_id"].astype(str).to_numpy(),
        "modality": _infer_modality_from_data_path(data_path),
        "source_dataset": dataset_name,
        "obs_name": pd.Index(obs.index).astype(str).to_numpy(),
    }, index=obs.index)

    optional_columns = [
        "cell_type",
        "meta_batch_id",
        "batch_id",
    ]
    for column in optional_columns:
        if column in obs.columns:
            manifest[column] = obs[column].astype(str).to_numpy()

    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "sample_ids",
            f"{dataset_name}.samples.tsv",
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    manifest.to_csv(output_path, sep="\t", index=False)
    return output_path


class myDataset(Dataset):
    def __init__(self, data, obs, var):
        """
        Initialize with data matrix, observation annotations, and variable annotations.
        Convert `cell_type` and `meta_batch_id` to integer codes.
        """
        # self.data = torch.tensor(data, dtype=torch.float32).to(device)
        self.data = torch.tensor(data, dtype=torch.float32)
        self.obs = obs
        self.var = var
        # self.device = device

        # Store cell types as integer codes and mapping
        self.cell_type_codes, self.cell_type_mapping = pd.factorize(obs["cell_type"])  # Integer codes and mapping
        self.cell_type_tensor = torch.tensor(self.cell_type_codes, dtype=torch.long)  # Tensor form for PyTorch

        # Store batch IDs
        if "meta_batch_id" in obs.columns:
            self.batch_id_codes, self.batch_id_mapping = pd.factorize(obs["meta_batch_id"])
            self.batch_id_tensor = torch.tensor(self.batch_id_codes, dtype=torch.long)
            self.has_batch_id = True
        else:
            self.has_batch_id = False

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # item = {
        #     'data': self.data[idx],
        #     'cell_type': self.cell_type_tensor[idx].to(self.device)  # Integer code as tensor
        # }
        item = {
            'data': self.data[idx],
            'cell_type': self.cell_type_tensor[idx] # Integer code as tensor
        }
        
        if self.has_batch_id:
            item['batch_id'] = self.batch_id_tensor[idx]
        if "sample_id" in self.obs.columns:
            item["sample_id"] = str(self.obs.iloc[idx]["sample_id"])
        
        return item

    def get_original_value(self, attr, idx):
        """
        Retrieve the original string value from its integer ID.
        Args:
            attr: Either 'cell_type' or 'batch_id'
            idx: The integer ID to convert back to string
        Returns:
            The original string value
        """
        if attr == 'cell_type':
            return self.cell_type_mapping[idx]
        elif attr == 'batch_id' and self.has_batch_id:
            return self.batch_id_mapping[idx]
        else:
            raise ValueError(f"Attribute {attr} is not valid or not available.")

    def get_cell_type_mapping(self):
        """
        Returns the mapping between cell type codes and their string names.
        """
        return dict(enumerate(self.cell_type_mapping))

class MyConcatDataset(ConcatDataset):
    def get_original_value(self, attr, global_idx):
        """
        Retrieve the original value (e.g., cell_type or meta_batch_id) from its integer ID.
        This method assumes each sub-dataset has get_original_value.
        """
        for ds_idx, cum_size in enumerate(self.cumulative_sizes):
            if global_idx < cum_size:
                local_idx = global_idx if ds_idx == 0 else global_idx - self.cumulative_sizes[ds_idx - 1]
                return self.datasets[ds_idx].get_original_value(attr, local_idx)
        raise IndexError(f"Index {global_idx} out of range")

    def get_cell_type_mapping(self):
        """
        Combine cell type mappings from all sub-datasets.
        """
        combined_mapping = {}
        offset = 0
        for ds in self.datasets:
            local_mapping = ds.get_cell_type_mapping()
            for k, v in local_mapping.items():
                combined_mapping[offset + k] = v
            offset += len(local_mapping)
        return combined_mapping

class myDataset_old(Dataset):
    def __init__(self, data, obs, var):
        """
        Initialize with data matrix, observation annotations, and variable annotations.
        Convert `cell_type` and `meta_batch_id` to integer codes.
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.obs = obs

        # Convert 'cell_type' and 'meta_batch_id' to integer codes using pandas Categorical
        self.cell_type_codes, self.cell_type_mapping = pd.factorize(obs["cell_type"])
        self.batch_id_codes, self.batch_id_mapping = pd.factorize(obs["meta_batch_id"])
        
        # Store them as tensors
        self.cell_type = torch.tensor(self.cell_type_codes, dtype=torch.long)
        self.batch_id = torch.tensor(self.batch_id_codes, dtype=torch.long)
        
        self.var = var  # Variable annotations (e.g., gene names, etc.)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return {
            'data': self.data[idx],  # Data matrix row
            'cell_type': self.cell_type[idx],  # Cell type integer ID
            'batch_id': self.batch_id[idx]  # Meta batch ID integer
        }

    def get_original_value(self, attr, idx):
        """
        Retrieve the original value (e.g., cell_type or meta_batch_id) from its integer ID.
        """
        if attr == 'cell_type':
            return self.cell_type_mapping[idx]
        elif attr == 'batch_id':
            return self.batch_id_mapping[idx]
        else:
            raise ValueError(f"Attribute {attr} is not valid. Use 'cell_type' or 'meta_batch_id'.")

def minmax_normalize_data(data):
    min_value = data.min()
    max_value = data.max()
    return (data - min_value) / (max_value - min_value)

def log_transform_data(data):
    return np.log1p(data)
def quantile_normalize(df):
    ranks = df.stack().groupby(df.rank(method='first').stack().astype(int)).mean()
    normalized_df = df.rank(method='min').stack().astype(int).map(ranks).unstack()
    return normalized_df

def load_and_normalize_data(cellsfile, cell_types):
    # Load data for all cell types and concatenate
    all_cells_data = []
    for cell in cell_types:
        cell_data = pd.read_csv(cellsfile, sep="\t", header=0, usecols=lambda x: x.split(".")[0] == cell)
        all_cells_data.append(cell_data.transpose())
    combined_data = pd.concat(all_cells_data, axis=0)

    # Apply quantile normalization
    normalized_data = quantile_normalize(combined_data)

    # Split data back into cell types
    normalized_cell_data = {}
    start = 0
    for cell, cell_data in zip(cell_types, all_cells_data):
        end = start + cell_data.shape[1]
        normalized_cell_data[cell] = normalized_data.iloc[start:end, :]
        start = end

    return normalized_cell_data

def load_raw_data(data_path, args):
    
    raw_dp = _resolve_raw_data_path(data_path)
    
    # current data is still in old folder, after move to new folder we need to update the path
    # raw_dp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "scvi",
    #     data_path,
    #     "data.h5ad")
    adata = ad.read_h5ad(raw_dp)

    if args.raw_data_normalization == "log1p":
        if 'log1p' not in adata.layers:
            adata.layers['log1p'] = np.log1p(adata.X)
    elif args.raw_data_normalization == "cpm":
        if 'cpm' not in adata.layers:
            # Standard CPM: counts per million, then log1p (matches scanpy normalize_total + log1p)
            from scipy import sparse as sp
            X = adata.X
            lib_size = np.asarray(X.sum(axis=1)).flatten()
            lib_size = np.clip(lib_size, 1.0, None)
            if sp.issparse(X):
                # Scale each row: X / lib_size * 1e6, then densify for log1p
                diag = sp.diags(1e6 / lib_size)
                cpm = (diag @ X).toarray()
            else:
                cpm = X / lib_size[:, None] * 1e6
            adata.layers['cpm'] = np.log1p(cpm).astype(np.float32)
    elif args.raw_data_normalization == "raw":
        pass
    else:
        raise ValueError(f"Invalid raw data normalization: {args.raw_data_normalization}")

    #load common genes and cell types, and subset adata to common genes and cell types, ignore the header of the csv files
    _data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    common_genes_df = pd.read_csv(os.path.join(_data_dir, "common_genes.csv"), header=0)
    common_cell_types_df = pd.read_csv(os.path.join(_data_dir, "common_cell_types.csv"), header=0)
    print(f"Common genes: {common_genes_df.shape}")
    print(f"Common cell types: {common_cell_types_df.shape}")
    # subset adata with obs['cell_type'] in common_cell_types_df['common_cell_types'].tolist()
    adata = adata[adata.obs['cell_type'].isin(common_cell_types_df['common_cell_types'].tolist())].copy()
    
    
    # subset adata with var['gene_names'] in common_genes_df['common_genes'].tolist()
    adata = adata[:, adata.var['gene_names'].isin(common_genes_df['common_genes'].tolist())].copy()
    print(f"Subset adata to common genes and cell types: {adata.shape}")

    # If a gene_list file is provided, use exactly those genes (skips detection rate + mitocarta filters)
    gene_list_path = getattr(args, 'gene_list', None)
    if gene_list_path:
        gene_list = pd.read_csv(gene_list_path, header=None)[0].tolist()
        adata = adata[:, adata.var['gene_names'].isin(gene_list)].copy()
        print(f"Gene list filter ({gene_list_path}): {adata.shape}")
    else:
        # filter genes by detection rate (fraction of cells expressing each gene)
        if args.detection_rate_threshold != 1.0:
            gene_det_rate = np.asarray((adata.X > 0).sum(axis=0) / adata.X.shape[0]).flatten()
            threshold = args.detection_rate_threshold
            if threshold > 1.0:
                threshold = threshold / 100.0  # convert percentage to fraction
            gene_mask = gene_det_rate >= threshold
            adata = adata[:, gene_mask].copy()
            print(f"Gene detection rate filter (>={threshold:.2%}): {adata.shape}")

        if args.exclude_mitocarta:
            mitocarta_genes = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "Mouse.MitoCarta.csv"), header=0)
            mitocarta_genes = mitocarta_genes['Symbol'].tolist()
            adata = adata[:, ~adata.var['gene_names'].isin(mitocarta_genes)].copy()
            print(f"Exclude mitocarta genes from adata: {adata.shape}")

    # sort adata by gene names
    adata = adata[:, adata.var['gene_names'].argsort()].copy()
    print(f"Sort adata by gene names: {adata.shape}")    

    adata = _ensure_obs_id_fields(adata, data_path)

    return adata


def filter_by_protocol(adata, args, batch_name=None):
    """Filter adata by protocol version (v2, v3, or both).

    Uses args.protocol if set; otherwise falls back to the original
    batch-name endswith filter for backward compatibility.

    Protocol is determined from the meta_batch_id field.  The naming
    convention is ``<donor>-<modality><version><sample>``, e.g.
    ``1-n3A`` (nuclei, v3, sample A) or ``2-c2A`` (cell, v2, sample A).
    The version digit is the second character of the last 3-char suffix.

    Args:
        adata: AnnData with 'meta_batch_id' in obs.
        args: Namespace with .protocol (str or None) and .batch (str).
        batch_name: explicit batch name override (used by some callers).

    Returns:
        Filtered AnnData copy.
    """
    protocol = getattr(args, 'protocol', None)

    if protocol is not None:
        # Determine modality letter from existing batch arg or from data
        # e.g. batch="n3A" -> modality_char='n'; batch="c3A" -> 'c'
        batch_str = (batch_name or args.batch or '').strip()
        if batch_str and batch_str[0] in ('n', 'c'):
            modality_char = batch_str[0]
        else:
            # Try to infer from the data
            sample_meta = adata.obs['meta_batch_id'].iloc[0] if len(adata) > 0 else ''
            # e.g. '1-n3A' -> last 3 chars -> 'n3A' -> 'n'
            suffix = sample_meta.split('-')[-1] if '-' in sample_meta else sample_meta
            modality_char = suffix[0] if suffix and suffix[0] in ('n', 'c') else 'n'

        # sample letter from existing batch arg, e.g. 'n3A' -> 'A'
        sample_char = batch_str[-1] if batch_str and batch_str[-1].isalpha() else 'A'

        if protocol == 'both':
            # Keep both v2 and v3 for this modality+sample
            v2_suffix = f'{modality_char}2{sample_char}'
            v3_suffix = f'{modality_char}3{sample_char}'
            mask = (adata.obs['meta_batch_id'].str.endswith(v2_suffix) |
                    adata.obs['meta_batch_id'].str.endswith(v3_suffix))
            adata = adata[mask].copy()
            print(f"Protocol filter: both ({v2_suffix} + {v3_suffix}), "
                  f"{adata.shape[0]} cells remaining")
        elif protocol == 'v2':
            target = f'{modality_char}2{sample_char}'
            adata = adata[adata.obs['meta_batch_id'].str.endswith(target)].copy()
            print(f"Protocol filter: v2 ({target}), {adata.shape[0]} cells remaining")
        elif protocol == 'v3':
            target = f'{modality_char}3{sample_char}'
            adata = adata[adata.obs['meta_batch_id'].str.endswith(target)].copy()
            print(f"Protocol filter: v3 ({target}), {adata.shape[0]} cells remaining")
    else:
        # Original behavior: filter by batch name suffix
        if batch_name is None:
            batch_name = args.batch
        batch_name = batch_name.strip()
        adata = adata[adata.obs['meta_batch_id'].str.endswith(batch_name)].copy()
        print(f"Batch filter: {batch_name}, {adata.shape[0]} cells remaining")

    return adata

def balanced_split_by_cell_type(adata, args, train_size=0.6, valid_size=0.2, test_size=0.2, min_samples=50):
    # Extract the data matrix, cell types, and observation annotations
    print("-----balanced_split_by_cell_type-----")
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X

    if X.shape[0] == 0:
        # If the input anndata is empty, return empty splits
        X_empty = np.empty((0, X.shape[1]), dtype=X.dtype)
        obs_empty = adata.obs.iloc[0:0]
        print("No data available after splitting. Skipping...")
        return (X_empty, obs_empty), (X_empty, obs_empty), (X_empty, obs_empty), adata.var

    y = adata.obs['cell_type']
    print("all cell type: ",y)
    print("data shape: ",X.shape)
    if args.subset:
        # Subset to only 10% of each cell type
        print("creating subset of data")
        indices_by_celltype = {}
        for cell_type in y.unique():
            print(cell_type)
            mask = y == cell_type
            cell_indices = np.where(mask)[0]
            print(len(cell_indices))
            # Take 10% of indices for this cell type
            subset_size = max(int(len(cell_indices) * args.subset_ratio), min(min_samples,int(len(cell_indices))))# Ensure at least 1 sample
            subset_indices = np.random.RandomState(42).choice(cell_indices, subset_size, replace=False)
            indices_by_celltype[cell_type] = subset_indices

        if not indices_by_celltype:
            subset_idx = np.array([], dtype=int)
            X = X[subset_idx]
            y = y.iloc[subset_idx] if hasattr(y, 'iloc') else y[subset_idx]
            print("after filtering for batch: ",args.batch, "no cell type selected for cell type: ", y.unique())

        else:
            # Combine all selected indices
            selected_indices = np.concatenate(list(indices_by_celltype.values()))

            # Subset the data
            X = X[selected_indices]
            y = y.iloc[selected_indices] if hasattr(y, 'iloc') else y[selected_indices]
            subset_idx = selected_indices
    else:
        subset_idx = np.arange(X.shape[0])

    # First, split into train + valid, and test
    X_train_valid, X_test, y_train_valid, y_test, train_valid_idx, test_idx = train_test_split(
        X, y, np.arange(X.shape[0]), stratify=y, test_size=test_size, random_state=42)

    # Then, split the train_valid into train and valid
    valid_ratio = valid_size / (train_size + valid_size)  # Adjust valid ratio accordingly
    X_train, X_valid, y_train, y_valid, train_idx, valid_idx = train_test_split(
        X_train_valid, y_train_valid, train_valid_idx, stratify=y_train_valid, test_size=valid_ratio, random_state=42)

    # Map back to original indices if subsetting was applied
    if args.subset:
        train_idx = subset_idx[train_idx]
        valid_idx = subset_idx[valid_idx]
        test_idx = subset_idx[test_idx]

    # Subset the obs and var from the AnnData object
    obs_train = adata.obs.iloc[train_idx]
    obs_valid = adata.obs.iloc[valid_idx]
    obs_test = adata.obs.iloc[test_idx]
    var = adata.var

    if X_train.shape[0] == 0 and X_valid.shape[0] == 0 and X_test.shape[0] == 0:
        print(f"No data available after splitting. Skipping...")
        empty_dataset = myDataset(np.empty((0, adata.n_vars)), pd.DataFrame(columns=adata.obs.columns), adata.var)
        empty_loader = DataLoader(empty_dataset, batch_size=args.batch_size)
        return empty_loader, empty_loader, empty_loader

    return (X_train, obs_train), (X_valid, obs_valid), (X_test, obs_test), var

def _modality_from_batch(batch_name):
    """'n3A' / '1-n3A' -> 'nuclei', 'c3A' -> 'cell' (modality letter = first char of the 3-char suffix)."""
    b = (batch_name or "").strip()
    if len(b) < 3:
        return None
    return {"n": "nuclei", "c": "cell"}.get(b[-3].lower())


def apply_train_exclusions(X, obs, args, modality, partition="train"):
    """Drop held-out cell types / donors from a TRAIN or VALID partition (ablations, 2026-11-04).

    Applied AFTER balanced_split_by_cell_type so the stratified split (random_state=42, which draws
    per class in sequence) is untouched: the test partition stays identical to data/split_manifest.tsv
    and the benchmark cache, and the held-out observations are still scored there.

    args.exclude_cell_types : comma list of cell types removed from training (N-1 type ablation)
    args.holdout_donors     : comma list of ``<modality>:<donor>`` (e.g. ``nuclei:5,cell:12``); the donor
                              is the prefix of meta_batch_id (``<donor>-<modality><version><sample>``),
                              the same field data_split.py uses for the manifest's donor column.
    """
    n0 = X.shape[0]
    keep = np.ones(n0, dtype=bool)
    ex_ct = [c.strip() for c in (getattr(args, "exclude_cell_types", None) or "").split(",") if c.strip()]
    if ex_ct:
        ct = obs["cell_type"].astype(str).to_numpy()
        keep &= ~np.isin(ct, ex_ct)
    ex_donor = [d.strip() for d in (getattr(args, "holdout_donors", None) or "").split(",") if d.strip()]
    if ex_donor:
        mine = []
        for spec in ex_donor:
            if ":" not in spec:
                raise ValueError(f"--holdout_donors entries must be <modality>:<donor>, got '{spec}'")
            mod, donor = spec.split(":", 1)
            if mod.strip() not in ("nuclei", "cell"):
                raise ValueError(f"--holdout_donors modality must be 'nuclei' or 'cell', got '{mod}'")
            if mod.strip() == modality:
                mine.append(donor.strip())
        if mine:
            src = obs["meta_batch_id"] if "meta_batch_id" in obs.columns else obs["batch_id"]
            donors = np.array([str(x).split(".")[0].split("-")[0] for x in src.to_numpy()], dtype=object)
            keep &= ~np.isin(donors, mine)
    if keep.all():
        return X, obs
    print(f"[train exclusions] {partition} {modality}: removed {int((~keep).sum())} of {n0} "
          f"(exclude_cell_types={ex_ct or None}, holdout_donors={ex_donor or None}); {int(keep.sum())} remain")
    return X[keep], obs.iloc[np.where(keep)[0]]


def create_dataloaders(adata, args):
    args.batch = args.batch.strip()
    adata = filter_by_protocol(adata, args)
    print(adata)

    print("raw datas shape: ",adata.X.shape)
    if adata.X.shape[0] == 0:
        print("No data available after filtering for batch: ", args.batch)
        return None, None, None

    # Perform balanced train-valid-test split
    (X_train, obs_train), (X_valid, obs_valid), (X_test, obs_test), var = balanced_split_by_cell_type(adata,args)

    train_dataset = myDataset(X_train, obs_train, var)
    valid_dataset = myDataset(X_valid, obs_valid, var)
    test_dataset = myDataset(X_test, obs_test, var)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader

def create_dataloaders_by_cell_type(adata, args, batch_name=None):

    args.batch = args.batch.strip()

    adata = filter_by_protocol(adata, args, batch_name=batch_name)
    print(adata)

    print("raw datas shape: ",adata.X.shape)
    # Only override adata.X with a normalized layer if NOT using raw counts.
    # For vae_nb with --raw_data_normalization raw, keep adata.X as-is (raw counts).
    raw_mode = getattr(args, 'use_raw_data', False) and getattr(args, 'raw_data_normalization', 'log1p') == 'raw'
    if not raw_mode:
        # if log1p layer exists, use it
        if 'log1p' in adata.layers:
            adata.X = adata.layers['log1p']
        elif 'cpm' in adata.layers:
            adata.X = adata.layers['cpm']
        elif 'raw' in adata.layers:
            adata.X = adata.layers['raw']
        else:
            print("No raw data normalization layer found in adata.layers")

    if adata.X.shape[0] == 0:
        print("No data available after filtering for batch: ", batch_name)
        return None, None, None

    # Perform balanced train-valid-test split
    (X_train, obs_train), (X_valid, obs_valid), (X_test, obs_test), var = balanced_split_by_cell_type(adata,args)

    # Ablations (2026-11-04): held-out cell types / donors leave train and valid only; test is untouched.
    _modality = _modality_from_batch(batch_name if batch_name is not None else args.batch)
    X_train, obs_train = apply_train_exclusions(X_train, obs_train, args, _modality, "train")
    X_valid, obs_valid = apply_train_exclusions(X_valid, obs_valid, args, _modality, "valid")

    # for each cell type, create a dataset for train, valid, and test
    cell_type_datasets_train = {}
    cell_type_datasets_valid = {}
    cell_type_datasets_test = {}
    for cell_type in obs_train["cell_type"].unique():
        cell_type_dataset = myDataset(X_train[obs_train["cell_type"] == cell_type], obs_train[obs_train["cell_type"] == cell_type], var)
        cell_type_datasets_train[cell_type] = cell_type_dataset

        cell_type_dataset_valid = myDataset(X_valid[obs_valid["cell_type"] == cell_type], obs_valid[obs_valid["cell_type"] == cell_type], var)
        cell_type_datasets_valid[cell_type] = cell_type_dataset_valid

        cell_type_dataset_test = myDataset(X_test[obs_test["cell_type"] == cell_type], obs_test[obs_test["cell_type"] == cell_type], var)
        cell_type_datasets_test[cell_type] = cell_type_dataset_test

    # create a dataloader for each cell type
    train_loaders = {}
    valid_loaders = {}
    test_loaders = {}
    for cell_type in obs_train["cell_type"].unique():
        train_loaders[cell_type] = DataLoader(cell_type_datasets_train[cell_type], batch_size=args.batch_size, shuffle=True)
        valid_loaders[cell_type] = DataLoader(cell_type_datasets_valid[cell_type], batch_size=args.batch_size, shuffle=False)
        test_loaders[cell_type] = DataLoader(cell_type_datasets_test[cell_type], batch_size=args.batch_size, shuffle=False)

    return train_loaders, valid_loaders, test_loaders
