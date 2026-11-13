"""
scVI baseline for cross-modality translation (nucleus -> cell RNA-seq).

Train scVI on combined cell+nucleus data with modality as batch covariate.
Use `get_normalized_expression(transform_batch=...)` to produce counterfactual
expression: "what would this nucleus look like as a cell?"

Usage:
    python analysis/baseline_scvi.py \
        --n_latent 30 --n_layers 2 --n_hidden 256 \
        --max_epochs 200 --output_dir run/scvi_baseline

Requires: pip install scvi-tools
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse

try:
    import scvi
except ImportError:
    print("ERROR: scvi-tools not installed. Run: pip install scvi-tools")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Data paths (matches imputeVAE/data.py and baseline_linear_regression.py)
# ---------------------------------------------------------------------------
DATA_ROOT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi"
CELL_DIR = "10X_cells_v2_v3_AIBS"
NUC_DIR = "10X_nuclei_v2_v3_AIBS"
DEFAULT_GENE_LIST = "data/training_gene_list_v3.csv"


def _resolve_raw_path(data_dir):
    return os.path.join(DATA_ROOT, data_dir, "data.h5ad")


def load_and_filter(data_dir, batch_suffix, gene_list):
    """Load raw data, filter to batch + common genes + common CTs + gene list.

    Matches the filtering in baseline_linear_regression.py exactly.
    Returns adata with RAW COUNTS in .X (scVI needs counts, not log1p).
    """
    adata = ad.read_h5ad(_resolve_raw_path(data_dir))

    # Add meta_batch_id if missing
    if 'meta_batch_id' not in adata.obs:
        adata.obs['meta_batch_id'] = adata.obs['batch_id'].apply(
            lambda x: x.split('.')[0]
        )

    # Filter to v3 protocol batch
    adata = adata[adata.obs['meta_batch_id'].str.endswith(batch_suffix)].copy()

    # Filter to common genes
    common_genes = pd.read_csv('data/common_genes.csv', header=0)['common_genes'].tolist()
    adata = adata[:, adata.var['gene_names'].isin(common_genes)].copy()

    # Filter to common cell types
    common_ct = pd.read_csv('data/common_cell_types.csv', header=0)['common_cell_types'].tolist()
    adata = adata[adata.obs['cell_type'].isin(common_ct)].copy()

    # Filter to gene list
    adata = adata[:, adata.var['gene_names'].isin(gene_list)].copy()

    # Sort by gene name for consistency
    adata = adata[:, adata.var['gene_names'].argsort()].copy()

    # Keep raw counts in .X (scVI requirement)
    if sparse.issparse(adata.X):
        adata.X = adata.X.toarray().astype(np.float32)
    else:
        adata.X = adata.X.astype(np.float32)

    print(f"  Loaded {data_dir} batch={batch_suffix}: {adata.shape}")
    return adata


def load_data(gene_list_path=DEFAULT_GENE_LIST):
    """Load and combine cell + nucleus data. Returns combined adata with raw counts."""
    gene_list = pd.read_csv(gene_list_path, header=None)[0].tolist()

    adata_cell = load_and_filter(CELL_DIR, "c3A", gene_list)
    adata_nuc = load_and_filter(NUC_DIR, "n3A", gene_list)

    # Add modality labels
    adata_cell.obs["modality"] = "cell"
    adata_nuc.obs["modality"] = "nucleus"

    # Make var_names unique before concat (use gene_names as index)
    adata_cell.var_names = adata_cell.var['gene_names'].values
    adata_nuc.var_names = adata_nuc.var['gene_names'].values

    adata = ad.concat([adata_cell, adata_nuc], label="modality_src",
                      keys=["cell", "nucleus"], merge="same")
    # Use our own modality column (concat's label creates a new one)
    # The obs["modality"] we set above is preserved in concat

    print(f"\nCombined: {adata.n_obs} samples x {adata.n_vars} genes")
    print(f"  Cell: {(adata.obs['modality'] == 'cell').sum()}")
    print(f"  Nucleus: {(adata.obs['modality'] == 'nucleus').sum()}")
    print(f"  Cell types: {adata.obs['cell_type'].nunique()}")

    return adata


def run_scvi(adata, n_latent=30, n_layers=2, n_hidden=256,
             max_epochs=200, batch_size=256, lr=1e-3,
             output_dir="run/scvi_baseline"):
    """Train scVI and generate translated expression."""
    os.makedirs(output_dir, exist_ok=True)

    # Verify raw counts
    max_val = adata.X.max()
    if max_val < 50:
        print(f"WARNING: max(X)={max_val:.1f} — data may be normalized. scVI needs raw counts.")

    # Setup: modality as batch covariate
    scvi.model.SCVI.setup_anndata(adata, batch_key="modality")

    model = scvi.model.SCVI(
        adata,
        n_latent=n_latent,
        n_layers=n_layers,
        n_hidden=n_hidden,
        gene_likelihood="nb",
    )

    print(f"\nscVI model: {n_latent}d latent, {n_layers} layers, {n_hidden} hidden")
    print(f"Training for {max_epochs} epochs, batch_size={batch_size}, lr={lr}")

    t0 = time.time()
    model.train(
        max_epochs=max_epochs,
        batch_size=batch_size,
        plan_kwargs={"lr": lr},
        early_stopping=True,
        early_stopping_patience=15,
    )
    elapsed = time.time() - t0
    print(f"Training completed in {elapsed:.0f}s ({elapsed / 60:.1f}min)")

    # Save model
    model_path = os.path.join(output_dir, "scvi_model")
    model.save(model_path, overwrite=True)
    print(f"Model saved to {model_path}")

    # -----------------------------------------------------------------------
    # Generate counterfactual expression
    # -----------------------------------------------------------------------
    nuc_mask = adata.obs["modality"] == "nucleus"
    print(f"\nTranslating {nuc_mask.sum()} nuclei -> cell expression...")

    imputed = model.get_normalized_expression(
        adata=adata[nuc_mask],
        transform_batch="cell",
        library_size=1e4,
    )

    if isinstance(imputed, pd.DataFrame):
        imputed_matrix = imputed.values
    else:
        imputed_matrix = np.array(imputed)

    # Apply log1p to match our VAE's output space
    imputed_log1p = np.log1p(imputed_matrix).astype(np.float32)
    print(f"Imputed matrix shape: {imputed_log1p.shape}")

    # Also get cell reconstruction for comparison
    cell_mask = adata.obs["modality"] == "cell"
    cell_recon = model.get_normalized_expression(
        adata=adata[cell_mask],
        library_size=1e4,
    )
    if isinstance(cell_recon, pd.DataFrame):
        cell_recon = cell_recon.values
    cell_recon_log1p = np.log1p(cell_recon).astype(np.float32)

    # Get latent representations
    latent = model.get_latent_representation(adata)

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    np.save(os.path.join(output_dir, "imputed_cell_from_nucleus.npy"), imputed_log1p)
    np.save(os.path.join(output_dir, "cell_reconstruction.npy"), cell_recon_log1p)
    np.save(os.path.join(output_dir, "latent_representations.npy"), latent)

    # Save metadata
    nuc_obs = adata.obs[nuc_mask][["modality", "cell_type"]].copy()
    nuc_obs.to_csv(os.path.join(output_dir, "nucleus_obs.csv"))

    cell_obs = adata.obs[cell_mask][["modality", "cell_type"]].copy()
    cell_obs.to_csv(os.path.join(output_dir, "cell_obs.csv"))

    pd.Series(adata.var_names).to_csv(
        os.path.join(output_dir, "gene_names.csv"), index=False, header=False
    )

    # Save training history
    history = model.history
    history_dict = {k: v.values.tolist() for k, v in history.items()}
    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump(history_dict, f)

    args_dict = {
        "method": "scvi",
        "n_latent": n_latent,
        "n_layers": n_layers,
        "n_hidden": n_hidden,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "gene_likelihood": "nb",
        "n_genes": adata.n_vars,
        "n_cells": int(cell_mask.sum()),
        "n_nuclei": int(nuc_mask.sum()),
        "training_time_s": elapsed,
        "scvi_version": scvi.__version__,
    }
    with open(os.path.join(output_dir, "arguments.json"), "w") as f:
        json.dump(args_dict, f, indent=2)

    print(f"\nOutputs saved to {output_dir}/")
    return model, imputed_log1p


def main():
    parser = argparse.ArgumentParser(description="scVI baseline for cell<->nucleus translation")
    parser.add_argument("--n_latent", type=int, default=30)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_hidden", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, default="run/scvi_baseline")
    parser.add_argument("--gene_list", type=str, default=DEFAULT_GENE_LIST)
    args = parser.parse_args()

    adata = load_data(gene_list_path=args.gene_list)
    run_scvi(
        adata,
        n_latent=args.n_latent,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
