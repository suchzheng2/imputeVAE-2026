"""
scANVI baseline for cross-modality translation (nucleus -> cell RNA-seq).

Train scANVI on combined cell+nucleus data with modality as batch covariate
and cell_type as the semi-supervised label. Use `get_normalized_expression(
transform_batch=...)` to produce counterfactual expression.

This is the fair, strongest-possible scANVI setup: it gets cell-type labels
(semi-supervised) and uses its intended counterfactual decode.

Usage:
    python analysis/baseline_scanvi.py \
        --n_latent 30 --n_layers 2 --n_hidden 256 \
        --max_epochs_scvi 200 --max_epochs_scanvi 200 \
        --output_dir run/scanvi_baseline

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

# Reuse data loading from the scVI baseline (identical filtering)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from analysis.baseline_scvi import load_data


def run_scanvi(adata, n_latent=30, n_layers=2, n_hidden=256,
               max_epochs_scvi=200, max_epochs_scanvi=200,
               batch_size=256, lr=1e-3,
               output_dir="run/scanvi_baseline"):
    """Train scANVI (via scVI pretrain) and generate translated expression."""
    os.makedirs(output_dir, exist_ok=True)

    # Verify raw counts
    max_val = adata.X.max()
    if max_val < 50:
        print(f"WARNING: max(X)={max_val:.1f} — data may be normalized. scANVI needs raw counts.")

    # -----------------------------------------------------------------------
    # Step 1: Pre-train scVI (unsupervised)
    # -----------------------------------------------------------------------
    scvi.model.SCVI.setup_anndata(adata, batch_key="modality")

    scvi_model = scvi.model.SCVI(
        adata,
        n_latent=n_latent,
        n_layers=n_layers,
        n_hidden=n_hidden,
        gene_likelihood="nb",
    )

    print(f"\n[Step 1] Pre-training scVI: {n_latent}d latent, {n_layers} layers, {n_hidden} hidden")
    print(f"  max_epochs={max_epochs_scvi}, batch_size={batch_size}, lr={lr}")

    t0 = time.time()
    scvi_model.train(
        max_epochs=max_epochs_scvi,
        batch_size=batch_size,
        plan_kwargs={"lr": lr},
        early_stopping=True,
        early_stopping_patience=15,
    )
    scvi_time = time.time() - t0
    print(f"  scVI pre-training completed in {scvi_time:.0f}s ({scvi_time / 60:.1f}min)")

    # -----------------------------------------------------------------------
    # Step 2: Initialize scANVI from scVI + train with labels
    # -----------------------------------------------------------------------
    print(f"\n[Step 2] Training scANVI with cell_type labels (semi-supervised)")
    print(f"  max_epochs={max_epochs_scanvi}")

    scanvi_model = scvi.model.SCANVI.from_scvi_model(
        scvi_model,
        unlabeled_category="Unknown",
        labels_key="cell_type",
    )

    t1 = time.time()
    scanvi_model.train(
        max_epochs=max_epochs_scanvi,
        batch_size=batch_size,
        plan_kwargs={"lr": lr},
        early_stopping=True,
        early_stopping_patience=15,
    )
    scanvi_time = time.time() - t1
    total_time = time.time() - t0
    print(f"  scANVI training completed in {scanvi_time:.0f}s ({scanvi_time / 60:.1f}min)")
    print(f"  Total training time: {total_time:.0f}s ({total_time / 60:.1f}min)")

    # Save model
    model_path = os.path.join(output_dir, "scanvi_model")
    scanvi_model.save(model_path, overwrite=True)
    print(f"  Model saved to {model_path}")

    # -----------------------------------------------------------------------
    # Step 3: Generate counterfactual expression (nuclei decoded as cells)
    # -----------------------------------------------------------------------
    nuc_mask = adata.obs["modality"] == "nucleus"
    print(f"\n[Step 3] Translating {nuc_mask.sum()} nuclei -> cell expression...")

    imputed = scanvi_model.get_normalized_expression(
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
    print(f"  Imputed matrix shape: {imputed_log1p.shape}")

    # Also get cell reconstruction for comparison
    cell_mask = adata.obs["modality"] == "cell"
    cell_recon = scanvi_model.get_normalized_expression(
        adata=adata[cell_mask],
        library_size=1e4,
    )
    if isinstance(cell_recon, pd.DataFrame):
        cell_recon = cell_recon.values
    cell_recon_log1p = np.log1p(cell_recon).astype(np.float32)

    # Get latent representations
    latent = scanvi_model.get_latent_representation(adata)

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
    history = scanvi_model.history
    history_dict = {k: v.values.tolist() for k, v in history.items()}
    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump(history_dict, f)

    args_dict = {
        "method": "scanvi",
        "n_latent": n_latent,
        "n_layers": n_layers,
        "n_hidden": n_hidden,
        "max_epochs_scvi": max_epochs_scvi,
        "max_epochs_scanvi": max_epochs_scanvi,
        "batch_size": batch_size,
        "lr": lr,
        "gene_likelihood": "nb",
        "labels_key": "cell_type",
        "n_genes": adata.n_vars,
        "n_cells": int(cell_mask.sum()),
        "n_nuclei": int(nuc_mask.sum()),
        "n_cell_types": adata.obs["cell_type"].nunique(),
        "training_time_scvi_s": scvi_time,
        "training_time_scanvi_s": scanvi_time,
        "training_time_total_s": total_time,
        "scvi_version": scvi.__version__,
    }
    with open(os.path.join(output_dir, "arguments.json"), "w") as f:
        json.dump(args_dict, f, indent=2)

    print(f"\nOutputs saved to {output_dir}/")
    return scanvi_model, imputed_log1p


def main():
    parser = argparse.ArgumentParser(
        description="scANVI baseline for cell<->nucleus translation"
    )
    parser.add_argument("--n_latent", type=int, default=30)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_hidden", type=int, default=256)
    parser.add_argument("--max_epochs_scvi", type=int, default=200)
    parser.add_argument("--max_epochs_scanvi", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, default="run/scanvi_baseline")
    parser.add_argument("--gene_list", type=str, default="data/training_gene_list_v3.csv")
    args = parser.parse_args()

    adata = load_data(gene_list_path=args.gene_list)
    run_scanvi(
        adata,
        n_latent=args.n_latent,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        max_epochs_scvi=args.max_epochs_scvi,
        max_epochs_scanvi=args.max_epochs_scanvi,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
