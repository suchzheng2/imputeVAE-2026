"""
scGen-style baseline for cross-modality translation (nucleus -> cell RNA-seq).

Implements scGen's latent arithmetic approach (Lotfollahi et al. 2019):
1. Train a VAE (scVI without batch correction) on combined cell+nucleus data
2. Compute per-cell-type shift vectors: delta_z = mean(z_cell) - mean(z_nuc)
3. For each nucleus: z_imputed = z_nuc + delta_z_ct
4. Decode z_imputed -> imputed cell expression

This is the directed-transformation family baseline: a fixed additive shift
in latent space, computed per cell type.

Usage:
    python analysis/baseline_scgen.py \
        --n_latent 30 --n_layers 2 --n_hidden 256 \
        --max_epochs 200 --output_dir run/scgen_baseline

Requires: pip install scvi-tools
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import pandas as pd
import torch

try:
    import scvi
except ImportError:
    print("ERROR: scvi-tools not installed. Run: pip install scvi-tools")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from analysis.baseline_scvi import load_data


def decode_from_latent(model, z_array, library_size=1e4):
    """Decode expression from latent vectors using scVI's internal module.

    Args:
        model: trained scVI model
        z_array: numpy array (n_cells, n_latent)
        library_size: target library size for normalization

    Returns:
        numpy array (n_cells, n_genes) of normalized expression (counts per library_size)
    """
    module = model.module
    device = next(module.parameters()).device

    z = torch.tensor(z_array, dtype=torch.float32).to(device)
    library = torch.full((z.shape[0], 1), np.log(library_size),
                         dtype=torch.float32).to(device)
    # No batch correction — single batch index
    batch_index = torch.zeros(z.shape[0], 1, dtype=torch.long).to(device)

    # Decode in chunks to avoid OOM
    chunk_size = 4096
    results = []
    with torch.no_grad():
        for i in range(0, z.shape[0], chunk_size):
            z_chunk = z[i:i + chunk_size]
            lib_chunk = library[i:i + chunk_size]
            batch_chunk = batch_index[i:i + chunk_size]

            gen_out = module.generative(z_chunk, lib_chunk, batch_chunk)
            # px is a distribution object; .mean gives the expected counts
            rate = gen_out["px"].mean.cpu().numpy()
            results.append(rate)

    return np.concatenate(results, axis=0)


def run_scgen(adata, n_latent=30, n_layers=2, n_hidden=256,
              max_epochs=200, batch_size=256, lr=1e-3,
              output_dir="run/scgen_baseline", resume=False):
    """Train vanilla VAE and perform scGen-style latent arithmetic."""
    os.makedirs(output_dir, exist_ok=True)

    max_val = adata.X.max()
    if max_val < 50:
        print(f"WARNING: max(X)={max_val:.1f} — data may be normalized. scVI needs raw counts.")

    model_path = os.path.join(output_dir, "scvi_model")

    if resume and os.path.exists(model_path):
        # -----------------------------------------------------------------------
        # Resume: load saved model, skip training
        # -----------------------------------------------------------------------
        print(f"\n[Step 1] Loading saved model from {model_path}")
        scvi.model.SCVI.setup_anndata(adata)
        model = scvi.model.SCVI.load(model_path, adata=adata)
        train_time = 0.0
    else:
        # -----------------------------------------------------------------------
        # Step 1: Train scVI WITHOUT batch_key (vanilla VAE, faithful to scGen)
        # -----------------------------------------------------------------------
        scvi.model.SCVI.setup_anndata(adata)  # No batch_key!

        model = scvi.model.SCVI(
            adata,
            n_latent=n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            gene_likelihood="nb",
        )

        print(f"\n[Step 1] Training vanilla scVI (no batch correction): "
              f"{n_latent}d latent, {n_layers} layers, {n_hidden} hidden")
        print(f"  max_epochs={max_epochs}, batch_size={batch_size}, lr={lr}")

        t0 = time.time()
        model.train(
            max_epochs=max_epochs,
            batch_size=batch_size,
            plan_kwargs={"lr": lr},
            early_stopping=True,
            early_stopping_patience=15,
        )
        train_time = time.time() - t0
        print(f"  Training completed in {train_time:.0f}s ({train_time / 60:.1f}min)")

        model.save(model_path, overwrite=True)

    # -----------------------------------------------------------------------
    # Step 2: Get latent representations
    # -----------------------------------------------------------------------
    print(f"\n[Step 2] Computing latent representations...")
    latent = model.get_latent_representation(adata)
    print(f"  Latent shape: {latent.shape}")

    nuc_mask = adata.obs["modality"].values == "nucleus"
    cell_mask = adata.obs["modality"].values == "cell"
    cell_types = adata.obs["cell_type"].values

    nuc_latent = latent[nuc_mask]
    cell_latent = latent[cell_mask]
    nuc_cts = cell_types[nuc_mask]
    cell_cts = cell_types[cell_mask]

    # -----------------------------------------------------------------------
    # Step 3: Compute per-CT shift vectors (scGen latent arithmetic)
    # -----------------------------------------------------------------------
    print(f"\n[Step 3] Computing per-cell-type shift vectors...")
    unique_cts = np.unique(nuc_cts)
    shifts = {}
    for ct in unique_cts:
        z_cell_ct = cell_latent[cell_cts == ct]
        z_nuc_ct = nuc_latent[nuc_cts == ct]
        if len(z_cell_ct) == 0 or len(z_nuc_ct) == 0:
            print(f"  WARNING: {ct} missing in one modality, using global shift")
            shifts[ct] = None
        else:
            shifts[ct] = z_cell_ct.mean(axis=0) - z_nuc_ct.mean(axis=0)
            print(f"  {ct}: n_cell={len(z_cell_ct)}, n_nuc={len(z_nuc_ct)}, "
                  f"|shift|={np.linalg.norm(shifts[ct]):.3f}")

    # Global shift as fallback
    global_shift = cell_latent.mean(axis=0) - nuc_latent.mean(axis=0)
    print(f"  Global shift |delta|={np.linalg.norm(global_shift):.3f}")

    # Apply shifts
    shifted_latent = nuc_latent.copy()
    for i, ct in enumerate(nuc_cts):
        shift = shifts.get(ct)
        if shift is None:
            shift = global_shift
        shifted_latent[i] += shift

    # -----------------------------------------------------------------------
    # Step 4: Decode shifted latent -> imputed expression
    # -----------------------------------------------------------------------
    print(f"\n[Step 4] Decoding {shifted_latent.shape[0]} shifted latent vectors...")
    imputed_raw = decode_from_latent(model, shifted_latent, library_size=1e4)
    imputed_log1p = np.log1p(imputed_raw).astype(np.float32)
    print(f"  Imputed matrix shape: {imputed_log1p.shape}")
    print(f"  Range: [{imputed_log1p.min():.3f}, {imputed_log1p.max():.3f}]")

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    np.save(os.path.join(output_dir, "imputed_cell_from_nucleus.npy"), imputed_log1p)
    np.save(os.path.join(output_dir, "latent_representations.npy"), latent)
    np.save(os.path.join(output_dir, "shifted_latent.npy"), shifted_latent)

    nuc_obs = adata.obs[nuc_mask][["modality", "cell_type"]].copy()
    nuc_obs.to_csv(os.path.join(output_dir, "nucleus_obs.csv"))

    cell_obs = adata.obs[cell_mask][["modality", "cell_type"]].copy()
    cell_obs.to_csv(os.path.join(output_dir, "cell_obs.csv"))

    pd.Series(adata.var_names).to_csv(
        os.path.join(output_dir, "gene_names.csv"), index=False, header=False
    )

    # Save shift vectors
    shift_dict = {}
    for ct, s in shifts.items():
        shift_dict[ct] = s.tolist() if s is not None else None
    shift_dict["_global"] = global_shift.tolist()
    with open(os.path.join(output_dir, "shift_vectors.json"), "w") as f:
        json.dump(shift_dict, f)

    args_dict = {
        "method": "scgen",
        "approach": "per-CT latent arithmetic (scGen-style)",
        "n_latent": n_latent,
        "n_layers": n_layers,
        "n_hidden": n_hidden,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "gene_likelihood": "nb",
        "batch_correction": False,
        "n_genes": adata.n_vars,
        "n_cells": int(cell_mask.sum()),
        "n_nuclei": int(nuc_mask.sum()),
        "n_cell_types": len(unique_cts),
        "training_time_s": train_time,
        "scvi_version": scvi.__version__,
    }
    with open(os.path.join(output_dir, "arguments.json"), "w") as f:
        json.dump(args_dict, f, indent=2)

    print(f"\nOutputs saved to {output_dir}/")
    return model, imputed_log1p


def main():
    parser = argparse.ArgumentParser(
        description="scGen-style baseline: latent arithmetic for cell<->nucleus translation"
    )
    parser.add_argument("--n_latent", type=int, default=30)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_hidden", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, default="run/scgen_baseline")
    parser.add_argument("--gene_list", type=str, default="data/training_gene_list_v3.csv")
    parser.add_argument("--resume", action="store_true",
                        help="Load saved model instead of retraining")
    args = parser.parse_args()

    adata = load_data(gene_list_path=args.gene_list)
    run_scgen(
        adata,
        n_latent=args.n_latent,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
