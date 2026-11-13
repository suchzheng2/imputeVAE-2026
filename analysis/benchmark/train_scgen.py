"""
scGen-style baseline (split-aware): vanilla scVI (no batch key) on training cells + nuclei,
per-cell-type latent shift vectors (train cell mean - train nucleus mean), encode eval nuclei,
add the shift, decode at the per-cell-type median training-cell library size.

Usage:
    python -m analysis.benchmark.train_scgen --seed 0 --eval_partition test \
        --output_dir run/benchmark/test/scgen_seed0
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import anndata as ad
import torch

try:
    import scvi
except ImportError:
    print("ERROR: scvi-tools not installed in this environment (use conda env scvi2)")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from analysis.benchmark.data_split import load_benchmark_data, save_standard_outputs, ct_median_library


def decode_from_latent(model, z_array, library_per_cell):
    module = model.module
    device = next(module.parameters()).device
    z = torch.tensor(z_array, dtype=torch.float32, device=device)
    library = torch.tensor(np.log(library_per_cell), dtype=torch.float32, device=device).reshape(-1, 1)
    batch_index = torch.zeros(z.shape[0], 1, dtype=torch.long, device=device)
    results = []
    with torch.no_grad():
        for i in range(0, z.shape[0], 4096):
            gen = module.generative(z[i:i+4096], library[i:i+4096], batch_index[i:i+4096])
            px = gen["px"]
            mean = px.mean if hasattr(px, "mean") else px.mu
            results.append(mean.cpu().numpy())
    return np.concatenate(results, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_latent", type=int, default=30)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--n_hidden", type=int, default=256)
    p.add_argument("--max_epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_partition", default="test", choices=["valid", "test"])
    p.add_argument("--output_dir", default="run/benchmark/test/scgen_seed0")
    args = p.parse_args()

    scvi.settings.seed = args.seed
    print("=" * 60)
    print(f"Split-aware benchmark: scGen-style latent arithmetic (seed {args.seed})")
    print("=" * 60)
    data = load_benchmark_data(args.eval_partition, return_raw_counts=True, include_eval_cells=False)
    gene_names = data["gene_names"]
    cell_tr, nuc_tr = data["cell_train"], data["nuc_train"]
    cell_tr.obs["modality"] = "cell"
    nuc_tr.obs["modality"] = "nucleus"
    adata_train = ad.concat([cell_tr, nuc_tr], merge="same", index_unique="-")
    lib_by_ct = ct_median_library(cell_tr.X, cell_tr.obs["cell_type"].values.astype(str), data["common_cts"])

    scvi.model.SCVI.setup_anndata(adata_train)          # no batch key: vanilla VAE
    model = scvi.model.SCVI(adata_train, n_latent=args.n_latent, n_layers=args.n_layers,
                            n_hidden=args.n_hidden, gene_likelihood="nb")
    t0 = time.time()
    model.train(max_epochs=args.max_epochs, batch_size=args.batch_size, plan_kwargs={"lr": args.lr},
                early_stopping=True, early_stopping_patience=15)
    elapsed = time.time() - t0
    print(f"Training: {elapsed:.0f}s")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save(os.path.join(args.output_dir, "scvi_model"), overwrite=True)

    train_latent = model.get_latent_representation(adata_train)
    n_cell = cell_tr.n_obs
    cell_latent, nuc_latent = train_latent[:n_cell], train_latent[n_cell:]
    cell_cts = cell_tr.obs["cell_type"].values.astype(str)
    nuc_cts = nuc_tr.obs["cell_type"].values.astype(str)

    adata_eval = data["nuc_eval"]
    ct_eval = adata_eval.obs["cell_type"].values.astype(str)
    shifts = {}
    for ct in sorted(set(ct_eval)):
        zc, zn = cell_latent[cell_cts == ct], nuc_latent[nuc_cts == ct]
        shifts[ct] = (zc.mean(axis=0) - zn.mean(axis=0)) if (len(zc) and len(zn)) else None
    global_shift = cell_latent.mean(axis=0) - nuc_latent.mean(axis=0)

    scvi.model.SCVI.setup_anndata(adata_eval)
    z_eval = model.get_latent_representation(adata_eval)
    z_shift = z_eval.copy()
    lib = np.empty(adata_eval.n_obs, dtype=np.float64)
    for i, ct in enumerate(ct_eval):
        s = shifts.get(ct)
        z_shift[i] += s if s is not None else global_shift
        lib[i] = lib_by_ct.get(ct, lib_by_ct["_global"])
    imputed = decode_from_latent(model, z_shift, lib)
    imputed_log1p = np.log1p(imputed).astype(np.float32)

    n_params = sum(p.numel() for p in model.module.parameters())
    args_dict = {
        "method": "scGen", "seed": args.seed, "approach": "per-CT latent arithmetic on vanilla scVI",
        "n_latent": args.n_latent, "n_layers": args.n_layers, "n_hidden": args.n_hidden,
        "max_epochs": args.max_epochs, "epochs_trained": len(model.history.get("elbo_train", [])),
        "batch_size": args.batch_size, "lr": args.lr, "gene_likelihood": "nb", "batch_correction": False,
        "n_params": int(n_params), "training_time_s": elapsed, "scvi_version": scvi.__version__,
        "checkpoint": "best (early stopping)", "decode_library": "median raw library of training cells per cell type",
        "library_by_ct": lib_by_ct, "eval_partition": args.eval_partition,
    }
    save_standard_outputs(args.output_dir, imputed_log1p, adata_eval.obs[["obs_id", "cell_type", "donor"]],
                          gene_names, args_dict, data["split_info"])
    with open(os.path.join(args.output_dir, "shift_vectors.json"), "w") as f:
        json.dump({ct: (s.tolist() if s is not None else None) for ct, s in shifts.items()} |
                  {"_global": global_shift.tolist()}, f)
    print("Done.")


if __name__ == "__main__":
    main()
