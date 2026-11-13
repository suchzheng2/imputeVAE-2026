"""
scVI baseline (split-aware).  Train on train_cell + train_nuc with modality as the
batch covariate; translate eval nuclei with transform_batch="cell".

Library size: scVI's decoder outputs proportions that must be multiplied by a
library size.  The old benchmark used a fixed 1e4, which is far below the real
v3 cell library (~15-30k UMI) and put every scale-sensitive metric (R2, MSE,
Wasserstein) at a disadvantage for no modelling reason.  Here each eval nucleus
is decoded at the median raw library size of TRAINING cells of its cell type.

Usage:
    python -m analysis.benchmark.train_scvi --seed 0 --eval_partition test \
        --output_dir run/benchmark/test/scvi_seed0
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import pandas as pd
import anndata as ad

try:
    import scvi
except ImportError:
    print("ERROR: scvi-tools not installed in this environment (use conda env scvi2)")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from analysis.benchmark.data_split import load_benchmark_data, save_standard_outputs, ct_median_library


def decode_per_ct(model, adata_eval, ct_labels, lib_by_ct, transform_batch="cell"):
    """get_normalized_expression with a per-cell-type library size."""
    out = np.zeros((adata_eval.n_obs, adata_eval.n_vars), dtype=np.float32)
    for ct in sorted(set(ct_labels)):
        idx = np.where(ct_labels == ct)[0]
        lib = lib_by_ct.get(ct, lib_by_ct["_global"])
        x = model.get_normalized_expression(adata=adata_eval, indices=idx, transform_batch=transform_batch,
                                            library_size=lib, return_numpy=True)
        out[idx] = np.asarray(x, dtype=np.float32)
    return out


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
    p.add_argument("--output_dir", default="run/benchmark/test/scvi_seed0")
    args = p.parse_args()

    scvi.settings.seed = args.seed
    print("=" * 60)
    print(f"Split-aware benchmark: scVI (seed {args.seed})")
    print("=" * 60)
    data = load_benchmark_data(args.eval_partition, return_raw_counts=True, include_eval_cells=False)
    gene_names = data["gene_names"]

    cell_tr, nuc_tr = data["cell_train"], data["nuc_train"]
    cell_tr.obs["modality"] = "cell"
    nuc_tr.obs["modality"] = "nucleus"
    adata_train = ad.concat([cell_tr, nuc_tr], merge="same", index_unique="-")
    adata_train.obs["modality"] = adata_train.obs["modality"].astype("category")
    print(f"Train: {adata_train.n_obs} obs ({cell_tr.n_obs} cells, {nuc_tr.n_obs} nuclei)")

    lib_by_ct = ct_median_library(cell_tr.X, cell_tr.obs["cell_type"].values.astype(str), data["common_cts"])
    print("Per-CT median training-cell library:", {k: int(v) for k, v in lib_by_ct.items()})

    scvi.model.SCVI.setup_anndata(adata_train, batch_key="modality")
    model = scvi.model.SCVI(adata_train, n_latent=args.n_latent, n_layers=args.n_layers,
                            n_hidden=args.n_hidden, gene_likelihood="nb")
    t0 = time.time()
    model.train(max_epochs=args.max_epochs, batch_size=args.batch_size, plan_kwargs={"lr": args.lr},
                early_stopping=True, early_stopping_patience=15)
    elapsed = time.time() - t0
    print(f"Training: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save(os.path.join(args.output_dir, "scvi_model"), overwrite=True)

    adata_eval = data["nuc_eval"]
    adata_eval.obs["modality"] = pd.Categorical(["nucleus"] * adata_eval.n_obs,
                                                categories=adata_train.obs["modality"].cat.categories)
    scvi.model.SCVI.setup_anndata(adata_eval, batch_key="modality")
    ct_eval = adata_eval.obs["cell_type"].values.astype(str)
    print(f"Translating {adata_eval.n_obs} eval nuclei -> cell (per-CT library)...")
    imputed = decode_per_ct(model, adata_eval, ct_eval, lib_by_ct)
    imputed_log1p = np.log1p(imputed).astype(np.float32)

    n_params = sum(p.numel() for p in model.module.parameters())
    args_dict = {
        "method": "scVI", "seed": args.seed, "n_latent": args.n_latent, "n_layers": args.n_layers,
        "n_hidden": args.n_hidden, "max_epochs": args.max_epochs,
        "epochs_trained": len(model.history.get("elbo_train", [])), "batch_size": args.batch_size,
        "lr": args.lr, "gene_likelihood": "nb", "n_params": int(n_params), "training_time_s": elapsed,
        "scvi_version": scvi.__version__, "early_stopping": True, "early_stopping_patience": 15,
        "checkpoint": "best (early stopping on scvi internal 10% validation split of training data)",
        "decode_library": "median raw library of training cells per cell type",
        "library_by_ct": lib_by_ct, "eval_partition": args.eval_partition,
    }
    save_standard_outputs(args.output_dir, imputed_log1p, adata_eval.obs[["obs_id", "cell_type", "donor"]],
                          gene_names, args_dict, data["split_info"])
    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump({k: v.values.ravel().tolist() for k, v in model.history.items()}, f)
    print("Done.")


if __name__ == "__main__":
    main()
