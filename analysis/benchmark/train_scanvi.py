"""
scANVI baseline (split-aware).  Pretrain scVI on train_cell + train_nuc (modality as batch),
then scANVI with cell_type labels; translate eval nuclei with transform_batch="cell"
at the per-cell-type median training-cell library size (see train_scvi.py).

Usage:
    python -m analysis.benchmark.train_scanvi --seed 0 --eval_partition test \
        --output_dir run/benchmark/test/scanvi_seed0
"""

import os
import sys
import argparse
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
from analysis.benchmark.train_scvi import decode_per_ct


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_latent", type=int, default=30)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--n_hidden", type=int, default=256)
    p.add_argument("--max_epochs_scvi", type=int, default=200)
    p.add_argument("--max_epochs_scanvi", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_partition", default="test", choices=["valid", "test"])
    p.add_argument("--output_dir", default="run/benchmark/test/scanvi_seed0")
    args = p.parse_args()

    scvi.settings.seed = args.seed
    print("=" * 60)
    print(f"Split-aware benchmark: scANVI (seed {args.seed})")
    print("=" * 60)
    data = load_benchmark_data(args.eval_partition, return_raw_counts=True, include_eval_cells=False)
    gene_names = data["gene_names"]
    cell_tr, nuc_tr = data["cell_train"], data["nuc_train"]
    cell_tr.obs["modality"] = "cell"
    nuc_tr.obs["modality"] = "nucleus"
    adata_train = ad.concat([cell_tr, nuc_tr], merge="same", index_unique="-")
    adata_train.obs["modality"] = adata_train.obs["modality"].astype("category")
    adata_train.obs["cell_type"] = adata_train.obs["cell_type"].astype(str)
    lib_by_ct = ct_median_library(cell_tr.X, cell_tr.obs["cell_type"].values.astype(str), data["common_cts"])

    scvi.model.SCVI.setup_anndata(adata_train, batch_key="modality")
    scvi_model = scvi.model.SCVI(adata_train, n_latent=args.n_latent, n_layers=args.n_layers,
                                 n_hidden=args.n_hidden, gene_likelihood="nb")
    print(f"\n[1] scVI pretraining ({args.max_epochs_scvi} epochs max)")
    t0 = time.time()
    scvi_model.train(max_epochs=args.max_epochs_scvi, batch_size=args.batch_size,
                     plan_kwargs={"lr": args.lr}, early_stopping=True, early_stopping_patience=15)
    scvi_time = time.time() - t0

    print(f"\n[2] scANVI ({args.max_epochs_scanvi} epochs max)")
    scanvi_model = scvi.model.SCANVI.from_scvi_model(scvi_model, unlabeled_category="Unknown",
                                                     labels_key="cell_type")
    t1 = time.time()
    scanvi_model.train(max_epochs=args.max_epochs_scanvi, batch_size=args.batch_size,
                       plan_kwargs={"lr": args.lr}, early_stopping=True, early_stopping_patience=15)
    scanvi_time = time.time() - t1
    total = time.time() - t0
    print(f"Training: scVI {scvi_time:.0f}s + scANVI {scanvi_time:.0f}s = {total:.0f}s")
    os.makedirs(args.output_dir, exist_ok=True)
    scanvi_model.save(os.path.join(args.output_dir, "scanvi_model"), overwrite=True)

    adata_eval = data["nuc_eval"]
    adata_eval.obs["modality"] = pd.Categorical(["nucleus"] * adata_eval.n_obs,
                                                categories=adata_train.obs["modality"].cat.categories)
    adata_eval.obs["cell_type"] = adata_eval.obs["cell_type"].astype(str)
    scvi.model.SCANVI.setup_anndata(adata_eval, batch_key="modality", labels_key="cell_type",
                                    unlabeled_category="Unknown")
    ct_eval = adata_eval.obs["cell_type"].values.astype(str)
    imputed = decode_per_ct(scanvi_model, adata_eval, ct_eval, lib_by_ct)
    imputed_log1p = np.log1p(imputed).astype(np.float32)

    n_params = sum(p.numel() for p in scanvi_model.module.parameters())
    args_dict = {
        "method": "scANVI", "seed": args.seed, "n_latent": args.n_latent, "n_layers": args.n_layers,
        "n_hidden": args.n_hidden, "max_epochs_scvi": args.max_epochs_scvi,
        "max_epochs_scanvi": args.max_epochs_scanvi, "batch_size": args.batch_size, "lr": args.lr,
        "gene_likelihood": "nb", "labels_key": "cell_type", "n_params": int(n_params),
        "training_time_scvi_s": scvi_time, "training_time_scanvi_s": scanvi_time, "training_time_s": total,
        "scvi_version": scvi.__version__, "checkpoint": "best (early stopping)",
        "decode_library": "median raw library of training cells per cell type",
        "library_by_ct": lib_by_ct, "eval_partition": args.eval_partition,
    }
    save_standard_outputs(args.output_dir, imputed_log1p, adata_eval.obs[["obs_id", "cell_type", "donor"]],
                          gene_names, args_dict, data["split_info"])
    print("Done.")


if __name__ == "__main__":
    main()
