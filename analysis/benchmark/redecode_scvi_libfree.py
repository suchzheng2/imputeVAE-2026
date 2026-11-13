"""
Label-free re-decode of saved scVI / scANVI benchmark models.

The main benchmark decodes the count models at the per-cell-type median library size of
training cells.  That hands the method the cell-type label AND the real cells' type-level RNA
content at inference time, which the cVAE does not get.  This variant uses no labels:

    imputed_counts = scVI normalized expression with library_size="latent"
                     (the model's own inferred library for the nucleus, nucleus scale)
                   x  (median training-cell library / median training-nucleus library)

The ratio is one global number from the training partition.  Output is log1p(imputed_counts).

Usage:
    python -m analysis.benchmark.redecode_scvi_libfree --method scvi \
        --model_dir run/benchmark/test/scvi_seed0 --eval_partition test \
        --output_dir run/benchmark/test/scvi_libfree_seed0
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import anndata as ad

try:
    import scvi
except ImportError:
    print("ERROR: scvi-tools not installed in this environment (use conda env scvi2)")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from analysis.benchmark.data_split import load_benchmark_data, save_standard_outputs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["scvi", "scanvi"], required=True)
    p.add_argument("--model_dir", required=True, help="benchmark method dir holding scvi_model/ or scanvi_model/")
    p.add_argument("--eval_partition", default="test", choices=["valid", "test"])
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    data = load_benchmark_data(args.eval_partition, return_raw_counts=True, include_eval_cells=False)
    gene_names = data["gene_names"]
    cell_tr, nuc_tr = data["cell_train"], data["nuc_train"]
    ratio = float(np.median(cell_tr.X.sum(1)) / np.median(nuc_tr.X.sum(1)))
    print(f"Global library ratio (median train cell / median train nucleus): {ratio:.3f}")

    adata_eval = data["nuc_eval"]
    adata_eval.obs["modality"] = pd.Categorical(["nucleus"] * adata_eval.n_obs, categories=["cell", "nucleus"])
    adata_eval.obs["cell_type"] = adata_eval.obs["cell_type"].astype(str)

    sub = "scvi_model" if args.method == "scvi" else "scanvi_model"
    cls = scvi.model.SCVI if args.method == "scvi" else scvi.model.SCANVI
    model = cls.load(os.path.join(args.model_dir, sub), adata=adata_eval)
    print(f"Loaded {args.method} from {args.model_dir}/{sub}")

    x = model.get_normalized_expression(adata=adata_eval, transform_batch="cell", library_size="latent",
                                        return_numpy=True)
    imputed = np.log1p(np.asarray(x, dtype=np.float64) * ratio).astype(np.float32)

    meta = json.load(open(os.path.join(args.model_dir, "arguments.json")))
    meta.update({
        "method": meta.get("method", args.method) + "_libfree",
        "decode_library": "latent (nucleus-scale) x global median cell/nucleus library ratio; no labels",
        "library_ratio": ratio, "source_model_dir": args.model_dir,
    })
    meta.pop("library_by_ct", None)
    save_standard_outputs(args.output_dir, imputed, adata_eval.obs[["obs_id", "cell_type", "donor"]],
                          gene_names, meta, data["split_info"])
    print("Done.")


if __name__ == "__main__":
    main()
