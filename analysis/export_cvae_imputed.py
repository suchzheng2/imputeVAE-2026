"""
Export cVAE cross-decoded imputed cell expression from nuclei input.

Produces imputed_cell_from_nucleus.npy and nucleus_obs.csv in the same
format as baseline_scvi.py, so evaluate_baseline_incremental.py can
consume them directly.

Usage:
    python analysis/export_cvae_imputed.py \
        --run_dir run/run_2026-05-11-19-52-25_134850 \
        --output_dir run/cvae_baseline/
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import anndata as ad
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from imputeVAE.models_cond import VAE_JSD_Cond


def load_nuclei_data(gene_list_path="data/training_gene_list_v3.csv",
                     normalization="log1p"):
    """Load nuclei data with same filtering/normalization as training."""
    DATA_ROOT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi"
    gene_list = pd.read_csv(gene_list_path, header=None)[0].tolist()
    common_genes = pd.read_csv("data/common_genes.csv", header=0)["common_genes"].tolist()
    common_ct = pd.read_csv("data/common_cell_types.csv", header=0)["common_cell_types"].tolist()

    adata = ad.read_h5ad(os.path.join(DATA_ROOT, "10X_nuclei_v2_v3_AIBS", "data.h5ad"))
    if "meta_batch_id" not in adata.obs:
        adata.obs["meta_batch_id"] = adata.obs["batch_id"].apply(lambda x: x.split(".")[0])
    adata = adata[adata.obs["meta_batch_id"].str.endswith("n3A")].copy()
    adata = adata[:, adata.var["gene_names"].isin(common_genes)].copy()
    adata = adata[adata.obs["cell_type"].isin(common_ct)].copy()
    adata = adata[:, adata.var["gene_names"].isin(gene_list)].copy()
    adata = adata[:, adata.var["gene_names"].argsort()].copy()

    X = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
    if normalization == "cpm":
        lib_size = np.clip(X.sum(axis=1, keepdims=True), 1.0, None)
        adata.X = np.log1p(X / lib_size * 1e6).astype(np.float32)
        print(f"Loaded nuclei (CPM): {adata.shape}")
    else:
        adata.X = np.log1p(X).astype(np.float32)
        print(f"Loaded nuclei (log1p): {adata.shape}")
    return adata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--target_libsize", default="auto",
                        choices=["auto", "ct_median_cell", "quantile_map", "source"],
                        help="libsize_cond runs only (2026-10-16): library the nuclei->cell decode uses; "
                             "auto = ct_median_cell for covariate models, ignored otherwise. "
                             "See analysis/benchmark/predict_cvae.py.")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.run_dir, "eval")
    os.makedirs(output_dir, exist_ok=True)

    # Load model config
    with open(os.path.join(args.run_dir, "arguments.json")) as f:
        config = json.load(f)

    # Build model
    num_internal = config["num_internal"]
    if isinstance(num_internal, str):
        num_internal = [int(x) for x in num_internal.split(",")]
    model = VAE_JSD_Cond(
        input_size=config["num_feature"],
        num_internal=num_internal,
        num_latent=config["num_latent"],
        embed_dim=config.get("embed_dim", 8),
        scalar_cond=config.get("scalar_cond", False),
        cond_encoder=config.get("cond_encoder", False),
        libsize_cond=config.get("libsize_cond", False),
    )

    # Load weights
    model_path = os.path.join(args.run_dir, "models", "model.pt")
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded from {model_path} -> {device}")

    # Load nuclei data (match training normalization)
    norm = config.get("raw_data_normalization", "log1p")
    adata_nuc = load_nuclei_data(config.get("gene_list", "data/training_gene_list_v3.csv"),
                                 normalization=norm)

    # Cross-decode: encode as nuclei (1), decode as cell (0)
    X_nuc = adata_nuc.X
    n_samples = X_nuc.shape[0]
    imputed = np.zeros_like(X_nuc)

    # Library-size covariate models: explicit decode target (per-type median training-cell library by default)
    target = None
    target_mode = "n/a (no covariate)"
    if model.libsize_cond:
        from analysis.benchmark.predict_cvae import build_target_log_libsize
        target_mode = "ct_median_cell" if args.target_libsize == "auto" else args.target_libsize
        target, _ = build_target_log_libsize(target_mode, args.run_dir, X_nuc,
                                             adata_nuc.obs["cell_type"].astype(str).values)
        print(f"libsize_cond model: decode target = {target_mode}")
    elif args.target_libsize not in ("auto", "source"):
        raise SystemExit(f"--target_libsize {args.target_libsize} needs a libsize_cond model")
    with open(os.path.join(output_dir, "export_args.json"), "w") as f:
        json.dump({"run_dir": args.run_dir, "checkpoint": model_path, "libsize_cond": bool(model.libsize_cond),
                   "target_libsize": target_mode, "normalization": norm}, f, indent=2)

    print(f"Cross-decoding {n_samples} nuclei -> cell ...")
    with torch.no_grad():
        for start in range(0, n_samples, args.batch_size):
            end = min(start + args.batch_size, n_samples)
            batch = torch.from_numpy(X_nuc[start:end]).to(device)
            kw = {} if target is None else {"target_log_libsize": torch.from_numpy(target[start:end]).to(device)}
            decoded = model.decode_cross(batch, source_modality_id=1, target_modality_id=0, **kw)
            imputed[start:end] = decoded.cpu().numpy()
            if (start // args.batch_size) % 50 == 0:
                print(f"  {end}/{n_samples}")

    # Save
    npy_path = os.path.join(output_dir, "imputed_cell_from_nucleus.npy")
    np.save(npy_path, imputed.astype(np.float32))
    print(f"Saved imputed: {npy_path} {imputed.shape}")

    obs_path = os.path.join(output_dir, "nucleus_obs.csv")
    adata_nuc.obs[["cell_type"]].to_csv(obs_path)
    print(f"Saved obs: {obs_path}")

    # Summary stats
    print(f"\nImputed stats: min={imputed.min():.3f}, max={imputed.max():.3f}, "
          f"mean={imputed.mean():.3f}, nonzero={(imputed > 0).mean():.1%}")


if __name__ == "__main__":
    main()
