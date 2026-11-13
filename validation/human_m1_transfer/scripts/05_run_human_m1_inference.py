#!/usr/bin/env python3
"""
Task Group 2, Part B: Run frozen mouse-trained ImputeVAE on Allen Human M1 nuclei.

Encodes human nuclei with nuclei modality (id=1), decodes with cell modality (id=0).
Output is log1p-scale imputed "cell-like" expression.
"""

import os
import sys
import json
import time
import argparse
import csv
import numpy as np
import torch
from scipy import sparse

ROOT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE"
sys.path.insert(0, ROOT)
from imputeVAE.models_cond import VAE_JSD_Cond


def main():
    parser = argparse.ArgumentParser(description="Human M1 zero-shot inference")
    parser.add_argument("--input-matrix",
                        default=os.path.join(ROOT, "validation/human_m1_transfer/processed/human_m1_nuclei_mouse_model_order.npz"))
    parser.add_argument("--obs",
                        default=os.path.join(ROOT, "validation/human_m1_transfer/processed/human_m1_nuclei_obs.csv"))
    parser.add_argument("--gene-map",
                        default=os.path.join(ROOT, "validation/human_m1_transfer/processed/human_m1_gene_mapping_for_model_order.csv"))
    parser.add_argument("--gene-mask",
                        default=os.path.join(ROOT, "validation/human_m1_transfer/processed/human_m1_available_gene_mask.npy"))
    parser.add_argument("--model-run",
                        default=os.path.join(ROOT, "run/run_2026-07-18-19-40-51_570984"))
    parser.add_argument("--output-dir",
                        default=os.path.join(ROOT, "validation/human_m1_transfer/inference"))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-full-output", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    t_start = time.time()

    # --- 1. Load input matrix ---
    print("Loading input matrix...")
    mat_sparse = sparse.load_npz(args.input_matrix)
    print(f"  Shape: {mat_sparse.shape}, NNZ: {mat_sparse.nnz:,}, dtype: {mat_sparse.dtype}")
    assert mat_sparse.shape[1] == 10134, f"Expected 10134 cols, got {mat_sparse.shape[1]}"
    n_nuclei = mat_sparse.shape[0]

    # Load obs and verify row count
    with open(args.obs) as f:
        reader = csv.DictReader(f)
        obs_rows = list(reader)
    assert len(obs_rows) == n_nuclei, f"Obs rows {len(obs_rows)} != matrix rows {n_nuclei}"
    print(f"  Obs verified: {len(obs_rows)} rows")

    # Load gene mask
    gene_mask = np.load(args.gene_mask)
    n_mapped = int(gene_mask.sum())
    print(f"  Gene mask: {n_mapped} mapped of {len(gene_mask)}")

    # Input stats (on non-zero entries)
    input_nz = mat_sparse.data
    input_min = float(input_nz.min()) if len(input_nz) > 0 else 0.0
    input_max = float(input_nz.max()) if len(input_nz) > 0 else 0.0
    input_mean_nz = float(input_nz.mean()) if len(input_nz) > 0 else 0.0
    input_frac_zero = 1.0 - mat_sparse.nnz / (mat_sparse.shape[0] * mat_sparse.shape[1])
    print(f"  Input: min={input_min:.3f}, max={input_max:.3f}, mean_nz={input_mean_nz:.3f}, frac_zero={input_frac_zero:.4f}")

    # --- 2. Load model ---
    print("\nLoading model...")
    config_path = os.path.join(args.model_run, "arguments.json")
    with open(config_path) as f:
        config = json.load(f)

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
    )

    model_path = os.path.join(args.model_run, "models", "model.pt")
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model = model.to(device)
    print(f"  Model: VAE_JSD_Cond, {config['num_feature']} features, {config['num_latent']} latent")
    print(f"  scalar_cond={config.get('scalar_cond')}, cond_encoder={config.get('cond_encoder')}")
    print(f"  Device: {device}")

    # --- 3. Run inference ---
    print(f"\nRunning inference: {n_nuclei} nuclei, batch_size={args.batch_size}")
    print("  Encoding with nuclei modality (id=1), decoding with cell modality (id=0)")

    # Convert sparse to dense in chunks to avoid OOM
    # Output will be dense (model produces dense output)
    imputed = np.zeros((n_nuclei, 10134), dtype=np.float32)
    mat_csr = mat_sparse.tocsr()

    t_inf_start = time.time()
    with torch.no_grad():
        for start in range(0, n_nuclei, args.batch_size):
            end = min(start + args.batch_size, n_nuclei)
            batch_dense = mat_csr[start:end].toarray().astype(np.float32)
            batch_tensor = torch.from_numpy(batch_dense).to(device)

            decoded = model.decode_cross(
                batch_tensor,
                source_modality_id=1,  # nuclei
                target_modality_id=0,  # cell
            )
            imputed[start:end] = decoded.cpu().numpy()

            if start % (args.batch_size * 20) == 0:
                print(f"  {end:,}/{n_nuclei:,} ({100*end/n_nuclei:.0f}%)")

    t_inf_end = time.time()
    inf_minutes = (t_inf_end - t_inf_start) / 60
    print(f"  Inference done in {inf_minutes:.1f} min")

    # --- 4. Output stats ---
    out_min = float(imputed.min())
    out_max = float(imputed.max())
    out_mean = float(imputed.mean())
    out_std = float(imputed.std())
    out_frac_neg = float((imputed < 0).mean())
    out_frac_zero = float((np.abs(imputed) < 1e-6).mean())

    print(f"\n  Output stats:")
    print(f"    min={out_min:.4f}, max={out_max:.4f}")
    print(f"    mean={out_mean:.4f}, std={out_std:.4f}")
    print(f"    frac_negative={out_frac_neg:.4f}")
    print(f"    frac_near_zero={out_frac_zero:.4f}")

    if out_frac_neg > 0:
        print(f"    NOTE: {out_frac_neg*100:.2f}% of output values are negative (not clamped)")

    # Check for NaN/Inf
    n_nan = int(np.isnan(imputed).sum())
    n_inf = int(np.isinf(imputed).sum())
    if n_nan > 0 or n_inf > 0:
        print(f"    WARNING: {n_nan} NaN, {n_inf} Inf values detected!")

    # --- 5. Save outputs ---
    print("\nSaving outputs...")

    # Imputed matrix
    imp_path = os.path.join(args.output_dir, "human_m1_imputed_cell_log1p.npy")
    np.save(imp_path, imputed)
    print(f"  Imputed: {imp_path} ({imputed.nbytes / 1e9:.1f} GB)")

    # Input nucleus (keep as sparse reference)
    inp_path = os.path.join(args.output_dir, "human_m1_input_nucleus_log1p.npz")
    sparse.save_npz(inp_path, mat_sparse)
    print(f"  Input: {inp_path}")

    # Obs copy
    obs_out = os.path.join(args.output_dir, "human_m1_inference_obs.csv")
    with open(args.obs) as fin, open(obs_out, "w") as fout:
        fout.write(fin.read())
    print(f"  Obs: {obs_out}")

    # Gene map copy
    gm_out = os.path.join(args.output_dir, "human_m1_inference_gene_map.csv")
    with open(args.gene_map) as fin, open(gm_out, "w") as fout:
        fout.write(fin.read())
    print(f"  Gene map: {gm_out}")

    # Run info JSON
    total_minutes = (time.time() - t_start) / 60
    run_info = {
        "model_run": args.model_run,
        "checkpoint_path": model_path,
        "model_class": "VAE_JSD_Cond",
        "model_config": {
            "num_feature": config["num_feature"],
            "num_internal": num_internal,
            "num_latent": config["num_latent"],
            "embed_dim": config.get("embed_dim", 8),
            "scalar_cond": config.get("scalar_cond", False),
            "cond_encoder": config.get("cond_encoder", False),
            "raw_data_normalization": config.get("raw_data_normalization", "log1p"),
        },
        "device": str(device),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "n_nuclei": n_nuclei,
        "n_model_genes": 10134,
        "n_mapped_genes": n_mapped,
        "input_matrix_path": args.input_matrix,
        "obs_path": args.obs,
        "output_saved_as": "dense_npy",
        "input_stats": {
            "min_nonzero": input_min,
            "max": input_max,
            "mean_nonzero": input_mean_nz,
            "nnz": int(mat_sparse.nnz),
            "fraction_zero": input_frac_zero,
        },
        "output_stats": {
            "min": out_min,
            "max": out_max,
            "mean": out_mean,
            "std": out_std,
            "fraction_negative": out_frac_neg,
            "fraction_near_zero": out_frac_zero,
            "n_nan": n_nan,
            "n_inf": n_inf,
        },
        "clamping_applied": False,
        "postprocessing": "none",
        "inference_minutes": round(inf_minutes, 2),
        "total_minutes": round(total_minutes, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "command": " ".join(sys.argv),
    }
    info_path = os.path.join(args.output_dir, "human_m1_inference_run_info.json")
    with open(info_path, "w") as f:
        json.dump(run_info, f, indent=2)
    print(f"  Run info: {info_path}")

    # QC CSV
    qc_path = os.path.join(args.output_dir, "human_m1_inference_qc.csv")
    with open(qc_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["n_nuclei", n_nuclei])
        w.writerow(["n_model_genes", 10134])
        w.writerow(["n_mapped_genes", n_mapped])
        w.writerow(["input_nnz", mat_sparse.nnz])
        w.writerow(["input_fraction_zero", f"{input_frac_zero:.4f}"])
        w.writerow(["output_min", f"{out_min:.6f}"])
        w.writerow(["output_max", f"{out_max:.6f}"])
        w.writerow(["output_mean", f"{out_mean:.6f}"])
        w.writerow(["output_std", f"{out_std:.6f}"])
        w.writerow(["output_fraction_negative", f"{out_frac_neg:.6f}"])
        w.writerow(["output_fraction_zero_or_near_zero", f"{out_frac_zero:.6f}"])
        w.writerow(["output_n_nan", n_nan])
        w.writerow(["output_n_inf", n_inf])
        w.writerow(["batch_size", args.batch_size])
        w.writerow(["device", str(device)])
        w.writerow(["inference_minutes", f"{inf_minutes:.2f}"])
        w.writerow(["total_minutes", f"{total_minutes:.2f}"])
    print(f"  QC: {qc_path}")

    print(f"\nDone. Total time: {total_minutes:.1f} min")


if __name__ == "__main__":
    main()
