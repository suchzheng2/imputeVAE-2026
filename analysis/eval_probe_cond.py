#!/usr/bin/env python3
"""
Post-training evaluation for conditioned VAE (VAE_NB_Cond) runs.

Loads the saved model, runs inference on both modalities to generate
epoch dumps (train + valid), then calls eval pipeline for UMAPs,
heatmaps, correlation, and scIB metrics.

Usage:
    python analysis/eval_probe_cond.py --run_folder run_2026-03-26-23-28-20_500560
    python analysis/eval_probe_cond.py --run_folder run_2026-03-26-23-28-20_500560 --skip_scib
    python analysis/eval_probe_cond.py --run_folder run_2026-03-26-23-28-20_500560 --splits train,valid
"""

import os
import sys
import gc
import json
import argparse
import traceback
import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, _PROJECT_DIR)
sys.path.insert(0, os.path.join(_PROJECT_DIR, "imputeVAE"))

from eval_utils import (
    find_run_folder, load_run_args, load_common_filters, load_raw_data,
    RAW_CELL_PATH, RAW_NUCLEI_PATH,
)
from imputeVAE.models_cond import VAE_NB_Cond
from imputeVAE.data import create_dataloaders_by_cell_type, load_raw_data as load_raw_adata

CELL_ID = 0
NUCLEI_ID = 1


def _load_model(run_folder, run_args):
    """Load conditioned VAE model from checkpoint."""
    model_path = os.path.join(run_folder, "models", "model.pt")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"No model found at {model_path}")

    model = VAE_NB_Cond(
        input_size=run_args["num_feature"],
        num_internal=run_args["num_internal"],
        num_latent=run_args["num_latent"],
        embed_dim=run_args.get("embed_dim", 8),
        use_zinb=run_args.get("use_zinb", False),
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def _make_args_namespace(run_args):
    """Convert dict to namespace for dataloader compatibility."""
    ns = argparse.Namespace(**run_args)
    if not hasattr(ns, "detection_rate_threshold"):
        ns.detection_rate_threshold = 1.0
    if not hasattr(ns, "exclude_mitocarta"):
        ns.exclude_mitocarta = False
    return ns


def _swap_batch(batch_name):
    if not batch_name:
        return batch_name
    batch_name = batch_name.strip()
    if batch_name.startswith("n"):
        return "c" + batch_name[1:]
    if batch_name.startswith("c"):
        return "n" + batch_name[1:]
    return batch_name


def _get_batch_ids(loader, celltype, batch_data, batch_size):
    """Extract original batch IDs from a batch."""
    if loader.dataset.has_batch_id:
        return [
            loader.dataset.get_original_value("batch_id", code)
            for code in batch_data["batch_id"].cpu().numpy()[:batch_size]
        ]
    return []


def generate_epoch_dump(run_folder, run_args, model, split="train"):
    """Run inference on both modalities and produce an epoch dump JSON.

    Args:
        split: "train" or "valid"
    Returns:
        path to the dump file
    """
    args_ns = _make_args_namespace(run_args)

    data_path = run_args.get("data_path", "")
    target_modality = "nuclei" if "nuclei" in data_path.lower() else "cell"
    reference_modality = "cell" if target_modality == "nuclei" else "nuclei"

    # Determine reference data path
    if target_modality == "nuclei":
        ref_data_path = RAW_CELL_PATH + "/"
    else:
        ref_data_path = RAW_NUCLEI_PATH + "/"

    ref_batch = _swap_batch(run_args.get("batch"))

    print(f"\n=== Generating {split} epoch dump ===")
    print(f"Target modality: {target_modality}, Reference: {reference_modality}")

    # Load data and create loaders
    print(f"Loading target data: {data_path}")
    target_adata = load_raw_adata(data_path, args_ns)
    target_train_loaders, target_valid_loaders, _ = create_dataloaders_by_cell_type(
        target_adata, args_ns
    )
    del target_adata
    gc.collect()

    print(f"Loading reference data: {ref_data_path}")
    ref_adata = load_raw_adata(ref_data_path, args_ns)
    ref_train_loaders, ref_valid_loaders, _ = create_dataloaders_by_cell_type(
        ref_adata, args_ns, batch_name=ref_batch
    )
    del ref_adata
    gc.collect()

    # Select train or valid loaders
    if split == "valid":
        if target_modality == "nuclei":
            nuclei_loaders = target_valid_loaders
            cell_loaders = ref_valid_loaders
        else:
            cell_loaders = target_valid_loaders
            nuclei_loaders = ref_valid_loaders
    else:
        if target_modality == "nuclei":
            nuclei_loaders = target_train_loaders
            cell_loaders = ref_train_loaders
        else:
            cell_loaders = target_train_loaders
            nuclei_loaders = ref_train_loaders

    common_celltypes = sorted(set(nuclei_loaders.keys()) & set(cell_loaders.keys()))
    print(f"Common cell types ({split}): {len(common_celltypes)}")

    if not common_celltypes:
        print(f"  No common cell types for {split}, skipping")
        return None

    epoch_save_dict = {}
    with torch.no_grad():
        for celltype in common_celltypes:
            epoch_save_dict[celltype] = {}
            nuclei_loader = nuclei_loaders[celltype]
            cell_batches = list(cell_loaders[celltype])
            if not cell_batches:
                continue

            for batch_idx, nuclei_batch in enumerate(nuclei_loader):
                if batch_idx >= len(cell_batches):
                    break
                cell_batch_data = cell_batches[batch_idx]
                nuclei_data = nuclei_batch["data"]
                cell_data = cell_batch_data["data"]

                batch_size = min(nuclei_data.size(0), cell_data.size(0))
                nuclei_data = nuclei_data[:batch_size]
                cell_data = cell_data[:batch_size]

                # Forward with modality IDs
                cell_encoded, cell_decoded, cell_mu, cell_log_var = model(
                    cell_data, modality_id=CELL_ID
                )
                nuclei_encoded, nuclei_decoded, nuclei_mu, nuclei_log_var = model(
                    nuclei_data, modality_id=NUCLEI_ID
                )

                # Cross-decode: nuclei -> cell decoder
                cross_decoded = model.decode_cross(
                    nuclei_data, source_modality_id=NUCLEI_ID,
                    target_modality_id=CELL_ID
                )

                # Metadata
                original_cell_types = [
                    nuclei_loader.dataset.get_original_value("cell_type", code)
                    for code in nuclei_batch["cell_type"].cpu().numpy()[:batch_size]
                ]
                nuclei_batch_ids = _get_batch_ids(
                    nuclei_loader, celltype, nuclei_batch, batch_size
                )
                cell_batch_ids = _get_batch_ids(
                    cell_loaders[celltype], celltype, cell_batch_data, batch_size
                )
                nuclei_sample_ids = list(
                    nuclei_batch["sample_id"][:batch_size]
                ) if "sample_id" in nuclei_batch else []
                cell_sample_ids = list(
                    cell_batch_data["sample_id"][:batch_size]
                ) if "sample_id" in cell_batch_data else []

                dump_entry = {
                    "batch_idx": batch_idx,
                    "cell_type": nuclei_batch["cell_type"].cpu().numpy()[:batch_size].tolist(),
                    "cell_type_original": original_cell_types,
                    "meta_batch_id": nuclei_batch["batch_id"].cpu().numpy()[:batch_size].tolist(),
                    "meta_batch_id_original": nuclei_batch_ids,
                    "nuclei_meta_batch_id_original": nuclei_batch_ids,
                    "cell_meta_batch_id_original": cell_batch_ids,
                    "nuclei_sample_id": nuclei_sample_ids,
                    "cell_sample_id": cell_sample_ids,
                    "nuclei_encoded": nuclei_encoded.detach().cpu().numpy().tolist(),
                    "nuclei_decoded": nuclei_decoded.detach().cpu().numpy().tolist(),
                    "nuclei_mu": nuclei_mu.detach().cpu().numpy().tolist(),
                    "nuclei_log_var": nuclei_log_var.detach().cpu().numpy().tolist(),
                    "cell_encoded": cell_encoded.detach().cpu().numpy().tolist(),
                    "cell_decoded": cell_decoded.detach().cpu().numpy().tolist(),
                    "cell_mu": cell_mu.detach().cpu().numpy().tolist(),
                    "cell_log_var": cell_log_var.detach().cpu().numpy().tolist(),
                    "cross_decoded": cross_decoded.detach().cpu().numpy().tolist(),
                }
                epoch_save_dict[celltype][batch_idx] = dump_entry

            print(f"  {celltype}: {len(epoch_save_dict[celltype])} batches")

    # Write dump — use epoch 99 for train, 98 for valid so
    # find_epoch_dump() can locate them independently.
    epoch_num = 99 if split == "train" else 98
    dump_dir = os.path.join(run_folder, "epoch_dump")
    os.makedirs(dump_dir, exist_ok=True)
    dump_path = os.path.join(dump_dir, f"train_epoch_{epoch_num}.json")
    with open(dump_path, "w") as f:
        json.dump(epoch_save_dict, f)
    file_size = os.path.getsize(dump_path) / (1024 * 1024)
    print(f"Epoch dump written: {dump_path} ({file_size:.0f} MB)")
    return dump_path, epoch_num


def main():
    parser = argparse.ArgumentParser(
        description="Post-training evaluation for conditioned VAE runs"
    )
    parser.add_argument("--run_folder", required=True, help="Run folder name or path")
    parser.add_argument("--splits", default="train,valid",
                        help="Comma-separated splits to evaluate (default: train,valid)")
    parser.add_argument("--skip_scib", action="store_true", help="Skip scIB metrics")
    parser.add_argument("--skip_umap", action="store_true", help="Skip UMAP generation")
    parser.add_argument("--skip_eval", action="store_true", help="Skip heatmaps/correlation")
    parser.add_argument("--delete_dump", action="store_true",
                        help="Delete dump after evaluation")
    args = parser.parse_args()

    run_folder = find_run_folder(args.run_folder)
    run_args = load_run_args(run_folder)
    splits = [s.strip() for s in args.splits.split(",")]

    if not run_args:
        print(f"ERROR: No arguments.json found in {run_folder}")
        sys.exit(1)

    print(f"Run folder: {run_folder}")
    print(f"Splits: {splits}")

    # Step 1: Load model once
    print("\n=== Loading conditioned model ===")
    model = _load_model(run_folder, run_args)

    # Step 2: Generate epoch dumps for each split
    # Use epoch 99 for train, 98 for valid so eval functions can find them
    dump_info = {}  # split -> (dump_path, epoch_num)
    for split in splits:
        result = generate_epoch_dump(run_folder, run_args, model, split=split)
        if result:
            dump_info[split] = result

    del model
    gc.collect()

    if not dump_info:
        print("ERROR: No dumps generated")
        sys.exit(1)

    # Step 3: Load raw data once for eval
    raw_cell = raw_nuclei = None
    if not args.skip_eval:
        det_rate = float(run_args.get('detection_rate_threshold', 1.0))
        exclude_mito = run_args.get('exclude_mitocarta', True)
        common_genes, common_cell_types, mitocarta_genes = load_common_filters()
        print("\n=== Loading raw reference data ===")
        raw_cell = load_raw_data(
            RAW_CELL_PATH, common_genes, common_cell_types,
            mitocarta_genes, det_rate, exclude_mito,
        )
        raw_nuclei = load_raw_data(
            RAW_NUCLEI_PATH, common_genes, common_cell_types,
            mitocarta_genes, det_rate, exclude_mito,
        )

    # The run uses fine_tune_cond_stage3, treat as stage3 for eval
    is_stage3 = True

    # Step 4: Run evaluation for each split
    for split, (dump_path, epoch_num) in dump_info.items():
        print(f"\n{'='*60}")
        print(f"Evaluating {split} split (epoch_num={epoch_num})")
        print(f"{'='*60}")

        data_split = "train" if split == "train" else "valid"

        # 4a. UMAPs
        if not args.skip_umap:
            print(f"\n  [{split}] Generating UMAPs...")
            try:
                from eval_post_training import run_umaps_for_epoch
                run_umaps_for_epoch(run_folder, epoch_num, dump_path)
            except Exception as e:
                print(f"  UMAP generation failed: {e}")
                traceback.print_exc()

        # 4b. Full eval (heatmaps, correlation)
        if not args.skip_eval:
            print(f"\n  [{split}] Running heatmaps + correlation...")
            try:
                from eval_post_training import run_full_eval_for_epoch
                run_full_eval_for_epoch(
                    run_folder, epoch_num, is_stage3,
                    raw_cell, raw_nuclei, run_args
                )
            except Exception as e:
                print(f"  Full eval failed: {e}")
                traceback.print_exc()

        # 4c. scIB metrics
        if not args.skip_scib:
            print(f"\n  [{split}] Running scIB metrics...")
            try:
                from eval_post_training import run_scib_for_epoch
                run_scib_for_epoch(run_folder, epoch_num, is_stage3, run_args)
            except Exception as e:
                print(f"  scIB failed: {e}")
                traceback.print_exc()

        # Rename evaluation outputs so train/valid results don't clobber
        eval_dir = os.path.join(run_folder, "evaluation")
        for subdir in ["scib_metrics"]:
            src = os.path.join(eval_dir, subdir, f"epoch_{epoch_num}")
            dst = os.path.join(eval_dir, subdir, f"{split}_epoch_{epoch_num}")
            if os.path.isdir(src) and not os.path.isdir(dst):
                os.rename(src, dst)
                print(f"  Renamed: {src} -> {dst}")

        # 4d. Cleanup dump if requested
        if args.delete_dump and os.path.isfile(dump_path):
            os.remove(dump_path)
            print(f"  Deleted dump: {dump_path}")

    # Cleanup
    if raw_cell is not None:
        del raw_cell, raw_nuclei
    gc.collect()

    print(f"\n{'='*60}")
    print("Conditioned model evaluation complete.")
    print(f"Results in: {os.path.join(run_folder, 'evaluation')}")


if __name__ == "__main__":
    main()
