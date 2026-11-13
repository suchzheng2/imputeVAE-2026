#!/usr/bin/env python3
"""
Quick evaluation for probe runs that have no epoch dumps.

Loads the saved model, runs inference on both modalities to generate
a single epoch dump, then calls the existing post-training eval pipeline
for UMAPs, heatmaps, and correlation analysis.

Usage:
    python analysis/eval_probe.py --run_folder run_2026-03-20-10-27-02_2073068
    python analysis/eval_probe.py --run_folder run_2026-03-20-10-27-02_2073068 --skip_scib
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
    is_stage3_run,
    RAW_CELL_PATH, RAW_NUCLEI_PATH,
)
from imputeVAE.models import VAE_NB
from imputeVAE.data import create_dataloaders_by_cell_type, load_raw_data as load_raw_adata


def _load_model(run_folder, run_args):
    model_path = os.path.join(run_folder, "models", "model.pt")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"No model found at {model_path}")

    model = VAE_NB(
        input_size=run_args["num_feature"],
        num_internal=run_args["num_internal"],
        num_latent=run_args["num_latent"],
        use_zinb=run_args.get("use_zinb", False),
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def _make_args_namespace(run_args):
    """Convert dict to namespace for dataloader compatibility."""
    ns = argparse.Namespace(**run_args)
    # Ensure required fields
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


def generate_epoch_dump(run_folder, run_args):
    """Run inference on both modalities and produce an epoch dump JSON."""
    print("Loading model...")
    model = _load_model(run_folder, run_args)
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

    print(f"Target modality: {target_modality}, Reference: {reference_modality}")
    print(f"Loading target data: {data_path}")
    target_adata = load_raw_adata(data_path, args_ns)
    target_loaders, _, _ = create_dataloaders_by_cell_type(target_adata, args_ns)
    del target_adata
    gc.collect()

    print(f"Loading reference data: {ref_data_path}")
    ref_adata = load_raw_adata(ref_data_path, args_ns)
    ref_loaders, _, _ = create_dataloaders_by_cell_type(ref_adata, args_ns, batch_name=ref_batch)
    del ref_adata
    gc.collect()

    common_celltypes = sorted(set(target_loaders.keys()) & set(ref_loaders.keys()))
    print(f"Common cell types: {len(common_celltypes)}")

    epoch_save_dict = {}
    with torch.no_grad():
        for celltype in common_celltypes:
            epoch_save_dict[celltype] = {}
            target_loader = target_loaders[celltype]
            ref_batches = list(ref_loaders[celltype])
            if not ref_batches:
                continue

            for batch_idx, target_batch in enumerate(target_loader):
                if batch_idx >= len(ref_batches):
                    break
                ref_batch_data = ref_batches[batch_idx]
                target_data = target_batch["data"]
                ref_data = ref_batch_data["data"]

                batch_size = min(target_data.size(0), ref_data.size(0))
                target_data = target_data[:batch_size]
                ref_data = ref_data[:batch_size]

                t_enc, t_dec, t_mu, t_lv = model(target_data)
                r_enc, r_dec, r_mu, r_lv = model(ref_data)

                if target_modality == "nuclei":
                    nuc_enc, nuc_dec, nuc_mu, nuc_lv = t_enc, t_dec, t_mu, t_lv
                    cel_enc, cel_dec, cel_mu, cel_lv = r_enc, r_dec, r_mu, r_lv
                    target_batch_ids = []
                    if target_loader.dataset.has_batch_id:
                        target_batch_ids = [
                            target_loader.dataset.get_original_value("batch_id", c)
                            for c in target_batch["batch_id"].numpy()[:batch_size]
                        ]
                    ref_batch_ids = []
                    if ref_loaders[celltype].dataset.has_batch_id:
                        ref_batch_ids = [
                            ref_loaders[celltype].dataset.get_original_value("batch_id", c)
                            for c in ref_batch_data["batch_id"].numpy()[:batch_size]
                        ]
                else:
                    cel_enc, cel_dec, cel_mu, cel_lv = t_enc, t_dec, t_mu, t_lv
                    nuc_enc, nuc_dec, nuc_mu, nuc_lv = r_enc, r_dec, r_mu, r_lv
                    target_batch_ids = []
                    if target_loader.dataset.has_batch_id:
                        target_batch_ids = [
                            target_loader.dataset.get_original_value("batch_id", c)
                            for c in target_batch["batch_id"].numpy()[:batch_size]
                        ]
                    ref_batch_ids = []
                    if ref_loaders[celltype].dataset.has_batch_id:
                        ref_batch_ids = [
                            ref_loaders[celltype].dataset.get_original_value("batch_id", c)
                            for c in ref_batch_data["batch_id"].numpy()[:batch_size]
                        ]

                original_cell_types = [
                    target_loader.dataset.get_original_value("cell_type", c)
                    for c in target_batch["cell_type"].numpy()[:batch_size]
                ]

                # Sample IDs
                target_sample_ids = list(target_batch.get("sample_id", [])[:batch_size])
                ref_sample_ids = list(ref_batch_data.get("sample_id", [])[:batch_size])

                dump_entry = {
                    "batch_idx": batch_idx,
                    "cell_type": target_batch["cell_type"].numpy()[:batch_size].tolist(),
                    "cell_type_original": original_cell_types,
                    "meta_batch_id": target_batch["batch_id"].numpy()[:batch_size].tolist(),
                    "meta_batch_id_original": target_batch_ids,
                    f"{target_modality}_meta_batch_id_original": target_batch_ids,
                    f"{reference_modality}_meta_batch_id_original": ref_batch_ids,
                    f"{target_modality}_sample_id": target_sample_ids,
                    f"{reference_modality}_sample_id": ref_sample_ids,
                    "nuclei_encoded": nuc_enc.numpy().tolist(),
                    "nuclei_decoded": nuc_dec.numpy().tolist(),
                    "nuclei_mu": nuc_mu.numpy().tolist(),
                    "nuclei_log_var": nuc_lv.numpy().tolist(),
                    "cell_encoded": cel_enc.numpy().tolist(),
                    "cell_decoded": cel_dec.numpy().tolist(),
                    "cell_mu": cel_mu.numpy().tolist(),
                    "cell_log_var": cel_lv.numpy().tolist(),
                }
                epoch_save_dict[celltype][batch_idx] = dump_entry

            print(f"  {celltype}: {len(epoch_save_dict[celltype])} batches")

    # Write dump
    dump_dir = os.path.join(run_folder, "epoch_dump")
    os.makedirs(dump_dir, exist_ok=True)
    dump_path = os.path.join(dump_dir, "train_epoch_99.json")
    with open(dump_path, "w") as f:
        json.dump(epoch_save_dict, f)
    print(f"Epoch dump written: {dump_path}")
    return dump_path


def main():
    parser = argparse.ArgumentParser(description="Quick probe evaluation")
    parser.add_argument("--run_folder", required=True, help="Run folder name")
    parser.add_argument("--skip_scib", action="store_true", help="Skip scIB metrics")
    parser.add_argument("--skip_umap", action="store_true", help="Skip UMAP generation")
    parser.add_argument("--skip_eval", action="store_true", help="Skip heatmaps/correlation")
    parser.add_argument("--delete_dump", action="store_true", help="Delete dump after eval")
    args = parser.parse_args()

    run_folder = find_run_folder(args.run_folder)
    run_args = load_run_args(run_folder)

    if not run_args:
        print(f"ERROR: No arguments.json found in {run_folder}")
        sys.exit(1)

    # Step 1: Generate epoch dump from saved model
    dump_path = generate_epoch_dump(run_folder, run_args)

    # Step 2: Run UMAPs
    if not args.skip_umap:
        print("\n=== Generating UMAPs ===")
        try:
            from eval_post_training import run_umaps_for_epoch
            run_umaps_for_epoch(run_folder, 99, dump_path)
        except Exception as e:
            print(f"UMAP generation failed: {e}")
            traceback.print_exc()

    # Step 3: Run full eval (heatmaps, correlation)
    if not args.skip_eval:
        print("\n=== Running heatmaps + correlation ===")
        try:
            from eval_post_training import run_full_eval_for_epoch
            raw_cell = None
            raw_nuclei = None
            try:
                common_genes, common_cell_types, mitocarta_genes = load_common_filters()
                raw_cell = load_raw_data(RAW_CELL_PATH, common_genes, common_cell_types, mitocarta_genes)
                raw_nuclei = load_raw_data(RAW_NUCLEI_PATH, common_genes, common_cell_types, mitocarta_genes)
            except Exception as e:
                print(f"Warning: Could not load raw data for eval: {e}")

            is_stage3 = is_stage3_run(run_args)
            run_full_eval_for_epoch(run_folder, 99, is_stage3, raw_cell, raw_nuclei, run_args)
        except Exception as e:
            print(f"Full eval failed: {e}")
            traceback.print_exc()

    # Step 4: Run scIB
    if not args.skip_scib:
        print("\n=== Running scIB metrics ===")
        try:
            from eval_post_training import run_scib_for_epoch
            run_scib_for_epoch(run_folder, 99)
        except Exception as e:
            print(f"scIB failed: {e}")
            traceback.print_exc()

    # Cleanup
    if args.delete_dump and os.path.isfile(dump_path):
        os.remove(dump_path)
        print(f"Deleted dump: {dump_path}")

    print("\n=== Probe evaluation complete ===")


if __name__ == "__main__":
    main()
