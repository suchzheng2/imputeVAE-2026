import gc
import json
import os
import time

import anndata as ad
import torch
import torch.optim as optim

from .data import create_dataloaders_by_cell_type, load_raw_data
from .divergences import compute_mc_jsd_from_two_latents
from .models import VAE_JSD, VAE_L1, VAE_NB
from .plot_loss import plot_loss_curve_stage3_v2
from .stage3_eval import (
    accumulate_epoch_batch,
    compute_loader_celltype_means,
    init_epoch_accumulator,
    load_raw_reference_data,
    run_stage3_full_evaluation,
    run_stage3_inline_evaluation,
    run_stage3_inline_umap_correlation,
    _prepare_raw_means_for_correlation,
)
from .training import train_loop_ABC
from .utils import device, verify_gpu_usage
from .visualization import plot_umap_after_train_by_celltype_stage3


def _create_model_from_args(args):
    if args.model == "vae_nb":
        return VAE_NB(
            input_size=args.num_feature,
            num_internal=args.num_internal,
            num_latent=args.num_latent,
            use_zinb=getattr(args, "use_zinb", False),
        )
    if args.model == "vae_js":
        return VAE_JSD(
            input_size=args.num_feature,
            num_internal=args.num_internal,
            num_latent=args.num_latent,
        )
    return VAE_L1(
        input_size=args.num_feature,
        num_internal=args.num_internal,
        num_latent=args.num_latent,
        l1_coef=args.l1_coef,
    )


def _setup_training_device():
    global device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"CUDA available. Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU")
    return device


def _infer_modality_from_data_path(data_path):
    data_path = (data_path or "").lower()
    if "nuclei" in data_path:
        return "nuclei"
    if "cell" in data_path:
        return "cell"
    raise ValueError(f"Unable to infer modality from data path: {data_path}")


def _swap_batch_modality(batch_name):
    if not batch_name:
        return batch_name
    batch_name = batch_name.strip()
    if batch_name.startswith("n"):
        return "c" + batch_name[1:]
    if batch_name.startswith("c"):
        return "n" + batch_name[1:]
    return batch_name


def _get_reference_data_path(args):
    if getattr(args, "reference_data_path", None):
        return args.reference_data_path
    if _infer_modality_from_data_path(args.data_path) == "nuclei":
        return "10X_cells_v2_v3_AIBS/"
    return "10X_nuclei_v2_v3_AIBS/"


def _get_reference_batch(args):
    if getattr(args, "reference_batch", None):
        return args.reference_batch
    return _swap_batch_modality(args.batch)


def _load_adata_for_path(data_path, args):
    if args.use_raw_data:
        return load_raw_data(data_path, args)
    integrated_dp = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "ABC_results",
        data_path,
        "integrated.h5ad",
    )
    return ad.read_h5ad(integrated_dp)


def _anneal_weight(epoch, start_epoch, anneal_epochs, final_value):
    if epoch < start_epoch:
        return 0.0
    if epoch < start_epoch + anneal_epochs:
        progress = (epoch - start_epoch) / max(1, anneal_epochs)
        return min(final_value, final_value * progress)
    return final_value


def _runtime_str(start_time):
    elapsed = time.time() - start_time
    return f"{int(elapsed // 3600)}:{int((elapsed % 3600) // 60)}:{elapsed % 60:.2f}"


def _stage3_weight_name(base_name):
    if base_name == "alignment":
        return "alignment_weight"
    return f"{base_name}_weight"


def _stage3_anneal_flag_name(base_name):
    if base_name == "alignment":
        return "anneal_alignment"
    return f"anneal_{base_name}_weight"


def _stage3_anneal_start_name(base_name):
    if base_name == "alignment":
        return "anneal_alignment_start_epoch"
    return f"{base_name}_weight_start_epoch"


def _stage3_anneal_epoch_name(base_name):
    if base_name == "alignment":
        return "anneal_alignment_epoch"
    return f"{base_name}_weight_anneal_epoch"


def _prepare_stage3_weights(args):
    weight_keys = [
        "nuclei_recons",
        "nuclei_jsd",
        "cell_recons",
        "cell_jsd",
        "alignment",
        "cross_recons",
        "residual",
    ]
    for key in weight_keys:
        attr_name = _stage3_weight_name(key)
        original_attr = f"{attr_name}_original"
        if not hasattr(args, original_attr):
            setattr(args, original_attr, getattr(args, attr_name))


def _update_stage3_weight(args, epoch, base_name):
    attr_name = _stage3_weight_name(base_name)
    original_attr = f"{attr_name}_original"
    anneal_flag = getattr(args, _stage3_anneal_flag_name(base_name), False)
    final_value = getattr(args, original_attr)

    if anneal_flag:
        start_epoch = getattr(args, _stage3_anneal_start_name(base_name))
        anneal_epochs = getattr(args, _stage3_anneal_epoch_name(base_name))
        value = _anneal_weight(epoch, start_epoch, anneal_epochs, final_value)
    else:
        value = final_value

    setattr(args, attr_name, value)
    return value


def _write_epoch_dump(save_path, epoch, epoch_save_dict):
    """Write epoch dump to train_epoch_N.json and return the file path."""
    dump_dir = os.path.join(save_path, "epoch_dump")
    os.makedirs(dump_dir, exist_ok=True)
    train_file_path = os.path.join(dump_dir, f"train_epoch_{epoch}.json")
    with open(train_file_path, "w") as fp:
        json.dump(epoch_save_dict, fp)
    print(f"  Epoch dump written: {train_file_path}")
    return train_file_path


def _save_stage3_epoch_outputs(save_path, epoch, epoch_save_dict, batch_loss_list):
    os.makedirs(os.path.join(save_path, "models"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "train_loss"), exist_ok=True)

    train_file_path = _write_epoch_dump(save_path, epoch, epoch_save_dict)

    plot_umap_after_train_by_celltype_stage3(
        save_path, feature="encoded", data_split="train", epoch=epoch, file_path=train_file_path
    )
    plot_umap_after_train_by_celltype_stage3(
        save_path, feature="decoded", data_split="train", epoch=epoch, file_path=train_file_path
    )

    with open(os.path.join(save_path, "train_loss", f"epoch_{epoch}_train_loss.txt"), "w") as fp:
        header = [
            "celltype",
            "batch",
            "loss",
            "nuclei_recons_loss",
            "nuclei_JSD_div",
            "cell_recons_loss",
            "cell_JSD_div",
            "alignment_JSD_loss",
            "cross_recons_loss",
            "data_size",
        ]
        fp.write("\t".join(header) + "\n")
        for row in batch_loss_list:
            fp.write("\t".join(map(str, row)) + "\n")


def _write_stage3_batch_losses(save_path, epoch, batch_loss_list):
    os.makedirs(os.path.join(save_path, "train_loss"), exist_ok=True)
    with open(os.path.join(save_path, "train_loss", f"epoch_{epoch}_train_loss.txt"), "w") as fp:
        header = [
            "celltype",
            "batch",
            "loss",
            "nuclei_recons_loss",
            "nuclei_JSD_div",
            "cell_recons_loss",
            "cell_JSD_div",
            "alignment_JSD_loss",
            "cross_recons_loss",
            "data_size",
        ]
        fp.write("\t".join(header) + "\n")
        for row in batch_loss_list:
            fp.write("\t".join(map(str, row)) + "\n")


def train_loop_ABC_stage2_v2(adata, cell_types, save_path, args):
    if not args.stage1_model_run:
        raise ValueError("--stage1_model_run is required for stage 2")

    # Stage 2 in the desired setups is simply target-modality training
    # initialized from stage 1 weights, with reconstruction + model JSD only.
    original_load_flag = args.load_trained_model
    original_load_path = args.load_trained_model_path
    try:
        args.load_trained_model = True
        args.load_trained_model_path = args.stage1_model_run
        return train_loop_ABC(adata, cell_types, save_path, args)
    finally:
        args.load_trained_model = original_load_flag
        args.load_trained_model_path = original_load_path


def train_loop_ABC_stage3(adata, cell_types, save_path, args):
    return train_loop_ABC_stage3_v2(adata, cell_types, save_path, args)


def _run_stage3_validation_epoch(
    model, target_loaders, reference_loaders, common_celltypes, args,
    target_modality, epoch_accumulator=None, build_dump=False,
):
    """Run a single validation epoch (no gradient) and return losses + optional dump."""
    use_cross_recons = args.cross_recons_weight > 0
    total_loss = 0.0
    total_nuclei_recons = 0.0
    total_nuclei_jsd = 0.0
    total_cell_recons = 0.0
    total_cell_jsd = 0.0
    total_alignment_jsd = 0.0
    total_cross_recons = 0.0
    num_data = 0
    epoch_save_dict = {} if build_dump else None

    model.eval()
    with torch.no_grad():
        for celltype in common_celltypes:
            target_loader = target_loaders[celltype]
            reference_batches = list(reference_loaders[celltype])
            if not reference_batches:
                continue

            if build_dump:
                epoch_save_dict[celltype] = {}
            for batch_idx, target_batch in enumerate(target_loader):
                if batch_idx >= len(reference_batches):
                    break

                reference_batch_data = reference_batches[batch_idx]
                target_data = target_batch["data"].to(device)
                reference_data = reference_batch_data["data"].to(device)

                batch_size = min(target_data.size(0), reference_data.size(0))
                target_data = target_data[:batch_size]
                reference_data = reference_data[:batch_size]

                target_encoded, target_decoded, target_mu, target_log_var = model(target_data)
                reference_encoded, reference_decoded, reference_mu, reference_log_var = model(reference_data)

                target_loss, target_recons_loss, target_jsd_val = model.loss_function(
                    target_decoded, target_data, target_mu, target_log_var, args
                )[:3]
                reference_loss, reference_recons_loss, reference_jsd_val = model.loss_function(
                    reference_decoded, reference_data, reference_mu, reference_log_var, args
                )[:3]
                alignment_jsd = compute_mc_jsd_from_two_latents(
                    target_mu, target_log_var, reference_mu, reference_log_var
                )

                cross_recons_loss_val = torch.tensor(0.0, device=device)
                if use_cross_recons:
                    if target_modality == "nuclei":
                        cross_recons_loss_val = model.cross_recons_loss(
                            target_decoded, target_data, reference_data
                        )
                    else:
                        cross_recons_loss_val = model.cross_recons_loss(
                            reference_decoded, reference_data, target_data
                        )

                _recon_norm = getattr(args, "recon_loss_norm", 0.0)
                _do_norm = getattr(args, "normalize_losses", False) and _recon_norm > 0
                _LN2 = 0.6931471805599453
                if target_modality == "nuclei":
                    if _do_norm:
                        loss = (
                            args.nuclei_recons_weight * target_recons_loss / _recon_norm
                            + args.nuclei_jsd_weight * target_jsd_val / _LN2
                            + args.cell_recons_weight * reference_recons_loss / _recon_norm
                            + args.cell_jsd_weight * reference_jsd_val / _LN2
                            + args.alignment_weight * alignment_jsd / _LN2
                            + args.cross_recons_weight * cross_recons_loss_val / _recon_norm
                        )
                    else:
                        loss = (
                            args.nuclei_recons_weight * target_recons_loss
                            + args.nuclei_jsd_weight * target_jsd_val
                            + args.cell_recons_weight * reference_recons_loss
                            + args.cell_jsd_weight * reference_jsd_val
                            + args.alignment_weight * alignment_jsd
                            + args.cross_recons_weight * cross_recons_loss_val
                        )
                    nuclei_recons_loss, nuclei_jsd = target_recons_loss, target_jsd_val
                    cell_recons_loss, cell_jsd = reference_recons_loss, reference_jsd_val
                    nuclei_encoded, nuclei_decoded = target_encoded, target_decoded
                    nuclei_mu, nuclei_log_var = target_mu, target_log_var
                    cell_encoded, cell_decoded = reference_encoded, reference_decoded
                    cell_mu, cell_log_var = reference_mu, reference_log_var
                else:
                    if _do_norm:
                        loss = (
                            args.cell_recons_weight * target_recons_loss / _recon_norm
                            + args.cell_jsd_weight * target_jsd_val / _LN2
                            + args.nuclei_recons_weight * reference_recons_loss / _recon_norm
                            + args.nuclei_jsd_weight * reference_jsd_val / _LN2
                            + args.alignment_weight * alignment_jsd / _LN2
                            + args.cross_recons_weight * cross_recons_loss_val / _recon_norm
                        )
                    else:
                        loss = (
                            args.cell_recons_weight * target_recons_loss
                            + args.cell_jsd_weight * target_jsd_val
                            + args.nuclei_recons_weight * reference_recons_loss
                            + args.nuclei_jsd_weight * reference_jsd_val
                            + args.alignment_weight * alignment_jsd
                            + args.cross_recons_weight * cross_recons_loss_val
                        )
                    cell_recons_loss, cell_jsd = target_recons_loss, target_jsd_val
                    nuclei_recons_loss, nuclei_jsd = reference_recons_loss, reference_jsd_val
                    cell_encoded, cell_decoded = target_encoded, target_decoded
                    cell_mu, cell_log_var = target_mu, target_log_var
                    nuclei_encoded, nuclei_decoded = reference_encoded, reference_decoded
                    nuclei_mu, nuclei_log_var = reference_mu, reference_log_var

                if epoch_accumulator is not None:
                    accumulate_epoch_batch(epoch_accumulator, celltype, "cell", cell_decoded)
                    accumulate_epoch_batch(epoch_accumulator, celltype, "nuclei", nuclei_decoded)

                total_loss += loss.item() * batch_size
                total_nuclei_recons += nuclei_recons_loss.item() * batch_size
                total_nuclei_jsd += nuclei_jsd.item() * batch_size
                total_cell_recons += cell_recons_loss.item() * batch_size
                total_cell_jsd += cell_jsd.item() * batch_size
                total_alignment_jsd += alignment_jsd.item() * batch_size
                total_cross_recons += cross_recons_loss_val.item() * batch_size
                num_data += batch_size

                if build_dump:
                    # Extract metadata from target loader
                    original_cell_types = [
                        target_loader.dataset.get_original_value("cell_type", code)
                        for code in target_batch["cell_type"].cpu().numpy()[:batch_size]
                    ]
                    target_batch_ids = []
                    if target_loader.dataset.has_batch_id:
                        target_batch_ids = [
                            target_loader.dataset.get_original_value("batch_id", code)
                            for code in target_batch["batch_id"].cpu().numpy()[:batch_size]
                        ]
                    ref_loader = reference_loaders[celltype]
                    reference_batch_ids = []
                    if ref_loader.dataset.has_batch_id:
                        reference_batch_ids = [
                            ref_loader.dataset.get_original_value("batch_id", code)
                            for code in reference_batch_data["batch_id"].cpu().numpy()[:batch_size]
                        ]
                    target_sample_ids = []
                    if "sample_id" in target_batch:
                        target_sample_ids = list(target_batch["sample_id"][:batch_size])
                    reference_sample_ids = []
                    if "sample_id" in reference_batch_data:
                        reference_sample_ids = list(reference_batch_data["sample_id"][:batch_size])

                    epoch_save_dict[celltype][batch_idx] = {
                        "batch_idx": batch_idx,
                        "cell_type": target_batch["cell_type"].cpu().numpy()[:batch_size].tolist(),
                        "cell_type_original": original_cell_types,
                        "meta_batch_id": target_batch["batch_id"].cpu().numpy()[:batch_size].tolist(),
                        "meta_batch_id_original": target_batch_ids,
                        f"{target_modality}_meta_batch_id_original": target_batch_ids,
                        f"{'cell' if target_modality == 'nuclei' else 'nuclei'}_meta_batch_id_original": reference_batch_ids,
                        f"{target_modality}_sample_id": target_sample_ids,
                        f"{'cell' if target_modality == 'nuclei' else 'nuclei'}_sample_id": reference_sample_ids,
                        "nuclei_encoded": nuclei_encoded.detach().cpu().numpy().tolist(),
                        "nuclei_decoded": nuclei_decoded.detach().cpu().numpy().tolist(),
                        "nuclei_mu": nuclei_mu.detach().cpu().numpy().tolist(),
                        "nuclei_log_var": nuclei_log_var.detach().cpu().numpy().tolist(),
                        "cell_encoded": cell_encoded.detach().cpu().numpy().tolist(),
                        "cell_decoded": cell_decoded.detach().cpu().numpy().tolist(),
                        "cell_mu": cell_mu.detach().cpu().numpy().tolist(),
                        "cell_log_var": cell_log_var.detach().cpu().numpy().tolist(),
                        "loss": float(loss.item()),
                        "nuclei_recons_loss": float(nuclei_recons_loss.item()),
                        "nuclei_JSD_div": float(nuclei_jsd.item()),
                        "cell_recons_loss": float(cell_recons_loss.item()),
                        "cell_JSD_div": float(cell_jsd.item()),
                        "alignment_JSD_loss": float(alignment_jsd.item()),
                        "cross_recons_loss": float(cross_recons_loss_val.item()),
                    }

    model.train()

    if num_data == 0:
        return None, epoch_save_dict

    losses = {
        "loss": total_loss / num_data,
        "nuclei_recons_loss": total_nuclei_recons / num_data,
        "nuclei_JSD_div": total_nuclei_jsd / num_data,
        "cell_recons_loss": total_cell_recons / num_data,
        "cell_JSD_div": total_cell_jsd / num_data,
        "alignment_JSD_loss": total_alignment_jsd / num_data,
        "cross_recons_loss": total_cross_recons / num_data,
    }
    return losses, epoch_save_dict


def _write_valid_epoch_dump(save_path, epoch, epoch_save_dict):
    """Write validation epoch dump and return file path."""
    dump_dir = os.path.join(save_path, "epoch_dump")
    os.makedirs(dump_dir, exist_ok=True)
    valid_file_path = os.path.join(dump_dir, f"valid_epoch_{epoch}.json")
    with open(valid_file_path, "w") as fp:
        json.dump(epoch_save_dict, fp)
    print(f"  Valid epoch dump written: {valid_file_path}")
    return valid_file_path


def train_loop_ABC_stage3_v2(adata, cell_types, save_path, args):
    if not args.stage1_model_run:
        raise ValueError("--stage1_model_run is required for stage 3")

    _setup_training_device()

    target_modality = _infer_modality_from_data_path(args.data_path)
    reference_modality = "cell" if target_modality == "nuclei" else "nuclei"
    reference_data_path = _get_reference_data_path(args)
    reference_batch = _get_reference_batch(args)
    init_model_path = args.stage2_model_run or args.stage1_model_run

    print(f"Stage 3 target modality: {target_modality}")
    print(f"Stage 3 reference modality: {reference_modality}")
    print(f"Stage 3 init model: {init_model_path}")
    print(f"Stage 3 reference data path: {reference_data_path}")

    model = _create_model_from_args(args)
    model.load_state_dict(torch.load(init_model_path, map_location=device))
    model.to(device)

    with open(f"{save_path}/config.txt", "w") as f:
        f.write(str(model))

    if args.no_lr:
        optimizer = optim.Adam(model.parameters())
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if not hasattr(args, "alignment_weight") or args.alignment_weight is None:
        args.alignment_weight = args.alpha
    _prepare_stage3_weights(args)
    loss_list = [[
        "epoch",
        "loss",
        "nuclei_recons_loss",
        "nuclei_JSD_div",
        "cell_recons_loss",
        "cell_JSD_div",
        "alignment_JSD_loss",
        "cross_recons_loss",
    ]]

    target_train_loaders, target_valid_loaders, _ = create_dataloaders_by_cell_type(adata, args)
    reference_adata = _load_adata_for_path(reference_data_path, args)
    reference_train_loaders, reference_valid_loaders, _ = create_dataloaders_by_cell_type(
        reference_adata, args, batch_name=reference_batch
    )
    del reference_adata
    gc.collect()

    common_celltypes = sorted(set(target_train_loaders.keys()) & set(reference_train_loaders.keys()))
    valid_celltypes = sorted(
        set(target_valid_loaders.keys()) & set(reference_valid_loaders.keys())
    ) if target_valid_loaders and reference_valid_loaders else []
    run_validation = bool(getattr(args, "valid", True) and valid_celltypes)
    if run_validation:
        print(f"Validation enabled: {len(valid_celltypes)} common cell types")
    if not common_celltypes:
        raise RuntimeError("No shared cell types across target and reference modality loaders")

    raw_means = {
        target_modality: compute_loader_celltype_means(target_train_loaders, common_celltypes),
        reference_modality: compute_loader_celltype_means(reference_train_loaders, common_celltypes),
    }
    save_epoch_dump = getattr(args, "stage3_save_epoch_dump", False)
    dump_every = max(1, int(getattr(args, "stage3_dump_every", 50)))
    eval_every = max(1, int(getattr(args, "stage3_eval_every", 50)))

    # Load raw reference data once for full inline evaluation
    skip_full_eval = getattr(args, 'stage3_skip_full_eval', False)
    if skip_full_eval:
        print("  Full evaluation disabled (--stage3_skip_full_eval True)")
        raw_cell_adata = raw_nuclei_adata = None
        _common_genes = common_cell_types_set = None
        raw_cell_mean = raw_nuclei_mean = raw_common_genes = None
        use_log1p = False
    else:
        raw_cell_adata, raw_nuclei_adata, _common_genes, common_cell_types_set = (
            load_raw_reference_data(args)
        )
        use_log1p = getattr(args, 'raw_data_normalization', 'raw') == 'log1p'
        raw_cell_mean, raw_nuclei_mean, raw_common_genes = _prepare_raw_means_for_correlation(
            raw_cell_adata, raw_nuclei_adata, use_log1p=use_log1p,
        )

    for epoch in range(args.start_epoch, args.num_epochs):
        start_time = time.time()
        print(f"======================Epoch {epoch}===================")

        epoch_weight_values = {
            "nuclei_recons_weight": _update_stage3_weight(args, epoch, "nuclei_recons"),
            "nuclei_jsd_weight": _update_stage3_weight(args, epoch, "nuclei_jsd"),
            "cell_recons_weight": _update_stage3_weight(args, epoch, "cell_recons"),
            "cell_jsd_weight": _update_stage3_weight(args, epoch, "cell_jsd"),
            "alignment_weight": _update_stage3_weight(args, epoch, "alignment"),
            "cross_recons_weight": _update_stage3_weight(args, epoch, "cross_recons"),
        }
        args.alpha = args.alignment_weight
        use_cross_recons = args.cross_recons_weight > 0
        print(
            f"[Epoch {epoch}] Stage3 weights: "
            f"nuclei_recons={args.nuclei_recons_weight}, nuclei_jsd={args.nuclei_jsd_weight}, "
            f"cell_recons={args.cell_recons_weight}, cell_jsd={args.cell_jsd_weight}, "
            f"alignment={args.alignment_weight}, cross_recons={args.cross_recons_weight}"
        )

        total_loss = 0.0
        total_nuclei_recons_loss = 0.0
        total_nuclei_jsd = 0.0
        total_cell_recons_loss = 0.0
        total_cell_jsd = 0.0
        total_alignment_jsd = 0.0
        total_cross_recons_loss = 0.0
        num_data = 0
        batch_loss_list = []
        epoch_accumulator = init_epoch_accumulator(common_celltypes)
        is_eval_epoch = (epoch % eval_every == 0 or epoch == args.num_epochs - 1)
        is_checkpoint_epoch = (epoch % 10 == 0 or epoch == args.num_epochs - 1)
        is_dump_epoch = (epoch % dump_every == 0 or epoch == args.num_epochs - 1)
        build_dump = is_eval_epoch or (save_epoch_dump and is_dump_epoch)
        epoch_save_dict = {} if build_dump else None

        for celltype in common_celltypes:
            target_loader = target_train_loaders[celltype]
            reference_batches = list(reference_train_loaders[celltype])
            if not reference_batches:
                continue

            if build_dump:
                epoch_save_dict[celltype] = {}
            for batch_idx, target_batch in enumerate(target_loader):
                if batch_idx >= len(reference_batches):
                    break

                reference_batch_data = reference_batches[batch_idx]
                target_data = target_batch["data"].to(device)
                reference_data = reference_batch_data["data"].to(device)
                verify_gpu_usage(model, target_data, f"Batch {batch_idx}")

                batch_size = min(target_data.size(0), reference_data.size(0))
                target_data = target_data[:batch_size]
                reference_data = reference_data[:batch_size]

                target_encoded, target_decoded, target_mu, target_log_var = model(target_data)
                reference_encoded, reference_decoded, reference_mu, reference_log_var = model(reference_data)

                target_loss, target_recons_loss, target_jsd = model.loss_function(
                    target_decoded, target_data, target_mu, target_log_var, args
                )[:3]
                reference_loss, reference_recons_loss, reference_jsd = model.loss_function(
                    reference_decoded, reference_data, reference_mu, reference_log_var, args
                )[:3]
                alignment_jsd = compute_mc_jsd_from_two_latents(
                    target_mu, target_log_var, reference_mu, reference_log_var
                )

                # Cross-modality reconstruction: push decoded nuclei toward cell expression
                cross_recons_loss_val = torch.tensor(0.0, device=device)
                if use_cross_recons:
                    if target_modality == "nuclei":
                        # decoded nuclei (target) vs raw cell (reference)
                        cross_recons_loss_val = model.cross_recons_loss(
                            target_decoded, target_data, reference_data
                        )
                    else:
                        # decoded nuclei (reference) vs raw cell (target)
                        cross_recons_loss_val = model.cross_recons_loss(
                            reference_decoded, reference_data, target_data
                        )

                _recon_norm = getattr(args, "recon_loss_norm", 0.0)
                _do_norm = getattr(args, "normalize_losses", False) and _recon_norm > 0
                _LN2 = 0.6931471805599453  # np.log(2)
                if target_modality == "nuclei":
                    if _do_norm:
                        loss = (
                            args.nuclei_recons_weight * target_recons_loss / _recon_norm
                            + args.nuclei_jsd_weight * target_jsd / _LN2
                            + args.cell_recons_weight * reference_recons_loss / _recon_norm
                            + args.cell_jsd_weight * reference_jsd / _LN2
                            + args.alignment_weight * alignment_jsd / _LN2
                            + args.cross_recons_weight * cross_recons_loss_val / _recon_norm
                        )
                    else:
                        loss = (
                            args.nuclei_recons_weight * target_recons_loss
                            + args.nuclei_jsd_weight * target_jsd
                            + args.cell_recons_weight * reference_recons_loss
                            + args.cell_jsd_weight * reference_jsd
                            + args.alignment_weight * alignment_jsd
                            + args.cross_recons_weight * cross_recons_loss_val
                        )
                else:
                    if _do_norm:
                        loss = (
                            args.cell_recons_weight * target_recons_loss / _recon_norm
                            + args.cell_jsd_weight * target_jsd / _LN2
                            + args.nuclei_recons_weight * reference_recons_loss / _recon_norm
                            + args.nuclei_jsd_weight * reference_jsd / _LN2
                            + args.alignment_weight * alignment_jsd / _LN2
                            + args.cross_recons_weight * cross_recons_loss_val / _recon_norm
                        )
                    else:
                        loss = (
                            args.cell_recons_weight * target_recons_loss
                            + args.cell_jsd_weight * target_jsd
                            + args.nuclei_recons_weight * reference_recons_loss
                            + args.nuclei_jsd_weight * reference_jsd
                            + args.alignment_weight * alignment_jsd
                            + args.cross_recons_weight * cross_recons_loss_val
                        )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if target_modality == "nuclei":
                    nuclei_encoded, nuclei_decoded, nuclei_mu, nuclei_log_var = (
                        target_encoded,
                        target_decoded,
                        target_mu,
                        target_log_var,
                    )
                    cell_encoded, cell_decoded, cell_mu, cell_log_var = (
                        reference_encoded,
                        reference_decoded,
                        reference_mu,
                        reference_log_var,
                    )
                    nuclei_recons_loss, nuclei_jsd = target_recons_loss, target_jsd
                    cell_recons_loss, cell_jsd = reference_recons_loss, reference_jsd
                else:
                    cell_encoded, cell_decoded, cell_mu, cell_log_var = (
                        target_encoded,
                        target_decoded,
                        target_mu,
                        target_log_var,
                    )
                    nuclei_encoded, nuclei_decoded, nuclei_mu, nuclei_log_var = (
                        reference_encoded,
                        reference_decoded,
                        reference_mu,
                        reference_log_var,
                    )
                    cell_recons_loss, cell_jsd = target_recons_loss, target_jsd
                    nuclei_recons_loss, nuclei_jsd = reference_recons_loss, reference_jsd

                accumulate_epoch_batch(epoch_accumulator, celltype, "cell", cell_decoded)
                accumulate_epoch_batch(epoch_accumulator, celltype, "nuclei", nuclei_decoded)

                total_loss += loss.item() * batch_size
                total_nuclei_recons_loss += nuclei_recons_loss.item() * batch_size
                total_nuclei_jsd += nuclei_jsd.item() * batch_size
                total_cell_recons_loss += cell_recons_loss.item() * batch_size
                total_cell_jsd += cell_jsd.item() * batch_size
                total_alignment_jsd += alignment_jsd.item() * batch_size
                total_cross_recons_loss += cross_recons_loss_val.item() * batch_size
                num_data += batch_size

                # Extract target modality metadata
                original_cell_types = [
                    target_loader.dataset.get_original_value("cell_type", code)
                    for code in target_batch["cell_type"].cpu().numpy()[:batch_size]
                ]
                target_batch_ids = []
                if target_loader.dataset.has_batch_id:
                    target_batch_ids = [
                        target_loader.dataset.get_original_value("batch_id", code)
                        for code in target_batch["batch_id"].cpu().numpy()[:batch_size]
                    ]

                # Extract reference modality metadata
                ref_loader = reference_train_loaders[celltype]
                reference_batch_ids = []
                if ref_loader.dataset.has_batch_id:
                    reference_batch_ids = [
                        ref_loader.dataset.get_original_value("batch_id", code)
                        for code in reference_batch_data["batch_id"].cpu().numpy()[:batch_size]
                    ]

                # Extract sample IDs from both dataloaders
                target_sample_ids = []
                if "sample_id" in target_batch:
                    target_sample_ids = list(target_batch["sample_id"][:batch_size])
                reference_sample_ids = []
                if "sample_id" in reference_batch_data:
                    reference_sample_ids = list(reference_batch_data["sample_id"][:batch_size])

                if build_dump:
                    dump_entry = {
                        "batch_idx": batch_idx,
                        "cell_type": target_batch["cell_type"].cpu().numpy()[:batch_size].tolist(),
                        "cell_type_original": original_cell_types,
                        "meta_batch_id": target_batch["batch_id"].cpu().numpy()[:batch_size].tolist(),
                        "meta_batch_id_original": target_batch_ids,
                        f"{target_modality}_meta_batch_id_original": target_batch_ids,
                        f"{reference_modality}_meta_batch_id_original": reference_batch_ids,
                        f"{target_modality}_sample_id": target_sample_ids,
                        f"{reference_modality}_sample_id": reference_sample_ids,
                        "nuclei_encoded": nuclei_encoded.detach().cpu().numpy().tolist(),
                        "nuclei_decoded": nuclei_decoded.detach().cpu().numpy().tolist(),
                        "nuclei_mu": nuclei_mu.detach().cpu().numpy().tolist(),
                        "nuclei_log_var": nuclei_log_var.detach().cpu().numpy().tolist(),
                        "cell_encoded": cell_encoded.detach().cpu().numpy().tolist(),
                        "cell_decoded": cell_decoded.detach().cpu().numpy().tolist(),
                        "cell_mu": cell_mu.detach().cpu().numpy().tolist(),
                        "cell_log_var": cell_log_var.detach().cpu().numpy().tolist(),
                        "loss": float(loss.detach().cpu().item()),
                        "nuclei_recons_loss": float(nuclei_recons_loss.detach().cpu().item()),
                        "nuclei_JSD_div": float(nuclei_jsd.detach().cpu().item()),
                        "cell_recons_loss": float(cell_recons_loss.detach().cpu().item()),
                        "cell_JSD_div": float(cell_jsd.detach().cpu().item()),
                        "alignment_JSD_loss": float(alignment_jsd.detach().cpu().item()),
                        "cross_recons_loss": float(cross_recons_loss_val.detach().cpu().item()),
                    }
                    epoch_save_dict[celltype][batch_idx] = dump_entry

                batch_loss_list.append([
                    celltype,
                    batch_idx,
                    float(loss.detach().cpu().item()),
                    float(nuclei_recons_loss.detach().cpu().item()),
                    float(nuclei_jsd.detach().cpu().item()),
                    float(cell_recons_loss.detach().cpu().item()),
                    float(cell_jsd.detach().cpu().item()),
                    float(alignment_jsd.detach().cpu().item()),
                    float(cross_recons_loss_val.detach().cpu().item()),
                    batch_size,
                ])

        if num_data == 0:
            raise RuntimeError("Stage 3 found no paired training batches across modalities")

        epoch_loss = total_loss / num_data
        epoch_nuclei_recons_loss = total_nuclei_recons_loss / num_data
        epoch_nuclei_jsd = total_nuclei_jsd / num_data
        epoch_cell_recons_loss = total_cell_recons_loss / num_data
        epoch_cell_jsd = total_cell_jsd / num_data
        epoch_alignment_jsd = total_alignment_jsd / num_data
        epoch_cross_recons_loss = total_cross_recons_loss / num_data

        # Auto-compute recon loss normalization constant from epoch 0
        if getattr(args, "normalize_losses", False) and epoch == 0 and getattr(args, "recon_loss_norm", 0.0) == 0.0:
            args.recon_loss_norm = epoch_cell_recons_loss
            print(f"[normalize_losses] Set recon_loss_norm = {args.recon_loss_norm:.6f} from epoch 0 cell recon")

        loss_list.append([
            epoch,
            epoch_loss,
            epoch_nuclei_recons_loss,
            epoch_nuclei_jsd,
            epoch_cell_recons_loss,
            epoch_cell_jsd,
            epoch_alignment_jsd,
            epoch_cross_recons_loss,
        ])

        if is_checkpoint_epoch:
            os.makedirs(os.path.join(save_path, "models"), exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_path, "models", "model.pt"))
            _write_stage3_batch_losses(save_path, epoch, batch_loss_list)

        if is_eval_epoch:
            # 1. Compact inline eval (lightweight, from accumulator)
            run_stage3_inline_evaluation(save_path, epoch, epoch_accumulator, raw_means)

            # 2. Write full epoch dump
            dump_path = _write_epoch_dump(save_path, epoch, epoch_save_dict)

            # 2b. Inline UMAP + correlation map (needs dump)
            run_stage3_inline_umap_correlation(
                save_path, epoch, dump_path, epoch_accumulator, raw_means
            )

            # 3. Full evaluation stack (heatmaps, correlation, scIB)
            if not getattr(args, 'stage3_skip_full_eval', False):
                run_stage3_full_evaluation(
                    save_path, epoch, dump_path,
                    raw_cell_adata, raw_nuclei_adata,
                    common_cell_types_set,
                    raw_cell_mean, raw_nuclei_mean, raw_common_genes,
                    use_log1p=use_log1p,
                )

            # 4. Cleanup dump if not retaining
            if not save_epoch_dump:
                try:
                    os.remove(dump_path)
                    print(f"  Epoch dump deleted: {dump_path}")
                except OSError:
                    pass
        elif save_epoch_dump and is_dump_epoch and build_dump:
            # Non-eval checkpoint epoch with dump retention enabled
            _save_stage3_epoch_outputs(save_path, epoch, epoch_save_dict, batch_loss_list)

        # ---- Validation ----
        valid_losses = None
        if run_validation:
            valid_accumulator = init_epoch_accumulator(valid_celltypes) if is_eval_epoch else None
            valid_losses, valid_save_dict = _run_stage3_validation_epoch(
                model, target_valid_loaders, reference_valid_loaders,
                valid_celltypes, args, target_modality,
                epoch_accumulator=valid_accumulator,
                build_dump=is_eval_epoch,
            )

            if valid_losses is not None:
                print(
                    f"  [Valid] loss={valid_losses['loss']:.6f}, "
                    f"nuc_recons={valid_losses['nuclei_recons_loss']:.6f}, "
                    f"cell_recons={valid_losses['cell_recons_loss']:.6f}, "
                    f"align_JSD={valid_losses['alignment_JSD_loss']:.6f}"
                )

                # Write valid epoch loss
                os.makedirs(os.path.join(save_path, "valid_epoch_loss"), exist_ok=True)
                valid_loss_file = os.path.join(save_path, "valid_epoch_loss", "loss.txt")
                valid_file_exists = os.path.isfile(valid_loss_file)
                with open(valid_loss_file, "a" if valid_file_exists else "w") as vf:
                    if not valid_file_exists:
                        vf.write(
                            "epoch,loss,nuclei_recons_loss,nuclei_JSD_div,"
                            "cell_recons_loss,cell_JSD_div,alignment_JSD_loss,"
                            "cross_recons_loss\n"
                        )
                    vf.write(",".join(map(str, [
                        epoch,
                        valid_losses["loss"],
                        valid_losses["nuclei_recons_loss"],
                        valid_losses["nuclei_JSD_div"],
                        valid_losses["cell_recons_loss"],
                        valid_losses["cell_JSD_div"],
                        valid_losses["alignment_JSD_loss"],
                        valid_losses["cross_recons_loss"],
                    ])) + "\n")

            if is_eval_epoch and valid_accumulator is not None and valid_losses is not None:
                # Inline eval on valid set (uses train raw_means as reference)
                valid_eval_dir = os.path.join(save_path, "valid_inline_eval")
                os.makedirs(valid_eval_dir, exist_ok=True)
                run_stage3_inline_evaluation(
                    save_path, epoch, valid_accumulator, raw_means,
                    output_subdir="valid_inline_eval",
                )

                # Write valid dump, generate UMAPs + correlation, then cleanup
                if valid_save_dict:
                    valid_dump_path = _write_valid_epoch_dump(save_path, epoch, valid_save_dict)
                    run_stage3_inline_umap_correlation(
                        save_path, epoch, valid_dump_path, valid_accumulator, raw_means,
                        output_subdir="valid_inline_eval",
                        umap_data_split="valid",
                    )
                    # Cleanup valid dump
                    try:
                        os.remove(valid_dump_path)
                        print(f"  Valid epoch dump deleted: {valid_dump_path}")
                    except OSError:
                        pass

        runtime_str = _runtime_str(start_time)
        cross_recons_str = f", cross_recons_loss={epoch_cross_recons_loss}" if use_cross_recons else ""
        print(
            f"Epoch {epoch+1}/{args.num_epochs} completed. "
            f"loss={epoch_loss}, nuclei_recons_loss={epoch_nuclei_recons_loss}, "
            f"nuclei_JSD_div={epoch_nuclei_jsd}, cell_recons_loss={epoch_cell_recons_loss}, "
            f"cell_JSD_div={epoch_cell_jsd}, alignment_JSD_loss={epoch_alignment_jsd}"
            f"{cross_recons_str}"
        )

        os.makedirs(os.path.join(save_path, "train_epoch_loss"), exist_ok=True)
        loss_file = os.path.join(save_path, "train_epoch_loss", "loss.txt")
        file_exists = os.path.isfile(loss_file)
        with open(loss_file, "a" if file_exists else "w") as file:
            if not file_exists:
                file.write(
                    "epoch,loss,nuclei_recons_loss,nuclei_JSD_div,cell_recons_loss,cell_JSD_div,alignment_JSD_loss,cross_recons_loss,nuclei_recons_weight,nuclei_jsd_weight,cell_recons_weight,cell_jsd_weight,alignment_weight,cross_recons_weight,runtime\n"
                )
            file.write(",".join(map(str, [
                epoch,
                epoch_loss,
                epoch_nuclei_recons_loss,
                epoch_nuclei_jsd,
                epoch_cell_recons_loss,
                epoch_cell_jsd,
                epoch_alignment_jsd,
                epoch_cross_recons_loss,
                epoch_weight_values["nuclei_recons_weight"],
                epoch_weight_values["nuclei_jsd_weight"],
                epoch_weight_values["cell_recons_weight"],
                epoch_weight_values["cell_jsd_weight"],
                epoch_weight_values["alignment_weight"],
                epoch_weight_values["cross_recons_weight"],
                runtime_str,
            ])) + "\n")

    plot_loss_curve_stage3_v2(save_path)
    return model, loss_list
