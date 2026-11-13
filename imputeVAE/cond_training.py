"""
Training loop for modality-conditioned VAE (VAE_NB_Cond).

The decoder receives a modality embedding alongside z. During training each
modality reconstructs with its own embedding; at eval time we cross-decode
(nuclei latent + cell embedding) to produce cell-like output.
"""

import gc
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim

from .data import create_dataloaders_by_cell_type, load_raw_data
from .divergences import compute_mc_jsd_from_two_latents
from .models_cond import VAE_JSD_Cond, VAE_NB_Cond
from .stage3_eval import (
    accumulate_epoch_batch,
    compute_loader_celltype_means,
    init_epoch_accumulator,
    load_raw_reference_data,
    run_stage3_full_evaluation,
    run_stage3_inline_evaluation,
    run_stage3_inline_umap_correlation,
    _prepare_raw_means_for_correlation,
    _plot_gene_corr_history,
    _plot_shift_history,
    _compute_decoded_means,
    _corr,
    _log_mae,
    _r_squared,
    _wasserstein_1d,
    _append_csv,
    _build_mean_dataframes,
)
from .staged_training import (
    _setup_training_device,
    _infer_modality_from_data_path,
    _get_reference_data_path,
    _get_reference_batch,
    _load_adata_for_path,
    _prepare_stage3_weights,
    _update_stage3_weight,
    _write_epoch_dump,
    _write_valid_epoch_dump,
    _write_stage3_batch_losses,
    _save_stage3_epoch_outputs,
    _runtime_str,
)
from .utils import device, verify_gpu_usage
from .visualization import plot_umap_after_train_by_celltype_stage3, plot_cond_umaps

# Modality ID constants
CELL_ID = 0
NUCLEI_ID = 1


# ============================================================================
# Cross-decode evaluation helpers
# ============================================================================


def _cross_decode_kwargs(model, libsize_targets, celltype):
    """Extra decode_cross kwargs: per-cell-type target library when the model has the covariate."""
    if libsize_targets and isinstance(model, VAE_JSD_Cond) and model.libsize_cond \
            and celltype in libsize_targets:
        return {"target_log_libsize": float(libsize_targets[celltype])}
    return {}


def _run_cross_decode_on_loaders(model, nuclei_loaders, celltypes, device_, libsize_targets=None,
                                 library_remaps=None):
    """Run cross-decode (nuclei→cell embedding) and return per-celltype means.

    Args:
        libsize_targets: optional {celltype: log1p library size} used as the decode
            target for libsize_cond models (per-type median training-cell library).
            None = decode at each nucleus's own library size.
        library_remaps: optional {celltype: LongTensor local batch code -> vocabulary id}
            for library_cond models (source library of each nucleus; the target library
            is UNKNOWN = the learned marginal library).

    Returns:
        cross_decoded_means: dict of celltype -> 1D numpy mean array
    """
    sums = {}
    counts = {}
    model.eval()
    with torch.no_grad():
        for celltype in celltypes:
            loader = nuclei_loaders[celltype]
            ct_sum = None
            ct_count = 0
            extra = _cross_decode_kwargs(model, libsize_targets, celltype)
            for batch in loader:
                data = batch["data"].to(device_)
                lib_kw = {}
                if library_remaps is not None and celltype in library_remaps:
                    lib_kw["source_library_id"] = library_remaps[celltype][batch["batch_id"]].to(device_)
                cross_dec = model.decode_cross(data, source_modality_id=NUCLEI_ID,
                                               target_modality_id=CELL_ID, **extra, **lib_kw)
                batch_np = cross_dec.detach().cpu().numpy().astype(np.float32)
                if ct_sum is None:
                    ct_sum = batch_np.sum(axis=0)
                else:
                    ct_sum += batch_np.sum(axis=0)
                ct_count += batch_np.shape[0]
            if ct_count > 0:
                sums[celltype] = ct_sum / ct_count
                counts[celltype] = ct_count
    model.train()
    return sums


def _run_cross_decode_eval(model, nuclei_loaders, celltypes, raw_means,
                           epoch, output_dir, device_, label_prefix="", libsize_targets=None,
                           library_remaps=None):
    """Compute cross-decode metrics: shift scores and gene correlation.

    Writes:
        cross_decode_shift_summary.csv
        cross_decode_gene_corr_summary.csv
    """
    cross_means = _run_cross_decode_on_loaders(model, nuclei_loaders, celltypes, device_,
                                               libsize_targets=libsize_targets, library_remaps=library_remaps)
    if not cross_means:
        return

    os.makedirs(output_dir, exist_ok=True)

    # Shift scores for cross-decoded output
    common_cts = sorted(
        set(cross_means.keys())
        & set(raw_means.get("cell", {}).keys())
        & set(raw_means.get("nuclei", {}).keys())
    )
    if not common_cts:
        return

    per_ct_shift = []
    decoded_matrix = []
    raw_cell_matrix = []
    raw_nuclei_matrix = []
    for celltype in common_cts:
        dec = cross_means[celltype]
        raw_cell = raw_means["cell"][celltype]
        raw_nuclei = raw_means["nuclei"][celltype]
        per_ct_shift.append({
            "epoch": epoch,
            "decoded_modality": "cross_dec",
            "cell_type": celltype,
            "corr_to_raw_cell": _corr(dec, raw_cell),
            "corr_to_raw_nuclei": _corr(dec, raw_nuclei),
        })
        decoded_matrix.append(dec)
        raw_cell_matrix.append(raw_cell)
        raw_nuclei_matrix.append(raw_nuclei)

    per_ct_df = pd.DataFrame(per_ct_shift)
    per_ct_df["shift"] = per_ct_df["corr_to_raw_cell"] - per_ct_df["corr_to_raw_nuclei"]

    per_ct_path = os.path.join(output_dir, f"{label_prefix}cross_decode_shift_per_celltype_epoch_{epoch}.csv")
    per_ct_df.to_csv(per_ct_path, index=False)

    shift_row = {
        "epoch": epoch,
        "decoded_modality": "cross_dec",
        "mean_shift": float(per_ct_df["shift"].mean()),
    }
    shift_csv = os.path.join(output_dir, f"{label_prefix}cross_decode_shift_summary.csv")
    _append_csv(pd.DataFrame([shift_row]), shift_csv)
    _plot_shift_history(shift_csv, output_dir, "cross_dec")

    # Gene-level metrics for cross-decoded
    decoded_matrix = np.stack(decoded_matrix, axis=0)
    raw_cell_matrix = np.stack(raw_cell_matrix, axis=0)
    raw_nuclei_matrix = np.stack(raw_nuclei_matrix, axis=0)
    corr_vs_cell = []
    corr_vs_nuclei = []
    mae_vs_cell = []
    mae_vs_nuclei = []
    r2_vs_cell = []
    r2_vs_nuclei = []
    wd_vs_cell = []
    wd_vs_nuclei = []
    for idx in range(decoded_matrix.shape[1]):
        dec_col = decoded_matrix[:, idx]
        raw_cell_col = raw_cell_matrix[:, idx]
        raw_nuc_col = raw_nuclei_matrix[:, idx]
        corr_vs_cell.append(_corr(dec_col, raw_cell_col))
        corr_vs_nuclei.append(_corr(dec_col, raw_nuc_col))
        mae_vs_cell.append(_log_mae(dec_col, raw_cell_col))
        mae_vs_nuclei.append(_log_mae(dec_col, raw_nuc_col))
        r2_vs_cell.append(_r_squared(dec_col, raw_cell_col))
        r2_vs_nuclei.append(_r_squared(dec_col, raw_nuc_col))
        wd_vs_cell.append(_wasserstein_1d(dec_col, raw_cell_col))
        wd_vs_nuclei.append(_wasserstein_1d(dec_col, raw_nuc_col))
    gene_row = {
        "epoch": epoch,
        "decoded_modality": "cross_dec",
        "median_corr_vs_raw_cell": float(np.nanmedian(corr_vs_cell)),
        "median_corr_vs_raw_nuclei": float(np.nanmedian(corr_vs_nuclei)),
        "median_log_mae_vs_raw_cell": float(np.nanmedian(mae_vs_cell)),
        "median_log_mae_vs_raw_nuclei": float(np.nanmedian(mae_vs_nuclei)),
        "median_r2_vs_raw_cell": float(np.nanmedian(r2_vs_cell)),
        "median_r2_vs_raw_nuclei": float(np.nanmedian(r2_vs_nuclei)),
        "median_wasserstein_vs_raw_cell": float(np.nanmedian(wd_vs_cell)),
        "median_wasserstein_vs_raw_nuclei": float(np.nanmedian(wd_vs_nuclei)),
    }
    gene_csv = os.path.join(output_dir, f"{label_prefix}cross_decode_gene_corr_summary.csv")
    _append_csv(pd.DataFrame([gene_row]), gene_csv)
    _plot_gene_corr_history(gene_csv, output_dir, "cross_dec")

    print(f"  Cross-decode eval: shift={shift_row['mean_shift']:.4f}, "
          f"gene_corr_vs_cell={gene_row['median_corr_vs_raw_cell']:.4f}, "
          f"log_mae_vs_cell={gene_row['median_log_mae_vs_raw_cell']:.4f}, "
          f"r2_vs_cell={gene_row['median_r2_vs_raw_cell']:.4f}")

    return cross_means


def _run_cross_decode_correlation(cross_means, decoded_means, raw_means,
                                  epoch, output_dir, label_prefix=""):
    """5x5 correlation matrix: raw_cell, raw_nuc, dec_cell, dec_nuc, cross_dec.

    Writes correlation heatmap and history CSV.
    """
    # Build mean DataFrames for all 5 sources
    mean_dict = _build_mean_dataframes(decoded_means, raw_means)
    if cross_means:
        cross_df = pd.DataFrame.from_dict(cross_means, orient="index")
        if not cross_df.empty:
            mean_dict["cross_dec"] = cross_df

    if len(mean_dict) < 2:
        return

    os.makedirs(output_dir, exist_ok=True)
    corr_dir = os.path.join(output_dir, f"{label_prefix}correlation_epoch_{epoch}")

    try:
        from eval_correlation import run_correlation_analysis
    except ImportError:
        import sys
        _analysis_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis")
        if _analysis_dir not in sys.path:
            sys.path.insert(0, _analysis_dir)
        from eval_correlation import run_correlation_analysis

    try:
        per_ct_corr, avg_corr = run_correlation_analysis(
            mean_dict, corr_dir, prefix=f"{label_prefix}5x5_"
        )
        # Save to history
        avg_row = {"epoch": epoch}
        for src1 in avg_corr.index:
            for src2 in avg_corr.columns:
                if src1 < src2:
                    avg_row[f"{src1}_vs_{src2}"] = float(avg_corr.loc[src1, src2])
        corr_history_csv = os.path.join(output_dir, f"{label_prefix}correlation_5x5_history.csv")
        _append_csv(pd.DataFrame([avg_row]), corr_history_csv)
        print(f"  5x5 correlation saved to {corr_dir}")
    except Exception as e:
        print(f"  5x5 correlation failed: {e}")


def _build_cross_decode_dump_entry(model, nuclei_data, device_, **decode_kwargs):
    """Compute cross-decoded output for a batch and return numpy array."""
    cross_dec = model.decode_cross(nuclei_data, source_modality_id=NUCLEI_ID,
                                   target_modality_id=CELL_ID, **decode_kwargs)
    return cross_dec.detach().cpu().numpy().tolist()


# ============================================================================
# Library (batch) covariate vocabulary (library_cond, 2026-11-06)
# ============================================================================


def _modality_of_library(name):
    """'4-c3A' -> 'cell', '2-n3A' -> 'nuclei' (meta_batch_id = <donor>-<c|n><version><sample>)."""
    tag = str(name).split("-", 1)[1] if "-" in str(name) else str(name)
    return "cell" if tag[:1] == "c" else "nuclei"


def build_library_vocab(cell_loaders, nuclei_loaders, celltypes):
    """One vocabulary of sequencing libraries (meta_batch_id) over both modalities' TRAINING loaders.

    Each per-type myDataset factorises meta_batch_id on its own, so the batch codes in the loaders are
    local; this builds the global, deterministic (sorted) mapping name -> id with 1..N (0 = UNKNOWN).
    """
    names = set()
    for loaders in (cell_loaders, nuclei_loaders):
        for ct in celltypes:
            ds = loaders[ct].dataset
            if getattr(ds, "has_batch_id", False):
                names.update(str(x) for x in ds.batch_id_mapping)
    names = sorted(names, key=lambda n: (_modality_of_library(n), n))
    vocab = {"unknown_id": 0, "names": names, "ids": {n: i + 1 for i, n in enumerate(names)},
             "modality": {n: _modality_of_library(n) for n in names},
             "donor": {n: str(n).split("-", 1)[0] for n in names}}
    return vocab


def build_library_remaps(loaders, celltypes, vocab):
    """{celltype: LongTensor} mapping each loader's local batch codes to vocabulary ids (unseen -> 0)."""
    remaps = {}
    for ct in celltypes:
        if ct not in loaders:
            continue
        ds = loaders[ct].dataset
        if not getattr(ds, "has_batch_id", False):
            continue
        remaps[ct] = torch.tensor([vocab["ids"].get(str(n), 0) for n in ds.batch_id_mapping], dtype=torch.long)
    return remaps


def _library_ids_for_batch(remaps, celltype, batch, n, device_):
    """Vocabulary ids [n] for the first n samples of a loader batch, or None when the model has no library covariate."""
    if remaps is None or celltype not in remaps:
        return None
    return remaps[celltype][batch["batch_id"][:n]].to(device_)


# ============================================================================
# Library-size covariate statistics (libsize_cond)
# ============================================================================


_LIBSIZE_QUANTILE_GRID = [round(q / 100.0, 2) for q in range(0, 101)]


def compute_libsize_stats(cell_loaders, nuclei_loaders, celltypes):
    """Library-size statistics of the TRAINING partition for the libsize_cond model.

    Library size = log1p(total raw counts over the training gene set), recovered from the
    log1p-normalised loader tensors (expm1 -> sum).  Returns a JSON-serialisable dict:
      mean / std          pooled over both modalities (standardisation constants, fixed at train time)
      ct_median_cell      per cell type, median training-cell library  -> `ct_median_cell` decode target
      ct_median_nuclei    per cell type, median training-nucleus library
      ct_quantiles_*      per cell type, 0..100 % quantiles          -> `quantile_map` decode target
    """
    per_ct = {"cell": {}, "nuclei": {}}
    for modality, loaders in (("cell", cell_loaders), ("nuclei", nuclei_loaders)):
        for ct in celltypes:
            vals = []
            for batch in loaders[ct]:
                vals.append(VAE_JSD_Cond.log_library_size(batch["data"]).squeeze(1).numpy())
            per_ct[modality][ct] = np.concatenate(vals) if vals else np.zeros(0, dtype=np.float32)
    pooled = np.concatenate([v for m in per_ct.values() for v in m.values()]).astype(np.float64)
    if pooled.size < 2:
        raise RuntimeError("compute_libsize_stats: no training samples found")
    grid = np.asarray(_LIBSIZE_QUANTILE_GRID)
    stats = {
        "definition": "log1p(sum of raw counts over the training gene set); computed on the training partition",
        "mean": float(pooled.mean()),
        "std": float(pooled.std()),
        "n_cell": int(sum(len(v) for v in per_ct["cell"].values())),
        "n_nuclei": int(sum(len(v) for v in per_ct["nuclei"].values())),
        "quantile_grid": _LIBSIZE_QUANTILE_GRID,
    }
    for modality in ("cell", "nuclei"):
        pooled_m = np.concatenate(list(per_ct[modality].values())).astype(np.float64)
        if pooled_m.size:  # pooled (label-free) statistics, added 2026-10-23 for the global_* decode modes
            stats[f"global_median_{modality}"] = float(np.median(pooled_m))
            stats[f"global_quantiles_{modality}"] = [float(x) for x in np.quantile(pooled_m, grid)]
        stats[f"ct_median_{modality}"] = {
            ct: float(np.median(v)) for ct, v in per_ct[modality].items() if len(v)
        }
        stats[f"ct_quantiles_{modality}"] = {
            ct: [float(x) for x in np.quantile(v, grid)] for ct, v in per_ct[modality].items() if len(v)
        }
    return stats


# ============================================================================
# Resumable-run bookkeeping (scavenge preemption restarts the job from scratch otherwise)
# ============================================================================


def _write_run_state(save_path, epoch, args):
    """Record the last checkpointed epoch and the auto-computed loss normaliser."""
    state = {
        "last_checkpoint_epoch": int(epoch),
        "recon_loss_norm": float(getattr(args, "recon_loss_norm", 0.0) or 0.0),
    }
    with open(os.path.join(save_path, "run_state.json"), "w") as f:
        json.dump(state, f, indent=2)


def _load_run_state(run_path):
    p = os.path.join(run_path, "run_state.json")
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        return json.load(f)


def _truncate_loss_file(path, start_epoch):
    """Drop rows with epoch >= start_epoch so a resumed run does not duplicate epochs."""
    if not os.path.isfile(path):
        return
    with open(path) as f:
        lines = f.readlines()
    if not lines:
        return
    kept = [lines[0]]
    for line in lines[1:]:
        try:
            if int(line.split(",")[0]) < start_epoch:
                kept.append(line)
        except ValueError:
            kept.append(line)
    with open(path, "w") as f:
        f.writelines(kept)


# ============================================================================
# Main training loop
# ============================================================================


def train_cond_stage3(adata, cell_types, save_path, args):
    """Train modality-conditioned VAE (stage 3 equivalent).

    When --stage1_model_run is provided, loads pretrained weights (two-stage).
    When omitted, trains from scratch (single-stage cVAE).
    """
    _setup_training_device()
    global device
    from .utils import device

    target_modality = _infer_modality_from_data_path(args.data_path)
    reference_modality = "cell" if target_modality == "nuclei" else "nuclei"
    reference_data_path = _get_reference_data_path(args)
    reference_batch = _get_reference_batch(args)
    init_model_path = getattr(args, "stage1_model_run", None)

    print(f"Conditioned Stage 3 target modality: {target_modality}")
    print(f"Conditioned Stage 3 reference modality: {reference_modality}")
    print(f"Conditioned Stage 3 init model: {init_model_path or '(from scratch)'}")
    print(f"Conditioned Stage 3 reference data path: {reference_data_path}")

    embed_dim = getattr(args, "embed_dim", 8)
    scalar_cond = getattr(args, "scalar_cond", False)
    cond_encoder = getattr(args, "cond_encoder", False)
    libsize_cond = bool(getattr(args, "libsize_cond", False))
    libsize_target_mode = getattr(args, "libsize_target", "ct_median_cell")
    if libsize_cond and args.model != "vae_js":
        raise ValueError("--libsize_cond is implemented for the MSE/JSD model (vae_js) only")
    library_cond = bool(getattr(args, "library_cond", False))
    if library_cond and args.model != "vae_js":
        raise ValueError("--library_cond is implemented for the MSE/JSD model (vae_js) only")
    if args.model == "vae_nb":
        model = VAE_NB_Cond(
            input_size=args.num_feature,
            num_internal=args.num_internal,
            num_latent=args.num_latent,
            embed_dim=embed_dim,
            use_zinb=getattr(args, "use_zinb", False),
            scalar_cond=scalar_cond,
        )
    elif args.model == "vae_js":
        model = VAE_JSD_Cond(
            input_size=args.num_feature,
            num_internal=args.num_internal,
            num_latent=args.num_latent,
            embed_dim=embed_dim,
            scalar_cond=scalar_cond,
            cond_encoder=cond_encoder,
            libsize_cond=libsize_cond,
            library_cond=library_cond,
            library_capacity=int(getattr(args, "library_capacity", 32)),
            library_dim=int(getattr(args, "library_dim", 4)),
            library_dropout=float(getattr(args, "library_dropout", 0.1)),
        )
        if libsize_cond:
            print(f"  Library-size covariate ON (encoder+decoder); inline cross-decode target = {libsize_target_mode}")
        if library_cond:
            print(f"  Library (batch) covariate ON (encoder+decoder): embedding dim {model.library_dim}, "
                  f"capacity {model.library_capacity}, dropout->UNKNOWN p={model.library_dropout}")
    else:
        raise ValueError(f"Unsupported model for conditioned training: {args.model}. "
                         f"Expected 'vae_nb' or 'vae_js'.")
    # If continuing a previous run, load full checkpoint;
    # if stage1 model provided, expand from stage1;
    # otherwise train from scratch.
    if getattr(args, "continue_training", False) and args.continue_training_run_path:
        ckpt_path = os.path.join(args.continue_training_run_path, "models", "model.pt")
        print(f"  Continuing from checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    elif init_model_path:
        model.load_pretrained_partial(init_model_path)
    else:
        print("  Training from scratch (no pretrained weights)")
    model.to(device)

    with open(f"{save_path}/config.txt", "w") as f:
        f.write(str(model))

    if args.no_lr:
        optimizer = optim.Adam(model.parameters())
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if not hasattr(args, "alignment_weight") or args.alignment_weight is None:
        args.alignment_weight = args.alpha
    if not hasattr(args, "cross_recons_weight") or args.cross_recons_weight is None:
        args.cross_recons_weight = 0.0
    if not hasattr(args, "residual_weight") or args.residual_weight is None:
        args.residual_weight = 0.0
    _prepare_stage3_weights(args)

    # Resume bookkeeping: recover the epoch-0 loss normaliser and drop loss rows past the checkpoint
    if getattr(args, "continue_training", False) and args.continue_training_run_path:
        prev_state = _load_run_state(args.continue_training_run_path)
        if prev_state:
            if getattr(args, "normalize_losses", False) and not getattr(args, "recon_loss_norm", 0.0):
                args.recon_loss_norm = prev_state.get("recon_loss_norm", 0.0)
                print(f"  Resumed recon_loss_norm = {args.recon_loss_norm:.6f} from run_state.json")
            print(f"  run_state.json: last checkpoint epoch {prev_state.get('last_checkpoint_epoch')}, "
                  f"resuming at start_epoch {args.start_epoch}")
        elif getattr(args, "normalize_losses", False) and not getattr(args, "recon_loss_norm", 0.0) \
                and args.start_epoch > 0:
            print("  WARNING: resuming a normalize_losses run without run_state.json; "
                  "recon_loss_norm unknown -> losses will be UNnormalised. Pass --recon_loss_norm explicitly.")
        for sub in ("train_epoch_loss", "valid_epoch_loss"):
            _truncate_loss_file(os.path.join(save_path, sub, "loss.txt"), args.start_epoch)

    loss_list = [[
        "epoch", "loss",
        "nuclei_recons_loss", "nuclei_JSD_div",
        "cell_recons_loss", "cell_JSD_div",
        "alignment_JSD_loss", "cross_recons_loss",
        "residual_loss",
    ]]

    # ---------- Data loading ----------
    target_train_loaders, target_valid_loaders, _ = create_dataloaders_by_cell_type(adata, args)
    reference_adata = _load_adata_for_path(reference_data_path, args)
    reference_train_loaders, reference_valid_loaders, _ = create_dataloaders_by_cell_type(
        reference_adata, args, batch_name=reference_batch
    )
    del reference_adata
    gc.collect()

    common_celltypes = sorted(
        set(target_train_loaders.keys()) & set(reference_train_loaders.keys())
    )
    valid_celltypes = sorted(
        set(target_valid_loaders.keys()) & set(reference_valid_loaders.keys())
    ) if target_valid_loaders and reference_valid_loaders else []
    run_validation = bool(getattr(args, "valid", True) and valid_celltypes)
    if run_validation:
        print(f"Validation enabled: {len(valid_celltypes)} common cell types")
    if not common_celltypes:
        raise RuntimeError("No shared cell types across target and reference modality loaders")

    # Determine which loader is which modality
    if target_modality == "nuclei":
        nuclei_train_loaders = target_train_loaders
        cell_train_loaders = reference_train_loaders
        nuclei_valid_loaders = target_valid_loaders
        cell_valid_loaders = reference_valid_loaders
    else:
        cell_train_loaders = target_train_loaders
        nuclei_train_loaders = reference_train_loaders
        cell_valid_loaders = target_valid_loaders
        nuclei_valid_loaders = reference_valid_loaders

    raw_means = {
        "cell": compute_loader_celltype_means(cell_train_loaders, common_celltypes),
        "nuclei": compute_loader_celltype_means(nuclei_train_loaders, common_celltypes),
    }

    # Library-size covariate: standardisation constants + per-type decode targets from the TRAIN partition
    libsize_targets = None
    if libsize_cond:
        libsize_stats = compute_libsize_stats(cell_train_loaders, nuclei_train_loaders, common_celltypes)
        model.set_libsize_stats(libsize_stats["mean"], libsize_stats["std"])
        with open(os.path.join(save_path, "libsize_stats.json"), "w") as f:
            json.dump(libsize_stats, f, indent=2)
        print(f"  libsize stats: mean={libsize_stats['mean']:.4f} std={libsize_stats['std']:.4f} "
              f"(n_cell={libsize_stats['n_cell']}, n_nuclei={libsize_stats['n_nuclei']}); "
              f"saved to libsize_stats.json")
        if libsize_target_mode == "ct_median_cell":
            libsize_targets = libsize_stats["ct_median_cell"]
        elif libsize_target_mode != "source":
            raise ValueError(f"--libsize_target {libsize_target_mode} is not valid during training "
                             f"(use 'ct_median_cell' or 'source'; 'quantile_map' is inference-only)")

    # Library covariate: global vocabulary of training libraries + per-loader code remaps
    lib_remaps = {"cell": None, "nuclei": None, "cell_valid": None, "nuclei_valid": None}
    if library_cond:
        library_vocab = build_library_vocab(cell_train_loaders, nuclei_train_loaders, common_celltypes)
        n_lib = len(library_vocab["names"])
        if n_lib == 0:
            raise RuntimeError("--library_cond True but the loaders carry no meta_batch_id")
        if n_lib > model.library_capacity:
            raise ValueError(f"{n_lib} training libraries exceed --library_capacity {model.library_capacity}")
        with open(os.path.join(save_path, "library_vocab.json"), "w") as f:
            json.dump(library_vocab, f, indent=2)
        lib_remaps["cell"] = build_library_remaps(cell_train_loaders, common_celltypes, library_vocab)
        lib_remaps["nuclei"] = build_library_remaps(nuclei_train_loaders, common_celltypes, library_vocab)
        if run_validation:
            lib_remaps["cell_valid"] = build_library_remaps(cell_valid_loaders, valid_celltypes, library_vocab)
            lib_remaps["nuclei_valid"] = build_library_remaps(nuclei_valid_loaders, valid_celltypes, library_vocab)
        n_cell_lib = sum(1 for n in library_vocab["names"] if library_vocab["modality"][n] == "cell")
        print(f"  library vocab: {n_lib} libraries ({n_cell_lib} cell, {n_lib - n_cell_lib} nuclei) -> "
              f"library_vocab.json; ids 1..{n_lib}, 0 = UNKNOWN")

    # Pre-compute CT mean tensors on GPU for residual MSE loss
    use_residual = getattr(args, "residual_weight", 0.0) > 0
    ct_mean_tensors = {"cell": {}, "nuclei": {}}
    if use_residual:
        for ct in common_celltypes:
            ct_mean_tensors["cell"][ct] = torch.from_numpy(
                raw_means["cell"][ct]
            ).float().to(device)
            ct_mean_tensors["nuclei"][ct] = torch.from_numpy(
                raw_means["nuclei"][ct]
            ).float().to(device)
        print(f"  Residual MSE enabled: pre-computed CT means for {len(common_celltypes)} cell types")

    save_epoch_dump = getattr(args, "stage3_save_epoch_dump", False)
    dump_every = max(1, int(getattr(args, "stage3_dump_every", 50)))
    eval_every = max(1, int(getattr(args, "stage3_eval_every", 50)))

    # Load raw reference data for full eval
    skip_full_eval = getattr(args, "stage3_skip_full_eval", False)
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
        use_log1p = getattr(args, "raw_data_normalization", "raw") == "log1p"
        raw_cell_mean, raw_nuclei_mean, raw_common_genes = _prepare_raw_means_for_correlation(
            raw_cell_adata, raw_nuclei_adata, use_log1p=use_log1p,
        )

    # ====================================================================
    # Training loop
    # ====================================================================
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
            "residual_weight": _update_stage3_weight(args, epoch, "residual"),
        }
        args.alpha = args.alignment_weight
        use_cross_recons = args.cross_recons_weight > 0
        cross_recons_str = f", cross_recons={args.cross_recons_weight}" if use_cross_recons else ""
        residual_str = f", residual={args.residual_weight}" if use_residual else ""
        print(
            f"[Epoch {epoch}] Cond weights: "
            f"nuclei_recons={args.nuclei_recons_weight}, nuclei_jsd={args.nuclei_jsd_weight}, "
            f"cell_recons={args.cell_recons_weight}, cell_jsd={args.cell_jsd_weight}, "
            f"alignment={args.alignment_weight}{cross_recons_str}{residual_str}"
        )

        total_loss = 0.0
        total_nuclei_recons_loss = 0.0
        total_nuclei_jsd = 0.0
        total_cell_recons_loss = 0.0
        total_cell_jsd = 0.0
        total_alignment_jsd = 0.0
        total_cross_recons_loss = 0.0
        total_residual_loss = 0.0
        num_data = 0
        batch_loss_list = []
        epoch_accumulator = init_epoch_accumulator(common_celltypes)
        is_eval_epoch = (epoch % eval_every == 0 or epoch == args.num_epochs - 1)
        is_checkpoint_epoch = (epoch % 10 == 0 or epoch == args.num_epochs - 1)
        is_dump_epoch = (epoch % dump_every == 0 or epoch == args.num_epochs - 1)
        build_dump = is_eval_epoch or (save_epoch_dump and is_dump_epoch)
        epoch_save_dict = {} if build_dump else None

        model.train()
        for celltype in common_celltypes:
            nuclei_loader = nuclei_train_loaders[celltype]
            cell_batches = list(cell_train_loaders[celltype])
            if not cell_batches:
                continue

            if build_dump:
                epoch_save_dict[celltype] = {}

            for batch_idx, nuclei_batch in enumerate(nuclei_loader):
                if batch_idx >= len(cell_batches):
                    break

                cell_batch_data = cell_batches[batch_idx]
                nuclei_data = nuclei_batch["data"].to(device)
                cell_data = cell_batch_data["data"].to(device)
                verify_gpu_usage(model, nuclei_data, f"Batch {batch_idx}")

                batch_size = min(nuclei_data.size(0), cell_data.size(0))
                nuclei_data = nuclei_data[:batch_size]
                cell_data = cell_data[:batch_size]
                cell_lib = _library_ids_for_batch(lib_remaps["cell"], celltype, cell_batch_data, batch_size, device)
                nuclei_lib = _library_ids_for_batch(lib_remaps["nuclei"], celltype, nuclei_batch, batch_size, device)

                # Forward with modality IDs (and library ids when library_cond)
                cell_encoded, cell_decoded, cell_mu, cell_log_var = model(
                    cell_data, modality_id=CELL_ID, library_id=cell_lib
                )
                nuclei_encoded, nuclei_decoded, nuclei_mu, nuclei_log_var = model(
                    nuclei_data, modality_id=NUCLEI_ID, library_id=nuclei_lib
                )

                # Per-modality losses
                _, cell_recons_loss, cell_jsd_val = model.loss_function(
                    cell_decoded, cell_data, cell_mu, cell_log_var, args
                )[:3]
                _, nuclei_recons_loss, nuclei_jsd_val = model.loss_function(
                    nuclei_decoded, nuclei_data, nuclei_mu, nuclei_log_var, args
                )[:3]

                # Alignment loss (latent space)
                alignment_jsd = compute_mc_jsd_from_two_latents(
                    cell_mu, cell_log_var, nuclei_mu, nuclei_log_var
                )

                # Cross-recons: encode nuclei, decode with cell embedding, compare to raw cell
                cross_recons_loss_val = torch.tensor(0.0, device=device)
                if use_cross_recons:
                    if isinstance(model, VAE_NB_Cond):
                        target_library = cell_data.sum(dim=1, keepdim=True).clamp(min=1.0)
                        cross_decoded = model.decode_cross_train(
                            nuclei_mu, target_modality_id=CELL_ID,
                            target_library_size=target_library
                        )
                        # NB loss against raw cell data
                        eps = 1e-8
                        theta = model._nb_params["theta"]
                        log_theta_mu = torch.log(theta + cross_decoded + eps)
                        nb_log_prob = (
                            torch.lgamma(cell_data + theta)
                            - torch.lgamma(theta)
                            - torch.lgamma(cell_data + 1)
                            + theta * (torch.log(theta + eps) - log_theta_mu)
                            + cell_data * (torch.log(cross_decoded + eps) - log_theta_mu)
                        )
                        cross_recons_loss_val = -torch.mean(nb_log_prob)
                    else:
                        # MSE: decode nuclei mu with cell embedding, compare to raw cell (log1p)
                        _xr_kwargs = {}
                        if libsize_cond:
                            _xr_kwargs["target_log_libsize"] = VAE_JSD_Cond.log_library_size(cell_data)
                        if cell_lib is not None:
                            _xr_kwargs["target_library_id"] = cell_lib
                        cross_decoded = model.decode_cross_train(
                            nuclei_mu, target_modality_id=CELL_ID, **_xr_kwargs
                        )
                        cross_recons_loss_val = torch.nn.functional.mse_loss(
                            cross_decoded, cell_data
                        )

                # Residual MSE: per-cell deviation reconstruction
                residual_loss_val = torch.tensor(0.0, device=device)
                if use_residual and args.residual_weight > 0:
                    cell_ct_mean = ct_mean_tensors["cell"][celltype]
                    nuclei_ct_mean = ct_mean_tensors["nuclei"][celltype]
                    cell_residual_loss = F.mse_loss(
                        cell_decoded - cell_ct_mean, cell_data - cell_ct_mean
                    )
                    nuclei_residual_loss = F.mse_loss(
                        nuclei_decoded - nuclei_ct_mean, nuclei_data - nuclei_ct_mean
                    )
                    residual_loss_val = cell_residual_loss + nuclei_residual_loss

                # Combined loss — optionally normalize recon and JSD to ~[0,1]
                _recon_norm = getattr(args, "recon_loss_norm", 0.0)
                _do_norm = getattr(args, "normalize_losses", False) and _recon_norm > 0
                _LN2 = 0.6931471805599453  # np.log(2)
                if _do_norm:
                    loss = (
                        args.cell_recons_weight * cell_recons_loss / _recon_norm
                        + args.cell_jsd_weight * cell_jsd_val / _LN2
                        + args.nuclei_recons_weight * nuclei_recons_loss / _recon_norm
                        + args.nuclei_jsd_weight * nuclei_jsd_val / _LN2
                        + args.alignment_weight * alignment_jsd / _LN2
                        + args.cross_recons_weight * cross_recons_loss_val / _recon_norm
                        + args.residual_weight * residual_loss_val / _recon_norm
                    )
                else:
                    loss = (
                        args.cell_recons_weight * cell_recons_loss
                        + args.cell_jsd_weight * cell_jsd_val
                        + args.nuclei_recons_weight * nuclei_recons_loss
                        + args.nuclei_jsd_weight * nuclei_jsd_val
                        + args.alignment_weight * alignment_jsd
                        + args.cross_recons_weight * cross_recons_loss_val
                        + args.residual_weight * residual_loss_val
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Accumulate stats
                accumulate_epoch_batch(epoch_accumulator, celltype, "cell", cell_decoded)
                accumulate_epoch_batch(epoch_accumulator, celltype, "nuclei", nuclei_decoded)

                total_loss += loss.item() * batch_size
                total_nuclei_recons_loss += nuclei_recons_loss.item() * batch_size
                total_nuclei_jsd += nuclei_jsd_val.item() * batch_size
                total_cell_recons_loss += cell_recons_loss.item() * batch_size
                total_cell_jsd += cell_jsd_val.item() * batch_size
                total_alignment_jsd += alignment_jsd.item() * batch_size
                total_cross_recons_loss += cross_recons_loss_val.item() * batch_size
                total_residual_loss += residual_loss_val.item() * batch_size
                num_data += batch_size

                # ---- Build dump entry ----
                if build_dump:
                    # Metadata from nuclei (target) loader
                    original_cell_types = [
                        nuclei_loader.dataset.get_original_value("cell_type", code)
                        for code in nuclei_batch["cell_type"].cpu().numpy()[:batch_size]
                    ]
                    nuclei_batch_ids = []
                    if nuclei_loader.dataset.has_batch_id:
                        nuclei_batch_ids = [
                            nuclei_loader.dataset.get_original_value("batch_id", code)
                            for code in nuclei_batch["batch_id"].cpu().numpy()[:batch_size]
                        ]
                    cell_loader = cell_train_loaders[celltype]
                    cell_batch_ids = []
                    if cell_loader.dataset.has_batch_id:
                        cell_batch_ids = [
                            cell_loader.dataset.get_original_value("batch_id", code)
                            for code in cell_batch_data["batch_id"].cpu().numpy()[:batch_size]
                        ]
                    nuclei_sample_ids = list(nuclei_batch["sample_id"][:batch_size]) if "sample_id" in nuclei_batch else []
                    cell_sample_ids = list(cell_batch_data["sample_id"][:batch_size]) if "sample_id" in cell_batch_data else []

                    # Cross-decode for dump (nuclei→cell embedding)
                    cross_decoded_list = _build_cross_decode_dump_entry(
                        model, nuclei_data, device,
                        **_cross_decode_kwargs(model, libsize_targets, celltype),
                        **({"source_library_id": nuclei_lib} if nuclei_lib is not None else {}))

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
                        "cross_decoded": cross_decoded_list,
                        "loss": float(loss.item()),
                        "nuclei_recons_loss": float(nuclei_recons_loss.item()),
                        "nuclei_JSD_div": float(nuclei_jsd_val.item()),
                        "cell_recons_loss": float(cell_recons_loss.item()),
                        "cell_JSD_div": float(cell_jsd_val.item()),
                        "alignment_JSD_loss": float(alignment_jsd.item()),
                        "cross_recons_loss": float(cross_recons_loss_val.item()),
                        "residual_loss": float(residual_loss_val.item()),
                    }
                    epoch_save_dict[celltype][batch_idx] = dump_entry

                batch_loss_list.append([
                    celltype, batch_idx,
                    float(loss.item()),
                    float(nuclei_recons_loss.item()),
                    float(nuclei_jsd_val.item()),
                    float(cell_recons_loss.item()),
                    float(cell_jsd_val.item()),
                    float(alignment_jsd.item()),
                    float(cross_recons_loss_val.item()),
                    float(residual_loss_val.item()),
                    batch_size,
                ])

        if num_data == 0:
            raise RuntimeError("Cond stage 3 found no paired training batches across modalities")

        epoch_loss = total_loss / num_data
        epoch_nuclei_recons_loss = total_nuclei_recons_loss / num_data
        epoch_nuclei_jsd = total_nuclei_jsd / num_data
        epoch_cell_recons_loss = total_cell_recons_loss / num_data
        epoch_cell_jsd = total_cell_jsd / num_data
        epoch_alignment_jsd = total_alignment_jsd / num_data
        epoch_cross_recons_loss = total_cross_recons_loss / num_data
        epoch_residual_loss = total_residual_loss / num_data

        # Auto-compute recon loss normalization constant from epoch 0
        if getattr(args, "normalize_losses", False) and epoch == 0 and getattr(args, "recon_loss_norm", 0.0) == 0.0:
            args.recon_loss_norm = epoch_cell_recons_loss
            print(f"[normalize_losses] Set recon_loss_norm = {args.recon_loss_norm:.6f} from epoch 0 cell recon")

        loss_list.append([
            epoch, epoch_loss,
            epoch_nuclei_recons_loss, epoch_nuclei_jsd,
            epoch_cell_recons_loss, epoch_cell_jsd,
            epoch_alignment_jsd, epoch_cross_recons_loss,
            epoch_residual_loss,
        ])

        # ---- Checkpoint ----
        if is_checkpoint_epoch:
            os.makedirs(os.path.join(save_path, "models"), exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_path, "models", "model.pt"))
            _write_stage3_batch_losses(save_path, epoch, batch_loss_list)
            _write_run_state(save_path, epoch, args)

        # ---- Eval epoch ----
        if is_eval_epoch:
            # 1. Compact inline eval (decoder scale, shift, gene corr)
            run_stage3_inline_evaluation(save_path, epoch, epoch_accumulator, raw_means)

            # 2. Write epoch dump
            dump_path = _write_epoch_dump(save_path, epoch, epoch_save_dict)

            # 3. Conditioned decoder UMAPs (latent, decoded, cross-decoded, cell_vs_cross)
            try:
                plot_cond_umaps(save_path, "train", epoch, file_path=dump_path)
            except Exception as e:
                print(f"  Cond UMAP (train) failed: {e}")

            # 3b. Inline correlation (uses existing 4-source infrastructure)
            run_stage3_inline_umap_correlation(
                save_path, epoch, dump_path, epoch_accumulator, raw_means
            )

            # 4. Cross-decode evaluation (shift, gene corr, 5x5 correlation)
            eval_dir = os.path.join(save_path, "inline_eval")
            cross_means = _run_cross_decode_eval(
                model, nuclei_train_loaders, common_celltypes, raw_means,
                epoch, eval_dir, device, libsize_targets=libsize_targets,
                library_remaps=lib_remaps["nuclei"],
            )
            decoded_means = {
                "cell": _compute_decoded_means(epoch_accumulator, "cell"),
                "nuclei": _compute_decoded_means(epoch_accumulator, "nuclei"),
            }
            if cross_means:
                _run_cross_decode_correlation(
                    cross_means, decoded_means, raw_means,
                    epoch, eval_dir,
                )

            # 5. Full evaluation stack (heatmaps, 4x4 corr, scIB) if enabled
            if not skip_full_eval:
                run_stage3_full_evaluation(
                    save_path, epoch, dump_path,
                    raw_cell_adata, raw_nuclei_adata,
                    common_cell_types_set,
                    raw_cell_mean, raw_nuclei_mean, raw_common_genes,
                    use_log1p=use_log1p,
                )

            # 6. Cleanup dump if not retaining
            if not save_epoch_dump:
                try:
                    os.remove(dump_path)
                    print(f"  Epoch dump deleted: {dump_path}")
                except OSError:
                    pass

        elif save_epoch_dump and is_dump_epoch and build_dump:
            _save_stage3_epoch_outputs(save_path, epoch, epoch_save_dict, batch_loss_list)

        # ---- Validation ----
        valid_losses = None
        if run_validation:
            valid_accumulator = init_epoch_accumulator(valid_celltypes) if is_eval_epoch else None
            valid_losses, valid_save_dict = _run_cond_validation_epoch(
                model, nuclei_valid_loaders, cell_valid_loaders,
                valid_celltypes, args,
                epoch_accumulator=valid_accumulator,
                build_dump=is_eval_epoch,
                ct_mean_tensors=ct_mean_tensors if use_residual else None,
                library_remaps=(lib_remaps["nuclei_valid"], lib_remaps["cell_valid"]),
            )

            if valid_losses is not None:
                print(
                    f"  [Valid] loss={valid_losses['loss']:.6f}, "
                    f"nuc_recons={valid_losses['nuclei_recons_loss']:.6f}, "
                    f"cell_recons={valid_losses['cell_recons_loss']:.6f}, "
                    f"align_JSD={valid_losses['alignment_JSD_loss']:.6f}"
                )

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
                        valid_losses.get("cross_recons_loss", 0.0),
                    ])) + "\n")

            if is_eval_epoch and valid_accumulator is not None and valid_losses is not None:
                valid_eval_dir = os.path.join(save_path, "valid_inline_eval")
                os.makedirs(valid_eval_dir, exist_ok=True)

                # Inline eval on valid set
                run_stage3_inline_evaluation(
                    save_path, epoch, valid_accumulator, raw_means,
                    output_subdir="valid_inline_eval",
                )

                # Valid dump, UMAPs, correlation
                if valid_save_dict:
                    valid_dump_path = _write_valid_epoch_dump(save_path, epoch, valid_save_dict)

                    # Conditioned decoder UMAPs for valid set
                    try:
                        plot_cond_umaps(save_path, "valid", epoch, file_path=valid_dump_path)
                    except Exception as e:
                        print(f"  Cond UMAP (valid) failed: {e}")

                    run_stage3_inline_umap_correlation(
                        save_path, epoch, valid_dump_path, valid_accumulator, raw_means,
                        output_subdir="valid_inline_eval",
                        umap_data_split="valid",
                    )
                    try:
                        os.remove(valid_dump_path)
                        print(f"  Valid epoch dump deleted: {valid_dump_path}")
                    except OSError:
                        pass

                # Valid cross-decode eval (shift, gene corr, 5x5 correlation)
                valid_nuclei_loaders = nuclei_valid_loaders
                valid_cross_means = _run_cross_decode_eval(
                    model, valid_nuclei_loaders, valid_celltypes, raw_means,
                    epoch, valid_eval_dir, device, libsize_targets=libsize_targets,
                    library_remaps=lib_remaps["nuclei_valid"],
                )
                valid_decoded_means = {
                    "cell": _compute_decoded_means(valid_accumulator, "cell"),
                    "nuclei": _compute_decoded_means(valid_accumulator, "nuclei"),
                }
                if valid_cross_means:
                    _run_cross_decode_correlation(
                        valid_cross_means, valid_decoded_means, raw_means,
                        epoch, valid_eval_dir,
                    )

        # ---- Logging ----
        runtime_str = _runtime_str(start_time)
        cross_recons_log = f", cross_recons={epoch_cross_recons_loss:.6f}" if use_cross_recons else ""
        residual_log = f", residual={epoch_residual_loss:.6f}" if use_residual else ""
        print(
            f"Epoch {epoch+1}/{args.num_epochs} completed. "
            f"loss={epoch_loss:.6f}, nuclei_recons={epoch_nuclei_recons_loss:.6f}, "
            f"nuclei_JSD={epoch_nuclei_jsd:.6f}, cell_recons={epoch_cell_recons_loss:.6f}, "
            f"cell_JSD={epoch_cell_jsd:.6f}, alignment_JSD={epoch_alignment_jsd:.6f}"
            f"{cross_recons_log}{residual_log}"
        )

        os.makedirs(os.path.join(save_path, "train_epoch_loss"), exist_ok=True)
        loss_file = os.path.join(save_path, "train_epoch_loss", "loss.txt")
        file_exists = os.path.isfile(loss_file)
        with open(loss_file, "a" if file_exists else "w") as f:
            if not file_exists:
                f.write(
                    "epoch,loss,nuclei_recons_loss,nuclei_JSD_div,"
                    "cell_recons_loss,cell_JSD_div,alignment_JSD_loss,"
                    "cross_recons_loss,residual_loss,"
                    "nuclei_recons_weight,nuclei_jsd_weight,"
                    "cell_recons_weight,cell_jsd_weight,alignment_weight,"
                    "cross_recons_weight,residual_weight,runtime\n"
                )
            f.write(",".join(map(str, [
                epoch,
                epoch_loss,
                epoch_nuclei_recons_loss,
                epoch_nuclei_jsd,
                epoch_cell_recons_loss,
                epoch_cell_jsd,
                epoch_alignment_jsd,
                epoch_cross_recons_loss,
                epoch_residual_loss,
                epoch_weight_values["nuclei_recons_weight"],
                epoch_weight_values["nuclei_jsd_weight"],
                epoch_weight_values["cell_recons_weight"],
                epoch_weight_values["cell_jsd_weight"],
                epoch_weight_values["alignment_weight"],
                epoch_weight_values["cross_recons_weight"],
                epoch_weight_values["residual_weight"],
                runtime_str,
            ])) + "\n")

    # ---- Loss curves ----
    try:
        from .plot_loss import plot_loss_curve_stage3_v2
        plot_loss_curve_stage3_v2(save_path)
    except Exception as e:
        print(f"  Loss curve plotting failed: {e}")

    print("Conditioned stage 3 training complete.")


# ============================================================================
# Validation epoch (no gradient)
# ============================================================================


def _run_cond_validation_epoch(
    model, nuclei_loaders, cell_loaders, common_celltypes, args,
    epoch_accumulator=None, build_dump=False, ct_mean_tensors=None, library_remaps=(None, None),
):
    """Run validation epoch with modality-conditioned model.

    library_remaps: (nuclei_remaps, cell_remaps) from build_library_remaps for library_cond models.
    """
    nuclei_remaps, cell_remaps = library_remaps
    total_loss = 0.0
    total_nuclei_recons = 0.0
    total_nuclei_jsd = 0.0
    total_cell_recons = 0.0
    total_cell_jsd = 0.0
    total_alignment_jsd = 0.0
    total_cross_recons = 0.0
    total_residual = 0.0
    num_data = 0
    epoch_save_dict = {} if build_dump else None
    use_cross_recons = getattr(args, "cross_recons_weight", 0.0) > 0
    use_residual_val = getattr(args, "residual_weight", 0.0) > 0

    from .utils import device

    model.eval()
    with torch.no_grad():
        for celltype in common_celltypes:
            nuclei_loader = nuclei_loaders[celltype]
            cell_batches = list(cell_loaders[celltype])
            if not cell_batches:
                continue

            if build_dump:
                epoch_save_dict[celltype] = {}

            for batch_idx, nuclei_batch in enumerate(nuclei_loader):
                if batch_idx >= len(cell_batches):
                    break

                cell_batch_data = cell_batches[batch_idx]
                nuclei_data = nuclei_batch["data"].to(device)
                cell_data = cell_batch_data["data"].to(device)

                batch_size = min(nuclei_data.size(0), cell_data.size(0))
                nuclei_data = nuclei_data[:batch_size]
                cell_data = cell_data[:batch_size]
                cell_lib = _library_ids_for_batch(cell_remaps, celltype, cell_batch_data, batch_size, device)
                nuclei_lib = _library_ids_for_batch(nuclei_remaps, celltype, nuclei_batch, batch_size, device)

                cell_encoded, cell_decoded, cell_mu, cell_log_var = model(
                    cell_data, modality_id=CELL_ID, library_id=cell_lib
                )
                nuclei_encoded, nuclei_decoded, nuclei_mu, nuclei_log_var = model(
                    nuclei_data, modality_id=NUCLEI_ID, library_id=nuclei_lib
                )

                _, cell_recons_loss, cell_jsd_val = model.loss_function(
                    cell_decoded, cell_data, cell_mu, cell_log_var, args
                )[:3]
                _, nuclei_recons_loss, nuclei_jsd_val = model.loss_function(
                    nuclei_decoded, nuclei_data, nuclei_mu, nuclei_log_var, args
                )[:3]
                alignment_jsd = compute_mc_jsd_from_two_latents(
                    cell_mu, cell_log_var, nuclei_mu, nuclei_log_var
                )

                cross_recons_loss_val = torch.tensor(0.0, device=device)
                if use_cross_recons:
                    _xr_kwargs = {}
                    if isinstance(model, VAE_JSD_Cond) and model.libsize_cond:
                        _xr_kwargs["target_log_libsize"] = VAE_JSD_Cond.log_library_size(cell_data)
                    if nuclei_lib is not None:
                        _xr_kwargs["source_library_id"] = nuclei_lib
                        _xr_kwargs["target_library_id"] = cell_lib
                    cross_decoded = model.decode_cross(
                        nuclei_data, source_modality_id=NUCLEI_ID,
                        target_modality_id=CELL_ID, **_xr_kwargs
                    )
                    if isinstance(model, VAE_NB_Cond):
                        eps = 1e-8
                        theta = model._nb_params["theta"]
                        target_library = cell_data.sum(dim=1, keepdim=True).clamp(min=1.0)
                        # Rescale cross-decoded to cell library size
                        source_library = nuclei_data.sum(dim=1, keepdim=True).clamp(min=1.0)
                        proportions = cross_decoded / source_library
                        cross_mu = target_library * proportions
                        log_theta_mu = torch.log(theta + cross_mu + eps)
                        nb_log_prob = (
                            torch.lgamma(cell_data + theta)
                            - torch.lgamma(theta)
                            - torch.lgamma(cell_data + 1)
                            + theta * (torch.log(theta + eps) - log_theta_mu)
                            + cell_data * (torch.log(cross_mu + eps) - log_theta_mu)
                        )
                        cross_recons_loss_val = -torch.mean(nb_log_prob)
                    else:
                        cross_recons_loss_val = torch.nn.functional.mse_loss(
                            cross_decoded, cell_data
                        )

                # Residual MSE in validation
                residual_loss_val = torch.tensor(0.0, device=device)
                if use_residual_val and ct_mean_tensors is not None and args.residual_weight > 0:
                    cell_ct_mean = ct_mean_tensors["cell"][celltype]
                    nuclei_ct_mean = ct_mean_tensors["nuclei"][celltype]
                    cell_residual_loss = F.mse_loss(
                        cell_decoded - cell_ct_mean, cell_data - cell_ct_mean
                    )
                    nuclei_residual_loss = F.mse_loss(
                        nuclei_decoded - nuclei_ct_mean, nuclei_data - nuclei_ct_mean
                    )
                    residual_loss_val = cell_residual_loss + nuclei_residual_loss

                _recon_norm = getattr(args, "recon_loss_norm", 0.0)
                _do_norm = getattr(args, "normalize_losses", False) and _recon_norm > 0
                _LN2 = 0.6931471805599453
                if _do_norm:
                    loss = (
                        args.cell_recons_weight * cell_recons_loss / _recon_norm
                        + args.cell_jsd_weight * cell_jsd_val / _LN2
                        + args.nuclei_recons_weight * nuclei_recons_loss / _recon_norm
                        + args.nuclei_jsd_weight * nuclei_jsd_val / _LN2
                        + args.alignment_weight * alignment_jsd / _LN2
                        + args.cross_recons_weight * cross_recons_loss_val / _recon_norm
                        + args.residual_weight * residual_loss_val / _recon_norm
                    )
                else:
                    loss = (
                        args.cell_recons_weight * cell_recons_loss
                        + args.cell_jsd_weight * cell_jsd_val
                        + args.nuclei_recons_weight * nuclei_recons_loss
                        + args.nuclei_jsd_weight * nuclei_jsd_val
                        + args.alignment_weight * alignment_jsd
                        + args.cross_recons_weight * cross_recons_loss_val
                        + args.residual_weight * residual_loss_val
                    )

                if epoch_accumulator is not None:
                    accumulate_epoch_batch(epoch_accumulator, celltype, "cell", cell_decoded)
                    accumulate_epoch_batch(epoch_accumulator, celltype, "nuclei", nuclei_decoded)

                total_loss += loss.item() * batch_size
                total_nuclei_recons += nuclei_recons_loss.item() * batch_size
                total_nuclei_jsd += nuclei_jsd_val.item() * batch_size
                total_cell_recons += cell_recons_loss.item() * batch_size
                total_cell_jsd += cell_jsd_val.item() * batch_size
                total_alignment_jsd += alignment_jsd.item() * batch_size
                total_cross_recons += cross_recons_loss_val.item() * batch_size
                total_residual += residual_loss_val.item() * batch_size
                num_data += batch_size

                if build_dump:
                    original_cell_types = [
                        nuclei_loader.dataset.get_original_value("cell_type", code)
                        for code in nuclei_batch["cell_type"].cpu().numpy()[:batch_size]
                    ]
                    nuclei_batch_ids = []
                    if nuclei_loader.dataset.has_batch_id:
                        nuclei_batch_ids = [
                            nuclei_loader.dataset.get_original_value("batch_id", code)
                            for code in nuclei_batch["batch_id"].cpu().numpy()[:batch_size]
                        ]
                    cell_loader = cell_loaders[celltype]
                    cell_batch_ids = []
                    if cell_loader.dataset.has_batch_id:
                        cell_batch_ids = [
                            cell_loader.dataset.get_original_value("batch_id", code)
                            for code in cell_batch_data["batch_id"].cpu().numpy()[:batch_size]
                        ]
                    nuclei_sample_ids = list(nuclei_batch["sample_id"][:batch_size]) if "sample_id" in nuclei_batch else []
                    cell_sample_ids = list(cell_batch_data["sample_id"][:batch_size]) if "sample_id" in cell_batch_data else []

                    # Cross-decode for dump
                    cross_decoded_list = _build_cross_decode_dump_entry(
                        model, nuclei_data, device,
                        **({"source_library_id": nuclei_lib} if nuclei_lib is not None else {}))

                    epoch_save_dict[celltype][batch_idx] = {
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
                        "cross_decoded": cross_decoded_list,
                        "loss": float(loss.item()),
                        "nuclei_recons_loss": float(nuclei_recons_loss.item()),
                        "nuclei_JSD_div": float(nuclei_jsd_val.item()),
                        "cell_recons_loss": float(cell_recons_loss.item()),
                        "cell_JSD_div": float(cell_jsd_val.item()),
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
        "residual_loss": total_residual / num_data,
    }
    return losses, epoch_save_dict
