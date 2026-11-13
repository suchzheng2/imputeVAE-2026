import gc
import json
import os
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse, stats

# ---------------------------------------------------------------------------
# Import analysis utilities via sys.path (analysis/ is not a package)
# ---------------------------------------------------------------------------
_ANALYSIS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis")
if _ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, _ANALYSIS_DIR)

from eval_utils import (
    load_common_filters, load_raw_data, load_both_decoded_stage3,
    compute_celltype_mean, stratified_subsample, stratified_subsample_adata,
    compute_target_counts, get_raw_data_for_correlation, print_memory_usage,
    RAW_CELL_PATH, RAW_NUCLEI_PATH, MAX_CELLS_PER_TYPE, SUBSAMPLE_FRACTION, RANDOM_SEED,
)
from eval_heatmaps import (
    plot_expression_heatmap, plot_difference_heatmap,
)
from eval_correlation import (
    run_correlation_analysis, compute_shift_scores, plot_shift_scores,
    plot_shift_over_epochs,
)

# ============================================================================
# Compact inline evaluation (existing, kept unchanged)
# ============================================================================


def compute_loader_celltype_means(loaders, celltypes):
    means = {}
    for celltype in celltypes:
        dataset = loaders[celltype].dataset
        data = dataset.data.cpu().numpy().astype(np.float32)
        means[celltype] = data.mean(axis=0)
    return means


def init_epoch_accumulator(celltypes):
    def _empty_stats():
        return {
            "sum": 0.0,
            "sumsq": 0.0,
            "count": 0,
            "min": None,
            "max": None,
        }

    return {
        "overall": {"cell": _empty_stats(), "nuclei": _empty_stats()},
        "per_celltype": {
            "cell": {celltype: _empty_stats() for celltype in celltypes},
            "nuclei": {celltype: _empty_stats() for celltype in celltypes},
        },
        "decoded_sum": {
            "cell": {celltype: None for celltype in celltypes},
            "nuclei": {celltype: None for celltype in celltypes},
        },
        "decoded_count": {
            "cell": {celltype: 0 for celltype in celltypes},
            "nuclei": {celltype: 0 for celltype in celltypes},
        },
    }


def _update_stats(stats, values):
    arr = np.asarray(values, dtype=np.float32)
    stats["sum"] += float(arr.sum())
    stats["sumsq"] += float(np.square(arr, dtype=np.float32).sum())
    stats["count"] += int(arr.size)
    arr_min = float(arr.min())
    arr_max = float(arr.max())
    stats["min"] = arr_min if stats["min"] is None else min(stats["min"], arr_min)
    stats["max"] = arr_max if stats["max"] is None else max(stats["max"], arr_max)


def accumulate_epoch_batch(epoch_accumulator, celltype, modality, decoded_batch):
    decoded_np = decoded_batch.detach().cpu().numpy().astype(np.float32)
    _update_stats(epoch_accumulator["overall"][modality], decoded_np)
    _update_stats(epoch_accumulator["per_celltype"][modality][celltype], decoded_np)

    batch_sum = decoded_np.sum(axis=0)
    if epoch_accumulator["decoded_sum"][modality][celltype] is None:
        epoch_accumulator["decoded_sum"][modality][celltype] = batch_sum
    else:
        epoch_accumulator["decoded_sum"][modality][celltype] += batch_sum
    epoch_accumulator["decoded_count"][modality][celltype] += decoded_np.shape[0]


def _stats_to_row(epoch, modality, stats):
    if stats["count"] == 0:
        return None
    mean = stats["sum"] / stats["count"]
    var = max(stats["sumsq"] / stats["count"] - mean ** 2, 0.0)
    return {
        "epoch": epoch,
        "modality": modality,
        "mean": mean,
        "std": float(np.sqrt(var)),
        "min": stats["min"],
        "max": stats["max"],
        "n_values": stats["count"],
    }


def _compute_decoded_means(epoch_accumulator, modality):
    means = {}
    for celltype, vec_sum in epoch_accumulator["decoded_sum"][modality].items():
        count = epoch_accumulator["decoded_count"][modality][celltype]
        if vec_sum is None or count == 0:
            continue
        means[celltype] = vec_sum / count
    return means


def _corr(a, b):
    if a is None or b is None:
        return np.nan
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _log_mae(a, b):
    """Mean absolute error in log-space: MAE(log(a+1), log(b+1))."""
    if a is None or b is None:
        return np.nan
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.mean(np.abs(np.log1p(a) - np.log1p(b))))


def _r_squared(a, b):
    """Per-gene R² (coefficient of determination) across cell types."""
    if a is None or b is None:
        return np.nan
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ss_res = np.sum((a - b) ** 2)
    ss_tot = np.sum((b - b.mean()) ** 2)
    if ss_tot == 0:
        return np.nan
    return float(1.0 - ss_res / ss_tot)


def _wasserstein_1d(a, b):
    """1D Wasserstein distance (Earth Mover's Distance) between two vectors.

    Treats the vectors as empirical distributions over cell types.
    """
    if a is None or b is None:
        return np.nan
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    # Normalize to distributions (handle zero-sum gracefully)
    a_sum = a.sum()
    b_sum = b.sum()
    if a_sum == 0 and b_sum == 0:
        return 0.0
    if a_sum > 0:
        a_norm = a / a_sum
    else:
        a_norm = np.ones_like(a) / len(a)
    if b_sum > 0:
        b_norm = b / b_sum
    else:
        b_norm = np.ones_like(b) / len(b)
    # Sort both and compute Wasserstein-1
    a_sorted = np.sort(a_norm)
    b_sorted = np.sort(b_norm)
    return float(np.mean(np.abs(np.cumsum(a_sorted) - np.cumsum(b_sorted))))


def _append_csv(row_df, csv_path):
    if os.path.isfile(csv_path):
        existing = pd.read_csv(csv_path)
        row_df = pd.concat([existing, row_df], ignore_index=True)
        row_df = row_df.drop_duplicates(subset=[c for c in row_df.columns if c != "runtime"], keep="last")
    row_df.to_csv(csv_path, index=False)


def _plot_scale_history(scale_csv, output_dir):
    df = pd.read_csv(scale_csv)
    fig, ax = plt.subplots(figsize=(10, 6))
    for modality in sorted(df["modality"].unique()):
        sub = df[df["modality"] == modality].sort_values("epoch")
        ax.plot(sub["epoch"], sub["mean"], marker="o", label=f"{modality} mean")
        ax.fill_between(sub["epoch"], sub["mean"] - sub["std"], sub["mean"] + sub["std"], alpha=0.15)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Decoded value")
    ax.set_title("Stage 3 Inline Decoder Scale")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "decoder_scale_over_time.png"), dpi=150)
    plt.close(fig)


def _plot_shift_history(shift_csv, output_dir, modality):
    df = pd.read_csv(shift_csv)
    sub = df[df["decoded_modality"] == modality].sort_values("epoch")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sub["epoch"], sub["mean_shift"], marker="o")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean shift")
    ax.set_title(f"Stage 3 Inline Shift: {modality}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{modality}_shift_over_time.png"), dpi=150)
    plt.close(fig)


def _plot_gene_corr_history(gene_csv, output_dir, modality):
    df = pd.read_csv(gene_csv)
    sub = df[df["decoded_modality"] == modality].sort_values("epoch")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sub["epoch"], sub["median_corr_vs_raw_cell"], marker="o", label="vs raw cell")
    ax.plot(sub["epoch"], sub["median_corr_vs_raw_nuclei"], marker="o", label="vs raw nuclei")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Median gene correlation")
    ax.set_title(f"Stage 3 Inline Gene Correlation: {modality}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{modality}_gene_corr_over_time.png"), dpi=150)
    plt.close(fig)


def _build_mean_dataframes(decoded_means, raw_means):
    """Convert decoded_means and raw_means dicts to DataFrames for correlation analysis.

    Both inputs: dict of modality -> celltype -> 1D numpy array.
    Returns: dict of source_name -> DataFrame (celltypes x genes).
    """
    mean_dict = {}
    for label, means in [("raw_cell", raw_means.get("cell", {})),
                         ("raw_nuclei", raw_means.get("nuclei", {})),
                         ("decoded_cell", decoded_means.get("cell", {})),
                         ("decoded_nuclei", decoded_means.get("nuclei", {}))]:
        if not means:
            continue
        df = pd.DataFrame.from_dict(means, orient="index")
        if not df.empty:
            mean_dict[label] = df
    return mean_dict


def run_stage3_inline_umap_correlation(save_path, epoch, dump_path,
                                        epoch_accumulator, raw_means,
                                        output_subdir="inline_eval",
                                        umap_data_split="train"):
    """Generate UMAP plots and correlation maps as part of inline eval.

    Called after the epoch dump is written so dump_path is available.
    """
    output_dir = os.path.join(save_path, output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    # --- UMAPs (encoded + decoded, 3-panel each) ---
    try:
        from imputeVAE.visualization import plot_umap_after_train_by_celltype_stage3
    except ImportError:
        from visualization import plot_umap_after_train_by_celltype_stage3

    for feature in ["encoded", "decoded"]:
        try:
            print(f"  Inline UMAP ({feature}) epoch {epoch} [{umap_data_split}]...")
            plot_umap_after_train_by_celltype_stage3(
                save_path, feature, umap_data_split, epoch, file_path=dump_path
            )
        except Exception as e:
            print(f"  Inline UMAP ({feature}) failed: {e}")

    # --- Correlation map (4-source: raw_cell, raw_nuclei, decoded_cell, decoded_nuclei) ---
    decoded_means = {
        "cell": _compute_decoded_means(epoch_accumulator, "cell"),
        "nuclei": _compute_decoded_means(epoch_accumulator, "nuclei"),
    }
    mean_dict = _build_mean_dataframes(decoded_means, raw_means)

    if len(mean_dict) >= 2:
        corr_dir = os.path.join(output_dir, f"correlation_epoch_{epoch}")
        try:
            print(f"  Inline correlation map epoch {epoch}...")
            per_ct_corr, avg_corr = run_correlation_analysis(
                mean_dict, corr_dir, prefix="inline_"
            )
            # Save average correlation summary to CSV for time-series tracking
            avg_row = {"epoch": epoch}
            for src1 in avg_corr.index:
                for src2 in avg_corr.columns:
                    if src1 < src2:
                        avg_row[f"{src1}_vs_{src2}"] = float(avg_corr.loc[src1, src2])
            corr_history_csv = os.path.join(output_dir, "correlation_history.csv")
            _append_csv(pd.DataFrame([avg_row]), corr_history_csv)
        except Exception as e:
            print(f"  Inline correlation map failed: {e}")


def run_stage3_inline_evaluation(save_path, epoch, epoch_accumulator, raw_means,
                                  output_subdir="inline_eval"):
    output_dir = os.path.join(save_path, output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    scale_rows = []
    for modality in ["cell", "nuclei"]:
        row = _stats_to_row(epoch, modality, epoch_accumulator["overall"][modality])
        if row is not None:
            scale_rows.append(row)
    if scale_rows:
        scale_csv = os.path.join(output_dir, "decoder_scale.csv")
        _append_csv(pd.DataFrame(scale_rows), scale_csv)
        _plot_scale_history(scale_csv, output_dir)

    decoded_means = {
        "cell": _compute_decoded_means(epoch_accumulator, "cell"),
        "nuclei": _compute_decoded_means(epoch_accumulator, "nuclei"),
    }

    shift_rows = []
    gene_rows = []
    for modality in ["cell", "nuclei"]:
        common_cts = sorted(
            set(decoded_means[modality].keys())
            & set(raw_means["cell"].keys())
            & set(raw_means["nuclei"].keys())
        )
        if not common_cts:
            continue

        per_ct_shift = []
        decoded_matrix = []
        raw_cell_matrix = []
        raw_nuclei_matrix = []
        for celltype in common_cts:
            dec = decoded_means[modality][celltype]
            raw_cell = raw_means["cell"][celltype]
            raw_nuclei = raw_means["nuclei"][celltype]
            per_ct_shift.append({
                "epoch": epoch,
                "decoded_modality": modality,
                "cell_type": celltype,
                "corr_to_raw_cell": _corr(dec, raw_cell),
                "corr_to_raw_nuclei": _corr(dec, raw_nuclei),
            })
            decoded_matrix.append(dec)
            raw_cell_matrix.append(raw_cell)
            raw_nuclei_matrix.append(raw_nuclei)

        per_ct_df = pd.DataFrame(per_ct_shift)
        per_ct_df["shift"] = per_ct_df["corr_to_raw_cell"] - per_ct_df["corr_to_raw_nuclei"]
        per_ct_path = os.path.join(output_dir, f"{modality}_shift_per_celltype_epoch_{epoch}.csv")
        per_ct_df.to_csv(per_ct_path, index=False)
        shift_rows.append({
            "epoch": epoch,
            "decoded_modality": modality,
            "mean_shift": float(per_ct_df["shift"].mean()),
        })

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
        gene_rows.append({
            "epoch": epoch,
            "decoded_modality": modality,
            "median_corr_vs_raw_cell": float(np.nanmedian(corr_vs_cell)),
            "median_corr_vs_raw_nuclei": float(np.nanmedian(corr_vs_nuclei)),
            "median_log_mae_vs_raw_cell": float(np.nanmedian(mae_vs_cell)),
            "median_log_mae_vs_raw_nuclei": float(np.nanmedian(mae_vs_nuclei)),
            "median_r2_vs_raw_cell": float(np.nanmedian(r2_vs_cell)),
            "median_r2_vs_raw_nuclei": float(np.nanmedian(r2_vs_nuclei)),
            "median_wasserstein_vs_raw_cell": float(np.nanmedian(wd_vs_cell)),
            "median_wasserstein_vs_raw_nuclei": float(np.nanmedian(wd_vs_nuclei)),
        })

    if shift_rows:
        shift_csv = os.path.join(output_dir, "shift_summary.csv")
        _append_csv(pd.DataFrame(shift_rows), shift_csv)
        for modality in sorted({row["decoded_modality"] for row in shift_rows}):
            _plot_shift_history(shift_csv, output_dir, modality)

    if gene_rows:
        gene_csv = os.path.join(output_dir, "gene_corr_summary.csv")
        _append_csv(pd.DataFrame(gene_rows), gene_csv)
        for modality in sorted({row["decoded_modality"] for row in gene_rows}):
            _plot_gene_corr_history(gene_csv, output_dir, modality)


# ============================================================================
# Full inline evaluation (NEW)
# ============================================================================


def load_raw_reference_data(args):
    """Load raw cell and nuclei h5ad data for evaluation. Call once at training start."""
    common_genes, common_cell_types, mitocarta_genes = load_common_filters()
    det_rate = float(getattr(args, 'detection_rate_threshold', 1.0))
    exclude_mito = getattr(args, 'exclude_mitocarta', True)
    protocol = getattr(args, 'protocol', None)
    batch = getattr(args, 'batch', None)

    print("\n=== Loading raw reference data for inline evaluation ===")
    raw_cell_adata = load_raw_data(
        RAW_CELL_PATH, common_genes, common_cell_types,
        mitocarta_genes, det_rate, exclude_mito,
        protocol=protocol, batch=batch,
    )
    print_memory_usage("After raw cell load")

    raw_nuclei_adata = load_raw_data(
        RAW_NUCLEI_PATH, common_genes, common_cell_types,
        mitocarta_genes, det_rate, exclude_mito,
        protocol=protocol, batch=batch,
    )
    print_memory_usage("After raw nuclei load")

    return raw_cell_adata, raw_nuclei_adata, common_genes, common_cell_types


def _prepare_raw_means_for_correlation(raw_cell_adata, raw_nuclei_adata, use_log1p=False):
    """Compute raw cell-type means from reference data (once, reused across epochs)."""
    raw_cell_sub = stratified_subsample_adata(raw_cell_adata, SUBSAMPLE_FRACTION, seed=RANDOM_SEED)
    raw_cell_X = get_raw_data_for_correlation(raw_cell_sub, use_log1p)
    raw_cell_gene_names = list(raw_cell_sub.var['gene_names'])
    raw_cell_mean = compute_celltype_mean(raw_cell_X, raw_cell_sub.obs['cell_type'].values, raw_cell_gene_names)
    del raw_cell_X, raw_cell_sub; gc.collect()

    raw_nuclei_sub = stratified_subsample_adata(raw_nuclei_adata, SUBSAMPLE_FRACTION, seed=RANDOM_SEED)
    raw_nuclei_X = get_raw_data_for_correlation(raw_nuclei_sub, use_log1p)
    raw_nuclei_gene_names = list(raw_nuclei_sub.var['gene_names'])
    raw_nuclei_mean = compute_celltype_mean(raw_nuclei_X, raw_nuclei_sub.obs['cell_type'].values, raw_nuclei_gene_names)
    del raw_nuclei_X, raw_nuclei_sub; gc.collect()

    raw_common_genes = sorted(set(raw_cell_gene_names) & set(raw_nuclei_gene_names))
    return raw_cell_mean, raw_nuclei_mean, raw_common_genes


def run_4source_heatmaps(dump_path, epoch, raw_cell_adata, raw_nuclei_adata,
                          common_cell_types, output_dir, use_log1p=False):
    """Generate 4-source heatmaps: raw_cell, raw_nuclei, decoded_cell, decoded_nuclei."""
    os.makedirs(output_dir, exist_ok=True)
    norm_label = "log1p" if use_log1p else "raw"

    cell_dec, cell_dec_types, nuclei_dec, nuclei_dec_types = load_both_decoded_stage3(
        dump_path, common_cell_types
    )

    raw_cell_ct = raw_cell_adata.obs['cell_type'].values
    raw_nuclei_ct = raw_nuclei_adata.obs['cell_type'].values
    tc = compute_target_counts(
        [cell_dec_types, nuclei_dec_types, raw_cell_ct, raw_nuclei_ct],
        MAX_CELLS_PER_TYPE,
    )
    print(f"  4-source heatmap target counts: {len(tc)} cell types, {sum(tc.values())} total")

    # Subsample decoded
    dec_cell_sub, dec_cell_ct = stratified_subsample(cell_dec, cell_dec_types,
                                                       seed=RANDOM_SEED, target_counts=tc)
    dec_nuclei_sub, dec_nuclei_ct = stratified_subsample(nuclei_dec, nuclei_dec_types,
                                                           seed=RANDOM_SEED, target_counts=tc)
    del cell_dec, nuclei_dec; gc.collect()

    # Subsample raw
    raw_cell_X = raw_cell_adata.X
    if sparse.issparse(raw_cell_X):
        raw_cell_X = raw_cell_X.toarray()
    rc_sub, rc_types = stratified_subsample(raw_cell_X, raw_cell_ct, seed=RANDOM_SEED, target_counts=tc)
    rc_ref = np.log1p(rc_sub) if use_log1p else rc_sub.astype(np.float32)

    raw_nuclei_X = raw_nuclei_adata.X
    if sparse.issparse(raw_nuclei_X):
        raw_nuclei_X = raw_nuclei_X.toarray()
    rn_sub, rn_types = stratified_subsample(raw_nuclei_X, raw_nuclei_ct, seed=RANDOM_SEED, target_counts=tc)
    rn_ref = np.log1p(rn_sub) if use_log1p else rn_sub.astype(np.float32)

    # Common color scale
    all_vals = np.concatenate([
        dec_cell_sub.flatten(), dec_nuclei_sub.flatten(),
        rc_ref.flatten(), rn_ref.flatten(),
    ])
    vmin = float(np.percentile(all_vals, 1))
    vmax = float(np.percentile(all_vals, 99))
    del all_vals

    # 4 expression heatmaps
    plot_expression_heatmap(dec_cell_sub, dec_cell_ct, f"Decoded Cell (epoch {epoch})",
                            os.path.join(output_dir, "decoded_cell_heatmap.png"), vmin, vmax)
    plot_expression_heatmap(dec_nuclei_sub, dec_nuclei_ct, f"Decoded Nuclei (epoch {epoch})",
                            os.path.join(output_dir, "decoded_nuclei_heatmap.png"), vmin, vmax)
    plot_expression_heatmap(rc_ref, rc_types, f"Raw Cell ({norm_label})",
                            os.path.join(output_dir, f"raw_cell_{norm_label}_heatmap.png"), vmin, vmax)
    plot_expression_heatmap(rn_ref, rn_types, f"Raw Nuclei ({norm_label})",
                            os.path.join(output_dir, f"raw_nuclei_{norm_label}_heatmap.png"), vmin, vmax)

    # 4 difference heatmaps (decoded_X vs raw_Y)
    pairs = [
        (dec_cell_sub, rc_ref, dec_cell_ct, "decoded_cell", f"raw_cell ({norm_label})"),
        (dec_cell_sub, rn_ref, dec_cell_ct, "decoded_cell", f"raw_nuclei ({norm_label})"),
        (dec_nuclei_sub, rc_ref, dec_nuclei_ct, "decoded_nuclei", f"raw_cell ({norm_label})"),
        (dec_nuclei_sub, rn_ref, dec_nuclei_ct, "decoded_nuclei", f"raw_nuclei ({norm_label})"),
    ]
    for dec, raw, ct, dec_label, raw_label in pairs:
        safe_raw = raw_label.replace(" ", "_").replace("(", "").replace(")", "")
        plot_difference_heatmap(
            dec, raw, ct,
            f"{dec_label} vs {raw_label}",
            os.path.join(output_dir, f"diff_{dec_label}_vs_{safe_raw}.png"),
            dec_label, raw_label,
        )

    del dec_cell_sub, dec_nuclei_sub, rc_sub, rn_sub, rc_ref, rn_ref
    gc.collect()
    print(f"  4-source heatmaps saved to {output_dir}")


def run_4source_correlation(dump_path, epoch, raw_cell_mean, raw_nuclei_mean,
                             raw_common_genes, common_cell_types, output_dir):
    """4x4 correlation analysis: raw_cell, raw_nuclei, decoded_cell, decoded_nuclei."""
    os.makedirs(output_dir, exist_ok=True)

    cell_dec, cell_dec_types, nuclei_dec, nuclei_dec_types = load_both_decoded_stage3(
        dump_path, common_cell_types
    )

    cell_dec_mean = compute_celltype_mean(cell_dec, cell_dec_types)
    nuclei_dec_mean = compute_celltype_mean(nuclei_dec, nuclei_dec_types)
    del cell_dec, nuclei_dec; gc.collect()

    n_raw = len(raw_common_genes)

    # Align decoded to raw gene order
    cell_dec_mean_c = cell_dec_mean.iloc[:, :min(cell_dec_mean.shape[1], n_raw)].copy()
    cell_dec_mean_c.columns = raw_common_genes[:cell_dec_mean_c.shape[1]]
    nuclei_dec_mean_c = nuclei_dec_mean.iloc[:, :min(nuclei_dec_mean.shape[1], n_raw)].copy()
    nuclei_dec_mean_c.columns = raw_common_genes[:nuclei_dec_mean_c.shape[1]]

    raw_cell_mean_c = raw_cell_mean[raw_common_genes] if hasattr(raw_cell_mean, '__getitem__') else raw_cell_mean
    raw_nuclei_mean_c = raw_nuclei_mean[raw_common_genes] if hasattr(raw_nuclei_mean, '__getitem__') else raw_nuclei_mean

    mean_dict = {
        'raw_cell': raw_cell_mean_c,
        'raw_nuclei': raw_nuclei_mean_c,
        'decoded_cell': cell_dec_mean_c,
        'decoded_nuclei': nuclei_dec_mean_c,
    }

    per_ct_corr, avg_corr = run_correlation_analysis(mean_dict, output_dir, prefix="4x4_")

    # Shift scores for both decoded modalities
    results = {}
    for decoded_label in ['decoded_cell', 'decoded_nuclei']:
        shift_df = compute_shift_scores(per_ct_corr, decoded_label)
        if not shift_df.empty:
            plot_shift_scores(shift_df, decoded_label, output_dir, prefix=f"4x4_{decoded_label}_")
            results[f'{decoded_label}_mean_shift'] = float(shift_df['shift'].mean())
        else:
            results[f'{decoded_label}_mean_shift'] = None

    del cell_dec_mean, nuclei_dec_mean, cell_dec_mean_c, nuclei_dec_mean_c
    gc.collect()

    print(f"  4x4 correlation saved to {output_dir}")
    return results


def run_inline_scib(save_path, epoch, output_dir):
    """Launch scIB stage3_full evaluation as subprocess in abc_env."""
    os.makedirs(output_dir, exist_ok=True)

    eval_scib_path = os.path.join(_ANALYSIS_DIR, 'eval_scib.py')

    # Find conda
    conda_path = None
    for candidate in ['conda', os.path.expanduser('~/miniconda3/bin/conda'),
                      os.path.expanduser('~/anaconda3/bin/conda')]:
        try:
            subprocess.run([candidate, '--version'], capture_output=True, timeout=10)
            conda_path = candidate
            break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    if conda_path is None:
        print("  Warning: conda not found, skipping inline scIB")
        return {'status': 'skipped', 'reason': 'conda not found'}

    # Check --no-banner support
    try:
        help_out = subprocess.run(
            [conda_path, 'run', '-h'], capture_output=True, text=True, timeout=10
        )
        supports_no_banner = '--no-banner' in f"{help_out.stdout}\n{help_out.stderr}"
    except Exception:
        supports_no_banner = False

    cmd = [conda_path, 'run', '-n', 'abc_env']
    if supports_no_banner:
        cmd.append('--no-banner')
    cmd.extend([
        'python', eval_scib_path,
        '--run_folder', save_path,
        '--epoch', str(epoch),
        '--output_dir', output_dir,
        '--mode', 'stage3_full',
    ])

    print(f"  Running inline scIB (stage3_full) for epoch {epoch}...")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if proc.returncode == 0:
            print(f"    scIB epoch {epoch}: success")
            results_path = os.path.join(output_dir, 'scib_results.json')
            if os.path.isfile(results_path):
                with open(results_path, 'r') as f:
                    return json.load(f)
            return {'status': 'success'}
        else:
            print(f"    scIB epoch {epoch}: failed (exit code {proc.returncode})")
            if proc.stderr:
                print(f"    stderr: {proc.stderr[:500]}")
            return {'status': 'failed', 'stderr': proc.stderr[:500]}
    except subprocess.TimeoutExpired:
        print(f"    scIB epoch {epoch}: timed out (60min)")
        return {'status': 'failed', 'reason': 'timeout'}
    except Exception as e:
        print(f"    scIB epoch {epoch}: error: {e}")
        return {'status': 'failed', 'reason': str(e)}


def run_stage3_full_evaluation(save_path, epoch, dump_path,
                                raw_cell_adata, raw_nuclei_adata,
                                common_cell_types, raw_cell_mean,
                                raw_nuclei_mean, raw_common_genes,
                                use_log1p=False):
    """Run the full evaluation stack on a stage-3 epoch dump.

    Called at every eval epoch during training.
    """
    full_eval_dir = os.path.join(save_path, "inline_eval", "full")
    os.makedirs(full_eval_dir, exist_ok=True)

    t_start = time.time()
    manifest = {'epoch': epoch, 'dump_path': dump_path}

    # 1. 4-source heatmaps
    print(f"\n--- [Epoch {epoch}] 4-source heatmaps ---")
    heatmap_dir = os.path.join(full_eval_dir, f"heatmaps_epoch_{epoch}")
    try:
        run_4source_heatmaps(
            dump_path, epoch, raw_cell_adata, raw_nuclei_adata,
            common_cell_types, heatmap_dir, use_log1p=use_log1p,
        )
        manifest['heatmaps'] = {'status': 'success', 'output_dir': heatmap_dir}
    except Exception as e:
        print(f"    Heatmaps failed: {e}")
        manifest['heatmaps'] = {'status': 'failed', 'error': str(e)}

    # 2. 4-source correlation
    print(f"\n--- [Epoch {epoch}] 4-source correlation ---")
    corr_dir = os.path.join(full_eval_dir, f"correlation_epoch_{epoch}")
    try:
        corr_results = run_4source_correlation(
            dump_path, epoch, raw_cell_mean, raw_nuclei_mean,
            raw_common_genes, common_cell_types, corr_dir,
        )
        manifest['correlation'] = {'status': 'success', **corr_results}
    except Exception as e:
        print(f"    Correlation failed: {e}")
        manifest['correlation'] = {'status': 'failed', 'error': str(e)}

    # 3. scIB via subprocess
    print(f"\n--- [Epoch {epoch}] scIB (subprocess) ---")
    scib_dir = os.path.join(full_eval_dir, f"scib_epoch_{epoch}")
    try:
        scib_results = run_inline_scib(save_path, epoch, scib_dir)
        manifest['scib'] = scib_results
    except Exception as e:
        print(f"    scIB failed: {e}")
        manifest['scib'] = {'status': 'failed', 'error': str(e)}

    elapsed = time.time() - t_start
    manifest['total_time_seconds'] = round(elapsed, 1)
    print(f"\n--- [Epoch {epoch}] Full inline evaluation completed in {elapsed:.1f}s ---")

    # Save manifest
    manifest_path = os.path.join(full_eval_dir, f"manifest_epoch_{epoch}.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)

    return manifest
