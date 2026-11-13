"""
Zero / non-zero partition analysis.

Treats the cell×gene matrix as a flat collection of entries, partitions by
whether the raw value is zero or non-zero, and compares raw vs decoded
statistics in each partition.

Four decoded sources:
  1. Cell reconstruction (encode cell → decode cell)
  2. Nuclei reconstruction (encode nuclei → decode nuclei)
  3. Cross-decoded (encode nuclei → decode as cell)
  4. Cell-type mean baseline (no model, just CT mean lookup)

Two cross-decoded comparison modes:
  - CT-matched: compare cross-decoded nuclei against raw cells of the same CT
  - CT-mean reference: compare cross-decoded nuclei against CT mean of raw cells

Groupings: global, by cell type, by gene sparsity bucket.

Usage:
    python -m analysis.zero_nonzero_analysis --model pct5v2_cond \
        --out notes/experiments/zero_nonzero_analysis/
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore", category=FutureWarning)

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_DIR)

from analysis.eval.registry import get_model
from analysis.eval.data import load_eval_data
from analysis.eval.inference import load_model, run_5source_inference, CELL_ID, NUCLEI_ID


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _partition_metrics(raw_flat, dec_flat, zero_mask):
    """Compute metrics for one partition (zero or nonzero entries).

    Parameters
    ----------
    raw_flat : 1-D array, raw values at these positions
    dec_flat : 1-D array, decoded values at these positions
    zero_mask : bool, True if this is the zero partition

    Returns
    -------
    dict of metrics
    """
    n = len(raw_flat)
    if n == 0:
        return {"count": 0}

    result = {
        "count": int(n),
        "dec_mean": float(np.mean(dec_flat)),
        "dec_std": float(np.std(dec_flat)),
        "dec_median": float(np.median(dec_flat)),
        "dec_q25": float(np.percentile(dec_flat, 25)),
        "dec_q75": float(np.percentile(dec_flat, 75)),
        "dec_min": float(np.min(dec_flat)),
        "dec_max": float(np.max(dec_flat)),
        "dec_frac_negative": float(np.mean(dec_flat < 0)),
        "dec_frac_near_zero": float(np.mean(np.abs(dec_flat) < 0.01)),
    }

    if not zero_mask:
        # Nonzero partition: raw has actual values
        result["raw_mean"] = float(np.mean(raw_flat))
        result["raw_std"] = float(np.std(raw_flat))
        result["raw_median"] = float(np.median(raw_flat))
        result["raw_q25"] = float(np.percentile(raw_flat, 25))
        result["raw_q75"] = float(np.percentile(raw_flat, 75))

        # Scale ratio
        raw_std = np.std(raw_flat)
        dec_std = np.std(dec_flat)
        result["std_ratio"] = float(dec_std / raw_std) if raw_std > 1e-10 else np.nan
        result["mean_ratio"] = float(np.mean(dec_flat) / np.mean(raw_flat)) if np.mean(raw_flat) > 1e-10 else np.nan

        # Pearson correlation
        if len(raw_flat) > 2 and np.std(raw_flat) > 1e-10 and np.std(dec_flat) > 1e-10:
            result["pearson_r"] = float(np.corrcoef(raw_flat, dec_flat)[0, 1])
        else:
            result["pearson_r"] = np.nan

    return result


def _separation_metrics(dec_at_zero, dec_at_nonzero):
    """How well can we separate zero vs nonzero from decoded values alone?"""
    if len(dec_at_zero) == 0 or len(dec_at_nonzero) == 0:
        return {}

    result = {}
    # Mean difference
    result["mean_gap"] = float(np.mean(dec_at_nonzero) - np.mean(dec_at_zero))

    # AUROC: can decoded value predict zero/nonzero?
    # Efficient approximation via Mann-Whitney U
    n0 = min(len(dec_at_zero), 50000)
    n1 = min(len(dec_at_nonzero), 50000)
    if n0 > 0 and n1 > 0:
        rng = np.random.RandomState(42)
        sample_0 = rng.choice(dec_at_zero, n0, replace=False) if n0 < len(dec_at_zero) else dec_at_zero
        sample_1 = rng.choice(dec_at_nonzero, n1, replace=False) if n1 < len(dec_at_nonzero) else dec_at_nonzero
        u_stat, _ = sp_stats.mannwhitneyu(sample_1, sample_0, alternative='greater')
        auroc = u_stat / (n0 * n1)
        result["auroc_nonzero_vs_zero"] = float(auroc)

    # Overlap: fraction of decoded-at-zero values that exceed the decoded-at-nonzero median
    nz_median = np.median(dec_at_nonzero)
    result["frac_zero_above_nonzero_median"] = float(np.mean(dec_at_zero > nz_median))

    return result


def analyze_pair(raw, decoded, label, ct_array, gene_sparsity, common_cts, gene_names):
    """Run full zero/nonzero analysis for one (raw, decoded) pair.

    Parameters
    ----------
    raw : ndarray [N, G]
    decoded : ndarray [N, G]
    label : str, e.g. "cell_recon"
    ct_array : ndarray [N] str, cell type per sample
    gene_sparsity : ndarray [G] float, fraction of zeros per gene in raw data
    common_cts : list of str
    gene_names : ndarray [G] str

    Returns
    -------
    dict of DataFrames: 'global', 'by_celltype', 'by_sparsity'
    """
    assert raw.shape == decoded.shape, f"Shape mismatch: {raw.shape} vs {decoded.shape}"

    zero_mask_full = (raw == 0)  # [N, G] bool

    results = {}

    # ---- Global ----
    raw_flat = raw.ravel()
    dec_flat = decoded.ravel()
    zmask_flat = zero_mask_full.ravel()

    global_row = {"group": "global", "label": label}
    global_row["total_entries"] = int(len(raw_flat))
    global_row["frac_zero"] = float(zmask_flat.mean())

    zero_metrics = _partition_metrics(raw_flat[zmask_flat], dec_flat[zmask_flat], zero_mask=True)
    nonzero_metrics = _partition_metrics(raw_flat[~zmask_flat], dec_flat[~zmask_flat], zero_mask=False)
    sep = _separation_metrics(dec_flat[zmask_flat], dec_flat[~zmask_flat])

    for k, v in zero_metrics.items():
        global_row[f"zero_{k}"] = v
    for k, v in nonzero_metrics.items():
        global_row[f"nonzero_{k}"] = v
    for k, v in sep.items():
        global_row[k] = v

    results['global'] = pd.DataFrame([global_row])

    # ---- By cell type ----
    ct_rows = []
    for ct in common_cts:
        mask = ct_array == ct
        if mask.sum() == 0:
            continue
        r = raw[mask]
        d = decoded[mask]
        zm = (r == 0)

        row = {"group": ct, "label": label}
        row["total_entries"] = int(r.size)
        row["n_cells"] = int(mask.sum())
        row["frac_zero"] = float(zm.mean())

        r_flat, d_flat, z_flat = r.ravel(), d.ravel(), zm.ravel()
        zm_z = _partition_metrics(r_flat[z_flat], d_flat[z_flat], zero_mask=True)
        zm_nz = _partition_metrics(r_flat[~z_flat], d_flat[~z_flat], zero_mask=False)
        s = _separation_metrics(d_flat[z_flat], d_flat[~z_flat])

        for k, v in zm_z.items():
            row[f"zero_{k}"] = v
        for k, v in zm_nz.items():
            row[f"nonzero_{k}"] = v
        for k, v in s.items():
            row[k] = v
        ct_rows.append(row)

    results['by_celltype'] = pd.DataFrame(ct_rows)

    # ---- By gene sparsity bucket ----
    sparsity_bins = [
        ("<30%", 0.0, 0.3),
        ("30-70%", 0.3, 0.7),
        ("70-90%", 0.7, 0.9),
        (">90%", 0.9, 1.01),
    ]
    sp_rows = []
    for bin_label, lo, hi in sparsity_bins:
        gene_mask = (gene_sparsity >= lo) & (gene_sparsity < hi)
        n_genes = gene_mask.sum()
        if n_genes == 0:
            continue

        r = raw[:, gene_mask]
        d = decoded[:, gene_mask]
        zm = (r == 0)

        row = {"group": bin_label, "label": label}
        row["total_entries"] = int(r.size)
        row["n_genes"] = int(n_genes)
        row["frac_zero"] = float(zm.mean())

        r_flat, d_flat, z_flat = r.ravel(), d.ravel(), zm.ravel()
        zm_z = _partition_metrics(r_flat[z_flat], d_flat[z_flat], zero_mask=True)
        zm_nz = _partition_metrics(r_flat[~z_flat], d_flat[~z_flat], zero_mask=False)
        s = _separation_metrics(d_flat[z_flat], d_flat[~z_flat])

        for k, v in zm_z.items():
            row[f"zero_{k}"] = v
        for k, v in zm_nz.items():
            row[f"nonzero_{k}"] = v
        for k, v in s.items():
            row[k] = v
        sp_rows.append(row)

    results['by_sparsity'] = pd.DataFrame(sp_rows)

    return results


def build_ct_mean_baseline(X, ct_array, common_cts):
    """Build cell-type mean 'decoded' output: each cell gets its CT mean profile.

    Parameters
    ----------
    X : ndarray [N, G]
    ct_array : ndarray [N] str
    common_cts : list of str

    Returns
    -------
    ndarray [N, G] — each row is the CT mean for that cell's type
    """
    baseline = np.zeros_like(X)
    for ct in common_cts:
        mask = ct_array == ct
        if mask.sum() > 0:
            baseline[mask] = X[mask].mean(axis=0)
    return baseline


def cross_decode_ct_matched_analysis(cross_decoded, ct_nuc, raw_cell, ct_cell,
                                      gene_sparsity, common_cts, gene_names):
    """Compare cross-decoded nuclei against raw cells matched by cell type.

    For each CT: pool cross-decoded nuclei and raw cells separately,
    then do zero/nonzero partition based on the raw cell side.
    """
    ct_rows = []
    # Also build global pools
    all_raw = []
    all_dec = []

    for ct in common_cts:
        nuc_mask = ct_nuc == ct
        cell_mask = ct_cell == ct
        if nuc_mask.sum() == 0 or cell_mask.sum() == 0:
            continue

        cd = cross_decoded[nuc_mask]  # [N_nuc_ct, G]
        rc = raw_cell[cell_mask]      # [N_cell_ct, G]

        # Truncate to min size for direct comparison
        n = min(cd.shape[0], rc.shape[0])
        cd_sub = cd[:n]
        rc_sub = rc[:n]

        all_raw.append(rc_sub)
        all_dec.append(cd_sub)

        zm = (rc_sub == 0)
        r_flat, d_flat, z_flat = rc_sub.ravel(), cd_sub.ravel(), zm.ravel()

        row = {"group": ct, "label": "cross_decoded_ct_matched"}
        row["total_entries"] = int(rc_sub.size)
        row["n_pairs"] = int(n)
        row["frac_zero"] = float(zm.mean())

        zm_z = _partition_metrics(r_flat[z_flat], d_flat[z_flat], zero_mask=True)
        zm_nz = _partition_metrics(r_flat[~z_flat], d_flat[~z_flat], zero_mask=False)
        s = _separation_metrics(d_flat[z_flat], d_flat[~z_flat])

        for k, v in zm_z.items():
            row[f"zero_{k}"] = v
        for k, v in zm_nz.items():
            row[f"nonzero_{k}"] = v
        for k, v in s.items():
            row[k] = v
        ct_rows.append(row)

    # Global across all CTs
    if all_raw:
        all_raw_np = np.concatenate(all_raw, axis=0)
        all_dec_np = np.concatenate(all_dec, axis=0)
        zm = (all_raw_np == 0)
        r_flat, d_flat, z_flat = all_raw_np.ravel(), all_dec_np.ravel(), zm.ravel()

        global_row = {"group": "global", "label": "cross_decoded_ct_matched"}
        global_row["total_entries"] = int(all_raw_np.size)
        global_row["frac_zero"] = float(zm.mean())

        zm_z = _partition_metrics(r_flat[z_flat], d_flat[z_flat], zero_mask=True)
        zm_nz = _partition_metrics(r_flat[~z_flat], d_flat[~z_flat], zero_mask=False)
        s = _separation_metrics(d_flat[z_flat], d_flat[~z_flat])

        for k, v in zm_z.items():
            global_row[f"zero_{k}"] = v
        for k, v in zm_nz.items():
            global_row[f"nonzero_{k}"] = v
        for k, v in s.items():
            global_row[k] = v
    else:
        global_row = {"group": "global", "label": "cross_decoded_ct_matched", "count": 0}

    return {
        'global': pd.DataFrame([global_row]),
        'by_celltype': pd.DataFrame(ct_rows),
    }


def cross_decode_ct_mean_analysis(cross_decoded, ct_nuc, raw_cell, ct_cell,
                                   gene_sparsity, common_cts, gene_names):
    """Compare each cross-decoded nuclei profile against its CT's raw cell mean.

    Zero/nonzero is defined on the raw cell CT mean (threshold: mean == 0 for
    genes where NO cell in that CT expressed it, otherwise nonzero).
    """
    ct_rows = []
    all_raw_mean = []
    all_dec = []

    for ct in common_cts:
        nuc_mask = ct_nuc == ct
        cell_mask = ct_cell == ct
        if nuc_mask.sum() == 0 or cell_mask.sum() == 0:
            continue

        cd = cross_decoded[nuc_mask]           # [N_nuc_ct, G]
        ct_mean = raw_cell[cell_mask].mean(axis=0)  # [G]

        # Expand ct_mean to match cd shape
        ct_mean_expanded = np.broadcast_to(ct_mean, cd.shape)  # [N_nuc_ct, G]

        all_raw_mean.append(ct_mean_expanded)
        all_dec.append(cd)

        # Zero partition: gene positions where ct_mean == 0
        # (no cells of this type expressed this gene at all)
        zm = (ct_mean_expanded == 0)
        r_flat = ct_mean_expanded.ravel()
        d_flat = cd.ravel()
        z_flat = zm.ravel()

        row = {"group": ct, "label": "cross_decoded_vs_ct_mean"}
        row["total_entries"] = int(cd.size)
        row["n_nuclei"] = int(nuc_mask.sum())
        row["frac_zero_in_ct_mean"] = float(zm.mean())

        zm_z = _partition_metrics(r_flat[z_flat], d_flat[z_flat], zero_mask=True)
        zm_nz = _partition_metrics(r_flat[~z_flat], d_flat[~z_flat], zero_mask=False)
        s = _separation_metrics(d_flat[z_flat], d_flat[~z_flat])

        for k, v in zm_z.items():
            row[f"zero_{k}"] = v
        for k, v in zm_nz.items():
            row[f"nonzero_{k}"] = v
        for k, v in s.items():
            row[k] = v
        ct_rows.append(row)

    # Global
    if all_raw_mean:
        all_raw_np = np.concatenate(all_raw_mean, axis=0)
        all_dec_np = np.concatenate(all_dec, axis=0)
        zm = (all_raw_np == 0)
        r_flat, d_flat, z_flat = all_raw_np.ravel(), all_dec_np.ravel(), zm.ravel()

        global_row = {"group": "global", "label": "cross_decoded_vs_ct_mean"}
        global_row["total_entries"] = int(all_raw_np.size)
        global_row["frac_zero_in_ct_mean"] = float(zm.mean())

        zm_z = _partition_metrics(r_flat[z_flat], d_flat[z_flat], zero_mask=True)
        zm_nz = _partition_metrics(r_flat[~z_flat], d_flat[~z_flat], zero_mask=False)
        s = _separation_metrics(d_flat[z_flat], d_flat[~z_flat])

        for k, v in zm_z.items():
            global_row[f"zero_{k}"] = v
        for k, v in zm_nz.items():
            global_row[f"nonzero_{k}"] = v
        for k, v in s.items():
            global_row[k] = v
    else:
        global_row = {"group": "global", "label": "cross_decoded_vs_ct_mean", "count": 0}

    return {
        'global': pd.DataFrame([global_row]),
        'by_celltype': pd.DataFrame(ct_rows),
    }


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Zero/nonzero partition analysis")
    parser.add_argument("--model", type=str, default="pct5v2_cond",
                        help="Model name from registry")
    parser.add_argument("--out", type=str,
                        default="notes/experiments/zero_nonzero_analysis/",
                        help="Output directory")
    parser.add_argument("--max-per-type", type=int, default=None,
                        help="Max cells per cell type (None = use all)")
    args = parser.parse_args()

    out_dir = os.path.join(_PROJECT_DIR, args.out, args.model)
    os.makedirs(out_dir, exist_ok=True)

    # Load model and data
    model_config = get_model(args.model)
    gene_list_path = model_config['gene_list']

    max_per_type = args.max_per_type if args.max_per_type else 99999
    data = load_eval_data(gene_list_path, max_per_type=max_per_type)
    model, device = load_model(model_config)

    print(f"\nModel: {args.model}")
    print(f"Cell: {data.X_cell.shape}, Nuclei: {data.X_nuc.shape}")
    print(f"Cell types: {len(data.common_cts)}")
    print(f"Genes: {data.n_genes}")

    # Run inference
    print("\nRunning inference...")
    sources = run_5source_inference(model, model_config,
                                     data.X_cell, data.X_nuc, device)

    # Gene sparsity (fraction of zeros in raw cell data per gene)
    gene_sparsity_cell = (data.X_cell == 0).mean(axis=0)
    gene_sparsity_nuc = (data.X_nuc == 0).mean(axis=0)

    # Build CT mean baselines
    ct_mean_cell = build_ct_mean_baseline(data.X_cell, data.ct_cell, data.common_cts)
    ct_mean_nuc = build_ct_mean_baseline(data.X_nuc, data.ct_nuc, data.common_cts)

    all_globals = []
    all_ct = []
    all_sp = []

    # ---- 1. Cell reconstruction ----
    print("\n--- Cell reconstruction ---")
    r = analyze_pair(data.X_cell, sources['decoded_cell'], "cell_recon",
                     data.ct_cell, gene_sparsity_cell, data.common_cts, data.gene_names)
    all_globals.append(r['global'])
    all_ct.append(r['by_celltype'])
    all_sp.append(r['by_sparsity'])

    # ---- 2. Nuclei reconstruction ----
    print("--- Nuclei reconstruction ---")
    r = analyze_pair(data.X_nuc, sources['decoded_nuclei'], "nuclei_recon",
                     data.ct_nuc, gene_sparsity_nuc, data.common_cts, data.gene_names)
    all_globals.append(r['global'])
    all_ct.append(r['by_celltype'])
    all_sp.append(r['by_sparsity'])

    # ---- 3. Cross-decoded (vs raw nuclei — same samples) ----
    print("--- Cross-decoded vs raw nuclei (same samples) ---")
    r = analyze_pair(data.X_nuc, sources['cross_decoded'], "cross_decoded_vs_raw_nuc",
                     data.ct_nuc, gene_sparsity_nuc, data.common_cts, data.gene_names)
    all_globals.append(r['global'])
    all_ct.append(r['by_celltype'])
    all_sp.append(r['by_sparsity'])

    # ---- 4. CT mean baseline for cells ----
    print("--- CT mean baseline (cell) ---")
    r = analyze_pair(data.X_cell, ct_mean_cell, "ct_mean_baseline_cell",
                     data.ct_cell, gene_sparsity_cell, data.common_cts, data.gene_names)
    all_globals.append(r['global'])
    all_ct.append(r['by_celltype'])
    all_sp.append(r['by_sparsity'])

    # ---- 5. CT mean baseline for nuclei ----
    print("--- CT mean baseline (nuclei) ---")
    r = analyze_pair(data.X_nuc, ct_mean_nuc, "ct_mean_baseline_nuc",
                     data.ct_nuc, gene_sparsity_nuc, data.common_cts, data.gene_names)
    all_globals.append(r['global'])
    all_ct.append(r['by_celltype'])
    all_sp.append(r['by_sparsity'])

    # ---- 6. Cross-decoded vs raw cell (CT-matched) ----
    print("--- Cross-decoded vs raw cell (CT-matched) ---")
    r = cross_decode_ct_matched_analysis(
        sources['cross_decoded'], data.ct_nuc,
        data.X_cell, data.ct_cell,
        gene_sparsity_cell, data.common_cts, data.gene_names
    )
    all_globals.append(r['global'])
    all_ct.append(r['by_celltype'])

    # ---- 7. Cross-decoded vs raw cell CT mean ----
    print("--- Cross-decoded vs raw cell CT mean ---")
    r = cross_decode_ct_mean_analysis(
        sources['cross_decoded'], data.ct_nuc,
        data.X_cell, data.ct_cell,
        gene_sparsity_cell, data.common_cts, data.gene_names
    )
    all_globals.append(r['global'])
    all_ct.append(r['by_celltype'])

    # ---- Save ----
    global_df = pd.concat(all_globals, ignore_index=True)
    ct_df = pd.concat(all_ct, ignore_index=True)
    sp_df = pd.concat(all_sp, ignore_index=True)

    global_df.to_csv(os.path.join(out_dir, "global_summary.csv"), index=False)
    ct_df.to_csv(os.path.join(out_dir, "by_celltype.csv"), index=False)
    sp_df.to_csv(os.path.join(out_dir, "by_sparsity.csv"), index=False)

    # ---- Print summary ----
    print("\n" + "=" * 70)
    print("GLOBAL SUMMARY")
    print("=" * 70)
    for _, row in global_df.iterrows():
        label = row.get('label', '?')
        frac_z = row.get('frac_zero', row.get('frac_zero_in_ct_mean', '?'))
        print(f"\n  [{label}]  frac_zero={frac_z}")
        print(f"    Zero entries:    count={row.get('zero_count','?')}, "
              f"dec_mean={row.get('zero_dec_mean','?'):.4f}, "
              f"dec_std={row.get('zero_dec_std','?'):.4f}")
        nz_count = row.get('nonzero_count', '?')
        if nz_count and nz_count != '?' and nz_count > 0:
            print(f"    Nonzero entries: count={nz_count}, "
                  f"raw_mean={row.get('nonzero_raw_mean','?'):.4f}, "
                  f"dec_mean={row.get('nonzero_dec_mean','?'):.4f}, "
                  f"std_ratio={row.get('nonzero_std_ratio','?')}, "
                  f"pearson={row.get('nonzero_pearson_r','?')}")
        auroc = row.get('auroc_nonzero_vs_zero', '?')
        if auroc != '?':
            print(f"    Separation:      AUROC={auroc:.4f}, "
                  f"mean_gap={row.get('mean_gap','?'):.4f}")

    print(f"\nResults saved to: {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
