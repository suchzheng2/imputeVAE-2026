"""Library size analysis comparing multiple models.

For each model:
  1. Library size preservation (raw vs decoded correlation)
  2. Variance decomposition: per-gene R² of expression ~ log(libsize)
  3. Proportion-normalized std_ratio (honest variance after removing libsize)
  4. Latent dimension correlation with library size

Usage:
    python analysis/libsize_comparison.py \
        --runs run/run_A run/run_B \
        --labels log1p_model cpm_model \
        --device cuda
"""

import argparse
import gc
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.delta_analysis import load_model
from imputeVAE.data import load_raw_data, filter_by_protocol

CELL_ID, NUCLEI_ID = 0, 1


def get_expression(adata, norm, use_raw):
    """Extract expression matrix from adata in the right normalization."""
    layer = norm if use_raw and norm and norm != "raw" else None
    if layer and layer in adata.layers:
        X = np.array(adata.layers[layer], dtype=np.float32)
    else:
        X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.array(X, dtype=np.float32)
    return X


def compute_library_sizes(X, norm):
    """Compute library sizes from expression matrix.
    For log1p: expm1 then sum. For cpm: sum directly (should be ~1e6)."""
    if norm == "log1p":
        raw = np.expm1(np.clip(X, 0, None))
    else:
        raw = np.clip(X, 0, None)
    return raw.sum(axis=1)


def run_inference(model, X, run_args, device, mode="same", batch_size=256):
    """Run model inference. mode='same' for same-modality decode, 'cross' for cross-decode.
    Returns (decoded, latent_mu)."""
    is_cond = run_args.get("fine_tune_cond_stage3", False)
    decoded_list, latent_list = [], []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
            if is_cond:
                # Encode as nuclei
                z, mu, _ = model.encode(batch, modality_id=NUCLEI_ID)
                if mode == "cross":
                    xd = model.decode(mu, modality_id=CELL_ID)
                else:
                    xd = model.decode(mu, modality_id=NUCLEI_ID)
            else:
                encoded = model.encoder(batch)
                mu = model.mu(encoded)
                xd = model.decoder(mu)
            decoded_list.append(xd.cpu().numpy())
            latent_list.append(mu.cpu().numpy())

    return np.concatenate(decoded_list), np.concatenate(latent_list)


def _pearson_r(x, y):
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def analyze_one_model(run_folder, label, device):
    """Run full library size analysis for one model."""
    print(f"\n{'='*70}")
    print(f"Model: {label} ({run_folder})")
    print(f"{'='*70}")

    with open(f"{run_folder}/arguments.json") as f:
        run_args = json.load(f)

    norm = run_args.get("raw_data_normalization", "log1p")
    use_raw = run_args.get("use_raw_data", False)
    args_ns = SimpleNamespace(**run_args)

    # Load data
    print("Loading nuclei data...")
    adata_nuc = load_raw_data("10X_nuclei_v2_v3_AIBS/", args_ns)
    adata_nuc = filter_by_protocol(adata_nuc, args_ns)
    X_nuc = get_expression(adata_nuc, norm, use_raw)
    ct_nuc = np.array(adata_nuc.obs["cell_type"].values)
    gene_names = np.array(adata_nuc.var["gene_names"].values)

    # Also load cell data for same-modality reference
    print("Loading cell data...")
    cell_args = SimpleNamespace(**{**run_args, "batch": "c3A"})
    adata_cell = load_raw_data("10X_cells_v2_v3_AIBS/", cell_args)
    adata_cell = filter_by_protocol(adata_cell, cell_args)
    X_cell = get_expression(adata_cell, norm, use_raw)
    ct_cell = np.array(adata_cell.obs["cell_type"].values)
    common_cts = sorted(set(ct_nuc) & set(ct_cell))

    del adata_nuc, adata_cell
    gc.collect()

    print(f"  Nuclei: {X_nuc.shape}, Cells: {X_cell.shape}, norm: {norm}")

    # Load model and run inference
    model, _ = load_model(run_folder, device)

    print("Cross-decoding nuclei -> cell...")
    X_cross, latent_nuc = run_inference(model, X_nuc, run_args, device, mode="cross")

    print("Same-modality decoding nuclei...")
    X_same, _ = run_inference(model, X_nuc, run_args, device, mode="same")

    del model
    gc.collect()

    # Output dir
    out_dir = os.path.join(run_folder, "library_size_analysis")
    os.makedirs(out_dir, exist_ok=True)

    # === Analysis 1: Library size preservation ===
    print("\n--- Library size preservation ---")
    lib_raw_nuc = compute_library_sizes(X_nuc, norm)
    lib_cross = compute_library_sizes(X_cross, norm)
    lib_same = compute_library_sizes(X_same, norm)
    lib_raw_cell = compute_library_sizes(X_cell, norm)

    r_cross = _pearson_r(lib_raw_nuc, lib_cross)
    r_same = _pearson_r(lib_raw_nuc, lib_same)
    print(f"  Libsize corr (raw_nuc vs cross-decoded): {r_cross:.4f}")
    print(f"  Libsize corr (raw_nuc vs same-decoded):  {r_same:.4f}")
    print(f"  Raw nuc libsize mean: {lib_raw_nuc.mean():.0f}, std: {lib_raw_nuc.std():.0f}")
    print(f"  Cross-dec libsize mean: {lib_cross.mean():.0f}, std: {lib_cross.std():.0f}")

    per_ct_lib = []
    for ct in common_cts:
        mask = ct_nuc == ct
        n = mask.sum()
        if n < 5:
            continue
        r = _pearson_r(lib_raw_nuc[mask], lib_cross[mask])
        per_ct_lib.append({
            "ct": ct, "n": n,
            "lib_raw_mean": lib_raw_nuc[mask].mean(),
            "lib_cross_mean": lib_cross[mask].mean(),
            "pearson": r,
        })
        print(f"    {ct:<15} n={n:>5}  raw={lib_raw_nuc[mask].mean():>10.0f}  "
              f"cross={lib_cross[mask].mean():>10.0f}  r={r:.3f}")

    pd.DataFrame(per_ct_lib).to_csv(os.path.join(out_dir, "libsize_per_ct.csv"), index=False)

    # === Analysis 2: Variance decomposition ===
    print("\n--- Variance decomposition (R² of expression ~ log libsize) ---")
    log_lib_raw = np.log1p(lib_raw_nuc)
    log_lib_cross = np.log1p(lib_cross)

    rows = []
    for ct in common_cts:
        mask = ct_nuc == ct
        n = mask.sum()
        if n < 10:
            continue
        X_ct = X_nuc[mask]
        D_ct = X_cross[mask]
        ll_raw = log_lib_raw[mask] - log_lib_raw[mask].mean()
        ll_cross = log_lib_cross[mask] - log_lib_cross[mask].mean()

        X_c = X_ct - X_ct.mean(axis=0, keepdims=True)
        D_c = D_ct - D_ct.mean(axis=0, keepdims=True)

        x_norms = np.sqrt((X_c ** 2).sum(axis=0))
        d_norms = np.sqrt((D_c ** 2).sum(axis=0))
        x_norms = np.where(x_norms == 0, 1, x_norms)
        d_norms = np.where(d_norms == 0, 1, d_norms)
        ll_raw_norm = np.sqrt((ll_raw ** 2).sum()) or 1
        ll_cross_norm = np.sqrt((ll_cross ** 2).sum()) or 1

        corr_raw = (X_c * ll_raw[:, None]).sum(axis=0) / (x_norms * ll_raw_norm)
        corr_cross = (D_c * ll_cross[:, None]).sum(axis=0) / (d_norms * ll_cross_norm)

        r2_raw = corr_raw ** 2
        r2_cross = corr_cross ** 2
        frac_nz = (X_ct > 0).mean(axis=0)

        for g in range(len(gene_names)):
            rows.append({
                "ct": ct, "gene": gene_names[g],
                "frac_nonzero": frac_nz[g],
                "r2_raw_libsize": r2_raw[g],
                "r2_cross_libsize": r2_cross[g],
            })

    df_decomp = pd.DataFrame(rows)

    bins = [(0.7, 1.01, ">70%"), (0.3, 0.7, "30-70%"),
            (0.1, 0.3, "10-30%"), (0, 0.1, "<10%")]
    print(f"  {'Sparsity':>10}  {'R²_raw':>8}  {'R²_cross':>8}  {'n':>8}")
    for lo, hi, lbl in bins:
        mask = (df_decomp["frac_nonzero"] >= lo) & (df_decomp["frac_nonzero"] < hi)
        sub = df_decomp[mask]
        print(f"  {lbl:>10}  {sub['r2_raw_libsize'].median():>8.4f}  "
              f"{sub['r2_cross_libsize'].median():>8.4f}  {len(sub):>8}")

    df_decomp.to_csv(os.path.join(out_dir, "variance_decomposition.csv"), index=False)

    # === Analysis 3: Proportion-normalized std ===
    print("\n--- Proportion-normalized std_ratio (honest variance) ---")
    if norm == "log1p":
        raw_counts = np.expm1(np.clip(X_nuc, 0, None))
        cross_counts = np.expm1(np.clip(X_cross, 0, None))
    else:
        raw_counts = np.clip(X_nuc, 0, None)
        cross_counts = np.clip(X_cross, 0, None)

    lib_raw_safe = np.where(lib_raw_nuc == 0, 1, lib_raw_nuc)
    lib_cross_safe = np.where(lib_cross == 0, 1, lib_cross)
    prop_raw = raw_counts / lib_raw_safe[:, None]
    prop_cross = cross_counts / lib_cross_safe[:, None]

    prop_rows = []
    for ct in common_cts:
        mask = ct_nuc == ct
        n = mask.sum()
        if n < 10:
            continue
        std_raw_expr = X_nuc[mask].std(axis=0)
        std_cross_expr = X_cross[mask].std(axis=0)
        std_raw_prop = prop_raw[mask].std(axis=0)
        std_cross_prop = prop_cross[mask].std(axis=0)
        frac_nz = (X_nuc[mask] > 0).mean(axis=0)

        for g in range(len(gene_names)):
            sr_expr = std_cross_expr[g] / std_raw_expr[g] if std_raw_expr[g] > 0 else 0
            sr_prop = std_cross_prop[g] / std_raw_prop[g] if std_raw_prop[g] > 0 else 0
            prop_rows.append({
                "ct": ct, "gene": gene_names[g],
                "frac_nonzero": frac_nz[g],
                "std_ratio_expr": sr_expr,
                "std_ratio_prop": sr_prop,
            })

    df_prop = pd.DataFrame(prop_rows)
    print(f"  {'Sparsity':>10}  {'expr':>8}  {'prop':>8}  {'n':>8}")
    for lo, hi, lbl in bins:
        mask = (df_prop["frac_nonzero"] >= lo) & (df_prop["frac_nonzero"] < hi)
        sub = df_prop[mask]
        print(f"  {lbl:>10}  {sub['std_ratio_expr'].median():>8.4f}  "
              f"{sub['std_ratio_prop'].median():>8.4f}  {len(sub):>8}")

    df_prop.to_csv(os.path.join(out_dir, "proportion_std_comparison.csv"), index=False)

    # === Analysis 4: Latent-libsize correlation ===
    print("\n--- Latent dimension correlation with log(library size) ---")
    log_lib = np.log1p(lib_raw_nuc)
    n_dims = latent_nuc.shape[1]
    dim_corrs = []
    for d in range(n_dims):
        r = _pearson_r(latent_nuc[:, d], log_lib)
        dim_corrs.append({"dim": d, "pearson": r, "abs_pearson": abs(r)})
    df_latent = pd.DataFrame(dim_corrs).sort_values("abs_pearson", ascending=False)

    print("  Top 5 dims correlated with log(libsize):")
    for _, row in df_latent.head(5).iterrows():
        print(f"    dim {int(row['dim']):3d}  r = {row['pearson']:+.4f}")

    df_latent.to_csv(os.path.join(out_dir, "latent_libsize_correlation.csv"), index=False)

    # === Figures ===
    # Libsize scatter
    fig, ax = plt.subplots(figsize=(8, 7))
    for ct in common_cts:
        mask = ct_nuc == ct
        ax.scatter(lib_raw_nuc[mask], lib_cross[mask], s=2, alpha=0.3, label=ct)
    mx = max(np.percentile(lib_raw_nuc, 99.5), np.percentile(lib_cross, 99.5))
    ax.plot([0, mx], [0, mx], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("Raw nuclei library size")
    ax.set_ylabel("Cross-decoded library size")
    ax.set_title(f"{label}: Library size preservation (r={r_cross:.3f})")
    ax.legend(fontsize=6, markerscale=4, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_libsize_scatter.png"), dpi=150)
    plt.close()

    # Latent-libsize bar
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in df_latent.sort_values("dim")["pearson"]]
    ax.bar(range(n_dims), df_latent.sort_values("dim")["pearson"].values, color=colors, alpha=0.7)
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel("Pearson r with log(libsize)")
    ax.set_title(f"{label}: Latent dims vs library size")
    ax.axhline(0, color="k", lw=0.5)
    top = df_latent.iloc[0]
    ax.annotate(f"dim {int(top['dim'])}: r={top['pearson']:.3f}",
                xy=(top["dim"], top["pearson"]),
                xytext=(top["dim"] + 3, top["pearson"] * 0.8),
                fontsize=10, arrowprops=dict(arrowstyle="->"))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_latent_libsize.png"), dpi=150)
    plt.close()

    print(f"\nOutputs saved to: {out_dir}")

    # Return summary
    dense_mask = df_decomp["frac_nonzero"] >= 0.7
    dense_prop = df_prop["frac_nonzero"] >= 0.7
    return {
        "label": label,
        "norm": norm,
        "libsize_corr_cross": r_cross,
        "libsize_corr_same": r_same,
        "r2_dense_raw": df_decomp.loc[dense_mask, "r2_raw_libsize"].median(),
        "r2_dense_cross": df_decomp.loc[dense_mask, "r2_cross_libsize"].median(),
        "std_ratio_dense_expr": df_prop.loc[dense_prop, "std_ratio_expr"].median(),
        "std_ratio_dense_prop": df_prop.loc[dense_prop, "std_ratio_prop"].median(),
        "top_latent_dim": int(df_latent.iloc[0]["dim"]),
        "top_latent_r": df_latent.iloc[0]["pearson"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    assert len(args.runs) == len(args.labels)
    device = torch.device(args.device)

    summaries = []
    for run_folder, label in zip(args.runs, args.labels):
        summary = analyze_one_model(run_folder, label, device)
        summaries.append(summary)
        gc.collect()

    # Print comparison
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Metric':<30}", end="")
    for s in summaries:
        print(f"  {s['label']:>15}", end="")
    print()
    print("-" * (30 + 17 * len(summaries)))

    for key in ["norm", "libsize_corr_cross", "libsize_corr_same",
                "r2_dense_raw", "r2_dense_cross",
                "std_ratio_dense_expr", "std_ratio_dense_prop",
                "top_latent_dim", "top_latent_r"]:
        print(f"{key:<30}", end="")
        for s in summaries:
            val = s[key]
            if isinstance(val, float):
                print(f"  {val:>15.4f}", end="")
            else:
                print(f"  {str(val):>15}", end="")
        print()


if __name__ == "__main__":
    main()
