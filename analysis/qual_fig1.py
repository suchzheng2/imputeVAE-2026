"""
Figure 1 for qualifying-exam report: cell-type variation in nucleus-cell expression relationship.

Panel A: Pearson correlation between mean whole-cell and nuclear expression per cell type.
Panel B: Mean absolute empirical shift (cell minus nucleus) per cell type.
Both panels sorted by correction magnitude, with 95% bootstrap CIs.
"""

import os
import gc
import numpy as np
import pandas as pd
import anndata as ad
from scipy import stats, sparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_BASE = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi"
CELL_PATH = os.path.join(RAW_BASE, "10X_cells_v2_v3_AIBS", "data.h5ad")
NUCLEI_PATH = os.path.join(RAW_BASE, "10X_nuclei_v2_v3_AIBS", "data.h5ad")
GENE_LIST_PATH = os.path.join(DATA_DIR, "training_gene_list_v3.csv")
FIG_DIR = os.path.join(PROJECT_ROOT, "fig")

# ── Cell types ─────────────────────────────────────────────────────────────
CELL_TYPES = [
    "Astro", "Endo", "L2/3 IT", "L5 ET", "L5 IT", "L5/6 NP",
    "L6 CT", "L6 IT", "L6b", "Lamp5", "Macrophage", "OPC", "Oligo",
    "Pvalb", "Sncg", "Sst", "Vip",
]

NEURONAL = {
    "L2/3 IT", "L5 ET", "L5 IT", "L5/6 NP", "L6 CT", "L6 IT", "L6b",
    "Lamp5", "Pvalb", "Sncg", "Sst", "Vip",
}

NON_NEURONAL = {"Astro", "Endo", "Macrophage", "OPC", "Oligo"}


def load_ct_means_and_data(h5ad_path, batch_suffix, gene_mask, gene_order):
    """Load h5ad, filter to v3 + cell types + genes. Return per-CT data dict."""
    print(f"Loading {h5ad_path} ...")
    adata = ad.read_h5ad(h5ad_path, backed="r")

    # Build meta_batch_id from batch_id
    meta_batch = adata.obs["batch_id"].apply(lambda x: x.split(".")[0])

    # v3 chemistry filter
    v3_mask = meta_batch.str.endswith(batch_suffix)
    ct_mask = adata.obs["cell_type"].isin(CELL_TYPES)
    row_mask = v3_mask & ct_mask
    indices = np.where(row_mask.values)[0]
    cell_types_filtered = adata.obs["cell_type"].values[indices]
    print(f"  After v3+CT filter: {len(indices)} observations")

    # Read data in chunks to avoid OOM, applying gene filter
    ct_data = {}
    CHUNK = 5000
    for start in range(0, len(indices), CHUNK):
        end = min(start + CHUNK, len(indices))
        chunk_idx = indices[start:end]
        chunk_cts = cell_types_filtered[start:end]

        # Read chunk from backed store
        X_chunk = adata.X[chunk_idx]
        if sparse.issparse(X_chunk):
            X_chunk = X_chunk.toarray()

        # Gene filter + reorder
        X_chunk = X_chunk[:, gene_mask][:, gene_order].astype(np.float32)
        # log1p
        np.log1p(X_chunk, out=X_chunk)

        for ct in CELL_TYPES:
            ct_rows = X_chunk[chunk_cts == ct]
            if ct_rows.shape[0] == 0:
                continue
            if ct not in ct_data:
                ct_data[ct] = []
            ct_data[ct].append(ct_rows)

        if (start // CHUNK) % 10 == 0:
            print(f"    Processed {end}/{len(indices)} rows")

    # Close backed file
    if hasattr(adata, "file") and adata.file is not None:
        adata.file.close()
    del adata
    gc.collect()

    # Concatenate chunks per CT
    for ct in ct_data:
        ct_data[ct] = np.concatenate(ct_data[ct], axis=0)

    return ct_data


def main():
    gene_list = pd.read_csv(GENE_LIST_PATH, header=None)[0].tolist()
    gene_set = set(gene_list)
    print(f"Gene list: {len(gene_list)} genes")

    # Get gene info from one file to build mask/order
    print("Reading gene metadata ...")
    adata_tmp = ad.read_h5ad(CELL_PATH, backed="r")
    all_gene_names = adata_tmp.var["gene_names"].values
    gene_mask = np.array([g in gene_set for g in all_gene_names])
    filtered_genes = all_gene_names[gene_mask]
    gene_order = np.argsort(filtered_genes)
    sorted_genes = filtered_genes[gene_order]
    n_genes = len(sorted_genes)
    print(f"  Genes after filter+sort: {n_genes}")
    if hasattr(adata_tmp, "file") and adata_tmp.file is not None:
        adata_tmp.file.close()
    del adata_tmp
    gc.collect()

    # Verify nuclei have same genes
    adata_tmp2 = ad.read_h5ad(NUCLEI_PATH, backed="r")
    nuc_genes = adata_tmp2.var["gene_names"].values
    nuc_filtered = nuc_genes[np.array([g in gene_set for g in nuc_genes])]
    nuc_sorted = nuc_filtered[np.argsort(nuc_filtered)]
    assert np.array_equal(sorted_genes, nuc_sorted), "Gene mismatch between cell and nuclei!"
    # Need nuclei gene mask/order too (may differ in original order)
    nuc_gene_mask = np.array([g in gene_set for g in nuc_genes])
    nuc_gene_order = np.argsort(nuc_genes[nuc_gene_mask])
    if hasattr(adata_tmp2, "file") and adata_tmp2.file is not None:
        adata_tmp2.file.close()
    del adata_tmp2
    gc.collect()

    print(f"Verified: {n_genes} genes in identical order across modalities")

    # Load data per cell type
    cell_data = load_ct_means_and_data(CELL_PATH, "c3A", gene_mask, gene_order)
    nuc_data = load_ct_means_and_data(NUCLEI_PATH, "n3A", nuc_gene_mask, nuc_gene_order)

    # ── Per-cell-type statistics ───────────────────────────────────────────
    np.random.seed(42)
    N_BOOT = 500
    results = []

    for ct in CELL_TYPES:
        Xc = cell_data.get(ct)
        Xn = nuc_data.get(ct)
        if Xc is None or Xn is None:
            print(f"  {ct}: MISSING in one modality, skipping")
            continue
        n_cell = Xc.shape[0]
        n_nuc = Xn.shape[0]
        print(f"  {ct}: {n_cell} cells, {n_nuc} nuclei")

        # Point estimates
        mean_c = Xc.mean(axis=0)
        mean_n = Xn.mean(axis=0)
        corr_point = float(np.corrcoef(mean_c, mean_n)[0, 1])
        mag_point = float(np.abs(mean_c - mean_n).mean())

        # Bootstrap
        boot_corrs = np.empty(N_BOOT)
        boot_mags = np.empty(N_BOOT)
        unstable = n_cell < 10 or n_nuc < 10

        for b in range(N_BOOT):
            idx_c = np.random.randint(0, n_cell, size=n_cell)
            idx_n = np.random.randint(0, n_nuc, size=n_nuc)
            mc = Xc[idx_c].mean(axis=0)
            mn = Xn[idx_n].mean(axis=0)
            boot_corrs[b] = np.corrcoef(mc, mn)[0, 1]
            boot_mags[b] = np.abs(mc - mn).mean()

        corr_lo, corr_hi = np.percentile(boot_corrs, [2.5, 97.5])
        mag_lo, mag_hi = np.percentile(boot_mags, [2.5, 97.5])

        results.append({
            "cell_type": ct,
            "class": "Neuronal" if ct in NEURONAL else "Non-neuronal",
            "n_cell": n_cell,
            "n_nuc": n_nuc,
            "corr": corr_point,
            "corr_lo": float(corr_lo),
            "corr_hi": float(corr_hi),
            "mag": mag_point,
            "mag_lo": float(mag_lo),
            "mag_hi": float(mag_hi),
            "unstable_ci": unstable,
        })

    df = pd.DataFrame(results)
    # Sort by correction magnitude: ascending so largest is at top of horizontal plot
    df = df.sort_values("mag", ascending=True).reset_index(drop=True)

    # ── Save data CSV ──────────────────────────────────────────────────────
    os.makedirs(FIG_DIR, exist_ok=True)
    df.to_csv(os.path.join(FIG_DIR, "qual-fig1-data.csv"), index=False)

    # ── Plot ───────────────────────────────────────────────────────────────
    COLOR_NEURONAL = "#3574B0"
    COLOR_NONNEURONAL = "#2CA02C"

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    fig.subplots_adjust(wspace=0.08, left=0.22, right=0.95, top=0.92, bottom=0.08)

    y_pos = np.arange(len(df))
    colors = [COLOR_NEURONAL if c == "Neuronal" else COLOR_NONNEURONAL for c in df["class"]]

    y_labels = [f"{row.cell_type} ({row.n_cell}/{row.n_nuc})" for _, row in df.iterrows()]

    # Panel A — clamp xerr to >= 0 (bootstrap jitter can cause tiny negatives)
    corr_lo_err = np.maximum(df["corr"].values - df["corr_lo"].values, 0)
    corr_hi_err = np.maximum(df["corr_hi"].values - df["corr"].values, 0)
    ax_a.errorbar(df["corr"], y_pos, xerr=[corr_lo_err, corr_hi_err],
                  fmt="none", ecolor="gray", elinewidth=0.8, capsize=2, zorder=1)
    ax_a.scatter(df["corr"], y_pos, c=colors, s=50, zorder=2, edgecolors="white", linewidths=0.5)
    for i, row in df.iterrows():
        ax_a.text(row["corr"] + 0.003, y_pos[i], f"{row['corr']:.3f}", va="center", fontsize=7)

    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(y_labels, fontsize=8)
    ax_a.set_xlabel("Pearson r", fontsize=10)
    ax_a.set_title("A. Expression concordance", fontsize=11, fontweight="bold", loc="left")
    ax_a.grid(axis="x", alpha=0.3, linewidth=0.5)
    ax_a.set_axisbelow(True)

    # Panel B — clamp xerr to >= 0
    mag_lo_err = np.maximum(df["mag"].values - df["mag_lo"].values, 0)
    mag_hi_err = np.maximum(df["mag_hi"].values - df["mag"].values, 0)
    ax_b.errorbar(df["mag"], y_pos, xerr=[mag_lo_err, mag_hi_err],
                  fmt="none", ecolor="gray", elinewidth=0.8, capsize=2, zorder=1)
    ax_b.scatter(df["mag"], y_pos, c=colors, s=50, zorder=2, edgecolors="white", linewidths=0.5)
    for i, row in df.iterrows():
        ax_b.text(row["mag"] + 0.002, y_pos[i], f"{row['mag']:.3f}", va="center", fontsize=7)

    ax_b.set_xlabel("Mean |cell \u2212 nucleus| shift", fontsize=10)
    ax_b.set_title("B. Empirical shift magnitude", fontsize=11, fontweight="bold", loc="left")
    ax_b.grid(axis="x", alpha=0.3, linewidth=0.5)
    ax_b.set_axisbelow(True)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_NEURONAL, markersize=8, label="Neuronal"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_NONNEURONAL, markersize=8, label="Non-neuronal"),
    ]
    ax_b.legend(handles=legend_elements, loc="lower right", fontsize=8, framealpha=0.9)

    unstable_cts = df[df["unstable_ci"]]["cell_type"].tolist()
    if unstable_cts:
        for i, row in df.iterrows():
            if row["unstable_ci"]:
                ax_a.scatter(row["corr"], y_pos[i], marker="*", c="red", s=20, zorder=3)

    for ax in (ax_a, ax_b):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ext in ["png", "pdf", "svg"]:
        path = os.path.join(FIG_DIR, f"qual-fig1.{ext}")
        fig.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight", facecolor="white")
        print(f"Saved {path}")
    plt.close()

    # ── Statistical summary ────────────────────────────────────────────────
    neuro = df[df["class"] == "Neuronal"]
    non_neuro = df[df["class"] == "Non-neuronal"]

    spearman_r, spearman_p = stats.spearmanr(df["corr"], df["mag"])

    summary_lines = [
        "Statistical Summary - Figure 1",
        "=" * 50,
        f"Number of cell types: {len(df)}",
        f"Number of genes: {n_genes}",
        "",
        f"Correlation range: {df['corr'].min():.4f} - {df['corr'].max():.4f}",
        f"Correction magnitude range: {df['mag'].min():.4f} - {df['mag'].max():.4f}",
        "",
        f"Median correlation (neuronal): {neuro['corr'].median():.4f}",
        f"Median correlation (non-neuronal): {non_neuro['corr'].median():.4f}",
        f"Median correction magnitude (neuronal): {neuro['mag'].median():.4f}",
        f"Median correction magnitude (non-neuronal): {non_neuro['mag'].median():.4f}",
        f"Neuronal/non-neuronal magnitude ratio: {neuro['mag'].median() / non_neuro['mag'].median():.3f}",
        "",
        f"Spearman rho (corr vs magnitude): {spearman_r:.4f} (p={spearman_p:.4g})",
        "",
        "Per-cell-type data:",
    ]

    for _, row in df.sort_values("mag", ascending=False).iterrows():
        flag = " *UNSTABLE CI*" if row["unstable_ci"] else ""
        summary_lines.append(
            f"  {row['cell_type']:15s} ({row['class']:13s}) "
            f"r={row['corr']:.4f} [{row['corr_lo']:.4f}, {row['corr_hi']:.4f}]  "
            f"mag={row['mag']:.4f} [{row['mag_lo']:.4f}, {row['mag_hi']:.4f}]  "
            f"n_cell={row['n_cell']}  n_nuc={row['n_nuc']}{flag}"
        )

    summary_lines.extend(["", "Interpretation:"])
    if neuro["corr"].median() < non_neuro["corr"].median():
        summary_lines.append("  Neuronal cell types have lower median correlation than non-neuronal,")
        summary_lines.append("  suggesting more divergent nucleus-cell expression patterns in neurons.")
    else:
        summary_lines.append("  No systematic difference: neuronal median correlation is not lower than non-neuronal.")

    if neuro["mag"].median() > non_neuro["mag"].median():
        summary_lines.append("  Neuronal cell types have larger median correction magnitude,")
        summary_lines.append("  consistent with greater cytoplasmic RNA content in neurons.")
    else:
        summary_lines.append("  No systematic difference in correction magnitude between classes.")

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    with open(os.path.join(FIG_DIR, "qual-fig1-summary.txt"), "w") as f:
        f.write(summary_text + "\n")

    # ── Caption ────────────────────────────────────────────────────────────
    caption = (
        "Figure 1. The empirical nucleus-cell relationship varies across cell types. "
        "(A) Pearson correlation across 10,134 genes between mean whole-cell and "
        "nuclear expression profiles for each of the 17 cell types represented in both "
        "modalities. (B) Mean absolute empirical cell-minus-nucleus expression shift "
        "across genes for each cell type. Correlation measures preservation of the "
        "relative expression pattern, whereas mean absolute shift measures the "
        "magnitude of the nucleus-cell difference. Points show cell-type estimates; "
        "intervals show 95% bootstrap confidence intervals obtained by independently "
        "resampling cells and nuclei within each cell type. Neuronal and non-neuronal "
        "populations are shown in blue and green, respectively."
    )
    with open(os.path.join(FIG_DIR, "qual-fig1-caption.txt"), "w") as f:
        f.write(caption + "\n")

    # ── Final report ───────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("FILES LOADED:")
    print(f"  Cells:   {CELL_PATH}")
    print(f"  Nuclei:  {NUCLEI_PATH}")
    print(f"  Genes:   {GENE_LIST_PATH}")
    print("\nFILTERING STEPS:")
    print("  1. v3 chemistry: c3A (cells), n3A (nuclei)")
    print("  2. Cell type: 17 common types (L6 IT Car3 excluded)")
    print(f"  3. Gene list: {n_genes} genes from training_gene_list_v3.csv")
    print("  4. log1p transformation on raw counts")
    print(f"\nDATA: Full working dataset (no train/valid/test split)")
    print(f"\nOBSERVATION COUNTS:")
    for _, row in df.sort_values("mag", ascending=False).iterrows():
        print(f"  {row['cell_type']:15s}  cells={row['n_cell']:6d}  nuclei={row['n_nuc']:6d}")
    if unstable_cts:
        print(f"\nUNSTABLE BOOTSTRAP: {', '.join(unstable_cts)} (<10 obs in one modality)")
    else:
        print("\nNo cell types with unstable bootstrap intervals.")


if __name__ == "__main__":
    main()
