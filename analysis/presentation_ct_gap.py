"""Slide 6 figure: cell vs nuclei mean expression scatter, per cell type.

Shows that the nuc-cell gap varies by cell type:
- Neurons (large, long processes) have bigger gaps
- Glia (compact) have smaller gaps

Output: multi-panel scatter + summary bar chart of per-CT correlation/shift.
"""

import os
import gc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "notes", "qualify-exam", "pre-qual", "figures")

GENE_LIST = os.path.join(PROJECT_ROOT, "data", "training_gene_list_v3.csv")

# Cell types to show (mix of neuronal + glial, with enough cells in both modalities)
SHOW_CTS = ["L2/3 IT", "L5 IT", "L5 ET", "Pvalb", "Sst", "Vip", "Lamp5", "Astro", "Oligo", "OPC"]

CELL_BATCH = "c3A"
NUC_BATCH = "n3A"


def compute_ct_means_backed(dataset_dir, batch_prefix, gene_list):
    """Load h5ad in backed mode, compute per-CT mean expression for gene list genes.

    Processes one cell type at a time to avoid loading full matrix into memory.
    Returns: dict {ct_name: mean_array}, gene_names array, dict {ct_name: n_cells}
    """
    path = os.path.join(DATA_ROOT, dataset_dir, "data.h5ad")
    print(f"Loading {path} (backed)...")
    adata = ad.read_h5ad(path, backed="r")

    # Get gene names and find indices for our gene list
    if "gene_names" in adata.var.columns:
        all_genes = adata.var["gene_names"].values.astype(str)
    else:
        all_genes = adata.var_names.values.astype(str)

    gene_set = set(gene_list)
    gene_indices = [i for i, g in enumerate(all_genes) if g in gene_set]
    matched_genes = all_genes[gene_indices]
    print(f"  Genes matched: {len(gene_indices)}")

    # Get batch and cell type info
    obs = adata.obs.copy()
    batch_mask = obs["batch_id"].str.contains(batch_prefix, regex=False).values
    print(f"  Cells in batch {batch_prefix}: {batch_mask.sum()}")

    ct_means = {}
    ct_counts = {}

    for ct in SHOW_CTS:
        ct_mask = (obs["cell_type"].values == ct) & batch_mask
        n = ct_mask.sum()
        if n < 20:
            print(f"  {ct}: {n} cells (skipping)")
            continue

        # Load only cells for this CT — use integer indexing
        row_idx = np.where(ct_mask)[0]

        # Process in chunks to limit memory
        chunk_size = 2000
        running_sum = np.zeros(len(gene_indices), dtype=np.float64)
        for start in range(0, len(row_idx), chunk_size):
            end = min(start + chunk_size, len(row_idx))
            chunk_idx = row_idx[start:end]
            # Read chunk, select gene columns
            X_chunk = adata.X[chunk_idx]
            if sp.issparse(X_chunk):
                X_chunk = X_chunk.toarray()
            running_sum += X_chunk[:, gene_indices].sum(axis=0)

        ct_means[ct] = running_sum / n
        ct_counts[ct] = n
        print(f"  {ct}: {n} cells, mean range [{ct_means[ct].min():.3f}, {ct_means[ct].max():.1f}]")

    if hasattr(adata, "file") and adata.file is not None:
        adata.file.close()

    gc.collect()
    return ct_means, matched_genes, ct_counts


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    gene_list = pd.read_csv(GENE_LIST, header=None)[0].values
    print(f"Gene list: {len(gene_list)} genes\n")

    print("=== Cell data ===")
    ct_means_cell, genes_cell, ct_n_cell = compute_ct_means_backed(
        "10X_cells_v2_v3_AIBS", CELL_BATCH, gene_list)

    print("\n=== Nuclei data ===")
    ct_means_nuc, genes_nuc, ct_n_nuc = compute_ct_means_backed(
        "10X_nuclei_v2_v3_AIBS", NUC_BATCH, gene_list)

    assert np.array_equal(genes_cell, genes_nuc), "Gene order mismatch!"

    common_cts = [ct for ct in SHOW_CTS if ct in ct_means_cell and ct in ct_means_nuc]
    print(f"\nCell types to plot: {common_cts}")

    # --- Figure 1: Multi-panel scatter ---
    n_cts = len(common_cts)
    ncols = min(5, n_cts)
    nrows = (n_cts + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)

    summary = []

    for i, ct in enumerate(common_cts):
        ax = axes[i // ncols][i % ncols]
        cm = ct_means_cell[ct] + 1e-6
        nm = ct_means_nuc[ct] + 1e-6

        log2fc = np.log2(cm / nm)
        median_shift = np.median(log2fc)
        r, _ = stats.pearsonr(np.log10(cm), np.log10(nm))

        abs_fc = np.abs(log2fc)
        colors = np.where(abs_fc > 3, "#DC3545",
                 np.where(abs_fc > 1, "#E8912D", "#4682B4"))

        ax.scatter(cm, nm, c=colors, s=3, alpha=0.4, edgecolors="none", rasterized=True)

        lims = [1e-3, max(cm.max(), nm.max()) * 2]
        ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        ax.set_title(f"{ct}\nr={r:.3f}, shift={median_shift:+.2f}", fontsize=11)
        ax.text(0.95, 0.05, f"c:{ct_n_cell[ct]} n:{ct_n_nuc[ct]}", transform=ax.transAxes,
                fontsize=8, ha="right", va="bottom", color="gray")

        if i % ncols == 0:
            ax.set_ylabel("Nuclei mean", fontsize=10)
        if i // ncols == nrows - 1:
            ax.set_xlabel("Cell mean", fontsize=10)

        summary.append({"cell_type": ct, "pearson_r": r, "median_log2fc": median_shift,
                         "n_cell": ct_n_cell[ct], "n_nuc": ct_n_nuc[ct]})

    for j in range(n_cts, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle("Cell vs Nuclei mean expression by cell type", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    out1 = os.path.join(OUTPUT_DIR, "slide6_ct_gap_scatter.png")
    fig.savefig(out1, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out1}")

    # --- Figure 2: Summary bar chart of Pearson r per CT ---
    df = pd.DataFrame(summary)
    df = df.sort_values("pearson_r", ascending=True)

    glial = {"Astro", "Oligo", "OPC"}
    bar_colors = ["#2ca02c" if ct in glial else "#1f77b4" for ct in df["cell_type"]]

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.barh(range(len(df)), df["pearson_r"].values, color=bar_colors, edgecolor="white")
    ax2.set_yticks(range(len(df)))
    ax2.set_yticklabels(df["cell_type"].values, fontsize=11)
    ax2.set_xlabel("Pearson r (log-scale cell vs nuclei mean expression)", fontsize=12)
    ax2.set_title("Cell-nucleus correlation by cell type", fontsize=14, fontweight="bold")
    ax2.set_xlim(0.4, 1.0)

    from matplotlib.patches import Patch
    ax2.legend(handles=[Patch(facecolor="#1f77b4", label="Neuronal"),
                        Patch(facecolor="#2ca02c", label="Glial")],
               loc="upper left", fontsize=10)

    fig2.tight_layout()
    out2 = os.path.join(OUTPUT_DIR, "slide6_ct_gap_bar.png")
    fig2.savefig(out2, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {out2}")

    # Print summary
    print("\n--- Summary ---")
    for _, row in df.iterrows():
        print(f"  {row['cell_type']:12s}  r={row['pearson_r']:.3f}  "
              f"median_log2FC={row['median_log2fc']:+.3f}  "
              f"(c:{row['n_cell']:,} n:{row['n_nuc']:,})")


if __name__ == "__main__":
    main()
