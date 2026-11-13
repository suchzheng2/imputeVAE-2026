"""Compare per-gene expression between cell and nuclei (v3 only, after Config B filtering).

Uses the same gene set as training: common genes across modalities, excluding
MitoCarta genes, with Config B detection rate filtering.
"""

import os

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp

DATA_ROOT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "analysis", "filtering_cutoffs_v3")

DATASETS = {
    "cell": ("10X_cells_v2_v3_AIBS", "c3A"),
    "nuclei": ("10X_nuclei_v2_v3_AIBS", "n3A"),
}

# Config B filtering thresholds
DET_RATE_THRESH = 0.05   # >5% detection rate (applied per-modality, then union)
MEAN_THRESH = 0.01       # mean expression > 0.01
MIN_LIB_SIZE = 2000
MIN_GENES = 1500


def load_common_genes():
    """Load the common gene set used in training."""
    path = os.path.join(PROJECT_ROOT, "data", "common_genes.csv")
    df = pd.read_csv(path, header=0)
    genes = set(df["common_genes"].tolist())
    print(f"Common genes from training: {len(genes)}")
    return genes


def load_mitocarta_genes():
    """Load MitoCarta gene symbols to exclude."""
    path = os.path.join(PROJECT_ROOT, "data", "Mouse.MitoCarta.csv")
    df = pd.read_csv(path, header=0)
    genes = set(df["Symbol"].tolist())
    print(f"MitoCarta genes: {len(genes)}")
    return genes


def load_v3_gene_stats(dataset_name, v3_suffix):
    """Load v3 data and compute per-gene stats. Returns DataFrame indexed by gene name."""
    path = os.path.join(DATA_ROOT, dataset_name, "data.h5ad")
    print(f"Loading {path}...")
    adata = ad.read_h5ad(path, backed="r")

    # Get gene names from var
    if "gene_names" in adata.var.columns:
        gene_names = np.array(adata.var["gene_names"])
    else:
        gene_names = np.array(adata.var_names)

    meta_batch = adata.obs["batch_id"].apply(lambda x: x.split(".")[0])
    v3_mask = meta_batch.str.endswith(v3_suffix).values
    v3_indices = np.where(v3_mask)[0]
    print(f"  Total: {adata.shape[0]}, v3: {len(v3_indices)}")

    n_v3 = len(v3_indices)
    n_genes = adata.shape[1]

    gene_sums = np.zeros(n_genes, dtype=np.float64)
    gene_nonzero = np.zeros(n_genes, dtype=np.int64)
    gene_sumsq = np.zeros(n_genes, dtype=np.float64)
    lib_sizes = np.zeros(n_v3, dtype=np.float64)
    genes_per_cell = np.zeros(n_v3, dtype=np.int64)

    chunk_size = 5000
    for start in range(0, n_v3, chunk_size):
        end = min(start + chunk_size, n_v3)
        idx = v3_indices[start:end]
        X_chunk = adata.X[idx]
        if sp.issparse(X_chunk):
            X_chunk = X_chunk.toarray()
        else:
            X_chunk = np.asarray(X_chunk)

        lib_sizes[start:end] = X_chunk.sum(axis=1)
        genes_per_cell[start:end] = (X_chunk > 0).sum(axis=1)
        gene_sums += X_chunk.sum(axis=0)
        gene_nonzero += (X_chunk > 0).sum(axis=0)
        gene_sumsq += (X_chunk ** 2).sum(axis=0)

        if start % 20000 == 0:
            print(f"  processed {start}/{n_v3}")

    adata.file.close()

    cell_mask = (lib_sizes >= MIN_LIB_SIZE) & (genes_per_cell >= MIN_GENES)
    n_pass = cell_mask.sum()
    print(f"  Cells passing QC: {n_pass}/{n_v3} ({100*n_pass/n_v3:.1f}%)")

    gene_means = gene_sums / n_v3
    detection_rate = gene_nonzero / n_v3
    gene_var = gene_sumsq / n_v3 - gene_means ** 2

    return pd.DataFrame({
        "gene_name": gene_names,
        "mean": gene_means,
        "det_rate": detection_rate,
        "nonzero_count": gene_nonzero,
        "variance": gene_var,
        "n_cells": n_v3,
    }).set_index("gene_name")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load gene filters
    common_genes = load_common_genes()
    mitocarta_genes = load_mitocarta_genes()

    # Load per-gene stats for both modalities
    stats = {}
    for modality, (dataset, suffix) in DATASETS.items():
        stats[modality] = load_v3_gene_stats(dataset, suffix)

    cell = stats["cell"]
    nuclei = stats["nuclei"]

    # Step 1: Restrict to common genes (same as training)
    cell_in_common = cell.index.isin(common_genes)
    nuclei_in_common = nuclei.index.isin(common_genes)
    cell = cell[cell_in_common]
    nuclei = nuclei[nuclei_in_common]
    print(f"\nAfter common gene filter: cell={len(cell)}, nuclei={len(nuclei)}")

    # Step 2: Intersect (genes present in both modalities)
    shared_genes = cell.index.intersection(nuclei.index)
    print(f"Shared genes: {len(shared_genes)}")
    cell = cell.loc[shared_genes]
    nuclei = nuclei.loc[shared_genes]

    # Step 3: Exclude MitoCarta genes
    mito_mask = shared_genes.isin(mitocarta_genes)
    n_mito = mito_mask.sum()
    print(f"MitoCarta genes in shared set: {n_mito}")
    cell = cell[~mito_mask]
    nuclei = nuclei[~mito_mask]
    shared_genes = cell.index
    print(f"After MitoCarta exclusion: {len(shared_genes)} genes")

    # Build comparison dataframe
    df = pd.DataFrame({
        "cell_mean": cell["mean"],
        "cell_det": cell["det_rate"],
        "cell_nonzero": cell["nonzero_count"],
        "cell_var": cell["variance"],
        "nuclei_mean": nuclei["mean"],
        "nuclei_det": nuclei["det_rate"],
        "nuclei_nonzero": nuclei["nonzero_count"],
        "nuclei_var": nuclei["variance"],
    }, index=shared_genes)

    # Apply Config B gene filter: det>5% in EITHER modality AND mean>0.01 in EITHER
    gene_mask = (
        ((df["cell_det"] >= DET_RATE_THRESH) | (df["nuclei_det"] >= DET_RATE_THRESH))
        & ((df["cell_mean"] >= MEAN_THRESH) | (df["nuclei_mean"] >= MEAN_THRESH))
    )
    df_filt = df[gene_mask].copy()
    print(f"Genes after Config B filter (union): {len(df_filt)}/{len(df)}")

    # Compute difference metrics
    df_filt["det_diff"] = df_filt["cell_det"] - df_filt["nuclei_det"]
    df_filt["log_mean_ratio"] = np.log2(
        (df_filt["cell_mean"] + 1e-6) / (df_filt["nuclei_mean"] + 1e-6)
    )
    df_filt["mean_diff"] = df_filt["cell_mean"] - df_filt["nuclei_mean"]

    # Sort by absolute detection rate difference
    df_filt = df_filt.sort_values("det_diff", key=abs, ascending=False)

    # Save full table
    csv_path = os.path.join(OUTPUT_DIR, "cell_vs_nuclei_gene_diff_v3.csv")
    df_filt.to_csv(csv_path, float_format="%.6f")
    print(f"Saved: {csv_path}")

    # ── Print top genes ──
    print("\n" + "=" * 80)
    print("TOP 30 GENES: Higher in CELL than NUCLEI (by detection rate difference)")
    print("=" * 80)
    top_cell = df_filt[df_filt["det_diff"] > 0].head(30)
    for gene, row in top_cell.iterrows():
        print(f"  {gene:15s}  cell_det={row['cell_det']:.3f}  nuc_det={row['nuclei_det']:.3f}  "
              f"diff={row['det_diff']:+.3f}  cell_mean={row['cell_mean']:.2f}  nuc_mean={row['nuclei_mean']:.2f}  "
              f"log2FC={row['log_mean_ratio']:+.2f}")

    print("\n" + "=" * 80)
    print("TOP 30 GENES: Higher in NUCLEI than CELL (by detection rate difference)")
    print("=" * 80)
    top_nuc = df_filt[df_filt["det_diff"] < -0.01].sort_values("det_diff").head(30)
    for gene, row in top_nuc.iterrows():
        print(f"  {gene:15s}  cell_det={row['cell_det']:.3f}  nuc_det={row['nuclei_det']:.3f}  "
              f"diff={row['det_diff']:+.3f}  cell_mean={row['cell_mean']:.2f}  nuc_mean={row['nuclei_mean']:.2f}  "
              f"log2FC={row['log_mean_ratio']:+.2f}")

    # ── PLOT 1: Detection rate scatter (cell vs nuclei) ──
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = np.where(
        abs(df_filt["det_diff"]) > 0.15, "crimson",
        np.where(abs(df_filt["det_diff"]) > 0.05, "darkorange", "steelblue")
    )
    ax.scatter(df_filt["cell_det"] * 100, df_filt["nuclei_det"] * 100,
               c=colors, s=4, alpha=0.5, edgecolors="none")
    ax.plot([0, 100], [0, 100], "k--", alpha=0.3, linewidth=1)
    ax.set_xlabel("Cell detection rate (%)")
    ax.set_ylabel("Nuclei detection rate (%)")
    ax.set_title("Gene detection rate: Cell vs Nuclei\n(v3, common genes, no MitoCarta, Config B)")

    extreme = df_filt[abs(df_filt["det_diff"]) > 0.50].head(25)
    for gene, row in extreme.iterrows():
        ax.annotate(gene, (row["cell_det"] * 100, row["nuclei_det"] * 100),
                     fontsize=5.5, alpha=0.8,
                     xytext=(3, 3), textcoords="offset points")

    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "cell_vs_nuclei_det_rate_scatter.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out}")

    # ── PLOT 2: Mean expression scatter (log scale) ──
    fig, ax = plt.subplots(figsize=(8, 8))
    cm = df_filt["cell_mean"].values + 1e-6
    nm = df_filt["nuclei_mean"].values + 1e-6
    colors2 = np.where(
        abs(df_filt["log_mean_ratio"]) > 3, "crimson",
        np.where(abs(df_filt["log_mean_ratio"]) > 1, "darkorange", "steelblue")
    )
    ax.scatter(cm, nm, c=colors2, s=4, alpha=0.5, edgecolors="none")
    lims = [1e-4, max(cm.max(), nm.max()) * 2]
    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Cell mean expression")
    ax.set_ylabel("Nuclei mean expression")
    ax.set_title("Gene mean expression: Cell vs Nuclei\n(v3, common genes, no MitoCarta, Config B)")

    extreme_mean = df_filt.reindex(df_filt["log_mean_ratio"].abs().sort_values(ascending=False).index).head(25)
    for gene, row in extreme_mean.iterrows():
        ax.annotate(gene, (row["cell_mean"] + 1e-6, row["nuclei_mean"] + 1e-6),
                     fontsize=5.5, alpha=0.8,
                     xytext=(3, 3), textcoords="offset points")

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "cell_vs_nuclei_mean_expr_scatter.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")

    # ── PLOT 3: MA plot (log2 fold change vs average expression) ──
    fig, ax = plt.subplots(figsize=(12, 6))
    avg_expr = 0.5 * (np.log2(cm) + np.log2(nm))
    log2fc = df_filt["log_mean_ratio"].values
    colors3 = np.where(
        abs(log2fc) > 3, "crimson",
        np.where(abs(log2fc) > 1, "darkorange", "steelblue")
    )
    ax.scatter(avg_expr, log2fc, c=colors3, s=4, alpha=0.5, edgecolors="none")
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.axhline(1, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axhline(-1, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_xlabel("Average log2(mean expression)")
    ax.set_ylabel("log2(cell / nuclei)")
    ax.set_title("MA plot: Cell vs Nuclei fold change\n(v3, common genes, no MitoCarta, Config B)")

    extreme_ma = df_filt.iloc[np.argsort(-abs(log2fc))[:25]]
    for gene, row in extreme_ma.iterrows():
        a = 0.5 * (np.log2(row["cell_mean"] + 1e-6) + np.log2(row["nuclei_mean"] + 1e-6))
        fc = row["log_mean_ratio"]
        ax.annotate(gene, (a, fc), fontsize=5.5, alpha=0.8,
                     xytext=(3, 3), textcoords="offset points")

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "cell_vs_nuclei_MA_plot.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")

    # ── PLOT 4: Detection rate difference histogram ──
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df_filt["det_diff"] * 100, bins=100, color="steelblue",
            edgecolor="none", alpha=0.8)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Detection rate difference (cell - nuclei, percentage points)")
    ax.set_ylabel("Number of genes")
    ax.set_title("Distribution of cell-nuclei detection rate difference\n(v3, common genes, no MitoCarta, Config B)")

    n_cell_higher = (df_filt["det_diff"] > 0.05).sum()
    n_nuc_higher = (df_filt["det_diff"] < -0.05).sum()
    n_similar = len(df_filt) - n_cell_higher - n_nuc_higher
    ax.text(0.02, 0.95, f"Cell-enriched (>5pp): {n_cell_higher}\n"
            f"Nuclei-enriched (>5pp): {n_nuc_higher}\n"
            f"Similar (<5pp): {n_similar}",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "cell_vs_nuclei_det_diff_hist.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")

    # ── PLOT 5: Quadrant classification ──
    fig, ax = plt.subplots(figsize=(10, 8))

    cell_detected = df_filt["cell_det"] >= DET_RATE_THRESH
    nuc_detected = df_filt["nuclei_det"] >= DET_RATE_THRESH

    categories = pd.Series("shared", index=df_filt.index)
    categories[cell_detected & ~nuc_detected] = "cell-specific"
    categories[~cell_detected & nuc_detected] = "nuclei-specific"
    categories[cell_detected & nuc_detected & (abs(df_filt["det_diff"]) > 0.15)] = "divergent"

    color_map = {
        "shared": "steelblue",
        "cell-specific": "crimson",
        "nuclei-specific": "forestgreen",
        "divergent": "darkorange",
    }
    for cat, color in color_map.items():
        mask = categories == cat
        ax.scatter(
            df_filt.loc[mask, "cell_det"] * 100,
            df_filt.loc[mask, "nuclei_det"] * 100,
            c=color, s=6 if cat != "shared" else 3,
            alpha=0.6 if cat != "shared" else 0.3,
            edgecolors="none", label=f"{cat} (n={mask.sum()})",
        )

    ax.plot([0, 100], [0, 100], "k--", alpha=0.3, linewidth=1)
    ax.axvline(DET_RATE_THRESH * 100, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(DET_RATE_THRESH * 100, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Cell detection rate (%)")
    ax.set_ylabel("Nuclei detection rate (%)")
    ax.set_title("Gene classification: Cell vs Nuclei\n(v3, common genes, no MitoCarta, Config B)")
    ax.legend(fontsize=9, loc="upper left")

    # Label top cell-specific and nuclei-specific genes by det_diff
    for cat in ["cell-specific", "nuclei-specific"]:
        subset = df_filt[categories == cat].sort_values("det_diff", key=abs, ascending=False).head(10)
        for gene, row in subset.iterrows():
            ax.annotate(gene, (row["cell_det"] * 100, row["nuclei_det"] * 100),
                         fontsize=5.5, alpha=0.8,
                         xytext=(3, 3), textcoords="offset points")

    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "cell_vs_nuclei_gene_classification.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")

    # ── Summary stats ──
    print("\n" + "=" * 80)
    print("GENE CLASSIFICATION SUMMARY")
    print(f"(common genes, no MitoCarta, Config B filter, det>5% threshold)")
    print("=" * 80)
    for cat in ["shared", "divergent", "cell-specific", "nuclei-specific"]:
        n = (categories == cat).sum()
        print(f"  {cat:20s}: {n:5d} genes ({100*n/len(df_filt):.1f}%)")

    print("\n--- Cell-specific genes (det>5% in cell, <5% in nuclei) ---")
    cs = df_filt[categories == "cell-specific"]
    if len(cs) > 0:
        print(f"  Count: {len(cs)}")
        print(f"  Cell det rate: median={cs['cell_det'].median():.3f}, mean={cs['cell_det'].mean():.3f}")
        print(f"  Nuclei det rate: median={cs['nuclei_det'].median():.3f}, mean={cs['nuclei_det'].mean():.3f}")
        print(f"  Cell mean expr: median={cs['cell_mean'].median():.3f}")
        print(f"  Nuclei mean expr: median={cs['nuclei_mean'].median():.3f}")
        print(f"  Top 10 by det_diff:")
        for gene, row in cs.head(10).iterrows():
            print(f"    {gene:15s}  cell_det={row['cell_det']:.3f}  nuc_det={row['nuclei_det']:.3f}  "
                  f"cell_mean={row['cell_mean']:.2f}  nuc_mean={row['nuclei_mean']:.2f}")

    print("\n--- Nuclei-specific genes (det>5% in nuclei, <5% in cell) ---")
    ns = df_filt[categories == "nuclei-specific"]
    if len(ns) > 0:
        print(f"  Count: {len(ns)}")
        print(f"  Nuclei det rate: median={ns['nuclei_det'].median():.3f}, mean={ns['nuclei_det'].mean():.3f}")
        print(f"  Cell det rate: median={ns['cell_det'].median():.3f}, mean={ns['cell_det'].mean():.3f}")
        print(f"  Top 10 by det_diff:")
        for gene, row in ns.sort_values("det_diff").head(10).iterrows():
            print(f"    {gene:15s}  cell_det={row['cell_det']:.3f}  nuc_det={row['nuclei_det']:.3f}  "
                  f"nuc_mean={row['nuclei_mean']:.2f}  cell_mean={row['cell_mean']:.2f}")

    print("\n--- Divergent genes (detected in both but |diff|>15pp) ---")
    dv = df_filt[categories == "divergent"]
    if len(dv) > 0:
        print(f"  Count: {len(dv)}")
        print(f"  Cell-biased: {(dv['det_diff'] > 0).sum()}")
        print(f"  Nuclei-biased: {(dv['det_diff'] < 0).sum()}")

    print("\nDone.")


if __name__ == "__main__":
    main()
