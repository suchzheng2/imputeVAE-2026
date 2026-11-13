"""Quick check: are the bimodal distributions driven by v2 vs v3 chemistry?"""
import re
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

DATA_ROOT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi"
OUTPUT_DIR = "analysis/filtering_cutoffs"

DATASETS = {
    "cell": ("10X_cells_v2_v3_AIBS", r"c(\d)A"),
    "nuclei": ("10X_nuclei_v2_v3_AIBS", r"n(\d)A"),
}

for modality, (dataset, pattern) in DATASETS.items():
    print(f"\n=== {modality} ===")
    path = f"{DATA_ROOT}/{dataset}/data.h5ad"
    adata = ad.read_h5ad(path, backed="r")

    # Parse chemistry version from batch_id
    chemistry = adata.obs["batch_id"].apply(lambda x: "v" + re.search(pattern, x).group(1))
    adata.obs["chemistry"] = chemistry

    # Compute library sizes from sparse matrix (chunked to avoid OOM)
    n_cells = adata.shape[0]
    chunk_size = 10000
    lib_sizes = np.zeros(n_cells, dtype=np.float64)
    genes_per_cell = np.zeros(n_cells, dtype=np.int64)
    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)
        X_chunk = adata.X[start:end]
        if sp.issparse(X_chunk):
            X_chunk = X_chunk.toarray()
        lib_sizes[start:end] = X_chunk.sum(axis=1)
        genes_per_cell[start:end] = (X_chunk > 0).sum(axis=1)
        if start % 50000 == 0:
            print(f"  processed {start}/{n_cells}")

    chem = adata.obs["chemistry"].values

    # Print stats
    for v in ["v2", "v3"]:
        mask = chem == v
        print(f"  {v}: n={mask.sum()}, lib_size median={np.median(lib_sizes[mask]):.0f}, "
              f"genes median={np.median(genes_per_cell[mask]):.0f}")

    # Plot: library size colored by chemistry
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for v, color in [("v2", "steelblue"), ("v3", "coral")]:
        mask = chem == v
        ax.hist(np.log10(lib_sizes[mask] + 1), bins=80, alpha=0.6, color=color,
                label=f"{v} (n={mask.sum():,}, med={np.median(lib_sizes[mask]):.0f})",
                edgecolor="none")
    ax.set_xlabel("log10(total counts + 1)")
    ax.set_ylabel("Number of cells")
    ax.set_title(f"{modality}: Library size by chemistry")
    ax.legend(fontsize=9)

    ax = axes[1]
    for v, color in [("v2", "steelblue"), ("v3", "coral")]:
        mask = chem == v
        ax.hist(genes_per_cell[mask], bins=80, alpha=0.6, color=color,
                label=f"{v} (n={mask.sum():,}, med={np.median(genes_per_cell[mask]):.0f})",
                edgecolor="none")
    ax.set_xlabel("Number of expressed genes")
    ax.set_ylabel("Number of cells")
    ax.set_title(f"{modality}: Genes expressed by chemistry")
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = f"{OUTPUT_DIR}/{modality}_v2_v3_batch_effect.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")

    adata.file.close()

print("\nDone.")
