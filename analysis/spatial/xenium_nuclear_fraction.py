"""Per-gene and per-cell nuclear fraction from the Xenium per-molecule table (2026-11-13).

transcripts.parquet (188.5 M rows, 720 row groups) carries `overlaps_nucleus` per molecule. For every gene this gives
the in-situ nuclear fraction of its transcripts, the quantity our model's cell-minus-nucleus delta predicts (a gene
with a large positive delta should have a low nuclear fraction). Aggregated by row group so memory stays small.

Outputs (notes/experiments/spatial_merfish_application/scoping/):
  gene_nuclear_fraction.csv   gene, n_molecules, n_nuclear, nuclear_fraction, n_cells_detected (q20 molecules, assigned to a cell)
  cell_nuclear_fraction.csv   cell_id, n_molecules, n_nuclear, nuclear_fraction
  gene_nuclear_fraction_by_region.csv  same per gene for cortex-only cells (region from the scoping clusters) -- to add once
                                       the cortex mask exists; for now all assigned cells.
"""
import argparse, os, time
import numpy as np, pandas as pd, pyarrow.parquet as pq

XEN = "/vast/palmer/scratch/gerstein/sz477/xenium_prime5k_mouse_brain"
OUT = "notes/experiments/spatial_merfish_application/scoping"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=OUT); ap.add_argument("--qv_min", type=float, default=20.0)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    pf = pq.ParquetFile(f"{XEN}/transcripts.parquet")
    gene_tot, gene_nuc, cell_tot, cell_nuc = {}, {}, {}, {}
    gene_cells = {}
    t0 = time.time()
    for rg in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(rg, columns=["cell_id", "overlaps_nucleus", "feature_name", "qv", "is_gene"]).to_pandas()
        t = t[t.is_gene & (t.qv >= a.qv_min) & (t.cell_id != "UNASSIGNED")]
        g = t.groupby("feature_name", observed=True)["overlaps_nucleus"].agg(["size", "sum"])
        for k, (n, s) in g.iterrows():
            gene_tot[k] = gene_tot.get(k, 0) + int(n); gene_nuc[k] = gene_nuc.get(k, 0) + int(s)
        c = t.groupby("cell_id", observed=True)["overlaps_nucleus"].agg(["size", "sum"])
        for k, (n, s) in c.iterrows():
            cell_tot[k] = cell_tot.get(k, 0) + int(n); cell_nuc[k] = cell_nuc.get(k, 0) + int(s)
        gc = t.groupby("feature_name", observed=True)["cell_id"].nunique()  # upper bound across row groups (cells span fovs rarely)
        for k, n in gc.items():
            gene_cells[k] = gene_cells.get(k, 0) + int(n)
        if rg % 50 == 0:
            print(f"row group {rg}/{pf.metadata.num_row_groups}  {time.time()-t0:.0f}s  genes {len(gene_tot)} cells {len(cell_tot)}", flush=True)
    gdf = pd.DataFrame({"n_molecules": pd.Series(gene_tot), "n_nuclear": pd.Series(gene_nuc), "n_cells_detected_ub": pd.Series(gene_cells)})
    gdf["nuclear_fraction"] = gdf.n_nuclear / gdf.n_molecules
    gdf.sort_values("nuclear_fraction").to_csv(f"{a.out}/gene_nuclear_fraction.csv", index_label="gene")
    cdf = pd.DataFrame({"n_molecules": pd.Series(cell_tot), "n_nuclear": pd.Series(cell_nuc)})
    cdf["nuclear_fraction"] = cdf.n_nuclear / cdf.n_molecules
    cdf.to_csv(f"{a.out}/cell_nuclear_fraction.csv", index_label="cell_id")
    print("genes", len(gdf), "cells", len(cdf), "overall nuclear fraction", round(gdf.n_nuclear.sum() / gdf.n_molecules.sum(), 3))
    print(gdf.sort_values("nuclear_fraction").head(10)); print(gdf.sort_values("nuclear_fraction").tail(10))


if __name__ == "__main__":
    main()
