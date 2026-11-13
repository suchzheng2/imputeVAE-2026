#!/usr/bin/env python3
"""Per-gene value distributions for a few L5 IT genes: real cell vs raw nucleus vs imputed.

Companion to analysis/variance_rank_check.py (Prashant, Sep 2026). Shows what the per-gene variance numbers
look like at the single-cell level: overlaid histograms of log1p values for real v3 cells, raw v3 nuclei and
the full-data cVAE imputation of those nuclei, within one cell type.

Inputs: raw h5ad (cells c3A, nuclei n3A), run/cvae_full_eval/imputed_cell_from_nucleus.npy (log1p scale, rows
        aligned to nuclei filtered as in variance_zero_nonzero_fullmodel.py), per-gene variance table.
Output: notes/experiments/variance_zero_nonzero_fullmodel/variance_rank/figures/V5_gene_examples_<CT>.{png,pdf,_data.csv}
"""
import argparse
import os
import numpy as np
import pandas as pd
import anndata as ad
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE"
DATA_ROOT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi"
FIG = f"{PROJECT}/notes/experiments/variance_zero_nonzero_fullmodel/variance_rank/figures"
SRC = {"cell": "#E69F00", "nucleus": "#7F7F7F", "imputed": "#0072B2"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#E5E5E5", "grid.linewidth": 0.6, "axes.axisbelow": True, "pdf.fonttype": 42})

DEFAULT_GENES = ["Pcp4", "Rplp1", "Rorb", "Calm2", "Malat1", "Xist", "Gm49164", "Gfra1"]
LABELS = {"Pcp4": "cytoplasmic, absent in nuclei", "Rplp1": "ribosomal, absent in nuclei",
          "Rorb": "L5 IT marker, well detected", "Calm2": "housekeeping, >70% bin",
          "Malat1": "nuclear lncRNA", "Xist": "top real-cell variance, imputed collapses",
          "Gm49164": "well detected, variance ratio 0.05", "Gfra1": "sparse (<10%), variance ratio 0.07"}


def gene_index(a, gene_list, common):
    """Columns kept and their order, exactly as in variance_zero_nonzero_fullmodel.load_counts."""
    cm = (a.var["gene_names"].isin(common) & a.var["gene_names"].isin(gene_list)).values
    idx = np.where(cm)[0]
    names = a.var["gene_names"].values[idx]
    order = np.argsort(names, kind="stable")
    return idx[order], names[order]


def load_subset(data_dir, batch, ct, cts, gene_list, common, genes):
    a = ad.read_h5ad(f"{DATA_ROOT}/{data_dir}/data.h5ad", backed="r")
    mb = a.obs["batch_id"].astype(str).str.split(".").str[0]
    rm = (mb.str.endswith(batch) & a.obs["cell_type"].isin(cts)).values
    kept_rows = np.where(rm)[0]
    ct_pos = np.where(a.obs["cell_type"].astype(str).values[kept_rows] == ct)[0]   # position among kept rows
    col_idx, names = gene_index(a, gene_list, common)
    gpos = np.array([int(np.where(names == g)[0][0]) for g in genes])
    a.file.close()
    # X is a dense int64 h5py dataset (cells: ~41 GB); stream row chunks and keep only the wanted columns
    rows = kept_rows[ct_pos]; cols = col_idx[gpos]
    X = np.zeros((len(rows), len(cols)), dtype=np.float32)
    with h5py.File(f"{DATA_ROOT}/{data_dir}/data.h5ad", "r") as f:
        D = f["X"]; step = 4000
        for lo in range(int(rows.min()), int(rows.max()) + 1, step):
            hi = min(lo + step, int(rows.max()) + 1)
            sel = np.where((rows >= lo) & (rows < hi))[0]
            if len(sel) == 0:
                continue
            block = D[lo:hi][:, cols]
            X[sel] = block[rows[sel] - lo]
    return np.log1p(X), ct_pos, gpos, len(kept_rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cell_type", default="L5 IT")
    p.add_argument("--genes", nargs="*", default=DEFAULT_GENES)
    args = p.parse_args()
    ct, genes = args.cell_type, args.genes
    gene_list = pd.read_csv(f"{PROJECT}/data/training_gene_list_v3.csv", header=None)[0].tolist()
    common = pd.read_csv(f"{PROJECT}/data/common_genes.csv")["common_genes"].tolist()
    cts = pd.read_csv(f"{PROJECT}/data/common_cell_types.csv")["common_cell_types"].tolist()

    Ln, npos, gpos, n_kept = load_subset("10X_nuclei_v2_v3_AIBS", "n3A", ct, cts, gene_list, common, genes)
    Lc, _, gpos_c, _ = load_subset("10X_cells_v2_v3_AIBS", "c3A", ct, cts, gene_list, common, genes)
    assert (gpos == gpos_c).all()
    I = np.load(f"{PROJECT}/run/cvae_full_eval/imputed_cell_from_nucleus.npy", mmap_mode="r")
    assert I.shape[0] == n_kept, (I.shape, n_kept)
    ict = pd.read_csv(f"{PROJECT}/run/cvae_full_eval/nucleus_obs.csv")["cell_type"].astype(str).values
    assert (ict[npos] == ct).all()
    Li = np.asarray(I[np.ix_(npos, gpos)], dtype=np.float32)
    print(f"{ct}: {Lc.shape[0]} cells, {Ln.shape[0]} nuclei, {Li.shape[0]} imputed; genes {genes}", flush=True)

    T = pd.read_csv(f"{PROJECT}/notes/experiments/variance_zero_nonzero_fullmodel/tables/per_ct_gene_variance.csv.gz")
    T = T[T.cell_type == ct].set_index("gene")

    ncol = 4; nrow = int(np.ceil(len(genes) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.0 * nrow)); axes = np.atleast_1d(axes).ravel()
    rows = []
    for k, g in enumerate(genes):
        ax = axes[k]
        vals = {"cell": Lc[:, k], "nucleus": Ln[:, k], "imputed": Li[:, k]}
        hi = max(v.max() for v in vals.values()); bins = np.linspace(0, hi * 1.02, 45)
        for name, v in vals.items():
            ax.hist(v, bins=bins, density=True, histtype="step", lw=1.8, color=SRC[name], label=name)
            ax.hist(v, bins=bins, density=True, histtype="stepfilled", alpha=0.12, color=SRC[name])
            rows.append(dict(cell_type=ct, gene=g, source=name, n=len(v), mean=float(v.mean()), var=float(v.var()),
                             frac_zero=float((v <= 0).mean()), q05=float(np.percentile(v, 5)), q50=float(np.median(v)),
                             q95=float(np.percentile(v, 95))))
        vc, vn, vi = (float(vals[s].var()) for s in ("cell", "nucleus", "imputed"))
        dc, dn = float((vals["cell"] > 0).mean()), float((vals["nucleus"] > 0).mean())
        ax.text(0.98, 0.97, f"var  cell {vc:.2f}  nuc {vn:.2f}  imp {vi:.2f}\n"
                            f"imp/cell {vi / vc:.2f}\n"
                            f"det  cell {dc:.0%}  nuc {dn:.0%}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7, family="monospace",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#CCCCCC", lw=0.6))
        ax.set_title(f"{g}\n{LABELS.get(g, '')}", fontsize=8.5)
        ax.set_xlabel("log1p expression"); ax.set_ylabel("density" if k % ncol == 0 else "")
        # clip the y axis so the zero spike does not flatten everything else
        ymax = max(np.histogram(v[v > 0], bins=bins, density=False)[0].max() / (len(v) * (bins[1] - bins[0]))
                   for v in vals.values() if (v > 0).sum() > 10)
        ax.set_ylim(0, ymax * 1.35)
        if k == 0:
            ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    for ax in axes[len(genes):]:
        ax.axis("off")
    fig.suptitle(f"{ct}: per-cell distribution of log1p values, real cells (n={Lc.shape[0]}), raw nuclei (n={Ln.shape[0]}), "
                 f"imputed from those nuclei. y axis clipped so the zero spike does not dominate.", fontsize=9, y=1.01)
    fig.tight_layout()
    tag = ct.replace("/", "").replace(" ", "")
    fig.savefig(f"{FIG}/V5_gene_examples_{tag}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{FIG}/V5_gene_examples_{tag}.pdf", bbox_inches="tight")
    pd.DataFrame(rows).to_csv(f"{FIG}/V5_gene_examples_{tag}_data.csv", index=False)
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).pivot(index="gene", columns="source", values=["var", "frac_zero", "mean"]).round(3).loc[genes].to_string())
    print("saved", f"{FIG}/V5_gene_examples_{tag}.png")


if __name__ == "__main__":
    main()
