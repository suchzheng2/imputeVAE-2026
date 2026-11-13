"""
Task 14 — Lake 2023 kidney: conserved-transformation test.

Tests whether the nucleus→cell shift learned by the model in mouse brain
is a conserved biological axis, using an independent matched human kidney dataset.
NO model inference — purely empirical delta comparison.

Data: Lake et al. 2023, human kidney, 10x 3' v3, donor-paired.
  - Cells: CELLxGENE 08648660-9898-4644-a9a1-ff3a159a0041.h5ad
  - Nuclei: CELLxGENE 02d5db96-d67e-45f2-b556-b6bd99665.h5ad

Usage:
    python analysis/conserved_transformation.py \
        --cell_h5ad data/external/lake2023_cells.h5ad \
        --nuc_h5ad data/external/lake2023_nuclei.h5ad \
        --output_dir notes/experiments/conserved_transformation
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse, stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Mouse brain delta from our model
MOUSE_DELTA_PATH = "notes/experiments/nuclear_lncrna_analysis/all_genes_delta.csv"

# Known compartment genes (human symbols)
RIBOSOMAL_PREFIXES = ("RPL", "RPS")
NUCLEAR_LNCRNAS = ["MALAT1", "NEAT1", "XIST", "MEG3", "KCNQ1OT1", "SNHG11"]


def load_and_normalize(h5ad_path, max_cells=None):
    """Load h5ad, extract raw counts, apply log1p normalization."""
    print(f"Loading {h5ad_path}...")
    # Load fully into memory (backed mode doesn't support raw.X materialization)
    adata = ad.read_h5ad(h5ad_path)
    print(f"  Shape: {adata.shape}")
    print(f"  obs columns: {list(adata.obs.columns[:20])}")

    # Use raw counts; resolve gene symbols from feature_name if available
    if adata.raw is not None:
        print("  Using adata.raw.X")
        X = adata.raw.X
        if "feature_name" in adata.raw.var.columns:
            var_names = list(adata.raw.var["feature_name"].values)
            print(f"  Mapped Ensembl IDs -> gene symbols via feature_name")
        else:
            var_names = list(adata.raw.var_names)
    else:
        print("  Using adata.X")
        X = adata.X
        if "feature_name" in adata.var.columns:
            var_names = list(adata.var["feature_name"].values)
        else:
            var_names = list(adata.var_names)

    obs = adata.obs.copy()
    del adata  # free memory

    if max_cells and X.shape[0] > max_cells:
        print(f"  Subsampling to {max_cells} cells...")
        idx = np.random.choice(X.shape[0], max_cells, replace=False)
        idx.sort()
        X = X[idx]
        obs = obs.iloc[idx].copy()

    if sparse.issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)

    # log1p normalize (same as our training pipeline)
    X = np.log1p(X)

    print(f"  Final: {X.shape}, log1p range [{X.min():.2f}, {X.max():.2f}]")
    return X, obs, var_names


def human_to_mouse_symbol(human_symbol):
    """Convert human gene symbol to mouse equivalent.

    Mouse gene symbols follow Title Case convention for most genes:
    ACTB -> Actb, RPL5 -> Rpl5, MALAT1 -> Malat1.
    Exceptions exist but this covers >90% of protein-coding genes.
    """
    if not human_symbol or not isinstance(human_symbol, str):
        return None
    # Skip Ensembl IDs that weren't resolved
    if human_symbol.startswith("ENSG"):
        return None
    # Mouse: first letter uppercase, rest lowercase
    return human_symbol[0].upper() + human_symbol[1:].lower()


def get_ortholog_mapping(human_genes, mouse_genes):
    """Build human->mouse mapping using symbol capitalization convention,
    validated against actual mouse gene list."""
    mouse_set = set(mouse_genes)
    mapping = {}
    for hg in human_genes:
        mg = human_to_mouse_symbol(hg)
        if mg and mg in mouse_set:
            mapping[hg] = mg
    print(f"  Mapped {len(mapping)}/{len(human_genes)} human genes to mouse symbols "
          f"(validated against {len(mouse_set)} mouse genes)")
    return mapping


def compute_empirical_delta(X_cell, X_nuc, obs_cell, obs_nuc, var_names, ct_col):
    """Compute empirical delta = mean(cell) - mean(nucleus) per gene, globally and per CT."""

    # Global delta
    cell_mean = X_cell.mean(axis=0)
    nuc_mean = X_nuc.mean(axis=0)
    global_delta = cell_mean - nuc_mean

    delta_df = pd.DataFrame({
        "gene": var_names,
        "cell_mean": cell_mean,
        "nuc_mean": nuc_mean,
        "delta": global_delta,
        "abs_delta": np.abs(global_delta),
    })

    # Per cell-type delta
    ct_cell = obs_cell[ct_col].values.astype(str)
    ct_nuc = obs_nuc[ct_col].values.astype(str)
    common_cts = sorted(set(ct_cell) & set(ct_nuc))
    print(f"  Common cell types: {len(common_cts)}")

    ct_deltas = {}
    for ct in common_cts:
        mask_c = ct_cell == ct
        mask_n = ct_nuc == ct
        if mask_c.sum() >= 10 and mask_n.sum() >= 10:
            ct_deltas[ct] = X_cell[mask_c].mean(axis=0) - X_nuc[mask_n].mean(axis=0)

    return delta_df, ct_deltas, common_cts


def check_compartment_genes(delta_df):
    """Check ribosomal and nuclear lncRNA genes in human kidney delta."""
    results = {}

    # Ribosomal genes
    ribo_mask = delta_df["gene"].str.startswith(RIBOSOMAL_PREFIXES)
    ribo = delta_df[ribo_mask].copy()
    results["ribosomal"] = {
        "n_found": int(ribo_mask.sum()),
        "mean_delta": float(ribo["delta"].mean()),
        "median_delta": float(ribo["delta"].median()),
        "frac_positive": float((ribo["delta"] > 0).mean()),
        "top_5": ribo.nlargest(5, "delta")[["gene", "delta"]].to_dict("records"),
    }
    print(f"\n  Ribosomal: {ribo_mask.sum()} genes, mean delta={ribo['delta'].mean():.3f}, "
          f"frac positive={results['ribosomal']['frac_positive']:.1%}")

    # Nuclear lncRNAs
    lncrna_mask = delta_df["gene"].isin(NUCLEAR_LNCRNAS)
    lncrna = delta_df[lncrna_mask].copy()
    results["nuclear_lncrna"] = {
        "n_found": int(lncrna_mask.sum()),
        "per_gene": lncrna[["gene", "delta"]].to_dict("records"),
        "mean_delta": float(lncrna["delta"].mean()) if len(lncrna) > 0 else None,
        "frac_negative": float((lncrna["delta"] < 0).mean()) if len(lncrna) > 0 else None,
    }
    if len(lncrna) > 0:
        print(f"  Nuclear lncRNAs found: {len(lncrna)}")
        for _, row in lncrna.iterrows():
            direction = "DOWN" if row["delta"] < 0 else "UP"
            print(f"    {row['gene']}: delta={row['delta']:.3f} ({direction})")

    # Percentile ranks for key genes
    delta_df = delta_df.copy()
    delta_df["percentile"] = delta_df["delta"].rank(pct=True) * 100
    key_genes = list(NUCLEAR_LNCRNAS) + [g for g in delta_df["gene"] if g.startswith("RPL") or g.startswith("RPS")][:10]
    key_info = delta_df[delta_df["gene"].isin(key_genes)][["gene", "delta", "percentile"]]
    results["key_gene_ranks"] = key_info.to_dict("records")

    return results


def cross_tissue_comparison(kidney_delta_df, mouse_delta_path, ortholog_map):
    """Compare human kidney delta to mouse brain delta at the gene-class level."""
    mouse_delta = pd.read_csv(mouse_delta_path)
    print(f"\n  Mouse brain delta: {len(mouse_delta)} genes")

    # Map kidney genes to mouse symbols
    kidney_delta_df = kidney_delta_df.copy()
    kidney_delta_df["mouse_symbol"] = kidney_delta_df["gene"].map(ortholog_map)
    kidney_delta_df = kidney_delta_df.dropna(subset=["mouse_symbol"])

    # Inner join on mouse symbol
    mouse_renamed = mouse_delta[["gene", "delta"]].rename(
        columns={"gene": "mouse_symbol", "delta": "mouse_delta"})
    mouse_renamed["mouse_symbol"] = mouse_renamed["mouse_symbol"].astype(str)
    kidney_delta_df["mouse_symbol"] = kidney_delta_df["mouse_symbol"].astype(str)
    merged = kidney_delta_df.merge(mouse_renamed, on="mouse_symbol", how="inner")
    merged = merged.rename(columns={"delta": "kidney_delta"})
    print(f"  Mapped genes (intersection): {len(merged)} / {len(kidney_delta_df)}")

    results = {"n_mapped": len(merged), "n_kidney_total": len(kidney_delta_df),
               "n_mouse_total": len(mouse_delta)}

    if len(merged) < 10:
        print("  WARNING: too few mapped genes for meaningful comparison")
        results["warning"] = "too few mapped genes"
        return results, merged

    # Overall correlation
    r, p = stats.spearmanr(merged["kidney_delta"], merged["mouse_delta"])
    results["spearman_r"] = float(r)
    results["spearman_p"] = float(p)
    print(f"  Spearman correlation (kidney vs brain delta): r={r:.3f}, p={p:.2e}")

    pearson_r, pearson_p = stats.pearsonr(merged["kidney_delta"], merged["mouse_delta"])
    results["pearson_r"] = float(pearson_r)
    results["pearson_p"] = float(pearson_p)

    # Gene-class breakdown
    merged["is_ribo"] = merged["mouse_symbol"].str.startswith(("Rpl", "Rps"))
    merged["is_lncrna"] = merged["mouse_symbol"].str.match(r"^(Gm\d|[A-Z][a-z]*os$|Malat1|Neat1|Xist|Meg3)")

    # Ribosomal: both should be positive
    ribo = merged[merged["is_ribo"]]
    if len(ribo) > 0:
        results["ribo_both_positive"] = float((
            (ribo["kidney_delta"] > 0) & (ribo["mouse_delta"] > 0)
        ).mean())
        results["ribo_n"] = len(ribo)
        print(f"  Ribosomal genes: {len(ribo)}, both positive: {results['ribo_both_positive']:.1%}")

    # Check if synaptic genes are brain-only (positive in brain, near-zero in kidney)
    synaptic_mouse = ["Snap25", "Syn1", "Syt1", "Vamp2", "Sv2a", "Sv2b",
                       "Stx1a", "Stx1b", "Nrxn1", "Nlgn1"]
    syn_mask = merged["mouse_symbol"].isin(synaptic_mouse)
    syn = merged[syn_mask]
    if len(syn) > 0:
        results["synaptic"] = {
            "n_found": len(syn),
            "kidney_mean_delta": float(syn["kidney_delta"].mean()),
            "brain_mean_delta": float(syn["mouse_delta"].mean()),
            "per_gene": syn[["mouse_symbol", "kidney_delta", "mouse_delta"]].to_dict("records"),
        }
        print(f"  Synaptic genes: {len(syn)}, kidney delta={syn['kidney_delta'].mean():.3f}, "
              f"brain delta={syn['mouse_delta'].mean():.3f}")

    # Top-100 overlap
    kidney_top100 = set(merged.nlargest(100, "kidney_delta")["mouse_symbol"])
    brain_top100 = set(merged.nlargest(100, "mouse_delta")["mouse_symbol"])
    overlap = kidney_top100 & brain_top100
    results["top100_overlap"] = len(overlap)
    results["top100_shared_genes"] = sorted(overlap)
    print(f"  Top-100 increased overlap: {len(overlap)}/100")

    kidney_bot100 = set(merged.nsmallest(100, "kidney_delta")["mouse_symbol"])
    brain_bot100 = set(merged.nsmallest(100, "mouse_delta")["mouse_symbol"])
    overlap_bot = kidney_bot100 & brain_bot100
    results["bot100_overlap"] = len(overlap_bot)
    results["bot100_shared_genes"] = sorted(overlap_bot)
    print(f"  Top-100 decreased overlap: {len(overlap_bot)}/100")

    return results, merged


def run_go_enrichment_human(gene_list, output_dir, label):
    """Run g:Profiler GO enrichment on human gene list."""
    try:
        from gprofiler import GProfiler
    except ImportError:
        print("  g:Profiler not available, skipping GO enrichment")
        return None

    print(f"\n  Running GO enrichment on {label} ({len(gene_list)} genes)...")
    gp = GProfiler(return_dataframe=True)
    result = gp.profile(
        organism="hsapiens",
        query=gene_list,
        sources=["GO:BP", "GO:CC", "GO:MF", "KEGG"],
        significance_threshold_method="fdr",
    )

    if len(result) == 0:
        print("    No significant GO terms")
        return None

    result = result.sort_values("p_value")
    out_path = os.path.join(output_dir, f"go_{label}.csv")
    result.to_csv(out_path, index=False)

    print(f"    Top GO terms ({label}):")
    for _, row in result.head(10).iterrows():
        print(f"      {row['source']}:{row['name']} (p={row['p_value']:.2e}, n={row['intersection_size']})")

    return result


def make_plots(merged, kidney_compartment, output_dir):
    """Generate figures."""

    # 1. Scatter: kidney delta vs brain delta
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(merged["mouse_delta"], merged["kidney_delta"], alpha=0.15, s=8, c="steelblue")

    # Highlight ribosomal
    ribo = merged[merged["is_ribo"]]
    if len(ribo) > 0:
        ax.scatter(ribo["mouse_delta"], ribo["kidney_delta"], alpha=0.7, s=20, c="red", label="Ribosomal")

    ax.set_xlabel("Mouse brain delta (log1p)", fontsize=12)
    ax.set_ylabel("Human kidney delta (log1p)", fontsize=12)
    ax.set_title("Conserved cell−nucleus shift: brain vs kidney", fontsize=14)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.5, ls="--")

    # Add correlation text
    r = stats.spearmanr(merged["mouse_delta"], merged["kidney_delta"])[0]
    ax.text(0.05, 0.95, f"Spearman r = {r:.3f}\nn = {len(merged)} genes",
            transform=ax.transAxes, va="top", fontsize=11,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "brain_vs_kidney_delta.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved brain_vs_kidney_delta.png")

    # 2. Ribosomal/lncRNA rank plot for kidney
    fig, ax = plt.subplots(figsize=(10, 5))
    delta_sorted = merged.sort_values("kidney_delta")
    delta_sorted["rank"] = range(len(delta_sorted))

    ax.scatter(delta_sorted["rank"], delta_sorted["kidney_delta"], s=3, alpha=0.3, c="gray")

    ribo_sorted = delta_sorted[delta_sorted["is_ribo"]]
    if len(ribo_sorted) > 0:
        ax.scatter(ribo_sorted["rank"], ribo_sorted["kidney_delta"], s=15, c="red",
                   alpha=0.7, label="Ribosomal", zorder=5)

    # Mark nuclear lncRNAs
    for gene in NUCLEAR_LNCRNAS:
        mask = delta_sorted["gene"] == gene
        if mask.any():
            row = delta_sorted[mask].iloc[0]
            ax.scatter(row["rank"], row["kidney_delta"], s=40, c="blue", zorder=6, marker="D")
            ax.annotate(gene, (row["rank"], row["kidney_delta"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xlabel("Gene rank", fontsize=12)
    ax.set_ylabel("Kidney delta (log1p)", fontsize=12)
    ax.set_title("Human kidney: gene delta ranks", fontsize=14)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "kidney_delta_ranks.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved kidney_delta_ranks.png")


def main():
    parser = argparse.ArgumentParser(description="Lake 2023 kidney conserved-transformation analysis")
    parser.add_argument("--cell_h5ad", type=str, required=True, help="Path to Lake 2023 cells h5ad")
    parser.add_argument("--nuc_h5ad", type=str, required=True, help="Path to Lake 2023 nuclei h5ad")
    parser.add_argument("--output_dir", type=str,
                        default="notes/experiments/conserved_transformation")
    parser.add_argument("--ct_col", type=str, default=None,
                        help="Cell type column in obs (auto-detected if not set)")
    parser.add_argument("--donor_col", type=str, default=None,
                        help="Donor column in obs (auto-detected if not set)")
    parser.add_argument("--max_cells", type=int, default=None,
                        help="Max cells per modality (subsample for speed)")
    parser.add_argument("--skip_go", action="store_true", help="Skip GO enrichment")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load data ---
    X_cell, obs_cell, var_cell = load_and_normalize(args.cell_h5ad, args.max_cells)
    X_nuc, obs_nuc, var_nuc = load_and_normalize(args.nuc_h5ad, args.max_cells)

    # --- Detect columns ---
    ct_col = args.ct_col
    if ct_col is None:
        for candidate in ["cell_type", "subclass.l1", "cell_type_ontology_term_id",
                          "subclass_label", "author_cell_type", "cell_ontology_class"]:
            if candidate in obs_cell.columns:
                ct_col = candidate
                break
        if ct_col is None:
            print("ERROR: could not auto-detect cell type column. Available:")
            print(obs_cell.columns.tolist())
            sys.exit(1)
    print(f"\nUsing cell type column: '{ct_col}'")
    print(f"  Cell types (cells): {obs_cell[ct_col].nunique()}")
    print(f"  Cell types (nuclei): {obs_nuc[ct_col].nunique()}")

    donor_col = args.donor_col
    if donor_col is None:
        for candidate in ["donor_id", "donor", "sample_id", "subject_id"]:
            if candidate in obs_cell.columns:
                donor_col = candidate
                break
    if donor_col:
        donors_cell = set(obs_cell[donor_col].unique())
        donors_nuc = set(obs_nuc[donor_col].unique())
        shared_donors = donors_cell & donors_nuc
        print(f"\nDonor column: '{donor_col}'")
        print(f"  Cell donors: {len(donors_cell)}, Nuclei donors: {len(donors_nuc)}, "
              f"Shared: {len(shared_donors)}")
        if shared_donors and len(shared_donors) < len(donors_cell):
            print(f"  Restricting to shared donors: {sorted(shared_donors)}")
            obs_cell_mask = obs_cell[donor_col].isin(shared_donors)
            obs_nuc_mask = obs_nuc[donor_col].isin(shared_donors)
            X_cell = X_cell[obs_cell_mask.values]
            obs_cell = obs_cell[obs_cell_mask].copy()
            X_nuc = X_nuc[obs_nuc_mask.values]
            obs_nuc = obs_nuc[obs_nuc_mask].copy()
            print(f"  After donor filter: {X_cell.shape[0]} cells, {X_nuc.shape[0]} nuclei")

    # --- Intersect gene space ---
    shared_genes = sorted(set(var_cell) & set(var_nuc))
    print(f"\nGene space: {len(var_cell)} cell, {len(var_nuc)} nuc, {len(shared_genes)} shared")

    cell_idx = [var_cell.index(g) for g in shared_genes]
    nuc_idx = [var_nuc.index(g) for g in shared_genes]
    X_cell = X_cell[:, cell_idx]
    X_nuc = X_nuc[:, nuc_idx]

    # --- Compute empirical delta ---
    print("\nComputing empirical delta...")
    delta_df, ct_deltas, common_cts = compute_empirical_delta(
        X_cell, X_nuc, obs_cell, obs_nuc, shared_genes, ct_col)

    delta_df.to_csv(os.path.join(args.output_dir, "kidney_empirical_delta.csv"), index=False)
    print(f"  Saved kidney_empirical_delta.csv ({len(delta_df)} genes)")

    # --- Compartment gene check ---
    print("\nChecking compartment genes in human kidney...")
    compartment = check_compartment_genes(delta_df)

    # --- Ortholog mapping + cross-tissue comparison ---
    print("\nCross-tissue comparison (human kidney vs mouse brain)...")
    mouse_delta = pd.read_csv(MOUSE_DELTA_PATH)
    mouse_gene_list = mouse_delta["gene"].tolist()
    ortholog_map = get_ortholog_mapping(delta_df["gene"].tolist(), mouse_gene_list)
    cross_results, merged = cross_tissue_comparison(
        delta_df, MOUSE_DELTA_PATH, ortholog_map)

    merged.to_csv(os.path.join(args.output_dir, "kidney_vs_brain_merged.csv"), index=False)

    # --- GO enrichment on top-increased kidney genes ---
    go_results = {}
    if not args.skip_go:
        top100_up = delta_df.nlargest(100, "delta")["gene"].tolist()
        go_up = run_go_enrichment_human(top100_up, args.output_dir, "top100_increased")
        if go_up is not None:
            go_results["top100_increased_top5"] = go_up.head(5)[
                ["source", "name", "p_value", "intersection_size"]].to_dict("records")

        top100_down = delta_df.nsmallest(100, "delta")["gene"].tolist()
        go_down = run_go_enrichment_human(top100_down, args.output_dir, "top100_decreased")
        if go_down is not None:
            go_results["top100_decreased_top5"] = go_down.head(5)[
                ["source", "name", "p_value", "intersection_size"]].to_dict("records")

    # --- Plots ---
    print("\nGenerating plots...")
    if len(merged) > 0:
        make_plots(merged, compartment, args.output_dir)

    # --- Save summary ---
    summary = {
        "dataset": "Lake 2023 human kidney",
        "n_cells": int(X_cell.shape[0]),
        "n_nuclei": int(X_nuc.shape[0]),
        "n_shared_genes": len(shared_genes),
        "n_common_cts": len(common_cts),
        "common_cts": common_cts,
        "compartment": compartment,
        "cross_tissue": cross_results,
        "go_enrichment": go_results,
    }

    out_path = os.path.join(args.output_dir, "conserved_transformation_stats.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved summary to {out_path}")

    # --- Print headline ---
    print(f"\n{'='*60}")
    print(f"  CONSERVED TRANSFORMATION — SUMMARY")
    print(f"{'='*60}")
    print(f"  Dataset: Lake 2023 human kidney ({X_cell.shape[0]} cells, {X_nuc.shape[0]} nuclei)")
    print(f"  Shared genes: {len(shared_genes)}")
    print(f"  Common cell types: {len(common_cts)}")
    ribo = compartment.get("ribosomal", {})
    print(f"  Ribosomal genes: {ribo.get('n_found', 0)}, "
          f"mean delta={ribo.get('mean_delta', 0):.3f}, "
          f"frac positive={ribo.get('frac_positive', 0):.1%}")
    lncrna = compartment.get("nuclear_lncrna", {})
    print(f"  Nuclear lncRNAs found: {lncrna.get('n_found', 0)}, "
          f"mean delta={lncrna.get('mean_delta', 'N/A')}")
    print(f"  Cross-tissue Spearman r: {cross_results.get('spearman_r', 'N/A')}")
    print(f"  Top-100 increased overlap: {cross_results.get('top100_overlap', 'N/A')}/100")


if __name__ == "__main__":
    main()
