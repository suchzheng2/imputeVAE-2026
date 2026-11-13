"""
Task 2 (revised): Localization concordance using lncATLAS CN-RCI.

Correlates model imputation delta with experimentally measured cytoplasmic/nuclear
RNA ratios (CN-RCI) from lncATLAS (Mas-Ponte et al., RNA 2017).

Steps:
  1. Load lncATLAS_all_data.csv, filter to CNRCI rows, median across cell lines
  2. Map mouse gene symbols → human orthologs via Ensembl BioMart
  3. Merge with model delta (all_genes_delta.csv)
  4. Compute Spearman, AUROC, scrambled-delta null, lncRNA-only subanalysis
  5. Save scatter + ROC plots

Prerequisites:
  - Download lncATLAS_all_data.csv manually from https://lncatlas.crg.eu
    (Get Raw Data tab → "Download All raw data" button)
    Place in data/external/lncATLAS_all_data.csv

Usage:
    python analysis/localization_lncatlas.py
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT, "notes/experiments/localization_concordance")

# =====================================================================
# Step 1: Load lncATLAS CN-RCI
# =====================================================================

def load_lncatlas_cnrci(path):
    """Load lncATLAS, filter to CN-RCI rows (ratio2), median across cell lines.

    Uses chunked reading to avoid OOM on large files (~2.4M rows).
    The CN-RCI = log2(cytosol/nucleus): positive = cytoplasm-enriched.
    In lncATLAS downloads, the Data Type column for CN-RCI is 'ratio2'.
    """
    print(f"Loading lncATLAS from {path} (chunked)...")

    # Columns: ENSEMBL ID, Data Source, Data Type, Value, Gene Name, Coding Type, Biotype
    # CN-RCI is stored as Data Type == "ratio2"
    CNRCI_LABELS = {"CNRCI", "ratio2"}

    chunks_cnrci = []
    total_rows = 0
    for chunk in pd.read_csv(path, chunksize=200000):
        total_rows += len(chunk)
        cn_chunk = chunk[chunk["Data Type"].isin(CNRCI_LABELS)].copy()
        if len(cn_chunk) > 0:
            cn_chunk["Value_num"] = pd.to_numeric(cn_chunk["Value"], errors="coerce")
            chunks_cnrci.append(cn_chunk[["Gene Name", "Value_num", "Coding Type"]].dropna(subset=["Value_num"]))

    cn = pd.concat(chunks_cnrci, ignore_index=True)
    print(f"  Total rows scanned: {total_rows}")
    print(f"  CN-RCI rows: {len(cn)}")

    # Median across cell lines per gene
    cnrci = cn.groupby("Gene Name")["Value_num"].median().reset_index()
    cnrci.columns = ["human_sym", "CN_RCI"]
    print(f"  Unique genes with CN-RCI: {len(cnrci)}")
    print(f"  CN-RCI range: [{cnrci['CN_RCI'].min():.2f}, {cnrci['CN_RCI'].max():.2f}]")

    # Track coding type for lncRNA-only analysis
    coding = cn.groupby("Gene Name")["Coding Type"].first().reset_index()
    coding.columns = ["human_sym", "coding_type"]

    return cnrci, coding


# =====================================================================
# Step 2: Mouse → human orthologs
# =====================================================================

def get_mouse_human_orthologs():
    """Query Ensembl BioMart for mouse→human ortholog mapping."""
    cache_path = os.path.join(PROJECT, "data/external/mouse_human_orthologs.csv")

    if os.path.exists(cache_path):
        print(f"Loading cached orthologs from {cache_path}...")
        orth = pd.read_csv(cache_path)
        print(f"  {len(orth)} mappings loaded")
        return orth

    print("Querying Ensembl BioMart for mouse→human orthologs...")
    try:
        from pybiomart import Server
        server = Server(host="http://www.ensembl.org")
        mm = server.marts["ENSEMBL_MART_ENSEMBL"].datasets["mmusculus_gene_ensembl"]
        orth = mm.query(attributes=[
            "external_gene_name",
            "hsapiens_homolog_associated_gene_name"
        ])
        orth.columns = ["mouse_sym", "human_sym"]
        orth = orth.dropna()
        orth = orth[orth["human_sym"].str.len() > 0]
        orth = orth.drop_duplicates("mouse_sym")  # keep 1 human per mouse
        print(f"  {len(orth)} mouse→human mappings from BioMart")

        # Cache for future use
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        orth.to_csv(cache_path, index=False)
        print(f"  Cached to {cache_path}")
        return orth

    except Exception as e:
        print(f"  BioMart query failed: {e}")
        print("  Falling back to uppercase-only mapping")
        return None


def map_mouse_to_human(delta_df, orthologs=None):
    """Add human_sym column to delta dataframe."""
    if orthologs is not None:
        delta_df = delta_df.merge(orthologs, left_on="gene", right_on="mouse_sym", how="left")
        mapped = delta_df["human_sym"].notna().sum()
        print(f"  BioMart mapped: {mapped} / {len(delta_df)}")
    else:
        delta_df["human_sym"] = None

    # Uppercase fallback for unmapped
    unmapped = delta_df["human_sym"].isna()
    delta_df.loc[unmapped, "human_sym"] = delta_df.loc[unmapped, "gene"].str.upper()
    print(f"  After uppercase fallback: {delta_df['human_sym'].notna().sum()} / {len(delta_df)}")
    return delta_df


# =====================================================================
# Step 3: Merge & compute stats
# =====================================================================

def compute_concordance(merged, label="lncATLAS"):
    """Compute Spearman, AUROC, sanity checks."""
    m = merged.dropna(subset=["delta", "CN_RCI"])
    print(f"\n{'='*60}")
    print(f"  {label}: {len(m)} genes matched")
    print(f"{'='*60}")

    delta = m["delta"].values
    cnrci = m["CN_RCI"].values

    # Spearman
    rho, p = stats.spearmanr(delta, cnrci)
    print(f"  Spearman(delta, CN-RCI) = {rho:.3f}  p = {p:.2e}")

    # Pearson
    r_pear, p_pear = stats.pearsonr(delta, cnrci)
    print(f"  Pearson(delta, CN-RCI)  = {r_pear:.3f}  p = {p_pear:.2e}")

    # AUROC: delta predicts cytoplasmic (CN_RCI > 0) vs nuclear (CN_RCI < 0)
    label_binary = (cnrci > 0).astype(int)
    n_pos = label_binary.sum()
    n_neg = len(label_binary) - n_pos
    auroc = roc_auc_score(label_binary, delta)
    auprc = average_precision_score(label_binary, delta)
    print(f"  AUROC (added=cytoplasmic) = {auroc:.3f}")
    print(f"  PR-AUC = {auprc:.3f}  (baseline={n_pos/len(label_binary):.3f})")

    # Sanity: canonical nuclear lncRNAs
    print(f"\n  Canonical gene checks:")
    for g in ["MALAT1", "NEAT1", "XIST", "ACTB", "EEF1A1", "RPL4", "RPS3"]:
        row = m[m["human_sym"] == g]
        if len(row):
            print(f"    {g:<10}  delta={row.delta.iloc[0]:+.3f}  CN-RCI={row.CN_RCI.iloc[0]:+.3f}")
        else:
            print(f"    {g:<10}  NOT FOUND")

    results = {
        "n_genes": int(len(m)),
        "spearman_rho": float(rho),
        "spearman_p": float(p),
        "pearson_r": float(r_pear),
        "pearson_p": float(p_pear),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "n_cytoplasmic": int(n_pos),
        "n_nuclear": int(n_neg),
        "baseline_prevalence": float(n_pos / len(label_binary)),
    }
    return results


def scrambled_null(merged, n_perm=1000, seed=0):
    """Scrambled-delta negative control."""
    m = merged.dropna(subset=["delta", "CN_RCI"])
    real_rho, _ = stats.spearmanr(m["delta"].values, m["CN_RCI"].values)

    rng = np.random.default_rng(seed)
    null_rhos = []
    for _ in range(n_perm):
        perm_delta = rng.permutation(m["delta"].values)
        null_rho, _ = stats.spearmanr(perm_delta, m["CN_RCI"].values)
        null_rhos.append(null_rho)

    null_rhos = np.array(null_rhos)
    pct_beaten = (abs(real_rho) > np.abs(null_rhos)).mean() * 100
    print(f"\n  Scrambled-delta null ({n_perm} permutations):")
    print(f"    Real |rho| = {abs(real_rho):.4f}")
    print(f"    Null mean = {null_rhos.mean():.4f}, std = {null_rhos.std():.4f}")
    print(f"    Beats {pct_beaten:.1f}% of permutations")

    return {
        "real_rho": float(real_rho),
        "null_mean": float(null_rhos.mean()),
        "null_std": float(null_rhos.std()),
        "pct_beaten": float(pct_beaten),
    }


def lncrna_subanalysis(merged, coding_info):
    """lncRNA-only subanalysis."""
    if coding_info is None:
        print("\n  lncRNA subanalysis: no coding_type info available, skipping")
        return None

    m = merged.merge(coding_info, on="human_sym", how="left")
    nc = m[m["coding_type"] == "nc"].dropna(subset=["delta", "CN_RCI"])
    if len(nc) < 20:
        print(f"\n  lncRNA subanalysis: only {len(nc)} lncRNAs matched, too few")
        return {"n_lncrna": int(len(nc))}

    rho, p = stats.spearmanr(nc["delta"].values, nc["CN_RCI"].values)
    print(f"\n  lncRNA-only subanalysis:")
    print(f"    n = {len(nc)}")
    print(f"    Spearman = {rho:.3f}  p = {p:.2e}")

    return {
        "n_lncrna": int(len(nc)),
        "spearman_rho": float(rho),
        "spearman_p": float(p),
    }


# =====================================================================
# Step 4: Plots
# =====================================================================

def plot_scatter_and_roc(merged, results, out_dir):
    """Scatter of delta vs CN-RCI + ROC curve."""
    m = merged.dropna(subset=["delta", "CN_RCI"])
    delta = m["delta"].values
    cnrci = m["CN_RCI"].values

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: scatter
    ax = axes[0]
    ax.scatter(cnrci, delta, alpha=0.08, s=4, c="steelblue", rasterized=True)

    # Highlight canonical genes
    highlights = {
        "MALAT1": ("red", "v"), "NEAT1": ("red", "v"), "XIST": ("red", "v"),
        "ACTB": ("green", "^"), "EEF1A1": ("green", "^"), "RPL4": ("green", "^"),
    }
    for gene, (color, marker) in highlights.items():
        row = m[m["human_sym"] == gene]
        if len(row):
            ax.scatter(row["CN_RCI"].iloc[0], row["delta"].iloc[0],
                       c=color, s=80, marker=marker, edgecolors="black",
                       linewidths=0.5, zorder=5)
            ax.annotate(gene, (row["CN_RCI"].iloc[0], row["delta"].iloc[0]),
                        fontsize=7, ha="left", va="bottom")

    # Trend line
    slope, intercept = np.polyfit(cnrci, delta, 1)
    x_fit = np.array([cnrci.min(), cnrci.max()])
    ax.plot(x_fit, slope * x_fit + intercept, "r--", linewidth=1.5, alpha=0.8)

    rho = results["spearman_rho"]
    ax.set_xlabel("lncATLAS CN-RCI\n(log2 cytoplasmic/nuclear), + = cytoplasmic", fontsize=11)
    ax.set_ylabel("Imputation delta\n(imputed cell − input nuclei, log1p)", fontsize=11)
    ax.set_title(f"Delta vs lncATLAS CN-RCI\n"
                 f"Spearman ρ = {rho:.3f}, n = {results['n_genes']}",
                 fontsize=12)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")

    # Right: ROC curve
    ax = axes[1]
    label_binary = (cnrci > 0).astype(int)
    fpr, tpr, _ = roc_curve(label_binary, delta)
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"AUROC = {results['auroc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"ROC: Delta predicts cytoplasmic localization\n"
                 f"({results['n_cytoplasmic']} cyto vs {results['n_nuclear']} nuc)",
                 fontsize=12)
    ax.legend(fontsize=11, loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "delta_vs_cnrci_scatter_roc.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {out_path}")


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lncatlas_csv",
                        default=os.path.join(PROJECT, "data/external/lncATLAS_all_data.csv"))
    parser.add_argument("--delta_csv",
                        default=os.path.join(PROJECT,
                            "notes/experiments/nuclear_lncrna_analysis/all_genes_delta.csv"))
    parser.add_argument("--output_dir", default=OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Check lncATLAS file
    if not os.path.exists(args.lncatlas_csv) or os.path.getsize(args.lncatlas_csv) < 1000:
        print("ERROR: lncATLAS_all_data.csv not found or too small.")
        print("Please download it manually from https://lncatlas.crg.eu/")
        print("  1. Go to 'Get Raw Data' tab")
        print("  2. Click 'Download All raw data'")
        print(f"  3. Save to {args.lncatlas_csv}")
        sys.exit(1)

    # Step 1: Load lncATLAS CN-RCI
    cnrci, coding_info = load_lncatlas_cnrci(args.lncatlas_csv)

    # Step 2: Load delta + ortholog mapping
    print(f"\nLoading delta from {args.delta_csv}...")
    delta_df = pd.read_csv(args.delta_csv)
    delta_df = delta_df.rename(columns={delta_df.columns[0]: "gene"})
    assert "delta" in delta_df.columns, f"Expected 'delta' column, got {delta_df.columns.tolist()}"
    delta_df = delta_df[["gene", "delta"]].dropna()
    print(f"  {len(delta_df)} genes with delta")

    orthologs = get_mouse_human_orthologs()
    delta_df = map_mouse_to_human(delta_df, orthologs)

    # Step 3: Merge
    merged = delta_df.merge(cnrci, on="human_sym", how="inner")
    merged = merged.dropna(subset=["delta", "CN_RCI"])
    print(f"\n  Mapped genes in lncATLAS: {len(merged)} of {len(delta_df)}")

    # Step 4: Concordance stats
    results = compute_concordance(merged, "lncATLAS CN-RCI")

    # Step 5: Scrambled-delta null
    null_results = scrambled_null(merged)
    results["scrambled_null"] = null_results

    # Step 6: lncRNA-only subanalysis
    lnc_results = lncrna_subanalysis(merged, coding_info)
    if lnc_results:
        results["lncrna_only"] = lnc_results

    # Step 7: Plots
    plot_scatter_and_roc(merged, results, args.output_dir)

    # Step 8: Save
    merged.to_csv(os.path.join(args.output_dir, "delta_vs_cnrci.csv"), index=False)
    print(f"  Saved per-gene table: {len(merged)} genes")

    out_json = os.path.join(args.output_dir, "lncatlas_stats.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {out_json}")

    # Step 9: Summary
    print(f"\n{'='*60}")
    print("  SUMMARY — lncATLAS CN-RCI concordance")
    print(f"{'='*60}")
    print(f"  Genes matched:     {results['n_genes']}")
    print(f"  Spearman rho:      {results['spearman_rho']:.3f}  (p={results['spearman_p']:.2e})")
    print(f"  AUROC:             {results['auroc']:.3f}")
    print(f"  PR-AUC:            {results['auprc']:.3f}")
    print(f"  Scrambled beaten:  {null_results['pct_beaten']:.1f}%")
    if lnc_results and "spearman_rho" in lnc_results:
        print(f"  lncRNA-only rho:   {lnc_results['spearman_rho']:.3f} (n={lnc_results['n_lncrna']})")


if __name__ == "__main__":
    main()
