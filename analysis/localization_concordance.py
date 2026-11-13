"""
Task 2: Localization concordance vs external databases.

Tests whether the model's imputation delta (imputed_cell - input_nuclei) correlates
with independently measured subcellular RNA localization, using references the model
never saw during training.

External references:
  1. COMPARTMENTS (Jensen lab) — mouse protein/RNA subcellular localization scores
     from integrated evidence (knowledge, text mining, predictions).
     CN ratio = cytoplasm_score - nucleus_score. Positive = cytoplasmic.
  2. (Optional) lncATLAS CN-RCI — human lncRNA localization (requires ortholog mapping).
  3. (Optional) APEX-seq (Fazal 2019) — human RNA localization (requires ortholog mapping).

Output: notes/experiments/localization_concordance/
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT, "notes/experiments/localization_concordance")


def load_delta(path):
    """Load per-gene delta from all_genes_delta.csv."""
    df = pd.read_csv(path)
    # Columns: gene, cell_mean, nuc_mean, delta, abs_delta, is_lncrna, lncrna_source
    return df.set_index("gene")


def load_compartments(path):
    """Load COMPARTMENTS integrated data and compute CN ratio per gene.

    CN ratio = max(cytoplasm_score) - max(nucleus_score).
    Positive = more cytoplasmic evidence, negative = more nuclear evidence.

    Accepts either the full integrated file or the pre-filtered nuc+cyto file.
    """
    # Try pre-filtered file first (much smaller)
    prefiltered = path.replace("_integrated_full.tsv", "_nuc_cyto.tsv")
    if os.path.exists(prefiltered):
        df = pd.read_csv(prefiltered, sep="\t")
    else:
        df = pd.read_csv(path, sep="\t", header=None,
                         names=["gene_id", "gene_name", "go_term", "go_name", "score"])

    # Extract nucleus (GO:0005634) and cytoplasm (GO:0005737) scores
    nuc = df[df.go_term == "GO:0005634"].groupby("gene_name")["score"].max()
    cyto = df[df.go_term == "GO:0005737"].groupby("gene_name")["score"].max()

    cn = pd.DataFrame({"cyto_score": cyto, "nuc_score": nuc}).dropna()
    cn["cn_ratio"] = cn["cyto_score"] - cn["nuc_score"]
    return cn


def load_compartments_knowledge(path):
    """Load COMPARTMENTS knowledge-channel data.

    Uses curated GO annotations (higher confidence than integrated).
    Assigns binary labels: cytoplasm-only, nucleus-only, or both.
    """
    df = pd.read_csv(path, sep="\t", header=None,
                     names=["ensembl_id", "gene_name", "go_term", "go_name",
                            "source", "evidence_code", "confidence"])

    # Only high-confidence annotations (score >= 3)
    df = df[df.confidence >= 3]

    nuc_genes = set(df[df.go_term == "GO:0005634"]["gene_name"])
    cyto_genes = set(df[df.go_term == "GO:0005737"]["gene_name"])

    labels = {}
    for g in nuc_genes | cyto_genes:
        in_nuc = g in nuc_genes
        in_cyto = g in cyto_genes
        if in_cyto and not in_nuc:
            labels[g] = "cytoplasm_only"
        elif in_nuc and not in_cyto:
            labels[g] = "nucleus_only"
        else:
            labels[g] = "both"

    return pd.Series(labels, name="localization")


def concordance_analysis(delta_df, cn_df, label="COMPARTMENTS"):
    """Compute concordance between imputation delta and external CN ratio."""
    # Intersect
    common = delta_df.index.intersection(cn_df.index)
    d = delta_df.loc[common, "delta"].values
    cn = cn_df.loc[common, "cn_ratio"].values

    print(f"\n{'='*60}")
    print(f"  {label}: {len(common)} genes in common")
    print(f"{'='*60}")

    # Spearman correlation
    rho, pval = stats.spearmanr(d, cn)
    print(f"  Spearman rho = {rho:.4f}, p = {pval:.2e}")

    # Pearson correlation
    r, p_pearson = stats.pearsonr(d, cn)
    print(f"  Pearson r    = {r:.4f}, p = {p_pearson:.2e}")

    # AUC: use delta as classifier to predict cytoplasmic (cn > 0) vs nuclear (cn < 0)
    # Exclude genes with cn == 0 (ambiguous)
    mask = cn != 0
    labels_binary = (cn[mask] > 0).astype(int)
    scores = d[mask]

    auroc = roc_auc_score(labels_binary, scores)
    auprc = average_precision_score(labels_binary, scores)
    n_pos = labels_binary.sum()
    n_neg = len(labels_binary) - n_pos
    print(f"  AUROC = {auroc:.4f}  (cytoplasmic={n_pos}, nuclear={n_neg})")
    print(f"  AUPRC = {auprc:.4f}  (baseline={n_pos/len(labels_binary):.4f})")

    # Check extreme cases
    print(f"\n  Extreme gene checks:")
    for gene in ["Malat1", "Xist", "Neat1", "Meg3", "Actb", "Eef1a1", "Rpl4", "Rps3"]:
        if gene in delta_df.index and gene in cn_df.index:
            print(f"    {gene:<10}  delta={delta_df.loc[gene, 'delta']:+.3f}  "
                  f"CN={cn_df.loc[gene, 'cn_ratio']:+.3f}")

    return {
        "n_genes": int(len(common)),
        "spearman_rho": float(rho),
        "spearman_p": float(pval),
        "pearson_r": float(r),
        "pearson_p": float(p_pearson),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "n_cytoplasmic": int(n_pos),
        "n_nuclear": int(n_neg),
        "baseline_prevalence": float(n_pos / len(labels_binary)),
    }


def knowledge_analysis(delta_df, loc_labels):
    """Compare delta distributions for cytoplasm-only vs nucleus-only genes."""
    common = delta_df.index.intersection(loc_labels.index)
    loc = loc_labels[common]
    d = delta_df.loc[common, "delta"]

    cyto_only = d[loc == "cytoplasm_only"]
    nuc_only = d[loc == "nucleus_only"]
    both = d[loc == "both"]

    print(f"\n  Knowledge-channel (curated GO, confidence >= 3):")
    print(f"    Cytoplasm-only: n={len(cyto_only)}, median delta={cyto_only.median():.3f}")
    print(f"    Nucleus-only:   n={len(nuc_only)}, median delta={nuc_only.median():.3f}")
    print(f"    Both:           n={len(both)}, median delta={both.median():.3f}")

    if len(cyto_only) > 0 and len(nuc_only) > 0:
        u_stat, u_p = stats.mannwhitneyu(cyto_only, nuc_only, alternative="greater")
        print(f"    Mann-Whitney U (cyto > nuc): p = {u_p:.2e}")
    else:
        u_p = None

    return {
        "n_cyto_only": int(len(cyto_only)),
        "n_nuc_only": int(len(nuc_only)),
        "n_both": int(len(both)),
        "median_delta_cyto": float(cyto_only.median()) if len(cyto_only) > 0 else None,
        "median_delta_nuc": float(nuc_only.median()) if len(nuc_only) > 0 else None,
        "median_delta_both": float(both.median()) if len(both) > 0 else None,
        "mannwhitney_p": float(u_p) if u_p is not None else None,
    }


def plot_scatter(delta_df, cn_df, stats_dict, out_path):
    """Scatter plot: delta vs CN ratio."""
    common = delta_df.index.intersection(cn_df.index)
    d = delta_df.loc[common, "delta"].values
    cn = cn_df.loc[common, "cn_ratio"].values

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: scatter
    ax = axes[0]
    ax.scatter(cn, d, alpha=0.08, s=4, c="steelblue", rasterized=True)

    # Highlight known genes
    highlights = {
        "Malat1": "red", "Xist": "red", "Neat1": "red",
        "Actb": "green", "Eef1a1": "green", "Rpl4": "green",
    }
    for gene, color in highlights.items():
        if gene in delta_df.index and gene in cn_df.index:
            ax.scatter(cn_df.loc[gene, "cn_ratio"], delta_df.loc[gene, "delta"],
                       c=color, s=60, edgecolors="black", linewidths=0.5, zorder=5)
            ax.annotate(gene, (cn_df.loc[gene, "cn_ratio"], delta_df.loc[gene, "delta"]),
                        fontsize=7, ha="left", va="bottom")

    # Trend line
    slope, intercept = np.polyfit(cn, d, 1)
    x_fit = np.array([cn.min(), cn.max()])
    ax.plot(x_fit, slope * x_fit + intercept, "r--", linewidth=1.5, alpha=0.8)

    rho = stats_dict["spearman_rho"]
    ax.set_xlabel("COMPARTMENTS CN ratio\n(cytoplasm − nucleus score)", fontsize=11)
    ax.set_ylabel("Imputation delta\n(imputed cell − input nuclei, log1p)", fontsize=11)
    ax.set_title(f"Delta vs External Localization\n"
                 f"Spearman ρ = {rho:.3f}, n = {stats_dict['n_genes']}",
                 fontsize=12)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")

    # Right: ROC curve
    ax = axes[1]
    mask = cn != 0
    labels_binary = (cn[mask] > 0).astype(int)
    scores = d[mask]

    # Sort by threshold to make ROC
    from sklearn.metrics import roc_curve, precision_recall_curve
    fpr, tpr, _ = roc_curve(labels_binary, scores)
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"AUROC = {stats_dict['auroc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"ROC: Delta predicts cytoplasmic localization\n"
                 f"({stats_dict['n_cytoplasmic']} cyto vs {stats_dict['n_nuclear']} nuc)",
                 fontsize=12)
    ax.legend(fontsize=11, loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {out_path}")


def plot_distributions(delta_df, loc_labels, out_path):
    """Box/violin plots of delta by localization category."""
    common = delta_df.index.intersection(loc_labels.index)
    loc = loc_labels[common]
    d = delta_df.loc[common, "delta"]

    fig, ax = plt.subplots(figsize=(8, 5))

    categories = ["cytoplasm_only", "both", "nucleus_only"]
    cat_labels = ["Cytoplasm\nonly", "Both", "Nucleus\nonly"]
    colors = ["#2ecc71", "#95a5a6", "#e74c3c"]

    data = [d[loc == cat].values for cat in categories]
    parts = ax.violinplot(data, positions=range(len(categories)),
                          showmedians=True, showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("black")

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(cat_labels, fontsize=11)
    ax.set_ylabel("Imputation delta (log1p)", fontsize=11)
    ax.set_title("Delta by curated localization category\n(COMPARTMENTS knowledge, confidence ≥ 3)",
                 fontsize=12)

    for i, cat_data in enumerate(data):
        ax.annotate(f"n={len(cat_data)}\nmed={np.median(cat_data):.2f}",
                    (i, np.median(cat_data)), fontsize=8,
                    ha="center", va="bottom", textcoords="offset points",
                    xytext=(0, 10))

    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_binned(delta_df, cn_df, out_path):
    """Binned mean delta by CN ratio quintile — cleaner trend visualization."""
    common = delta_df.index.intersection(cn_df.index)
    d = delta_df.loc[common, "delta"].values
    cn = cn_df.loc[common, "cn_ratio"].values

    # Quintile bins
    n_bins = 10
    bin_edges = np.percentile(cn, np.linspace(0, 100, n_bins + 1))
    bin_centers = []
    bin_means = []
    bin_sems = []
    bin_ns = []

    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (cn >= bin_edges[i]) & (cn < bin_edges[i + 1])
        else:
            mask = (cn >= bin_edges[i]) & (cn <= bin_edges[i + 1])
        if mask.sum() > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_means.append(d[mask].mean())
            bin_sems.append(d[mask].std() / np.sqrt(mask.sum()))
            bin_ns.append(mask.sum())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(bin_centers, bin_means, yerr=bin_sems, fmt="o-",
                color="steelblue", capsize=4, linewidth=2, markersize=6)
    ax.set_xlabel("COMPARTMENTS CN ratio (binned deciles)", fontsize=11)
    ax.set_ylabel("Mean imputation delta (log1p)", fontsize=11)
    ax.set_title("Monotonic trend: more cytoplasmic genes get larger delta", fontsize=12)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Localization concordance analysis")
    parser.add_argument("--delta_csv",
                        default=os.path.join(PROJECT,
                            "notes/experiments/nuclear_lncrna_analysis/all_genes_delta.csv"))
    parser.add_argument("--compartments_integrated",
                        default=os.path.join(PROJECT,
                            "data/external/mouse_compartment_integrated_full.tsv"))
    parser.add_argument("--compartments_knowledge",
                        default=os.path.join(PROJECT,
                            "data/external/mouse_compartment_knowledge_full.tsv"))
    parser.add_argument("--output_dir", default=OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("Loading imputation delta...")
    delta_df = load_delta(args.delta_csv)
    print(f"  {len(delta_df)} genes")

    results = {}

    # === COMPARTMENTS integrated ===
    if os.path.exists(args.compartments_integrated):
        print("\nLoading COMPARTMENTS integrated data...")
        cn_df = load_compartments(args.compartments_integrated)
        print(f"  {len(cn_df)} genes with CN ratio")

        stats_int = concordance_analysis(delta_df, cn_df, "COMPARTMENTS (integrated)")
        results["compartments_integrated"] = stats_int

        plot_scatter(delta_df, cn_df, stats_int,
                     os.path.join(args.output_dir, "delta_vs_cn_scatter.png"))
        plot_binned(delta_df, cn_df,
                    os.path.join(args.output_dir, "delta_vs_cn_binned.png"))

        # Save the per-gene table
        common = delta_df.index.intersection(cn_df.index)
        merge = delta_df.loc[common, ["delta", "cell_mean", "nuc_mean"]].copy()
        merge["cn_ratio"] = cn_df.loc[common, "cn_ratio"]
        merge["cyto_score"] = cn_df.loc[common, "cyto_score"]
        merge["nuc_score"] = cn_df.loc[common, "nuc_score"]
        merge.to_csv(os.path.join(args.output_dir, "gene_delta_vs_cn.csv"))
        print(f"  Saved per-gene table: {len(merge)} genes")

    # === COMPARTMENTS knowledge (curated) ===
    if os.path.exists(args.compartments_knowledge):
        print("\nLoading COMPARTMENTS knowledge data...")
        loc_labels = load_compartments_knowledge(args.compartments_knowledge)
        print(f"  {len(loc_labels)} genes with curated localization")

        know_stats = knowledge_analysis(delta_df, loc_labels)
        results["compartments_knowledge"] = know_stats

        plot_distributions(delta_df, loc_labels,
                           os.path.join(args.output_dir, "delta_by_localization.png"))

    # === Summary ===
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    if "compartments_integrated" in results:
        s = results["compartments_integrated"]
        print(f"  Spearman rho:  {s['spearman_rho']:.4f}  (p={s['spearman_p']:.2e})")
        print(f"  AUROC:         {s['auroc']:.4f}")
        print(f"  AUPRC:         {s['auprc']:.4f}  (baseline={s['baseline_prevalence']:.4f})")
    if "compartments_knowledge" in results:
        k = results["compartments_knowledge"]
        print(f"  Curated cyto-only delta: {k['median_delta_cyto']:.3f}")
        print(f"  Curated nuc-only delta:  {k['median_delta_nuc']:.3f}")
        print(f"  Mann-Whitney p:          {k['mannwhitney_p']:.2e}")

    # Save
    out_json = os.path.join(args.output_dir, "localization_stats.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_json}")


if __name__ == "__main__":
    main()
