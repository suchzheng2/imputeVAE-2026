"""
Task 3: Dissociation / IEG confound audit.

Checks whether the model's imputation delta is driven by dissociation-induced
immediate-early genes (IEGs) rather than genuine cytoplasmic biology.

Dissociation/stress gene module from van den Brink 2017 + Marsh 2022.
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELTA_CSV = os.path.join(PROJECT, "notes/experiments/nuclear_lncrna_analysis/all_genes_delta.csv")
TOP100_FILE = os.path.join(PROJECT, "notes/experiments/delta_analysis/top100_increased_genes.txt")
OUT_DIR = os.path.join(PROJECT, "notes/experiments/dissociation_audit")

# ── Dissociation / stress gene module ──────────────────────────────────────
# Core IEGs (van den Brink 2017, Marsh 2022, Wu 2017)
DISSOCIATION_GENES = [
    # Immediate-early / AP-1
    "Fos", "Fosb", "Jun", "Junb", "Jund",
    # Early growth response
    "Egr1", "Egr2", "Egr3", "Egr4",
    # Activity-regulated
    "Arc", "Npas4",
    # Heat shock
    "Hspa1a", "Hspa1b", "Hsp90aa1",
    # Immediate-early response
    "Ier2", "Ier3", "Ier5",
    # Dual-specificity phosphatases
    "Dusp1", "Dusp5", "Dusp6",
    # Other stress/dissociation
    "Zfp36", "Btg2", "Nr4a1", "Nr4a2", "Nr4a3",
    "Socs3", "Atf3", "Nfkbia", "Nfkbiz",
    # Marsh 2022 microglial ex-vivo activation
    "Ccl2", "Ccl3", "Ccl4", "Cxcl2", "Il1b",
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load delta
    df = pd.read_csv(DELTA_CSV)
    n_total = len(df)
    print(f"Loaded {n_total} genes from delta CSV")

    # Compute percentile rank (0-100, higher = larger positive delta)
    df["delta_rank"] = df["delta"].rank(pct=True) * 100

    # Load top-100 increased
    with open(TOP100_FILE) as f:
        top100 = [line.strip() for line in f if line.strip()]

    # ── Find module genes in our gene set ──────────────────────────────────
    module_genes = [g for g in DISSOCIATION_GENES if g in df["gene"].values]
    missing_genes = [g for g in DISSOCIATION_GENES if g not in df["gene"].values]

    print(f"\nDissociation module: {len(DISSOCIATION_GENES)} genes defined, "
          f"{len(module_genes)} found in dataset, {len(missing_genes)} missing")
    if missing_genes:
        print(f"  Missing: {', '.join(missing_genes)}")

    # ── Per-gene delta and rank ────────────────────────────────────────────
    module_df = df[df["gene"].isin(module_genes)].copy()
    module_df = module_df.sort_values("delta", ascending=False)

    print(f"\nDissociation module gene deltas:")
    print(f"{'Gene':<12} {'Delta':>8} {'Pctile':>8} {'In Top100':>10}")
    print("-" * 42)
    for _, row in module_df.iterrows():
        in_top100 = "YES" if row["gene"] in top100 else ""
        print(f"{row['gene']:<12} {row['delta']:>8.4f} {row['delta_rank']:>7.1f}% {in_top100:>10}")

    # ── Summary stats ──────────────────────────────────────────────────────
    module_deltas = module_df["delta"].values
    all_deltas = df["delta"].values
    median_module = np.median(module_deltas)
    median_all = np.median(all_deltas)
    mean_module = np.mean(module_deltas)
    mean_all = np.mean(all_deltas)

    print(f"\nModule median delta: {median_module:.4f} (all genes: {median_all:.4f})")
    print(f"Module mean delta:   {mean_module:.4f} (all genes: {mean_all:.4f})")
    print(f"Module median percentile: {module_df['delta_rank'].median():.1f}%")

    # ── Mann-Whitney U: are module genes shifted higher than background? ──
    non_module = df[~df["gene"].isin(module_genes)]["delta"].values
    u_stat, u_pval = stats.mannwhitneyu(module_deltas, non_module, alternative="greater")
    print(f"\nMann-Whitney U (module > background): U={u_stat:.0f}, p={u_pval:.4g}")

    # ── Hypergeometric test: enrichment in top-100 ─────────────────────────
    module_in_top100 = [g for g in module_genes if g in top100]
    # hypergeom.sf(k-1, M, n, N) = P(X >= k)
    # M = total genes, n = module size, N = top-100 size, k = overlap
    M = n_total
    n = len(module_genes)
    N = len(top100)
    k = len(module_in_top100)
    hyper_pval = stats.hypergeom.sf(k - 1, M, n, N) if k > 0 else 1.0
    expected = n * N / M

    print(f"\nHypergeometric enrichment in top-100 increased:")
    print(f"  Module genes in top-100: {k} / {n} (expected by chance: {expected:.2f})")
    if module_in_top100:
        print(f"  Genes: {', '.join(module_in_top100)}")
    print(f"  p-value: {hyper_pval:.4g}")
    print(f"  Enriched: {'YES' if hyper_pval < 0.05 else 'NO'}")

    # ── Also check top-200 and top-500 ─────────────────────────────────────
    top_genes_by_delta = df.nlargest(500, "delta")["gene"].values
    top200 = top_genes_by_delta[:200]
    top500 = top_genes_by_delta[:500]

    k200 = len([g for g in module_genes if g in top200])
    k500 = len([g for g in module_genes if g in top500])
    p200 = stats.hypergeom.sf(k200 - 1, M, n, 200) if k200 > 0 else 1.0
    p500 = stats.hypergeom.sf(k500 - 1, M, n, 500) if k500 > 0 else 1.0

    print(f"\n  In top-200: {k200}/{n} (expected {n*200/M:.2f}), p={p200:.4g}")
    print(f"  In top-500: {k500}/{n} (expected {n*500/M:.2f}), p={p500:.4g}")

    # ── Compare to ribosomal genes as positive control ─────────────────────
    ribo_genes = [g for g in df["gene"].values if g.startswith(("Rpl", "Rps"))]
    ribo_in_top100 = [g for g in ribo_genes if g in top100]
    ribo_median_pctile = df[df["gene"].isin(ribo_genes)]["delta_rank"].median()

    print(f"\nPositive control — ribosomal genes (Rpl*/Rps*):")
    print(f"  {len(ribo_genes)} found, {len(ribo_in_top100)} in top-100")
    print(f"  Median percentile: {ribo_median_pctile:.1f}%")

    # ── Save results ───────────────────────────────────────────────────────
    # Module gene table
    module_df[["gene", "delta", "delta_rank"]].to_csv(
        os.path.join(OUT_DIR, "module_gene_deltas.csv"), index=False
    )

    # Gene list used
    with open(os.path.join(OUT_DIR, "dissociation_gene_list.txt"), "w") as f:
        for g in DISSOCIATION_GENES:
            found = "found" if g in module_genes else "NOT_IN_DATASET"
            f.write(f"{g}\t{found}\n")

    # Stats JSON
    results = {
        "n_module_defined": len(DISSOCIATION_GENES),
        "n_module_found": len(module_genes),
        "n_missing": len(missing_genes),
        "missing_genes": missing_genes,
        "module_median_delta": float(median_module),
        "module_mean_delta": float(mean_module),
        "all_genes_median_delta": float(median_all),
        "all_genes_mean_delta": float(mean_all),
        "module_median_percentile": float(module_df["delta_rank"].median()),
        "mannwhitney_U": float(u_stat),
        "mannwhitney_p": float(u_pval),
        "mannwhitney_direction": "module > background" if u_pval < 0.05 else "not significant",
        "top100_overlap": k,
        "top100_overlap_genes": module_in_top100,
        "top100_expected": float(expected),
        "top100_hypergeom_p": float(hyper_pval),
        "top100_enriched": hyper_pval < 0.05,
        "top200_overlap": k200,
        "top200_hypergeom_p": float(p200),
        "top500_overlap": k500,
        "top500_hypergeom_p": float(p500),
        "ribo_n_found": len(ribo_genes),
        "ribo_in_top100": len(ribo_in_top100),
        "ribo_median_percentile": float(ribo_median_pctile),
    }
    with open(os.path.join(OUT_DIR, "dissociation_audit_stats.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ── Plot: module gene ranks vs all genes ───────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: histogram of all gene deltas with module genes marked
    ax = axes[0]
    ax.hist(all_deltas, bins=80, color="lightgray", edgecolor="gray", alpha=0.7, label="All genes")
    for _, row in module_df.iterrows():
        ax.axvline(row["delta"], color="red", alpha=0.5, linewidth=1)
    # Add invisible line for legend
    ax.axvline(0, color="red", alpha=0, linewidth=0)
    ax.plot([], [], color="red", linewidth=2, label=f"Dissociation module ({len(module_genes)} genes)")
    ax.set_xlabel("Delta (imputed - nuclei, log1p scale)")
    ax.set_ylabel("Number of genes")
    ax.set_title("Dissociation gene deltas vs all genes")
    ax.legend()

    # Right: percentile ranks of module genes, sorted
    ax = axes[1]
    sorted_module = module_df.sort_values("delta_rank")
    colors = ["red" if g in top100 else "steelblue" for g in sorted_module["gene"]]
    bars = ax.barh(range(len(sorted_module)), sorted_module["delta_rank"].values, color=colors)
    ax.set_yticks(range(len(sorted_module)))
    ax.set_yticklabels(sorted_module["gene"].values, fontsize=8)
    ax.set_xlabel("Percentile rank (higher = larger delta)")
    ax.set_title("Dissociation module: percentile ranks")
    ax.axvline(50, color="gray", linestyle="--", alpha=0.5, label="Median")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "dissociation_audit_plot.png"), dpi=150)
    plt.close()
    print(f"\nPlot saved to {OUT_DIR}/dissociation_audit_plot.png")

    # ── Conclusion ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    if hyper_pval >= 0.05 and module_df["delta_rank"].median() < 80:
        print("The dissociation/IEG module is NOT enriched among top-increased")
        print("genes. The imputation delta is driven by cytoplasmic biology")
        print("(ribosomal, synaptic), not dissociation artifacts.")
    elif hyper_pval < 0.05:
        print("WARNING: The dissociation module IS enriched among top-increased")
        print("genes. Some of the imputation delta may reflect dissociation")
        print("artifacts rather than pure cytoplasmic biology. Report honestly.")
    else:
        print("The dissociation module genes tend to have high deltas but are")
        print("not statistically enriched in the top-100. Investigate further.")


if __name__ == "__main__":
    main()
