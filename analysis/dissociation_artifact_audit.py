"""
Dissociation artifact audit — check whether the model's predicted delta
is contaminated by known dissociation-induced gene expression artifacts.

Tests:
1. Are IEG/stress genes (van den Brink 2017) enriched in our top added genes?
2. Are glial activation artifacts (Marsh 2022) in our added set?
3. What are the specific deltas for known artifact genes?
4. Hypergeometric enrichment test of artifact genes in top-N added genes.

Usage:
    python analysis/dissociation_artifact_audit.py \
        --delta_csv notes/experiments/nuclear_lncrna_analysis/all_genes_delta.csv \
        --output_dir notes/experiments/dissociation_audit
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy import stats


# ============================================================================
# Dissociation-induced gene lists
# ============================================================================

# van den Brink et al. 2017 (Nature Methods) — core dissociation module
# These genes are reproducibly induced by warm enzymatic dissociation.
# Curated from their supplementary data + commonly cited subset.
VAN_DEN_BRINK_CORE = [
    # Immediate early genes (IEGs)
    "Fos", "Fosb", "Jun", "Junb", "Jund",
    "Egr1", "Egr2", "Egr3", "Egr4",
    "Arc", "Nr4a1", "Nr4a2", "Nr4a3",
    "Ier2", "Ier3", "Ier5",
    # Heat shock proteins
    "Hspa1a", "Hspa1b", "Hspa5", "Hspa8",
    "Hsp90aa1", "Hsp90ab1", "Hsp90b1",
    "Hsph1", "Dnajb1", "Dnaja1",
    # Stress response / AP-1 targets
    "Atf3", "Btg2", "Dusp1", "Dusp5", "Dusp6",
    "Gadd45b", "Gadd45g",
    "Nfkbia", "Nfkbiz",
    "Socs3", "Cyr61", "Ctgf",
    "Klf2", "Klf4", "Klf6",
    "Rhob", "Maff",
    # Cytokines / chemokines (dissociation-responsive)
    "Ccl2", "Ccl3", "Ccl4", "Cxcl1", "Cxcl2", "Cxcl10",
    "Il1b", "Il6", "Tnf",
]

# Marsh et al. 2022 (Nature Neuroscience) — glial/microglial activation artifacts
# Genes specifically activated by enzymatic dissociation in glia.
MARSH_GLIAL_ACTIVATION = [
    # Microglial activation markers (artifactual in dissociated tissue)
    "Cd83", "Cd69", "Ccl3", "Ccl4", "Ccl2",
    "Il1b", "Il1a", "Tnf", "Nfkbia", "Nfkbiz",
    "Socs3", "Cxcl2", "Cxcl10",
    # Astrocyte dissociation artifacts
    "Fos", "Fosb", "Jun", "Junb",
    "Egr1", "Egr2", "Egr3",
    "Nr4a1", "Nr4a2", "Nr4a3",
    "Hspa1a", "Hspa1b",
    "Atf3", "Btg2", "Dusp1",
    # Extended glial activation
    "Spp1", "Lpl", "Cst7", "Apoe",
]

# Combined unique set
ALL_ARTIFACT_GENES = sorted(set(VAN_DEN_BRINK_CORE + MARSH_GLIAL_ACTIVATION))


def run_audit(delta_csv, output_dir, top_n_list=(50, 100, 200, 500)):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(delta_csv)
    n_total = len(df)
    gene_set = set(df["gene"].values)

    # Which artifact genes are in our gene list?
    artifact_in_data = sorted(set(ALL_ARTIFACT_GENES) & gene_set)
    artifact_missing = sorted(set(ALL_ARTIFACT_GENES) - gene_set)

    print(f"Total genes in delta: {n_total}")
    print(f"Artifact genes in data: {len(artifact_in_data)} / {len(ALL_ARTIFACT_GENES)}")
    print(f"Artifact genes missing: {len(artifact_missing)}")

    # ---- Per-gene artifact delta table ----
    artifact_df = df[df["gene"].isin(artifact_in_data)].copy()
    artifact_df = artifact_df.sort_values("delta", ascending=False)
    artifact_df["source"] = artifact_df["gene"].apply(lambda g:
        "both" if g in set(VAN_DEN_BRINK_CORE) & set(MARSH_GLIAL_ACTIVATION)
        else "van_den_Brink" if g in VAN_DEN_BRINK_CORE
        else "Marsh")
    artifact_df.to_csv(os.path.join(output_dir, "artifact_gene_deltas.csv"), index=False)

    print(f"\n{'='*70}")
    print("Dissociation artifact genes — per-gene delta")
    print(f"{'='*70}")
    print(f"{'Gene':<12} {'Delta':>8} {'Cell mean':>10} {'Nuc mean':>10} {'Source':<16}")
    print(f"{'-'*56}")
    for _, row in artifact_df.iterrows():
        print(f"{row['gene']:<12} {row['delta']:>8.4f} {row['cell_mean']:>10.5f} "
              f"{row['nuc_mean']:>10.5f} {row['source']:<16}")

    # ---- Summary stats ----
    n_positive = (artifact_df["delta"] > 0).sum()
    n_negative = (artifact_df["delta"] <= 0).sum()
    mean_delta = artifact_df["delta"].mean()
    median_delta = artifact_df["delta"].median()

    print(f"\nArtifact gene summary:")
    print(f"  Positive delta (added by model): {n_positive}/{len(artifact_df)}")
    print(f"  Negative delta (removed):        {n_negative}/{len(artifact_df)}")
    print(f"  Mean delta:  {mean_delta:.4f}")
    print(f"  Median delta: {median_delta:.4f}")

    # ---- Compare artifact delta distribution vs all genes ----
    all_deltas = df["delta"].values
    artifact_deltas = artifact_df["delta"].values

    # Mann-Whitney U test: are artifact genes shifted toward positive delta?
    if len(artifact_deltas) > 1:
        u_stat, u_pval = stats.mannwhitneyu(artifact_deltas, all_deltas, alternative="greater")
    else:
        u_stat, u_pval = np.nan, np.nan

    print(f"\n  Mann-Whitney U (artifact > all genes): U={u_stat:.0f}, p={u_pval:.2e}")
    print(f"  All genes mean delta: {all_deltas.mean():.4f}")
    print(f"  All genes median delta: {np.median(all_deltas):.4f}")

    # ---- Enrichment in top-N added genes ----
    df_sorted = df.sort_values("delta", ascending=False)

    print(f"\n{'='*70}")
    print("Hypergeometric enrichment of artifact genes in top-N added genes")
    print(f"{'='*70}")
    print(f"{'Top-N':>6} {'Artifact in top-N':>18} {'Expected':>10} {'Fold':>6} {'p-value':>12}")
    print(f"{'-'*54}")

    enrichment_results = []
    for top_n in top_n_list:
        top_genes = set(df_sorted.head(top_n)["gene"].values)
        overlap = top_genes & set(artifact_in_data)
        k = len(overlap)  # successes in sample
        M = n_total       # population size
        n = len(artifact_in_data)  # successes in population
        N = top_n         # sample size
        expected = N * n / M
        fold = k / expected if expected > 0 else 0

        # Hypergeometric test (probability of >= k successes)
        pval = stats.hypergeom.sf(k - 1, M, n, N)

        print(f"{top_n:>6} {k:>18} {expected:>10.1f} {fold:>6.2f} {pval:>12.2e}")
        enrichment_results.append({
            "top_n": top_n, "artifact_count": k, "expected": expected,
            "fold_enrichment": fold, "p_value": pval,
            "artifact_genes": sorted(overlap),
        })

        if overlap:
            print(f"         Genes: {', '.join(sorted(overlap))}")

    # ---- Specific spotlight genes ----
    spotlight = ["Fos", "Fosb", "Jun", "Junb", "Arc", "Egr1", "Egr2",
                 "Hspa1a", "Hspa1b", "Nr4a1", "Atf3", "Dusp1", "Socs3"]
    print(f"\n{'='*70}")
    print("Spotlight: most-cited dissociation genes")
    print(f"{'='*70}")
    print(f"{'Gene':<10} {'Delta':>8} {'Rank':>6} {'Percentile':>12} {'In top 100?':>12}")
    print(f"{'-'*50}")
    top100 = set(df_sorted.head(100)["gene"].values)
    for gene in spotlight:
        row = df[df["gene"] == gene]
        if len(row) == 0:
            print(f"{gene:<10} {'NOT IN DATA':>8}")
            continue
        delta = row["delta"].values[0]
        rank = (df_sorted["gene"] == gene).values.argmax() + 1
        pctile = (1 - rank / n_total) * 100
        in_top = "YES" if gene in top100 else "no"
        print(f"{gene:<10} {delta:>8.4f} {rank:>6} {pctile:>11.1f}% {in_top:>12}")

    # ---- What ARE the top-20 added genes? (for context) ----
    print(f"\n{'='*70}")
    print("Top 20 added genes (positive delta) — for reference")
    print(f"{'='*70}")
    for i, (_, row) in enumerate(df_sorted.head(20).iterrows()):
        is_artifact = "* ARTIFACT" if row["gene"] in set(ALL_ARTIFACT_GENES) else ""
        print(f"  {i+1:>2}. {row['gene']:<15} delta={row['delta']:>7.4f} {is_artifact}")

    # ---- Save summary ----
    summary = {
        "n_total_genes": n_total,
        "n_artifact_genes_in_data": len(artifact_in_data),
        "n_artifact_genes_total": len(ALL_ARTIFACT_GENES),
        "artifact_positive_delta": int(n_positive),
        "artifact_negative_delta": int(n_negative),
        "artifact_mean_delta": float(mean_delta),
        "artifact_median_delta": float(median_delta),
        "all_genes_mean_delta": float(all_deltas.mean()),
        "all_genes_median_delta": float(np.median(all_deltas)),
        "mann_whitney_U": float(u_stat) if not np.isnan(u_stat) else None,
        "mann_whitney_p": float(u_pval) if not np.isnan(u_pval) else None,
        "enrichment": enrichment_results,
        "artifact_genes_missing_from_data": artifact_missing,
    }

    import json
    with open(os.path.join(output_dir, "audit_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved results to {output_dir}/")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta_csv", type=str,
                        default="notes/experiments/nuclear_lncrna_analysis/all_genes_delta.csv")
    parser.add_argument("--output_dir", type=str,
                        default="notes/experiments/dissociation_audit")
    args = parser.parse_args()
    run_audit(args.delta_csv, args.output_dir)


if __name__ == "__main__":
    main()
