"""Validate ImputeVAE compartment predictions against brain fractionation-seq data.

Uses two independent fractionation datasets where nuclear and cytosolic RNA
were physically separated and sequenced:

1. Buckley et al. 2016 (GSE73391) — Mouse brain, cell-type-specific nuclear
   RNA (neurons, astrocytes, oligodendrocytes) + bulk cytoplasmic fraction.
   Direct species match.

2. Moutinho et al. 2021 (GSE110727) — Human adult frontal cortex,
   matched nuclear and cytosolic fractions (6 replicates each).
   Cross-species but direct brain tissue match.

For each dataset, compute per-gene cytosolic/nuclear ratio and correlate
with our GT delta and ImputeVAE predicted delta.

Usage:
    python analysis/fractionation_validation.py --output_dir compartment_classification/fractionation_validation
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FRAC_DIR = "data/external/fractionation"
GENE_LIST_PATH = "data/training_gene_list_v3.csv"


# ---------------------------------------------------------------------------
# Gene ID mapping
# ---------------------------------------------------------------------------

def load_ensembl_mapping(species="mouse"):
    """Load Ensembl ID -> gene name mapping."""
    if species == "mouse":
        path = os.path.join(FRAC_DIR, "ensembl_mouse_id2name.csv")
    else:
        path = os.path.join(FRAC_DIR, "ensembl_human_id2name.csv")
    df = pd.read_csv(path)
    col_id = [c for c in df.columns if "stable" in c.lower() or "ensembl" in c.lower()][0]
    col_name = [c for c in df.columns if "name" in c.lower()][0]
    # Drop empty names and duplicates (keep first)
    df = df[[col_id, col_name]].dropna()
    df = df[df[col_name].str.len() > 0]
    df = df.drop_duplicates(subset=col_id, keep="first")
    mapping = dict(zip(df[col_id], df[col_name]))
    print(f"  Loaded {len(mapping)} {species} Ensembl->name mappings")
    return mapping


# ---------------------------------------------------------------------------
# Dataset 1: Buckley 2016 — Mouse brain
# ---------------------------------------------------------------------------

def load_buckley():
    """Load Buckley 2016 mouse brain fractionation data.

    Returns per-gene CN ratio using:
    - Nuclear: Sorted samples (flow-sorted nuclei) for each cell type
    - Cytoplasmic: Bulk cytoplasmic fraction (2 replicates, not cell-type-specific)
    """
    print("\n=== Loading Buckley 2016 (GSE73391) — Mouse brain fractionation ===")
    df = pd.read_csv(os.path.join(FRAC_DIR, "GSE73391_Counts.csv.gz"), index_col=0)

    # Map Ensembl IDs to gene names
    mapping = load_ensembl_mapping("mouse")
    df.index = df.index.map(lambda x: mapping.get(x, x))
    # Drop unmapped (still Ensembl IDs)
    mapped = ~df.index.str.startswith("ENSMUSG")
    df = df[mapped]
    # Drop duplicate gene names (keep highest total)
    df["_total"] = df.sum(axis=1)
    df = df.sort_values("_total", ascending=False).drop(columns="_total")
    df = df[~df.index.duplicated(keep="first")]
    print(f"  {df.shape[0]} genes after mapping and dedup")

    # Separate nuclear (Sorted) and cytoplasmic columns
    nuc_neuro_cols = [c for c in df.columns if "Sorted" in c and "Neuro" in c]
    nuc_astro_cols = [c for c in df.columns if "Sorted" in c and "Astro" in c]
    nuc_oligo_cols = [c for c in df.columns if "Sorted" in c and "Oligo" in c]
    nuc_all_cols = nuc_neuro_cols + nuc_astro_cols + nuc_oligo_cols
    cyto_cols = [c for c in df.columns if "Cytoplasmic" in c]

    print(f"  Nuclear neuron: {len(nuc_neuro_cols)} reps, Astro: {len(nuc_astro_cols)}, "
          f"Oligo: {len(nuc_oligo_cols)}")
    print(f"  Cytoplasmic (bulk): {len(cyto_cols)} reps")

    results = {}

    # Overall CN ratio: cytoplasmic vs all nuclear
    cyto_cpm = df[cyto_cols].mean(axis=1)
    cyto_cpm = cyto_cpm / cyto_cpm.sum() * 1e6

    for label, nuc_cols in [("all", nuc_all_cols),
                            ("neuron", nuc_neuro_cols),
                            ("astro", nuc_astro_cols),
                            ("oligo", nuc_oligo_cols)]:
        nuc_cpm = df[nuc_cols].mean(axis=1)
        nuc_cpm = nuc_cpm / nuc_cpm.sum() * 1e6

        # CN ratio = log2(cyto / nuc), require both > 1 CPM
        valid = (cyto_cpm > 1) & (nuc_cpm > 1)
        cn_ratio = np.log2(cyto_cpm[valid] / nuc_cpm[valid])
        results[f"buckley_{label}"] = cn_ratio
        print(f"  {label}: {valid.sum()} genes with CPM>1 in both fractions, "
              f"median CN ratio = {cn_ratio.median():.3f}")

    return results


# ---------------------------------------------------------------------------
# Dataset 2: Moutinho 2021 — Human adult frontal cortex
# ---------------------------------------------------------------------------

def load_moutinho():
    """Load Moutinho 2021 human brain fractionation data.

    Uses adult frontal cortex samples (6 cytosolic + 6 nuclear replicates).
    Maps human gene names to mouse via case conversion.
    """
    print("\n=== Loading Moutinho 2021 (GSE110727) — Human brain fractionation ===")
    df = pd.read_csv(os.path.join(FRAC_DIR, "GSE110727_countMatrix_all.txt.gz"),
                     sep="\t", index_col=0)

    # Map Ensembl IDs to gene names
    mapping = load_ensembl_mapping("human")
    df.index = df.index.map(lambda x: mapping.get(x, x))
    mapped = ~df.index.str.startswith("ENSG")
    df = df[mapped]
    df["_total"] = df.sum(axis=1)
    df = df.sort_values("_total", ascending=False).drop(columns="_total")
    df = df[~df.index.duplicated(keep="first")]
    print(f"  {df.shape[0]} genes after mapping and dedup")

    # Use adult frontal cortex only
    cyto_cols = [c for c in df.columns if c.startswith("fc_adult_cyto")]
    nuc_cols = [c for c in df.columns if c.startswith("fc_adult_nuc")]
    print(f"  Adult FC: {len(cyto_cols)} cyto reps, {len(nuc_cols)} nuc reps")

    cyto_cpm = df[cyto_cols].mean(axis=1)
    cyto_cpm = cyto_cpm / cyto_cpm.sum() * 1e6
    nuc_cpm = df[nuc_cols].mean(axis=1)
    nuc_cpm = nuc_cpm / nuc_cpm.sum() * 1e6

    valid = (cyto_cpm > 1) & (nuc_cpm > 1)
    cn_ratio = np.log2(cyto_cpm[valid] / nuc_cpm[valid])

    # Convert human gene names to mouse-style (capitalize first letter only)
    # e.g., MALAT1 -> Malat1, RPL3 -> Rpl3
    cn_ratio_mouse = cn_ratio.copy()
    cn_ratio_mouse.index = cn_ratio_mouse.index.map(
        lambda x: x[0].upper() + x[1:].lower() if len(x) > 1 else x
    )
    # Deduplicate after case conversion
    cn_ratio_mouse = cn_ratio_mouse[~cn_ratio_mouse.index.duplicated(keep="first")]

    print(f"  {valid.sum()} genes with CPM>1, {len(cn_ratio_mouse)} after mouse-name conversion")
    print(f"  Median CN ratio = {cn_ratio.median():.3f}")

    return {"moutinho_adult_fc": cn_ratio_mouse}


# ---------------------------------------------------------------------------
# Correlation with GT delta and predicted deltas
# ---------------------------------------------------------------------------

def correlate_with_deltas(cn_ratios, output_dir):
    """Correlate fractionation CN ratios with our GT delta and predicted deltas."""
    print("\n=== Correlating fractionation CN ratios with model deltas ===")

    # Load deltas from compartment_classification
    gt_global = pd.read_csv("compartment_classification/gt_delta_global.csv", index_col=0)
    pred_global = pd.read_csv("compartment_classification/predicted_deltas_global.csv", index_col=0)

    gene_list = pd.read_csv(GENE_LIST_PATH, header=None)[0].tolist()

    all_results = {}

    for frac_name, cn_ratio in cn_ratios.items():
        print(f"\n--- {frac_name} ---")

        # Match genes
        common = cn_ratio.index.intersection(gt_global.index).intersection(gene_list)
        if len(common) < 100:
            print(f"  Only {len(common)} genes in common, skipping")
            continue
        print(f"  {len(common)} genes in common with our gene list")

        cn = cn_ratio.loc[common].values
        gt = gt_global.loc[common, "gt_delta"].values

        # GT delta vs fractionation CN ratio
        rho_gt, p_gt = stats.spearmanr(cn, gt)
        r_gt, pr_gt = stats.pearsonr(cn, gt)

        # AUROC: can CN ratio classify our GT delta sign?
        gt_sign = (gt > 0).astype(int)
        if gt_sign.sum() > 10 and (1 - gt_sign).sum() > 10:
            auroc_gt = roc_auc_score(gt_sign, cn)
        else:
            auroc_gt = float("nan")

        # Also: can our GT delta classify fractionation sign?
        cn_sign = (cn > 0).astype(int)
        if cn_sign.sum() > 10 and (1 - cn_sign).sum() > 10:
            auroc_cn = roc_auc_score(cn_sign, gt)
        else:
            auroc_cn = float("nan")

        frac_result = {
            "n_genes": len(common),
            "GT_delta": {
                "spearman_rho": float(rho_gt),
                "spearman_p": float(p_gt),
                "pearson_r": float(r_gt),
                "pearson_p": float(pr_gt),
                "auroc_cn_classifies_gt": float(auroc_gt),
                "auroc_gt_classifies_cn": float(auroc_cn),
            },
        }
        print(f"  GT delta vs CN ratio: Spearman rho = {rho_gt:.4f} (p = {p_gt:.2e}), "
              f"AUROC = {auroc_cn:.4f}")

        # Each method's predicted delta
        methods = ["ImputeVAE", "CT-mean", "Linear_reg", "scVI", "scANVI",
                    "scGen", "scButterfly"]
        for method in methods:
            if method not in pred_global.columns:
                continue
            pred = pred_global.loc[common, method].values
            if np.std(pred) < 1e-12:
                frac_result[method] = {"spearman_rho": 0.0, "note": "constant prediction"}
                continue

            rho_m, p_m = stats.spearmanr(cn, pred)
            r_m, pr_m = stats.pearsonr(cn, pred)

            if cn_sign.sum() > 10 and (1 - cn_sign).sum() > 10:
                auroc_m = roc_auc_score(cn_sign, pred)
            else:
                auroc_m = float("nan")

            frac_result[method] = {
                "spearman_rho": float(rho_m),
                "spearman_p": float(p_m),
                "pearson_r": float(r_m),
                "pearson_p": float(pr_m),
                "auroc_pred_classifies_cn": float(auroc_m),
            }

        # Print comparison table
        print(f"\n  {'Method':<15s} {'Spearman rho':>12s} {'Pearson r':>10s} {'AUROC':>8s}")
        print(f"  {'-'*15} {'-'*12} {'-'*10} {'-'*8}")
        print(f"  {'GT delta':<15s} {rho_gt:>12.4f} {r_gt:>10.4f} {auroc_cn:>8.4f}")
        for method in methods:
            d = frac_result.get(method, {})
            if "spearman_rho" in d and "pearson_r" in d:
                auroc_s = f"{d.get('auroc_pred_classifies_cn', float('nan')):>8.4f}"
                print(f"  {method:<15s} {d['spearman_rho']:>12.4f} "
                      f"{d['pearson_r']:>10.4f} {auroc_s}")

        all_results[frac_name] = frac_result

        # Save per-gene data
        per_gene = pd.DataFrame({
            "cn_ratio": cn_ratio.reindex(common),
            "gt_delta": gt_global.loc[common, "gt_delta"],
        })
        for method in methods:
            if method in pred_global.columns:
                per_gene[f"pred_{method}"] = pred_global.loc[common, method]
        per_gene.to_csv(os.path.join(output_dir, f"{frac_name}_per_gene.csv"))

    return all_results


# ---------------------------------------------------------------------------
# Spot-check known genes
# ---------------------------------------------------------------------------

def spot_check(cn_ratios):
    """Print CN ratios for well-known compartment markers."""
    print("\n=== Spot-check: known compartment markers ===")
    markers = {
        "Cytoplasmic (expect CN > 0)": [
            "Rpl3", "Rpl7", "Rps3", "Actb", "Eef1a1", "Calm1", "Vamp2",
            "Snap25", "Gapdh",
        ],
        "Nuclear (expect CN < 0)": [
            "Malat1", "Neat1", "Xist", "Meg3",
        ],
    }

    for frac_name, cn_ratio in cn_ratios.items():
        print(f"\n  --- {frac_name} ---")
        for category, genes in markers.items():
            print(f"  {category}:")
            for g in genes:
                if g in cn_ratio.index:
                    val = cn_ratio[g]
                    flag = "OK" if ("CN > 0" in category) == (val > 0) else "MISMATCH"
                    print(f"    {g:12s}  CN ratio = {val:+.3f}  [{flag}]")
                else:
                    print(f"    {g:12s}  not found")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",
                        default="compartment_classification/fractionation_validation")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load both fractionation datasets
    buckley_ratios = load_buckley()
    moutinho_ratios = load_moutinho()

    all_cn_ratios = {**buckley_ratios, **moutinho_ratios}

    # Spot-check known genes
    spot_check(all_cn_ratios)

    # Correlate with our deltas
    results = correlate_with_deltas(all_cn_ratios, args.output_dir)

    # Save results
    output = {
        "buckley_2016": {
            "source": "GSE73391, Buckley et al. 2016, mouse brain",
            "description": "Cell-type-specific nuclear RNA (neurons/astro/oligo) + bulk cytoplasmic",
            "species": "mouse",
            "note": "Cytoplasmic fraction is bulk (not cell-type-specific); nuclear is cell-type-sorted",
        },
        "moutinho_2021": {
            "source": "GSE110727, Moutinho et al. 2021, human adult frontal cortex",
            "description": "Matched nuclear and cytosolic fractions, 6 replicates each",
            "species": "human (mapped to mouse gene names via capitalization)",
            "note": "Cross-species; adult frontal cortex used",
        },
        "results": results,
    }

    with open(os.path.join(args.output_dir, "fractionation_results.json"), "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for frac_name, res in results.items():
        gt_rho = res["GT_delta"]["spearman_rho"]
        gt_auroc = res["GT_delta"]["auroc_gt_classifies_cn"]
        vae_rho = res.get("ImputeVAE", {}).get("spearman_rho", float("nan"))
        n = res["n_genes"]
        print(f"  {frac_name}: {n} genes, "
              f"GT rho={gt_rho:.4f}, VAE rho={vae_rho:.4f}, AUROC={gt_auroc:.4f}")

    print(f"\nResults saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
