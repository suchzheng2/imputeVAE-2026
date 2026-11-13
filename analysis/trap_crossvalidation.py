"""Cross-validate ImputeVAE predictions against TRAP/RiboTag data.

Uses two independent mouse cortex datasets:
  1. GSE141464 (Schmidt et al.) — S100a10-TRAP in cortical L5a pyramidal neurons.
     Has both IP (ribosome-bound) and Input (total RNA) for saline condition.
     log2(IP/Input) = cytoplasmic translational enrichment per gene.

  2. GSE77243 (Greenberg lab, Spiegel et al. 2014) — RiboTag in 5 neuron types
     (Emx1, Gad2, PV, SST, VIP) in mouse visual cortex, standard housing.
     IP-only but enables cell-type-specific comparisons.

Validation logic:
  - TRAP enrichment (log2 IP/Input) is an INDEPENDENT measure of cytoplasmic
    RNA engagement, from a different lab, technology, and brain region.
  - If our model's predicted delta (imputed_cell - raw_nucleus) correlates
    with TRAP enrichment, it confirms the model captures real subcellular
    biology, not just dataset-specific artifacts.
  - This is NON-CIRCULAR: the model never saw TRAP data during training.

Usage:
    python analysis/trap_crossvalidation.py --output_dir compartment_classification/trap_validation
"""

import argparse
import json
import os
import sys
import gzip

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GENE_LIST_PATH = "data/training_gene_list_v3.csv"
COMMON_GENES_PATH = "data/common_genes.csv"
COMMON_CT_PATH = "data/common_cell_types.csv"
DATA_ROOT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi"
TRAP_DIR = "data/external/furlanis2019"
GREENBERG_XLSX = os.path.join(TRAP_DIR, "GSE77243_Greenberg_SupplementaryTable1.xlsx")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_raw_modality(data_dir, batch_suffix, gene_list, common_genes, common_ct):
    """Load one modality from raw h5ad, filter, log1p-normalize."""
    import anndata as ad
    raw_path = os.path.join(DATA_ROOT, data_dir, "data.h5ad")
    print(f"  Loading {raw_path} ...")
    adata = ad.read_h5ad(raw_path)
    if "meta_batch_id" not in adata.obs:
        adata.obs["meta_batch_id"] = adata.obs["batch_id"].apply(lambda x: x.split(".")[0])
    row_mask = (adata.obs["meta_batch_id"].str.endswith(batch_suffix) &
                adata.obs["cell_type"].isin(common_ct))
    col_mask = (adata.var["gene_names"].isin(common_genes) &
                adata.var["gene_names"].isin(gene_list))
    adata = adata[row_mask, col_mask].copy()
    adata = adata[:, adata.var["gene_names"].argsort()].copy()
    X = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
    X = np.log1p(X).astype(np.float32)
    gene_names_sorted = list(adata.var["gene_names"].values)
    ct_labels = adata.obs["cell_type"].values.astype(str)
    n = adata.shape[0]
    print(f"    {n} observations, {len(gene_names_sorted)} genes")
    del adata
    return X, ct_labels, gene_names_sorted, n


def load_raw_data():
    """Load raw cell and nucleus matrices."""
    gene_list = pd.read_csv(GENE_LIST_PATH, header=None)[0].tolist()
    common_genes = pd.read_csv(COMMON_GENES_PATH, header=0)["common_genes"].tolist()
    common_ct = pd.read_csv(COMMON_CT_PATH, header=0)["common_cell_types"].tolist()
    print("Loading raw cell data...")
    X_cell, ct_cell, gene_names, n_cell = load_raw_modality(
        "10X_cells_v2_v3_AIBS", "c3A", gene_list, common_genes, common_ct)
    print("Loading raw nucleus data...")
    X_nuc, ct_nuc, gene_names2, n_nuc = load_raw_modality(
        "10X_nuclei_v2_v3_AIBS", "n3A", gene_list, common_genes, common_ct)
    assert gene_names == gene_names2
    return X_cell, ct_cell, X_nuc, ct_nuc, gene_names, n_cell, n_nuc


def compute_wasserstein_per_ct(X_cell, ct_cell, X_nuc, ct_nuc, gene_names, cell_types):
    """Compute signed Wasserstein distance between cell and nucleus distributions per gene per CT.

    Signed Wasserstein = sign(mean_cell - mean_nucleus) × W1(cell_dist, nucleus_dist)
    This preserves directionality: positive = cytoplasm-enriched, negative = nuclear-enriched.
    """
    print("\n=== Computing per-gene per-CT Wasserstein distances ===")
    results = {}
    for ct in cell_types:
        cell_mask = ct_cell == ct
        nuc_mask = ct_nuc == ct
        n_c, n_n = cell_mask.sum(), nuc_mask.sum()
        if n_c < 10 or n_n < 10:
            print(f"  {ct}: skipping (n_cell={n_c}, n_nuc={n_n})")
            continue

        X_c = X_cell[cell_mask]  # (n_cells, n_genes)
        X_n = X_nuc[nuc_mask]    # (n_nuclei, n_genes)

        w_signed = np.zeros(len(gene_names), dtype=np.float64)
        w_unsigned = np.zeros(len(gene_names), dtype=np.float64)
        for j in range(len(gene_names)):
            cell_vals = X_c[:, j]
            nuc_vals = X_n[:, j]
            w = wasserstein_distance(cell_vals, nuc_vals)
            sign = 1.0 if cell_vals.mean() >= nuc_vals.mean() else -1.0
            w_unsigned[j] = w
            w_signed[j] = sign * w

        results[ct] = pd.DataFrame({
            "wasserstein_signed": w_signed,
            "wasserstein_unsigned": w_unsigned,
        }, index=gene_names)
        print(f"  {ct}: n_cell={n_c}, n_nuc={n_n}, median |W|={np.median(w_unsigned):.4f}")

    return results


def load_gse141464_enrichment():
    """Load GSE141464 IP and Input counts, compute log2(IP/Input) per gene."""
    print("=== Loading GSE141464 (S100a10-TRAP, cortical L5a pyramidal neurons) ===")

    def read_counts(pattern):
        files = sorted([
            f for f in os.listdir(TRAP_DIR)
            if pattern in f and f.endswith(".gz") and f.startswith("GSM")
        ])
        dfs = []
        for f in files:
            path = os.path.join(TRAP_DIR, f)
            d = pd.read_csv(path, sep="\t", header=None, names=["gene", "count"])
            d = d.set_index("gene")
            dfs.append(d)
        combined = pd.concat(dfs, axis=1)
        combined.columns = [f"rep{i}" for i in range(len(dfs))]
        return combined

    ip_df = read_counts("SAL_IP")
    inp_df = read_counts("SAL_INPUT")
    print(f"  IP: {ip_df.shape[1]} reps, Input: {inp_df.shape[1]} reps, {ip_df.shape[0]} genes")

    # Average across replicates
    ip_mean = ip_df.mean(axis=1)
    inp_mean = inp_df.mean(axis=1)

    # CPM normalize each fraction independently
    ip_cpm = ip_mean / ip_mean.sum() * 1e6
    inp_cpm = inp_mean / inp_mean.sum() * 1e6

    # log2 enrichment; require both > 0
    valid = (ip_cpm > 1) & (inp_cpm > 1)  # min 1 CPM in both
    enrichment = np.log2(ip_cpm[valid] / inp_cpm[valid])
    print(f"  {valid.sum()} genes with CPM > 1 in both fractions")
    return enrichment


def load_greenberg_ribotag():
    """Load GSE77243 RiboTag expression for 5 neuron types at standard housing."""
    print("\n=== Loading GSE77243 (Greenberg RiboTag, 5 neuron types, visual cortex) ===")
    df = pd.read_excel(GREENBERG_XLSX, sheet_name="allgenes.csv")
    df = df.set_index("Genenames")

    # Use standard-housing mean columns only
    std_cols = {
        "Emx1": "Emx1_Std_Mean",
        "Gad2": "Gad2_Std_Mean",
        "PV": "Pv_Std_Mean",
        "SST": "Sst_Std_Mean",
        "VIP": "Vip_Std_Mean",
    }
    result = pd.DataFrame({name: df[col] for name, col in std_cols.items()})
    print(f"  {result.shape[0]} genes, {result.shape[1]} neuron types")
    return result


def load_model_deltas():
    """Load ImputeVAE predicted deltas (global and per-CT) from compartment_classification."""
    print("\n=== Loading ImputeVAE predicted deltas ===")
    pred_global = pd.read_csv(
        "compartment_classification/predicted_deltas_global.csv", index_col=0
    )
    gt_global = pd.read_csv(
        "compartment_classification/gt_delta_global.csv", index_col=0
    )
    gt_per_ct = pd.read_csv(
        "compartment_classification/gt_delta_per_ct.csv", index_col=0
    )
    print(f"  {pred_global.shape[0]} genes, methods: {list(pred_global.columns)}")
    return pred_global, gt_global, gt_per_ct


# ---------------------------------------------------------------------------
# Validation 1: GSE141464 IP/Input enrichment vs predicted delta
# ---------------------------------------------------------------------------

def validate_ip_input(enrichment, pred_global, gt_global, gene_list, output_dir):
    """Correlate TRAP enrichment with model's predicted delta and GT delta."""
    print("\n=== Validation 1: TRAP enrichment vs predicted delta ===")
    print("  (S100a10-TRAP, L5a pyramidal neurons, mouse cerebral cortex)")

    # Restrict to our gene list
    common = enrichment.index.intersection(gene_list)
    enrichment = enrichment.loc[common]
    print(f"  {len(common)} genes in common with our gene list")

    results = {}

    # Compare with each method's global predicted delta
    methods = ["ImputeVAE", "CT-mean", "Linear_reg", "scVI", "scANVI",
               "scGen", "scButterfly"]
    for method in methods:
        if method not in pred_global.columns:
            continue
        pred_delta = pred_global[method].reindex(common).dropna()
        shared = enrichment.index.intersection(pred_delta.index)
        if len(shared) < 100:
            continue

        e = enrichment.loc[shared].values
        p = pred_delta.loc[shared].values

        rho, p_rho = stats.spearmanr(e, p)
        r, p_r = stats.pearsonr(e, p)

        # AUROC: can predicted delta discriminate ribosome-enriched (e > 0)
        # from ribosome-depleted (e < 0)?
        binary = (e > 0).astype(int)
        if binary.sum() > 10 and (1 - binary).sum() > 10:
            auroc = roc_auc_score(binary, p)
        else:
            auroc = float("nan")

        results[method] = {
            "n_genes": len(shared),
            "spearman_rho": float(rho),
            "spearman_p": float(p_rho),
            "pearson_r": float(r),
            "pearson_p": float(p_r),
            "auroc_enriched_vs_depleted": float(auroc),
        }

        if method == "ImputeVAE":
            print(f"  ImputeVAE: Spearman rho = {rho:.4f} (p = {p_rho:.2e}), "
                  f"AUROC = {auroc:.4f}")

    # Also compare GT delta with enrichment (sanity check: does the biology match?)
    gt_delta = gt_global["gt_delta"].reindex(common).dropna()
    shared_gt = enrichment.index.intersection(gt_delta.index)
    e_gt = enrichment.loc[shared_gt].values
    g_gt = gt_delta.loc[shared_gt].values
    rho_gt, p_gt = stats.spearmanr(e_gt, g_gt)
    r_gt, pr_gt = stats.pearsonr(e_gt, g_gt)
    binary_gt = (e_gt > 0).astype(int)
    auroc_gt = roc_auc_score(binary_gt, g_gt) if binary_gt.sum() > 10 else float("nan")

    results["GT_delta"] = {
        "n_genes": len(shared_gt),
        "spearman_rho": float(rho_gt),
        "spearman_p": float(p_gt),
        "pearson_r": float(r_gt),
        "pearson_p": float(pr_gt),
        "auroc_enriched_vs_depleted": float(auroc_gt),
        "note": "Empirical cell-nucleus delta vs TRAP enrichment (both are ground truth, different experiments)",
    }
    print(f"  GT delta: Spearman rho = {rho_gt:.4f} (p = {p_gt:.2e}), AUROC = {auroc_gt:.4f}")

    # Save per-gene file
    merged = pd.DataFrame({
        "trap_enrichment": enrichment.reindex(common),
        "gt_delta": gt_global["gt_delta"].reindex(common),
        "pred_delta_vae": pred_global["ImputeVAE"].reindex(common) if "ImputeVAE" in pred_global.columns else np.nan,
    })
    merged.to_csv(os.path.join(output_dir, "trap_enrichment_vs_delta.csv"))

    # Spot-check known genes
    print("\n  Known gene spot-checks (TRAP enrichment | GT delta | ImputeVAE pred delta):")
    spot_genes = ["Malat1", "Neat1", "Xist", "Snap25", "Actb", "Rpl3", "Rpl7",
                  "Eef1a1", "Calm1", "Vamp2", "Gad1", "Sst", "Pvalb"]
    for g in spot_genes:
        if g in merged.index:
            row = merged.loc[g]
            trap = row["trap_enrichment"]
            gt = row["gt_delta"]
            pred = row["pred_delta_vae"]
            trap_s = f"{trap:+.3f}" if not np.isnan(trap) else "N/A"
            gt_s = f"{gt:+.3f}" if not np.isnan(gt) else "N/A"
            pred_s = f"{pred:+.3f}" if not np.isnan(pred) else "N/A"
            print(f"    {g:12s}  TRAP={trap_s:>8s}  GT={gt_s:>8s}  pred={pred_s:>8s}")

    return results


# ---------------------------------------------------------------------------
# Validation 2: Cell-type-specific TRAP vs cell-type-specific delta
# ---------------------------------------------------------------------------

def validate_celltype_specific(greenberg, pred_global, gt_per_ct, gene_list, output_dir):
    """Test if cell-type-specific ribosome engagement matches cell-type-specific delta."""
    print("\n=== Validation 2: Cell-type-specific TRAP vs delta ===")
    print("  (Greenberg RiboTag: Emx1/Gad2/PV/SST/VIP in visual cortex)")
    print("  (Our model: predicted delta per cell type in MOp)")

    # Map Greenberg neuron types to our cell types
    # Emx1 = excitatory neurons → best match: L5 IT (largest excitatory type)
    # PV → Pvalb, SST → Sst, VIP → Vip
    # Gad2 = all GABAergic → average of Pvalb, Sst, Vip, Lamp5, Sncg
    ct_mapping = {
        "PV": "Pvalb",
        "SST": "Sst",
        "VIP": "Vip",
    }

    # Restrict to common genes
    common = greenberg.index.intersection(gene_list)
    greenberg = greenberg.loc[common]
    print(f"  {len(common)} genes in common")

    results = {}

    # For each mapped pair, correlate log(TRAP_expression) with GT delta
    for trap_type, our_type in ct_mapping.items():
        if our_type not in gt_per_ct.columns:
            continue

        trap_expr = greenberg[trap_type].reindex(common)
        gt_delta = gt_per_ct[our_type].reindex(common)

        valid = (trap_expr > 0) & ~np.isnan(gt_delta)
        if valid.sum() < 100:
            continue

        log_trap = np.log2(trap_expr[valid].values + 1)
        gt_d = gt_delta[valid].values

        rho, p_rho = stats.spearmanr(log_trap, gt_d)
        results[f"{trap_type}_vs_{our_type}_GT"] = {
            "n_genes": int(valid.sum()),
            "spearman_rho": float(rho),
            "spearman_p": float(p_rho),
        }
        print(f"  {trap_type} TRAP vs {our_type} GT delta: rho = {rho:.4f} (p = {p_rho:.2e})")

    # Cross-type specificity test:
    # For PV vs SST, compute differential TRAP = log2(PV/SST).
    # Then compute differential delta = GT_delta_Pvalb - GT_delta_Sst.
    # If our model captures cell-type-specific biology, these should correlate.
    print("\n  Cross-type specificity tests:")
    pairs = [("PV", "SST", "Pvalb", "Sst"),
             ("PV", "VIP", "Pvalb", "Vip"),
             ("SST", "VIP", "Sst", "Vip")]
    for trap_a, trap_b, our_a, our_b in pairs:
        if our_a not in gt_per_ct.columns or our_b not in gt_per_ct.columns:
            continue

        ta = greenberg[trap_a].reindex(common)
        tb = greenberg[trap_b].reindex(common)
        da = gt_per_ct[our_a].reindex(common)
        db = gt_per_ct[our_b].reindex(common)

        valid = (ta > 10) & (tb > 10) & ~np.isnan(da) & ~np.isnan(db)
        if valid.sum() < 100:
            continue

        diff_trap = np.log2(ta[valid].values / tb[valid].values)
        diff_delta = da[valid].values - db[valid].values

        rho, p_rho = stats.spearmanr(diff_trap, diff_delta)
        results[f"diff_{trap_a}vs{trap_b}_{our_a}vs{our_b}"] = {
            "n_genes": int(valid.sum()),
            "spearman_rho": float(rho),
            "spearman_p": float(p_rho),
        }
        print(f"    diff TRAP({trap_a}/{trap_b}) vs diff delta({our_a}-{our_b}): "
              f"rho = {rho:.4f} (p = {p_rho:.2e})")

    # Cell-type marker validation:
    # Genes that are differentially ribosome-engaged in one neuron type should
    # show corresponding differential delta.
    print("\n  Marker gene validation:")
    markers = {
        "Gad1": {"expect_high": ["PV", "SST", "VIP"], "expect_low": ["Emx1"],
                 "biology": "GABAergic marker"},
        "Gad2": {"expect_high": ["PV", "SST", "VIP"], "expect_low": ["Emx1"],
                 "biology": "GABAergic marker"},
        "Slc17a7": {"expect_high": ["Emx1"], "expect_low": ["PV", "SST", "VIP"],
                    "biology": "Vesicular glutamate transporter (excitatory)"},
        "Pvalb": {"expect_high": ["PV"], "expect_low": ["Emx1", "SST", "VIP"],
                  "biology": "Parvalbumin (PV marker)"},
        "Sst": {"expect_high": ["SST"], "expect_low": ["Emx1", "PV", "VIP"],
                "biology": "Somatostatin (SST marker)"},
        "Vip": {"expect_high": ["VIP"], "expect_low": ["Emx1", "PV", "SST"],
                "biology": "VIP (VIP marker)"},
    }
    for gene, info in markers.items():
        if gene not in greenberg.index:
            continue
        vals = greenberg.loc[gene]
        high_mean = np.mean([vals[t] for t in info["expect_high"] if t in vals.index])
        low_mean = np.mean([vals[t] for t in info["expect_low"] if t in vals.index])
        ratio = high_mean / max(low_mean, 1)
        print(f"    {gene:12s} ({info['biology']:35s}): "
              f"high={high_mean:.0f}, low={low_mean:.0f}, ratio={ratio:.1f}x")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def validate_celltype_wasserstein(greenberg, wasserstein_dists, gene_list, output_dir):
    """Correlate TRAP expression with Wasserstein distance instead of mean delta."""
    print("\n=== Validation 3: Cell-type-specific TRAP vs Wasserstein distance ===")

    ct_mapping = {"PV": "Pvalb", "SST": "Sst", "VIP": "Vip"}
    common = greenberg.index.intersection(gene_list)
    greenberg = greenberg.loc[common]

    results = {}
    rows = []

    for trap_type, our_type in ct_mapping.items():
        if our_type not in wasserstein_dists:
            continue

        w_df = wasserstein_dists[our_type]
        trap_expr = greenberg[trap_type].reindex(common)

        # Match genes
        shared = common.intersection(w_df.index)
        trap_vals = trap_expr.loc[shared]
        w_signed = w_df.loc[shared, "wasserstein_signed"]
        w_unsigned = w_df.loc[shared, "wasserstein_unsigned"]

        valid = (trap_vals > 0) & ~np.isnan(w_signed)
        if valid.sum() < 100:
            continue

        log_trap = np.log2(trap_vals[valid].values + 1)
        ws = w_signed[valid].values
        wu = w_unsigned[valid].values

        rho_signed, p_signed = stats.spearmanr(log_trap, ws)
        rho_unsigned, p_unsigned = stats.spearmanr(log_trap, wu)

        results[f"{trap_type}_vs_{our_type}_signed_W"] = {
            "n_genes": int(valid.sum()),
            "spearman_rho": float(rho_signed),
            "spearman_p": float(p_signed),
        }
        results[f"{trap_type}_vs_{our_type}_unsigned_W"] = {
            "n_genes": int(valid.sum()),
            "spearman_rho": float(rho_unsigned),
            "spearman_p": float(p_unsigned),
        }
        print(f"  {trap_type} TRAP vs {our_type} signed W1:   rho = {rho_signed:.4f} (n={valid.sum()})")
        print(f"  {trap_type} TRAP vs {our_type} unsigned W1: rho = {rho_unsigned:.4f}")

        # Save per-gene data
        for g in shared[valid]:
            rows.append({
                "gene": g,
                "trap_type": trap_type,
                "our_type": our_type,
                "log2_trap": np.log2(trap_vals[g] + 1),
                "wasserstein_signed": w_df.loc[g, "wasserstein_signed"],
                "wasserstein_unsigned": w_df.loc[g, "wasserstein_unsigned"],
            })

    # Cross-type differential Wasserstein
    print("\n  Cross-type differential Wasserstein tests:")
    pairs = [("PV", "SST", "Pvalb", "Sst"),
             ("PV", "VIP", "Pvalb", "Vip"),
             ("SST", "VIP", "Sst", "Vip")]
    for trap_a, trap_b, our_a, our_b in pairs:
        if our_a not in wasserstein_dists or our_b not in wasserstein_dists:
            continue

        ta = greenberg[trap_a].reindex(common)
        tb = greenberg[trap_b].reindex(common)
        wa = wasserstein_dists[our_a]["wasserstein_signed"].reindex(common)
        wb = wasserstein_dists[our_b]["wasserstein_signed"].reindex(common)

        valid = (ta > 10) & (tb > 10) & ~np.isnan(wa) & ~np.isnan(wb)
        if valid.sum() < 100:
            continue

        diff_trap = np.log2(ta[valid].values / tb[valid].values)
        diff_w = wa[valid].values - wb[valid].values

        rho, p_rho = stats.spearmanr(diff_trap, diff_w)
        results[f"diff_{trap_a}vs{trap_b}_{our_a}vs{our_b}_W"] = {
            "n_genes": int(valid.sum()),
            "spearman_rho": float(rho),
            "spearman_p": float(p_rho),
        }
        print(f"    diff TRAP({trap_a}/{trap_b}) vs diff W({our_a}-{our_b}): "
              f"rho = {rho:.4f} (n={valid.sum()})")

    # Save per-gene file
    if rows:
        pd.DataFrame(rows).to_csv(
            os.path.join(output_dir, "trap_vs_wasserstein_per_gene.csv"), index=False)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="compartment_classification/trap_validation")
    parser.add_argument("--skip_wasserstein", action="store_true",
                        help="Skip Wasserstein computation (requires loading raw h5ad data)")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    gene_list = pd.read_csv(GENE_LIST_PATH, header=None)[0].tolist()
    gene_set = set(gene_list)

    # Load model predictions
    pred_global, gt_global, gt_per_ct = load_model_deltas()

    # Validation 1: IP/Input enrichment
    enrichment = load_gse141464_enrichment()
    ip_results = validate_ip_input(enrichment, pred_global, gt_global, gene_list, args.output_dir)

    # Validation 2: Cell-type-specific TRAP (mean-based)
    greenberg = load_greenberg_ribotag()
    ct_results = validate_celltype_specific(greenberg, pred_global, gt_per_ct, gene_list, args.output_dir)

    # Validation 3: Cell-type-specific TRAP vs Wasserstein distance
    wasserstein_results = {}
    if not args.skip_wasserstein:
        X_cell, ct_cell, X_nuc, ct_nuc, gene_names, _, _ = load_raw_data()
        ct_mapping_types = ["Pvalb", "Sst", "Vip"]
        wasserstein_dists = compute_wasserstein_per_ct(
            X_cell, ct_cell, X_nuc, ct_nuc, gene_names, ct_mapping_types)
        wasserstein_results = validate_celltype_wasserstein(
            greenberg, wasserstein_dists, gene_list, args.output_dir)

        # Save Wasserstein distances
        for ct, df in wasserstein_dists.items():
            df.to_csv(os.path.join(args.output_dir, f"wasserstein_{ct}.csv"))

    # Save all results
    all_results = {
        "validation_1_trap_enrichment": {
            "source": "GSE141464 (Schmidt et al.), S100a10-TRAP, cortical L5a pyramidal neurons",
            "measure": "log2(IP_CPM / Input_CPM) per gene, saline condition, 3 IP reps + 1 Input rep",
            "comparison": "Correlate with ImputeVAE predicted delta and GT empirical delta",
            "note": "INDEPENDENT experiment: different lab, technology (TRAP), brain region (cerebral cortex vs MOp)",
            "results": ip_results,
        },
        "validation_2_celltype_trap": {
            "source": "GSE77243 (Spiegel/Bhatt/Bhatt/Greenberg), RiboTag, 5 neuron types, visual cortex",
            "measure": "Ribosome-bound expression per gene per neuron type (IP-only, standard housing)",
            "comparison": "Test cell-type-specific enrichment patterns match cell-type-specific GT delta",
            "note": "IP-only (no input), but cross-cell-type differential TRAP tests specificity",
            "results": ct_results,
        },
        "validation_3_wasserstein": {
            "source": "Same as validation_2, but using Wasserstein distance instead of mean delta",
            "measure": "signed W1(cell_dist, nucleus_dist) = sign(mean_diff) × W1",
            "comparison": "Correlate TRAP expression with distributional distance, not just mean shift",
            "note": "Captures full distribution shape: spread, skewness, bimodality — not just location",
            "results": wasserstein_results,
        },
    }

    with open(os.path.join(args.output_dir, "trap_validation_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Generate summary
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    print("\nValidation 1: TRAP enrichment (IP/Input) vs predicted delta")
    print("  Source: GSE141464, L5a pyramidal neurons, cerebral cortex")
    print("  Metric: Spearman rho between log2(IP/Input) and predicted delta")
    print("")
    print(f"  {'Method':<15s} {'Spearman rho':>12s} {'AUROC':>8s} {'N genes':>8s}")
    print(f"  {'-'*15} {'-'*12} {'-'*8} {'-'*8}")
    for method in ["GT_delta", "ImputeVAE", "CT-mean", "Linear_reg", "scVI",
                   "scANVI", "scGen", "scButterfly"]:
        d = ip_results.get(method, {})
        if d:
            print(f"  {method:<15s} {d['spearman_rho']:>12.4f} "
                  f"{d['auroc_enriched_vs_depleted']:>8.4f} {d['n_genes']:>8d}")

    print("\nValidation 2: Cell-type-specific TRAP vs mean delta")
    for k, v in ct_results.items():
        print(f"  {k}: rho = {v['spearman_rho']:.4f} (p = {v['spearman_p']:.2e})")

    if wasserstein_results:
        print("\nValidation 3: Cell-type-specific TRAP vs Wasserstein distance")
        for k, v in wasserstein_results.items():
            print(f"  {k}: rho = {v['spearman_rho']:.4f} (n = {v['n_genes']})")

    print(f"\nResults saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
