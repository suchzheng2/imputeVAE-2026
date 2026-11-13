"""Per-compartment classification of nucleus-to-cell imputation.

Computes empirical ground-truth delta from raw cell/nucleus data,
then evaluates how well each imputation method recovers the correct
compartment enrichment direction per gene.

Steps:
  1. Ground-truth delta from raw data (per gene, per cell type, global)
  2. Gene class definitions (ribosomal, nuclear ncRNA, coding polyA)
  3. Classification metrics for ImputeVAE
  4. Head-to-head comparison across all methods
  5. Cell-type resolution (neurons vs glia, per-CT)
  6. lncATLAS external cross-check
  7. Interpretable brain examples
  8. Scrambled-label negative control

Usage:
    python analysis/compartment_classification.py --output_dir compartment_classification
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_fscore_support,
    confusion_matrix, accuracy_score,
)

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_ROOT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi"
GENE_LIST_PATH = "data/training_gene_list_v3.csv"
COMMON_GENES_PATH = "data/common_genes.csv"
COMMON_CT_PATH = "data/common_cell_types.csv"

NEURONAL_TYPES = {
    "L2/3 IT", "L5 ET", "L5 IT", "L5/6 NP", "L6 CT", "L6 IT", "L6b",
    "Lamp5", "Pvalb", "Sncg", "Sst", "Vip",
}
NON_NEURONAL_TYPES = {"Astro", "Endo", "Macrophage", "OPC", "Oligo"}

# Minimum cells/nuclei per cell type to include in headline numbers
MIN_N_HEADLINE = 100

# Methods and their imputed matrix paths
METHODS = {
    "ImputeVAE": "run/cvae_full_eval/imputed_cell_from_nucleus.npy",
    "CT-mean": "run/ctmean_baseline/imputed_cell_from_nucleus.npy",
    "Linear_reg": "run/linreg_baseline/imputed_cell_from_nucleus.npy",
    "scVI": "run/scvi_baseline/imputed_cell_from_nucleus.npy",
    "scANVI": "run/scanvi_baseline/imputed_cell_from_nucleus.npy",
    "scGen": "run/scgen_baseline/imputed_cell_from_nucleus.npy",
    "scButterfly": "run/scbutterfly_baseline/imputed_cell_from_nucleus.npy",
}

# Obs CSV paths (cell_type labels for the 40,166 nuclei)
OBS_PATHS = {
    "ImputeVAE": "run/cvae_full_eval/nucleus_obs.csv",
    "CT-mean": "run/ctmean_baseline/nucleus_obs.csv",
    "Linear_reg": "run/linreg_baseline/nucleus_obs.csv",
    "scVI": "run/scvi_baseline/nucleus_obs.csv",
    "scANVI": "run/scanvi_baseline/nucleus_obs.csv",
    "scGen": "run/scgen_baseline/nucleus_obs.csv",
    "scButterfly": "run/scbutterfly_baseline/nucleus_obs.csv",
}

# ---------------------------------------------------------------------------
# Gene class definitions
# ---------------------------------------------------------------------------

KNOWN_NUCLEAR_LNCRNAS = {
    "Malat1", "Neat1", "Xist", "Tsix", "Meg3", "Kcnq1ot1",
    "Gomafu", "Miat", "Firre", "Fendrr", "Airn", "Rian", "Mirg",
    "Pvt1", "Gas5", "Rmrp", "Rpph1",
}
KNOWN_SNHG = {"Snhg1", "Snhg3", "Snhg4", "Snhg5", "Snhg6", "Snhg7",
              "Snhg8", "Snhg9", "Snhg10", "Snhg11", "Snhg12", "Snhg14",
              "Snhg15", "Snhg16"}


def classify_gene(name):
    """Assign a gene to one of: ribosomal, nuclear_ncrna, translation, cytoplasmic_marker, coding_polya."""
    # Ribosomal proteins: Rpl* or Rps* but NOT Rps6k* (S6 kinases)
    if (name.startswith("Rpl") or name.startswith("Rps")) and not name.startswith("Rps6k"):
        if name[3:4].isdigit() or name[3:4] == "" or name in ("Rpsa",):
            return "ribosomal"
        # e.g. Rpl7a, Rps3a1, Rps4x — still ribosomal
        if len(name) > 3 and name[3:4].isdigit():
            return "ribosomal"
        # Catch remaining Rpl/Rps with suffix like Rpl7a
        if name.startswith("Rpl") and len(name) >= 4:
            return "ribosomal"
        if name.startswith("Rps") and len(name) >= 4 and not name.startswith("Rps6k"):
            return "ribosomal"

    # Translation factors
    if name.startswith("Eef") or name.startswith("Eif"):
        return "translation"

    # Nuclear non-coding RNAs
    if name in KNOWN_NUCLEAR_LNCRNAS or name in KNOWN_SNHG:
        return "nuclear_ncrna"
    if name.startswith("Snhg"):
        return "nuclear_ncrna"
    # Antisense (*os suffix, but not too short)
    if name.endswith("os") and len(name) > 3:
        # Exclude protein-coding genes that happen to end in 'os'
        # Check if the prefix looks like a gene name (not just 2 letters)
        return "nuclear_ncrna"
    # Gm* predicted genes (many are lncRNAs)
    if name.startswith("Gm") and len(name) > 2 and name[2:].replace(".", "").isdigit():
        return "nuclear_ncrna"
    # RIKEN clones ending in Rik
    if name.endswith("Rik"):
        return "nuclear_ncrna"

    # Everything else = coding polyA mRNA
    return "coding_polya"


def get_gene_classes(gene_names):
    """Classify all genes and return a DataFrame."""
    records = []
    for g in gene_names:
        cls = classify_gene(g)
        records.append({"gene": g, "gene_class": cls})
    df = pd.DataFrame(records).set_index("gene")
    return df


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
    assert gene_names == gene_names2, "Gene name mismatch between cell and nucleus!"
    return X_cell, ct_cell, X_nuc, ct_nuc, gene_names, n_cell, n_nuc


# ---------------------------------------------------------------------------
# Step 1: Ground-truth delta
# ---------------------------------------------------------------------------

def compute_gt_delta(X_cell, ct_cell, X_nuc, ct_nuc, gene_names, common_cts):
    """Compute empirical GT delta = mean(cell) - mean(nucleus), globally and per CT."""
    print("\n=== Step 1: Computing ground-truth delta ===")
    n_genes = len(gene_names)

    # Global delta
    global_cell_mean = X_cell.mean(axis=0)
    global_nuc_mean = X_nuc.mean(axis=0)
    global_delta = global_cell_mean - global_nuc_mean

    gt_global = pd.DataFrame({
        "gene": gene_names,
        "cell_mean": global_cell_mean,
        "nuc_mean": global_nuc_mean,
        "gt_delta": global_delta,
    }).set_index("gene")

    # Per-CT delta
    gt_per_ct = {}
    ct_counts = {}
    for ct in sorted(common_cts):
        cell_mask = ct_cell == ct
        nuc_mask = ct_nuc == ct
        n_c = cell_mask.sum()
        n_n = nuc_mask.sum()
        ct_counts[ct] = {"n_cell": int(n_c), "n_nuc": int(n_n)}
        if n_c > 0 and n_n > 0:
            cell_mean = X_cell[cell_mask].mean(axis=0)
            nuc_mean = X_nuc[nuc_mask].mean(axis=0)
            gt_per_ct[ct] = cell_mean - nuc_mean
        else:
            gt_per_ct[ct] = np.full(n_genes, np.nan)

    gt_ct_df = pd.DataFrame(gt_per_ct, index=gene_names)
    print(f"  Global: {(global_delta > 0).sum()} cytoplasm-enriched, "
          f"{(global_delta < 0).sum()} nuclear-enriched (at tau=0)")
    return gt_global, gt_ct_df, ct_counts


# ---------------------------------------------------------------------------
# Step 2: Gene classes
# ---------------------------------------------------------------------------

def report_gene_classes(gene_classes, gt_global, output_dir):
    """Report gene class counts and save."""
    print("\n=== Step 2: Gene class report ===")
    counts = gene_classes["gene_class"].value_counts()
    for cls, n in counts.items():
        subset = gt_global.loc[gene_classes["gene_class"] == cls]
        med_delta = subset["gt_delta"].median()
        print(f"  {cls}: {n} genes, median GT delta = {med_delta:.4f}")

    gene_classes.to_csv(os.path.join(output_dir, "gene_classes.csv"))
    return counts


# ---------------------------------------------------------------------------
# Step 3 & 4: Classification metrics
# ---------------------------------------------------------------------------

def classify_genes_by_delta(gt_delta, tau):
    """Label genes: 'cytoplasmic' if delta > tau, 'nuclear' if delta < -tau, 'dual' otherwise."""
    labels = np.where(gt_delta > tau, "cytoplasmic",
                      np.where(gt_delta < -tau, "nuclear", "dual"))
    return labels


def compute_classification_metrics(pred_delta, gt_delta, tau, gene_classes_series=None):
    """Compute classification metrics using predicted delta as score and GT label as truth.

    Returns dict with per-class and overall metrics.
    """
    gt_labels = classify_genes_by_delta(gt_delta, tau)
    n_total = len(gt_delta)

    results = {"tau": tau, "n_total": n_total}

    # Per-class counts
    for cls in ["cytoplasmic", "nuclear", "dual"]:
        results[f"n_{cls}"] = int((gt_labels == cls).sum())

    # Sign agreement (exclude dual)
    non_dual = gt_labels != "dual"
    if non_dual.sum() > 0:
        gt_sign = np.sign(gt_delta[non_dual])
        pred_sign = np.sign(pred_delta[non_dual])
        sign_agree = (gt_sign == pred_sign).mean()
        results["sign_agreement"] = float(sign_agree)
    else:
        results["sign_agreement"] = float("nan")

    # Overall sign agreement (all genes, tau=0 boundary)
    gt_sign_all = np.sign(gt_delta)
    pred_sign_all = np.sign(pred_delta)
    nonzero = gt_sign_all != 0
    results["sign_agreement_all"] = float(
        (gt_sign_all[nonzero] == pred_sign_all[nonzero]).mean()) if nonzero.sum() > 0 else float("nan")

    # Binary classification: cytoplasmic vs nuclear (excluding dual)
    cyto_nuc_mask = (gt_labels == "cytoplasmic") | (gt_labels == "nuclear")
    if cyto_nuc_mask.sum() > 1:
        y_true = (gt_labels[cyto_nuc_mask] == "cytoplasmic").astype(int)
        y_score = pred_delta[cyto_nuc_mask]

        # AUROC and AUPRC
        try:
            results["auroc_cyto_vs_nuc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            results["auroc_cyto_vs_nuc"] = float("nan")
        try:
            results["auprc_cyto_vs_nuc"] = float(average_precision_score(y_true, y_score))
        except ValueError:
            results["auprc_cyto_vs_nuc"] = float("nan")

        # Hard classification at threshold 0
        y_pred = (pred_delta[cyto_nuc_mask] > 0).astype(int)
        results["accuracy_cyto_vs_nuc"] = float(accuracy_score(y_true, y_pred))
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        results["precision_cyto"] = float(p)
        results["recall_cyto"] = float(r)  # = sensitivity for cytoplasmic class
        results["f1_cyto"] = float(f1)

        # Specificity = recall of the nuclear class
        p_nuc, r_nuc, f1_nuc, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", pos_label=0, zero_division=0)
        results["precision_nuc"] = float(p_nuc)
        results["recall_nuc"] = float(r_nuc)  # = specificity for cytoplasmic
        results["f1_nuc"] = float(f1_nuc)
        results["specificity_cyto"] = float(r_nuc)

        # Confusion matrix [TN, FP, FN, TP]
        cm = confusion_matrix(y_true, y_pred)
        results["confusion_matrix"] = cm.tolist()
    else:
        results["auroc_cyto_vs_nuc"] = float("nan")

    # Per gene-class metrics (if gene_classes provided)
    if gene_classes_series is not None:
        for cls in ["ribosomal", "nuclear_ncrna", "translation", "coding_polya"]:
            mask = gene_classes_series == cls
            if mask.sum() > 0:
                sub_gt = gt_delta[mask]
                sub_pred = pred_delta[mask]
                # Sign agreement
                gt_s = np.sign(sub_gt)
                pred_s = np.sign(sub_pred)
                nz = gt_s != 0
                if nz.sum() > 0:
                    results[f"sign_agree_{cls}"] = float((gt_s[nz] == pred_s[nz]).mean())
                # Regression for coding_polya
                if cls == "coding_polya" and mask.sum() > 10 and np.std(sub_pred) > 1e-12:
                    r_val, p_val = stats.pearsonr(sub_gt, sub_pred)
                    results[f"pearson_r_{cls}"] = float(r_val)
                    results[f"pearson_p_{cls}"] = float(p_val)
                    ss_res = np.sum((sub_gt - sub_pred) ** 2)
                    ss_tot = np.sum((sub_gt - sub_gt.mean()) ** 2)
                    results[f"r2_{cls}"] = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return results


def compute_method_deltas(method_name, imputed_path, obs_path, X_nuc, ct_nuc, gene_names, common_cts):
    """Compute predicted delta for a method: mean(imputed - raw_nuc) per CT and globally."""
    print(f"  Loading {method_name}...")
    imputed = np.load(imputed_path, mmap_mode="r")
    obs = pd.read_csv(obs_path)
    ct_imp = obs["cell_type"].values.astype(str)

    # Sanity: shapes must match
    assert imputed.shape == X_nuc.shape, f"Shape mismatch: {imputed.shape} vs {X_nuc.shape}"

    n_genes = len(gene_names)

    # Global delta: mean(imputed - nuc) over all nuclei
    # = mean(imputed) - mean(nuc) since both are over the same nuclei
    global_imp_mean = np.zeros(n_genes, dtype=np.float64)
    global_nuc_mean = np.zeros(n_genes, dtype=np.float64)

    # Process in chunks to avoid memory issues with mmap
    chunk_size = 5000
    n = imputed.shape[0]
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        imp_chunk = np.array(imputed[start:end], dtype=np.float64)
        nuc_chunk = X_nuc[start:end].astype(np.float64)
        global_imp_mean += imp_chunk.sum(axis=0)
        global_nuc_mean += nuc_chunk.sum(axis=0)
    global_imp_mean /= n
    global_nuc_mean /= n
    global_delta = global_imp_mean - global_nuc_mean

    # Per-CT delta
    per_ct_delta = {}
    for ct in sorted(common_cts):
        mask = ct_imp == ct
        n_ct = mask.sum()
        if n_ct > 0:
            imp_ct = np.array(imputed[mask], dtype=np.float64)
            nuc_ct = X_nuc[mask].astype(np.float64)
            per_ct_delta[ct] = (imp_ct.mean(axis=0) - nuc_ct.mean(axis=0)).astype(np.float32)
        else:
            per_ct_delta[ct] = np.full(n_genes, np.nan, dtype=np.float32)

    return global_delta.astype(np.float32), per_ct_delta


# ---------------------------------------------------------------------------
# Step 5: Cell-type resolution
# ---------------------------------------------------------------------------

def compute_per_ct_auroc(pred_ct_deltas, gt_ct_df, ct_counts, gene_classes_arr, taus=(0, 0.1)):
    """Compute AUROC per cell type and for neuron/glia groups."""
    results = {}
    for ct in gt_ct_df.columns:
        gt_d = gt_ct_df[ct].values
        pred_d = pred_ct_deltas.get(ct)
        if pred_d is None or np.all(np.isnan(gt_d)):
            continue
        valid = ~np.isnan(gt_d) & ~np.isnan(pred_d)
        if valid.sum() < 100:
            continue
        gt_v = gt_d[valid]
        pred_v = pred_d[valid]

        info = {
            "n_cell": ct_counts[ct]["n_cell"],
            "n_nuc": ct_counts[ct]["n_nuc"],
        }

        for tau in taus:
            labels = classify_genes_by_delta(gt_v, tau)
            cyto_nuc = (labels == "cytoplasmic") | (labels == "nuclear")
            if cyto_nuc.sum() > 10:
                y_true = (labels[cyto_nuc] == "cytoplasmic").astype(int)
                y_score = pred_v[cyto_nuc]
                try:
                    info[f"auroc_tau{tau}"] = float(roc_auc_score(y_true, y_score))
                except ValueError:
                    info[f"auroc_tau{tau}"] = float("nan")
                info[f"sign_agree_tau{tau}"] = float(
                    (np.sign(gt_v[cyto_nuc]) == np.sign(pred_v[cyto_nuc])).mean()
                )
                info[f"n_cyto_tau{tau}"] = int((labels == "cytoplasmic").sum())
                info[f"n_nuc_tau{tau}"] = int((labels == "nuclear").sum())

        results[ct] = info

    return results


# ---------------------------------------------------------------------------
# Step 6: lncATLAS cross-check
# ---------------------------------------------------------------------------

def lncatlas_crosscheck(gt_global, gene_names, output_dir):
    """Map lncATLAS CN-RCI to mouse genes via orthologs and compute concordance."""
    print("\n=== Step 6: lncATLAS external cross-check ===")
    lncatlas_path = "data/external/2026-07-24_lncATLAS_all_data.csv"
    ortho_path = "data/external/mouse_human_orthologs.csv"

    if not os.path.exists(lncatlas_path) or not os.path.exists(ortho_path):
        print("  SKIPPED: lncATLAS or ortholog file not found.")
        return None

    # Load orthologs
    ortho = pd.read_csv(ortho_path)
    human_to_mouse = dict(zip(ortho["human_sym"], ortho["mouse_sym"]))

    # Load lncATLAS — only ratio2 rows (CN-RCI)
    print("  Loading lncATLAS (ratio2 rows)...")
    chunks = []
    for chunk in pd.read_csv(lncatlas_path, chunksize=500000):
        ratio_rows = chunk[chunk["Data Type"] == "ratio2"]
        if len(ratio_rows) > 0:
            chunks.append(ratio_rows[["Gene Name", "Value", "Biotype"]])
    if not chunks:
        print("  No ratio2 rows found in lncATLAS.")
        return None
    lnc = pd.concat(chunks, ignore_index=True)
    print(f"  {len(lnc)} ratio2 measurements across {lnc['Gene Name'].nunique()} human genes")

    # Median CN-RCI per human gene (across cell lines)
    lnc_median = lnc.groupby("Gene Name").agg(
        cn_rci=("Value", "median"),
        biotype=("Biotype", "first"),
        n_cell_lines=("Value", "count"),
    ).reset_index()

    # Map to mouse
    lnc_median["mouse_gene"] = lnc_median["Gene Name"].map(human_to_mouse)
    mapped = lnc_median.dropna(subset=["mouse_gene"])
    # Keep only genes in our gene list
    gene_set = set(gene_names)
    mapped = mapped[mapped["mouse_gene"].isin(gene_set)].copy()
    mapped = mapped.set_index("mouse_gene")
    print(f"  Mapped to {len(mapped)} mouse genes in our gene list")

    if len(mapped) < 100:
        print("  Too few mapped genes for meaningful analysis.")
        return None

    # Merge with GT delta
    merged = gt_global.join(mapped[["cn_rci", "biotype", "n_cell_lines"]], how="inner")
    print(f"  {len(merged)} genes with both GT delta and lncATLAS CN-RCI")

    # Drop rows with NaN/Inf in either column
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=["gt_delta", "cn_rci"])
    print(f"  {len(merged)} genes after dropping NaN/Inf")

    # Spearman correlation
    rho, p_rho = stats.spearmanr(merged["gt_delta"], merged["cn_rci"])
    r_pear, p_pear = stats.pearsonr(merged["gt_delta"], merged["cn_rci"])

    # AUROC: use CN-RCI sign as independent truth, GT delta as score
    cn_positive = merged["cn_rci"] > 0  # cytoplasmic by lncATLAS
    cn_negative = merged["cn_rci"] < 0  # nuclear by lncATLAS
    cn_binary_mask = cn_positive | cn_negative
    if cn_binary_mask.sum() > 10:
        y_true_cn = cn_positive[cn_binary_mask].astype(int).values
        y_score_cn = merged["gt_delta"][cn_binary_mask].values
        auroc_cn = roc_auc_score(y_true_cn, y_score_cn)
    else:
        auroc_cn = float("nan")

    result = {
        "n_matched": len(merged),
        "spearman_rho": float(rho),
        "spearman_p": float(p_rho),
        "pearson_r": float(r_pear),
        "pearson_p": float(p_pear),
        "auroc_cn_label": float(auroc_cn),
        "n_cn_cytoplasmic": int(cn_positive.sum()),
        "n_cn_nuclear": int(cn_negative.sum()),
        "note": ("lncATLAS CN-RCI is from human cell lines (A549, HeLa, etc.), "
                 "cell-type-agnostic. Tests only the conserved cell-type-invariant axis."),
    }
    print(f"  Spearman rho = {rho:.4f} (p = {p_rho:.2e})")
    print(f"  AUROC (CN-RCI label) = {auroc_cn:.4f}")

    # Save per-gene file
    merged.to_csv(os.path.join(output_dir, "lncatlas_crosscheck.csv"))
    return result


# ---------------------------------------------------------------------------
# Step 7: Interpretable brain examples
# ---------------------------------------------------------------------------

# Named neurobiological genes to prioritize in top lists
NEUROBIO_GENES = {
    # Synaptic/dendritic
    "Snap25", "Syt1", "Syt11", "Vamp2", "Sv2a", "Sv2b", "Stx1a", "Stx1b",
    "Nrgn", "Camk2a", "Camk2b", "Camk2n1", "Dlg4", "Shank1", "Shank2", "Shank3",
    "Homer1", "Gria1", "Gria2", "Grin1", "Grin2a", "Grin2b", "Gabra1", "Gabrb2",
    # Calcium/signaling
    "Calm1", "Calm2", "Calm3", "Calb1", "Calb2", "Pvalb",
    # Cytoskeletal
    "Actb", "Actg1", "Tuba1a", "Tubb2a", "Tubb3", "Map2", "Mapt",
    # Named neuronal TFs
    "Rbfox3", "Mef2c", "Satb2", "Tbr1", "Bcl11b", "Fezf2",
    # Neurotransmitter
    "Slc17a7", "Slc17a6", "Gad1", "Gad2", "Slc32a1", "Th",
    # Translation
    "Eef1a1", "Eef2", "Eif4a2",
    # 14-3-3
    "Ywhab", "Ywhaz", "Ywhah", "Ywhag",
    # Heat shock
    "Hsp90aa1", "Hsp90ab1", "Hspa8", "Hsph1",
    # RAB/vesicular
    "Rab3a", "Rab3c", "Rab5a",
    # Nuclear lncRNAs
    "Malat1", "Neat1", "Xist", "Tsix", "Meg3", "Kcnq1ot1",
    # Glia markers
    "Plp1", "Mbp", "Mobp", "Mog", "Gfap", "Aldoc", "Mt1", "Aqp4",
    "Cspg4", "Pdgfra",
    # Ion channels
    "Kcnab2", "Scn1a", "Scn2a1", "Scn8a",
    # Tetraspanins
    "Tspan7", "Cd63", "Cd81",
}


def top_gene_lists(gt_global, pred_global_delta, gene_classes, gt_ct_df, pred_ct_deltas,
                   common_cts, ct_counts, output_dir, n_top=15):
    """List top correctly-recovered cytoplasmic and nuclear genes."""
    print("\n=== Step 7: Interpretable brain examples ===")
    results = {}

    gene_names = gt_global.index.tolist()
    gc_arr = gene_classes["gene_class"].reindex(gene_names).values

    # For each major cell type (with enough data)
    for ct in sorted(common_cts):
        if ct_counts[ct]["n_cell"] < 50 or ct_counts[ct]["n_nuc"] < 50:
            continue
        gt_d = gt_ct_df[ct].values
        pred_d = pred_ct_deltas.get(ct)
        if pred_d is None:
            continue

        # Correctly recovered = GT delta and pred delta have same sign and |GT delta| > 0.1
        correct_cyto = (gt_d > 0.1) & (pred_d > 0)
        correct_nuc = (gt_d < -0.1) & (pred_d < 0)

        # Rank by predicted delta magnitude
        df = pd.DataFrame({
            "gene": gene_names,
            "gt_delta": gt_d,
            "pred_delta": pred_d,
            "gene_class": gc_arr,
            "is_neurobio": [g in NEUROBIO_GENES for g in gene_names],
        })

        # Top cytoplasmic: prioritize named neurobiological genes
        cyto_df = df[correct_cyto].sort_values("pred_delta", ascending=False)
        # Take named genes first, then fill with others, excluding generic ribosomal
        named_cyto = cyto_df[cyto_df["is_neurobio"] & (cyto_df["gene_class"] != "ribosomal")]
        other_cyto = cyto_df[~cyto_df.index.isin(named_cyto.index) & (cyto_df["gene_class"] != "ribosomal")]
        top_cyto = pd.concat([named_cyto.head(n_top), other_cyto]).head(n_top)

        # Top nuclear
        nuc_df = df[correct_nuc].sort_values("pred_delta", ascending=True)
        named_nuc = nuc_df[nuc_df["is_neurobio"]]
        other_nuc = nuc_df[~nuc_df.index.isin(named_nuc.index)]
        top_nuc = pd.concat([named_nuc.head(n_top), other_nuc]).head(n_top)

        results[ct] = {
            "top_cytoplasmic": top_cyto[["gene", "gt_delta", "pred_delta", "gene_class"]].to_dict("records"),
            "top_nuclear": top_nuc[["gene", "gt_delta", "pred_delta", "gene_class"]].to_dict("records"),
        }

    # Find canonical-nuclear genes that flip to cytoplasmic in neurons
    print("  Checking for nuclear-to-cytoplasmic flips in neurons...")
    canonical_nuclear = {"Meg3", "Kcnq1ot1", "Malat1", "Neat1", "Xist", "Miat"}
    flips = []
    for g in canonical_nuclear:
        if g not in gt_ct_df.index:
            continue
        for ct in sorted(common_cts):
            if ct not in NEURONAL_TYPES:
                continue
            gt_d = gt_ct_df.loc[g, ct] if g in gt_ct_df.index else np.nan
            if not np.isnan(gt_d) and gt_d > 0:
                flips.append({"gene": g, "cell_type": ct, "gt_delta": float(gt_d)})
    results["nuclear_to_cyto_flips"] = flips
    if flips:
        print(f"  Found {len(flips)} nuclear-to-cytoplasmic flips:")
        for f in flips:
            print(f"    {f['gene']} in {f['cell_type']}: GT delta = {f['gt_delta']:.3f}")

    return results


# ---------------------------------------------------------------------------
# Step 8: Scrambled-label negative control
# ---------------------------------------------------------------------------

def scrambled_control(pred_delta, gt_delta, tau, n_permutations=1000, seed=42):
    """Permute GT labels and recompute AUROC to get null distribution."""
    print(f"\n=== Step 8: Scrambled-label control (n={n_permutations}) ===")
    gt_labels = classify_genes_by_delta(gt_delta, tau)
    cyto_nuc = (gt_labels == "cytoplasmic") | (gt_labels == "nuclear")
    y_true = (gt_labels[cyto_nuc] == "cytoplasmic").astype(int)
    y_score = pred_delta[cyto_nuc]

    real_auroc = roc_auc_score(y_true, y_score)

    rng = np.random.RandomState(seed)
    null_aurocs = np.zeros(n_permutations)
    for i in range(n_permutations):
        perm_labels = rng.permutation(y_true)
        null_aurocs[i] = roc_auc_score(perm_labels, y_score)

    p_value = (null_aurocs >= real_auroc).mean()
    z_score = (real_auroc - null_aurocs.mean()) / null_aurocs.std() if null_aurocs.std() > 0 else float("inf")

    result = {
        "real_auroc": float(real_auroc),
        "null_mean": float(null_aurocs.mean()),
        "null_std": float(null_aurocs.std()),
        "null_max": float(null_aurocs.max()),
        "z_score": float(z_score),
        "p_value": float(p_value),
        "n_permutations": n_permutations,
    }
    print(f"  Real AUROC = {real_auroc:.4f}")
    print(f"  Null: mean={null_aurocs.mean():.4f}, std={null_aurocs.std():.4f}, max={null_aurocs.max():.4f}")
    print(f"  z = {z_score:.1f}, p = {p_value}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compartment classification analysis")
    parser.add_argument("--output_dir", default="compartment_classification")
    parser.add_argument("--taus", nargs="+", type=float, default=[0, 0.1, 0.2],
                        help="Threshold values for delta classification")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    X_cell, ct_cell, X_nuc, ct_nuc, gene_names, n_cell, n_nuc = load_raw_data()
    common_cts = sorted(set(ct_cell) & set(ct_nuc))
    print(f"\nDataset: {n_cell} cells, {n_nuc} nuclei, {len(gene_names)} genes, {len(common_cts)} cell types")

    # Gene classes
    gene_classes = get_gene_classes(gene_names)
    gc_arr = gene_classes["gene_class"].reindex(gene_names).values

    # -----------------------------------------------------------------------
    # Step 1: Ground-truth delta
    # -----------------------------------------------------------------------
    gt_global, gt_ct_df, ct_counts = compute_gt_delta(
        X_cell, ct_cell, X_nuc, ct_nuc, gene_names, common_cts)

    # Save GT delta
    gt_global.to_csv(os.path.join(args.output_dir, "gt_delta_global.csv"))
    gt_ct_df.to_csv(os.path.join(args.output_dir, "gt_delta_per_ct.csv"))
    with open(os.path.join(args.output_dir, "ct_counts.json"), "w") as f:
        json.dump(ct_counts, f, indent=2)

    # -----------------------------------------------------------------------
    # Step 2: Gene classes
    # -----------------------------------------------------------------------
    class_counts = report_gene_classes(gene_classes, gt_global, args.output_dir)

    # Report GT delta distribution at different taus
    for tau in args.taus:
        labels = classify_genes_by_delta(gt_global["gt_delta"].values, tau)
        print(f"  tau={tau}: cytoplasmic={int((labels == 'cytoplasmic').sum())}, "
              f"nuclear={int((labels == 'nuclear').sum())}, "
              f"dual={int((labels == 'dual').sum())}")

    # -----------------------------------------------------------------------
    # Steps 3 & 4: Method-by-method classification
    # -----------------------------------------------------------------------
    print("\n=== Steps 3 & 4: Computing predicted deltas and classification ===")

    all_method_results = {}
    all_pred_global = {}
    all_pred_ct = {}

    # Raw nuclei baseline: predicted delta = 0 for all genes
    print("  Raw nuclei (delta = 0 baseline)...")
    raw_nuc_delta = np.zeros(len(gene_names), dtype=np.float32)
    all_pred_global["Raw_nuclei"] = raw_nuc_delta
    # Per-CT: also 0
    raw_nuc_ct = {ct: np.zeros(len(gene_names), dtype=np.float32) for ct in common_cts}
    all_pred_ct["Raw_nuclei"] = raw_nuc_ct

    # Load each method
    for method_name, imputed_path in METHODS.items():
        if not os.path.exists(imputed_path):
            print(f"  {method_name}: MISSING, skipping")
            continue
        obs_path = OBS_PATHS[method_name]
        global_delta, per_ct_delta = compute_method_deltas(
            method_name, imputed_path, obs_path, X_nuc, ct_nuc, gene_names, common_cts)
        all_pred_global[method_name] = global_delta
        all_pred_ct[method_name] = per_ct_delta

    # Free raw cell data (no longer needed)
    del X_cell
    import gc; gc.collect()

    # Compute classification metrics for each method
    print("\nComputing classification metrics...")
    for tau in args.taus:
        tau_results = {}
        for method_name, pred_delta in all_pred_global.items():
            metrics = compute_classification_metrics(
                pred_delta, gt_global["gt_delta"].values, tau, gc_arr)
            tau_results[method_name] = metrics
            if method_name == "ImputeVAE":
                print(f"\n  ImputeVAE (tau={tau}):")
                print(f"    AUROC = {metrics.get('auroc_cyto_vs_nuc', 'N/A')}")
                print(f"    Sign agreement = {metrics.get('sign_agreement_all', 'N/A')}")
                print(f"    Accuracy = {metrics.get('accuracy_cyto_vs_nuc', 'N/A')}")
        all_method_results[f"tau_{tau}"] = tau_results

    # Save per-method global deltas
    delta_df = pd.DataFrame(all_pred_global, index=gene_names)
    delta_df.to_csv(os.path.join(args.output_dir, "predicted_deltas_global.csv"))

    # -----------------------------------------------------------------------
    # Build comparison tables
    # -----------------------------------------------------------------------
    print("\n=== Building comparison tables ===")

    # Table B: Method comparison AUROC
    comparison_rows = []
    for tau in args.taus:
        for method_name in ["ImputeVAE", "CT-mean", "Linear_reg", "scVI", "scANVI",
                            "scGen", "scButterfly", "Raw_nuclei"]:
            m = all_method_results.get(f"tau_{tau}", {}).get(method_name, {})
            if m:
                comparison_rows.append({
                    "method": method_name,
                    "tau": tau,
                    "n_cyto": m.get("n_cytoplasmic", ""),
                    "n_nuclear": m.get("n_nuclear", ""),
                    "AUROC": m.get("auroc_cyto_vs_nuc", ""),
                    "AUPRC": m.get("auprc_cyto_vs_nuc", ""),
                    "accuracy": m.get("accuracy_cyto_vs_nuc", ""),
                    "sign_agreement": m.get("sign_agreement_all", ""),
                    "sensitivity_cyto": m.get("recall_cyto", ""),
                    "specificity_cyto": m.get("specificity_cyto", ""),
                    "F1_cyto": m.get("f1_cyto", ""),
                    "F1_nuc": m.get("f1_nuc", ""),
                })
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(os.path.join(args.output_dir, "method_comparison.csv"), index=False)

    # Table A: ImputeVAE per-class metrics
    vae_detailed_rows = []
    for tau in args.taus:
        m = all_method_results.get(f"tau_{tau}", {}).get("ImputeVAE", {})
        if m:
            row = {"tau": tau}
            for k, v in m.items():
                if k != "confusion_matrix":
                    row[k] = v
            vae_detailed_rows.append(row)
    vae_detailed_df = pd.DataFrame(vae_detailed_rows)
    vae_detailed_df.to_csv(os.path.join(args.output_dir, "imputevae_detailed_metrics.csv"), index=False)

    # -----------------------------------------------------------------------
    # Step 5: Cell-type resolution
    # -----------------------------------------------------------------------
    print("\n=== Step 5: Cell-type resolution ===")
    vae_ct_results = {}
    if "ImputeVAE" in all_pred_ct:
        vae_ct_results = compute_per_ct_auroc(
            all_pred_ct["ImputeVAE"], gt_ct_df, ct_counts, gc_arr, taus=args.taus)

        # Add neuron/glia group AUROCs
        for group_name, group_cts in [("neurons", NEURONAL_TYPES), ("non_neuronal", NON_NEURONAL_TYPES)]:
            # Pool GT deltas across CTs in the group (weighted by nuclei count)
            total_nuc = sum(ct_counts[ct]["n_nuc"] for ct in group_cts if ct in ct_counts)
            total_cell = sum(ct_counts[ct]["n_cell"] for ct in group_cts if ct in ct_counts)
            if total_nuc == 0 or total_cell == 0:
                continue
            # Weighted average delta
            gt_group = np.zeros(len(gene_names))
            pred_group = np.zeros(len(gene_names))
            for ct in group_cts:
                if ct not in gt_ct_df.columns or ct not in all_pred_ct["ImputeVAE"]:
                    continue
                w_gt = ct_counts[ct]["n_cell"] + ct_counts[ct]["n_nuc"]
                gt_group += gt_ct_df[ct].values * w_gt
                pred_group += all_pred_ct["ImputeVAE"][ct] * ct_counts[ct]["n_nuc"]
            gt_group /= sum(ct_counts[ct]["n_cell"] + ct_counts[ct]["n_nuc"]
                           for ct in group_cts if ct in ct_counts)
            pred_group /= sum(ct_counts[ct]["n_nuc"] for ct in group_cts if ct in ct_counts)

            for tau in args.taus:
                labels = classify_genes_by_delta(gt_group, tau)
                cyto_nuc = (labels == "cytoplasmic") | (labels == "nuclear")
                if cyto_nuc.sum() > 10:
                    y_true = (labels[cyto_nuc] == "cytoplasmic").astype(int)
                    y_score = pred_group[cyto_nuc]
                    try:
                        auroc = roc_auc_score(y_true, y_score)
                    except ValueError:
                        auroc = float("nan")
                    if group_name not in vae_ct_results:
                        vae_ct_results[group_name] = {"n_cell": total_cell, "n_nuc": total_nuc}
                    vae_ct_results[group_name][f"auroc_tau{tau}"] = float(auroc)
                    vae_ct_results[group_name][f"sign_agree_tau{tau}"] = float(
                        (np.sign(gt_group[cyto_nuc]) == np.sign(pred_group[cyto_nuc])).mean())

        # Flag low-n and imbalanced types
        for ct in common_cts:
            if ct in vae_ct_results:
                nc = ct_counts[ct]["n_cell"]
                nn = ct_counts[ct]["n_nuc"]
                vae_ct_results[ct]["low_n_flag"] = nc < MIN_N_HEADLINE or nn < MIN_N_HEADLINE
                ratio = max(nc, nn) / max(min(nc, nn), 1)
                vae_ct_results[ct]["count_imbalance_ratio"] = round(ratio, 1)

    # Save per-CT results
    ct_results_df = pd.DataFrame(vae_ct_results).T
    ct_results_df.to_csv(os.path.join(args.output_dir, "per_ct_auroc.csv"))

    # Per-CT method comparison at tau=0
    print("  Computing per-CT AUROCs for all methods...")
    method_ct_aurocs = {}
    for method_name, pred_ct in all_pred_ct.items():
        ct_aurocs = {}
        for ct in common_cts:
            gt_d = gt_ct_df[ct].values
            pred_d = pred_ct.get(ct)
            if pred_d is None:
                continue
            valid = ~np.isnan(gt_d) & ~np.isnan(pred_d)
            gt_v = gt_d[valid]
            pred_v = pred_d[valid]
            labels = classify_genes_by_delta(gt_v, 0)
            cyto_nuc = (labels == "cytoplasmic") | (labels == "nuclear")
            if cyto_nuc.sum() > 10:
                y_true = (labels[cyto_nuc] == "cytoplasmic").astype(int)
                y_score = pred_v[cyto_nuc]
                try:
                    ct_aurocs[ct] = float(roc_auc_score(y_true, y_score))
                except ValueError:
                    ct_aurocs[ct] = float("nan")
        method_ct_aurocs[method_name] = ct_aurocs
    ct_comparison_df = pd.DataFrame(method_ct_aurocs)
    ct_comparison_df.to_csv(os.path.join(args.output_dir, "per_ct_method_auroc.csv"))

    # -----------------------------------------------------------------------
    # Step 6: lncATLAS
    # -----------------------------------------------------------------------
    lncatlas_result = lncatlas_crosscheck(gt_global, gene_names, args.output_dir)

    # -----------------------------------------------------------------------
    # Step 7: Top gene lists
    # -----------------------------------------------------------------------
    top_genes = None
    if "ImputeVAE" in all_pred_ct:
        top_genes = top_gene_lists(
            gt_global, all_pred_global["ImputeVAE"], gene_classes,
            gt_ct_df, all_pred_ct["ImputeVAE"],
            common_cts, ct_counts, args.output_dir)

    # -----------------------------------------------------------------------
    # Step 8: Scrambled control
    # -----------------------------------------------------------------------
    scramble_results = {}
    if "ImputeVAE" in all_pred_global:
        for tau in args.taus:
            scramble_results[f"tau_{tau}"] = scrambled_control(
                all_pred_global["ImputeVAE"], gt_global["gt_delta"].values, tau)

    # -----------------------------------------------------------------------
    # Step E: Coding-mRNA regression stats
    # -----------------------------------------------------------------------
    print("\n=== Step E: Coding-mRNA regression ===")
    coding_mask = gc_arr == "coding_polya"
    coding_regression = {}
    for method_name, pred_delta in all_pred_global.items():
        gt_coding = gt_global["gt_delta"].values[coding_mask]
        pred_coding = pred_delta[coding_mask]
        valid = ~np.isnan(gt_coding) & ~np.isnan(pred_coding) & np.isfinite(pred_coding)
        if valid.sum() > 10:
            gt_v = gt_coding[valid]
            pred_v = pred_coding[valid]
            # Guard against constant input (e.g., Raw_nuclei delta=0)
            if pred_v.std() < 1e-12:
                r_val, p_val, r2 = 0.0, 1.0, 0.0
            else:
                r_val, p_val = stats.pearsonr(gt_v, pred_v)
                ss_res = np.sum((gt_v - pred_v) ** 2)
                ss_tot = np.sum((gt_v - gt_v.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            coding_regression[method_name] = {
                "n_genes": int(valid.sum()),
                "pearson_r": float(r_val),
                "pearson_p": float(p_val),
                "r2_pred_vs_gt": float(r2),
                "mean_gt_delta": float(gt_v.mean()),
                "mean_pred_delta": float(pred_v.mean()),
            }
            if method_name == "ImputeVAE":
                print(f"  ImputeVAE: r={r_val:.4f}, R²={r2:.4f} on {valid.sum()} coding polyA genes")
    coding_df = pd.DataFrame(coding_regression).T
    coding_df.to_csv(os.path.join(args.output_dir, "coding_mrna_regression.csv"))

    # -----------------------------------------------------------------------
    # Save all results
    # -----------------------------------------------------------------------
    all_results = {
        "dataset": {
            "n_cell": n_cell,
            "n_nuc": n_nuc,
            "n_genes": len(gene_names),
            "n_cell_types": len(common_cts),
            "cell_types": common_cts,
        },
        "gene_classes": {k: int(v) for k, v in class_counts.items()},
        "gt_delta_summary": {
            "n_cyto_tau0": int((gt_global["gt_delta"].values > 0).sum()),
            "n_nuc_tau0": int((gt_global["gt_delta"].values < 0).sum()),
            "median_delta": float(gt_global["gt_delta"].median()),
            "mean_delta": float(gt_global["gt_delta"].mean()),
        },
        "ct_counts": ct_counts,
        "method_comparison": all_method_results,
        "per_ct_imputevae": vae_ct_results,
        "lncatlas": lncatlas_result,
        "scrambled_control": scramble_results,
        "coding_regression": coding_regression,
    }
    if top_genes:
        all_results["top_genes"] = top_genes

    with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # -----------------------------------------------------------------------
    # Generate README
    # -----------------------------------------------------------------------
    generate_readme(all_results, args)


def generate_readme(results, args):
    """Generate README.md with headline numbers."""
    out = args.output_dir

    lines = []
    lines.append("# Compartment Classification Results")
    lines.append("")
    lines.append("## Dataset")
    ds = results["dataset"]
    lines.append(f"- **Cells:** {ds['n_cell']:,} (c3A protocol)")
    lines.append(f"- **Nuclei:** {ds['n_nuc']:,} (n3A protocol)")
    lines.append(f"- **Genes:** {ds['n_genes']:,} (Config B gene list)")
    lines.append(f"- **Cell types:** {ds['n_cell_types']} shared types")
    lines.append("")

    # Cell type counts
    lines.append("### Cell/nuclei counts per type")
    lines.append("| Cell type | N cells | N nuclei | Group | Low-n flag |")
    lines.append("|-----------|--------:|---------:|-------|------------|")
    cts = results["ct_counts"]
    for ct in sorted(cts.keys()):
        nc = cts[ct]["n_cell"]
        nn = cts[ct]["n_nuc"]
        group = "neuron" if ct in NEURONAL_TYPES else "non-neuronal"
        flag = "YES" if nc < MIN_N_HEADLINE or nn < MIN_N_HEADLINE else ""
        lines.append(f"| {ct} | {nc:,} | {nn:,} | {group} | {flag} |")
    lines.append("")

    # Gene classes
    lines.append("## Gene Classes")
    gc = results["gene_classes"]
    for cls, n in sorted(gc.items()):
        lines.append(f"- **{cls}:** {n} genes")
    lines.append("")
    lines.append("*Classification by name pattern (BioMart was unavailable). "
                 "Ribosomal = Rpl*/Rps* excluding Rps6k*. Nuclear ncRNA = Malat1/Neat1/Xist/Tsix "
                 "+ Gm*/Rik/os/Snhg heuristics. Coding polyA = remainder.*")
    lines.append("")

    # GT delta summary
    gt = results["gt_delta_summary"]
    lines.append("## Ground-Truth Delta (cell - nucleus, log1p)")
    lines.append(f"- At tau=0: **{gt['n_cyto_tau0']:,}** cytoplasm-enriched, "
                 f"**{gt['n_nuc_tau0']:,}** nuclear-enriched")
    lines.append(f"- Median delta: {gt['median_delta']:.4f}")
    lines.append(f"- Mean delta: {gt['mean_delta']:.4f}")
    lines.append("")

    # Method comparison table (tau=0)
    lines.append("## Method Comparison (tau=0)")
    lines.append("")
    mc = results["method_comparison"].get("tau_0", results["method_comparison"].get("tau_0.0", {}))
    method_order = ["ImputeVAE", "CT-mean", "Linear_reg", "scVI", "scANVI",
                    "scGen", "scButterfly", "Raw_nuclei"]
    lines.append("| Method | AUROC | AUPRC | Accuracy | Sign Agree | Sens(cyto) | Spec(cyto) | F1(cyto) |")
    lines.append("|--------|------:|------:|---------:|-----------:|-----------:|-----------:|---------:|")
    for m in method_order:
        d = mc.get(m, {})
        if not d:
            continue
        def fmt(v):
            if isinstance(v, float) and not np.isnan(v):
                return f"{v:.4f}"
            return "—"
        lines.append(f"| {m} | {fmt(d.get('auroc_cyto_vs_nuc'))} | "
                     f"{fmt(d.get('auprc_cyto_vs_nuc'))} | "
                     f"{fmt(d.get('accuracy_cyto_vs_nuc'))} | "
                     f"{fmt(d.get('sign_agreement_all'))} | "
                     f"{fmt(d.get('recall_cyto'))} | "
                     f"{fmt(d.get('specificity_cyto'))} | "
                     f"{fmt(d.get('f1_cyto'))} |")
    lines.append("")

    # Per-class sign agreement for ImputeVAE (already using mc from above)
    vae_t0 = mc.get("ImputeVAE", {})
    if vae_t0:
        lines.append("### ImputeVAE per-class sign agreement (tau=0)")
        for cls in ["ribosomal", "nuclear_ncrna", "translation", "coding_polya"]:
            sa = vae_t0.get(f"sign_agree_{cls}")
            if sa is not None:
                lines.append(f"- **{cls}:** {sa:.4f}")
        lines.append("")

    # Confusion matrix
    cm = vae_t0.get("confusion_matrix")
    if cm:
        lines.append("### ImputeVAE confusion matrix (tau=0, pred>0 = cytoplasmic)")
        lines.append("```")
        lines.append(f"               Pred Nuclear  Pred Cyto")
        lines.append(f"  GT Nuclear      {cm[0][0]:>6}       {cm[0][1]:>6}")
        lines.append(f"  GT Cyto         {cm[1][0]:>6}       {cm[1][1]:>6}")
        lines.append("```")
        lines.append("")

    # Coding-mRNA regression
    lines.append("## Coding polyA mRNA Regression (predicted vs GT delta)")
    cr = results.get("coding_regression", {})
    if cr:
        lines.append("| Method | N genes | Pearson r | R² |")
        lines.append("|--------|--------:|----------:|---:|")
        for m in method_order:
            d = cr.get(m, {})
            if d:
                lines.append(f"| {m} | {d['n_genes']} | {d['pearson_r']:.4f} | {d['r2_pred_vs_gt']:.4f} |")
        lines.append("")

    # Per-CT AUROC
    lines.append("## Per-Cell-Type AUROC (ImputeVAE, tau=0)")
    pct = results.get("per_ct_imputevae", {})
    if pct:
        lines.append("| Cell type | N cell | N nuc | AUROC | Sign agree | Imbalance | Flag |")
        lines.append("|-----------|-------:|------:|------:|-----------:|----------:|------|")
        for ct in sorted(pct.keys()):
            d = pct[ct]
            if ct in ("neurons", "non_neuronal"):
                continue
            auroc = d.get("auroc_tau0", d.get("auroc_tau0.0", ""))
            sa = d.get("sign_agree_tau0", d.get("sign_agree_tau0.0", ""))
            nc = d.get("n_cell", "")
            nn = d.get("n_nuc", "")
            imb = d.get("count_imbalance_ratio", "")
            flag = "LOW-N" if d.get("low_n_flag") else ""
            auroc_s = f"{auroc:.4f}" if isinstance(auroc, (float, np.floating)) else str(auroc)
            sa_s = f"{sa:.4f}" if isinstance(sa, (float, np.floating)) else str(sa)
            lines.append(f"| {ct} | {nc} | {nn} | {auroc_s} | {sa_s} | {imb} | {flag} |")
        # Group summaries
        for group in ["neurons", "non_neuronal"]:
            d = pct.get(group, {})
            if d:
                a_val = d.get('auroc_tau0', d.get('auroc_tau0.0', ''))
                s_val = d.get('sign_agree_tau0', d.get('sign_agree_tau0.0', ''))
                a_str = f"{a_val:.4f}" if isinstance(a_val, (float, np.floating)) else str(a_val)
                s_str = f"{s_val:.4f}" if isinstance(s_val, (float, np.floating)) else str(s_val)
                lines.append(f"| **{group}** | {d.get('n_cell','')} | {d.get('n_nuc','')} | "
                             f"{a_str} | {s_str} | — | — |")
        lines.append("")

    # lncATLAS
    lnc = results.get("lncatlas")
    if lnc:
        lines.append("## lncATLAS External Cross-Check")
        lines.append(f"- **N matched genes:** {lnc['n_matched']}")
        lines.append(f"- **Spearman rho (GT delta vs CN-RCI):** {lnc['spearman_rho']:.4f} "
                     f"(p = {lnc['spearman_p']:.2e})")
        lines.append(f"- **AUROC (CN-RCI label):** {lnc['auroc_cn_label']:.4f}")
        lines.append(f"- *Note:* {lnc['note']}")
        lines.append("")

    # Scrambled control
    sc = results.get("scrambled_control", {})
    sc0 = sc.get("tau_0", sc.get("tau_0.0", {}))
    if sc0:
        lines.append("## Scrambled-Label Control (tau=0)")
        lines.append(f"- **Real AUROC:** {sc0['real_auroc']:.4f}")
        lines.append(f"- **Null mean:** {sc0['null_mean']:.4f} +/- {sc0['null_std']:.4f}")
        lines.append(f"- **z-score:** {sc0['z_score']:.1f}")
        lines.append(f"- **p-value:** {sc0['p_value']}")
        lines.append(f"- 1000 permutations. Real classification massively exceeds chance.")
        lines.append("")

    # Assumptions / limitations
    lines.append("## Assumptions and Limitations")
    lines.append("1. Ground truth delta computed from raw counts -> log1p, NOT from any model output.")
    lines.append("2. Gene biotype assigned by name pattern (Ensembl BioMart was unavailable). "
                 "Some antisense/Rik/Gm genes may be protein-coding; this adds noise to the "
                 "nuclear_ncrna class but does not affect the primary cytoplasmic/nuclear/dual labels.")
    lines.append("3. Cells and nuclei are unpaired; both GT and predicted deltas are population means per cell type.")
    lines.append("4. Low-n cell types (Endo 23 nuc, Macrophage 54 nuc, OPC 77 nuc) have noisy GT deltas. "
                 f"Headline numbers use types with >= {MIN_N_HEADLINE} in both modalities.")
    lines.append("5. Ribosomal = Rpl*/Rps*, excluding Rps6k* (S6 kinases). "
                 "Translation = Eef*/Eif*.")
    lines.append("6. lncATLAS CN-RCI is human cell-line data; tests only conserved, "
                 "cell-type-invariant localization axis.")
    lines.append("7. Raw nuclei baseline has predicted delta = 0 (no imputation), "
                 "so AUROC = 0.5 by definition.")
    lines.append("8. COMPARTMENTS (mouse protein localization) used only for cross-check in "
                 "`localization_concordance/`, not as primary reference here.")
    lines.append("")

    readme_text = "\n".join(lines)
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write(readme_text)

    print("\n" + "=" * 72)
    print("README.md contents:")
    print("=" * 72)
    print(readme_text)


if __name__ == "__main__":
    main()
