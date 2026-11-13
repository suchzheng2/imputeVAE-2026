#!/usr/bin/env python3
"""
Task Group 2, Parts E+F: Compare human predicted shifts with mouse validation shifts.

- SynGO and TF transfer summaries
- Mouse-human predicted-shift similarity by class
- Donor consistency
"""

import os
import sys
import csv
import json
import numpy as np
from scipy import stats
from collections import defaultdict

ROOT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE"
RESULTS_DIR = os.path.join(ROOT, "validation/human_m1_transfer/results")
INF_DIR = os.path.join(ROOT, "validation/human_m1_transfer/inference")
PROC_DIR = os.path.join(ROOT, "validation/human_m1_transfer/processed")

SYNGO_DELTAS = os.path.join(ROOT, "validation/syngo/syngo_brain_validation/syngo_gene_celltype_deltas.csv")
SYNGO_META = os.path.join(ROOT, "validation/syngo/syngo_brain_validation/syngo_gene_metadata.csv")
TF_DELTAS = os.path.join(ROOT, "validation/tf/tf_brain_validation/tf_gene_celltype_deltas.csv")
TF_META = os.path.join(ROOT, "validation/tf/tf_brain_validation/tf_gene_metadata.csv")

# Mouse all-gene per-CT delta tables (from delta_analysis, covers all 10,134 genes)
MOUSE_DELTA_TABLES = os.path.join(ROOT, "notes/experiments/delta_analysis/tables")

os.makedirs(RESULTS_DIR, exist_ok=True)


def load_human_class_deltas(class_key="mapped_mouse_like_class"):
    """Load human predicted deltas from the long-format pseudobulk CSV.
    Returns dict: {(class, mouse_gene): delta}
    """
    if class_key == "mapped_mouse_like_class":
        path = os.path.join(RESULTS_DIR, "human_m1_mouse_like_class_gene_means.csv")
    else:
        path = os.path.join(RESULTS_DIR, "human_m1_broad_class_gene_means.csv")

    result = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (row["group"], row["mouse_model_gene"])
            result[key] = float(row["predicted_delta"])
    return result


def load_human_donor_deltas(class_key="mapped_mouse_like_class"):
    """Load per-donor human predicted deltas.
    Returns dict: {(donor, class, mouse_gene): delta}
    """
    if class_key == "mapped_mouse_like_class":
        path = os.path.join(RESULTS_DIR, "human_m1_donor_mouse_like_class_gene_means.csv")
    else:
        path = os.path.join(RESULTS_DIR, "human_m1_donor_broad_class_gene_means.csv")

    result = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            parts = row["group"].split("|", 1)
            donor, cls = parts[0], parts[1]
            result[(donor, cls, row["mouse_model_gene"])] = float(row["predicted_delta"])
    return result


def load_mouse_deltas(deltas_path):
    """Load mouse per-CT predicted deltas.
    Returns dict: {(cell_type, gene): predicted_delta}
    """
    result = {}
    with open(deltas_path) as f:
        for row in csv.DictReader(f):
            result[(row["cell_type"], row["gene"])] = float(row["predicted_delta"])
    return result


def load_gene_metadata(meta_path):
    """Load gene metadata. Returns dict keyed by mouse symbol."""
    result = {}
    with open(meta_path) as f:
        for row in csv.DictReader(f):
            # SynGO has mouse_symbol, TF has model_gene
            mouse_key = row.get("mouse_symbol", row.get("model_gene", ""))
            if mouse_key:
                result[mouse_key] = row
    return result


def load_human_gene_map():
    """Load human gene mapping to get mouse->human symbol map for mapped genes."""
    path = os.path.join(INF_DIR, "human_m1_inference_gene_map.csv")
    mouse_to_human = {}
    mapped_mouse_genes = set()
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["filled_value_status"] == "mapped":
                mapped_mouse_genes.add(row["mouse_model_gene"])
                mouse_to_human[row["mouse_model_gene"]] = row["human_gene_symbol"]
    return mapped_mouse_genes, mouse_to_human


# Mouse cell types that map to human mouse-like classes
MOUSE_TO_HUMAN_CT = {
    "L2/3 IT": "L2/3_IT",
    "L5 IT": "L5_IT",
    "L5 ET": "L5_ET",
    "L5/6 NP": "L5/6_NP",
    "L6 CT": "L6_CT",
    "L6 IT": "L6_IT",
    "L6b": "L6b",
    "Lamp5": "Lamp5",
    "Sncg": "Sncg",
    "Vip": "Vip",
    "Sst": "Sst",
    "Pvalb": "Pvalb",
    "Astro": "Astro",
    "Oligo": "Oligo",
    "OPC": "OPC",
    "Endo": "Endo",
    "Micro": "Micro",
}

NEURONAL_BROAD = {"excitatory_neuron", "inhibitory_neuron"}
NONNEURONAL_BROAD = {"astrocyte", "oligodendrocyte", "opc", "endothelial",
                     "microglia_macrophage", "vlmc_pericyte_smc"}


def part_e_syngo_tf_transfer():
    """Compare human predicted shifts for SynGO and TF gene sets."""
    print("=== Part E: SynGO/TF transfer analysis ===")

    mapped_genes, mouse_to_human = load_human_gene_map()
    human_deltas_fine = load_human_class_deltas("mapped_mouse_like_class")
    human_deltas_broad = load_human_class_deltas("mapped_broad_class")

    mouse_syngo = load_mouse_deltas(SYNGO_DELTAS)
    mouse_tf = load_mouse_deltas(TF_DELTAS)

    # Use the non-ribosomal SynGO gene set from the deltas file (1,275 genes),
    # NOT syngo_gene_metadata.csv which includes 84 ribosomal genes
    syngo_genes_in_deltas = set(g for (_, g) in mouse_syngo.keys())
    tf_genes_in_deltas = set(g for (_, g) in mouse_tf.keys())

    # Filter to mapped genes
    syngo_mapped = syngo_genes_in_deltas & mapped_genes
    tf_mapped = tf_genes_in_deltas & mapped_genes
    print(f"  SynGO mapped genes: {len(syngo_mapped)} / {len(syngo_genes_in_deltas)} (non-ribosomal)")
    print(f"  TF mapped genes: {len(tf_mapped)} / {len(tf_genes_in_deltas)}")

    # --- SynGO transfer summary ---
    syngo_rows = []
    for (cls, gene), delta in human_deltas_fine.items():
        if gene in syngo_mapped:
            syngo_rows.append({
                "human_class": cls,
                "mouse_model_gene": gene,
                "human_gene_symbol": mouse_to_human.get(gene, ""),
                "human_predicted_delta": f"{delta:.6f}",
            })
    save_csv(syngo_rows, os.path.join(RESULTS_DIR, "human_m1_syngo_transfer_summary.csv"))

    # --- TF transfer summary ---
    tf_rows = []
    for (cls, gene), delta in human_deltas_fine.items():
        if gene in tf_mapped:
            tf_rows.append({
                "human_class": cls,
                "mouse_model_gene": gene,
                "human_gene_symbol": mouse_to_human.get(gene, ""),
                "human_predicted_delta": f"{delta:.6f}",
            })
    save_csv(tf_rows, os.path.join(RESULTS_DIR, "human_m1_tf_transfer_summary.csv"))

    # --- Neuronal vs non-neuronal contrast ---
    contrast_rows = []
    for gene_set_name, gene_set in [("SynGO", syngo_mapped), ("TF", tf_mapped), ("all_mapped", mapped_genes)]:
        for broad_cls in sorted(set(r["group"] for (r_group, _), _ in
                                     [(((r,), None), None) for r in set()] )):
            pass  # Will compute below

        neuronal_deltas = []
        nonneuronal_deltas = []
        for (cls, gene), delta in human_deltas_broad.items():
            if gene not in gene_set:
                continue
            if cls in NEURONAL_BROAD:
                neuronal_deltas.append(delta)
            elif cls in NONNEURONAL_BROAD:
                nonneuronal_deltas.append(delta)

        if neuronal_deltas and nonneuronal_deltas:
            n_mean = np.mean(neuronal_deltas)
            nn_mean = np.mean(nonneuronal_deltas)
            contrast_rows.append({
                "gene_set": gene_set_name,
                "n_genes": len(gene_set),
                "neuronal_mean_delta": f"{n_mean:.6f}",
                "nonneuronal_mean_delta": f"{nn_mean:.6f}",
                "neuronal_minus_nonneuronal": f"{n_mean - nn_mean:.6f}",
                "neuronal_n_obs": len(neuronal_deltas),
                "nonneuronal_n_obs": len(nonneuronal_deltas),
            })
            print(f"  {gene_set_name}: neuronal={n_mean:.4f}, non-neuronal={nn_mean:.4f}, "
                  f"diff={n_mean - nn_mean:.4f}")

    save_csv(contrast_rows,
             os.path.join(RESULTS_DIR, "human_m1_neuronal_vs_nonneuronal_predicted_shift.csv"))

    # --- SynGO vs TF contrast ---
    syngo_tf_rows = []
    for cls in sorted(set(c for (c, _) in human_deltas_fine.keys())):
        syngo_d = [human_deltas_fine[(cls, g)] for g in syngo_mapped if (cls, g) in human_deltas_fine]
        tf_d = [human_deltas_fine[(cls, g)] for g in tf_mapped if (cls, g) in human_deltas_fine]
        if syngo_d and tf_d:
            syngo_tf_rows.append({
                "class": cls,
                "syngo_mean_delta": f"{np.mean(syngo_d):.6f}",
                "tf_mean_delta": f"{np.mean(tf_d):.6f}",
                "syngo_minus_tf": f"{np.mean(syngo_d) - np.mean(tf_d):.6f}",
                "n_syngo": len(syngo_d),
                "n_tf": len(tf_d),
            })
    save_csv(syngo_tf_rows,
             os.path.join(RESULTS_DIR, "human_m1_syngo_vs_tf_transfer_contrast.csv"))

    return syngo_mapped, tf_mapped


def load_mouse_allgene_deltas():
    """Load mouse per-CT predicted deltas for ALL genes from delta_analysis tables.

    These tables have columns: gene, median_delta, mean_delta, std_delta, frac_positive, n_cells
    and cover all 10,134 model genes per cell type.
    Returns dict: {(cell_type, gene): mean_delta}
    """
    import glob
    result = {}
    # Map file names to cell type names matching MOUSE_TO_HUMAN_CT keys
    ct_name_map = {
        "L2_3 IT": "L2/3 IT",
        "L5 IT": "L5 IT",
        "L5 ET": "L5 ET",
        "L5_6 NP": "L5/6 NP",
        "L6 CT": "L6 CT",
        "L6 IT": "L6 IT",
        "L6b": "L6b",
        "Lamp5": "Lamp5",
        "Sncg": "Sncg",
        "Vip": "Vip",
        "Sst": "Sst",
        "Pvalb": "Pvalb",
        "Astro": "Astro",
        "Oligo": "Oligo",
        "OPC": "OPC",
        "Endo": "Endo",
        "Macrophage": "Micro",
    }
    for fname in glob.glob(os.path.join(MOUSE_DELTA_TABLES, "*_delta.csv")):
        basename = os.path.basename(fname).replace("_delta.csv", "")
        ct = ct_name_map.get(basename)
        if ct is None:
            continue
        with open(fname) as f:
            for row in csv.DictReader(f):
                result[(ct, row["gene"])] = float(row["mean_delta"])
    print(f"  Loaded mouse all-gene deltas: {len(result)} (cell_type, gene) pairs "
          f"from {len(set(ct for ct, _ in result))} cell types")
    return result


def part_e_mouse_human_similarity(syngo_mapped, tf_mapped):
    """Compare mouse and human predicted deltas by shared cell class."""
    print("\n=== Part E4: Mouse-human predicted-shift similarity ===")

    mapped_genes, _ = load_human_gene_map()
    human_deltas = load_human_class_deltas("mapped_mouse_like_class")

    # Load true all-gene mouse deltas from delta_analysis tables
    mouse_all = load_mouse_allgene_deltas()

    # Also load SynGO/TF-specific mouse deltas for subset analyses
    mouse_syngo = load_mouse_deltas(SYNGO_DELTAS)
    mouse_tf = load_mouse_deltas(TF_DELTAS)

    similarity_rows = []

    for mouse_ct, human_ct in MOUSE_TO_HUMAN_CT.items():
        for gene_set_name, gene_filter, mouse_source in [
            ("all_mapped", mapped_genes, mouse_all),
            ("SynGO", syngo_mapped, mouse_syngo),
            ("TF", tf_mapped, mouse_tf),
        ]:
            mouse_vals = []
            human_vals = []
            for gene in gene_filter:
                m_key = (mouse_ct, gene)
                h_key = (human_ct, gene)
                if m_key in mouse_source and h_key in human_deltas:
                    mouse_vals.append(mouse_source[m_key])
                    human_vals.append(human_deltas[h_key])

            if len(mouse_vals) < 5:
                continue

            m_arr = np.array(mouse_vals)
            h_arr = np.array(human_vals)

            pearson_r, pearson_p = stats.pearsonr(m_arr, h_arr)
            spearman_r, spearman_p = stats.spearmanr(m_arr, h_arr)
            mae = np.mean(np.abs(m_arr - h_arr))
            rmse = np.sqrt(np.mean((m_arr - h_arr) ** 2))
            sign_agree = np.mean(np.sign(m_arr) == np.sign(h_arr))

            # Slope/intercept
            if np.std(m_arr) > 1e-8:
                slope, intercept = np.polyfit(m_arr, h_arr, 1)
            else:
                slope, intercept = 0.0, 0.0

            similarity_rows.append({
                "mouse_cell_type": mouse_ct,
                "human_class": human_ct,
                "gene_set": gene_set_name,
                "n_genes": len(mouse_vals),
                "pearson_r": f"{pearson_r:.4f}",
                "pearson_p": f"{pearson_p:.2e}",
                "spearman_r": f"{spearman_r:.4f}",
                "spearman_p": f"{spearman_p:.2e}",
                "mae": f"{mae:.4f}",
                "rmse": f"{rmse:.4f}",
                "sign_agreement": f"{sign_agree:.4f}",
                "slope": f"{slope:.4f}",
                "intercept": f"{intercept:.4f}",
            })

    save_csv(similarity_rows,
             os.path.join(RESULTS_DIR, "human_mouse_predicted_shift_similarity_by_class.csv"))

    # Print summary
    all_mapped = [r for r in similarity_rows if r["gene_set"] == "all_mapped"]
    if all_mapped:
        rs = [float(r["pearson_r"]) for r in all_mapped]
        print(f"  All-mapped Pearson: median={np.median(rs):.3f}, "
              f"range=[{min(rs):.3f}, {max(rs):.3f}], n_classes={len(rs)}")


def part_f_donor_consistency():
    """Donor consistency of predicted shifts."""
    print("\n=== Part F: Donor consistency ===")

    mapped_genes, _ = load_human_gene_map()

    # Use the non-ribosomal SynGO/TF gene sets from the deltas files
    mouse_syngo = load_mouse_deltas(SYNGO_DELTAS)
    mouse_tf = load_mouse_deltas(TF_DELTAS)
    syngo_mapped = set(g for (_, g) in mouse_syngo.keys()) & mapped_genes
    tf_mapped = set(g for (_, g) in mouse_tf.keys()) & mapped_genes

    consistency_rows = []

    for class_key, label in [("mapped_broad_class", "broad"), ("mapped_mouse_like_class", "fine")]:
        donor_deltas = load_human_donor_deltas(class_key)

        # Get donors and classes
        donors = sorted(set(d for d, _, _ in donor_deltas.keys()))
        classes = sorted(set(c for _, c, _ in donor_deltas.keys()))

        if len(donors) != 2:
            print(f"  WARNING: Expected 2 donors, got {len(donors)}: {donors}")
            continue

        d1, d2 = donors[0], donors[1]

        for cls in classes:
            for gene_set_name, gene_filter in [
                ("all_mapped", mapped_genes),
                ("SynGO", syngo_mapped),
                ("TF", tf_mapped),
            ]:
                d1_vals = []
                d2_vals = []
                for gene in gene_filter:
                    k1 = (d1, cls, gene)
                    k2 = (d2, cls, gene)
                    if k1 in donor_deltas and k2 in donor_deltas:
                        d1_vals.append(donor_deltas[k1])
                        d2_vals.append(donor_deltas[k2])

                if len(d1_vals) < 10:
                    continue

                d1_arr = np.array(d1_vals)
                d2_arr = np.array(d2_vals)
                r, p = stats.pearsonr(d1_arr, d2_arr)
                mad = np.mean(np.abs(d1_arr - d2_arr))
                sign_agree = np.mean(np.sign(d1_arr) == np.sign(d2_arr))

                consistency_rows.append({
                    "class_level": label,
                    "class": cls,
                    "gene_set": gene_set_name,
                    "n_genes": len(d1_vals),
                    "donor1": d1,
                    "donor2": d2,
                    "pearson_r": f"{r:.4f}",
                    "pearson_p": f"{p:.2e}",
                    "mean_abs_diff": f"{mad:.4f}",
                    "sign_agreement": f"{sign_agree:.4f}",
                })

    save_csv(consistency_rows,
             os.path.join(RESULTS_DIR, "human_m1_donor_consistency_by_class.csv"))

    # Print summary
    all_broad = [r for r in consistency_rows
                 if r["class_level"] == "broad" and r["gene_set"] == "all_mapped"]
    if all_broad:
        rs = [float(r["pearson_r"]) for r in all_broad]
        print(f"  Broad class donor consistency: median r={np.median(rs):.3f}")


def save_csv(rows, path):
    if not rows:
        print(f"  Skipped (empty): {path}")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {path} ({len(rows)} rows)")


def main():
    syngo_mapped, tf_mapped = part_e_syngo_tf_transfer()
    part_e_mouse_human_similarity(syngo_mapped, tf_mapped)
    part_f_donor_consistency()
    print("\nDone.")


if __name__ == "__main__":
    main()
