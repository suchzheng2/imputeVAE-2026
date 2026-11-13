#!/usr/bin/env python3
"""
Task Group 2, Part I: Acceptance checks.

Validates all deliverables exist and are consistent.
"""

import os
import sys
import csv
import json
import numpy as np
from scipy import sparse

ROOT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE"
INF_DIR = os.path.join(ROOT, "validation/human_m1_transfer/inference")
RESULTS_DIR = os.path.join(ROOT, "validation/human_m1_transfer/results")
FIG_DIR = os.path.join(ROOT, "validation/human_m1_transfer/figures")
REPORT_DIR = os.path.join(ROOT, "validation/human_m1_transfer/reports")
PROC_DIR = os.path.join(ROOT, "validation/human_m1_transfer/processed")

checks = []
warnings = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    checks.append((name, status, detail))
    symbol = "[OK]" if condition else "[FAIL]"
    print(f"  {symbol} {name}" + (f" -- {detail}" if detail else ""))
    return condition


def main():
    print("=== Task Group 2 Acceptance Checks ===\n")

    # --- Inference outputs ---
    print("Inference outputs:")
    imp_path = os.path.join(INF_DIR, "human_m1_imputed_cell_log1p.npy")
    check("Imputed matrix exists", os.path.exists(imp_path))

    imputed = np.load(imp_path, mmap_mode="r")
    check("Output shape rows = 76533", imputed.shape[0] == 76533, f"got {imputed.shape[0]}")
    check("Output shape cols = 10134", imputed.shape[1] == 10134, f"got {imputed.shape[1]}")

    # Sample for NaN/Inf check (mmap doesn't load all)
    sample = np.array(imputed[:1000])
    check("No NaNs in sample", not np.any(np.isnan(sample)))
    check("No Infs in sample", not np.any(np.isinf(sample)))

    inp_path = os.path.join(INF_DIR, "human_m1_input_nucleus_log1p.npz")
    check("Input matrix exists", os.path.exists(inp_path))
    inp = sparse.load_npz(inp_path)
    check("Input shape = (76533, 10134)", inp.shape == (76533, 10134), f"got {inp.shape}")

    mask = np.load(os.path.join(PROC_DIR, "human_m1_available_gene_mask.npy"))
    check("Mask sum = 8215", int(mask.sum()) == 8215, f"got {int(mask.sum())}")

    obs_path = os.path.join(INF_DIR, "human_m1_inference_obs.csv")
    check("Obs exists", os.path.exists(obs_path))
    with open(obs_path) as f:
        n_obs = sum(1 for _ in f) - 1
    check("Obs rows = 76533", n_obs == 76533, f"got {n_obs}")

    info_path = os.path.join(INF_DIR, "human_m1_inference_run_info.json")
    check("Run info JSON exists", os.path.exists(info_path))
    with open(info_path) as f:
        info = json.load(f)

    check("Model checkpoint loaded",
          "model_run" in info and "VAE_JSD_Cond" in info.get("model_class", ""))
    check("Inference completed for all nuclei",
          info.get("n_nuclei") == 76533, f"got {info.get('n_nuclei')}")

    # Output stats
    ostat = info.get("output_stats", {})
    out_min = ostat.get("min", None)
    out_max = ostat.get("max", None)
    out_mean = ostat.get("mean", None)
    frac_neg = ostat.get("fraction_negative", None)
    print(f"\n  Output: min={out_min}, max={out_max}, mean={out_mean}, frac_neg={frac_neg}")

    # --- Pseudobulk summaries ---
    print("\nPseudobulk summaries:")
    for fname in [
        "human_m1_broad_class_gene_means.csv",
        "human_m1_mouse_like_class_gene_means.csv",
        "human_m1_donor_broad_class_gene_means.csv",
        "human_m1_donor_mouse_like_class_gene_means.csv",
        "human_m1_nucleus_broad_class_mean_matrix.csv",
        "human_m1_imputed_broad_class_mean_matrix.csv",
        "human_m1_predicted_delta_broad_class_matrix.csv",
        "human_m1_nucleus_mouse_like_class_mean_matrix.csv",
        "human_m1_imputed_mouse_like_class_mean_matrix.csv",
        "human_m1_predicted_delta_mouse_like_class_matrix.csv",
    ]:
        path = os.path.join(RESULTS_DIR, fname)
        check(f"Pseudobulk: {fname}", os.path.exists(path))

    # Check mapped-genes-only
    broad_data = list(csv.DictReader(open(os.path.join(RESULTS_DIR, "human_m1_broad_class_gene_means.csv"))))
    all_mapped = all(r["filled_value_status"] == "mapped" for r in broad_data)
    check("Pseudobulk includes only mapped genes", all_mapped)

    # --- SynGO/TF ---
    print("\nSynGO/TF transfer:")
    syngo_path = os.path.join(RESULTS_DIR, "human_m1_syngo_transfer_summary.csv")
    tf_path = os.path.join(RESULTS_DIR, "human_m1_tf_transfer_summary.csv")
    check("SynGO transfer summary exists", os.path.exists(syngo_path))
    check("TF transfer summary exists", os.path.exists(tf_path))

    if os.path.exists(syngo_path):
        syngo_genes = set(r["mouse_model_gene"] for r in csv.DictReader(open(syngo_path)))
        print(f"  SynGO mapped genes used: {len(syngo_genes)}")
    if os.path.exists(tf_path):
        tf_genes = set(r["mouse_model_gene"] for r in csv.DictReader(open(tf_path)))
        print(f"  TF mapped genes used: {len(tf_genes)}")

    for fname in [
        "human_m1_syngo_vs_tf_transfer_contrast.csv",
        "human_m1_neuronal_vs_nonneuronal_predicted_shift.csv",
        "human_mouse_predicted_shift_similarity_by_class.csv",
    ]:
        check(f"Comparison: {fname}", os.path.exists(os.path.join(RESULTS_DIR, fname)))

    # --- Donor consistency ---
    print("\nDonor consistency:")
    cons_path = os.path.join(RESULTS_DIR, "human_m1_donor_consistency_by_class.csv")
    check("Donor consistency exists", os.path.exists(cons_path))
    if os.path.exists(cons_path):
        cons = list(csv.DictReader(open(cons_path)))
        broad_rs = [float(r["pearson_r"]) for r in cons
                    if r["class_level"] == "broad" and r["gene_set"] == "all_mapped"]
        if broad_rs:
            print(f"  Donor consistency median Pearson (broad): {np.median(broad_rs):.3f}")

    # --- Figures ---
    print("\nFigures:")
    for fname in [
        "human_m1_transfer_qc.png",
        "human_m1_predicted_delta_summary.png",
        "human_mouse_shift_similarity.png",
        "human_m1_donor_consistency.png",
    ]:
        check(f"Figure: {fname}", os.path.exists(os.path.join(FIG_DIR, fname)))

    # --- Summary ---
    n_pass = sum(1 for _, s, _ in checks if s == "PASS")
    n_fail = sum(1 for _, s, _ in checks if s == "FAIL")
    print(f"\n{'='*50}")
    print(f"RESULTS: {n_pass} passed, {n_fail} failed")
    print(f"{'='*50}")

    # --- Print expected output summary ---
    print(f"\n=== TASK GROUP 2 OUTPUT SUMMARY ===")
    print(f"  n_nuclei imputed: {info.get('n_nuclei', 'N/A')}")
    print(f"  n_model_genes: 10134")
    print(f"  n_interpretable_mapped_genes: 8215")
    print(f"  output file: {imp_path}")

    # Broad class delta summary
    broad_deltas = {}
    for r in broad_data:
        cls = r["group"]
        d = float(r["predicted_delta"])
        broad_deltas.setdefault(cls, []).append(d)
    print(f"\n  Broad classes summarized: {len(broad_deltas)}")
    fine_path = os.path.join(RESULTS_DIR, "human_m1_mouse_like_class_gene_means.csv")
    fine_classes = set(r["group"] for r in csv.DictReader(open(fine_path)))
    print(f"  Mouse-like classes summarized: {len(fine_classes)}")

    if os.path.exists(syngo_path):
        print(f"  SynGO mapped genes: {len(syngo_genes)}")
    if os.path.exists(tf_path):
        print(f"  TF mapped genes: {len(tf_genes)}")

    print(f"\n  Top 5 broad classes by mean predicted delta:")
    sorted_broad = sorted(broad_deltas.items(), key=lambda x: np.mean(x[1]), reverse=True)
    for cls, ds in sorted_broad[:5]:
        print(f"    {cls}: {np.mean(ds):.4f}")

    if os.path.exists(cons_path) and broad_rs:
        print(f"\n  Donor consistency median Pearson: {np.median(broad_rs):.3f}")

    print(f"\n  Warnings/caveats:")
    print(f"    - Zero-shot cross-species transfer (no human ground truth)")
    print(f"    - Only 8215/10134 genes interpretable (one-to-one orthologs)")
    print(f"    - 2 donors only")
    if frac_neg and frac_neg > 0.01:
        print(f"    - {frac_neg*100:.1f}% negative output values (not clamped)")
    print(f"    - L5 ET: interpret cautiously (mouse validation calibration issue)")

    # Save to file
    out_path = os.path.join(REPORT_DIR, "task_group_2_acceptance_checks.txt")
    with open(out_path, "w") as f:
        f.write("Task Group 2 Acceptance Checks\n")
        f.write(f"{'='*50}\n\n")
        for name, status, detail in checks:
            f.write(f"[{status}] {name}")
            if detail:
                f.write(f" -- {detail}")
            f.write("\n")
        f.write(f"\nTotal: {n_pass} passed, {n_fail} failed\n")
    print(f"\nSaved: {out_path}")

    return n_fail == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
