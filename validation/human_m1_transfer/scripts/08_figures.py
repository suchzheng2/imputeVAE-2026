#!/usr/bin/env python3
"""
Task Group 2, Part G: Generate all figures for human M1 transfer pilot.
"""

import os
import sys
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats

ROOT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE"
RESULTS_DIR = os.path.join(ROOT, "validation/human_m1_transfer/results")
INF_DIR = os.path.join(ROOT, "validation/human_m1_transfer/inference")
FIG_DIR = os.path.join(ROOT, "validation/human_m1_transfer/figures")
os.makedirs(FIG_DIR, exist_ok=True)


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fig1_transfer_qc():
    """QC figure: input/output distributions, gene coverage, class counts."""
    print("=== Figure 1: Transfer QC ===")

    # Load QC info
    with open(os.path.join(INF_DIR, "human_m1_inference_run_info.json")) as f:
        info = json.load(f)

    obs = load_csv(os.path.join(INF_DIR, "human_m1_inference_obs.csv"))
    gene_map = load_csv(os.path.join(INF_DIR, "human_m1_inference_gene_map.csv"))

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Human M1 Zero-Shot Transfer QC", fontsize=14, fontweight="bold")

    # Panel 1: Nuclei by broad class
    ax = axes[0, 0]
    from collections import Counter
    broad_counts = Counter(r["mapped_broad_class"] for r in obs)
    classes = [c for c, _ in broad_counts.most_common()]
    counts = [broad_counts[c] for c in classes]
    ax.barh(range(len(classes)), counts, color=plt.cm.Set2(np.linspace(0, 1, len(classes))))
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Nuclei")
    ax.set_title("Nuclei by broad class")
    for i, c in enumerate(counts):
        ax.text(c + max(counts)*0.01, i, f"{c:,}", va="center", fontsize=7)

    # Panel 2: Nuclei by mouse-like class (top 15)
    ax = axes[0, 1]
    fine_counts = Counter(r["mapped_mouse_like_class"] for r in obs)
    classes_f = [c for c, _ in fine_counts.most_common()[:15]]
    counts_f = [fine_counts[c] for c in classes_f]
    ax.barh(range(len(classes_f)), counts_f, color=plt.cm.tab20(np.linspace(0, 1, len(classes_f))))
    ax.set_yticks(range(len(classes_f)))
    ax.set_yticklabels(classes_f, fontsize=8)
    ax.set_xlabel("Nuclei")
    ax.set_title("Mouse-like classes (top 15)")

    # Panel 3: Gene coverage
    ax = axes[0, 2]
    status_counts = Counter(r["filled_value_status"] for r in gene_map)
    labels = list(status_counts.keys())
    sizes = [status_counts[l] for l in labels]
    colors = {"mapped": "#2ecc71", "no_ortholog": "#e74c3c", "missing_human_gene": "#f39c12"}
    c = [colors.get(l, "#95a5a6") for l in labels]
    ax.pie(sizes, labels=[f"{l}\n({s:,})" for l, s in zip(labels, sizes)],
           colors=c, autopct="%1.1f%%", textprops={"fontsize": 8})
    ax.set_title("Model gene coverage")

    # Panel 4: Output value distribution (sample)
    ax = axes[1, 0]
    # Load a small sample to avoid OOM
    imputed_sample = np.load(os.path.join(INF_DIR, "human_m1_imputed_cell_log1p.npy"),
                             mmap_mode="r")[:5000].flatten()
    # Subsample for histogram
    sub = imputed_sample[imputed_sample != 0]
    if len(sub) > 100000:
        sub = np.random.choice(sub, 100000, replace=False)
    ax.hist(sub, bins=100, color="steelblue", edgecolor="none", alpha=0.8)
    ax.set_xlabel("Imputed log1p value (non-zero)")
    ax.set_ylabel("Count")
    ax.set_title("Output distribution (sample)")
    ax.axvline(0, color="red", linestyle="--", alpha=0.5)

    # Panel 5: Output stats text
    ax = axes[1, 1]
    ax.axis("off")
    ostat = info["output_stats"]
    istat = info["input_stats"]
    text = (
        f"Inference Summary\n"
        f"{'='*35}\n"
        f"Nuclei:          {info['n_nuclei']:,}\n"
        f"Model genes:     {info['n_model_genes']:,}\n"
        f"Mapped genes:    {info['n_mapped_genes']:,}\n"
        f"Device:          {info['device']}\n"
        f"Time:            {info['inference_minutes']:.1f} min\n"
        f"\nInput Stats\n"
        f"{'='*35}\n"
        f"NNZ:             {istat['nnz']:,}\n"
        f"Frac zero:       {istat['fraction_zero']:.4f}\n"
        f"\nOutput Stats\n"
        f"{'='*35}\n"
        f"Min:             {ostat['min']:.4f}\n"
        f"Max:             {ostat['max']:.4f}\n"
        f"Mean:            {ostat['mean']:.4f}\n"
        f"Std:             {ostat['std']:.4f}\n"
        f"Frac negative:   {ostat['fraction_negative']:.4f}\n"
        f"NaN/Inf:         {ostat['n_nan']}/{ostat['n_inf']}\n"
    )
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9,
            va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    # Panel 6: Detected genes per nucleus
    ax = axes[1, 2]
    n_det = [int(r["n_genes_detected"]) for r in obs]
    ax.hist(n_det, bins=50, color="coral", edgecolor="white", alpha=0.8)
    ax.axvline(np.median(n_det), color="red", linestyle="--",
               label=f"median={np.median(n_det):.0f}")
    ax.set_xlabel("Genes detected (mapped)")
    ax.set_ylabel("Nuclei")
    ax.set_title("Genes detected per nucleus")
    ax.legend(fontsize=8)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(os.path.join(FIG_DIR, f"human_m1_transfer_qc.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved human_m1_transfer_qc.png/pdf")


def fig2_predicted_delta():
    """Delta summary by class, SynGO vs TF contrast."""
    print("=== Figure 2: Predicted delta summary ===")

    broad_data = load_csv(os.path.join(RESULTS_DIR, "human_m1_broad_class_gene_means.csv"))
    fine_data = load_csv(os.path.join(RESULTS_DIR, "human_m1_mouse_like_class_gene_means.csv"))
    contrast = load_csv(os.path.join(RESULTS_DIR, "human_m1_syngo_vs_tf_transfer_contrast.csv"))
    nn_contrast = load_csv(os.path.join(RESULTS_DIR,
                           "human_m1_neuronal_vs_nonneuronal_predicted_shift.csv"))

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Human M1 Predicted Nucleus-to-Cell Shifts", fontsize=14, fontweight="bold")

    # Panel 1: Mean delta by broad class
    ax = axes[0, 0]
    broad_deltas = defaultdict(list)
    for r in broad_data:
        broad_deltas[r["group"]].append(float(r["predicted_delta"]))
    classes = sorted(broad_deltas.keys(), key=lambda c: np.mean(broad_deltas[c]), reverse=True)
    means = [np.mean(broad_deltas[c]) for c in classes]
    colors = ["#e74c3c" if m < 0 else "#2ecc71" for m in means]
    ax.barh(range(len(classes)), means, color=colors)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Mean predicted delta (log1p)")
    ax.set_title("All mapped genes: broad class")
    ax.axvline(0, color="black", linewidth=0.5)

    # Panel 2: Mean delta by mouse-like class (top 15)
    ax = axes[0, 1]
    fine_deltas = defaultdict(list)
    for r in fine_data:
        fine_deltas[r["group"]].append(float(r["predicted_delta"]))
    classes_f = sorted(fine_deltas.keys(), key=lambda c: np.mean(fine_deltas[c]), reverse=True)[:15]
    means_f = [np.mean(fine_deltas[c]) for c in classes_f]
    colors_f = ["#e74c3c" if m < 0 else "#2ecc71" for m in means_f]
    ax.barh(range(len(classes_f)), means_f, color=colors_f)
    ax.set_yticks(range(len(classes_f)))
    ax.set_yticklabels(classes_f, fontsize=8)
    ax.set_xlabel("Mean predicted delta (log1p)")
    ax.set_title("All mapped genes: mouse-like class")
    ax.axvline(0, color="black", linewidth=0.5)

    # Panel 3: SynGO delta by broad class
    ax = axes[0, 2]
    syngo_broad = load_csv(os.path.join(RESULTS_DIR, "human_m1_syngo_transfer_summary.csv"))
    # Aggregate SynGO deltas by matching to broad class via fine->broad lookup
    # For simplicity, use the contrast file
    if nn_contrast:
        gene_sets = [r["gene_set"] for r in nn_contrast]
        n_vals = [float(r["neuronal_mean_delta"]) for r in nn_contrast]
        nn_vals = [float(r["nonneuronal_mean_delta"]) for r in nn_contrast]

        x = np.arange(len(gene_sets))
        width = 0.35
        ax.bar(x - width/2, n_vals, width, label="Neuronal", color="#3498db")
        ax.bar(x + width/2, nn_vals, width, label="Non-neuronal", color="#e67e22")
        ax.set_xticks(x)
        ax.set_xticklabels(gene_sets, fontsize=9)
        ax.set_ylabel("Mean predicted delta")
        ax.set_title("Neuronal vs non-neuronal shift")
        ax.legend(fontsize=8)
        ax.axhline(0, color="black", linewidth=0.5)

    # Panel 4: SynGO vs TF contrast by class
    ax = axes[1, 0]
    if contrast:
        classes_c = [r["class"] for r in contrast[:15]]
        syngo_d = [float(r["syngo_mean_delta"]) for r in contrast[:15]]
        tf_d = [float(r["tf_mean_delta"]) for r in contrast[:15]]
        x = np.arange(len(classes_c))
        width = 0.35
        ax.barh(x - width/2, syngo_d, width, label="SynGO", color="#9b59b6")
        ax.barh(x + width/2, tf_d, width, label="TF", color="#1abc9c")
        ax.set_yticks(x)
        ax.set_yticklabels(classes_c, fontsize=7)
        ax.set_xlabel("Mean predicted delta")
        ax.set_title("SynGO vs TF by class")
        ax.legend(fontsize=8)
        ax.axvline(0, color="black", linewidth=0.5)

    # Panel 5: Delta distribution violin (broad neuronal vs non-neuronal)
    ax = axes[1, 1]
    neuronal_d = []
    nonneuronal_d = []
    NEURONAL = {"excitatory_neuron", "inhibitory_neuron"}
    for r in broad_data:
        d = float(r["predicted_delta"])
        if r["group"] in NEURONAL:
            neuronal_d.append(d)
        else:
            nonneuronal_d.append(d)
    # Subsample for plotting
    np.random.seed(42)
    if len(neuronal_d) > 50000:
        neuronal_d = list(np.random.choice(neuronal_d, 50000, replace=False))
    if len(nonneuronal_d) > 50000:
        nonneuronal_d = list(np.random.choice(nonneuronal_d, 50000, replace=False))
    parts = ax.violinplot([neuronal_d, nonneuronal_d], positions=[0, 1], showmedians=True)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Neuronal", "Non-neuronal"])
    ax.set_ylabel("Predicted delta")
    ax.set_title("Delta distribution by neuronality")
    ax.axhline(0, color="black", linewidth=0.5)

    # Panel 6: Summary text
    ax = axes[1, 2]
    ax.axis("off")
    text_lines = ["Predicted Delta Summary", "=" * 35]
    for cls in sorted(broad_deltas.keys()):
        d = broad_deltas[cls]
        n_genes = len(set(r["mouse_model_gene"] for r in broad_data if r["group"] == cls))
        text_lines.append(f"{cls}: {np.mean(d):.4f} (n={n_genes} genes)")
    ax.text(0.05, 0.95, "\n".join(text_lines[:15]), transform=ax.transAxes, fontsize=8,
            va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(os.path.join(FIG_DIR, f"human_m1_predicted_delta_summary.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved human_m1_predicted_delta_summary.png/pdf")


def fig3_mouse_human_similarity():
    """Heatmap and scatter of mouse-human shift similarity."""
    print("=== Figure 3: Mouse-human shift similarity ===")

    sim_data = load_csv(os.path.join(RESULTS_DIR,
                        "human_mouse_predicted_shift_similarity_by_class.csv"))
    if not sim_data:
        print("  No similarity data found, skipping.")
        return

    # Filter to all_mapped
    all_mapped = [r for r in sim_data if r["gene_set"] == "all_mapped"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Mouse-Human Predicted Shift Similarity", fontsize=14, fontweight="bold")

    # Panel 1: Heatmap of Pearson by class
    ax = axes[0]
    classes = sorted(set(r["human_class"] for r in all_mapped))
    gene_sets = ["all_mapped", "SynGO", "TF"]
    heatmap = np.full((len(classes), len(gene_sets)), np.nan)
    for r in sim_data:
        if r["human_class"] in classes:
            ci = classes.index(r["human_class"])
            if r["gene_set"] in gene_sets:
                gi = gene_sets.index(r["gene_set"])
                heatmap[ci, gi] = float(r["pearson_r"])

    im = ax.imshow(heatmap, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(gene_sets)))
    ax.set_xticklabels(gene_sets, fontsize=9)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_title("Pearson r: mouse vs human delta")
    plt.colorbar(im, ax=ax, shrink=0.8)
    # Annotate
    for i in range(len(classes)):
        for j in range(len(gene_sets)):
            if not np.isnan(heatmap[i, j]):
                ax.text(j, i, f"{heatmap[i, j]:.2f}", ha="center", va="center", fontsize=7)

    # Panel 2: Sign agreement bar plot
    ax = axes[1]
    sign_data = [r for r in all_mapped]
    classes_s = [r["human_class"] for r in sign_data]
    signs = [float(r["sign_agreement"]) for r in sign_data]
    y = range(len(classes_s))
    ax.barh(y, signs, color="steelblue")
    ax.set_yticks(y)
    ax.set_yticklabels(classes_s, fontsize=8)
    ax.set_xlabel("Sign agreement fraction")
    ax.set_title("Delta sign agreement (all mapped)")
    ax.axvline(0.5, color="red", linestyle="--", alpha=0.5, label="chance")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8)

    # Panel 3: Summary Pearson distribution
    ax = axes[2]
    for gs, color in [("all_mapped", "steelblue"), ("SynGO", "#9b59b6"), ("TF", "#1abc9c")]:
        rs = [float(r["pearson_r"]) for r in sim_data if r["gene_set"] == gs]
        if rs:
            ax.hist(rs, bins=15, alpha=0.5, label=f"{gs} (n={len(rs)}, med={np.median(rs):.2f})",
                    color=color, edgecolor="white")
    ax.set_xlabel("Pearson r")
    ax.set_ylabel("Cell classes")
    ax.set_title("Distribution of mouse-human Pearson r")
    ax.axvline(0, color="red", linestyle="--", alpha=0.5)
    ax.legend(fontsize=8)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(os.path.join(FIG_DIR, f"human_mouse_shift_similarity.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved human_mouse_shift_similarity.png/pdf")


def fig4_donor_consistency():
    """Donor consistency figures."""
    print("=== Figure 4: Donor consistency ===")

    cons_data = load_csv(os.path.join(RESULTS_DIR, "human_m1_donor_consistency_by_class.csv"))
    if not cons_data:
        print("  No consistency data, skipping.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Donor Consistency of Predicted Shifts", fontsize=14, fontweight="bold")

    # Panel 1: Pearson by class (broad, all_mapped)
    ax = axes[0]
    broad_all = [r for r in cons_data
                 if r["class_level"] == "broad" and r["gene_set"] == "all_mapped"]
    if broad_all:
        classes = [r["class"] for r in broad_all]
        rs = [float(r["pearson_r"]) for r in broad_all]
        colors = plt.cm.Set2(np.linspace(0, 1, len(classes)))
        ax.barh(range(len(classes)), rs, color=colors)
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes, fontsize=8)
        ax.set_xlabel("Donor Pearson r")
        ax.set_title("Broad classes (all mapped)")
        ax.set_xlim(0, 1)

    # Panel 2: Pearson by class (fine, all_mapped)
    ax = axes[1]
    fine_all = [r for r in cons_data
                if r["class_level"] == "fine" and r["gene_set"] == "all_mapped"]
    if fine_all:
        fine_all.sort(key=lambda r: float(r["pearson_r"]), reverse=True)
        classes_f = [r["class"] for r in fine_all[:15]]
        rs_f = [float(r["pearson_r"]) for r in fine_all[:15]]
        ax.barh(range(len(classes_f)), rs_f, color="steelblue")
        ax.set_yticks(range(len(classes_f)))
        ax.set_yticklabels(classes_f, fontsize=8)
        ax.set_xlabel("Donor Pearson r")
        ax.set_title("Mouse-like classes (all mapped)")
        ax.set_xlim(0, 1)

    # Panel 3: Mean absolute donor difference
    ax = axes[2]
    if fine_all:
        classes_f2 = [r["class"] for r in fine_all[:15]]
        mads = [float(r["mean_abs_diff"]) for r in fine_all[:15]]
        ax.barh(range(len(classes_f2)), mads, color="coral")
        ax.set_yticks(range(len(classes_f2)))
        ax.set_yticklabels(classes_f2, fontsize=8)
        ax.set_xlabel("Mean |donor1 - donor2| delta")
        ax.set_title("Donor difference (mouse-like)")

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(os.path.join(FIG_DIR, f"human_m1_donor_consistency.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved human_m1_donor_consistency.png/pdf")


def main():
    fig1_transfer_qc()
    fig2_predicted_delta()
    fig3_mouse_human_similarity()
    fig4_donor_consistency()
    print("\nAll figures saved.")


if __name__ == "__main__":
    main()
