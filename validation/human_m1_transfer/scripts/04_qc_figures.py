#!/usr/bin/env python3
"""Part E: Generate QC figures for human M1 transfer input."""

import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

ROOT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE"
PROC_DIR = os.path.join(ROOT, "validation/human_m1_transfer/processed")
REPORT_DIR = os.path.join(ROOT, "validation/human_m1_transfer/reports")
ORTH_DIR = os.path.join(ROOT, "validation/human_m1_transfer/orthologs")

# Load obs
obs_path = os.path.join(PROC_DIR, "human_m1_nuclei_obs.csv")
print(f"Loading {obs_path}...")
with open(obs_path) as f:
    reader = csv.DictReader(f)
    obs = list(reader)
print(f"  {len(obs)} nuclei")

broad_classes = [r["mapped_broad_class"] for r in obs]
fine_classes = [r["mapped_mouse_like_class"] for r in obs]
n_detected = [int(r["n_genes_detected"]) for r in obs]
total_counts = [float(r["total_counts"]) for r in obs]

# Load gene mapping
gene_map_path = os.path.join(PROC_DIR, "human_m1_gene_mapping_for_model_order.csv")
with open(gene_map_path) as f:
    gene_map = list(csv.DictReader(f))

status_counts = Counter(r["filled_value_status"] for r in gene_map)

# --- Figure ---
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Human M1 10x Transfer Input QC", fontsize=14, fontweight="bold")

# Panel 1: Nuclei by broad class
ax = axes[0, 0]
broad_counter = Counter(broad_classes)
classes = [c for c, _ in broad_counter.most_common()]
counts = [broad_counter[c] for c in classes]
colors = plt.cm.Set2(np.linspace(0, 1, len(classes)))
bars = ax.barh(range(len(classes)), counts, color=colors)
ax.set_yticks(range(len(classes)))
ax.set_yticklabels(classes, fontsize=9)
ax.set_xlabel("Number of nuclei")
ax.set_title("Nuclei by broad class")
for bar, count in zip(bars, counts):
    ax.text(bar.get_width() + max(counts)*0.01, bar.get_y() + bar.get_height()/2,
            f"{count:,}", va="center", fontsize=8)

# Panel 2: Nuclei by mouse-like class
ax = axes[0, 1]
fine_counter = Counter(fine_classes)
classes_f = [c for c, _ in fine_counter.most_common()[:15]]  # top 15
counts_f = [fine_counter[c] for c in classes_f]
colors_f = plt.cm.tab20(np.linspace(0, 1, len(classes_f)))
bars = ax.barh(range(len(classes_f)), counts_f, color=colors_f)
ax.set_yticks(range(len(classes_f)))
ax.set_yticklabels(classes_f, fontsize=9)
ax.set_xlabel("Number of nuclei")
ax.set_title("Nuclei by mouse-like class (top 15)")
for bar, count in zip(bars, counts_f):
    ax.text(bar.get_width() + max(counts_f)*0.01, bar.get_y() + bar.get_height()/2,
            f"{count:,}", va="center", fontsize=8)

# Panel 3: Genes detected per nucleus
ax = axes[0, 2]
ax.hist(n_detected, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
ax.axvline(np.median(n_detected), color="red", linestyle="--", label=f"median={np.median(n_detected):.0f}")
ax.set_xlabel("Genes detected (mapped orthologs)")
ax.set_ylabel("Number of nuclei")
ax.set_title("Detected genes per nucleus")
ax.legend(fontsize=9)

# Panel 4: Total counts per nucleus
ax = axes[1, 0]
ax.hist(total_counts, bins=50, color="coral", edgecolor="white", alpha=0.8)
ax.axvline(np.median(total_counts), color="red", linestyle="--", label=f"median={np.median(total_counts):.0f}")
ax.set_xlabel("Total counts (mapped orthologs)")
ax.set_ylabel("Number of nuclei")
ax.set_title("Total counts per nucleus")
ax.legend(fontsize=9)

# Panel 5: Model gene coverage
ax = axes[1, 1]
labels = ["Mapped\northolog", "No ortholog", "Missing in\nhuman matrix"]
sizes = [status_counts.get("mapped", 0),
         status_counts.get("no_ortholog", 0),
         status_counts.get("missing_human_gene", 0)]
colors_pie = ["#2ecc71", "#e74c3c", "#f39c12"]
wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_pie,
                                   autopct="%1.1f%%", startangle=90, textprops={"fontsize": 9})
ax.set_title(f"Model gene coverage\n({sizes[0]:,} / {sum(sizes):,} mapped)")

# Panel 6: Ortholog mapping summary text
ax = axes[1, 2]
ax.axis("off")
summary_text = (
    f"Ortholog Mapping Summary\n"
    f"{'='*35}\n"
    f"Mouse model genes:     {sum(sizes):,}\n"
    f"Mapped (1:1 ortholog): {sizes[0]:,} ({100*sizes[0]/sum(sizes):.1f}%)\n"
    f"No ortholog:           {sizes[1]:,}\n"
    f"Missing in human:      {sizes[2]:,}\n"
    f"\n"
    f"Human M1 Data\n"
    f"{'='*35}\n"
    f"Total nuclei:          {len(obs):,}\n"
    f"Donors:                2\n"
    f"Broad classes:         {len(Counter(broad_classes))}\n"
    f"Mouse-like classes:    {len(Counter(fine_classes))}\n"
    f"Median genes/nucleus:  {np.median(n_detected):.0f}\n"
    f"Median counts/nucleus: {np.median(total_counts):.0f}\n"
)
ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

plt.tight_layout()
out_png = os.path.join(REPORT_DIR, "human_m1_input_qc.png")
out_pdf = os.path.join(REPORT_DIR, "human_m1_input_qc.pdf")
plt.savefig(out_png, dpi=150, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
