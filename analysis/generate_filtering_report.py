"""Generate a PDF report of filtering cutoff analysis with plots and commentary."""

import os
from datetime import date

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


PLOT_DIR = "analysis/filtering_cutoffs"
OUTPUT = "analysis/filtering_cutoffs/filtering_cutoff_report.pdf"

# Each section: (title, [(image_path, caption_lines), ...], commentary_lines)
SECTIONS = [
    # ── Section 1: Cell-level QC ──
    (
        "1. Cell-Level QC: Genes Expressed per Cell",
        [
            (
                "cell_cell_genes_expressed.png",
                [
                    "Left: histogram of the number of expressed genes per cell.",
                    "Right: cumulative curve — how many cells survive at each min-genes cutoff.",
                    "Dashed lines mark the 1st, 5th, and 10th percentiles of the distribution.",
                ],
            ),
            (
                "nuclei_cell_genes_expressed.png",
                [
                    "Same plots for nuclei. Note the bimodal distribution (peaks ~1000 and ~3000),",
                    "likely reflecting different cell types or sequencing depth batches.",
                ],
            ),
        ],
        [
            "Conclusions:",
            "• Cell data: all cells express >1300 genes. The old cutoff of 200 filters nothing (100% pass).",
            "  Meaningful cutoffs start around 2100 (1st %ile) — even that only removes 0.9%.",
            "• Nuclei data: broader spread, 1st percentile at ~1000 genes. A cutoff of 1500 removes ~20%",
            "  of nuclei — mostly from the low-UMI tail of the bimodal distribution.",
            "• Nuclei are the constraint: aggressive min-genes filtering (>2000) would lose ~48% of nuclei.",
        ],
    ),
    (
        "2. Cell-Level QC: Library Size (Total UMI Counts)",
        [
            (
                "cell_cell_library_size.png",
                [
                    "Left: histogram of log10(total counts) per cell.",
                    "Right: cumulative survival curve on a log x-axis.",
                    "Dashed lines at data-driven percentiles (1st/5th/10th).",
                ],
            ),
            (
                "nuclei_cell_library_size.png",
                [
                    "Nuclei library sizes. Strongly bimodal: peaks at ~2500 and ~12000.",
                    "The low-UMI peak is the population most affected by filtering.",
                ],
            ),
        ],
        [
            "Conclusions:",
            "• Cell median library size is 17,103; nuclei median is 4,150 — a ~4x difference.",
            "• The old cutoffs (500/1000/2000) are far below the data range for cells (min ~3800).",
            "• For nuclei, lib>2000 removes ~7.4% (mostly from the low-UMI peak); lib>5000 removes 58%.",
            "• Recommendation: lib>1000 is effectively no-op for both. lib>2000 is a light nuclei filter.",
        ],
    ),
    # ── Section 2: Gene-level filtering ──
    (
        "3. Gene-Level Filtering: Detection Rate",
        [
            (
                "cell_gene_detection_rate.png",
                [
                    "Left: histogram of per-gene detection rate (% of cells expressing each gene). Log y-axis.",
                    "Right: genes surviving at each detection rate threshold.",
                    "The steep drop near 0% shows thousands of genes expressed in very few cells.",
                ],
            ),
            (
                "nuclei_gene_detection_rate.png",
                [
                    "Nuclei gene detection rates. Same pattern but shifted — fewer genes pass",
                    "each threshold because nuclei detect fewer genes overall.",
                ],
            ),
        ],
        [
            "Conclusions:",
            "• ~40% of genes (cell) / ~45% of genes (nuclei) have detection rate <1%.",
            "  These are noise genes contributing to softmax dimension without biological signal.",
            "• At det>5%: 12,659 cell genes (47%) and 9,938 nuclei genes (38%) survive.",
            "• At det>10%: 11,084 cell genes but only 7,238 nuclei genes — nuclei is the bottleneck.",
            "• The 1%→5% range is where the most noise genes get removed per unit of threshold increase.",
        ],
    ),
    (
        "4. Gene-Level Filtering: Mean Expression",
        [
            (
                "cell_gene_mean_expression.png",
                [
                    "Left: histogram of log10(mean expression) per gene.",
                    "Right: genes surviving at each mean expression cutoff.",
                    "Cutoff lines at mean > 0.001, 0.01, and 0.1.",
                ],
            ),
            (
                "nuclei_gene_mean_expression.png",
                [
                    "Nuclei mean expression. The distribution is shifted left — nuclei have lower",
                    "per-gene means due to lower total UMI counts.",
                ],
            ),
        ],
        [
            "Conclusions:",
            "• Cell: mean>0.01 keeps 16,710 genes (62%); mean>0.1 keeps 11,815 (44%).",
            "• Nuclei: mean>0.01 keeps 14,839 (57%); mean>0.1 keeps only 8,757 (34%).",
            "  Nuclei have dramatically fewer genes at mean>1.0: only 1,255 (5%) vs 5,047 (19%) for cells.",
            "• Mean expression is correlated with detection rate — combining both filters is slightly",
            "  more selective than either alone (e.g., det>1% AND mean>0.01 → 16,228 cell / 14,195 nuclei).",
        ],
    ),
    # ── Section 3: Joint and softmax ──
    (
        "5. Joint: Detection Rate vs Mean Expression",
        [
            (
                "cell_joint_detection_expression.png",
                [
                    "Hexbin density plot: each hexagon shows how many genes fall at that",
                    "(log10 mean expression, detection rate) coordinate. Color = gene count (log scale).",
                    "Dashed lines show candidate cutoffs on each axis.",
                ],
            ),
            (
                "nuclei_joint_detection_expression.png",
                [
                    "Same for nuclei. The dense cluster at bottom-left represents the noise genes",
                    "that would be removed by filtering.",
                ],
            ),
        ],
        [
            "Conclusions:",
            "• Detection rate and mean expression are strongly correlated (as expected),",
            "  but the relationship is not perfectly linear — some genes have moderate detection",
            "  rate but low mean expression (sparsely expressed at low levels).",
            "• The bottom-left quadrant (det<1%, mean<0.01) contains ~10,000 noise genes in both modalities.",
            "  Removing these is the conservative filtering recommendation.",
            "• Genes in the upper-right (det>5%, mean>0.1) are the well-detected, well-expressed core.",
        ],
    ),
    (
        "6. Softmax Proportion: Expression Mass Concentration",
        [
            (
                "cell_softmax_proportion.png",
                [
                    "Left: cumulative expression mass — what fraction of total expression is held",
                    "by the top N genes (ranked by mean expression). Log x-axis.",
                    "Right: per-gene proportion of total expression (log-log scale).",
                ],
            ),
            (
                "nuclei_softmax_proportion.png",
                [
                    "Same for nuclei. Note the steeper initial rise — top 10 nuclei genes hold 19%",
                    "of expression (vs 7% for cells), meaning nuclei are more concentrated.",
                ],
            ),
        ],
        [
            "Conclusions:",
            "• Expression is extremely concentrated: top 1000 genes hold 53% (cell) / 63% (nuclei) of mass.",
            "• The bottom ~15,000 genes collectively hold <5% of expression — they are nearly invisible",
            "  to the softmax, but they still compete for probability mass and waste model capacity.",
            "• This is the core of the softmax bottleneck: a few dominant genes (Malat1, ribosomal proteins)",
            "  eat the softmax, suppressing thousands of biologically important low-expression genes.",
            "• Nuclei are worse: Malat1 alone holds ~13% of nuclei expression. Gene filtering reduces",
            "  the softmax dimension but does not change this concentration — architectural fixes needed.",
        ],
    ),
]

SUMMARY_LINES = [
    "SUMMARY & RECOMMENDATIONS",
    "",
    "Dataset overview:",
    "  • Cell data:   193,824 cells × 26,852 genes, median lib size 17,103, median 5,287 genes/cell",
    "  • Nuclei data: 116,971 cells × 25,871 genes, median lib size  4,150, median 2,048 genes/cell",
    "",
    "Current filtering (min_genes=200, min_cells=3) removes 0 cells from either dataset.",
    "It is effectively no filtering.",
    "",
    "Recommended filtering configurations:",
    "",
    "  Config A (recommended): det>1%, mean>0.01, lib>1000, genes>1000",
    "    → Cell: 16,228 genes, 193,824 cells (100%)",
    "    → Nuclei: 14,195 genes, 116,194 cells (99.3%)",
    "    → Softmax reduction: ~1.8x. Conservative. Retains essentially all cells.",
    "",
    "  Config B (moderate):    det>5%, mean>0.01, lib>1000, genes>1000",
    "    → Cell: 12,659 genes, 193,824 cells (100%)",
    "    → Nuclei:  9,938 genes, 116,194 cells (99.3%)",
    "    → Softmax reduction: ~2.3x. Removes more noise genes.",
    "",
    "  Config C (aggressive):  det>5%, mean>0.1, lib>2000, genes>1500",
    "    → Cell: 11,779 genes, 193,174 cells (99.7%)",
    "    → Nuclei:  8,743 genes,  93,171 cells (79.7%)",
    "    → Softmax reduction: ~2.7x. Loses 20% of nuclei — use with caution.",
    "",
    "Key insight: Gene filtering is the main lever for reducing softmax bottleneck.",
    "Cell filtering has minimal impact (data is already well-QC'd). However, gene",
    "filtering alone cannot fix the fundamental softmax zero-sum issue — the top",
    "genes still dominate even at 14K genes. Architectural changes (softplus decoder,",
    "gene-specific bias) are needed as a complementary fix.",
]


def add_text_page(pdf, lines, fontsize=10, title=None, title_fontsize=13):
    """Add a text-only page to the PDF."""
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.08, 0.05, 0.84, 0.90])
    ax.axis("off")

    y = 0.97
    if title:
        ax.text(0.0, y, title, fontsize=title_fontsize, fontweight="bold",
                va="top", ha="left", family="sans-serif")
        y -= 0.04

    for line in lines:
        if line == "":
            y -= 0.015
            continue
        # Bold lines that look like headers
        weight = "bold" if line.isupper() or line.endswith(":") and not line.startswith(" ") else "normal"
        fs = fontsize + 1 if weight == "bold" else fontsize
        ax.text(0.0, y, line, fontsize=fs, fontweight=weight,
                va="top", ha="left", family="monospace", wrap=True,
                transform=ax.transAxes)
        y -= 0.025
    pdf.savefig(fig)
    plt.close(fig)


def add_section(pdf, title, plots_with_captions, commentary):
    """Add a section: title page with commentary, then plot pages."""
    for img_file, caption_lines in plots_with_captions:
        img_path = os.path.join(PLOT_DIR, img_file)
        if not os.path.exists(img_path):
            print(f"  WARNING: {img_path} not found, skipping")
            continue

        fig = plt.figure(figsize=(11, 8.5))  # landscape for wide plots

        # Title at top
        fig.text(0.5, 0.96, title, fontsize=12, fontweight="bold",
                 ha="center", va="top", family="sans-serif")

        # Image
        img = mpimg.imread(img_path)
        ax_img = fig.add_axes([0.03, 0.18, 0.94, 0.74])
        ax_img.imshow(img)
        ax_img.axis("off")

        # Caption below image
        caption = "\n".join(caption_lines)
        fig.text(0.5, 0.13, caption, fontsize=9, ha="center", va="top",
                 family="sans-serif", style="italic",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                           edgecolor="gray", alpha=0.8))

        pdf.savefig(fig)
        plt.close(fig)

    # Commentary page
    add_text_page(pdf, commentary, fontsize=10, title=title)


def main():
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

    with PdfPages(OUTPUT) as pdf:
        # ── Title page ──
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.axis("off")
        ax.text(0.5, 0.70, "Filtering Cutoff Analysis", fontsize=24,
                fontweight="bold", ha="center", va="center", family="sans-serif")
        ax.text(0.5, 0.62, "imputeVAE — Cell vs Nucleus Imputation", fontsize=14,
                ha="center", va="center", family="sans-serif", color="gray")
        ax.text(0.5, 0.55, f"Generated: {date.today().isoformat()}", fontsize=11,
                ha="center", va="center", family="sans-serif", color="gray")

        ax.text(0.5, 0.40, "Contents", fontsize=14, fontweight="bold",
                ha="center", va="center", family="sans-serif")
        toc = [
            "1. Cell-Level QC: Genes Expressed per Cell",
            "2. Cell-Level QC: Library Size",
            "3. Gene-Level Filtering: Detection Rate",
            "4. Gene-Level Filtering: Mean Expression",
            "5. Joint: Detection Rate vs Mean Expression",
            "6. Softmax Proportion: Expression Mass Concentration",
            "7. Summary & Recommendations",
        ]
        for i, item in enumerate(toc):
            ax.text(0.5, 0.33 - i * 0.035, item, fontsize=10,
                    ha="center", va="center", family="sans-serif")

        pdf.savefig(fig)
        plt.close(fig)

        # ── Content sections ──
        for title, plots, commentary in SECTIONS:
            print(f"Adding section: {title}")
            add_section(pdf, title, plots, commentary)

        # ── Summary page ──
        print("Adding summary page")
        add_text_page(pdf, SUMMARY_LINES, fontsize=9,
                      title="7. Summary & Recommendations", title_fontsize=14)

    print(f"\nReport saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
