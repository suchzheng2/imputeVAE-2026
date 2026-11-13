"""Figure: in-situ nuclear fraction (Xenium, per-molecule `overlaps_nucleus`) vs our predicted cell-minus-nucleus delta
(type-balanced ground-truth delta on the test partition, `gt_delta_per_gene_test_normalised.csv`), 2026-11-13.
Panel A: per-gene scatter (2,477 shared genes with >= 200 molecules), single-hue density; Panel B: nuclear fraction by
delta class (nuclear-enriched < -0.3, neutral |d| <= 0.1, cytoplasm-enriched > 0.3) as box plots.
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

OUT = "notes/experiments/spatial_merfish_application/scoping"
INK, MUTED, AXIS, BLUE = "#0b0b0b", "#898781", "#c3c2b7", "#2a78d6"


def main():
    m = pd.read_csv(f"{OUT}/gene_nuclear_fraction_vs_gt_delta.csv", index_col=0)
    rho = spearmanr(m.nuclear_fraction, m.delta_norm).statistic
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6), facecolor="#fcfcfb", gridspec_kw={"width_ratios": [1.3, 1]})
    a.scatter(m.delta_norm, m.nuclear_fraction, s=7, c=BLUE, alpha=0.35, lw=0, rasterized=True)
    for g in ["Rab3a", "Aldoa", "Tpi1", "Cbx1", "Rsrp1", "Xist", "Ywhaz", "Vamp2"]:
        if g in m.index:
            a.annotate(g, (m.delta_norm[g], m.nuclear_fraction[g]), fontsize=7.5, color=INK, xytext=(3, 3), textcoords="offset points")
    a.axvline(0, color=AXIS, lw=1); a.axhline(m.nuclear_fraction.median(), color=AXIS, lw=1, ls=":")
    a.set_xlabel("predicted delta, cell minus nucleus (log1p CP10k, type-balanced)", color=MUTED, fontsize=9)
    a.set_ylabel("in-situ nuclear fraction of molecules", color=MUTED, fontsize=9)
    a.set_title(f"A  {len(m):,} shared genes, Spearman rho = {rho:.2f}", loc="left", fontsize=11, color=INK)
    cls = pd.cut(m.delta_norm, [-9, -0.3, -0.1, 0.1, 0.3, 9], labels=["< -0.3\nnuclear", "-0.3..-0.1", "|d| <= 0.1\nneutral", "0.1..0.3", "> 0.3\ncytoplasm"])
    groups = [m.nuclear_fraction[cls == c].values for c in cls.cat.categories]
    bp = b.boxplot(groups, widths=0.55, patch_artist=True, showfliers=False, medianprops={"color": INK, "lw": 1.5},
                   boxprops={"facecolor": "#cde2fb", "edgecolor": BLUE, "lw": 1}, whiskerprops={"color": BLUE}, capprops={"color": BLUE})
    b.set_xticks(range(1, 6)); b.set_xticklabels([f"{c}\nn={len(g)}" for c, g in zip(cls.cat.categories, groups)], fontsize=7.5, color=MUTED)
    b.set_ylabel("in-situ nuclear fraction", color=MUTED, fontsize=9)
    b.set_title("B  by predicted delta class", loc="left", fontsize=11, color=INK)
    for ax in (a, b):
        ax.set_facecolor("#fcfcfb"); ax.tick_params(colors=MUTED, labelsize=8)
        for s in ["top", "right"]: ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]: ax.spines[s].set_color(AXIS)
    fig.suptitle("Xenium per-molecule nuclear localisation agrees with the predicted cell-vs-nucleus compartment shift",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(); os.makedirs(f"{OUT}/figures", exist_ok=True)
    fig.savefig(f"{OUT}/figures/fig_xenium_nuclear_fraction_vs_delta.png", dpi=150); print("wrote figure; rho", round(rho, 3))


if __name__ == "__main__":
    main()
