"""Per-cell-type DE overlap figure from pre-computed eval results.

Uses data from notes/eval_results/*/ct_level/summary.json
"""

import os
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "notes", "qualify-exam", "pre-qual", "figures")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load per-CT DE from eval results
    with open(os.path.join(PROJECT_ROOT, "notes/eval_results/pct5v2_cond_xr/ct_level/summary.json")) as f:
        xr = json.load(f)
    with open(os.path.join(PROJECT_ROOT, "notes/eval_results/pct5v2_cond/ct_level/summary.json")) as f:
        cond = json.load(f)

    de_xr = xr["de_imputed_per_ct"]
    de_cond = cond["de_imputed_per_ct"]
    raw_nuc_overall = 0.45  # overall mean baseline

    # All cell types, sorted by cond+xr DE overlap
    all_cts = sorted(de_xr.keys(), key=lambda ct: de_xr[ct])

    # Separate neuronal vs non-neuronal for coloring context
    non_neuronal = {"Astro", "Oligo", "OPC", "Endo", "Macrophage"}

    fig, ax = plt.subplots(figsize=(10, 8))

    y = np.arange(len(all_cts))
    height = 0.35

    vals_cond = [de_cond.get(ct, 0) * 100 for ct in all_cts]
    vals_xr = [de_xr.get(ct, 0) * 100 for ct in all_cts]

    bars_cond = ax.barh(y + height/2, vals_cond, height,
                        color='#7bafd4', label='Cond (no xr)', edgecolor='white')
    bars_xr = ax.barh(y - height/2, vals_xr, height,
                      color='#e8912d', label='Cond + xr', edgecolor='white')

    # Raw nuclei baseline line
    ax.axvline(raw_nuc_overall * 100, color='#999999', linestyle='--', linewidth=1.2)
    ax.text(raw_nuc_overall * 100 + 1, len(all_cts) - 0.5, 'raw nuclei\nmean (45%)',
            fontsize=9, color='#999999', va='top')

    # Cell type labels — gray for non-neuronal
    ct_labels = []
    for ct in all_cts:
        if ct in non_neuronal:
            ct_labels.append(f"{ct} *")
        else:
            ct_labels.append(ct)

    ax.set_yticks(y)
    ax.set_yticklabels(ct_labels, fontsize=10)
    ax.set_xlabel('DE overlap with real cells (%, top 100 genes)', fontsize=12)
    ax.set_title('Per-cell-type DE gene recovery (PerCT5, 16K genes)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.set_xlim(0, 100)

    # Annotation
    ax.text(0.98, 0.02, '* non-neuronal (low cell count)',
            transform=ax.transAxes, fontsize=9, ha='right', va='bottom', color='#666666')

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "slide_de_per_ct.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # Print summary
    print("\n--- Per-CT DE overlap ---")
    print(f"{'Cell type':15s}  {'cond':>6s}  {'cond+xr':>8s}  {'Δ':>6s}")
    for ct in reversed(all_cts):
        c = de_cond.get(ct, 0)
        x = de_xr.get(ct, 0)
        print(f"  {ct:15s}  {c:5.1%}  {x:7.1%}  {x-c:+5.1%}")


if __name__ == "__main__":
    main()
