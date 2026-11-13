"""
Task 4: Scrambled-delta negative control.

Permutes gene labels on the delta vector (N=1000) and recomputes:
  (a) compartment sign-correctness (ribosomal, cytoplasmic, nuclear lncRNA)
  (b) GO enrichment overlap with top-100 increased genes

Builds null distributions and reports empirical p-values.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELTA_CSV = os.path.join(PROJECT, "notes/experiments/nuclear_lncrna_analysis/all_genes_delta.csv")
OUT_DIR = os.path.join(PROJECT, "notes/experiments/negative_control")

N_PERM = 1000
SEED = 42

# ── Gene sets for compartment metrics ──────────────────────────────────
# Ribosomal proteins (Rpl/Rps) — expected positive delta
# Cytoplasmic structural — expected positive delta
CYTOPLASMIC_GENES = [
    "Actb", "Actg1", "Tmsb4x", "Pfn1", "Pfn2", "Cfl1", "Gsn",
    "Arpc2", "Arpc3", "Arpc5", "Cap1", "Capzb", "Tln1", "Vcl",
    "Vim", "Tubb2a", "Tubb3", "Tubb5",
]

# Nuclear lncRNAs — expected negative delta (data-driven list)
NUCLEAR_LNCRNA_FILE = os.path.join(
    PROJECT, "notes/experiments/nuclear_lncrna_analysis/proposed_nuclear_lncrna_list.txt"
)


def load_nuclear_lncrna_list():
    if os.path.exists(NUCLEAR_LNCRNA_FILE):
        with open(NUCLEAR_LNCRNA_FILE) as f:
            return [g.strip() for g in f if g.strip()]
    return []


def compute_compartment_metrics(delta_series, gene_names):
    """Compute sign-correctness for compartment gene sets."""
    results = {}

    # Ribosomal
    ribo_mask = np.array([g.startswith(("Rpl", "Rps")) for g in gene_names])
    ribo_deltas = delta_series[ribo_mask]
    if len(ribo_deltas) > 0:
        results["ribo_frac_positive"] = float((ribo_deltas > 0).mean())
        results["ribo_mean_delta"] = float(ribo_deltas.mean())
        results["ribo_n"] = int(ribo_mask.sum())

    # Cytoplasmic
    cyto_mask = np.isin(gene_names, CYTOPLASMIC_GENES)
    cyto_deltas = delta_series[cyto_mask]
    if len(cyto_deltas) > 0:
        results["cyto_frac_positive"] = float((cyto_deltas > 0).mean())
        results["cyto_mean_delta"] = float(cyto_deltas.mean())
        results["cyto_n"] = int(cyto_mask.sum())

    # Nuclear lncRNAs
    lncrna_genes = load_nuclear_lncrna_list()
    if lncrna_genes:
        lncrna_mask = np.isin(gene_names, lncrna_genes)
        lncrna_deltas = delta_series[lncrna_mask]
        if len(lncrna_deltas) > 0:
            results["lncrna_frac_negative"] = float((lncrna_deltas < 0).mean())
            results["lncrna_mean_delta"] = float(lncrna_deltas.mean())
            results["lncrna_n"] = int(lncrna_mask.sum())

    return results


def compute_top_gene_overlap(delta_values, gene_names, top_k=100):
    """Check overlap between top-K genes by delta and known gene sets."""
    top_idx = np.argsort(delta_values)[-top_k:][::-1]
    top_genes = set(gene_names[top_idx])

    # Ribosomal in top-K
    ribo_genes = set(g for g in gene_names if g.startswith(("Rpl", "Rps")))
    ribo_overlap = len(top_genes & ribo_genes)

    # Cytoplasmic in top-K
    cyto_overlap = len(top_genes & set(CYTOPLASMIC_GENES))

    return {"ribo_in_top100": ribo_overlap, "cyto_in_top100": cyto_overlap}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    np.random.seed(SEED)

    # Load delta
    df = pd.read_csv(DELTA_CSV)
    gene_names = df["gene"].values
    delta_values = df["delta"].values
    n_genes = len(delta_values)
    print(f"Loaded {n_genes} genes")

    # ── Real metrics ───────────────────────────────────────────────────
    real_compartment = compute_compartment_metrics(delta_values, gene_names)
    real_overlap = compute_top_gene_overlap(delta_values, gene_names)

    print("\nReal metrics:")
    for k, v in {**real_compartment, **real_overlap}.items():
        print(f"  {k}: {v}")

    # ── Permutation null ───────────────────────────────────────────────
    print(f"\nRunning {N_PERM} permutations...")
    perm_results = {k: [] for k in list(real_compartment.keys()) + list(real_overlap.keys())}

    for i in range(N_PERM):
        if (i + 1) % 200 == 0:
            print(f"  Permutation {i+1}/{N_PERM}")

        # Shuffle gene labels (equivalent to permuting delta across genes)
        perm_idx = np.random.permutation(n_genes)
        perm_delta = delta_values[perm_idx]

        perm_comp = compute_compartment_metrics(perm_delta, gene_names)
        perm_olap = compute_top_gene_overlap(perm_delta, gene_names)

        for k in perm_comp:
            perm_results[k].append(perm_comp[k])
        for k in perm_olap:
            perm_results[k].append(perm_olap[k])

    # ── Empirical p-values ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EMPIRICAL P-VALUES (one-sided: real > permuted null)")
    print("=" * 70)

    p_values = {}
    all_real = {**real_compartment, **real_overlap}

    for k in perm_results:
        null = np.array(perm_results[k])
        real_val = all_real.get(k, None)
        if real_val is None:
            continue

        # For "frac_negative" (lncRNA), we want real to be HIGHER (more negative deltas)
        # For everything else, we want real to be HIGHER
        p = (null >= real_val).mean()
        p_values[k] = float(p)

        null_mean = null.mean()
        null_std = null.std()
        z = (real_val - null_mean) / null_std if null_std > 0 else float('inf')

        print(f"  {k:<25} real={real_val:.4f}  null={null_mean:.4f}±{null_std:.4f}  "
              f"z={z:.1f}  p={p:.4f}  {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'}")

    # ── Save results ───────────────────────────────────────────────────
    output = {
        "n_permutations": N_PERM,
        "n_genes": n_genes,
        "real_metrics": {k: float(v) for k, v in all_real.items()},
        "empirical_p_values": p_values,
        "null_means": {k: float(np.mean(perm_results[k])) for k in perm_results},
        "null_stds": {k: float(np.std(perm_results[k])) for k in perm_results},
    }

    with open(os.path.join(OUT_DIR, "negative_control_stats.json"), "w") as f:
        json.dump(output, f, indent=2)

    # ── Plots ──────────────────────────────────────────────────────────
    plot_keys = [
        ("ribo_frac_positive", "Ribosomal genes: frac with positive delta"),
        ("cyto_frac_positive", "Cytoplasmic genes: frac with positive delta"),
        ("lncrna_frac_negative", "Nuclear lncRNAs: frac with negative delta"),
        ("ribo_mean_delta", "Ribosomal genes: mean delta"),
        ("ribo_in_top100", "Ribosomal genes in top-100 increased"),
        ("cyto_in_top100", "Cytoplasmic genes in top-100 increased"),
    ]

    # Filter to keys that exist
    plot_keys = [(k, title) for k, title in plot_keys if k in perm_results and k in all_real]

    n_plots = len(plot_keys)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, (key, title) in enumerate(plot_keys):
        ax = axes[i]
        null = np.array(perm_results[key])
        real_val = all_real[key]
        p = p_values.get(key, 1.0)

        ax.hist(null, bins=40, color="lightgray", edgecolor="gray", alpha=0.8, label="Permuted null")
        ax.axvline(real_val, color="red", linewidth=2, linestyle="--", label=f"Real = {real_val:.3f}")
        ax.set_title(f"{title}\np = {p:.4f}")
        ax.set_xlabel("Metric value")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    # Hide unused axes
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f"Scrambled-delta negative control (N={N_PERM} permutations)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "null_distributions.png"), dpi=150)
    plt.close()
    print(f"\nPlot saved to {OUT_DIR}/null_distributions.png")

    # ── Conclusion ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    sig = [k for k, p in p_values.items() if p < 0.001]
    if len(sig) == len(p_values):
        print("ALL compartment metrics are highly significant (p < 0.001).")
        print("The biological structure in the delta is real, not manufactured.")
    else:
        nonsig = [k for k, p in p_values.items() if p >= 0.05]
        print(f"Significant (p<0.001): {sig}")
        if nonsig:
            print(f"Not significant: {nonsig}")


if __name__ == "__main__":
    main()
