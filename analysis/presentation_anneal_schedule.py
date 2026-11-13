"""Slide figure: cascading loss annealing schedule for stage 2 training.

Shows how loss term weights ramp up sequentially over epochs.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "notes", "qualify-exam", "pre-qual", "figures")


def anneal_weight(epoch, start_epoch, anneal_epochs, final_value):
    if epoch < start_epoch:
        return 0.0
    if epoch < start_epoch + anneal_epochs:
        progress = (epoch - start_epoch) / max(1, anneal_epochs)
        return final_value * progress
    return final_value


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    epochs = np.arange(0, 601)

    # Stage 1: already trained (epochs shown as reference)
    # Stage 2 schedule (from run_s2_cond_normloss.sh):
    #   cell_recons (1.0)     — always on
    #   cell_jsd (0.1)        — always on
    #   nuclei_recons (1.0)   — anneals 0→1.0 over epochs 0-100
    #   nuclei_jsd (0.1)      — delayed to epoch 100, anneals over 100-200
    #   alignment (1.0)       — delayed to epoch 200, anneals over 200-300

    cell_recons = np.ones_like(epochs, dtype=float)
    cell_jsd = np.full_like(epochs, 0.1, dtype=float)
    nuclei_recons = np.array([anneal_weight(e, 0, 100, 1.0) for e in epochs])
    nuclei_jsd = np.array([anneal_weight(e, 100, 100, 0.1) for e in epochs])
    alignment = np.array([anneal_weight(e, 200, 100, 1.0) for e in epochs])

    fig, ax = plt.subplots(figsize=(10, 4.5))

    # Plot each loss weight
    ax.plot(epochs, cell_recons, color="#1f77b4", linewidth=2.5, label="Cell reconstruction (1.0)")
    ax.plot(epochs, nuclei_recons, color="#ff7f0e", linewidth=2.5, label="Nuclei reconstruction (0→1.0)")
    ax.plot(epochs, cell_jsd, color="#1f77b4", linewidth=2.5, linestyle="--", alpha=0.6, label="Cell JSD (0.1)")
    ax.plot(epochs, nuclei_jsd, color="#ff7f0e", linewidth=2.5, linestyle="--", alpha=0.6, label="Nuclei JSD (0→0.1)")
    ax.plot(epochs, alignment, color="#2ca02c", linewidth=2.5, label="Alignment (0→1.0)")

    # Phase labels
    ax.axvspan(0, 100, alpha=0.06, color="orange")
    ax.axvspan(100, 200, alpha=0.06, color="blue")
    ax.axvspan(200, 300, alpha=0.06, color="green")

    ax.text(50, 1.08, "Phase 1\nNuclei recons\nramps in", ha="center", va="bottom", fontsize=9, color="#555")
    ax.text(150, 1.08, "Phase 2\nNuclei JSD\nramps in", ha="center", va="bottom", fontsize=9, color="#555")
    ax.text(250, 1.08, "Phase 3\nAlignment\nramps in", ha="center", va="bottom", fontsize=9, color="#555")
    ax.text(450, 1.08, "Phase 4\nAll losses active\n(fine-tuning)", ha="center", va="bottom", fontsize=9, color="#555")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss weight", fontsize=12)
    ax.set_title("Cascading loss annealing — Stage 2 training", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 600)
    ax.set_ylim(-0.02, 1.35)
    ax.legend(loc="right", fontsize=9, framealpha=0.9)
    ax.tick_params(labelsize=10)

    # Vertical dashed lines at phase boundaries
    for x in [100, 200, 300]:
        ax.axvline(x, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "slide_cascading_anneal.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
