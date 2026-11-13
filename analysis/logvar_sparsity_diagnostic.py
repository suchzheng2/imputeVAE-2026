"""
Diagnostic: do extreme log_var values map to zero vs nonzero expression / sparsity?

For a trained stage 1 model, runs inference on a data sample and checks:
1. Per-cell sparsity (fraction of zeros) vs mean log_var per cell
2. Per-cell sparsity vs mean |mu| per cell
3. Scatter plots + Pearson/Spearman correlations

Usage:
    python analysis/logvar_sparsity_diagnostic.py \
        --run_path run/run_XXXX \
        --data_path "10X_cells_v2_v3_AIBS/" \
        --output_dir notes/experiments/logvar_sparsity
"""

import argparse
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imputeVAE.models import VAE_JSD
from imputeVAE.data import create_dataloaders_by_cell_type


def load_model_and_args(run_path):
    args_path = os.path.join(run_path, "arguments.json")
    with open(args_path) as f:
        args_dict = json.load(f)

    # Convert to namespace
    args = argparse.Namespace(**args_dict)

    model = VAE_JSD(
        input_size=args.num_feature,
        num_internal=args.num_internal,
        num_latent=args.num_latent
    )
    model_path = os.path.join(run_path, "models", "model.pt")
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model, args


def run_diagnostic(run_path, data_path, output_dir, max_cells=5000):
    os.makedirs(output_dir, exist_ok=True)
    model, args = load_model_and_args(run_path)

    # Override data path if provided
    if data_path:
        args.data_path = data_path

    # Load data
    from imputeVAE.data import load_raw_data
    adata = load_raw_data(args.data_path, args)

    # Create dataloaders
    args.subset = True
    args.subset_ratio = 0.2
    train_loaders, _, _ = create_dataloaders_by_cell_type(adata, args)
    del adata

    # Collect mu, log_var, and sparsity for all cells
    all_mu = []
    all_logvar = []
    all_sparsity = []  # fraction of zeros per cell
    all_total_expr = []  # total expression per cell (library size proxy)
    all_celltypes = []
    n_collected = 0

    with torch.no_grad():
        for celltype, loader in train_loaders.items():
            for batch in loader:
                data = batch['data']
                _, _, mu, log_var = model(data)

                # Sparsity: fraction of zeros in the input (log1p space, so 0 = zero count)
                sparsity = (data == 0).float().mean(dim=1).numpy()
                total_expr = data.sum(dim=1).numpy()

                all_mu.append(mu.numpy())
                all_logvar.append(log_var.numpy())
                all_sparsity.append(sparsity)
                all_total_expr.append(total_expr)

                ct_labels = []
                for code in batch['cell_type'].numpy():
                    ct_labels.append(loader.dataset.get_original_value('cell_type', code))
                all_celltypes.extend(ct_labels)

                n_collected += data.size(0)
                if n_collected >= max_cells:
                    break
            if n_collected >= max_cells:
                break

    mu = np.concatenate(all_mu, axis=0)[:max_cells]
    logvar = np.concatenate(all_logvar, axis=0)[:max_cells]
    sparsity = np.concatenate(all_sparsity)[:max_cells]
    total_expr = np.concatenate(all_total_expr)[:max_cells]
    celltypes = all_celltypes[:max_cells]

    # Per-cell summary stats
    mean_logvar = logvar.mean(axis=1)
    std_logvar = logvar.std(axis=1)
    min_logvar = logvar.min(axis=1)
    max_logvar = logvar.max(axis=1)
    mean_abs_mu = np.abs(mu).mean(axis=1)
    mean_mu = mu.mean(axis=1)
    mu_range = mu.max(axis=1) - mu.min(axis=1)

    print(f"Collected {len(sparsity)} cells")
    print(f"Sparsity range: [{sparsity.min():.3f}, {sparsity.max():.3f}]")
    print(f"Mean log_var range: [{mean_logvar.min():.3f}, {mean_logvar.max():.3f}]")
    print(f"Mean |mu| range: [{mean_abs_mu.min():.3f}, {mean_abs_mu.max():.3f}]")

    # Correlations
    results = {}
    pairs = [
        ("sparsity", "mean_logvar", sparsity, mean_logvar),
        ("sparsity", "min_logvar", sparsity, min_logvar),
        ("sparsity", "max_logvar", sparsity, max_logvar),
        ("sparsity", "mean_abs_mu", sparsity, mean_abs_mu),
        ("sparsity", "mu_range", sparsity, mu_range),
        ("total_expr", "mean_logvar", total_expr, mean_logvar),
        ("total_expr", "mean_abs_mu", total_expr, mean_abs_mu),
    ]

    print("\n=== Correlations ===")
    for xname, yname, x, y in pairs:
        r_pearson, p_pearson = stats.pearsonr(x, y)
        r_spearman, p_spearman = stats.spearmanr(x, y)
        print(f"  {xname} vs {yname}: Pearson r={r_pearson:.4f} (p={p_pearson:.2e}), "
              f"Spearman r={r_spearman:.4f} (p={p_spearman:.2e})")
        results[f"{xname}_vs_{yname}"] = {
            "pearson_r": float(r_pearson), "pearson_p": float(p_pearson),
            "spearman_r": float(r_spearman), "spearman_p": float(p_spearman),
        }

    # Save results
    with open(os.path.join(output_dir, "correlations.json"), 'w') as f:
        json.dump(results, f, indent=2)

    # Plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].scatter(sparsity, mean_logvar, s=1, alpha=0.3)
    axes[0, 0].set_xlabel("Sparsity (frac zeros)")
    axes[0, 0].set_ylabel("Mean log_var per cell")
    axes[0, 0].set_title(f"r={results['sparsity_vs_mean_logvar']['pearson_r']:.3f}")

    axes[0, 1].scatter(sparsity, min_logvar, s=1, alpha=0.3)
    axes[0, 1].set_xlabel("Sparsity (frac zeros)")
    axes[0, 1].set_ylabel("Min log_var per cell")
    axes[0, 1].set_title(f"r={results['sparsity_vs_min_logvar']['pearson_r']:.3f}")

    axes[0, 2].scatter(sparsity, mean_abs_mu, s=1, alpha=0.3)
    axes[0, 2].set_xlabel("Sparsity (frac zeros)")
    axes[0, 2].set_ylabel("Mean |mu| per cell")
    axes[0, 2].set_title(f"r={results['sparsity_vs_mean_abs_mu']['pearson_r']:.3f}")

    axes[1, 0].scatter(total_expr, mean_logvar, s=1, alpha=0.3)
    axes[1, 0].set_xlabel("Total expression (lib size proxy)")
    axes[1, 0].set_ylabel("Mean log_var per cell")
    axes[1, 0].set_title(f"r={results['total_expr_vs_mean_logvar']['pearson_r']:.3f}")

    axes[1, 1].scatter(total_expr, mean_abs_mu, s=1, alpha=0.3)
    axes[1, 1].set_xlabel("Total expression")
    axes[1, 1].set_ylabel("Mean |mu| per cell")
    axes[1, 1].set_title(f"r={results['total_expr_vs_mean_abs_mu']['pearson_r']:.3f}")

    # Histogram of log_var values (all latent dims, all cells)
    axes[1, 2].hist(logvar.flatten(), bins=100, density=True, alpha=0.7)
    axes[1, 2].set_xlabel("log_var value")
    axes[1, 2].set_ylabel("Density")
    axes[1, 2].set_title(f"log_var distribution (all dims)\nmean={logvar.mean():.2f}, std={logvar.std():.2f}")

    plt.suptitle(f"Log-var / Sparsity Diagnostic\n{run_path}", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "logvar_sparsity.png"), dpi=150)
    plt.close()
    print(f"\nPlots saved to {output_dir}/logvar_sparsity.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_path", required=True, help="Path to trained run directory")
    parser.add_argument("--data_path", default=None, help="Override data path")
    parser.add_argument("--output_dir", default="notes/experiments/logvar_sparsity")
    parser.add_argument("--max_cells", type=int, default=5000)
    args = parser.parse_args()
    run_diagnostic(args.run_path, args.data_path, args.output_dir, args.max_cells)
