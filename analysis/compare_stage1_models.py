"""Compare stage1 models: old NB (24.7K) vs new NB (10K) vs MSE (10K).
Computes per-gene Pearson R, R², mean Pearson, and Wasserstein distance
between raw input and decoded output on validation data.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from scipy.stats import pearsonr, wasserstein_distance
from collections import defaultdict

def load_model_and_args(run_path):
    args_path = os.path.join(run_path, "arguments.json")
    with open(args_path) as f:
        args_dict = json.load(f)

    from types import SimpleNamespace
    args = SimpleNamespace(**args_dict)

    model_path = os.path.join(run_path, "models", "model.pt")

    if args.model == "vae_js":
        from imputeVAE.models import VAE_JSD
        model = VAE_JSD(args.num_feature, args.num_internal, args.num_latent)
    elif args.model == "vae_nb":
        from imputeVAE.models import VAE_NB
        model = VAE_NB(args.num_feature, args.num_internal, args.num_latent)

    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    return model, args

def compute_metrics(model, data_loader, args, device='cpu'):
    """Compute reconstruction metrics on a data loader."""
    model = model.to(device)

    all_raw = []
    all_decoded = []

    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, dict):
                data = batch['data'].to(device)
            else:
                data = batch.to(device)
            encoded, decoded, mu, log_var = model(data)

            # For NB model, decoded is log-rate; apply softmax to get proportions
            if args.model == "vae_nb":
                decoded_out = torch.softmax(decoded, dim=-1)
            else:
                decoded_out = decoded

            all_raw.append(data.cpu().numpy())
            all_decoded.append(decoded_out.cpu().numpy())

    raw = np.concatenate(all_raw, axis=0)
    dec = np.concatenate(all_decoded, axis=0)

    n_genes = raw.shape[1]

    # Per-gene Pearson correlation
    gene_corrs = []
    gene_r2 = []
    for g in range(n_genes):
        r_col = raw[:, g]
        d_col = dec[:, g]
        if np.std(r_col) > 0 and np.std(d_col) > 0:
            r, _ = pearsonr(r_col, d_col)
            gene_corrs.append(r)
            gene_r2.append(r**2)
        else:
            gene_corrs.append(np.nan)
            gene_r2.append(np.nan)

    # Cell-level Pearson (average across cells)
    cell_corrs = []
    for i in range(min(raw.shape[0], 5000)):  # cap at 5000 cells
        r_row = raw[i]
        d_row = dec[i]
        if np.std(r_row) > 0 and np.std(d_row) > 0:
            r, _ = pearsonr(r_row, d_row)
            cell_corrs.append(r)

    # Wasserstein distance per gene (sample 1000 genes)
    gene_idx = np.random.choice(n_genes, min(1000, n_genes), replace=False)
    wd_list = []
    for g in gene_idx:
        wd = wasserstein_distance(raw[:, g], dec[:, g])
        wd_list.append(wd)

    # Global correlation (flatten)
    flat_r, _ = pearsonr(raw.flatten()[:500000], dec.flatten()[:500000])

    results = {
        "n_cells": raw.shape[0],
        "n_genes": n_genes,
        "mean_gene_pearson": float(np.nanmean(gene_corrs)),
        "median_gene_pearson": float(np.nanmedian(gene_corrs)),
        "mean_gene_r2": float(np.nanmean(gene_r2)),
        "median_gene_r2": float(np.nanmedian(gene_r2)),
        "mean_cell_pearson": float(np.mean(cell_corrs)),
        "median_cell_pearson": float(np.median(cell_corrs)),
        "mean_wasserstein": float(np.mean(wd_list)),
        "median_wasserstein": float(np.median(wd_list)),
        "global_pearson": float(flat_r),
        "raw_mean": float(raw.mean()),
        "raw_std": float(raw.std()),
        "decoded_mean": float(dec.mean()),
        "decoded_std": float(dec.std()),
    }
    return results

def main():
    runs = {
        "old_NB_24K": "run/run_2026-03-12-17-29-06_928562",
        "new_NB_10K": "run/run_2026-04-01-09-29-27_2189144",
        "MSE_10K":    "run/run_2026-04-01-09-29-26_3250320",
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    all_results = {}

    for label, run_path in runs.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {label} ({run_path})")
        print(f"{'='*60}")

        model, args = load_model_and_args(run_path)

        # Load validation data
        cache_dir = os.path.join("cache", "dataloaders", args.data_path.rstrip('/'))

        # Find valid loader
        valid_loader_files = [f for f in os.listdir(cache_dir) if 'valid' in f.lower()] if os.path.exists(cache_dir) else []

        if not valid_loader_files:
            print(f"  No cached valid loader found in {cache_dir}, loading data fresh...")
            from imputeVAE.data import load_raw_data, filter_by_protocol, create_dataloaders_by_cell_type
            adata = load_raw_data(args.data_path, args)
            adata = filter_by_protocol(adata, args)
            train_loaders, valid_loaders, test_loaders = create_dataloaders_by_cell_type(adata, args)

            # Combine all cell type valid loaders
            all_valid_data = []
            for ct, loader in valid_loaders.items():
                for batch in loader:
                    if isinstance(batch, dict):
                        all_valid_data.append(batch['data'])
                    else:
                        all_valid_data.append(batch)
            combined = torch.cat(all_valid_data, dim=0)
            valid_loader = torch.utils.data.DataLoader(combined, batch_size=256, shuffle=False)
        else:
            # Try loading cached
            for f in valid_loader_files:
                try:
                    valid_loaders = torch.load(os.path.join(cache_dir, f), map_location='cpu', weights_only=False)
                    if isinstance(valid_loaders, dict):
                        all_valid_data = []
                        for ct, loader in valid_loaders.items():
                            for batch in loader:
                                if isinstance(batch, dict):
                                    all_valid_data.append(batch['data'])
                                else:
                                    all_valid_data.append(batch)
                        combined = torch.cat(all_valid_data, dim=0)
                        valid_loader = torch.utils.data.DataLoader(combined, batch_size=256, shuffle=False)
                    else:
                        valid_loader = valid_loaders
                    print(f"  Loaded cached valid loader: {f}")
                    break
                except Exception as e:
                    print(f"  Failed to load {f}: {e}")
                    continue

        results = compute_metrics(model, valid_loader, args, device)
        all_results[label] = results

        print(f"\n  Results for {label}:")
        for k, v in results.items():
            print(f"    {k}: {v:.6f}" if isinstance(v, float) else f"    {k}: {v}")

    # Summary comparison table
    print(f"\n\n{'='*80}")
    print("COMPARISON TABLE")
    print(f"{'='*80}")

    metrics_to_show = [
        "n_genes", "n_cells",
        "mean_gene_pearson", "median_gene_pearson",
        "mean_gene_r2", "median_gene_r2",
        "mean_cell_pearson", "median_cell_pearson",
        "mean_wasserstein", "median_wasserstein",
        "global_pearson",
        "raw_mean", "decoded_mean",
        "raw_std", "decoded_std",
    ]

    header = f"{'Metric':<25}" + "".join(f"{label:>18}" for label in all_results.keys())
    print(header)
    print("-" * len(header))

    for metric in metrics_to_show:
        row = f"{metric:<25}"
        for label in all_results.keys():
            val = all_results[label].get(metric, "N/A")
            if isinstance(val, float):
                row += f"{val:>18.6f}"
            else:
                row += f"{val:>18}"

        print(row)

if __name__ == "__main__":
    main()
