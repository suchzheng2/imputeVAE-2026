"""
Cross-tissue imputation test: apply brain-trained cVAE to kidney data.

Loads the best brain-trained cVAE model, feeds kidney nuclei through it
(encode as nuclei → decode as cell), and compares the imputed output
to actual kidney cell expression.

This tests whether the nucleus→cell transformation learned in mouse brain
generalizes to human kidney.

Usage:
    python analysis/cross_tissue_imputation.py \
        --model_run run/run_2026-07-18-19-40-51_570984 \
        --cell_h5ad data/external/lake2023_cells.h5ad \
        --nuc_h5ad data/external/lake2023_nuclei.h5ad \
        --output_dir notes/experiments/cross_tissue_imputation
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse, stats
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from imputeVAE.models_cond import VAE_JSD_Cond


# ---------------------------------------------------------------------------
# Gene mapping
# ---------------------------------------------------------------------------

def mouse_to_human_symbol(mouse_symbol):
    """Convert mouse gene symbol to human equivalent (uppercase)."""
    if not mouse_symbol or not isinstance(mouse_symbol, str):
        return None
    return mouse_symbol.upper()


def human_to_mouse_symbol(human_symbol):
    """Convert human gene symbol to mouse equivalent (title case)."""
    if not human_symbol or not isinstance(human_symbol, str):
        return None
    if human_symbol.startswith("ENSG"):
        return None
    return human_symbol[0].upper() + human_symbol[1:].lower()


def build_gene_mapping(training_genes, kidney_genes):
    """Build mapping between training gene list (mouse) and kidney genes (human).

    Returns:
        mouse2human: dict mapping mouse gene name -> human gene name (for genes found in kidney)
        human2mouse: dict mapping human gene name -> mouse gene name
        mapped_training_indices: list of indices into training_genes that have a kidney match
        mapped_kidney_indices: list of corresponding indices into kidney_genes
    """
    # Build human -> kidney index lookup
    kidney_lookup = {g: i for i, g in enumerate(kidney_genes)}

    mouse2human = {}
    mapped_training_indices = []
    mapped_kidney_indices = []

    for i, mg in enumerate(training_genes):
        hg = mouse_to_human_symbol(mg)
        if hg and hg in kidney_lookup:
            mouse2human[mg] = hg
            mapped_training_indices.append(i)
            mapped_kidney_indices.append(kidney_lookup[hg])

    human2mouse = {v: k for k, v in mouse2human.items()}
    print(f"  Mapped {len(mouse2human)}/{len(training_genes)} training genes to kidney genes")
    return mouse2human, human2mouse, mapped_training_indices, mapped_kidney_indices


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_kidney_h5ad(h5ad_path, max_cells=None):
    """Load kidney h5ad, return raw counts matrix and metadata."""
    print(f"Loading {h5ad_path}...")
    adata = ad.read_h5ad(h5ad_path)
    print(f"  Shape: {adata.shape}")

    if adata.raw is not None:
        X = adata.raw.X
        if "feature_name" in adata.raw.var.columns:
            var_names = list(adata.raw.var["feature_name"].values)
        else:
            var_names = list(adata.raw.var_names)
    else:
        X = adata.X
        if "feature_name" in adata.var.columns:
            var_names = list(adata.var["feature_name"].values)
        else:
            var_names = list(adata.var_names)

    obs = adata.obs.copy()
    del adata

    if max_cells and X.shape[0] > max_cells:
        print(f"  Subsampling to {max_cells} cells...")
        idx = np.random.choice(X.shape[0], max_cells, replace=False)
        idx.sort()
        X = X[idx]
        obs = obs.iloc[idx].copy()

    if sparse.issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)

    print(f"  Final: {X.shape}, raw count range [{X.min():.0f}, {X.max():.0f}]")
    return X, obs, var_names


def prepare_model_input(X_raw, kidney_genes, training_genes, mapped_training_indices,
                        mapped_kidney_indices):
    """Build [N, num_training_genes] matrix from kidney data, log1p normalized.

    For training genes with no kidney match, the column is filled with zeros (log1p(0)=0).
    """
    N = X_raw.shape[0]
    G = len(training_genes)

    # Extract only the mapped kidney columns
    X_subset = X_raw[:, mapped_kidney_indices]  # [N, n_mapped]

    # log1p normalize
    X_subset = np.log1p(X_subset)

    # Build full input matrix (zeros for unmapped genes)
    X_model = np.zeros((N, G), dtype=np.float32)
    for out_idx, train_idx in enumerate(mapped_training_indices):
        X_model[:, train_idx] = X_subset[:, out_idx]

    n_mapped = len(mapped_training_indices)
    n_zero = G - n_mapped
    print(f"  Model input: {X_model.shape}, {n_mapped} mapped genes, {n_zero} zero-filled")
    return X_model


# ---------------------------------------------------------------------------
# Model loading and inference
# ---------------------------------------------------------------------------

def load_model(model_run_path):
    """Load the trained cVAE model."""
    args_path = os.path.join(model_run_path, "arguments.json")
    with open(args_path) as f:
        model_args = json.load(f)

    model = VAE_JSD_Cond(
        input_size=model_args["num_feature"],
        num_internal=model_args["num_internal"],
        num_latent=model_args["num_latent"],
        embed_dim=model_args.get("embed_dim", 8),
        scalar_cond=model_args.get("scalar_cond", False),
        cond_encoder=model_args.get("cond_encoder", False),
    )

    model_path = os.path.join(model_run_path, "models", "model.pt")
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  Model loaded: {model_args['num_feature']} genes, "
          f"latent={model_args['num_latent']}, device={device}")
    print(f"  scalar_cond={model_args.get('scalar_cond')}, "
          f"cond_encoder={model_args.get('cond_encoder')}")
    return model, model_args, device


def run_imputation(model, X_input, device, batch_size=512):
    """Run nucleus→cell cross-modal imputation in batches.

    Encodes with modality_id=1 (nuclei), decodes with modality_id=0 (cell).
    Returns imputed output on log1p scale.
    """
    model.eval()
    N = X_input.shape[0]
    outputs = []

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            x_batch = torch.tensor(X_input[start:end], dtype=torch.float32, device=device)
            # decode_cross: encode as nuclei (1), decode as cell (0)
            imputed = model.decode_cross(x_batch, source_modality_id=1, target_modality_id=0)
            outputs.append(imputed.cpu().numpy())

            if (start // batch_size) % 50 == 0:
                print(f"    Batch {start // batch_size}/{(N + batch_size - 1) // batch_size}")

    imputed_all = np.concatenate(outputs, axis=0)
    print(f"  Imputed output: {imputed_all.shape}, "
          f"range [{imputed_all.min():.3f}, {imputed_all.max():.3f}]")
    return imputed_all


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def evaluate_imputation(imputed, X_cell_log1p, training_genes, mapped_training_indices,
                        obs_nuc, obs_cell, ct_col):
    """Compare imputed cell profiles to actual kidney cells on mapped genes only."""
    results = {}
    mapped_idx = np.array(mapped_training_indices)
    mapped_genes = [training_genes[i] for i in mapped_idx]

    # Extract mapped columns
    imputed_mapped = imputed[:, mapped_idx]  # [N_nuc, n_mapped]
    cell_mapped = X_cell_log1p[:, mapped_idx]  # [N_cell, n_mapped]

    # --- Global gene-level correlation ---
    imputed_gene_mean = imputed_mapped.mean(axis=0)
    cell_gene_mean = cell_mapped.mean(axis=0)

    r_global, p_global = stats.pearsonr(imputed_gene_mean, cell_gene_mean)
    rho_global, _ = stats.spearmanr(imputed_gene_mean, cell_gene_mean)
    results["global_gene_corr_pearson"] = float(r_global)
    results["global_gene_corr_spearman"] = float(rho_global)
    print(f"\n  Global gene correlation (imputed nuc-mean vs actual cell-mean):")
    print(f"    Pearson r = {r_global:.4f}, Spearman rho = {rho_global:.4f}")

    # --- Per cell-type gene correlation ---
    ct_nuc = obs_nuc[ct_col].values.astype(str)
    ct_cell = obs_cell[ct_col].values.astype(str)
    common_cts = sorted(set(ct_nuc) & set(ct_cell))

    ct_corrs = {}
    for ct in common_cts:
        mask_nuc = ct_nuc == ct
        mask_cell = ct_cell == ct
        if mask_nuc.sum() < 10 or mask_cell.sum() < 10:
            continue
        imp_mean = imputed_mapped[mask_nuc].mean(axis=0)
        cell_mean = cell_mapped[mask_cell].mean(axis=0)
        r, _ = stats.pearsonr(imp_mean, cell_mean)
        ct_corrs[ct] = float(r)

    results["per_ct_gene_corr"] = ct_corrs
    if ct_corrs:
        mean_ct_corr = np.mean(list(ct_corrs.values()))
        results["mean_ct_gene_corr"] = float(mean_ct_corr)
        print(f"  Per-CT gene correlation: mean={mean_ct_corr:.4f} across {len(ct_corrs)} CTs")
        # Show best and worst
        sorted_cts = sorted(ct_corrs.items(), key=lambda x: x[1])
        print(f"    Worst 3: {sorted_cts[:3]}")
        print(f"    Best 3:  {sorted_cts[-3:]}")

    # --- Delta comparison: does imputation shift genes in the right direction? ---
    # Ground truth delta = actual_cell_mean - actual_nuc_mean (in log1p)
    nuc_gene_mean = np.zeros(len(mapped_idx), dtype=np.float32)
    for i, ti in enumerate(mapped_idx):
        # We need actual nuclei log1p mean on mapped genes
        # X_cell_log1p columns are aligned to training genes
        pass

    # Actually compute from the raw inputs we already have
    # imputed_gene_mean is the model's prediction of cell expression
    # cell_gene_mean is the ground truth cell expression
    # We also need actual nuclei expression to compute the delta
    # The nuclei input is already in X_input (log1p), so we can use that

    # --- Compartment gene analysis ---
    ribo_mask = np.array([g.startswith(("Rpl", "Rps")) for g in mapped_genes])
    lncrna_names = {"Malat1", "Neat1", "Xist", "Meg3"}
    lncrna_mask = np.array([g in lncrna_names for g in mapped_genes])

    if ribo_mask.any():
        ribo_imp = imputed_gene_mean[ribo_mask]
        ribo_cell = cell_gene_mean[ribo_mask]
        ribo_corr = float(stats.pearsonr(ribo_imp, ribo_cell)[0])
        ribo_bias = float(np.mean(ribo_imp - ribo_cell))
        results["ribosomal"] = {
            "n": int(ribo_mask.sum()),
            "corr": ribo_corr,
            "mean_bias": ribo_bias,
            "imputed_mean": float(ribo_imp.mean()),
            "actual_cell_mean": float(ribo_cell.mean()),
        }
        print(f"  Ribosomal genes ({ribo_mask.sum()}): corr={ribo_corr:.3f}, "
              f"bias={ribo_bias:.3f} (imputed={ribo_imp.mean():.3f}, actual={ribo_cell.mean():.3f})")

    if lncrna_mask.any():
        lncrna_genes = [g for g, m in zip(mapped_genes, lncrna_mask) if m]
        lncrna_imp = imputed_gene_mean[lncrna_mask]
        lncrna_cell = cell_gene_mean[lncrna_mask]
        results["nuclear_lncrna"] = {
            "genes": lncrna_genes,
            "imputed_mean": [float(v) for v in lncrna_imp],
            "actual_cell_mean": [float(v) for v in lncrna_cell],
        }
        for g, imp_v, cell_v in zip(lncrna_genes, lncrna_imp, lncrna_cell):
            print(f"  {g}: imputed={imp_v:.3f}, actual_cell={cell_v:.3f}")

    # --- R² on gene means ---
    ss_res = np.sum((imputed_gene_mean - cell_gene_mean) ** 2)
    ss_tot = np.sum((cell_gene_mean - cell_gene_mean.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    results["r2_gene_means"] = float(r2)
    print(f"  R² (gene means): {r2:.4f}")

    results["n_mapped_genes"] = len(mapped_idx)
    results["n_common_cts"] = len(ct_corrs)

    return results


def evaluate_delta_direction(imputed, X_nuc_model_input, X_cell_log1p,
                             training_genes, mapped_training_indices):
    """Check if imputation shifts genes in the correct direction.

    For each gene: does the model shift nuclei expression toward cell expression?
    ground_truth_delta = cell_mean - nuc_mean
    imputed_delta = imputed_mean - nuc_mean
    We want sign(imputed_delta) == sign(ground_truth_delta) for most genes.
    """
    mapped_idx = np.array(mapped_training_indices)

    nuc_mean = X_nuc_model_input[:, mapped_idx].mean(axis=0)
    imp_mean = imputed[:, mapped_idx].mean(axis=0)
    cell_mean = X_cell_log1p[:, mapped_idx].mean(axis=0)

    gt_delta = cell_mean - nuc_mean
    imp_delta = imp_mean - nuc_mean

    # Sign agreement
    nonzero = np.abs(gt_delta) > 0.01  # only count genes with meaningful delta
    if nonzero.sum() > 0:
        sign_agree = float(np.mean(np.sign(imp_delta[nonzero]) == np.sign(gt_delta[nonzero])))
    else:
        sign_agree = float("nan")

    # Delta correlation
    r_delta, _ = stats.pearsonr(gt_delta, imp_delta)
    rho_delta, _ = stats.spearmanr(gt_delta, imp_delta)

    results = {
        "delta_pearson": float(r_delta),
        "delta_spearman": float(rho_delta),
        "sign_agreement": sign_agree,
        "n_genes_tested": int(nonzero.sum()),
        "mean_gt_delta": float(gt_delta.mean()),
        "mean_imp_delta": float(imp_delta.mean()),
    }

    print(f"\n  Delta direction analysis:")
    print(f"    Delta correlation: Pearson={r_delta:.4f}, Spearman={rho_delta:.4f}")
    print(f"    Sign agreement: {sign_agree:.1%} ({nonzero.sum()} genes with |delta|>0.01)")
    print(f"    Mean ground truth delta: {gt_delta.mean():.4f}")
    print(f"    Mean imputed delta: {imp_delta.mean():.4f}")

    return results


def make_plots(imputed, X_cell_log1p, X_nuc_model_input, training_genes,
               mapped_training_indices, output_dir):
    """Generate diagnostic plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mapped_idx = np.array(mapped_training_indices)
    mapped_genes = [training_genes[i] for i in mapped_idx]

    imp_mean = imputed[:, mapped_idx].mean(axis=0)
    cell_mean = X_cell_log1p[:, mapped_idx].mean(axis=0)
    nuc_mean = X_nuc_model_input[:, mapped_idx].mean(axis=0)

    # 1. Imputed vs actual cell mean
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(cell_mean, imp_mean, alpha=0.15, s=8, c="steelblue")

    ribo_mask = np.array([g.startswith(("Rpl", "Rps")) for g in mapped_genes])
    if ribo_mask.any():
        ax.scatter(cell_mean[ribo_mask], imp_mean[ribo_mask], alpha=0.7, s=20, c="red",
                   label="Ribosomal")

    r = stats.pearsonr(cell_mean, imp_mean)[0]
    ax.plot([0, cell_mean.max()], [0, cell_mean.max()], "k--", alpha=0.3, label="y=x")
    ax.set_xlabel("Actual kidney cell mean (log1p)", fontsize=12)
    ax.set_ylabel("Imputed from kidney nuclei (log1p)", fontsize=12)
    ax.set_title("Cross-tissue imputation: kidney", fontsize=14)
    ax.text(0.05, 0.95, f"Pearson r = {r:.3f}\nn = {len(mapped_idx)} genes",
            transform=ax.transAxes, va="top", fontsize=11,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "imputed_vs_actual_cell.png"), dpi=150)
    plt.close(fig)

    # 2. Delta scatter: ground truth delta vs imputed delta
    gt_delta = cell_mean - nuc_mean
    imp_delta = imp_mean - nuc_mean

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(gt_delta, imp_delta, alpha=0.15, s=8, c="steelblue")
    if ribo_mask.any():
        ax.scatter(gt_delta[ribo_mask], imp_delta[ribo_mask], alpha=0.7, s=20, c="red",
                   label="Ribosomal")

    r_d = stats.pearsonr(gt_delta, imp_delta)[0]
    lim = max(abs(gt_delta).max(), abs(imp_delta).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.3)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel("Ground truth delta (cell - nuc, log1p)", fontsize=12)
    ax.set_ylabel("Imputed delta (imputed - nuc, log1p)", fontsize=12)
    ax.set_title("Cross-tissue: imputed vs ground truth shift", fontsize=14)
    ax.text(0.05, 0.95, f"Pearson r = {r_d:.3f}",
            transform=ax.transAxes, va="top", fontsize=11,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "delta_imputed_vs_groundtruth.png"), dpi=150)
    plt.close(fig)

    # 3. Nucleus input vs imputed (to show the model actually changes things)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(nuc_mean, imp_mean, alpha=0.15, s=8, c="steelblue")
    if ribo_mask.any():
        ax.scatter(nuc_mean[ribo_mask], imp_mean[ribo_mask], alpha=0.7, s=20, c="red",
                   label="Ribosomal")
    ax.plot([0, nuc_mean.max()], [0, nuc_mean.max()], "k--", alpha=0.3, label="y=x")
    ax.set_xlabel("Kidney nuclei input (log1p)", fontsize=12)
    ax.set_ylabel("Imputed cell output (log1p)", fontsize=12)
    ax.set_title("What the model does to kidney nuclei", fontsize=14)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "nuc_input_vs_imputed.png"), dpi=150)
    plt.close(fig)

    print(f"  Saved 3 plots to {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cross-tissue imputation: apply brain cVAE to kidney data")
    parser.add_argument("--model_run", type=str, required=True,
                        help="Path to trained cVAE run directory")
    parser.add_argument("--cell_h5ad", type=str, required=True,
                        help="Path to Lake 2023 kidney cells h5ad")
    parser.add_argument("--nuc_h5ad", type=str, required=True,
                        help="Path to Lake 2023 kidney nuclei h5ad")
    parser.add_argument("--output_dir", type=str,
                        default="notes/experiments/cross_tissue_imputation")
    parser.add_argument("--ct_col", type=str, default=None)
    parser.add_argument("--donor_col", type=str, default=None)
    parser.add_argument("--max_cells", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--gene_list", type=str, default="data/training_gene_list_v3.csv")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load training gene list ---
    training_genes = pd.read_csv(args.gene_list, header=None)[0].tolist()
    print(f"Training gene list: {len(training_genes)} genes")

    # --- Load kidney data ---
    X_nuc_raw, obs_nuc, var_nuc = load_kidney_h5ad(args.nuc_h5ad, args.max_cells)
    X_cell_raw, obs_cell, var_cell = load_kidney_h5ad(args.cell_h5ad, args.max_cells)

    # --- Detect cell type column ---
    ct_col = args.ct_col
    if ct_col is None:
        for candidate in ["cell_type", "subclass.l1", "cell_type_ontology_term_id",
                          "subclass_label", "author_cell_type", "cell_ontology_class"]:
            if candidate in obs_nuc.columns:
                ct_col = candidate
                break
    if ct_col is None:
        print(f"WARNING: could not detect cell type column. Available: {list(obs_nuc.columns[:20])}")
        ct_col = obs_nuc.columns[0]
    print(f"Cell type column: '{ct_col}'")

    # --- Restrict to shared donors ---
    donor_col = args.donor_col
    if donor_col is None:
        for candidate in ["donor_id", "donor", "sample_id", "subject_id"]:
            if candidate in obs_nuc.columns:
                donor_col = candidate
                break
    if donor_col:
        shared_donors = set(obs_nuc[donor_col].unique()) & set(obs_cell[donor_col].unique())
        if shared_donors and len(shared_donors) < len(set(obs_nuc[donor_col].unique())):
            mask_n = obs_nuc[donor_col].isin(shared_donors).values
            mask_c = obs_cell[donor_col].isin(shared_donors).values
            X_nuc_raw, obs_nuc = X_nuc_raw[mask_n], obs_nuc[mask_n].copy()
            X_cell_raw, obs_cell = X_cell_raw[mask_c], obs_cell[mask_c].copy()
            print(f"  Restricted to {len(shared_donors)} shared donors: "
                  f"{X_nuc_raw.shape[0]} nuclei, {X_cell_raw.shape[0]} cells")

    # --- Build gene mapping ---
    print("\nBuilding gene mapping (human kidney -> mouse training genes)...")
    mouse2human, human2mouse, mapped_train_idx, mapped_kidney_nuc_idx = \
        build_gene_mapping(training_genes, var_nuc)

    # Also map for cell data (should be same genes but verify)
    _, _, mapped_train_idx_cell, mapped_kidney_cell_idx = \
        build_gene_mapping(training_genes, var_cell)

    # Use intersection of mapped genes (present in both nuc and cell kidney data)
    mapped_set_nuc = set(mapped_train_idx)
    mapped_set_cell = set(mapped_train_idx_cell)
    common_mapped = sorted(mapped_set_nuc & mapped_set_cell)
    print(f"  Genes mapped in both nuc & cell kidney data: {len(common_mapped)}")

    # Rebuild index lists for common mapped genes only
    # For nuclei
    kidney_nuc_lookup = {g: i for i, g in enumerate(var_nuc)}
    kidney_cell_lookup = {g: i for i, g in enumerate(var_cell)}
    final_train_idx = []
    final_nuc_idx = []
    final_cell_idx = []
    for ti in common_mapped:
        mg = training_genes[ti]
        hg = mouse2human[mg]
        final_train_idx.append(ti)
        final_nuc_idx.append(kidney_nuc_lookup[hg])
        final_cell_idx.append(kidney_cell_lookup[hg])

    print(f"  Final common gene set: {len(final_train_idx)}")

    # --- Prepare model input for nuclei ---
    print("\nPreparing model input...")
    X_nuc_model = prepare_model_input(X_nuc_raw, var_nuc, training_genes,
                                       final_train_idx, final_nuc_idx)

    # Prepare cell ground truth (log1p, aligned to training gene order)
    X_cell_log1p = np.zeros((X_cell_raw.shape[0], len(training_genes)), dtype=np.float32)
    X_cell_subset = np.log1p(X_cell_raw[:, final_cell_idx])
    for out_i, ti in enumerate(final_train_idx):
        X_cell_log1p[:, ti] = X_cell_subset[:, out_i]

    # --- Load model and run imputation ---
    print("\nLoading model...")
    model, model_args, device = load_model(args.model_run)

    print("\nRunning imputation (nuclei -> cell)...")
    imputed = run_imputation(model, X_nuc_model, device, batch_size=args.batch_size)

    # --- Evaluate ---
    print("\n" + "=" * 60)
    print("  EVALUATION")
    print("=" * 60)

    eval_results = evaluate_imputation(
        imputed, X_cell_log1p, training_genes, final_train_idx,
        obs_nuc, obs_cell, ct_col)

    delta_results = evaluate_delta_direction(
        imputed, X_nuc_model, X_cell_log1p, training_genes, final_train_idx)

    # --- Plots ---
    print("\nGenerating plots...")
    make_plots(imputed, X_cell_log1p, X_nuc_model, training_genes, final_train_idx,
               args.output_dir)

    # --- Save results ---
    summary = {
        "model_run": args.model_run,
        "dataset": "Lake 2023 human kidney",
        "n_nuclei": int(X_nuc_raw.shape[0]),
        "n_cells": int(X_cell_raw.shape[0]),
        "n_training_genes": len(training_genes),
        "n_mapped_genes": len(final_train_idx),
        "n_unmapped_genes": len(training_genes) - len(final_train_idx),
        "evaluation": eval_results,
        "delta_analysis": delta_results,
    }

    out_path = os.path.join(args.output_dir, "cross_tissue_imputation_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved results to {out_path}")

    # --- Headline ---
    print(f"\n{'=' * 60}")
    print(f"  CROSS-TISSUE IMPUTATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Model: {args.model_run}")
    print(f"  Kidney data: {X_nuc_raw.shape[0]} nuclei, {X_cell_raw.shape[0]} cells")
    print(f"  Genes: {len(final_train_idx)} mapped / {len(training_genes)} training")
    print(f"  Gene corr (global): r = {eval_results['global_gene_corr_pearson']:.4f}")
    print(f"  Gene corr (per-CT mean): r = {eval_results.get('mean_ct_gene_corr', 'N/A')}")
    print(f"  R² (gene means): {eval_results['r2_gene_means']:.4f}")
    print(f"  Delta correlation: r = {delta_results['delta_pearson']:.4f}")
    print(f"  Delta sign agreement: {delta_results['sign_agreement']:.1%}")


if __name__ == "__main__":
    main()
