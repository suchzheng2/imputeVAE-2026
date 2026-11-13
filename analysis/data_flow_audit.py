"""
End-to-end data flow audit: raw → latent → decoded → cross-decoded.

Traces mean, variance, scale, and distribution shape at every pipeline stage,
stratified by gene, sample, cell type, and modality.

Usage:
    python -m analysis.data_flow_audit --model pct5v2_cond --out audit_results/
    python -m analysis.data_flow_audit --model pct5v2_cond --out audit_results/ --skip-plots
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from scipy import stats as sp_stats

warnings.filterwarnings("ignore", category=FutureWarning)

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_DIR)

from analysis.eval.registry import get_model, list_models
from analysis.eval.data import load_eval_data
from analysis.eval.inference import load_model, CELL_ID, NUCLEI_ID


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _per_ct_mask(ct_array, ct_list):
    """Yield (ct_name, bool_mask) for each cell type."""
    for ct in ct_list:
        yield ct, ct_array == ct


def _var_ratio(a, b):
    """var(a) / var(b), safe for zero denominators."""
    vb = np.var(b, axis=0)
    va = np.var(a, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.where(vb > 0, va / vb, np.nan)
    return r


def _per_gene_corr(X, Y):
    """Per-gene Pearson correlation across samples. X, Y: [N, G]."""
    assert X.shape == Y.shape
    G = X.shape[1]
    corrs = np.full(G, np.nan)
    for g in range(G):
        if np.std(X[:, g]) > 1e-10 and np.std(Y[:, g]) > 1e-10:
            corrs[g] = np.corrcoef(X[:, g], Y[:, g])[0, 1]
    return corrs


def _per_sample_corr(X, Y):
    """Per-sample Pearson correlation across genes. X, Y: [N, G]."""
    assert X.shape == Y.shape
    N = X.shape[0]
    corrs = np.full(N, np.nan)
    for i in range(N):
        if np.std(X[i, :]) > 1e-10 and np.std(Y[i, :]) > 1e-10:
            corrs[i] = np.corrcoef(X[i, :], Y[i, :])[0, 1]
    return corrs


def _within_ct_var(X, ct_array, ct_list):
    """Mean within-cell-type variance per gene. Returns [G]."""
    G = X.shape[1]
    var_sum = np.zeros(G)
    weight_sum = 0
    for ct, mask in _per_ct_mask(ct_array, ct_list):
        n = mask.sum()
        if n > 1:
            var_sum += np.var(X[mask], axis=0) * n
            weight_sum += n
    return var_sum / max(weight_sum, 1)


def _between_ct_var(X, ct_array, ct_list):
    """Between-cell-type variance per gene (variance of CT means). Returns [G]."""
    means = []
    weights = []
    for ct, mask in _per_ct_mask(ct_array, ct_list):
        n = mask.sum()
        if n > 0:
            means.append(X[mask].mean(axis=0))
            weights.append(n)
    means = np.array(means)  # [K, G]
    weights = np.array(weights)
    grand_mean = np.average(means, axis=0, weights=weights)
    return np.average((means - grand_mean) ** 2, axis=0, weights=weights)


# ──────────────────────────────────────────────────────────────────────
# Section A: Raw data characterization
# ──────────────────────────────────────────────────────────────────────

def section_a_raw(data, out_dir):
    """Raw data: gene-level, CT-level, sample-level, modality-level."""
    print("\n=== Section A: Raw Data Characterization ===")
    os.makedirs(out_dir, exist_ok=True)

    X_cell, X_nuc = data.X_cell, data.X_nuc
    ct_cell, ct_nuc = data.ct_cell, data.ct_nuc
    genes = data.gene_names
    cts = data.common_cts

    # ---- A1: Gene-level ----
    gene_df = pd.DataFrame({
        'gene': genes,
        'mean_cell': X_cell.mean(axis=0),
        'mean_nuc': X_nuc.mean(axis=0),
        'var_cell': X_cell.var(axis=0),
        'var_nuc': X_nuc.var(axis=0),
        'cv_cell': np.where(X_cell.mean(axis=0) > 1e-10,
                            X_cell.std(axis=0) / X_cell.mean(axis=0), 0),
        'cv_nuc': np.where(X_nuc.mean(axis=0) > 1e-10,
                           X_nuc.std(axis=0) / X_nuc.mean(axis=0), 0),
        'frac_zero_cell': (X_cell == 0).mean(axis=0),
        'frac_zero_nuc': (X_nuc == 0).mean(axis=0),
        'within_ct_var_cell': _within_ct_var(X_cell, ct_cell, cts),
        'within_ct_var_nuc': _within_ct_var(X_nuc, ct_nuc, cts),
        'between_ct_var_cell': _between_ct_var(X_cell, ct_cell, cts),
        'between_ct_var_nuc': _between_ct_var(X_nuc, ct_nuc, cts),
    })
    gene_df['mean_diff'] = gene_df['mean_cell'] - gene_df['mean_nuc']
    gene_df['mean_ratio'] = np.where(gene_df['mean_nuc'] > 1e-10,
                                     gene_df['mean_cell'] / gene_df['mean_nuc'], np.nan)
    gene_df['var_ratio_cell_nuc'] = np.where(gene_df['var_nuc'] > 1e-10,
                                             gene_df['var_cell'] / gene_df['var_nuc'], np.nan)
    gene_df['within_frac_cell'] = np.where(gene_df['var_cell'] > 1e-10,
                                           gene_df['within_ct_var_cell'] / gene_df['var_cell'], np.nan)
    gene_df['within_frac_nuc'] = np.where(gene_df['var_nuc'] > 1e-10,
                                          gene_df['within_ct_var_nuc'] / gene_df['var_nuc'], np.nan)
    gene_df.to_csv(os.path.join(out_dir, 'A1_gene_level.csv'), index=False)

    # Summary
    print(f"  Genes: {len(genes)}")
    print(f"  Cell mean expression: {X_cell.mean():.4f} (std {X_cell.std():.4f})")
    print(f"  Nuclei mean expression: {X_nuc.mean():.4f} (std {X_nuc.std():.4f})")
    print(f"  Genes where cell > nuclei (mean): {(gene_df['mean_diff'] > 0).sum()}")
    print(f"  Genes where nuclei > cell (mean): {(gene_df['mean_diff'] < 0).sum()}")
    print(f"  Median within-CT variance fraction — cell: {np.nanmedian(gene_df['within_frac_cell']):.3f}, "
          f"nuc: {np.nanmedian(gene_df['within_frac_nuc']):.3f}")

    # ---- A2: Cell-type level ----
    ct_rows = []
    for ct in cts:
        c_mask = ct_cell == ct
        n_mask = ct_nuc == ct
        ct_rows.append({
            'cell_type': ct,
            'n_cell': c_mask.sum(),
            'n_nuc': n_mask.sum(),
            'mean_expr_cell': X_cell[c_mask].mean(),
            'mean_expr_nuc': X_nuc[n_mask].mean(),
            'total_var_cell': X_cell[c_mask].var(),
            'total_var_nuc': X_nuc[n_mask].var(),
            'mean_gene_var_cell': X_cell[c_mask].var(axis=0).mean(),
            'mean_gene_var_nuc': X_nuc[n_mask].var(axis=0).mean(),
        })
    ct_df = pd.DataFrame(ct_rows)
    ct_df['var_ratio_cell_nuc'] = np.where(ct_df['total_var_nuc'] > 0,
                                           ct_df['total_var_cell'] / ct_df['total_var_nuc'], np.nan)
    ct_df.to_csv(os.path.join(out_dir, 'A2_celltype_level.csv'), index=False)
    print(f"  Cell types: {len(cts)}")
    print(f"  Cell counts — cell: {ct_df['n_cell'].sum()}, nuc: {ct_df['n_nuc'].sum()}")

    # ---- A3: Sample-level ----
    lib_cell = X_cell.sum(axis=1)  # in log1p space — "pseudo library size"
    lib_nuc = X_nuc.sum(axis=1)
    sample_summary = {
        'lib_size_cell_mean': float(lib_cell.mean()),
        'lib_size_cell_std': float(lib_cell.std()),
        'lib_size_nuc_mean': float(lib_nuc.mean()),
        'lib_size_nuc_std': float(lib_nuc.std()),
        'lib_ratio_cell_nuc': float(lib_cell.mean() / lib_nuc.mean()),
        'dynamic_range_cell': float(X_cell.max() - X_cell.min()),
        'dynamic_range_nuc': float(X_nuc.max() - X_nuc.min()),
        'n_cell': int(X_cell.shape[0]),
        'n_nuc': int(X_nuc.shape[0]),
    }
    with open(os.path.join(out_dir, 'A3_sample_level.json'), 'w') as f:
        json.dump(sample_summary, f, indent=2)
    print(f"  Library size (log1p sum) — cell: {lib_cell.mean():.1f} ± {lib_cell.std():.1f}, "
          f"nuc: {lib_nuc.mean():.1f} ± {lib_nuc.std():.1f}")

    # ---- A4: Representative gene distributions ----
    # Pick genes at different expression levels
    gene_means = gene_df['mean_cell'].values
    sorted_idx = np.argsort(gene_means)
    n_genes = len(sorted_idx)
    representative_genes = {}
    # Low, medium, high expression
    for label, pct in [('low_expr', 0.1), ('mid_expr', 0.5), ('high_expr', 0.9)]:
        idx = sorted_idx[int(pct * n_genes)]
        representative_genes[label] = genes[idx]
    # Most variable gene (by CV in cells)
    most_var_idx = np.argmax(gene_df['cv_cell'].values)
    representative_genes['high_cv'] = genes[most_var_idx]
    # Biggest cell-nuclei gap
    biggest_gap_idx = np.argmax(np.abs(gene_df['mean_diff'].values))
    representative_genes['biggest_gap'] = genes[biggest_gap_idx]

    with open(os.path.join(out_dir, 'A4_representative_genes.json'), 'w') as f:
        json.dump(representative_genes, f, indent=2)
    print(f"  Representative genes: {representative_genes}")

    # ---- A5: Save library size arrays and per-CT mean profiles for new plots ----
    lib_cell = X_cell.sum(axis=1)
    lib_nuc = X_nuc.sum(axis=1)
    np.savez_compressed(
        os.path.join(out_dir, 'A5_library_sizes.npz'),
        lib_cell=lib_cell, lib_nuc=lib_nuc,
    )

    # Per-CT mean gene profiles
    ct_mean_profiles = {}
    for ct in cts:
        c_mask = ct_cell == ct
        n_mask = ct_nuc == ct
        ct_mean_profiles[f"{ct}_cell"] = X_cell[c_mask].mean(axis=0)
        ct_mean_profiles[f"{ct}_nuc"] = X_nuc[n_mask].mean(axis=0)
    np.savez_compressed(
        os.path.join(out_dir, 'A5_ct_mean_profiles.npz'),
        **ct_mean_profiles,
        cell_types=json.dumps(cts),
        gene_names=genes,
    )

    # Save per-gene distributions for representative genes
    rep_dist = {}
    for label, gname in representative_genes.items():
        gidx = np.where(genes == gname)[0][0]
        rep_dist[label] = {
            'gene': gname,
            'cell': X_cell[:, gidx].tolist(),
            'nuc': X_nuc[:, gidx].tolist(),
        }
    np.savez_compressed(
        os.path.join(out_dir, 'A4_representative_distributions.npz'),
        **{f"{k}_cell": np.array(v['cell']) for k, v in rep_dist.items()},
        **{f"{k}_nuc": np.array(v['nuc']) for k, v in rep_dist.items()},
        gene_names=json.dumps({k: v['gene'] for k, v in rep_dist.items()}),
    )

    return gene_df, ct_df, representative_genes


# ──────────────────────────────────────────────────────────────────────
# Section B: Latent space
# ──────────────────────────────────────────────────────────────────────

def run_latent_extraction(model, X, modality_id, device, batch_size=256):
    """Extract mu and log_var from encoder. Returns (mu, log_var) as numpy arrays."""
    X_t = torch.tensor(X, dtype=torch.float32)
    mu_parts, logvar_parts = [], []
    with torch.no_grad():
        for i in range(0, X_t.shape[0], batch_size):
            batch = X_t[i:i + batch_size].to(device)
            encoded = model.encoder(batch)
            mu = model.mu(encoded)
            log_var = model.log_var(encoded)
            mu_parts.append(mu.cpu().numpy())
            logvar_parts.append(log_var.cpu().numpy())
    return np.concatenate(mu_parts), np.concatenate(logvar_parts)


def section_b_latent(model, data, device, out_dir):
    """Latent space: per-dim, per-CT, per-sample statistics."""
    print("\n=== Section B: Latent Space ===")
    os.makedirs(out_dir, exist_ok=True)

    cts = data.common_cts

    # Extract latent for both modalities
    mu_cell, logvar_cell = run_latent_extraction(model, data.X_cell, CELL_ID, device)
    mu_nuc, logvar_nuc = run_latent_extraction(model, data.X_nuc, NUCLEI_ID, device)
    std_cell = np.exp(logvar_cell * 0.5)
    std_nuc = np.exp(logvar_nuc * 0.5)

    L = mu_cell.shape[1]
    print(f"  Latent dims: {L}")

    # ---- B1: Per-dimension statistics ----
    dim_df = pd.DataFrame({
        'dim': np.arange(L),
        'mean_cell': mu_cell.mean(axis=0),
        'mean_nuc': mu_nuc.mean(axis=0),
        'var_cell': mu_cell.var(axis=0),
        'var_nuc': mu_nuc.var(axis=0),
        'std_posterior_cell': std_cell.mean(axis=0),
        'std_posterior_nuc': std_nuc.mean(axis=0),
        'mean_shift': mu_cell.mean(axis=0) - mu_nuc.mean(axis=0),
    })

    # Within-CT and between-CT variance decomposition
    dim_df['within_ct_var_cell'] = _within_ct_var(mu_cell, data.ct_cell, cts).ravel()
    dim_df['within_ct_var_nuc'] = _within_ct_var(mu_nuc, data.ct_nuc, cts).ravel()
    dim_df['between_ct_var_cell'] = _between_ct_var(mu_cell, data.ct_cell, cts).ravel()
    dim_df['between_ct_var_nuc'] = _between_ct_var(mu_nuc, data.ct_nuc, cts).ravel()

    # Inter-modality variance: combine both, compute modality-level variance
    mu_all = np.concatenate([mu_cell, mu_nuc], axis=0)
    mod_labels = np.array(['cell'] * len(mu_cell) + ['nuc'] * len(mu_nuc))
    dim_df['inter_modality_var'] = _between_ct_var(mu_all, mod_labels, ['cell', 'nuc']).ravel()
    dim_df['total_var'] = mu_all.var(axis=0)
    dim_df['inter_modality_frac'] = np.where(dim_df['total_var'] > 1e-10,
                                             dim_df['inter_modality_var'] / dim_df['total_var'], 0)

    dim_df.to_csv(os.path.join(out_dir, 'B1_per_dim.csv'), index=False)
    print(f"  Inter-modality variance fraction: {dim_df['inter_modality_frac'].mean():.3f} "
          f"(range {dim_df['inter_modality_frac'].min():.3f}–{dim_df['inter_modality_frac'].max():.3f})")
    print(f"  Mean posterior std — cell: {std_cell.mean():.4f}, nuc: {std_nuc.mean():.4f}")

    # ---- B2: Per-CT latent statistics ----
    ct_rows = []
    for ct in cts:
        c_mask = data.ct_cell == ct
        n_mask = data.ct_nuc == ct
        mu_c = mu_cell[c_mask]
        mu_n = mu_nuc[n_mask]
        centroid_c = mu_c.mean(axis=0)
        centroid_n = mu_n.mean(axis=0)
        ct_rows.append({
            'cell_type': ct,
            'n_cell': c_mask.sum(),
            'n_nuc': n_mask.sum(),
            'centroid_dist': float(np.linalg.norm(centroid_c - centroid_n)),
            'within_ct_var_cell': float(mu_c.var()),
            'within_ct_var_nuc': float(mu_n.var()),
            'latent_norm_cell': float(np.linalg.norm(mu_c, axis=1).mean()),
            'latent_norm_nuc': float(np.linalg.norm(mu_n, axis=1).mean()),
        })
    ct_lat_df = pd.DataFrame(ct_rows)
    ct_lat_df.to_csv(os.path.join(out_dir, 'B2_per_ct_latent.csv'), index=False)
    print(f"  CT centroid distance — mean: {ct_lat_df['centroid_dist'].mean():.4f}, "
          f"max: {ct_lat_df['centroid_dist'].max():.4f}")

    # ---- B3: Sample-level ----
    sample_summary = {
        'latent_norm_cell_mean': float(np.linalg.norm(mu_cell, axis=1).mean()),
        'latent_norm_cell_std': float(np.linalg.norm(mu_cell, axis=1).std()),
        'latent_norm_nuc_mean': float(np.linalg.norm(mu_nuc, axis=1).mean()),
        'latent_norm_nuc_std': float(np.linalg.norm(mu_nuc, axis=1).std()),
    }
    with open(os.path.join(out_dir, 'B3_sample_latent.json'), 'w') as f:
        json.dump(sample_summary, f, indent=2)

    # Save raw latent arrays for plotting
    np.savez_compressed(
        os.path.join(out_dir, 'B_latent_arrays.npz'),
        mu_cell=mu_cell, mu_nuc=mu_nuc,
        logvar_cell=logvar_cell, logvar_nuc=logvar_nuc,
    )

    return mu_cell, mu_nuc


# ──────────────────────────────────────────────────────────────────────
# Section C: Decoded (same-modality reconstruction)
# ──────────────────────────────────────────────────────────────────────

def run_decoded(model, X, modality_id, device, batch_size=256):
    """Run deterministic reconstruction: encode → mu → decode."""
    X_t = torch.tensor(X, dtype=torch.float32)
    parts = []
    has_cond = hasattr(model, 'modality_emb')
    with torch.no_grad():
        for i in range(0, X_t.shape[0], batch_size):
            batch = X_t[i:i + batch_size].to(device)
            encoded = model.encoder(batch)
            mu = model.mu(encoded)
            if has_cond:
                emb = model._get_modality_emb(batch.shape[0], modality_id, device)
                z_cond = torch.cat([mu, emb], dim=1)
                decoded = model.decoder(z_cond)
            else:
                decoded = model.decoder(mu)
            parts.append(decoded.cpu().numpy())
    return np.concatenate(parts, axis=0)


def run_cross_decode(model, X_source, source_id, target_id, device, batch_size=256):
    """Cross-decode: encode source modality, decode with target embedding."""
    X_t = torch.tensor(X_source, dtype=torch.float32)
    parts = []
    with torch.no_grad():
        for i in range(0, X_t.shape[0], batch_size):
            batch = X_t[i:i + batch_size].to(device)
            decoded = model.decode_cross(batch, source_id, target_id)
            parts.append(decoded.cpu().numpy())
    return np.concatenate(parts, axis=0)


def section_c_decoded(model, data, device, out_dir, gene_df, representative_genes):
    """Same-modality reconstruction statistics."""
    print("\n=== Section C: Same-Modality Reconstruction ===")
    os.makedirs(out_dir, exist_ok=True)

    cts = data.common_cts
    genes = data.gene_names

    # Reconstruct both modalities
    dec_cell = run_decoded(model, data.X_cell, CELL_ID, device)
    dec_nuc = run_decoded(model, data.X_nuc, NUCLEI_ID, device)

    # ---- C1: Gene-level ----
    c1 = pd.DataFrame({
        'gene': genes,
        # Cell reconstruction
        'mean_raw_cell': data.X_cell.mean(axis=0),
        'mean_dec_cell': dec_cell.mean(axis=0),
        'var_raw_cell': data.X_cell.var(axis=0),
        'var_dec_cell': dec_cell.var(axis=0),
        'var_ratio_cell': _var_ratio(dec_cell, data.X_cell),
        'gene_corr_cell': _per_gene_corr(dec_cell, data.X_cell),
        # Nuclei reconstruction
        'mean_raw_nuc': data.X_nuc.mean(axis=0),
        'mean_dec_nuc': dec_nuc.mean(axis=0),
        'var_raw_nuc': data.X_nuc.var(axis=0),
        'var_dec_nuc': dec_nuc.var(axis=0),
        'var_ratio_nuc': _var_ratio(dec_nuc, data.X_nuc),
        'gene_corr_nuc': _per_gene_corr(dec_nuc, data.X_nuc),
    })
    c1['mean_bias_cell'] = c1['mean_dec_cell'] - c1['mean_raw_cell']
    c1['mean_bias_nuc'] = c1['mean_dec_nuc'] - c1['mean_raw_nuc']

    # Within-CT variance in decoded
    c1['within_ct_var_dec_cell'] = _within_ct_var(dec_cell, data.ct_cell, cts)
    c1['within_ct_var_dec_nuc'] = _within_ct_var(dec_nuc, data.ct_nuc, cts)
    c1['within_ct_var_ratio_cell'] = np.where(
        gene_df['within_ct_var_cell'] > 1e-10,
        c1['within_ct_var_dec_cell'] / gene_df['within_ct_var_cell'], np.nan)
    c1['within_ct_var_ratio_nuc'] = np.where(
        gene_df['within_ct_var_nuc'] > 1e-10,
        c1['within_ct_var_dec_nuc'] / gene_df['within_ct_var_nuc'], np.nan)

    c1.to_csv(os.path.join(out_dir, 'C1_gene_level_recon.csv'), index=False)
    print(f"  Cell recon — mean gene corr: {np.nanmean(c1['gene_corr_cell']):.4f}, "
          f"median var ratio: {np.nanmedian(c1['var_ratio_cell']):.4f}")
    print(f"  Nuc recon — mean gene corr: {np.nanmean(c1['gene_corr_nuc']):.4f}, "
          f"median var ratio: {np.nanmedian(c1['var_ratio_nuc']):.4f}")
    print(f"  Cell within-CT var ratio — median: {np.nanmedian(c1['within_ct_var_ratio_cell']):.4f}")
    print(f"  Nuc within-CT var ratio — median: {np.nanmedian(c1['within_ct_var_ratio_nuc']):.4f}")

    # ---- C2: Cell-type level ----
    ct_rows = []
    for ct in cts:
        c_mask = data.ct_cell == ct
        n_mask = data.ct_nuc == ct
        raw_c, raw_n = data.X_cell[c_mask], data.X_nuc[n_mask]
        d_c, d_n = dec_cell[c_mask], dec_nuc[n_mask]
        ct_rows.append({
            'cell_type': ct,
            # Cell
            'mean_raw_cell': raw_c.mean(),
            'mean_dec_cell': d_c.mean(),
            'var_raw_cell': raw_c.var(),
            'var_dec_cell': d_c.var(),
            'var_ratio_cell': d_c.var() / max(raw_c.var(), 1e-10),
            'per_sample_corr_cell': np.nanmean(_per_sample_corr(d_c, raw_c)),
            # Nuclei
            'mean_raw_nuc': raw_n.mean(),
            'mean_dec_nuc': d_n.mean(),
            'var_raw_nuc': raw_n.var(),
            'var_dec_nuc': d_n.var(),
            'var_ratio_nuc': d_n.var() / max(raw_n.var(), 1e-10),
            'per_sample_corr_nuc': np.nanmean(_per_sample_corr(d_n, raw_n)),
        })
    ct_recon_df = pd.DataFrame(ct_rows)
    ct_recon_df.to_csv(os.path.join(out_dir, 'C2_ct_level_recon.csv'), index=False)
    print(f"  Per-CT var ratio — cell mean: {ct_recon_df['var_ratio_cell'].mean():.4f}, "
          f"nuc mean: {ct_recon_df['var_ratio_nuc'].mean():.4f}")

    # ---- C3: Sample-level ----
    sample_corr_cell = _per_sample_corr(dec_cell, data.X_cell)
    sample_corr_nuc = _per_sample_corr(dec_nuc, data.X_nuc)
    sample_summary = {
        'per_sample_corr_cell_mean': float(np.nanmean(sample_corr_cell)),
        'per_sample_corr_cell_std': float(np.nanstd(sample_corr_cell)),
        'per_sample_corr_cell_min': float(np.nanmin(sample_corr_cell)),
        'per_sample_corr_nuc_mean': float(np.nanmean(sample_corr_nuc)),
        'per_sample_corr_nuc_std': float(np.nanstd(sample_corr_nuc)),
        'per_sample_corr_nuc_min': float(np.nanmin(sample_corr_nuc)),
    }
    with open(os.path.join(out_dir, 'C3_sample_level_recon.json'), 'w') as f:
        json.dump(sample_summary, f, indent=2)
    print(f"  Per-sample corr — cell: {sample_summary['per_sample_corr_cell_mean']:.4f} "
          f"(min {sample_summary['per_sample_corr_cell_min']:.4f}), "
          f"nuc: {sample_summary['per_sample_corr_nuc_mean']:.4f}")

    # ---- C4: Representative gene distributions (decoded vs raw) ----
    rep_dist = {}
    for label, gname in representative_genes.items():
        gidx = np.where(genes == gname)[0][0]
        rep_dist[f"{label}_dec_cell"] = dec_cell[:, gidx]
        rep_dist[f"{label}_dec_nuc"] = dec_nuc[:, gidx]
    np.savez_compressed(os.path.join(out_dir, 'C4_representative_decoded.npz'), **rep_dist)

    return dec_cell, dec_nuc


# ──────────────────────────────────────────────────────────────────────
# Section D: Cross-decode (nuclei → cell)
# ──────────────────────────────────────────────────────────────────────

def section_d_cross(model, data, device, out_dir, dec_cell, gene_df, representative_genes):
    """Cross-decode: nuclei encoded → cell decoded."""
    print("\n=== Section D: Cross-Decode (Nuclei → Cell) ===")
    os.makedirs(out_dir, exist_ok=True)

    cts = data.common_cts
    genes = data.gene_names

    cross_dec = run_cross_decode(model, data.X_nuc, NUCLEI_ID, CELL_ID, device)

    # ---- D1: Gene-level ----
    d1 = pd.DataFrame({
        'gene': genes,
        'mean_raw_cell': data.X_cell.mean(axis=0),
        'mean_cross': cross_dec.mean(axis=0),
        'mean_raw_nuc': data.X_nuc.mean(axis=0),
        'var_raw_cell': data.X_cell.var(axis=0),
        'var_cross': cross_dec.var(axis=0),
        'var_raw_nuc': data.X_nuc.var(axis=0),
        'gene_corr_vs_cell': _per_gene_corr(cross_dec, data.X_nuc),  # limited: unpaired
    })
    d1['mean_bias_vs_cell'] = d1['mean_cross'] - d1['mean_raw_cell']
    d1['var_ratio_vs_cell'] = _var_ratio(cross_dec, data.X_cell)

    # Within-CT stats
    d1['within_ct_var_cross'] = _within_ct_var(cross_dec, data.ct_nuc, cts)
    d1['within_ct_var_raw_cell'] = gene_df['within_ct_var_cell'].values
    d1['within_ct_var_ratio'] = np.where(
        d1['within_ct_var_raw_cell'] > 1e-10,
        d1['within_ct_var_cross'] / d1['within_ct_var_raw_cell'], np.nan)

    # How much additional variance is lost at cross-decode vs same-modality recon?
    d1['var_dec_cell'] = dec_cell.var(axis=0)
    d1['additional_var_loss'] = np.where(
        d1['var_dec_cell'] > 1e-10,
        d1['var_cross'] / d1['var_dec_cell'], np.nan)  # <1 means cross loses more

    d1.to_csv(os.path.join(out_dir, 'D1_gene_level_cross.csv'), index=False)
    print(f"  Cross-decode — mean var ratio vs cell: {np.nanmedian(d1['var_ratio_vs_cell']):.4f}")
    print(f"  Within-CT var ratio vs cell — median: {np.nanmedian(d1['within_ct_var_ratio']):.4f}")
    print(f"  Additional var loss (cross / same-mod recon) — median: "
          f"{np.nanmedian(d1['additional_var_loss']):.4f}")

    # ---- D2: Cell-type level ----
    ct_rows = []
    for ct in cts:
        n_mask = data.ct_nuc == ct
        c_mask = data.ct_cell == ct
        xd = cross_dec[n_mask]
        raw_c = data.X_cell[c_mask]
        # CT mean comparison (unpaired, so compare means)
        ct_mean_cross = xd.mean(axis=0)
        ct_mean_cell = raw_c.mean(axis=0)
        mean_corr = np.corrcoef(ct_mean_cross, ct_mean_cell)[0, 1] if len(ct_mean_cross) > 1 else np.nan

        ct_rows.append({
            'cell_type': ct,
            'n_nuc': n_mask.sum(),
            'n_cell': c_mask.sum(),
            'mean_cross': xd.mean(),
            'mean_raw_cell': raw_c.mean(),
            'var_cross': xd.var(),
            'var_raw_cell': raw_c.var(),
            'var_ratio': xd.var() / max(raw_c.var(), 1e-10),
            'ct_mean_gene_corr': float(mean_corr),
            'within_ct_var_cross': xd.var(axis=0).mean(),
            'within_ct_var_cell': raw_c.var(axis=0).mean(),
        })
    ct_cross_df = pd.DataFrame(ct_rows)
    ct_cross_df['within_ct_var_ratio'] = np.where(
        ct_cross_df['within_ct_var_cell'] > 0,
        ct_cross_df['within_ct_var_cross'] / ct_cross_df['within_ct_var_cell'], np.nan)
    ct_cross_df.to_csv(os.path.join(out_dir, 'D2_ct_level_cross.csv'), index=False)
    print(f"  Per-CT var ratio — mean: {ct_cross_df['var_ratio'].mean():.4f}, "
          f"range: {ct_cross_df['var_ratio'].min():.4f}–{ct_cross_df['var_ratio'].max():.4f}")
    print(f"  Per-CT mean gene corr — mean: {ct_cross_df['ct_mean_gene_corr'].mean():.4f}")

    # ---- D3: Representative gene distributions ----
    rep_dist = {}
    for label, gname in representative_genes.items():
        gidx = np.where(genes == gname)[0][0]
        rep_dist[f"{label}_cross"] = cross_dec[:, gidx]
    np.savez_compressed(os.path.join(out_dir, 'D3_representative_cross.npz'), **rep_dist)

    return cross_dec


# ──────────────────────────────────────────────────────────────────────
# Section E: Stage-by-stage variance tracking
# ──────────────────────────────────────────────────────────────────────

def section_e_variance_flow(data, dec_cell, dec_nuc, cross_dec, mu_cell, mu_nuc, out_dir):
    """Track variance from raw → latent → decoded → cross-decoded."""
    print("\n=== Section E: Variance Flow (Punchline) ===")
    os.makedirs(out_dir, exist_ok=True)

    cts = data.common_cts
    genes = data.gene_names

    # ---- E1: Per-gene variance flow ----
    e1 = pd.DataFrame({
        'gene': genes,
        # Total variance
        'total_var_raw_cell': data.X_cell.var(axis=0),
        'total_var_raw_nuc': data.X_nuc.var(axis=0),
        'total_var_dec_cell': dec_cell.var(axis=0),
        'total_var_dec_nuc': dec_nuc.var(axis=0),
        'total_var_cross': cross_dec.var(axis=0),
        # Within-CT variance
        'within_var_raw_cell': _within_ct_var(data.X_cell, data.ct_cell, cts),
        'within_var_raw_nuc': _within_ct_var(data.X_nuc, data.ct_nuc, cts),
        'within_var_dec_cell': _within_ct_var(dec_cell, data.ct_cell, cts),
        'within_var_dec_nuc': _within_ct_var(dec_nuc, data.ct_nuc, cts),
        'within_var_cross': _within_ct_var(cross_dec, data.ct_nuc, cts),
        # Between-CT variance
        'between_var_raw_cell': _between_ct_var(data.X_cell, data.ct_cell, cts),
        'between_var_raw_nuc': _between_ct_var(data.X_nuc, data.ct_nuc, cts),
        'between_var_dec_cell': _between_ct_var(dec_cell, data.ct_cell, cts),
        'between_var_dec_nuc': _between_ct_var(dec_nuc, data.ct_nuc, cts),
        'between_var_cross': _between_ct_var(cross_dec, data.ct_nuc, cts),
    })

    # Ratios
    for stage in ['dec_cell', 'dec_nuc', 'cross']:
        ref = 'raw_cell' if stage != 'dec_nuc' else 'raw_nuc'
        for vtype in ['total_var', 'within_var', 'between_var']:
            num_col = f"{vtype}_{stage}"
            den_col = f"{vtype}_{ref}"
            e1[f"{vtype}_ratio_{stage}"] = np.where(
                e1[den_col] > 1e-10, e1[num_col] / e1[den_col], np.nan)

    e1.to_csv(os.path.join(out_dir, 'E1_gene_variance_flow.csv'), index=False)

    # ---- E2: Aggregate variance flow summary ----
    summary_rows = []

    def _agg(X, ct, label):
        return {
            'stage': label,
            'total_var': float(X.var()),
            'mean_gene_var': float(X.var(axis=0).mean()),
            'mean_within_ct_var': float(_within_ct_var(X, ct, cts).mean()),
            'mean_between_ct_var': float(_between_ct_var(X, ct, cts).mean()),
        }

    summary_rows.append(_agg(data.X_cell, data.ct_cell, 'raw_cell'))
    summary_rows.append(_agg(data.X_nuc, data.ct_nuc, 'raw_nuc'))
    summary_rows.append(_agg(dec_cell, data.ct_cell, 'decoded_cell'))
    summary_rows.append(_agg(dec_nuc, data.ct_nuc, 'decoded_nuc'))
    summary_rows.append(_agg(cross_dec, data.ct_nuc, 'cross_decoded'))

    # Latent space (different dimensionality, report separately)
    latent_summary = {
        'mu_cell_var': float(mu_cell.var()),
        'mu_nuc_var': float(mu_nuc.var()),
        'mu_cell_within_ct_var': float(_within_ct_var(mu_cell, data.ct_cell, cts).mean()),
        'mu_nuc_within_ct_var': float(_within_ct_var(mu_nuc, data.ct_nuc, cts).mean()),
        'mu_cell_between_ct_var': float(_between_ct_var(mu_cell, data.ct_cell, cts).mean()),
        'mu_nuc_between_ct_var': float(_between_ct_var(mu_nuc, data.ct_nuc, cts).mean()),
    }

    summary_df = pd.DataFrame(summary_rows)
    summary_df['within_frac'] = summary_df['mean_within_ct_var'] / (
        summary_df['mean_within_ct_var'] + summary_df['mean_between_ct_var'])
    summary_df.to_csv(os.path.join(out_dir, 'E2_variance_flow_summary.csv'), index=False)

    with open(os.path.join(out_dir, 'E2_latent_variance.json'), 'w') as f:
        json.dump(latent_summary, f, indent=2)

    # ---- E3: Per-CT variance flow ----
    ct_rows = []
    for ct in cts:
        c_mask = data.ct_cell == ct
        n_mask = data.ct_nuc == ct
        ct_rows.append({
            'cell_type': ct,
            'var_raw_cell': float(data.X_cell[c_mask].var()),
            'var_raw_nuc': float(data.X_nuc[n_mask].var()),
            'var_dec_cell': float(dec_cell[c_mask].var()),
            'var_dec_nuc': float(dec_nuc[n_mask].var()),
            'var_cross': float(cross_dec[n_mask].var()),
            # Gene-mean variance (within this CT, per gene, averaged)
            'gene_var_raw_cell': float(data.X_cell[c_mask].var(axis=0).mean()),
            'gene_var_raw_nuc': float(data.X_nuc[n_mask].var(axis=0).mean()),
            'gene_var_dec_cell': float(dec_cell[c_mask].var(axis=0).mean()),
            'gene_var_dec_nuc': float(dec_nuc[n_mask].var(axis=0).mean()),
            'gene_var_cross': float(cross_dec[n_mask].var(axis=0).mean()),
        })
    ct_var_df = pd.DataFrame(ct_rows)
    # Ratios
    ct_var_df['var_ratio_dec_cell'] = ct_var_df['var_dec_cell'] / ct_var_df['var_raw_cell'].clip(1e-10)
    ct_var_df['var_ratio_dec_nuc'] = ct_var_df['var_dec_nuc'] / ct_var_df['var_raw_nuc'].clip(1e-10)
    ct_var_df['var_ratio_cross'] = ct_var_df['var_cross'] / ct_var_df['var_raw_cell'].clip(1e-10)
    ct_var_df.to_csv(os.path.join(out_dir, 'E3_ct_variance_flow.csv'), index=False)

    # ---- Print summary table ----
    print("\n  Variance flow summary:")
    print(f"  {'Stage':<20} {'Mean gene var':>14} {'Within-CT':>12} {'Between-CT':>12} {'Within frac':>12}")
    for _, row in summary_df.iterrows():
        print(f"  {row['stage']:<20} {row['mean_gene_var']:>14.6f} "
              f"{row['mean_within_ct_var']:>12.6f} {row['mean_between_ct_var']:>12.6f} "
              f"{row['within_frac']:>12.3f}")

    print(f"\n  Latent variance:")
    print(f"    Cell — total: {latent_summary['mu_cell_var']:.4f}, "
          f"within-CT: {latent_summary['mu_cell_within_ct_var']:.4f}, "
          f"between-CT: {latent_summary['mu_cell_between_ct_var']:.4f}")
    print(f"    Nuc  — total: {latent_summary['mu_nuc_var']:.4f}, "
          f"within-CT: {latent_summary['mu_nuc_within_ct_var']:.4f}, "
          f"between-CT: {latent_summary['mu_nuc_between_ct_var']:.4f}")

    # Key ratios
    raw_cell_within = summary_df.loc[summary_df['stage'] == 'raw_cell', 'mean_within_ct_var'].values[0]
    dec_cell_within = summary_df.loc[summary_df['stage'] == 'decoded_cell', 'mean_within_ct_var'].values[0]
    cross_within = summary_df.loc[summary_df['stage'] == 'cross_decoded', 'mean_within_ct_var'].values[0]
    print(f"\n  Variance retention (within-CT):")
    print(f"    Cell recon / raw cell: {dec_cell_within / max(raw_cell_within, 1e-10):.4f}")
    print(f"    Cross-decode / raw cell: {cross_within / max(raw_cell_within, 1e-10):.4f}")


# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────

def make_plots(out_dir):
    """Generate all plots from saved CSV/NPZ data."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    plot_dir = os.path.join(out_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    print("\n=== Generating Plots ===")

    # ---- Plot 1: Raw cell vs nuclei (gene-level scatter) ----
    gene_df = pd.read_csv(os.path.join(out_dir, 'A1_gene_level.csv'))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].scatter(gene_df['mean_cell'], gene_df['mean_nuc'], s=1, alpha=0.3)
    lim = max(gene_df['mean_cell'].max(), gene_df['mean_nuc'].max()) * 1.05
    axes[0].plot([0, lim], [0, lim], 'r--', lw=0.5)
    axes[0].set_xlabel('Cell mean (log1p)')
    axes[0].set_ylabel('Nuclei mean (log1p)')
    axes[0].set_title('A1: Gene mean — cell vs nuclei')

    axes[1].scatter(gene_df['var_cell'], gene_df['var_nuc'], s=1, alpha=0.3)
    lim = max(gene_df['var_cell'].max(), gene_df['var_nuc'].max()) * 1.05
    axes[1].plot([0, lim], [0, lim], 'r--', lw=0.5)
    axes[1].set_xlabel('Cell variance (log1p)')
    axes[1].set_ylabel('Nuclei variance (log1p)')
    axes[1].set_title('A1: Gene variance — cell vs nuclei')

    axes[2].scatter(gene_df['frac_zero_cell'], gene_df['frac_zero_nuc'], s=1, alpha=0.3)
    axes[2].plot([0, 1], [0, 1], 'r--', lw=0.5)
    axes[2].set_xlabel('Fraction zeros (cell)')
    axes[2].set_ylabel('Fraction zeros (nuclei)')
    axes[2].set_title('A1: Dropout — cell vs nuclei')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '01_raw_cell_vs_nuc.png'), dpi=150)
    plt.close()

    # ---- Plot 2: Variance decomposition (raw) ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mod in zip(axes, ['cell', 'nuc']):
        within = gene_df[f'within_ct_var_{mod}']
        between = gene_df[f'between_ct_var_{mod}']
        total = within + between
        frac_within = np.where(total > 1e-10, within / total, 0)
        ax.hist(frac_within, bins=50, edgecolor='black', linewidth=0.3)
        ax.axvline(np.nanmedian(frac_within), color='red', ls='--',
                   label=f'median={np.nanmedian(frac_within):.2f}')
        ax.set_xlabel('Within-CT fraction of total variance')
        ax.set_title(f'A1: Within-CT variance fraction ({mod})')
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '02_raw_variance_decomposition.png'), dpi=150)
    plt.close()

    # ---- Plot 2b: Library size distributions ----
    lib_data = np.load(os.path.join(out_dir, 'A5_library_sizes.npz'))
    lib_cell = lib_data['lib_cell']
    lib_nuc = lib_data['lib_nuc']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Overlaid histogram
    axes[0].hist(lib_cell, bins=80, alpha=0.5, label=f'Cell (n={len(lib_cell)})', color='#2196F3', density=True)
    axes[0].hist(lib_nuc, bins=80, alpha=0.5, label=f'Nuclei (n={len(lib_nuc)})', color='#FF9800', density=True)
    axes[0].axvline(lib_cell.mean(), color='#1565C0', ls='--', lw=1, label=f'Cell mean={lib_cell.mean():.0f}')
    axes[0].axvline(lib_nuc.mean(), color='#E65100', ls='--', lw=1, label=f'Nuc mean={lib_nuc.mean():.0f}')
    axes[0].set_xlabel('Library size (sum of log1p)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('A5: Library size distributions')
    axes[0].legend(fontsize=8)

    # Ratio histogram
    axes[1].hist(lib_cell, bins=80, alpha=0.5, label='Cell', color='#2196F3')
    axes[1].hist(lib_nuc, bins=80, alpha=0.5, label='Nuclei', color='#FF9800')
    axes[1].set_xlabel('Library size (sum of log1p)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('A5: Library size (absolute counts)')
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '02b_library_size_distributions.png'), dpi=150)
    plt.close()

    # ---- Plot 2c: Mean-variance relationship per modality ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (mean_col, var_col, mod, color) in zip(axes, [
        ('mean_cell', 'var_cell', 'Cell', '#2196F3'),
        ('mean_nuc', 'var_nuc', 'Nuclei', '#FF9800'),
    ]):
        ax.scatter(gene_df[mean_col], gene_df[var_col], s=1, alpha=0.3, color=color)
        ax.set_xlabel(f'Gene mean expression (log1p)')
        ax.set_ylabel(f'Gene variance (log1p)')
        ax.set_title(f'A5: Mean-variance relationship ({mod})')
        # Add y=x line for Poisson reference
        lim = max(gene_df[mean_col].max(), gene_df[var_col].max()) * 1.05
        ax.plot([0, lim], [0, lim], 'r--', lw=0.5, label='y=x (Poisson)')
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '02c_mean_variance_relationship.png'), dpi=150)
    plt.close()

    # ---- Plot 2d: Per-CT expression heatmap (top 50 HVGs) ----
    ct_profiles = np.load(os.path.join(out_dir, 'A5_ct_mean_profiles.npz'), allow_pickle=True)
    ct_list = json.loads(str(ct_profiles['cell_types']))
    ct_gene_names = ct_profiles['gene_names']

    # Build CT-mean matrices [n_CT, G]
    cell_mat = np.array([ct_profiles[f"{ct}_cell"] for ct in ct_list])
    nuc_mat = np.array([ct_profiles[f"{ct}_nuc"] for ct in ct_list])

    # Top 50 most variable genes (by variance of CT means in cells)
    ct_mean_var = cell_mat.var(axis=0)
    top50_idx = np.argsort(ct_mean_var)[-50:][::-1]
    top50_genes = ct_gene_names[top50_idx]

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    im0 = axes[0].imshow(cell_mat[:, top50_idx], aspect='auto', cmap='viridis')
    axes[0].set_yticks(range(len(ct_list)))
    axes[0].set_yticklabels(ct_list, fontsize=7)
    axes[0].set_xticks(range(len(top50_genes)))
    axes[0].set_xticklabels(top50_genes, rotation=90, fontsize=5)
    axes[0].set_title('Cell: CT-mean expression (top 50 HVGs)')
    plt.colorbar(im0, ax=axes[0], shrink=0.5)

    im1 = axes[1].imshow(nuc_mat[:, top50_idx], aspect='auto', cmap='viridis',
                         vmin=im0.get_clim()[0], vmax=im0.get_clim()[1])
    axes[1].set_yticks(range(len(ct_list)))
    axes[1].set_yticklabels(ct_list, fontsize=7)
    axes[1].set_xticks(range(len(top50_genes)))
    axes[1].set_xticklabels(top50_genes, rotation=90, fontsize=5)
    axes[1].set_title('Nuclei: CT-mean expression (top 50 HVGs)')
    plt.colorbar(im1, ax=axes[1], shrink=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '02d_ct_expression_heatmap.png'), dpi=150)
    plt.close()

    # ---- Plot 2e: Cell-nuclei gap vs within-CT variance ----
    fig, ax = plt.subplots(figsize=(8, 6))
    gap = np.abs(gene_df['mean_cell'] - gene_df['mean_nuc'])
    within_var = gene_df['within_ct_var_cell']
    ax.scatter(gap, within_var, s=1, alpha=0.3, color='#2196F3')
    ax.set_xlabel('|Cell mean - Nuclei mean| (log1p)')
    ax.set_ylabel('Within-CT variance in cells (log1p)')
    ax.set_title('A5: Cell-nuclei gap vs within-CT variance')
    # Add correlation
    valid = (~np.isnan(gap)) & (~np.isnan(within_var)) & (within_var > 0)
    if valid.sum() > 2:
        r = np.corrcoef(gap[valid], within_var[valid])[0, 1]
        ax.annotate(f'r = {r:.3f}', xy=(0.95, 0.95), xycoords='axes fraction',
                    ha='right', va='top', fontsize=12,
                    bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '02e_gap_vs_within_ct_var.png'), dpi=150)
    plt.close()

    # ---- Plot 2f: Per-CT mean scatter (cell vs nuclei per gene) ----
    n_cts = len(ct_list)
    ncols = min(4, n_cts)
    nrows = (n_cts + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_flat = axes.flatten() if n_cts > 1 else [axes]

    for i, ct in enumerate(ct_list):
        ax = axes_flat[i]
        c_mean = ct_profiles[f"{ct}_cell"]
        n_mean = ct_profiles[f"{ct}_nuc"]
        ax.scatter(c_mean, n_mean, s=1, alpha=0.3, color='#2196F3')
        lim = max(c_mean.max(), n_mean.max()) * 1.05
        ax.plot([0, lim], [0, lim], 'r--', lw=0.5)
        # Per-CT correlation
        r = np.corrcoef(c_mean, n_mean)[0, 1]
        ax.set_title(f'{ct} (r={r:.3f})', fontsize=9)
        ax.set_xlabel('Cell mean', fontsize=7)
        ax.set_ylabel('Nuc mean', fontsize=7)
        ax.tick_params(labelsize=6)

    # Hide unused axes
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle('A5: Per-CT gene mean — cell vs nuclei', fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '02f_per_ct_mean_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ---- Plot 3: Same-modality reconstruction ----
    c1 = pd.read_csv(os.path.join(out_dir, 'C1_gene_level_recon.csv'))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for row, mod in enumerate(['cell', 'nuc']):
        # Mean: decoded vs raw
        axes[row, 0].scatter(c1[f'mean_raw_{mod}'], c1[f'mean_dec_{mod}'], s=1, alpha=0.3)
        lim = max(c1[f'mean_raw_{mod}'].max(), c1[f'mean_dec_{mod}'].max()) * 1.05
        axes[row, 0].plot([0, lim], [0, lim], 'r--', lw=0.5)
        axes[row, 0].set_xlabel(f'Raw {mod} mean')
        axes[row, 0].set_ylabel(f'Decoded {mod} mean')
        axes[row, 0].set_title(f'C1: Gene mean recon ({mod})')

        # Variance: decoded vs raw
        axes[row, 1].scatter(c1[f'var_raw_{mod}'], c1[f'var_dec_{mod}'], s=1, alpha=0.3)
        lim = max(c1[f'var_raw_{mod}'].max(), c1[f'var_dec_{mod}'].max()) * 1.05
        axes[row, 1].plot([0, lim], [0, lim], 'r--', lw=0.5)
        axes[row, 1].set_xlabel(f'Raw {mod} variance')
        axes[row, 1].set_ylabel(f'Decoded {mod} variance')
        axes[row, 1].set_title(f'C1: Gene variance recon ({mod})')

        # Var ratio histogram
        vr = c1[f'var_ratio_{mod}'].dropna()
        axes[row, 2].hist(vr.clip(0, 3), bins=50, edgecolor='black', linewidth=0.3)
        axes[row, 2].axvline(1.0, color='red', ls='--')
        axes[row, 2].axvline(vr.median(), color='blue', ls='--',
                             label=f'median={vr.median():.3f}')
        axes[row, 2].set_xlabel('Var ratio (decoded/raw)')
        axes[row, 2].set_title(f'C1: Variance ratio distribution ({mod})')
        axes[row, 2].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '03_same_modality_recon.png'), dpi=150)
    plt.close()

    # ---- Plot 4: Within-CT variance ratio (same-modality) ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mod in zip(axes, ['cell', 'nuc']):
        vr = c1[f'within_ct_var_ratio_{mod}'].dropna()
        ax.hist(vr.clip(0, 3), bins=50, edgecolor='black', linewidth=0.3)
        ax.axvline(1.0, color='red', ls='--')
        ax.axvline(vr.median(), color='blue', ls='--',
                   label=f'median={vr.median():.3f}')
        ax.set_xlabel('Within-CT var ratio (decoded/raw)')
        ax.set_title(f'C1: Within-CT variance ratio ({mod})')
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '04_within_ct_var_ratio_recon.png'), dpi=150)
    plt.close()

    # ---- Plot 5: Cross-decode gene-level ----
    d1 = pd.read_csv(os.path.join(out_dir, 'D1_gene_level_cross.csv'))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # Mean: cross vs raw cell
    axes[0].scatter(d1['mean_raw_cell'], d1['mean_cross'], s=1, alpha=0.3)
    lim = max(d1['mean_raw_cell'].max(), d1['mean_cross'].max()) * 1.05
    axes[0].plot([0, lim], [0, lim], 'r--', lw=0.5)
    axes[0].set_xlabel('Raw cell mean')
    axes[0].set_ylabel('Cross-decoded mean')
    axes[0].set_title('D1: Cross-decode gene mean vs raw cell')

    # Variance: cross vs raw cell
    axes[1].scatter(d1['var_raw_cell'], d1['var_cross'], s=1, alpha=0.3)
    lim = max(d1['var_raw_cell'].max(), d1['var_cross'].max()) * 1.05
    axes[1].plot([0, lim], [0, lim], 'r--', lw=0.5)
    axes[1].set_xlabel('Raw cell variance')
    axes[1].set_ylabel('Cross-decoded variance')
    axes[1].set_title('D1: Cross-decode gene variance vs raw cell')

    # Additional var loss (cross / same-mod)
    vr = d1['additional_var_loss'].dropna()
    axes[2].hist(vr.clip(0, 3), bins=50, edgecolor='black', linewidth=0.3)
    axes[2].axvline(1.0, color='red', ls='--')
    axes[2].axvline(vr.median(), color='blue', ls='--',
                    label=f'median={vr.median():.3f}')
    axes[2].set_xlabel('Cross var / Same-mod recon var')
    axes[2].set_title('D1: Additional variance loss at cross-decode')
    axes[2].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '05_cross_decode.png'), dpi=150)
    plt.close()

    # ---- Plot 6: Variance flow bar chart (the punchline) ----
    e2 = pd.read_csv(os.path.join(out_dir, 'E2_variance_flow_summary.csv'))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Total gene variance flow
    stages = e2['stage'].tolist()
    x = np.arange(len(stages))
    axes[0].bar(x, e2['mean_gene_var'], color=['#2196F3', '#FF9800', '#4CAF50', '#FFC107', '#F44336'])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(stages, rotation=30, ha='right')
    axes[0].set_ylabel('Mean per-gene variance')
    axes[0].set_title('E2: Total gene variance at each stage')

    # Within-CT / Between-CT stacked
    within = e2['mean_within_ct_var'].values
    between = e2['mean_between_ct_var'].values
    axes[1].bar(x, between, label='Between-CT', color='#FF9800')
    axes[1].bar(x, within, bottom=between, label='Within-CT', color='#2196F3')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(stages, rotation=30, ha='right')
    axes[1].set_ylabel('Variance')
    axes[1].set_title('E2: Within-CT vs Between-CT variance')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '06_variance_flow.png'), dpi=150)
    plt.close()

    # ---- Plot 7: Per-CT variance flow ----
    e3 = pd.read_csv(os.path.join(out_dir, 'E3_ct_variance_flow.csv'))

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(e3))
    w = 0.15
    ax.bar(x - 2 * w, e3['gene_var_raw_cell'], w, label='Raw cell', color='#2196F3')
    ax.bar(x - w, e3['gene_var_raw_nuc'], w, label='Raw nuc', color='#FF9800')
    ax.bar(x, e3['gene_var_dec_cell'], w, label='Dec cell', color='#4CAF50')
    ax.bar(x + w, e3['gene_var_dec_nuc'], w, label='Dec nuc', color='#FFC107')
    ax.bar(x + 2 * w, e3['gene_var_cross'], w, label='Cross-dec', color='#F44336')
    ax.set_xticks(x)
    ax.set_xticklabels(e3['cell_type'], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean per-gene variance (within CT)')
    ax.set_title('E3: Variance flow per cell type')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '07_ct_variance_flow.png'), dpi=150)
    plt.close()

    # ---- Plot 8: Representative gene distributions ----
    rep_genes = json.load(open(os.path.join(out_dir, 'A4_representative_genes.json')))
    raw_dist = np.load(os.path.join(out_dir, 'A4_representative_distributions.npz'), allow_pickle=True)
    dec_dist = np.load(os.path.join(out_dir, 'C4_representative_decoded.npz'), allow_pickle=True)
    cross_dist = np.load(os.path.join(out_dir, 'D3_representative_cross.npz'), allow_pickle=True)

    labels = list(rep_genes.keys())
    fig, axes = plt.subplots(len(labels), 1, figsize=(10, 4 * len(labels)))
    if len(labels) == 1:
        axes = [axes]

    for i, label in enumerate(labels):
        ax = axes[i]
        gname = rep_genes[label]
        bins = 50

        # Raw distributions
        raw_c = raw_dist[f"{label}_cell"]
        raw_n = raw_dist[f"{label}_nuc"]
        ax.hist(raw_c, bins=bins, alpha=0.4, label='Raw cell', color='#2196F3', density=True)
        ax.hist(raw_n, bins=bins, alpha=0.4, label='Raw nuc', color='#FF9800', density=True)

        # Decoded cell (same-modality)
        d_c = dec_dist[f"{label}_dec_cell"]
        ax.hist(d_c, bins=bins, alpha=0.4, label='Dec cell', color='#4CAF50',
                density=True, histtype='step', linewidth=2)

        # Cross-decoded
        xd = cross_dist[f"{label}_cross"]
        ax.hist(xd, bins=bins, alpha=0.4, label='Cross-dec', color='#F44336',
                density=True, histtype='step', linewidth=2)

        ax.set_title(f'{label}: {gname}')
        ax.legend(fontsize=8)
        ax.set_xlabel('Expression (log1p)')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '08_representative_distributions.png'), dpi=150)
    plt.close()

    # ---- Plot 9: Latent per-dim statistics ----
    dim_df = pd.read_csv(os.path.join(out_dir, 'B1_per_dim.csv'))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    L = len(dim_df)
    x = np.arange(L)

    # Mean per dim
    axes[0, 0].bar(x - 0.2, dim_df['mean_cell'], 0.4, label='Cell', color='#2196F3')
    axes[0, 0].bar(x + 0.2, dim_df['mean_nuc'], 0.4, label='Nuclei', color='#FF9800')
    axes[0, 0].set_xlabel('Latent dimension')
    axes[0, 0].set_title('B1: Latent mean per dim')
    axes[0, 0].legend()

    # Variance per dim
    axes[0, 1].bar(x - 0.2, dim_df['var_cell'], 0.4, label='Cell', color='#2196F3')
    axes[0, 1].bar(x + 0.2, dim_df['var_nuc'], 0.4, label='Nuclei', color='#FF9800')
    axes[0, 1].set_xlabel('Latent dimension')
    axes[0, 1].set_title('B1: Latent variance per dim')
    axes[0, 1].legend()

    # Inter-modality fraction per dim
    axes[1, 0].bar(x, dim_df['inter_modality_frac'], color='#9C27B0')
    axes[1, 0].axhline(0.5, color='red', ls='--', lw=0.5)
    axes[1, 0].set_xlabel('Latent dimension')
    axes[1, 0].set_title('B1: Inter-modality variance fraction per dim')

    # Posterior std per dim
    axes[1, 1].bar(x - 0.2, dim_df['std_posterior_cell'], 0.4, label='Cell', color='#2196F3')
    axes[1, 1].bar(x + 0.2, dim_df['std_posterior_nuc'], 0.4, label='Nuclei', color='#FF9800')
    axes[1, 1].set_xlabel('Latent dimension')
    axes[1, 1].set_title('B1: Posterior std per dim')
    axes[1, 1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '09_latent_per_dim.png'), dpi=150)
    plt.close()

    # ---- Plot 10: Gene-level var ratio vs expression level ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (vr_col, mean_col, title) in zip(axes, [
        ('var_ratio_cell', 'mean_raw_cell', 'Same-mod cell recon'),
        ('var_ratio_vs_cell', 'mean_raw_cell', 'Cross-decode vs raw cell'),
    ]):
        source = c1 if 'var_ratio_cell' in vr_col else d1
        vr = source[vr_col].clip(0, 5)
        mn = source[mean_col] if mean_col in source.columns else c1[mean_col]
        ax.scatter(mn, vr, s=1, alpha=0.3)
        ax.axhline(1.0, color='red', ls='--', lw=0.5)
        ax.set_xlabel('Gene mean expression (log1p)')
        ax.set_ylabel('Variance ratio (output/raw)')
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, '10_var_ratio_vs_expression.png'), dpi=150)
    plt.close()

    print(f"  Saved 10 plots to {plot_dir}/")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='End-to-end data flow audit')
    parser.add_argument('--model', type=str, default='pct5v2_cond',
                        help=f'Model name from registry ({", ".join(list_models())})')
    parser.add_argument('--out', type=str, default='audit_results',
                        help='Output directory')
    parser.add_argument('--max-per-type', type=int, default=2000,
                        help='Max cells per type (subsampling)')
    parser.add_argument('--skip-plots', action='store_true',
                        help='Skip plot generation')
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()

    # Resolve output dir relative to project root
    out_dir = os.path.join(_PROJECT_DIR, args.out, args.model)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Data flow audit: model={args.model}, out={out_dir}")

    # Load model config and data
    model_config = get_model(args.model)
    print(f"  Run folder: {model_config['run_folder']}")
    print(f"  Gene list: {os.path.basename(model_config['gene_list'])}")

    data = load_eval_data(model_config['gene_list'], max_per_type=args.max_per_type)

    # Load model (needs GPU)
    model, device = load_model(model_config)
    print(f"  Device: {device}")

    # Run all sections
    gene_df, ct_df, rep_genes = section_a_raw(data, out_dir)
    mu_cell, mu_nuc = section_b_latent(model, data, device, out_dir)
    dec_cell, dec_nuc = section_c_decoded(model, data, device, out_dir, gene_df, rep_genes)

    if model_config['has_cross_decode']:
        cross_dec = section_d_cross(model, data, device, out_dir, dec_cell, gene_df, rep_genes)
    else:
        print("\n=== Section D: Skipped (no cross-decode for shared model) ===")
        cross_dec = dec_nuc  # fallback

    section_e_variance_flow(data, dec_cell, dec_nuc, cross_dec, mu_cell, mu_nuc, out_dir)

    # Cleanup GPU
    del model
    torch.cuda.empty_cache()

    # Plots
    if not args.skip_plots:
        make_plots(out_dir)

    print(f"\nAudit complete. Results in {out_dir}/")


if __name__ == '__main__':
    main()
