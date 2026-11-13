"""
Latent space metrics for hallucination detection.

Metrics:
  - z_ablation: replace per-cell z with CT mean z, measure imputation change
  - z_variance_decomposition: fraction of z variance between vs within cell types
"""

import numpy as np
import torch
from scipy import stats


def z_ablation(model, latent_nuc, ct_nuc, common_cts, device, model_config):
    """z ablation test: replace each cell's z with its CT mean, measure change.

    If imputed output barely changes when z is replaced with CT mean,
    the model is ignoring per-cell variation (hallucinating).

    Parameters
    ----------
    model : nn.Module (VAE_JSD_Cond only)
    latent_nuc : [N, latent_dim] nuclei latent means
    ct_nuc : [N] cell type labels
    common_cts : list of str
    device : torch.device
    model_config : dict

    Returns
    -------
    per_ct : dict, ct -> {corr_original_ablated, mse_change}
    summary : dict with mean_corr, mean_mse_change
    """
    if not model_config.get('has_cross_decode', False):
        return {}, {'mean_corr': None, 'mean_mse_change': None}

    CELL_ID = 0

    # Compute CT mean z
    z_ablated = latent_nuc.copy()
    for ct in common_cts:
        mask = ct_nuc == ct
        if mask.sum() > 0:
            z_ablated[mask] = latent_nuc[mask].mean(axis=0)

    # Decode both original and ablated with cell embedding
    def _decode_with_cell_emb(z_array):
        z_t = torch.tensor(z_array, dtype=torch.float32).to(device)
        parts = []
        with torch.no_grad():
            for i in range(0, z_t.shape[0], 256):
                batch_z = z_t[i:i+256]
                emb = model._get_modality_emb(batch_z.shape[0], CELL_ID, device)
                z_cond = torch.cat([batch_z, emb], dim=1)
                decoded = model.decoder(z_cond)
                parts.append(decoded.cpu().numpy())
        return np.concatenate(parts, axis=0)

    original_decoded = _decode_with_cell_emb(latent_nuc)
    ablated_decoded = _decode_with_cell_emb(z_ablated)

    per_ct = {}
    all_corrs, all_mse = [], []

    for ct in common_cts:
        mask = ct_nuc == ct
        if mask.sum() < 5:
            continue

        orig_ct = original_decoded[mask]
        abl_ct = ablated_decoded[mask]

        # Per-cell correlation between original and ablated
        ct_corrs = []
        for i in range(orig_ct.shape[0]):
            if np.std(orig_ct[i]) > 0 and np.std(abl_ct[i]) > 0:
                r, _ = stats.pearsonr(orig_ct[i], abl_ct[i])
                ct_corrs.append(r)

        mse = float(np.mean((orig_ct - abl_ct)**2))

        per_ct[ct] = {
            'corr_original_ablated': float(np.median(ct_corrs)) if ct_corrs else 1.0,
            'mse_change': mse,
        }
        all_corrs.extend(ct_corrs)
        all_mse.append(mse)

    summary = {
        'mean_corr': float(np.mean(all_corrs)) if all_corrs else 1.0,
        'mean_mse_change': float(np.mean(all_mse)) if all_mse else 0.0,
    }
    return per_ct, summary


def z_variance_decomposition(latent_nuc, ct_nuc, common_cts):
    """Variance decomposition: fraction of z variance between cell types.

    High between-fraction means z encodes mostly cell type identity.
    Low between-fraction means z encodes per-cell variation too.

    Parameters
    ----------
    latent_nuc : [N, latent_dim]
    ct_nuc : [N]
    common_cts : list of str

    Returns
    -------
    per_dim : [latent_dim] fraction of variance between CTs
    summary : dict with mean_between_frac, median_between_frac
    """
    latent_dim = latent_nuc.shape[1]
    grand_mean = latent_nuc.mean(axis=0)

    ss_between = np.zeros(latent_dim)
    ss_total = np.zeros(latent_dim)

    for ct in common_cts:
        mask = ct_nuc == ct
        n_ct = mask.sum()
        if n_ct == 0:
            continue
        ct_mean = latent_nuc[mask].mean(axis=0)
        ss_between += n_ct * (ct_mean - grand_mean)**2

    for d in range(latent_dim):
        ss_total[d] = np.sum((latent_nuc[:, d] - grand_mean[d])**2)

    eps = 1e-10
    per_dim = ss_between / (ss_total + eps)

    summary = {
        'mean_between_frac': float(np.mean(per_dim)),
        'median_between_frac': float(np.median(per_dim)),
    }
    return per_dim, summary


def run_all_latent(sources, eval_data, model, model_config, device):
    """Run all latent-space metrics.

    Parameters
    ----------
    sources : dict from inference
    eval_data : EvalData
    model : nn.Module (needed for z ablation decode)
    model_config : dict
    device : torch.device

    Returns
    -------
    dict of results
    """
    latent_nuc = sources.get('latent_nuc')
    ct_nuc = eval_data.ct_nuc
    common_cts = eval_data.common_cts

    results = {}

    if latent_nuc is None:
        return results

    # z variance decomposition
    print("    Computing z variance decomposition...")
    per_dim, zvar_summary = z_variance_decomposition(latent_nuc, ct_nuc, common_cts)
    results['z_variance_decomp'] = zvar_summary
    results['z_variance_per_dim'] = per_dim

    # z ablation (requires model, only for cond models)
    if model_config.get('has_cross_decode', False):
        print("    Computing z ablation...")
        abl_per_ct, abl_summary = z_ablation(
            model, latent_nuc, ct_nuc, common_cts, device, model_config
        )
        results['z_ablation'] = abl_summary
        results['z_ablation_per_ct'] = abl_per_ct

    return results
