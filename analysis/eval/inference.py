"""
Model loading and 5-source inference.

Produces a standardized dict with all 5 expression sources:
  raw_cell, raw_nuclei, decoded_cell, decoded_nuclei, cross_decoded

Baseline models (no trained weights) produce imputed output directly from data.
"""

import gc
import json
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import Ridge

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_DIR)

from imputeVAE.models import VAE_JSD
from imputeVAE.models_cond import VAE_JSD_Cond

CELL_ID = 0
NUCLEI_ID = 1


def load_model(model_config, device=None):
    """Load a trained model from its registry config.

    Parameters
    ----------
    model_config : dict
        Entry from registry.MODELS.
    device : torch.device or None
        If None, auto-detect GPU.

    Returns
    -------
    model : nn.Module
        Loaded model in eval mode.
    device : torch.device
    """
    if model_config['model_class'] == 'baseline':
        return None, torch.device('cpu')

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    run_folder = model_config['run_folder']
    args_path = os.path.join(run_folder, 'arguments.json')
    model_path = os.path.join(run_folder, 'models', 'model.pt')

    with open(args_path) as f:
        args = json.load(f)

    num_feature = args['num_feature']
    num_internal = args['num_internal']
    num_latent = args['num_latent']

    if model_config['model_class'] == 'VAE_JSD_Cond':
        # Use model_kwargs from registry if present, else fall back to saved args
        kwargs = model_config.get('model_kwargs', {})
        embed_dim = kwargs.get('embed_dim', args.get('embed_dim', 8))
        scalar_cond = kwargs.get('scalar_cond', args.get('scalar_cond', False))
        cond_encoder = kwargs.get('cond_encoder', args.get('cond_encoder', False))

        model = VAE_JSD_Cond(
            input_size=num_feature,
            num_internal=num_internal,
            num_latent=num_latent,
            embed_dim=embed_dim,
            scalar_cond=scalar_cond,
            cond_encoder=cond_encoder,
        )
    else:
        model = VAE_JSD(
            input_size=num_feature,
            num_internal=num_internal,
            num_latent=num_latent,
        )

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model, device


def _run_baseline_inference(model_config, X_cell, X_nuc, ct_cell, ct_nuc):
    """Compute baseline 'imputed' output without a trained model.

    Parameters
    ----------
    model_config : dict
    X_cell : np.ndarray [N_cell, G]
    X_nuc : np.ndarray [N_nuc, G]
    ct_cell : np.ndarray [N_cell] str
    ct_nuc : np.ndarray [N_nuc] str

    Returns
    -------
    dict with standard 5-source keys
    """
    baseline_type = model_config['baseline_type']
    common_cts = sorted(set(ct_cell) & set(ct_nuc))

    if baseline_type == 'raw_nuclei':
        # "Imputed" = raw nuclei unchanged
        imputed = X_nuc.copy()

    elif baseline_type == 'ct_mean':
        # Replace each nucleus with its cell-type mean from raw CELL data
        cell_ct_means = {}
        for ct in common_cts:
            mask = ct_cell == ct
            if mask.sum() > 0:
                cell_ct_means[ct] = X_cell[mask].mean(axis=0)

        imputed = np.zeros_like(X_nuc)
        for i, ct in enumerate(ct_nuc):
            if ct in cell_ct_means:
                imputed[i] = cell_ct_means[ct]
            else:
                imputed[i] = X_nuc[i]  # fallback for unknown CT

    elif baseline_type == 'linear_regression':
        # Per-gene Ridge regression trained within each cell type
        # Train: for each CT, learn gene_cell = f(gene_nuc) using paired CT means
        # Apply: for each nucleus, predict cell expression gene-by-gene
        imputed = _linear_regression_baseline(X_cell, X_nuc, ct_cell, ct_nuc, common_cts)

    else:
        raise ValueError(f"Unknown baseline_type: {baseline_type}")

    return {
        'raw_cell': X_cell,
        'raw_nuclei': X_nuc,
        'decoded_cell': X_cell.copy(),  # no model, passthrough
        'decoded_nuclei': X_nuc.copy(),  # no model, passthrough
        'cross_decoded': imputed,
        'latent_cell': None,
        'latent_nuc': None,
    }


def _linear_regression_baseline(X_cell, X_nuc, ct_cell, ct_nuc, common_cts):
    """Per-CT per-gene linear regression: predict cell expression from nuclei.

    For each cell type, trains a Ridge regression model across cells:
        For each gene g: cell_expression[g] ~ nuclei_expression[g]

    Since we don't have paired cells/nuclei, we use a cross-CT approach:
    Train on CT means (each CT provides one data point per gene), then
    apply to individual nuclei.

    Actually, the more useful approach: for each gene, fit a simple linear
    model across ALL cell types (CT mean nuclei -> CT mean cell), then
    apply to individual nuclei. This is equivalent to learning a per-gene
    scaling/shift that maps nuclei expression to cell expression.
    """
    n_genes = X_cell.shape[1]

    # Compute CT means
    cell_means = np.zeros((len(common_cts), n_genes), dtype=np.float32)
    nuc_means = np.zeros((len(common_cts), n_genes), dtype=np.float32)
    for i, ct in enumerate(common_cts):
        cell_mask = ct_cell == ct
        nuc_mask = ct_nuc == ct
        if cell_mask.sum() > 0:
            cell_means[i] = X_cell[cell_mask].mean(axis=0)
        if nuc_mask.sum() > 0:
            nuc_means[i] = X_nuc[nuc_mask].mean(axis=0)

    # Per-gene linear regression: cell_mean[g] = a[g] * nuc_mean[g] + b[g]
    # Fit across cell types
    slopes = np.ones(n_genes, dtype=np.float32)
    intercepts = np.zeros(n_genes, dtype=np.float32)

    for g in range(n_genes):
        x = nuc_means[:, g]
        y = cell_means[:, g]
        if np.std(x) > 1e-8:
            # Simple OLS: y = ax + b
            x_mean = x.mean()
            y_mean = y.mean()
            slope = np.sum((x - x_mean) * (y - y_mean)) / (np.sum((x - x_mean)**2) + 1e-10)
            intercept = y_mean - slope * x_mean
            slopes[g] = slope
            intercepts[g] = intercept
        else:
            # Gene has no variance across CTs in nuclei — use cell mean
            slopes[g] = 0.0
            intercepts[g] = cell_means[:, g].mean()

    # Apply to individual nuclei
    imputed = X_nuc * slopes[np.newaxis, :] + intercepts[np.newaxis, :]
    return imputed.astype(np.float32)


def run_5source_inference(model, model_config, X_cell, X_nuc, device,
                          batch_size=256, ct_cell=None, ct_nuc=None):
    """Run model inference to produce all 5 expression sources.

    Parameters
    ----------
    model : nn.Module or None (for baselines)
    model_config : dict
    X_cell : np.ndarray [N_cell, G], log1p
    X_nuc : np.ndarray [N_nuc, G], log1p
    device : torch.device
    batch_size : int
    ct_cell : np.ndarray [N_cell] str (required for baselines)
    ct_nuc : np.ndarray [N_nuc] str (required for baselines)

    Returns
    -------
    dict with keys:
        'raw_cell': X_cell (passthrough)
        'raw_nuclei': X_nuc (passthrough)
        'decoded_cell': cell reconstruction [N_cell, G]
        'decoded_nuclei': nuclei reconstruction [N_nuc, G]
        'cross_decoded': nuclei-to-cell imputation [N_nuc, G] (None for shared models)
        'latent_nuc': nuclei latent means [N_nuc, latent_dim] (for hallucination tests)
    """
    # Handle baselines
    if model_config['model_class'] == 'baseline':
        if ct_cell is None or ct_nuc is None:
            raise ValueError("Baselines require ct_cell and ct_nuc arrays")
        return _run_baseline_inference(model_config, X_cell, X_nuc, ct_cell, ct_nuc)

    has_cross = model_config['has_cross_decode']

    def _run_batched(X, modality_id, do_cross=False):
        """Run forward + optional cross-decode in batches."""
        X_t = torch.tensor(X, dtype=torch.float32)
        recon_parts, cross_parts, latent_parts = [], [], []

        with torch.no_grad():
            for i in range(0, X_t.shape[0], batch_size):
                batch = X_t[i:i+batch_size].to(device)

                if has_cross:
                    # Conditioned model: manual encode + decode with embedding
                    if getattr(model, 'cond_encoder', False):
                        source_emb = model._get_modality_emb(
                            batch.shape[0], modality_id, device)
                        encoder_input = torch.cat([batch, source_emb], dim=1)
                    else:
                        encoder_input = batch
                    encoded = model.encoder(encoder_input)
                    mu = model.mu(encoded)
                    emb = model._get_modality_emb(batch.shape[0], modality_id, device)
                    z_cond = torch.cat([mu, emb], dim=1)
                    decoded = model.decoder(z_cond)
                    recon_parts.append(decoded.cpu().numpy())
                    latent_parts.append(mu.cpu().numpy())

                    if do_cross:
                        target_id = CELL_ID if modality_id == NUCLEI_ID else NUCLEI_ID
                        cross_dec = model.decode_cross(
                            batch, source_modality_id=modality_id,
                            target_modality_id=target_id
                        )
                        cross_parts.append(cross_dec.cpu().numpy())
                else:
                    _, decoded, mu, _ = model(batch)
                    recon_parts.append(decoded.cpu().numpy())
                    latent_parts.append(mu.cpu().numpy())

        recon = np.concatenate(recon_parts, axis=0)
        latent = np.concatenate(latent_parts, axis=0)
        cross = np.concatenate(cross_parts, axis=0) if cross_parts else None
        return recon, cross, latent

    # Cell reconstruction
    decoded_cell, _, latent_cell = _run_batched(X_cell, CELL_ID, do_cross=False)

    # Nuclei reconstruction + cross-decode
    decoded_nuc, cross_decoded, latent_nuc = _run_batched(
        X_nuc, NUCLEI_ID, do_cross=has_cross
    )

    sources = {
        'raw_cell': X_cell,
        'raw_nuclei': X_nuc,
        'decoded_cell': decoded_cell,
        'decoded_nuclei': decoded_nuc,
        'cross_decoded': cross_decoded,
        'latent_cell': latent_cell,
        'latent_nuc': latent_nuc,
    }

    # For shared models, the "imputed" output is just decoded_nuclei
    if not has_cross:
        sources['cross_decoded'] = decoded_nuc

    return sources


def get_imputed(sources, model_config):
    """Return the imputation output: cross_decoded for cond, decoded_nuclei for shared."""
    if model_config['has_cross_decode']:
        return sources['cross_decoded']
    return sources['decoded_nuclei']


def cleanup_model(model):
    """Delete model and free GPU memory."""
    if model is not None:
        del model
        torch.cuda.empty_cache()
    gc.collect()
