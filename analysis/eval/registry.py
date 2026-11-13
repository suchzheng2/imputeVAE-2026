"""
Model registry: single source of truth for all model configurations.

Each entry defines where the model lives, what class it uses,
and what gene list it was trained with.

Baseline entries (model_class='baseline') have no run_folder or model —
their 'imputed' output is computed directly from data in inference.py.
"""

import os

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Gene list paths (relative to project root)
GENE_LIST_CONFIG_B = os.path.join(_PROJECT_DIR, 'data', 'training_gene_list_v3.csv')
GENE_LIST_PCT5 = os.path.join(_PROJECT_DIR, 'data', 'gene_list_perct_5pct.csv')

MODELS = {
    # ===================================================================
    # BASELINES (no trained model)
    # ===================================================================
    'baseline_raw_nuclei': {
        'run_folder': None,
        'model_class': 'baseline',
        'baseline_type': 'raw_nuclei',
        'has_cross_decode': False,
        'gene_list': GENE_LIST_CONFIG_B,
        'normalization': 'log1p',
        'description': 'Raw nuclei (no imputation) — lower bound',
    },
    'baseline_ct_mean': {
        'run_folder': None,
        'model_class': 'baseline',
        'baseline_type': 'ct_mean',
        'has_cross_decode': True,
        'gene_list': GENE_LIST_CONFIG_B,
        'normalization': 'log1p',
        'description': 'CT-mean imputation (replace each nucleus with cell CT mean) — hallucination ceiling',
    },
    'baseline_linear_reg': {
        'run_folder': None,
        'model_class': 'baseline',
        'baseline_type': 'linear_regression',
        'has_cross_decode': True,
        'gene_list': GENE_LIST_CONFIG_B,
        'normalization': 'log1p',
        'description': 'Per-gene linear regression (nuclei -> cell, trained per CT)',
    },

    # ===================================================================
    # Config B (10,134 genes) — paper main models
    # ===================================================================
    'configB_shared': {
        'run_folder': os.path.join(_PROJECT_DIR, 'run', 'run_2026-04-01-14-01-16_1946018'),
        'model_class': 'VAE_JSD',
        'has_cross_decode': False,
        'gene_list': GENE_LIST_CONFIG_B,
        'normalization': 'log1p',
        'description': 'Config B, shared decoder (no modality conditioning)',
    },
    'configB_cond': {
        'run_folder': os.path.join(_PROJECT_DIR, 'run', 'run_2026-04-02-02-02-35_3345839'),
        'model_class': 'VAE_JSD_Cond',
        'has_cross_decode': True,
        'gene_list': GENE_LIST_CONFIG_B,
        'normalization': 'log1p',
        'description': 'Config B, learned 8d embed, dec-only (no cross-recons)',
    },
    'configB_cond_xr': {
        'run_folder': os.path.join(_PROJECT_DIR, 'run', 'run_2026-04-02-10-51-32_3738876'),
        'model_class': 'VAE_JSD_Cond',
        'has_cross_decode': True,
        'gene_list': GENE_LIST_CONFIG_B,
        'normalization': 'log1p',
        'description': 'Config B, learned 8d embed, dec-only + cross-recons',
    },

    # ===================================================================
    # Config B — conditioning experiment (May 2026, normloss exploration)
    # ===================================================================
    'configB_cond_normloss': {
        'run_folder': os.path.join(_PROJECT_DIR, 'run', 'run_2026-05-25-17-12-57_168455'),
        'model_class': 'VAE_JSD_Cond',
        'has_cross_decode': True,
        'gene_list': GENE_LIST_CONFIG_B,
        'normalization': 'log1p',
        'description': 'Config B, learned 8d embed, dec-only, normloss fixed',
    },
    'configB_cond_no_normloss': {
        'run_folder': os.path.join(_PROJECT_DIR, 'run', 'run_2026-05-08-05-21-27_3694365'),
        'model_class': 'VAE_JSD_Cond',
        'has_cross_decode': True,
        'gene_list': GENE_LIST_CONFIG_B,
        'normalization': 'log1p',
        'description': 'Config B, learned 8d embed, dec-only, no normloss',
    },
    'configB_single_stage_cVAE': {
        'run_folder': os.path.join(_PROJECT_DIR, 'run', 'run_2026-05-11-19-52-25_134850'),
        'model_class': 'VAE_JSD_Cond',
        'model_kwargs': {'scalar_cond': True, 'cond_encoder': True, 'embed_dim': 1},
        'has_cross_decode': True,
        'gene_list': GENE_LIST_CONFIG_B,
        'normalization': 'log1p',
        'description': 'Config B, scalar cond, enc+dec, single-stage from scratch',
    },
    'configB_2stage_scalar_enc_dec': {
        'run_folder': os.path.join(_PROJECT_DIR, 'run', 'run_2026-05-11-06-59-55_3983979'),
        'model_class': 'VAE_JSD_Cond',
        'model_kwargs': {'scalar_cond': True, 'cond_encoder': True, 'embed_dim': 1},
        'has_cross_decode': True,
        'gene_list': GENE_LIST_CONFIG_B,
        'normalization': 'log1p',
        'description': 'Config B, scalar cond, enc+dec, 2-stage pretrained init',
    },
    'configB_2stage_scalar_dec': {
        'run_folder': os.path.join(_PROJECT_DIR, 'run', 'run_2026-05-11-06-02-02_625347'),
        'model_class': 'VAE_JSD_Cond',
        'model_kwargs': {'scalar_cond': True, 'cond_encoder': False, 'embed_dim': 1},
        'has_cross_decode': True,
        'gene_list': GENE_LIST_CONFIG_B,
        'normalization': 'log1p',
        'description': 'Config B, scalar cond, dec-only, 2-stage pretrained init',
    },

    # ===================================================================
    # Config B — CPM normalization
    # ===================================================================
    'configB_cpm_cond': {
        'run_folder': os.path.join(_PROJECT_DIR, 'run', 'run_2026-05-11-04-43-20_620735'),
        'model_class': 'VAE_JSD_Cond',
        'has_cross_decode': True,
        'gene_list': GENE_LIST_CONFIG_B,
        'normalization': 'cpm',
        'description': 'Config B, CPM, learned 8d embed, dec-only',
    },

    # ===================================================================
    # Normloss-fixed systematic sweep (May 28-29, 2026)
    # All single-stage from scratch, 600 epochs, normloss actually working
    # 2x4 factorial: {log1p, CPM} x {learned_8d dec, scalar dec, scalar cVAE, learned_8d cVAE}
    # ===================================================================

    # -- log1p variants --
    'sweep_log1p_8d_dec': {
        'run_folder': os.path.join(_PROJECT_DIR, 'run', 'run_2026-05-28-14-48-36_278184'),
        'model_class': 'VAE_JSD_Cond',
        'has_cross_decode': True,
        'gene_list': GENE_LIST_CONFIG_B,
        'normalization': 'log1p',
        'description': 'Sweep: log1p, learned 8d embed, dec-only, from scratch',
    },
    'sweep_log1p_scalar_dec': {
        'run_folder': os.path.join(_PROJECT_DIR, 'run', 'run_2026-05-28-15-15-37_31917'),
        'model_class': 'VAE_JSD_Cond',
        'model_kwargs': {'scalar_cond': True, 'cond_encoder': False, 'embed_dim': 1},
        'has_cross_decode': True,
        'gene_list': GENE_LIST_CONFIG_B,
        'normalization': 'log1p',
        'description': 'Sweep: log1p, scalar cond, dec-only, from scratch',
    },
    'sweep_log1p_scalar_cvae': {
        'run_folder': os.path.join(_PROJECT_DIR, 'run', 'run_2026-05-28-15-15-37_1836209'),
        'model_class': 'VAE_JSD_Cond',
        'model_kwargs': {'scalar_cond': True, 'cond_encoder': True, 'embed_dim': 1},
        'has_cross_decode': True,
        'gene_list': GENE_LIST_CONFIG_B,
        'normalization': 'log1p',
        'description': 'Sweep: log1p, scalar cond, enc+dec (cVAE), from scratch',
    },
    'sweep_log1p_8d_cvae': {
        'run_folder': os.path.join(_PROJECT_DIR, 'run', 'run_2026-05-28-21-37-31_3423409'),
        'model_class': 'VAE_JSD_Cond',
        'model_kwargs': {'cond_encoder': True},
        'has_cross_decode': True,
        'gene_list': GENE_LIST_CONFIG_B,
        'normalization': 'log1p',
        'description': 'Sweep: log1p, learned 8d embed, enc+dec (cVAE), from scratch',
    },

    # -- CPM variants --
    'sweep_cpm_8d_dec': {
        'run_folder': os.path.join(_PROJECT_DIR, 'run', 'run_2026-05-28-15-15-38_1204566'),
        'model_class': 'VAE_JSD_Cond',
        'has_cross_decode': True,
        'gene_list': GENE_LIST_CONFIG_B,
        'normalization': 'cpm',
        'description': 'Sweep: CPM, learned 8d embed, dec-only, from scratch',
    },
    'sweep_cpm_scalar_dec': {
        'run_folder': os.path.join(_PROJECT_DIR, 'run', 'run_2026-05-28-15-23-50_1234766'),
        'model_class': 'VAE_JSD_Cond',
        'model_kwargs': {'scalar_cond': True, 'cond_encoder': False, 'embed_dim': 1},
        'has_cross_decode': True,
        'gene_list': GENE_LIST_CONFIG_B,
        'normalization': 'cpm',
        'description': 'Sweep: CPM, scalar cond, dec-only, from scratch',
    },
    'sweep_cpm_scalar_cvae': {
        'run_folder': os.path.join(_PROJECT_DIR, 'run', 'run_2026-05-28-17-30-21_1880162'),
        'model_class': 'VAE_JSD_Cond',
        'model_kwargs': {'scalar_cond': True, 'cond_encoder': True, 'embed_dim': 1},
        'has_cross_decode': True,
        'gene_list': GENE_LIST_CONFIG_B,
        'normalization': 'cpm',
        'description': 'Sweep: CPM, scalar cond, enc+dec (cVAE), from scratch',
    },
    'sweep_cpm_8d_cvae': {
        'run_folder': os.path.join(_PROJECT_DIR, 'run', 'run_2026-05-29-03-10-28_3375023'),
        'model_class': 'VAE_JSD_Cond',
        'model_kwargs': {'cond_encoder': True},
        'has_cross_decode': True,
        'gene_list': GENE_LIST_CONFIG_B,
        'normalization': 'cpm',
        'description': 'Sweep: CPM, learned 8d embed, enc+dec (cVAE), from scratch',
    },

    # ===================================================================
    # PerCT 5% (16,439 genes) — older gene list
    # ===================================================================
    'pct5v2_cond': {
        'run_folder': os.path.join(_PROJECT_DIR, 'run', 'run_2026-04-07-02-12-05_3779214'),
        'model_class': 'VAE_JSD_Cond',
        'has_cross_decode': True,
        'gene_list': GENE_LIST_PCT5,
        'normalization': 'log1p',
        'description': 'PerCT 5% v2, conditioned decoder (no cross-recons)',
    },
    'pct5v2_cond_xr': {
        'run_folder': os.path.join(_PROJECT_DIR, 'run', 'run_2026-04-07-02-12-05_3551465'),
        'model_class': 'VAE_JSD_Cond',
        'has_cross_decode': True,
        'gene_list': GENE_LIST_PCT5,
        'normalization': 'log1p',
        'description': 'PerCT 5% v2, conditioned decoder + cross-recons',
    },
}


def get_model(name):
    """Get model config by name. Raises KeyError if not found."""
    if name not in MODELS:
        available = ', '.join(sorted(MODELS.keys()))
        raise KeyError(f"Unknown model '{name}'. Available: {available}")
    return MODELS[name]


def list_models():
    """Return list of all registered model names."""
    return sorted(MODELS.keys())


def models_by_gene_list():
    """Group models by gene list path."""
    groups = {}
    for name, cfg in MODELS.items():
        gl = cfg['gene_list']
        groups.setdefault(gl, []).append(name)
    return groups


def paper_models():
    """Return the subset of models intended for the main paper comparison."""
    return [
        'baseline_raw_nuclei',
        'baseline_ct_mean',
        'baseline_linear_reg',
        'configB_shared',
        'configB_cond',
        'configB_cond_xr',
        'configB_cond_normloss',
        'configB_single_stage_cVAE',
    ]


def sweep_models():
    """Return the normloss-fixed systematic sweep models (2x4 factorial)."""
    return [
        'sweep_log1p_8d_dec',
        'sweep_log1p_scalar_dec',
        'sweep_log1p_scalar_cvae',
        'sweep_log1p_8d_cvae',
        'sweep_cpm_8d_dec',
        'sweep_cpm_scalar_dec',
        'sweep_cpm_scalar_cvae',
        'sweep_cpm_8d_cvae',
    ]
