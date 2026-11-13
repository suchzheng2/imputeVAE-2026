"""Quick within-CT variance ratio check — loads data once, tests multiple models.

Usage:
    python analysis/quick_variance_check.py <run_folder1> [run_folder2] ...
"""
import sys
import os
import json
import gc
import numpy as np
import torch
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

run_folders = sys.argv[1:]
if not run_folders:
    print("Usage: python analysis/quick_variance_check.py <run_folder1> [run_folder2] ...")
    sys.exit(1)

# Use first run's args to load data (assume same gene list / norm across runs)
with open(f"{run_folders[0]}/arguments.json") as f:
    first_args = json.load(f)

norm = first_args.get('raw_data_normalization', 'log1p')
use_raw = first_args.get('use_raw_data', False)

# Load data once, filtering by protocol to match training data
from imputeVAE.data import load_raw_data, filter_by_protocol
args_ns = SimpleNamespace(**first_args)

# Derive cell batch from nuclei batch (n3A -> c3A)
nuc_batch = first_args.get('batch', 'n3A')
cell_batch = nuc_batch.replace('n', 'c', 1)

print("Loading cell data...")
adata_cell = load_raw_data("10X_cells_v2_v3_AIBS/", args_ns)
cell_args = SimpleNamespace(**{**first_args, 'batch': cell_batch})
adata_cell = filter_by_protocol(adata_cell, cell_args)
print(f"Cell: {adata_cell.shape}")

print("Loading nuclei data...")
adata_nuc = load_raw_data("10X_nuclei_v2_v3_AIBS/", args_ns)
adata_nuc = filter_by_protocol(adata_nuc, args_ns)
print(f"Nuclei: {adata_nuc.shape}")

# Get expression
layer = norm if use_raw and norm and norm != 'raw' else None
if layer and layer in adata_cell.layers:
    X_cell = np.array(adata_cell.layers[layer], dtype=np.float32)
    X_nuc = np.array(adata_nuc.layers[layer], dtype=np.float32)
else:
    X_cell = adata_cell.X
    X_nuc = adata_nuc.X
    if hasattr(X_cell, 'toarray'):
        X_cell = X_cell.toarray()
    if hasattr(X_nuc, 'toarray'):
        X_nuc = X_nuc.toarray()
    X_cell = np.array(X_cell, dtype=np.float32)
    X_nuc = np.array(X_nuc, dtype=np.float32)

ct_cell = np.array(adata_cell.obs['cell_type'].values)
ct_nuc = np.array(adata_nuc.obs['cell_type'].values)
common_cts = sorted(set(ct_nuc) & set(ct_cell))

# Pre-compute raw cell within-CT variance
raw_cell_var = {}
for ct in common_cts:
    mask = ct_cell == ct
    if mask.sum() >= 5:
        raw_cell_var[ct] = X_cell[mask].var(axis=0).mean()

del adata_cell, adata_nuc
gc.collect()

print(f"\nRaw cell within-CT variance computed for {len(raw_cell_var)} cell types")
print(f"X_nuc shape: {X_nuc.shape}, X_cell shape: {X_cell.shape}")


def run_variance_for_model(run_folder):
    """Load model, cross-decode, compute variance ratio."""
    with open(f"{run_folder}/arguments.json") as f:
        run_args = json.load(f)

    is_cond = run_args.get('fine_tune_cond_stage3', False)
    scalar_cond = run_args.get('scalar_cond', False)
    cond_encoder = run_args.get('cond_encoder', False)
    embed_dim = run_args.get('embed_dim', 8)
    model_norm = run_args.get('raw_data_normalization', 'log1p')

    print(f"\n{'='*70}")
    print(f"Run: {run_folder}")
    print(f"Norm: {model_norm}, cond: {is_cond}, scalar: {scalar_cond}, cond_enc: {cond_encoder}")

    # Check if this model uses same data normalization
    if model_norm != norm:
        print(f"WARNING: Model uses {model_norm} but data loaded with {norm}")
        print("Loading separate data for this model...")
        model_args_ns = SimpleNamespace(**run_args)
        from imputeVAE.data import load_raw_data as lrd, filter_by_protocol as fbp
        model_nuc_batch = run_args.get('batch', 'n3A')
        model_cell_batch = model_nuc_batch.replace('n', 'c', 1)
        adata_c = lrd("10X_cells_v2_v3_AIBS/", model_args_ns)
        adata_c = fbp(adata_c, SimpleNamespace(**{**run_args, 'batch': model_cell_batch}))
        adata_n = lrd("10X_nuclei_v2_v3_AIBS/", model_args_ns)
        adata_n = fbp(adata_n, model_args_ns)
        ml = model_norm if run_args.get('use_raw_data', False) and model_norm != 'raw' else None
        if ml and ml in adata_c.layers:
            Xc = np.array(adata_c.layers[ml], dtype=np.float32)
            Xn = np.array(adata_n.layers[ml], dtype=np.float32)
        else:
            Xc = adata_c.X.toarray() if hasattr(adata_c.X, 'toarray') else np.array(adata_c.X, dtype=np.float32)
            Xn = adata_n.X.toarray() if hasattr(adata_n.X, 'toarray') else np.array(adata_n.X, dtype=np.float32)
        ctc = np.array(adata_c.obs['cell_type'].values)
        ctn = np.array(adata_n.obs['cell_type'].values)
        rcv = {}
        for ct in sorted(set(ctn) & set(ctc)):
            mask = ctc == ct
            if mask.sum() >= 5:
                rcv[ct] = Xc[mask].var(axis=0).mean()
        del adata_c, adata_n
    else:
        Xn = X_nuc
        ctn = ct_nuc
        rcv = raw_cell_var

    num_features = Xn.shape[1]
    num_internal = run_args.get('num_internal', [1024, 512])
    num_latent = run_args.get('num_latent', 32)

    device = torch.device('cpu')
    model_path = f"{run_folder}/models/model.pt"

    if is_cond:
        from imputeVAE.models_cond import VAE_JSD_Cond
        kwargs = dict(
            input_size=num_features,
            num_internal=num_internal,
            num_latent=num_latent,
            embed_dim=1 if scalar_cond else embed_dim,
            scalar_cond=scalar_cond,
        )
        if cond_encoder:
            kwargs['cond_encoder'] = True
        model = VAE_JSD_Cond(**kwargs)
    else:
        from imputeVAE.models import VAE_JSD
        model = VAE_JSD(
            input_size=num_features,
            num_internal=num_internal,
            num_latent=num_latent,
        )

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Cross-decode
    CELL_ID, NUCLEI_ID = 0, 1
    cross_dec_list = []
    with torch.no_grad():
        for i in range(0, len(Xn), 256):
            batch = torch.tensor(Xn[i:i+256], device=device)
            if is_cond:
                xd = model.decode_cross(batch, source_modality_id=NUCLEI_ID, target_modality_id=CELL_ID)
            else:
                encoded, mu, log_var = model.encode(batch)
                xd = model.decode(encoded)
            cross_dec_list.append(xd.cpu().numpy())

    X_xd = np.concatenate(cross_dec_list, axis=0)
    del model, cross_dec_list
    gc.collect()

    # Compute variance ratios
    cts_used = sorted(set(rcv.keys()) & set(np.unique(ctn)))
    print(f"\n{'Cell Type':<15} {'n_nuc':>6} {'var_raw_cell':>12} {'var_xd':>10} {'xd/cell':>8}")
    print("-" * 55)

    ratios = []
    for ct in cts_used:
        mask = ctn == ct
        n = mask.sum()
        if n < 5:
            continue
        xd_var = X_xd[mask].var(axis=0).mean()
        ratio = xd_var / rcv[ct] if rcv[ct] > 0 else float('nan')
        ratios.append(ratio)
        print(f"{ct:<15} {n:>6} {rcv[ct]:>12.4f} {xd_var:>10.4f} {ratio:>8.3f}")

    print("-" * 55)
    mean_r = np.mean(ratios)
    median_r = np.median(ratios)
    print(f"Mean ratio:   {mean_r:.3f} ({mean_r*100:.1f}% preserved)")
    print(f"Median ratio: {median_r:.3f} ({median_r*100:.1f}% preserved)")

    del X_xd
    if model_norm != norm:
        del Xn, rcv
    gc.collect()
    return mean_r, median_r


# Run all models
results = {}
for rf in run_folders:
    try:
        mean_r, median_r = run_variance_for_model(rf)
        results[rf] = (mean_r, median_r)
    except Exception as e:
        print(f"ERROR for {rf}: {e}")
        import traceback; traceback.print_exc()

# Summary
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
for rf, (mean_r, median_r) in results.items():
    with open(f"{rf}/arguments.json") as f:
        ra = json.load(f)
    label = f"{ra.get('raw_data_normalization','log1p')}, cond={ra.get('fine_tune_cond_stage3',False)}, scalar={ra.get('scalar_cond',False)}, enc={ra.get('cond_encoder',False)}"
    print(f"{os.path.basename(rf)}: mean={mean_r:.3f} median={median_r:.3f}  [{label}]")
