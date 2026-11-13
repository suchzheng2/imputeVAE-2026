"""
Examine specific conserved genes in cross-tissue imputation results.

Loads the brain-trained cVAE, runs imputation on kidney nuclei mean profile,
and reports per-gene results for conserved, housekeeping, lncRNA, and
brain-specific gene groups.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from scipy import sparse, stats
import anndata as ad

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from imputeVAE.models_cond import VAE_JSD_Cond


# Gene groups
CONSERVED_INCREASED = {
    'RPL41': 'Rpl41', 'RPL32': 'Rpl32', 'RPLP1': 'Rplp1', 'RPL26': 'Rpl26',
    'RPS27A': 'Rps27a', 'RPS24': 'Rps24', 'RPL19': 'Rpl19', 'TPT1': 'Tpt1',
    'EEF1A1': 'Eef1a1', 'RPS8': 'Rps8', 'RPS14': 'Rps14', 'RPL8': 'Rpl8',
    'RPS3': 'Rps3', 'TMSB4X': 'Tmsb4x', 'EIF1': 'Eif1', 'RPS29': 'Rps29',
    'PTMA': 'Ptma', 'ACTB': 'Actb', 'RPL27A': 'Rpl27a', 'ACTG1': 'Actg1',
    'CFL1': 'Cfl1', 'DYNLL1': 'Dynll1',
}

HOUSEKEEPING = {
    'FTH1': 'Fth1', 'B2M': 'B2m', 'FTL': 'Ftl', 'GAPDH': 'Gapdh',
    'UBB': 'Ubb', 'UBC': 'Ubc', 'HSP90AB1': 'Hsp90ab1', 'HSP90AA1': 'Hsp90aa1',
    'EEF2': 'Eef2', 'TUBA1B': 'Tuba1b', 'TUBB': 'Tubb',
}

LNCRNAS = {
    'MALAT1': 'Malat1', 'NEAT1': 'Neat1', 'XIST': 'Xist',
    'MEG3': 'Meg3', 'KCNQ1OT1': 'Kcnq1ot1',
}

BRAIN_SPECIFIC = {
    'SNAP25': 'Snap25', 'SYN1': 'Syn1', 'SYT1': 'Syt1', 'VAMP2': 'Vamp2',
    'SV2A': 'Sv2a', 'SV2B': 'Sv2b', 'NRXN1': 'Nrxn1', 'CAMK2A': 'Camk2a',
    'GAD1': 'Gad1', 'GAD2': 'Gad2', 'SLC17A7': 'Slc17a7',
}

# All ribosomal (not just the top-22)
def get_all_ribosomal(training_genes):
    return {g.upper(): g for g in training_genes if g.startswith(('Rpl', 'Rps')) and not any(c in g for c in ['-', 'os'])}


def load_mean_profile(h5ad_path):
    """Load h5ad and return per-gene mean (raw counts) + gene names."""
    print(f"Loading {h5ad_path}...")
    adata = ad.read_h5ad(h5ad_path)
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
    del adata

    if sparse.issparse(X):
        mean_raw = np.array(X.mean(axis=0)).flatten()
    else:
        mean_raw = np.asarray(X.mean(axis=0)).flatten()

    print(f"  {len(var_names)} genes, mean count range [{mean_raw.min():.3f}, {mean_raw.max():.1f}]")
    return mean_raw, var_names


def analyze_group(gene_dict, group_name, model_input, imputed_vals, nuc_recon_vals,
                  gene_to_idx, cell_gene_lookup, cell_mean_log1p):
    print(f'\n{"=" * 85}')
    print(f'{group_name}')
    print(f'{"=" * 85}')
    print(f'{"Gene":>12s} {"nuc_in":>8s} {"nuc_rec":>8s} {"imp_cell":>8s} {"act_cell":>8s} '
          f'{"gt_delta":>9s} {"mdl_delta":>9s} {"dir":>5s}')
    print('-' * 85)

    correct_dir = 0
    total = 0
    deltas_gt = []
    deltas_mdl = []
    rows = []

    for human_gene, mouse_gene in sorted(gene_dict.items()):
        if mouse_gene not in gene_to_idx:
            continue
        idx = gene_to_idx[mouse_gene]
        nuc_in = model_input[idx]
        imp_val = imputed_vals[idx]
        nuc_rec = nuc_recon_vals[idx]

        if human_gene not in cell_gene_lookup:
            continue
        act_cell = cell_mean_log1p[cell_gene_lookup[human_gene]]

        gt_delta = act_cell - nuc_in
        mdl_delta = imp_val - nuc_in

        if abs(gt_delta) > 0.001:
            total += 1
            correct = (np.sign(mdl_delta) == np.sign(gt_delta))
            if correct:
                correct_dir += 1
            direction = 'OK' if correct else 'WRONG'
        else:
            direction = 'tiny'

        deltas_gt.append(gt_delta)
        deltas_mdl.append(mdl_delta)
        rows.append({
            'gene': mouse_gene, 'nuc_in': nuc_in, 'nuc_rec': nuc_rec,
            'imp_cell': imp_val, 'act_cell': act_cell,
            'gt_delta': gt_delta, 'mdl_delta': mdl_delta, 'direction': direction,
        })

        print(f'{mouse_gene:>12s} {nuc_in:8.3f} {nuc_rec:8.3f} {imp_val:8.3f} {act_cell:8.3f} '
              f'{gt_delta:+9.3f} {mdl_delta:+9.3f} {direction:>5s}')

    if total > 0:
        print(f'\nDirection accuracy: {correct_dir}/{total} ({correct_dir/total:.1%})')
    if len(deltas_gt) > 2:
        r, p = stats.pearsonr(deltas_gt, deltas_mdl)
        print(f'Delta correlation: r={r:.3f} (p={p:.2e})')

    return rows


def main():
    model_run = "run/run_2026-07-18-19-40-51_570984"
    output_dir = "notes/experiments/cross_tissue_imputation"
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    with open(f'{model_run}/arguments.json') as f:
        margs = json.load(f)

    model = VAE_JSD_Cond(
        input_size=margs['num_feature'], num_internal=margs['num_internal'],
        num_latent=margs['num_latent'], embed_dim=margs.get('embed_dim', 8),
        scalar_cond=margs.get('scalar_cond', False),
        cond_encoder=margs.get('cond_encoder', False))
    model.load_state_dict(torch.load(f'{model_run}/models/model.pt', map_location='cpu'))
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Model loaded on {device}")

    # Training genes
    training_genes = pd.read_csv('data/training_gene_list_v3.csv', header=None)[0].tolist()
    gene_to_idx = {g: i for i, g in enumerate(training_genes)}

    # Load kidney mean profiles
    nuc_mean_raw, var_nuc = load_mean_profile("data/external/lake2023_nuclei.h5ad")
    cell_mean_raw, var_cell = load_mean_profile("data/external/lake2023_cells.h5ad")

    nuc_gene_lookup = {g: i for i, g in enumerate(var_nuc)}
    cell_gene_lookup = {g: i for i, g in enumerate(var_cell)}

    # Build model input (log1p of mean nuclei counts, mapped to training genes)
    model_input = np.zeros(len(training_genes), dtype=np.float32)
    for i, mg in enumerate(training_genes):
        hg = mg.upper()
        if hg in nuc_gene_lookup:
            model_input[i] = np.log1p(nuc_mean_raw[nuc_gene_lookup[hg]])

    # Cell ground truth (log1p)
    cell_mean_log1p = np.log1p(cell_mean_raw)

    # Run through model: nuclei->cell imputation
    x = torch.tensor(model_input, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        imputed = model.decode_cross(x, source_modality_id=1, target_modality_id=0)
        nuc_recon = model.decode_cross(x, source_modality_id=1, target_modality_id=1)
    imputed_vals = imputed.cpu().numpy().flatten()
    nuc_recon_vals = nuc_recon.cpu().numpy().flatten()

    print(f"\nImputed range: [{imputed_vals.min():.3f}, {imputed_vals.max():.3f}]")
    print(f"Nuc recon range: [{nuc_recon_vals.min():.3f}, {nuc_recon_vals.max():.3f}]")

    # Analyze each gene group
    common_args = dict(
        model_input=model_input, imputed_vals=imputed_vals, nuc_recon_vals=nuc_recon_vals,
        gene_to_idx=gene_to_idx, cell_gene_lookup=cell_gene_lookup,
        cell_mean_log1p=cell_mean_log1p)

    all_results = {}

    rows = analyze_group(CONSERVED_INCREASED,
        'CONSERVED TOP-INCREASED (22 genes in brain & kidney top-100)', **common_args)
    all_results['conserved_increased'] = rows

    rows = analyze_group(HOUSEKEEPING,
        'HOUSEKEEPING / CYTOSKELETAL', **common_args)
    all_results['housekeeping'] = rows

    rows = analyze_group(get_all_ribosomal(training_genes),
        'ALL RIBOSOMAL GENES', **common_args)
    all_results['all_ribosomal'] = rows

    rows = analyze_group(LNCRNAS,
        'NUCLEAR lncRNAs', **common_args)
    all_results['nuclear_lncrna'] = rows

    rows = analyze_group(BRAIN_SPECIFIC,
        'BRAIN-SPECIFIC (synaptic — expect WRONG in kidney)', **common_args)
    all_results['brain_specific'] = rows

    # Summary table
    print(f'\n{"=" * 85}')
    print('SUMMARY BY GENE GROUP')
    print(f'{"=" * 85}')
    print(f'{"Group":>30s} {"N":>4s} {"Dir OK":>7s} {"Delta r":>8s}')
    print('-' * 55)

    for group_name, rows in all_results.items():
        if not rows:
            continue
        gt_d = [r['gt_delta'] for r in rows]
        md_d = [r['mdl_delta'] for r in rows]
        n = len(rows)
        n_ok = sum(1 for r in rows if r['direction'] == 'OK')
        if len(gt_d) > 2:
            r, _ = stats.pearsonr(gt_d, md_d)
        else:
            r = float('nan')
        print(f'{group_name:>30s} {n:4d} {n_ok}/{n:>5s} {r:8.3f}')

    # Save detailed results
    out_path = os.path.join(output_dir, "conserved_gene_analysis.json")
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
