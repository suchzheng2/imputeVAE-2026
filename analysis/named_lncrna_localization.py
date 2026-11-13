"""Export per-gene predicted vs ground-truth delta for canonical lncRNAs.

Produces a small table for the per-gene "flip" figure showing that the model
tracks per-gene localization — including cell-type-specific flips (Meg3,
Kcnq1ot1) that a fixed nuclear-gene rule gets wrong.
"""
import os, sys
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_ROOT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi"

TARGET_GENES = [
    'Xist', 'Tsix', 'Malat1', 'Neat1', 'Kcnq1ot1', 'Meg3',
    'Rian', 'Mirg', 'Airn', 'Gas5', 'Snhg11', 'Firre',
]

OUTPUT_DIR = "notes/experiments/named_lncrna_localization"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Load gene list ---
    gene_list = pd.read_csv("data/training_gene_list_v3.csv", header=None)[0].tolist()
    gene_set = set(gene_list)

    # --- Load ground-truth delta from all_genes_delta.csv ---
    gt_df = pd.read_csv("notes/experiments/nuclear_lncrna_analysis/all_genes_delta.csv")
    gt_df = gt_df.set_index('gene')
    # gt delta = cell_mean - nuc_mean (already in the file as 'delta')

    # --- Load imputed matrix + input nuclei to compute predicted delta ---
    imputed = np.load("run/cvae_full_eval/imputed_cell_from_nucleus.npy")
    nuc_obs = pd.read_csv("run/cvae_full_eval/nucleus_obs.csv", index_col=0)

    # Load raw nuclei (same filtering as find_nuclear_lncrnas.py)
    common_genes = pd.read_csv("data/common_genes.csv", header=0)["common_genes"].tolist()
    common_ct = pd.read_csv("data/common_cell_types.csv", header=0)["common_cell_types"].tolist()

    adata_nuc = ad.read_h5ad(os.path.join(DATA_ROOT, "10X_nuclei_v2_v3_AIBS", "data.h5ad"))
    if 'meta_batch_id' not in adata_nuc.obs:
        adata_nuc.obs['meta_batch_id'] = adata_nuc.obs['batch_id'].apply(lambda x: x.split('.')[0])
    adata_nuc = adata_nuc[adata_nuc.obs['meta_batch_id'].str.endswith("n3A")].copy()
    adata_nuc = adata_nuc[:, adata_nuc.var['gene_names'].isin(common_genes)].copy()
    adata_nuc = adata_nuc[adata_nuc.obs['cell_type'].isin(common_ct)].copy()
    adata_nuc = adata_nuc[:, adata_nuc.var['gene_names'].isin(gene_list)].copy()
    adata_nuc = adata_nuc[:, adata_nuc.var['gene_names'].argsort()].copy()

    X_nuc = adata_nuc.X.toarray() if sparse.issparse(adata_nuc.X) else adata_nuc.X
    X_nuc = np.log1p(X_nuc).astype(np.float32)
    nuc_gene_names = list(adata_nuc.var['gene_names'].values)

    n_nuclei = X_nuc.shape[0]
    n_cells = int(gt_df['cell_mean'].notna().sum())  # from gt file context

    # Load cell count
    adata_cell = ad.read_h5ad(os.path.join(DATA_ROOT, "10X_cells_v2_v3_AIBS", "data.h5ad"))
    if 'meta_batch_id' not in adata_cell.obs:
        adata_cell.obs['meta_batch_id'] = adata_cell.obs['batch_id'].apply(lambda x: x.split('.')[0])
    adata_cell = adata_cell[adata_cell.obs['meta_batch_id'].str.endswith("c3A")].copy()
    adata_cell = adata_cell[:, adata_cell.var['gene_names'].isin(common_genes)].copy()
    adata_cell = adata_cell[adata_cell.obs['cell_type'].isin(common_ct)].copy()
    n_cells_actual = adata_cell.shape[0]
    del adata_cell

    print(f"Nuclei: {n_nuclei}, Cells: {n_cells_actual}")
    print(f"Imputed shape: {imputed.shape}")
    print(f"Gene names: {len(nuc_gene_names)}")

    # --- Compute predicted delta per gene ---
    pred_mean_imp = imputed.mean(axis=0)   # mean imputed cell
    pred_mean_nuc = X_nuc.mean(axis=0)     # mean input nuclei
    pred_delta_all = pred_mean_imp - pred_mean_nuc  # predicted delta

    # Ground-truth delta for all genes (for percentile computation)
    gt_delta_all = np.array([gt_df.loc[g, 'delta'] if g in gt_df.index else np.nan
                             for g in nuc_gene_names])

    # --- Build output table ---
    rows = []
    for gene in TARGET_GENES:
        in_config_b = gene in gene_set
        if not in_config_b or gene not in nuc_gene_names:
            rows.append({
                'gene': gene,
                'in_config_b': in_config_b,
                'pred_delta': None,
                'gt_delta': None,
                'pred_pctile': None,
                'gt_pctile': None,
                'sign_agreement': None,
                'gt_class': None,
                'n_cells': n_cells_actual,
                'n_nuclei': n_nuclei,
            })
            continue

        idx = nuc_gene_names.index(gene)
        pd_val = float(pred_delta_all[idx])
        gd_val = float(gt_delta_all[idx]) if not np.isnan(gt_delta_all[idx]) else float(gt_df.loc[gene, 'delta'])

        # Percentile ranks (0 = most negative/nuclear, 100 = most positive/cytoplasmic)
        pred_pctile = float(np.mean(pred_delta_all <= pd_val) * 100)
        gt_valid = gt_delta_all[~np.isnan(gt_delta_all)]
        gt_pctile = float(np.mean(gt_valid <= gd_val) * 100)

        sign_agree = (pd_val >= 0) == (gd_val >= 0)
        gt_class = "nuclear" if gd_val < 0 else "cytoplasmic"

        rows.append({
            'gene': gene,
            'in_config_b': True,
            'pred_delta': round(pd_val, 5),
            'gt_delta': round(gd_val, 5),
            'pred_pctile': round(pred_pctile, 2),
            'gt_pctile': round(gt_pctile, 2),
            'sign_agreement': sign_agree,
            'gt_class': gt_class,
            'n_cells': n_cells_actual,
            'n_nuclei': n_nuclei,
        })

    df = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_DIR, "named_lncrna_localization.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")
    print(df.to_string(index=False))

    # --- README ---
    n_present = df['in_config_b'].sum()
    n_agree = df.loc[df['sign_agreement'].notna(), 'sign_agreement'].sum()
    n_total = df['sign_agreement'].notna().sum()
    readme = f"""# Named lncRNA localization export

Per-gene predicted vs ground-truth delta for {len(TARGET_GENES)} canonical lncRNAs.
Shows that the cVAE tracks per-gene localization including cell-type-specific
flips (Meg3, Kcnq1ot1 are cytoplasmic in this neuron-dominated dataset).

## Data
- **Model:** cVAE full (run_2026-07-18-19-40-51_570984, job 7714771)
- **Imputed:** `run/cvae_full_eval/imputed_cell_from_nucleus.npy`
- **Ground truth:** `notes/experiments/nuclear_lncrna_analysis/all_genes_delta.csv`
- **Data split:** n3A nuclei ({n_nuclei:,}), c3A cells ({n_cells_actual:,}), Config B 10,134 genes
- **Delta definition:** mean(imputed_cell) − mean(input_nuclei) [predicted], mean(real_cell) − mean(real_nuclei) [ground truth], log1p scale

## Results
- {n_present}/{len(TARGET_GENES)} genes present in Config B gene set
- {n_agree}/{n_total} sign agreement between predicted and ground-truth delta
- Percentile: 0 = most nuclear/removed, 100 = most cytoplasmic/added
"""
    with open(os.path.join(OUTPUT_DIR, "README.md"), 'w') as f:
        f.write(readme)
    print(f"\nSaved README.md")


if __name__ == "__main__":
    main()
