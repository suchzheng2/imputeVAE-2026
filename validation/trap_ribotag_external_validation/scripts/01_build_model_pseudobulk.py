#!/usr/bin/env python3
"""Build per-cell-type pseudobulk matrices and deltas from the full-data ImputeVAE model.

Sources (project root relative):
  raw nuclei  : Cytoplasm_imputation/scvi/10X_nuclei_v2_v3_AIBS/data.h5ad, n3A, 17 common CTs,
                10,134-gene list, genes sorted by name, log1p(counts)   [as analysis/export_cvae_imputed.py]
  raw cells   : .../10X_cells_v2_v3_AIBS/data.h5ad, c3A, same filtering                [for ground truth]
  imputed all : run/cvae_full_eval/imputed_cell_from_nucleus.npy + nucleus_obs.csv    (40,166 x 10,134)
  imputed val : run/eval_v3_splitaware/cvae_imputed_valid.npy + valid_nuc_obs.csv      (8,033 x 10,134)
  split       : data/split_manifest.tsv (validation partition indices, as in validation/syngo)

Outputs -> validation/trap_ribotag_external_validation/outputs/model_pseudobulk/
  {nucleus,imputed,cell}_pseudobulk_by_ct_{all,valid}.csv   genes x CT mean log1p (+ 'ALL' column = all nuclei/cells pooled)
  predicted_delta_by_ct_{all,valid}.csv                     imputed - nucleus, per CT + ALL + pooled groups
  gt_delta_by_ct_{all,valid}.csv                            cell - nucleus, per CT + ALL + pooled groups
  nucleus_within_ct_sd_{all,valid}.csv                      per-gene SD within CT (for abundance matching)
  ct_counts_{all,valid}.csv                                 n nuclei / n cells per CT
  README.md                                                 provenance + alignment checks
Aggregation: arithmetic mean of log1p values over cells of a CT (pseudobulk on log scale, matches
compartment_classification and SynGO validations). Pooled groups = unweighted mean of per-CT deltas.
"""
import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse

PROJECT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE"
DATA_ROOT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi"
OUT = os.path.join(PROJECT, "validation/trap_ribotag_external_validation/outputs/model_pseudobulk")
os.makedirs(OUT, exist_ok=True)

NEURONAL = ['L2/3 IT', 'L5 ET', 'L5 IT', 'L5/6 NP', 'L6 CT', 'L6 IT', 'L6b', 'Lamp5', 'Pvalb', 'Sncg', 'Sst', 'Vip']
INHIBITORY = ['Lamp5', 'Pvalb', 'Sncg', 'Sst', 'Vip']
EXCITATORY = ['L2/3 IT', 'L5 ET', 'L5 IT', 'L5/6 NP', 'L6 CT', 'L6 IT', 'L6b']
NON_NEURONAL = ['Astro', 'Endo', 'Macrophage', 'OPC', 'Oligo']


def load_h5ad(data_dir, batch, gene_list, common_genes, common_ct):
    path = os.path.join(DATA_ROOT, data_dir, "data.h5ad")
    print(f"Loading {path}", flush=True)
    adata = ad.read_h5ad(path)
    if 'meta_batch_id' not in adata.obs:
        adata.obs['meta_batch_id'] = adata.obs['batch_id'].apply(lambda x: x.split('.')[0])
    row_mask = adata.obs['meta_batch_id'].str.endswith(batch) & adata.obs['cell_type'].isin(common_ct)
    col_mask = adata.var['gene_names'].isin(common_genes) & adata.var['gene_names'].isin(gene_list)
    adata = adata[row_mask.values, col_mask.values].copy()
    adata = adata[:, adata.var['gene_names'].argsort()].copy()
    X = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
    X = np.log1p(X).astype(np.float32)
    ct = adata.obs['cell_type'].astype(str).values
    genes = adata.var['gene_names'].values.tolist()
    print(f"  -> {X.shape}", flush=True)
    return X, ct, genes


def pseudobulk(X, ct, genes, cts):
    df = pd.DataFrame({c: X[ct == c].mean(axis=0) for c in cts if (ct == c).sum() > 0}, index=genes)
    df['ALL'] = X.mean(axis=0)
    return df


def within_ct_sd(X, ct, genes, cts):
    return pd.DataFrame({c: X[ct == c].std(axis=0) for c in cts if (ct == c).sum() > 0}, index=genes)


def add_pooled(delta):
    d = delta.copy()
    for name, grp in [('POOL_neuronal', NEURONAL), ('POOL_inhibitory', INHIBITORY),
                      ('POOL_excitatory', EXCITATORY), ('POOL_nonneuronal', NON_NEURONAL),
                      ('POOL_all17', NEURONAL + NON_NEURONAL)]:
        cols = [c for c in grp if c in d.columns]
        d[name] = d[cols].mean(axis=1)
    return d


def run(tag, nuc_X, nuc_ct, cell_X, cell_ct, imp_X, imp_ct, genes, cts, notes):
    assert imp_X.shape == nuc_X.shape, (imp_X.shape, nuc_X.shape)
    match = (imp_ct == nuc_ct).mean()
    notes.append(f"[{tag}] imputed/nucleus CT-label alignment: {match*100:.2f}% over {len(nuc_ct)} nuclei")
    assert match > 0.99, f"{tag}: CT alignment {match}"
    pb_n = pseudobulk(nuc_X, nuc_ct, genes, cts)
    pb_i = pseudobulk(imp_X, nuc_ct, genes, cts)   # imputed rows are the same nuclei -> nucleus labels
    pb_c = pseudobulk(cell_X, cell_ct, genes, cts)
    common = [c for c in cts if c in pb_n.columns and c in pb_c.columns] + ['ALL']
    pred = add_pooled((pb_i[common] - pb_n[common]))
    gt = add_pooled((pb_c[common] - pb_n[common]))
    pb_n.to_csv(f"{OUT}/nucleus_pseudobulk_by_ct_{tag}.csv")
    pb_i.to_csv(f"{OUT}/imputed_pseudobulk_by_ct_{tag}.csv")
    pb_c.to_csv(f"{OUT}/cell_pseudobulk_by_ct_{tag}.csv")
    pred.to_csv(f"{OUT}/predicted_delta_by_ct_{tag}.csv")
    gt.to_csv(f"{OUT}/gt_delta_by_ct_{tag}.csv")
    within_ct_sd(nuc_X, nuc_ct, genes, cts).to_csv(f"{OUT}/nucleus_within_ct_sd_{tag}.csv")
    counts = pd.DataFrame({'n_nuclei': pd.Series(nuc_ct).value_counts(),
                           'n_cells': pd.Series(cell_ct).value_counts()}).reindex(cts).fillna(0).astype(int)
    counts.to_csv(f"{OUT}/ct_counts_{tag}.csv")
    notes.append(f"[{tag}] wrote pseudobulks for {len(common)-1} CTs; n_nuclei={len(nuc_ct)}, n_cells={len(cell_ct)}")
    notes.append(f"[{tag}] predicted vs GT delta (ALL) Pearson r = {np.corrcoef(pred['ALL'], gt['ALL'])[0,1]:.4f}")
    return pred, gt


def main():
    gene_list = pd.read_csv(f"{PROJECT}/data/training_gene_list_v3.csv", header=None)[0].tolist()
    common_genes = pd.read_csv(f"{PROJECT}/data/common_genes.csv")['common_genes'].tolist()
    cts = pd.read_csv(f"{PROJECT}/data/common_cell_types.csv")['common_cell_types'].tolist()
    notes = []

    nuc_X, nuc_ct, genes = load_h5ad("10X_nuclei_v2_v3_AIBS", "n3A", gene_list, common_genes, cts)
    cell_X, cell_ct, genes_c = load_h5ad("10X_cells_v2_v3_AIBS", "c3A", gene_list, common_genes, cts)
    assert genes == genes_c
    pd.Series(genes).to_csv(f"{OUT}/gene_order.txt", index=False, header=False)

    # ---- all nuclei
    imp_all = np.load(f"{PROJECT}/run/cvae_full_eval/imputed_cell_from_nucleus.npy")
    imp_all_ct = pd.read_csv(f"{PROJECT}/run/cvae_full_eval/nucleus_obs.csv")['cell_type'].astype(str).values
    pred_all, gt_all = run("all", nuc_X, nuc_ct, cell_X, cell_ct, imp_all, imp_all_ct, genes, cts, notes)

    # cross-check against compartment_classification ground truth
    ref = pd.read_csv(f"{PROJECT}/compartment_classification/gt_delta_per_ct.csv", index_col=0)
    ref = ref.reindex(gt_all.index)
    rs = [np.corrcoef(ref[c].values, gt_all[c].values)[0, 1] for c in ref.columns if c in gt_all.columns]
    notes.append(f"[all] GT delta vs compartment_classification/gt_delta_per_ct.csv: min Pearson r over CTs = {min(rs):.5f}")
    refp = pd.read_csv(f"{PROJECT}/compartment_classification/predicted_deltas_global.csv", index_col=0).reindex(gt_all.index)
    notes.append(f"[all] predicted ALL delta vs compartment_classification ImputeVAE global: r = {np.corrcoef(refp['ImputeVAE'].values, pred_all['ALL'].values)[0,1]:.5f}")
    del imp_all

    # ---- validation split
    manifest = pd.read_csv(f"{PROJECT}/data/split_manifest.tsv", sep='\t')
    def valid_idx(mod):
        m = manifest[manifest.modality == mod].sort_values('obs_id').reset_index(drop=True)
        return np.where(m.partition == 'valid')[0]
    vn, vc = valid_idx('nuclei'), valid_idx('cell')
    imp_v = np.load(f"{PROJECT}/run/eval_v3_splitaware/cvae_imputed_valid.npy")
    imp_v_ct = pd.read_csv(f"{PROJECT}/run/eval_v3_splitaware/valid_nuc_obs.csv")['cell_type'].astype(str).values
    run("valid", nuc_X[vn], nuc_ct[vn], cell_X[vc], cell_ct[vc], imp_v, imp_v_ct, genes, cts, notes)

    with open(f"{OUT}/README.md", "w") as f:
        f.write("# Model pseudobulk matrices\n\nGenerated by scripts/01_build_model_pseudobulk.py\n\n")
        f.write("Aggregation: mean of log1p(counts) over cells within cell type; 'ALL' = all cells pooled; "
                "POOL_* = unweighted mean of per-CT deltas.\n\n## Alignment checks\n\n")
        f.write("\n".join(f"- {n}" for n in notes) + "\n")
    print("\n".join(notes))


if __name__ == "__main__":
    main()
