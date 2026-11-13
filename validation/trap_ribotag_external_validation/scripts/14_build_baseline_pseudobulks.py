#!/usr/bin/env python3
"""Extension: per-cell-type imputed pseudobulks for the baseline methods (same 40,166 nuclei, same gene order as ImputeVAE).
Sources: run/<method>_baseline/imputed_cell_from_nucleus.npy + nucleus_obs.csv (as in analysis/compartment_classification.py)."""
import os, numpy as np, pandas as pd
PROJECT = "/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE"; TASK = f"{PROJECT}/validation/trap_ribotag_external_validation"
OUT = f"{TASK}/outputs/extension/model_pseudobulk"; os.makedirs(OUT, exist_ok=True)
METHODS = {"CT-mean": "ctmean_baseline", "Linear_reg": "linreg_baseline", "scVI": "scvi_baseline", "scANVI": "scanvi_baseline", "scGen": "scgen_baseline", "scButterfly": "scbutterfly_baseline", "cVAE_20pct": "cvae_baseline"}
genes = pd.read_csv(f"{TASK}/outputs/model_pseudobulk/gene_order.txt", header=None)[0].tolist()
ref_ct = pd.read_csv(f"{PROJECT}/run/cvae_full_eval/nucleus_obs.csv")["cell_type"].astype(str).values
ref = pd.read_csv(f"{TASK}/outputs/model_pseudobulk/imputed_pseudobulk_by_ct_all.csv", index_col=0)
notes = []
for m, d in METHODS.items():
    p = f"{PROJECT}/run/{d}/imputed_cell_from_nucleus.npy"
    if not os.path.exists(p): notes.append(f"{m}: MISSING {p}"); continue
    X = np.load(p, mmap_mode="r"); ct = pd.read_csv(f"{PROJECT}/run/{d}/nucleus_obs.csv")["cell_type"].astype(str).values
    assert X.shape == (len(ref_ct), len(genes)), (m, X.shape)
    agree = (ct == ref_ct).mean(); assert agree > 0.999, (m, agree)
    gn = f"{PROJECT}/run/{d}/gene_names.csv"
    if os.path.exists(gn):
        g2 = pd.read_csv(gn, header=None).iloc[:, -1].astype(str).tolist(); same = g2 == genes   # file has no header
        assert same, f"{m}: gene order differs"
    pb = pd.DataFrame({c: np.asarray(X[ct == c]).mean(axis=0) for c in ref.columns if c != "ALL"}, index=genes); pb["ALL"] = np.asarray(X).mean(axis=0)
    pb.to_csv(f"{OUT}/imputed_pseudobulk_by_ct__{m}.csv")
    notes.append(f"{m}: shape {X.shape}, CT-label agreement {agree:.4f}, value range [{float(pb.values.min()):.2f}, {float(pb.values.max()):.2f}], Spearman(Pvalb profile vs ImputeVAE) {pb['Pvalb'].corr(ref['Pvalb'], method='spearman'):.3f}")
open(f"{OUT}/README.md", "w").write("# Baseline per-CT imputed pseudobulks\n\n" + "\n".join(f"- {n}" for n in notes) + "\n"); print("\n".join(notes))
