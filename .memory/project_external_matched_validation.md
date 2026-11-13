---
name: project-external-matched-validation
description: External validation 2026-10-26 — Bakken 2018 (GSE123454, VISp SMART-seq, matched nuclei/cells, 11 clusters) is the independent matched hold-out; Ding 2020 (GSE132044) cortex has NO cells (4 snRNA-seq chemistries + bulk). Transformation transfers (sign at ceiling, delta r 0.58 vs ceiling 0.92) but the per-type training delta scores the same; residual decode is the cross-platform mode; latent kNN type vote fails off-chemistry, pca_lr classifier (0.982) replaces it
metadata:
  type: project
---
Notes: `notes/experiments/bakken2018_holdout/README.md`, `notes/experiments/cross_dataset_ding/README.md`; storyline gap 4 updated.
Code: `analysis/external/{ingest_bakken2018,ingest_ding2020,eval_matched_external,eval_ding2020,delta_reliability,type_classifier}.py`.
Data: `data/external/bakken2018_GSE123454/processed/*.h5ad` (461 nuclei / 445 cells, 8,928 of 10,134 genes present, depth-thinned to training medians), `data/external/ding2020_GSE132044/processed/nuclei.h5ad` (14,255 nuclei, 9,666 genes, Smart-seq2 thinned). Results: `run/external_eval/{bakken2018,bakken2018_exon,ding2020}/`.

- **Bakken**: predicted delta (imputed - nucleus) vs empirical (cell - nucleus): arm A quantile-map decode, delta on nucleus 0.58 pooled / 0.54 per-cluster median / 0.73 composition-free; sign agreement 0.94 = split-half ceiling; reliability ceiling 0.92; nucleus + training per-type delta 0.56 / 0.59 / 0.73 (same). Full regeneration 0.41-0.48 and nuclear-lncRNA sign 0.33-0.39 (platform profile), so **report the residual mode cross-platform**. Per cluster 0.43-0.80; lowest where type assignment is split (L6.Car12). "Gene corr across clusters" is 0.1-0.4 for every row incl. raw nuclei (11 similar clusters): not informative there.
- **Ding**: pseudobulk-vs-bulk Pearson raw 0.27-0.56 -> imputed 0.58-0.65 on 10x v2 / DroNc / sci-RNA-seq / Smart-seq2; training delta 0.58-0.68 (>= model); predicted delta agreement across chemistries 0.82 (paper model 0.91). Regeneration better at 10x depth, residual mode better on DroNc / Smart-seq2.
- **Type assignment off-chemistry**: latent kNN vote gives 14-23 % L6b and < 1 % Astro on 2-6 k UMI nuclei; `type_classifier.pca_lr_predict` (log-CPM / 3,000 HVG / PCA 50 / LR on training nuclei) validates at 0.982 on v3 and gives plausible compositions; on Bakken it maps L5.Chrna6 to L5 ET (1.00) and L4 clusters to L5 IT (0.99). Use `--type_predictor pca_lr` (also in `analysis/cross_dataset_eval.py` since 2026-10-27).
- **v2 re-run with the classifier (2026-10-27, job 14189131_13-14, `run/cross_dataset_eval/v2_batch_armA_lrtype_*`)**: type accuracy 0.77 -> 0.84, label-free gene corr 0.888 -> 0.903 (label 0.926), train-shift mode 0.814 -> 0.833. Residual: Pvalb -> Sst 56 %, L6 IT <-> L5 IT split, L5 ET -> L6 CT 25 %; v2-specific confusions, not depth. Paper number for v2 label-free: 0.90 vs 0.93 labelled.

**Why:** the storyline assumed Ding 2020 had matched cells; the GEO design shows it does not. Bakken is the only public mouse cortex matched set outside AIBS 10x.
**How to apply:** quote Bakken with the ceiling and the training-delta baseline in the same table; never claim the model beats the per-type shift on external data; SMART-seq input must be depth-thinned (model takes log1p raw counts). See [[project-libsize-covariate]], [[project-benchmark-redo]], [[project-scorch-scoping]].
