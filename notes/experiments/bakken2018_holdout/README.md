# Bakken 2018 matched nucleus/cell hold-out (GSE123454) — independent test of the transformation (2026-10-26)

Analysis-task 8 ("simulation replacement"). First evaluation of the model on a matched sc/sn dataset that is
not AIBS MOp 10x: Bakken et al. 2018 PLoS One, mouse **VISp**, **SMART-seq v4**, 463 nuclei + 463 cells
(cells are a subset of Tasic 2018 / GSE115746), **11 matched clusters** (L4/L5/L6 glutamatergic + Pvalb + Sst;
no glia). The nuclei and cells share cluster ids (cl1-cl11), so the per-type comparison needs no mapping to
our 17 training types. Region, platform (full-length plate vs 10x droplet) and lab all differ from training.

Note: **Ding 2020 (GSE132044) cortex has no whole cells** — the GEO design says the mouse cortex sample was
profiled with four *single-nucleus* methods only (10x v2, DroNc-seq, sci-RNA-seq, Smart-seq2) plus bulk RNA-seq.
The storyline's "Ding 2020 gives both" (gap 4) was wrong; Ding is the nuclei-only / chemistry test
(`notes/experiments/cross_dataset_ding/README.md`). Bakken 2018 is the matched test.

## Data and ingestion (`analysis/external/ingest_bakken2018.py`)

- GEO supplementary exon and intron count tables + the GEO submission sheet (cell_cluster, QC flag, donor, layer).
  42 cell columns are listed under a different sequencing-run suffix (`_E1` vs `_E2`); matched by plate/well.
  QC "Fail" dropped (2 nuclei, 17 cells) -> **461 nuclei, 445 cells**, 11 clusters (9-119 per cluster per modality).
- Counts: exon + intron (default; nuclei are 58 % intronic, cells 36 %); exon-only as a robustness variant.
- Gene space: training list v3 (10,134). **8,928 present** in the 2018 annotation; 1,206 absent (452 `Gm*`,
  `*Rik`), zero-filled on input and **excluded from every metric** (`var['present']`).
- Depth: SMART-seq libraries are ~1-2 M reads over the training genes vs 15 k (nuclei) / 42 k (cells) UMIs in
  training, and the model takes log1p(raw counts). Each nucleus / cell is binomially thinned to a library
  quantile-matched to the training partition of its modality (seed 0; medians 15,488 / 42,126 vs 15,450 / 42,177).
  This matches depth, not gene-length bias (full-length reads inflate long genes relative to 3' UMIs); the empirical
  delta (cell - nucleus, both SMART-seq) is internally consistent, the model output is 10x-like.
- Files: `data/external/bakken2018_GSE123454/processed/{nuclei,cells}_{exon_intron,exon}.h5ad`.

## Evaluation (`analysis/external/eval_matched_external.py`, results `run/external_eval/bakken2018[_exon]/`)

Rows per run: **model** (full regeneration: encode nucleus, decode as cell), **model delta on nucleus**
(X_nuc + [cell decode - nucleus decode of the same latent]: the observed nucleus plus the model's own
nucleus->cell shift; keeps the input's platform profile), **raw nuclei**, **training CT-mean of the predicted
type**, **nucleus + training per-type delta** (mean train cell - mean train nucleus of the predicted type; the
"type mean + per-gene shift" baseline). Predicted type = majority vote over the 50 nearest training nuclei in the
latent (no label at inference; the same label-free decode as the benchmark). Covariate model (arm A,
`run_2026-10-16_libsize_armA_jsd0.1_seed1`) decoded with the Bakken cluster's own cell library
(`quantile_map_target`, uses the cluster label), at the training per-type median of the predicted type
(`knn_type_train_shift_target`, no label), pooled quantile map, latent-kNN and `source`; paper model
(`run_2026-07-18-19-40-51_570984`) for reference.

Core metric: **predicted delta (imputed - nucleus) vs empirical delta (cell - nucleus) per cluster**, Pearson /
Spearman over the 8,928 present genes, pooled over clusters, and sign agreement on genes with |empirical| > 0.25.
Ceiling from split-half reliability of the empirical delta (`analysis/external/delta_reliability.py`, 50 splits):
pooled split-half r 0.857 -> **full-sample reliability 0.92** (Spearman 0.79; sign agreement 0.94); per-cluster
median 0.94 (0.85 for the smallest cluster).

### Results (exon + intron; pooled over 11 clusters)

| row | label at inference | delta Pearson (pool) | per-cluster median | sign agreement | cluster-mean delta r | Spearman (pool) | overall corr | W1 | per-cell corr | ribo / cyto / lncRNA sign |
|---|---|---|---|---|---|---|---|---|---|---|
| ceiling (split-half) | | 0.92 | 0.94 | 0.94 | | 0.79 | | | | |
| paper model, full regeneration | no | 0.41 | 0.43 | 0.83 | | | 0.742 | 0.455 | 0.754 | 1.00 / 1.00 / 0.39 |
| paper model, delta on nucleus | no | 0.52 | 0.52 | 0.94 | 0.71 | | 0.807 | 0.252 | 0.706 | 1.00 / 1.00 / 1.00 |
| arm A, quantile map at cluster cell library, full | yes | 0.48 | 0.48 | 0.91 | | | 0.761 | 0.387 | 0.749 | 1.00 / 1.00 / 0.39 |
| **arm A, same, delta on nucleus** | yes | **0.58** | 0.54 | 0.94 | **0.73** | 0.59 | **0.822** | **0.243** | 0.708 | 1.00 / 1.00 / 1.00 |
| arm A, predicted-type training library, full | no | 0.43 | 0.46 | 0.90 | | | 0.747 | 0.541 | **0.762** | 1.00 / 1.00 / 0.33 |
| arm A, same, delta on nucleus | no | 0.54 | 0.51 | 0.94 | | | 0.809 | 0.272 | 0.720 | 1.00 / 1.00 / 1.00 |
| arm A, pooled quantile map, delta on nucleus | no | 0.55 | 0.51 | 0.95 | | | 0.812 | 0.245 | 0.713 | |
| arm A, `source` (nucleus's own library), delta on nucleus | no | 0.52 | 0.51 | 0.71 | | | 0.800 | 0.390 | 0.641 | |
| raw nuclei | | 0 | 0 | 0 | | | 0.716 | 0.419 | 0.548 | |
| training CT-mean of predicted type | no | 0.43 | 0.46 | 0.91 | | | 0.748 | 0.569 | 0.765 | 1.00 / 1.00 / 0.33 |
| **nucleus + training per-type delta** (predicted type) | no | **0.56** | **0.59** | 0.94 | 0.73 | 0.56 | 0.813 | 0.287 | 0.715 | 1.00 / 1.00 / 1.00 |

Exon-only counts give the same picture (arm A delta on nucleus 0.61 pooled / 0.60 median; training delta 0.56 / 0.65;
full regeneration 0.29-0.38). "Gene corr across clusters" (the benchmark's headline metric) is 0.1-0.4 for every
row including raw nuclei (11 closely related clusters, 8.9 k genes, 10-100 cells each) and is not informative here.
Per-cluster table: `per_cluster_summary_exon_intron.csv`.

Per cluster (arm A delta on nucleus / training delta / ceiling): L5a.Batf3 0.65 / 0.66 / 0.97; L4.Arf5 0.80 / 0.77 /
0.95; L4.Hsd11b1 0.54 / 0.56 / 0.97; L6.Car12 0.43 / 0.42 / 0.96; Pvalb 0.46 / 0.47 / 0.95; L5b.Cdh13 0.52 / 0.59 /
0.91; L6a.Plcxd3 0.48 / 0.45 / 0.94; L5b.Samd3 0.50 / 0.56 / 0.94; Sst 0.69 / 0.66 / 0.88; L6a.Mgp 0.69 / 0.69 / 0.88;
L5.Chrna6 0.62 / 0.61 / 0.85. Slope of predicted on empirical 0.3-1.1 (median 0.4): magnitudes are shrunk, directions right.

Predicted training type per Bakken cluster (majority / purity), latent kNN vote: Pvalb.Wt1 -> Pvalb 0.94;
Sst.Cbln4 -> Sst 0.91; L5b.Cdh13 -> L5 ET 0.94; L5b.Samd3 -> L5 ET 0.97; L6a.Mgp -> L6 CT 0.95; L6a.Plcxd3
(Slc17a8) -> L5/6 NP 1.00; L4.Arf5 / L4.Hsd11b1 / L5a.Batf3 -> L5 IT 0.97 / 0.97 / 0.78 (MOp has no L4; the VISp
L4 types map onto the L5 IT / IT continuum); L6.Car12 -> L6 IT 0.51; L5.Chrna6 -> L5/6 NP 0.55 (11 nuclei). The
latent assigns an unseen region and platform to the expected training types without any label.
**Depth-robust reference classifier** (`analysis/external/type_classifier.py`: log1p-CPM over the present genes,
3,000 HVGs, PCA 50, logistic regression on the training nuclei; `--type_predictor pca_lr`; validation-partition
accuracy in `run/external_eval/bakken2018/eval_armA_lrtype.json`): purities rise to L5 IT 0.98-0.99 for the three
L4/L5a clusters, L5 ET 1.00 for L5b.Cdh13, L5b.Samd3 **and L5.Chrna6** (Chrna6 is an L5 PT/ET marker; the kNN vote
had it as NP), Sst 0.91, L6 CT 0.95, Pvalb 0.83; L6.Car12 stays split (L6 IT 0.51). The delta metrics do not change
(0.53 pooled / 0.51 median; training-delta baseline 0.55 / 0.59): the type assignment was already good enough
where it matters. Crosstabs: `predicted_type_crosstab_paper_source.csv` (kNN), `predicted_type_crosstab_pca_lr.csv`.

Gene classes (arm A, delta on nucleus): ribosomal (53 genes) empirical mean delta +1.12, predicted +1.69, 100 %
sign both; cytoplasmic (17) +2.12 / +2.81, 100 %; nuclear lncRNA (18 of 45 present) empirical **+0.17** (0 % expected
sign: at the depth-matched log1p scale every gene gains ~+1 from the 2.7x larger cell library, so the lncRNAs are
*relatively* depleted but not absolutely), predicted -0.41 (100 %): the model over-depletes nuclear lncRNAs on
SMART-seq input (Malat1 is captured differently by poly-A full-length vs 3' chemistry).

## Reading

1. **The transformation transfers to an independent region, lab and platform**: direction of the nucleus->cell
   change is predicted at the split-half ceiling (sign agreement 0.94 vs ceiling 0.94; ribosomal / cytoplasmic
   100 %), and the gene-level magnitude correlates 0.58 pooled / 0.73 composition-free against a ceiling of 0.92,
   i.e. ~40 % of the reliable per-cluster delta variance; per-cluster 0.43-0.80.
2. **The model is level with the per-type training delta** (0.56 / 0.59, same sign agreement, same cluster-mean r
   0.73). On this dataset a learned 22M-parameter map adds nothing over "predict the type, add that type's mean
   nucleus->cell shift"; what the model contributes is the label-free type assignment that the baseline also
   uses. This is the same conclusion as the split-aware benchmark, now on external data.
3. **Full regeneration is the wrong way to apply the model across platforms** (0.41-0.48, lncRNA sign 0.33-0.39,
   W1 0.39-0.54): the decoder emits a 10x-like profile and the platform difference enters the delta. Adding the
   model's own shift to the observed nucleus (delta on nucleus) is the cross-platform mode: best W1 (0.24),
   overall corr 0.82, lncRNA 100 %. For the per-cell metric (imputed vs the cluster's cell mean) regeneration is
   still best (0.76 vs 0.55 raw), i.e. regeneration denoises, the residual mode transfers.
4. The remaining gap to the ceiling (0.58 -> 0.92) is where the empirical delta is cluster-specific beyond what
   the training types encode: the three clusters the model maps to one training type (L4.Arf5, L4.Hsd11b1,
   L5a.Batf3 -> L5 IT) have different empirical deltas (r 0.54-0.80 against one predicted delta), and the two
   with mixed assignment (L6.Car12, L5.Chrna6) are the lowest. A finer identity code (or VISp types in training)
   would be needed, not more capacity.
5. For the paper (storyline Fig 5 / gap 4): report Bakken as the independent matched validation with the
   ceiling, the training-delta baseline in the same table, and the residual decode as the recommended
   cross-platform mode. State the depth matching and the 1,206 absent genes.

## Open

- Bakken is SMART-seq; the only 10x matched external set would be a second AIBS region (VISp 10x nuclei + cells
  exist in the Yao 2021 whole-cortex data, but they are the same lab/protocol family as training).
- Composition-free ceiling (cluster-mean split-half) not computed; the pooled ceiling is the conservative one.
- Label-free type assignment across platform: the reference classifier (`pca_lr`) resolves L5.Chrna6 and the L4
  clusters; L6.Car12 (41 nuclei) remains split between L6 IT and L6 CT, which may be a true VISp/MOp difference.
  The same classifier is what the Ding 2020 nuclei-only evaluation uses. **Done for v2 2026-10-27** (job 14189131,
  `cross_dataset_eval.py --type_predictor pca_lr`): type accuracy 77 % -> 84 %, label-free gene corr 0.888 -> 0.903
  (label 0.926); remaining errors are Pvalb -> Sst and L6 IT <-> L5 IT on v2 input (libsize review §6.4a).

## Figure (2026-11-06)

`figures/fig_bakken_delta_per_cluster.png`: per-cluster Pearson of predicted vs empirical delta (covariate model delta-on-nucleus, training per-type delta baseline, paper model) against the split-half reliability ceiling, plus the pooled bars. Source: `run/external_eval/bakken2018/per_cluster_summary_exon_intron.csv`, `delta_reliability_exon_intron.json`.
