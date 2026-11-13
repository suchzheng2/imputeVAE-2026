# Ding 2020 (GSE132044) mouse cortex: nuclei-only, four chemistries, bulk reference (2026-10-26)

Analysis-task 7. **Correction to the plan**: the GEO series design states that the mouse cortex sample was profiled
with four *single-nucleus* methods only (10x Chromium v2, DroNc-seq, sci-RNA-seq, Smart-seq2; two experiments
Cortex1 / Cortex2 run on different days) plus a bulk RNA-seq control per experiment. **There are no matched whole
cells**, so Ding cannot be the matched hold-out (that is Bakken 2018, `notes/experiments/bakken2018_holdout/`).
It is the nuclei-only application / chemistry-robustness test: same tissue, four chemistries, one whole-tissue
bulk profile per experiment as the only whole-cell reference.

## Data (`analysis/external/ingest_ding2020.py`; `data/external/ding2020_GSE132044/`)

GEO count matrix 28,692 genes x 15,175 nuclei; 9,666 / 10,134 training genes present (zero-filled otherwise,
excluded from metrics); nuclei with < 500 counts over the training genes dropped -> **14,255**. Libraries over the
training genes (median): 10x v2 3.2-6.2 k, DroNc-seq 1.8-2.7 k, sci-RNA-seq 2.0-3.2 k UMIs (training nuclei 15.4 k);
Smart-seq2 ~1 M reads, binomially thinned to the training-nuclei library distribution (14-17 k). Bulk: TPM over the
present genes, renormalised, log1p.

## Method (`analysis/external/eval_ding2020.py`; results `run/external_eval/ding2020/`)

Per group (chemistry x experiment): training type predicted label-free with the depth-robust reference classifier
(`analysis/external/type_classifier.py`, log1p-CPM / 3,000 HVG / PCA 50 / logistic regression on training nuclei;
v3 validation accuracy **0.982**), covariate model arm A decoded at the training per-type cell library (quantile map
within type: the nuclei-only decode of the benchmark), plus the paper model (`source`). Rows: imputed (full
regeneration); model delta on nucleus (X_nuc + cell decode - nucleus decode); **nucleus + training per-type delta**
of the predicted type (baseline). Metric: pseudobulk (mean counts -> CPM -> log1p) vs the bulk of the same
experiment, Pearson / Spearman over 9,666 genes; and (bulk - nuclei pseudobulk) vs (row - nuclei pseudobulk),
Pearson and sign agreement on |delta| > 0.25. Composition is not matched (bulk = all cells of the tissue, nuclei =
whatever the chemistry captured), so the comparison is relative: does imputation move the pseudobulk toward the
whole-tissue RNA, and does the per-type training delta do the same?

### Pseudobulk vs bulk, Pearson of log CPM (arm A, reference-classifier types)

| group | n | lib | raw nuclei | imputed | model delta on nucleus | nucleus + training delta | sign agr. imputed / delta / training delta |
|---|---|---|---|---|---|---|---|
| 10x v2, Cortex1 | 1,476 | 6.2 k | 0.27 | 0.63 | 0.52 | 0.58 | 0.72 / 0.72 / 0.75 |
| 10x v2, Cortex2 | 3,953 | 3.2 k | 0.34 | 0.61 | 0.54 | 0.62 | 0.70 / 0.70 / 0.75 |
| DroNc-seq, Cortex1 | 2,153 | 1.8 k | 0.56 | 0.62 | 0.65 | 0.68 | 0.62 / 0.64 / 0.68 |
| DroNc-seq, Cortex2 | 935 | 2.7 k | 0.55 | 0.61 | 0.64 | 0.66 | 0.61 / 0.64 / 0.67 |
| Smart-seq2, Cortex1 | 295 | 14 k | 0.52 | 0.60 | 0.65 | 0.66 | 0.65 / 0.64 / 0.66 |
| Smart-seq2, Cortex2 | 349 | 17 k | 0.50 | 0.58 | 0.63 | 0.63 | 0.64 / 0.63 / 0.63 |
| sci-RNA-seq, Cortex1 | 1,752 | 3.2 k | 0.29 | 0.62 | 0.53 | 0.60 | 0.74 / 0.71 / 0.77 |
| sci-RNA-seq, Cortex2 | 3,342 | 2.0 k | 0.37 | 0.60 | 0.55 | 0.63 | 0.70 / 0.68 / 0.75 |

Paper model (no covariate, own library): imputed 0.58-0.63, delta on nucleus 0.56-0.69, same pattern. Spearman
tracks Pearson (raw 0.27-0.55 -> imputed 0.55-0.61). Predicted gene-class deltas: ribosomal +0.7 to +1.8 (positive in
all 8 groups), nuclear lncRNA -0.24 to +0.49 (negative only on 10x v2; Smart-seq2 +0.4-0.5).

**Chemistry robustness of the predicted transformation**: per-gene mean predicted delta (imputed - nucleus) per
group, Pearson across the 8 groups: same chemistry (Cortex1 vs Cortex2) 0.96-0.99, different chemistries median
0.82 (min 0.70; matrix `predicted_delta_cross_chemistry_pearson_armA.csv`); paper model 0.91 median.

**Type assignment**: the latent kNN vote (the benchmark's label-free decode, 97 % on v3 test) does **not** transfer
to these shallow / foreign-chemistry nuclei: it assigns 14-23 % L6b, 12-33 % L5/6 NP or L6 CT and < 1 % astrocytes on
the droplet methods (`predicted_type_composition_latent_knn.csv`), while the reference classifier gives a plausible
cortex composition on every chemistry (L5 IT 20-29 %, L2/3 IT 5-17 %, L6 CT/IT 8-12 % each, Astro 3-13 %, Oligo
4-9 %, OPC 5-30 %, Sst 6-16 %, Pvalb 3-5 %; `predicted_type_composition_pca_lr.csv`; mean confidence 0.84). The
pseudobulk metrics barely depend on which predictor supplies the library (differences <= 0.01).

## Reading

1. **Imputation moves nuclei-only data toward whole-tissue RNA on all four chemistries** (raw 0.27-0.56 ->
   0.58-0.65), including chemistries never seen in training (DroNc-seq, sci-RNA-seq, Smart-seq2), and the predicted
   transformation is the same across chemistries (0.82). This is the application-level statement for a nuclei-only
   atlas, and it holds without labels.
2. **The per-type training delta achieves the same** (0.58-0.68, sign agreement 0.63-0.77, >= the model in every
   group). As in Bakken 2018 and the split-aware benchmark, what transfers is the cell-type-level compartment shift;
   the model's contribution is a label-free, chemistry-robust way to apply it, not a better shift.
3. **Full regeneration vs delta on nucleus** depends on depth: at 10x-like depth (10x v2, sci-RNA-seq) regeneration
   is better (0.60-0.63 vs 0.52-0.55, it denoises sparse profiles); on DroNc-seq and Smart-seq2 the delta-on-nucleus
   mode is better (0.63-0.65 vs 0.58-0.62). Report both, recommend regeneration for shallow droplet data and the
   residual mode for deep / full-length data.
4. **The latent is not a cross-chemistry type classifier**: the kNN vote that works on v3 (97 %) and roughly on v2
   (77 %) breaks on 2-6 k UMI nuclei. For nuclei-only applications the type should come from a reference-mapping
   classifier on the nuclei (as here); this also resolves the v2 cross-chemistry gap noted in the storyline
   (`paper_storyline_2026-10-23.md` gap 3) and should be re-run there.
5. Limits: one bulk profile per experiment, composition unmatched, no per-type truth; the Smart-seq2 lncRNA sign is
   wrong (full-length poly-A capture of Malat1 differs from 3' chemistries). Not a figure-grade validation; a
   supplementary application panel at most.

## Figure (2026-11-06)

`figures/fig_ding_pseudobulk_vs_bulk.png`: pseudobulk-vs-bulk Pearson per chemistry x experiment for raw nuclei / imputed / delta-on-nucleus / training per-type delta, and the 8 x 8 cross-group Pearson of the predicted per-gene delta. Source: `run/external_eval/ding2020/eval_armA_lrtypeq.json`.
