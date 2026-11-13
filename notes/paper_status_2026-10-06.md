# Paper status: what we have and what is missing (2026-10-06)

Survey of the repo as of 2026-10-06. Model under test throughout: single-stage cVAE
`run/run_2026-07-18-19-40-51_570984` (scalar modality condition on encoder + decoder, MSE + JSD,
log1p, Config B 10,134 genes, **32-dim latent**, 21.9M parameters, one seed).

No manuscript draft exists yet. Existing writing: ISMB 2026 abstract (describes the *old* two-stage,
16,439-gene, cross-recon model), pre-qual proposal (May), qual reading guide, Aim-1 figure set
(`validation/aim1_biological_validation_figures/`).

---

## 1. What we have, organized as candidate paper figures

### Fig 1. Problem and method
- Schematic exists (`validation/aim1_biological_validation_figures/aim1_figure_A1_1_validation_schematic.*`).
- Design-choice evidence from the 16-run factorial (`notes/experiments/session_2026-06-12_results.md`,
  `notes/design_decisions.md`): log1p > CPM, enc+dec > dec-only, scalar ~ learned 8-d, normloss allows
  single-stage training, cross-recon dropped because it compresses variance.

### Fig 2. Benchmark against 6 baselines (`notes/experiments/baseline_comparison.md`, eval JSONs in `run/*/eval/`)
| Metric | cVAE | Lin. reg | CT-mean | scVI | scANVI | scGen | scButterfly |
|---|---|---|---|---|---|---|---|
| Gene corr (median) | 0.927 | 0.822 | 1.0* | 0.386 | 0.373 | 0.501 | 0.786 |
| R2 (median) | 0.798 | 0.672 | 1.0* | -0.67 | -0.75 | -0.69 | -0.87 |
| Delta corr | 0.950 | 0.860 | 1.0* | 0.524 | 0.533 | 0.565 | 0.614 |
| Classification | 0.963 | 0.926 | 1.0* | 0.929 | 0.936 | 0.946 | 0.995 |
| Variance ratio | 0.158 | 3.1 | 0 | 0.034 | 0.032 | 0.031 | 0.006 |
| Wasserstein | 0.300 | 0.268 | 0.399 | 0.947 | 0.928 | 0.922 | 0.991 |
| Compartment AUROC | 0.991 | 0.984 | 0.995 | 0.842 | 0.829 | 0.834 | 0.803 |

scGen and scButterfly were run 2026-07-24 but are missing from `baseline_comparison.md`.
**Caveat: none of these numbers are split-aware (see Gap A1).**

### Fig 3. The learned delta is compartment biology
- Ribosomal 100% sign, translation 100%, nuclear ncRNA 92%, compartment AUROC 0.991
  (neurons 0.997, glia 0.956) (`notes/experiments/compartment_classification/`).
- GO: ribosome p=3e-26, synapse p=3e-14, cytoplasm p=8e-08 (`analysis/go_analysis.py`).
- Cell-type specificity: Plp1 in Oligo, Mt1 in Astro, neuron/glia cytoplasmic ratio 4.4x (scVI 1.3x).
- Scrambled-delta control z=39.8 (`notes/experiments/negative_control/`).
- Gene-stratified: gain largest in top gap quartile (+0.125 corr), no harm in bottom (`gene_stratified_eval/`).
- SynGO positive control (emp diff 0.673) vs JASPAR TF negative control (0.257) (`validation/syngo`, `validation/tf`).

### Fig 4. Non-circular external validation (mouse)
- Fractionation-seq: Buckley 2016 mouse rho 0.44, Moutinho 2021 human rho 0.40 (`fractionation_validation/`).
- TRAP/RiboTag: H1 GSE210071 rho 0.38-0.41, AUROC 0.80-0.82; H2 Pvalb/Sst 0.43 -> 0.73 (ceiling 0.74);
  extension to 9 newer sets / 19 populations holds; L5 ET/IT mapping audited against literature and
  Yao 2021 external markers (`validation/trap_ribotag_external_validation/`).
- Dissociation/IEG audit (`dissociation_audit/`), L5 ET diagnosis (`l5et_diagnosis/`).

### Fig 5. Generalization
- Unseen v2 chemistry, same tissue: QQ 0.894, Wasserstein 0.263 hold; gene corr 0.59, R2 -0.03 drop
  (`notes/experiments/cross_dataset_validation.md`).
- Human M1 zero-shot (76,533 nuclei, 8,215 genes): neuronal > glial shift, SynGO diff 0.243,
  mouse-human r=0.715, donor r=0.986 (`validation/human_m1_transfer/`).
- GSE219278 sorted human populations: SynGO delta 1.8x genome-wide, TF at genome-wide level
  (`validation/gse219278_sorted_population_validation/`).
- Lake 2023 kidney: conserved ribosomal-up / MALAT1-NEAT1-down core; genome-wide r=0.03
  (`conserved_transformation/`).

### Fig 6 (or supplement). Honest per-cell characterization
- Variance: rank kept (Spearman 0.92 L5 IT, 0.93 Oligo, 0.85 pooled), distribution stretched not
  collapsed; but like-for-like retention 0.12, 82% of imputed variance is the library-size axis,
  5-6 of 32 latent dims active (`variance_zero_nonzero_fullmodel/`, `variance_rank/`, `latent_usage/`).
- Zeros treated as dropouts (imputed value barely depends on nucleus zero vs nonzero).

---

## 2. Gaps, ranked by how much a reviewer would care

### A. Must fix before any submission

**A1. The benchmark is not split-aware.** `run/eval_v3_splitaware/audit_report.md` (2026-08-06) found
all baselines trained on all data and all metrics computed on all nuclei, including the cVAE's own
60% training nuclei. The fix was started and never finished:
- `jobs/eval/run_eval_splitaware.sh` (job 8575090) computed every cVAE validation metric, then crashed on
  `assert var < 1e-8` for the CT-mean baseline (actual 2.6e-6, float32 noise) before writing any JSON.
- Benchmark retrains (jobs 8578258-62): CT-mean + lin. reg finished; scVI, scANVI, scGen failed with
  "scvi-tools not installed" (wrong env in job script); scButterfly was OOM-killed.
- Also: split is per observation, so every donor is in train and valid. A leave-one-donor-out check
  would answer the obvious reviewer question.

**A2. The method advantage rests on per-cell structure, where the model is weakest.** On pseudobulk /
cell-type-level tests, CT-mean and linear regression match or beat the cVAE (compartment AUROC 0.995 /
0.984 vs 0.991; TRAP H2 margins 0.001-0.03, lin. reg wins on glia and Vip). What only a per-cell model
can offer is within-type variation, and that is currently one library-size axis. The planned retrain
(library size as an explicit covariate + free-bits or weaker JSD, then re-run the variance diagnostic)
has not been started. Either this retrain works, or the paper must be framed around the delta/biology
and the cell-type-level translation rather than per-cell imputation.

**A3. No downstream use case.** Gerstein review item 1 and SUC-10 are still open. Candidates: PsychENCODE
or other nuclei-only disease DE re-analysis (does a DE result change?), cell-type eQTL, atlas
integration of cell + nucleus data. Without this, the paper is a translator plus validation.

**A4. Ablations on the final architecture.** Tasks 11-13 not started: remove conditioning, remove JSD,
NB/ZINB decoder, latent size, held-out cell type (N-1). Existing ablations come from the factorial
(older runs) and the old two-stage MSE-vs-NB comparison, not the final model. Only one seed.

### B. Strongly expected for a methods journal

**B1. Matched sc/sn holdout datasets not started.** Ding 2020 (GSE132044; 10x, Drop-seq, Smart-seq2) and
Bakken 2018 (VISp; confirm accession) would test the transformation on independent matched data.
Scala 2021 Patch-seq gap-closing also not started.

**B2. Cross-chemistry result needs an answer.** On v2, cell-type-level R2 is -0.03. Explain (library size
/ sensitivity shift) or fix (size normalization from A2 may help).

**B3. Benchmark framing vs scButterfly.** cVAE has 21.9M parameters vs scButterfly 10.8M, so "parsimony"
cannot mean fewer parameters. Frame it as no translator, no discriminator, no pseudo-pairs, one
model; report training time. scButterfly wins classification (0.995 vs 0.963).

**B4. Confound and control analyses from the qual plan, not done:**
- Technical covariates: regress delta on gene length, GC, mean expression.
- Intron retention: overlap of negative-delta genes with Xia 2019 nuclear-retained list.
- Gene-filter bias: Config B drops low-detection genes, which are often the most compartment-shifted;
  most IEGs are absent from the gene set, so the dissociation audit is underpowered.
- Sex confound: cells are 6M + 6F, nuclei all female; Xist is top-variance in every type. Exclude or
  model sex-linked genes and report.
- Localization concordance with COMPARTMENTS is weak (rho 0.08, AUROC 0.55). The lncATLAS / RNALocate /
  APEX-seq comparison planned in Task 2 was not done; fractionation and TRAP carry this argument now.

**B5. Statistics.** No seed replicates, no bootstrap CIs on per-gene / per-cell-type metrics, few
permutation nulls outside the delta scramble.

### C. Packaging and writing

- **C1. No manuscript draft**, no figure plan, no methods section for the final model.
- **C2. Tool release.** No `pyproject.toml`, no CLI (h5ad in -> imputed h5ad out), no pretrained-weights release.
- **C3. Stale or conflicting notes.** `project_summary.md` (July), `project_scope_document.md` (April,
  two-stage, 16,439 genes), ISMB abstract, and the task list (says 128-dim latent) all describe
  different models. `baseline_comparison.md` lacks scGen / scButterfly. One canonical model
  description is needed before writing.
- **C4. ISMB reviewer points still open:** truncated y-axes, figure resolution, justification of JSD over KL.

---

## 3. Suggested order
1. A1: fix the assertion tolerance, point the scVI-family jobs at the right env, give scButterfly more
   memory, re-run the split-aware benchmark. Every table in the paper depends on it.
2. A2 retrain (library-size covariate + free bits), re-run the variance diagnostic. Its outcome decides
   the paper's framing.
3. A4 ablations and seeds, run alongside A2 on the same GPU queue.
4. B1 Ding 2020 (data already public, cheapest matched holdout), then A3 downstream use case.
5. C1-C3: freeze the model description, draft the paper around Figs 1-6 above.
