# Project Memory - imputeVAE

## Project Overview
- **Working dir:** `/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE`
- **Conda envs:** `impute2` (main), `abc_env` (scib metrics only)
- **Platform:** HPC cluster (SLURM), partition `pi_gerstein`, `scavenge_gpu`, `priority_gpu`
- **Default partition:** `scavenge` (user preference — always use scavenge unless specified)
- **GPU jobs needing 400G:** use `--partition=pi_gerstein_gpu,scavenge_gpu` — `pi_gerstein_gpu` is group-owned, non-preemptible, includes r813u29n03 (1 TB, 4×A100); scavenge_gpu alone waited 4 days for a 400G slot (2026-10-16). The cVAE full-data config peaks at 326 GiB RSS. **r813u29n03 GPU 0000:31:00.0 is broken** (nvidia-smi Unknown Error; training silently falls back to CPU) — submit there with `--gpus=2`; the libsize job script has a healthy-GPU guard to copy.
- **Script path note:** Scripts are at `analysis/` relative to project root

## Benchmark Redo (2026-10-09) — DONE incl. cVAE seed replicates (2026-10-12; 24 method dirs)
- [Benchmark redo](project_benchmark_redo.md) — split-aware pipeline in analysis/benchmark, review+results in notes/benchmark_review_2026-10-09.md (§5a = seeds). KEY: label-using baselines (scGen 0.975, scANVI 0.962) beat cVAE 0.900 ± 0.002 (3 runs) at CT level; label-free scVI/scANVI 0.86 trail it. cVAE within-type variance 0.151 ± 0.009 = library pass-through (tied with label-free scVI). `--seed` does not pin GPU nondeterminism.
- [Library-size covariate](project_libsize_covariate.md) — DONE 2026-10-23: labelled decode 0.970 / R² 0.93; label-free predicted-type decode 0.962 / 0.91 / energy 2.6 (reported nuclei-only number); no labels anywhere 0.921; v2 label-free 0.89 (type prediction 77 %); latent width: 2 dims suffice, 32 kept. Record `notes/libsize_covariate_review_2026-10-09.md` §6.4-6.7, benchmark review §5b, storyline `notes/paper_storyline_2026-10-23.md`
- [External matched validation](project_external_matched_validation.md) — DONE 2026-10-26: Bakken 2018 (GSE123454) matched hold-out: sign at ceiling, delta r 0.58 (ceiling 0.92) = per-type training delta; residual decode for cross-platform; Ding 2020 has NO cells (nuclei-only 4-chemistry test, pseudobulk 0.3-0.5 -> 0.6); latent kNN type vote fails off-chemistry -> `analysis/external/type_classifier.py` (0.982); v2 label-free re-run with it 2026-10-27: 0.90 (type 84 %), residual Pvalb->Sst confusion
- [Spatial application](project_spatial_application.md) — Xenium Prime 5K mouse brain on palmer scratch (hippocampal level, no MOp -> subclass only); 2026-11-13 in-situ nuclear fraction vs GT delta Spearman -0.57 / AUROC 0.90; next subclass label transfer + gene-filling; record `notes/experiments/spatial_merfish_application/`
- [SCORCH scoping](project_scorch_scoping.md) — human snRNA-seq 655k nuclei (PFC/INS/VST, HIV vs CTR) on the group share; no mouse spatial data on the HPC; the consortium's mouse spatial set = Allen SCORCH EcoHIV+cocaine MERFISH (NeMO portal), Yale's own spatial is human and lives on cantor:/banach2/SCORCH (2026-11-06); candidate nuclei-only application
- **Disk:** group quota 4087/4096 GiB (2026-10-15); ~37 GB of preempted partial run folders listed in `notes/training_runs.md` (2026-10-16 entry) await deletion by the user

- [Prashant meeting 11-06](project_prashant_meeting_1106.md) — variance questions answered (L5 IT per-gene distribution; Poisson-sampled decode recovers 75 % of real variance at rank 0.96; cost function unchanged, decode +7 pts), validation praised; OPEN: human single-cell motor cortex data ask (email draft in `notes/meeting_2026-11-06_prashant.md`)
- [Prashant update 10-30](project_prashant_update_1030.md) — status/storyline/gaps note `notes/prashant_update_2026-10-30.md`; open decisions: freeze covariate model, SCORCH application, ablations, manuscript skeleton; nothing launched yet
- [Ablations final model](project_ablations_final_model.md) — DONE 2026-11-06: JSD optional (0.964), alignment essential (0.947, clf 0.92), enc-cond +0.005; N-1 type delta r 0.93/0.85 with identity collapse; donor hold-out ~0 cost. Chain (after leak fix): predict 15559184_3 / 15559684_[4-6] -> evaluate 15559685; results -> `run/benchmark/test/results_ablations/`; README `notes/experiments/ablations_final_model/`. Library covariate run (train 15562732 -> evaluate 15563057; `results_libcond/`) read 2026-11-13: 0.956 vs 0.970, L5 ET identity lost, but female/male target-library decode reproduces sex (Xist 3.6 vs 0.0) -> supplementary only
- [CBB RIP talk](project_cbb_rip_talk.md) — Nov 2026 research-in-progress talk, prepared 2026-10-28: deck + figures + 23-min script + host intro in `notes/presentation/CBB_RIP_2026-11/`; final deck keeps the full-dataset benchmark table (speak the 0.90 vs 0.86 split-aware caveat)

## Paper Status (2026-10-06)
- [Paper status](project_paper_status.md) — inventory + ranked gaps; split-aware benchmark unfinished is #1

## Current Best Model
- **Best overall (full data)**: single-stage cVAE (scalar, enc+dec, log1p, 100%)
  - `run/run_2026-07-18-19-40-51_570984` (job 7714771)
  - R² 0.798, gene corr 0.927, delta corr 0.950, variance 0.158, classif 96.3%
  - Eval: `run/cvae_full_eval/eval/eval_cvae_full.json` (v2: `eval_cvae_full_v2.json`)
- See [experiment_history.md](experiment_history.md) for full model/job details

## Prashant Meeting (Sep 2026, summarized 2026-09-24) — NEXT STEPS
- [Meeting memory](project_prashant_meeting_0924.md) — notes: `notes/meeting_2026-09-24_prashant.md`
- TRAP validation "very promising" but: drop L4 comparison, decide L5a/L5b ET vs IT from projection evidence (not by name), or report vs all L5 types
- Variance: per-CT per-gene variance distributions + Spearman(var_real, var_imputed), L5 IT and Oligo first
- **Both done 2026-09-24**: L5 mapping audit (`validation/trap_ribotag_external_validation/outputs/extension/metadata/l5_mapping_evidence.md`, all lines agree with literature; L4 excluded; Rbp4 -> IT+ET pool; rerun job 11168301) and variance rank check (`notes/experiments/variance_zero_nonzero_fullmodel/variance_rank/`, Spearman 0.92 L5 IT / 0.93 Oligo, stretched not collapsed)
- Next meeting Friday following week (likely 2026-10-02)

## Human M1 Zero-Shot Transfer (2026-08-16/17) — DONE
- See [human_m1_transfer.md](human_m1_transfer.md) for full details
- Location: `validation/human_m1_transfer/`
- 76,533 Allen Human M1 nuclei, 8,215/10,134 genes mapped via MGI orthologs
- Predicted shifts: neuronal > non-neuronal (consistent with mouse)
- SynGO (1,236 genes): neuronal=0.775, diff=0.243; Mouse-human r=0.715 (17 classes)
- Donor consistency r=0.986; 34/34 acceptance checks passed
- SLURM jobs: 9231013 (prep), 9231508 (inference), 9243944 (rerun with fixes)

## SynGO / JASPAR TF / L5 ET Diagnostic (2026-08-15)
- See [validation_syngo_tf_l5et.md](validation_syngo_tf_l5et.md) for full details
- **SynGO** (1,275 synaptic genes): emp_diff=0.673, positive control confirmed
- **JASPAR TF** (63 TF genes): emp_diff=0.257, 2.6x weaker (specificity)
- **L5 ET:** +0.47 intercept bias, 96% overpredicted — interpret cautiously
- Locations: `validation/syngo/`, `validation/tf/`, `validation/l5et_diagnostic/`

## Compartment Classification (2026-08-05)
- Script: `analysis/compartment_classification.py` (job 8272238)
- **ImputeVAE AUROC 0.991**, 100% ribosomal sign agree, 92% nuclear ncRNA
- Per-CT: neurons 0.997, non-neuronal 0.956; scrambled z=39.8
- Results: `compartment_classification/`

## TRAP Cross-Validation (2026-08-05, Task 10)
- Script: `analysis/trap_crossvalidation.py`
- CT-specific TRAP: rho=0.80 (PV), 0.81 (SST), 0.79 (VIP) — STRONG
- Global IP/Input: negative rho (-0.15) — expected (TRAP ≠ compartment)
- Results: `compartment_classification/trap_validation/`

## Fractionation Validation (2026-08-06)
- Script: `analysis/fractionation_validation.py` (job 8558579)
- Buckley mouse brain rho=0.441, Moutinho human FC rho=0.395
- Gold-standard non-circular validation (biochemical N:C separation)

## Baseline Comparison (updated 2026-07-25)
- Results: `notes/experiments/baseline_comparison.md`
- Evals: `run/{ctmean,scvi,scanvi,cvae,linreg}_baseline/eval/`
- cVAE full-data: `run/cvae_full_eval/eval/eval_cvae_full.json`
- scANVI ≈ scVI; labels don't help (job 7909104)

## Conserved Transformation (Task 14, 2026-07-24)
- Script: `analysis/conserved_transformation.py`
- Kidney (Lake 2023): ribosomal positive (82%), MALAT1/NEAT1/XIST negative
- Cross-tissue Spearman: r=0.028 genome-wide (signal in tails only)
- Results: `notes/experiments/conserved_transformation/`

## Analysis Task Progress (from notes/ImputeVAE_analysis_tasks.md)
- **Phase 1 (reanalysis):** ALL DONE (Tasks 1-6)
- **Phase 2 (external validation):**
  - Task 10 — TRAP/RiboTag — **DONE** (2026-08-05)
  - Task 14 — Lake 2023 kidney — **DONE** (job 7900702)
  - Task 15 — scANVI baseline — **DONE** (job 7909104)
  - Task 7 — Ding 2020 cross-dataset — DONE 2026-10-26 (nuclei only; see external validation memory)
  - Task 8 — Bakken 2018 holdout — DONE 2026-10-26
  - Task 9 — Scala 2021 Patch-seq — NOT STARTED
  - Task 16 — scButterfly + scGen baselines — DONE 2026-07-24 (`run/{scbutterfly,scgen}_baseline/eval/`), not yet in baseline_comparison.md
- **Phase 3 (retraining):** NOT STARTED (Tasks 11-13)
- **Human M1 transfer (TG1+TG2):** DONE (2026-08-17)
  - **GSE219278 sorted-population validation:** DONE (2026-08-18)

## TRAP/RiboTag + Compartment External Validation (2026-09-18) — DONE
- Location: `validation/trap_ribotag_external_validation/` (README.md, outputs/report/)
- H1 GSE210071 rho 0.38-0.41, AUROC 0.80-0.82 (GT at same level); H2 Pvalb/Sst RiboTag 0.43->0.73 (ceiling 0.74); H3 rho 0.74-0.80
- Extension (9 newer RNA-seq sets, 19 populations): H2 holds everywhere incl. excitatory + glia; but pseudobulk H2 does not separate methods (negative result)
- See [project_trap_ribotag_external.md](project_trap_ribotag_external.md)

## GSE219278 Sorted-Population Validation (2026-08-18) — DONE
- Location: `validation/gse219278_sorted_population_validation/`
- GSE219278: FACS/FANS-sorted human motor cortex (6 donors x 3 populations x Control)
- 8,188 harmonized genes; SynGO 1,302; JASPAR TF 273
- Profile corr: low (r=0.14-0.30) — expected for cross-species/platform
- **SynGO positive control:** neurons delta=1.20 (1.8x genome-wide), sign agree 63-64%
- JASPAR TF: delta=0.60 (close to genome-wide 0.66) — correct negative control
- Reference matching: raw nuclei have diagonal dominance; imputed do not (neuron homogenization)
- Donor robustness excellent: M1 LOO |delta r|=0.0002, GSE LOO r=0.997
- SLURM: 9331550 (pipeline), 9331695 (rerun with JASPAR fix + figures)
- Report: `outputs/reports/gse219278_validation_report.md`

## Key Design Decisions
- See `notes/design_decisions.md`
- log1p > CPM; single-stage cVAE > two-stage; scalar ≈ emb8
- Deprioritizing cross-recons (xr) — compresses variance
- Normloss enables single-stage training (no annealing needed)

## GO Enrichment (2026-06-12)
- Script: `analysis/go_analysis.py`
- Increased: ribosome (p=3e-26), cytoplasm (p=8e-08), synapse (p=3e-14)
- Decreased: no significant GO terms (Gm* + lncRNAs)

## Variance Diagnostic, Full Model (2026-09-18) — DONE
- `notes/experiments/variance_zero_nonzero_fullmodel/README.md`
- Real-cell variance 52% Poisson / 25% library size / 23% biological; imputed variance is 82% library-size axis, ~1-D; like-for-like retention 0.12
- Latent-usage check: posterior collapse, 5-6 of 32 dims active, dim 19 = library size carries 95% of imputed within-CT variance (latent is 32-dim, not 128)
- Decision: no variance loss; retrain with explicit library-size covariate + weaker/free-bits JSD
- See [project_variance_zero_nonzero_fullmodel.md](project_variance_zero_nonzero_fullmodel.md)

## Variance Status (corrected 2026-07-20)
- Within-CT per-gene variance is the correct metric
- Protocol filtering bug fixed in `quick_variance_check.py`
- Full-data cVAE: **0.158** median; 20% cVAE: **0.083**; scVI: 0.034; CT-mean: ~0

## Nuclear lncRNA Metric (fixed 2026-07-20)
- Data-driven list: 45 genes with GT delta < -0.2
- Re-eval (job 7754430): **100% correct sign** (mean delta -0.36)

## Notes Structure
- `notes/training_runs.md` — Master run inventory
- `notes/design_decisions.md` — Architecture rationale
- `notes/experiments/` — All experiment results
- `notes/ISMB2026/` — Poster/slides

## Linear Issues
- **SUC-5** PerCT_5pct (**Done**), **SUC-8** Baseline (**Done**)
- **SUC-13** Eval framework (**In Progress** — Phase 1 done)
- SUC-6, 7, 9, 10, 11, 12 — Backlog

- [Sex composition](project_sex_composition.md) — cells 6M+6F libraries (sex=library), nuclei all female; Xist top-variance in every CT, imputes to one mode

## Older Context
- [Experiment history](experiment_history.md) — full model/job details
- [Prashant meeting 04-15](project_prashant_meeting_0415.md)
- [Metric rethink](project_metric_rethink.md)
