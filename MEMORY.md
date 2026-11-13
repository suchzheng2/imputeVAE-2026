# Project Memory - imputeVAE

## Project Overview
- **Working dir:** `/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE`
- **Conda envs:** `impute2` (main), `abc_env` (scib metrics only)
- **Platform:** HPC cluster (SLURM), partition `pi_gerstein`, `scavenge_gpu` for faster scheduling
- **Script path note:** Scripts are at `analysis/` relative to project root (NOT `imputeVAE/analysis/`)

## Notes Structure (reorganized 2026-04-06)
```
notes/
├── training_runs.md              # Master run inventory
├── design_decisions.md           # Architecture rationale
├── project_scope_document.md     # Full project overview
├── gerstein_perspective_review.md # Advisor feedback (actionable TODOs)
├── archive/                      # Superseded docs (mse_vs_nb, softmax, old plans)
├── presentation/                 # Plans, outlines, figures, slides
└── experiments/
    ├── README.md                 # Index of all tracked issues
    ├── SUC-5_perct5_retraining/  # PerCT retraining
    ├── SUC-6_variance_retention/ # Variance fix (4 phases)
    ├── SUC-7_publication_roadmap/ # Nature Methods submission plan
    ├── SUC-8_baseline/           # Linear regression + scVI baselines
    ├── SUC-9_gene_stratified_eval/ # Gene-stratified eval by gap
    ├── SUC-10_biological_validation/ # Downstream use case demo
    ├── SUC-11_cross_species_transfer/ # Human nuclei transfer
    ├── SUC-12_cross_tissue_characterization/ # Nuc-cell diff analysis
    ├── SUC-13_eval_refactor/     # Eval framework refactor
    └── pre_linear_configB/       # Config B eval data, hallucination tests
```

## Linear Issues
- **SUC-5** — PerCT_5pct gene list retraining (**Done** — eval complete 2026-04-07)
- **SUC-6** — Per-cell variation retention (Backlog, blocked by SUC-13)
- **SUC-7** — Publication roadmap: Nature Methods (Backlog, umbrella issue)
- **SUC-8** — Baseline comparison: linear regression + scVI (**Done** — linreg complete 2026-04-06)
- **SUC-9** — Gene-stratified eval by cell-nucleus gap size (Backlog, blocked by SUC-13)
- **SUC-10** — Biological validation / downstream use case (Backlog, low priority)
- **SUC-11** — Cross-species transfer: human nuclei (Backlog, blocked by SUC-12)
- **SUC-12** — Characterize nuc-cell diff across tissues/species (Backlog, prereq for SUC-11)
- **SUC-13** — Refactor eval framework: modular, 5-source, reusable (**In Progress** — Phase 1 done 2026-04-07)

## Key Design Decisions
- See `notes/design_decisions.md` in the repo
- **2026-04-06**: Deprioritizing cross-recons (xr) — compresses variance from 11% to 3%. Focus on per-cell variation first.
- **2026-04-06**: Must run baseline comparison (SUC-8) before investing in variance fixes (SUC-6). If linear regression also collapses → problem is fundamental, not architectural.
- **2026-04-07**: Latent diagnostic invalidated bottleneck hypothesis. Phase 1 (larger latent) SKIPPED. See [latent diagnostic](project_latent_diagnostic.md).

## Current Best Models (Config B, 10,134 genes)
- **MSE cond+xr**: `run/run_2026-04-02-10-51-32_3738876` — best cell-type-level (classif 95.8%, DE +5.1%)
- **MSE cond**: `run/run_2026-04-02-02-02-35_3345839` — better per-cell variation (var ratio 11.5% vs 3%)
- **MSE shared**: `run/run_2026-04-01-14-01-16_1946018` — baseline, worse than raw nuclei

## PerCT_5pct Models (16,439 genes)
- **Stage 1**: `run/run_2026-04-02-19-27-35_3767601` — COMPLETED
- **Stage 2 cond**: `run/run_2026-04-07-02-12-05_3779214` — COMPLETED
- **Stage 2 cond+xr**: `run/run_2026-04-07-02-12-05_3551465` — COMPLETED
- **Eval**: compeval job 3250927 COMPLETED, hallucination job 3250913 COMPLETED
- **Key results**: cond+xr DE overlap 54.7% (best yet), gene corr 0.963, classif 94.2%
- **Variance**: still collapsed (var ratio 0.035 cond+xr, 0.115 cond)

## Eval Framework (SUC-13, built 2026-04-07)
- **Location**: `analysis/eval/` — 11 modules
- **CLI**: `python -m analysis.eval.runner --model <name> --eval all [--slurm]`
- **Registry**: 7 models (5 log1p + 2 CPM pending)
- **Phase 1 done**: metrics, runner, registry, data, inference, report
- **2026-04-28**: Added CPM normalization support to data loader + registry
- **Phase 2 TODO**: refactor inline eval (stage3_eval.py) to share metrics
- **Phase 3 TODO**: deprecate old scripts

## CPM Experiment (2026-04-16, updated 2026-04-28)
- **Motivation**: Library size shortcut — decoder uses 1 dim for libsize, ignores rest
- **Stage 1**: job 3964307 — COMPLETED (4h 29m, 600 epochs)
  - Run folder: `run/run_2026-04-23-23-59-34_1163351`
  - JSD stuck at ln(2)=0.693 — same as log1p stage 1, not CPM-specific
  - Recons ~1.5 (unclamped, uses `--clamp_recons_loss False`)
- **Stage 2**: submitted 2026-04-28
  - cond: job 4061786
  - cond+xr: job 4061787
- **Eval registry**: `cpm_pct5_cond`, `cpm_pct5_cond_xr` (run_folder=None until jobs finish)
- **After jobs complete**: grep run folder from logs → fill registry → run eval
- **Stage 1 JSD note**: Both CPM and log1p stage 1 have JSD stuck at ln(2).
  beta=0.1 is too small relative to recons loss. Considered norm_recons_by_dim
  but MSELoss already uses mean reduction; extra /G over-corrects. Not a problem
  in practice — log1p model worked fine in stage 2 despite this.
- **Notes**: `notes/experiments/SUC-6_variance_retention/cpm_experiment.md`

## Next Steps
1. **Wait for CPM stage 2** (jobs 4061786, 4061787) → fill registry → run eval
2. **Compare CPM vs log1p** variance retention (the key question)
3. SUC-8: scVI baseline still TODO (only linreg done so far)
4. SUC-9: gene-stratified eval — unblocked by SUC-13 Phase 1
5. Always report imputed stats separately from cell-recon stats
6. Always report proportion-normalized std_ratio alongside log1p std_ratio

## Prashant Meeting (2026-04-15)
- Agreed on CPM + log1p normalization as next experiment
- Discussed ZINB loss, zero/nonzero weighting as medium-term options
- Action items: CPM implementation (priority), follow-up next week
- [Full notes](project_prashant_meeting_0415.md)

## Older Context
- [Metric rethink](project_metric_rethink.md)
- [Cross-recons design](project_cross_recons_design.md)
- [MSE vs NB results](project_mse_nb_results.md)
- [Training runs pointer](project_training_runs.md)
- [Feedback: HPC persistence](feedback_save_to_repo.md)
- [Feedback: Linear issue → notes folder](feedback_linear_issue_notes.md)
- [Feedback: Eval scripts no GPU](feedback_eval_no_gpu.md)
