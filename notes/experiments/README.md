# Experiments

Each experiment has its own folder with a standard structure:
- `README.md` — goal, hypothesis, link to Linear issue
- `config.md` — model configs, job scripts, hyperparameters
- `results.md` — metrics, tables, comparisons
- `figures/` — evaluation plots

## Active

| ID | Name | Status | Notes |
|----|------|--------|-------|
| [SUC-5](SUC-5_perct5_retraining/) | PerCT_5pct gene list retraining | In Progress | Stage 2 done, eval done |
| [SUC-6](SUC-6_variance_retention/) | Per-cell variation retention | Backlog | Blocked by SUC-5, SUC-8, SUC-13 |
| [SUC-7](SUC-7_publication_roadmap/) | Publication roadmap (Nature Methods) | Backlog | Umbrella issue |
| [SUC-8](SUC-8_baseline/) | Baseline comparison (linear reg + scVI) | In Progress | Linear reg done |
| [SUC-9](SUC-9_gene_stratified_eval/) | Gene-stratified eval by gap size | Backlog | Blocked by SUC-5, SUC-13 |
| [SUC-10](SUC-10_biological_validation/) | Biological validation / downstream use case | Backlog | Low priority, paper-oriented |
| [SUC-11](SUC-11_cross_species_transfer/) | Cross-species transfer (human nuclei) | Backlog | Blocked by SUC-12 |
| [SUC-12](SUC-12_cross_tissue_characterization/) | Characterize nuc-cell diff across tissues/species | Backlog | Prerequisite for SUC-11 |
| [SUC-13](SUC-13_eval_refactor/) | Refactor eval framework (modular, 5-source) | Backlog | Blocks SUC-6, SUC-9 |

## Analysis Tasks (2026-08)

| ID | Name | Status | Notes |
|----|------|--------|-------|
| [compartment_classification](compartment_classification/) | Compartment classification (AUROC, sign agreement, gene classes) | Done | Job 8272238, 7 methods + raw baseline |
| [trap_validation](trap_validation/) | TRAP/RiboTag cross-validation (non-circular) | Done | GSE141464 + GSE77243, rho ~0.80 within-type |
| [conserved_transformation](conserved_transformation/) | Lake 2023 kidney conserved transformation | Done | Job 7900702, cross-tissue ribosomal core |
| [fractionation_validation](fractionation_validation/) | Brain fractionation-seq validation (non-circular) | Done | Buckley 2016 mouse + Moutinho 2021 human, rho ~0.44 |
| [trap_ribotag_external](trap_ribotag_external/) | Compartment fractionation + TRAP/RiboTag external validation (GSE210071, GSE121199, GSE263082, GSE174303, GSE13379) | Done | 2026-09-18; H1 rho 0.38-0.41 AUROC 0.80-0.82; H2 Pvalb/Sst 0.43->0.73 (ceiling 0.74); jobs 10905766-10906009 |
| [variance_zero_nonzero_fullmodel](variance_zero_nonzero_fullmodel/) | Zero/nonzero + technical-vs-biological variance diagnostic on full-data cVAE | Done | 2026-09-18; jobs 10910494, 10910608; real-cell var 52% Poisson/25% lib/23% bio; imputed var 82% library-size axis, ~1-D; like-for-like retention 0.12 |
| [variance_zero_nonzero_fullmodel/variance_rank](variance_zero_nonzero_fullmodel/variance_rank/) | Variance-rank check (Prashant): per-gene variance distributions + Spearman real vs imputed | Done | 2026-09-24; Spearman 0.92 L5 IT, 0.93 Oligo, 0.85 pooled; distribution stretched not collapsed |
| [trap_ribotag_external](trap_ribotag_external/) L5 mapping audit | ET/IT identity of L5 TRAP lines from literature + real-cell markers; L4 excluded, Rbp4 -> L5 IT+ET pool | Done (rerun job 11168301) | 2026-09-24; all lines agree with literature; H2 numbers change <= 0.03 |

## Completed (pre-Linear)

These experiments predate the Linear issue tracker. Results are in:
- `pre_linear_configB/` — Config B model eval, hallucination tests, marker/gene filtering analysis
- `../training_runs.md` — master run inventory
