# ImputeVAE external validation with compartment fractionation and TRAP/RiboTag data

**Date:** 2026-09-18. Task: `validation/trap_ribotag_external_validation/TASK.md`. Model: `run/run_2026-07-18-19-40-51_570984` (frozen).

This report bundles four documents in `outputs/report/`:

- [executive_summary.md](executive_summary.md): the five committee questions answered plainly.
- [methods.md](methods.md): data, preprocessing, statistics.
- [results.md](results.md): all headline tables (H1 to H5, secondary datasets).
- [limitations.md](limitations.md): what the evidence does and does not support.

## Headline

| Test | Evidence | Number |
|---|---|---|
| H1 compartment concordance (GSE210071 cytosol - nucleus) | Spearman, AUROC, matched null | 0.38 [0.37, 0.40], AUROC 0.80, null -0.02, p 0.001 |
| H1 compartment concordance (GSE121199 D2 whole cell - nuclear) | Spearman, AUROC | 0.50, AUROC 0.82 |
| H2 profile improvement, Pvalb (GSE263082, 8 reps) | r(nucleus) -> r(imputed) vs ceiling | 0.43 -> 0.73 (real cell 0.74) |
| H2 profile improvement, Sst (GSE263082, 7 reps) | same | 0.45 -> 0.73 (real cell 0.73) |
| H2 cross-lab (GSE77243) | same | 0.50 -> 0.78 (Pvalb), 0.50 -> 0.78 (Sst) |
| H3 RiboTag abundance vs predicted delta | Spearman, partial given nucleus expr | 0.74 to 0.80, partial 0.69 to 0.74 |
| H4 45-gene nuclear set in GSE121199 | fraction externally nuclear | 89.5% (38 genes) |
| H5 filters (GSE210071) | range of rho | 0.33 to 0.41 |
| Property mismatch (IP/input, IP/UB) | Spearman | -0.27 to +0.24, ground truth identical |

## Figures (`outputs/figures/`, PNG + PDF + data CSV)
fig1 design schematic; fig2 QC; fig3 GSE210071 concordance; fig4 GSE121199 concordance; fig5 H2 profile improvement; fig6 H3 decile curves and enrichment; fig7 showcase genes; fig8 summary forest plot.

## Key tables (`outputs/tables/`)
`concordance_summary_GSE210071_GSE121199.tsv`, `concordance_genes_*.tsv`, `concordance_nuc45_*.tsv`, `merged_<dataset>_<contrast>.tsv` (per-gene), `profile_improvement_H2.tsv`, `translational_support_H3.tsv`, `celltype_specificity.tsv`, `concordance_summary_secondary.tsv`, dataset contrast tables (`GSE210071_contrasts_*.tsv`, `GSE121199_contrasts_*.tsv`, `GSE263082_*`, `GSE174303_contrasts_WT.tsv`).

## Reproduction
`outputs/logs/run_log.md` lists commands, versions, SLURM job IDs (10905766, 10905806, 10905946, 10905990, 10905994, 10906009) and warnings. All scripts run from the project root.
