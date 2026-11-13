# TRAP/RiboTag and compartment-fractionation external validation

**Date:** 2026-09-17/18. **Status:** complete (all phases run; 1,000-iteration final statistics).
**Plan:** `TASK.md` (Codex draft revised against the repo; Section 0 lists corrections).
**Report:** `outputs/report/external_validation_report.md` (executive summary, methods, results, limitations).
**Run log / job IDs:** `outputs/logs/run_log.md`.

## Headline results (ImputeVAE full-data cVAE, frozen; ground-truth MOp delta reported beside every number)

| Test | Result |
|---|---|
| H1 GSE210071 cytosol - nucleus (cultured cortical cells, triptolide) | rho 0.38 [0.37, 0.40], AUROC 0.80, matched null -0.02 (GT 0.37) |
| H1 GSE210071 whole cell - nucleus | rho 0.41 [0.39, 0.43], AUROC 0.82 (GT 0.40) |
| H1 GSE121199 D2 MSN whole cell - nuclear | rho 0.50, AUROC 0.82 (D1 fails: shallow nuclear libraries; GT identical) |
| H2 Pvalb RiboTag (GSE263082, 3 batches, 8 reps) | r nucleus 0.43 -> imputed 0.73; real cell 0.74 |
| H2 Sst RiboTag (GSE263082, 2 batches, 7 reps) | 0.45 -> 0.73; real cell 0.73 |
| H2 cross-lab GSE77243 | Pvalb 0.50 -> 0.78, Sst 0.50 -> 0.78 |
| H3 RiboTag abundance vs predicted delta | rho 0.74 to 0.80; partial given nucleus expression 0.69 to 0.74; top-10% enrichment 0.74 vs null 0.24 |
| H4 45-gene nuclear set (GSE121199) | 89.5% externally nuclear |
| H5 filters | rho 0.33 to 0.41 across all exclusions |
| Enrichment-type contrasts (IP/input, IP/UB) | -0.27 to +0.24, same for GT (property mismatch, as in GSE141464) |

## Layout
```
data/<GSE>/                 downloaded GEO files (never modified); outputs/data_manifest.tsv has checksums
scripts/00..11, extval_utils.py, jobs/*.sh   (see TASK.md Section 8)
outputs/metadata/           per-dataset and combined sample tables
outputs/gene_mapping/       mapping audits; outputs/harmonized/ symbol-level matrices
outputs/model_pseudobulk/   per-CT nucleus/imputed/cell pseudobulks, predicted and GT deltas (all + validation split)
outputs/normalized/         log2 FPKM / TMM logCPM matrices; outputs/qc/ QC tables and plots
outputs/tables/             contrast tables, merged per-gene tables, concordance/H2/H3/specificity summaries
outputs/figures/            fig1..fig8 (PNG, PDF, _data.csv)
outputs/report/             executive_summary, methods, results, limitations, external_validation_report
outputs/logs/               script logs, R sessionInfo, run_log.md, 11_summary.txt
```

## Extension (2026-09-18): newer RNA-seq RiboTag/TRAP datasets

Report: `outputs/report/extension_newer_datasets.md`. Scripts `scripts/12`-`16`, job `jobs/14_15_extension.sh` (SLURM 10910535). Outputs under `outputs/extension/` (metadata, harmonized, normalized, model_pseudobulk for 7 baseline methods, tables, figures E1-E5, logs, `data_manifest_extension.tsv`).
Nine sequencing datasets (GSE133291, GSE172484, GSE202172, GSE141337, GSE142028, GSE141874, GSE139684, GSE99791, GSE211162) extend H2/H3 to excitatory, Vip/Htr3a and glial populations.
- H2: imputed vs external profile Spearman 0.54-0.77 for neurons (nucleus 0.26-0.48; real-cell ceiling 0.54-0.75) and 0.40-0.56 for glia (nucleus 0.23-0.39; ceiling 0.37-0.54); gap closed 0.96-1.20; all CIs exclude zero.
- H3: abundance vs predicted delta rho 0.60-0.78 neurons, 0.42-0.55 glia (2008 microarray gave 0.24-0.40).
- Specificity: matched cell type ranked first for 14/18 external populations with imputed profiles vs 5/18 with raw nuclei (real cells 12/18). (Updated 2026-09-24 after the L5 mapping audit: Scnn1a_L4 excluded, Rbp4 -> L5 IT+ET pool; before: 15/19, 5/19, 13/19. H2/H3 ranges unchanged; Rbp4 vs pool: nucleus 0.46 -> imputed 0.73, real cell 0.72.)
- Method comparison: H2 does NOT discriminate methods at pseudobulk level (all within ~0.03; linear regression best on glia).

## L5 cell-type mapping audit (2026-09-24)

Prashant (Sep 2026 meeting): do not map external L5 populations to L5 IT / L5 ET by name; drop L4 (motor cortex has no L4); decide ET vs IT from projection evidence in the source papers, marker genes, and each L5 type individually.
Evidence table with quotes and DOIs: `outputs/extension/metadata/l5_mapping_evidence.md`. Script `scripts/17_l5_mapping_audit.py`, tables `outputs/extension/tables/l5_audit_*.tsv`, figure `outputs/extension/figures/L5_audit_centered_and_markers.png`, log `outputs/extension/logs/17_l5_audit.log`.
- **Literature:** Gprin3 (GSE172484) = corticospinal ET (82% of spinal CTB+ cells EGFP+, Moya 2022); Colgalt2/Glt25d2 (GSE172484, GSE141337, GSE13379) = corticopontine ET (Moya 2022; Schmidt 2012; Doyle 2008); S100a10 (GSE202172) = L5a corticostriatal IT (Schmidt 2012; Chottekalapanda 2020); Rbp4-Cre KL100 (GSE142028) = pan-L5 IT + ET, not NP (Gerfen 2013; Tasic 2018); Scnn1a-Tg3-Cre (GSE133291) = L4 spiny stellate (Furlanis 2019), no MOp counterpart; Etv1 (GSE13379) = IT-leaning but impure (Doyle 2008 flag vascular contamination).
- **Data-driven cross-check (real-cell markers only, never imputed):** marker z (ET minus IT) +0.49 Gprin3, +0.40 Colgalt2, +0.27 Glt25d2, -0.19 S100a10, +0.07 Rbp4, -0.60 Scnn1a; centred Spearman picks L5 ET for Gprin3 and Colgalt2 with real cells, nuclei and imputed alike. All agree with the literature.
- **Decisions encoded in `13_prepare_extension.py` / `15_extension_h2_h3.py`:** Scnn1a_L4 excluded from H2/H3 (`exclude` column in `extension_populations.tsv`); Rbp4 mapped to `POOL_L5_IT_ET` (mean of L5 IT and L5 ET) instead of L5 IT; Gprin3, Colgalt2, Glt25d2 stay L5 ET; S100a10 stays L5 IT. Pre-audit tables kept as `*__pre_audit_2026-09-18.tsv`; rerun job `jobs/17_rerun_15_after_mapping_audit.sh` (SLURM 11168301).
- **Effect on numbers:** uncentered H2 Spearman differs by at most 0.03 between L5 IT, L5 ET and L5/6 NP for every L5 population (`l5_audit_each_type_H2.tsv`), so the mapping changes presentation, not the H2 conclusion. Region caveat: GSE141337, GSE142028, GSE202172 and the GSE172484 SOD-negative batch pooled whole cortex; GSE172484 baseline is M1.
