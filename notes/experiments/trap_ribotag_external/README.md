# TRAP/RiboTag + Compartment Fractionation External Validation

**Date:** 2026-09-18
**Location:** `validation/trap_ribotag_external_validation/` (plan `TASK.md`, report `outputs/report/`)
**SLURM:** 10905766 (pseudobulk), 10905946 (limma-voom GSE121199/GSE263082), 10905990 (GSE174303), 10905994 (final stats), 10906009 (secondary)

## Question
Does the frozen full-data cVAE's predicted nucleus-to-whole-cell delta agree with independent biochemical compartment data (GSE210071 cytosol/nucleus/whole-cell fractionation; GSE121199 whole-cell vs nuclear FACS) and does imputation move nuclear profiles toward external RiboTag profiles of matched interneuron types (GSE263082 Pvalb/Sst; GSE77243 rerun on the same pipeline)? Secondary: GSE174303 CA1 TRAP/input, GSE13379 bacTRAP atlas.

## Results
- **H1 compartment:** GSE210071 cytosol - nucleus rho 0.38 [0.37, 0.40], AUROC 0.80, abundance-matched null -0.02; whole - nucleus rho 0.41, AUROC 0.82. Ground-truth MOp delta gives 0.37 / 0.40 (model at the biological ceiling). Robust to bicuculline and to all H5 filters (0.33 to 0.41). GSE121199 D2 whole cell - nuclear rho 0.50, AUROC 0.82; D1 fails (shallow nuclear libraries; GT identical).
- **H2 profile improvement:** Pvalb r(RiboTag) 0.43 (nucleus) -> 0.73 (imputed), real cell 0.74; Sst 0.45 -> 0.73, ceiling 0.73. Gene and replicate bootstrap CIs exclude zero in every batch. GSE77243: 0.50 -> 0.78 for both. Imputed profiles show cell-type diagonal dominance like real cells.
- **H3:** RiboTag abundance vs predicted delta rho 0.74 (GSE263082 META) and 0.79 to 0.80 (GSE77243); partial given nucleus expression 0.69 to 0.74; top-10% predicted-positive genes 74% in top-20% RiboTag vs 24% matched null.
- **H4:** 45-gene nuclear set 89.5% externally nuclear in GSE121199; Xist/Meg3/Snhg11 nuclear in all external data; Meg3 and Kcnq1ot1 remain model-discordant (pre-registered); Malat1/Neat1 absent from GSE210071; Neat1/Pvalb absent from the model gene set.
- **Property mismatch confirmed:** IP/input and IP/UB contrasts rho -0.27 to +0.24 for model and ground truth alike. GSE13379 TRAP abundance vs matched-CT delta 0.24 to 0.40 across nine cortical populations.
- **Caveat on method discrimination:** global H1 rank concordance is similar for all seven methods (0.37 to 0.41); differences appear in AUROC and in the cell-type-resolved H2/H3 tests.

## Limitations
All GSE210071 compartment samples were triptolide-treated (transcription blocked 3 h) in embryonic mixed cultures. Model deltas are un-normalized log1p differences (rank statistics only). No matched cell type for MSNs. RiboTag is ribosome-associated RNA, not cytoplasm. Mouse only; no sub-cytoplasmic or human claims.

## Extension (2026-09-18): newer RNA-seq RiboTag/TRAP datasets

Report: `outputs/report/extension_newer_datasets.md`. Scripts `scripts/12`-`16`, job `jobs/14_15_extension.sh` (SLURM 10910535). Outputs under `outputs/extension/` (metadata, harmonized, normalized, model_pseudobulk for 7 baseline methods, tables, figures E1-E5, logs, `data_manifest_extension.tsv`).
Nine sequencing datasets (GSE133291, GSE172484, GSE202172, GSE141337, GSE142028, GSE141874, GSE139684, GSE99791, GSE211162) extend H2/H3 to excitatory, Vip/Htr3a and glial populations.
- H2: imputed vs external profile Spearman 0.54-0.77 for neurons (nucleus 0.26-0.48; real-cell ceiling 0.54-0.75) and 0.40-0.56 for glia (nucleus 0.23-0.39; ceiling 0.37-0.54); gap closed 0.96-1.20; all CIs exclude zero.
- H3: abundance vs predicted delta rho 0.60-0.78 neurons, 0.42-0.55 glia (2008 microarray gave 0.24-0.40).
- Specificity: matched cell type ranked first for 13/17 external populations with imputed profiles vs 5/17 with raw nuclei (real cells 11/17). (Updated 2026-09-25: Scnn1a_L4 and Rbp4_L5 excluded, job 11181236; with Rbp4 pooled to L5 IT+ET it was 14/18, 5/18, 12/18; before the audit 15/19, 5/19, 13/19. H2/H3 of every kept population unchanged.)
- Method comparison: H2 does NOT discriminate methods at pseudobulk level (all within ~0.03; linear regression best on glia).

## L5 cell-type mapping audit (2026-09-24)

Prashant (Sep 2026 meeting): do not map external L5 populations to L5 IT / L5 ET by name; drop L4 (motor cortex has no L4); decide ET vs IT from projection evidence in the source papers, marker genes, and each L5 type individually.
Evidence table with quotes and DOIs: `outputs/extension/metadata/l5_mapping_evidence.md`. Script `scripts/17_l5_mapping_audit.py`, tables `outputs/extension/tables/l5_audit_*.tsv`, figure `outputs/extension/figures/L5_audit_centered_and_markers.png`, log `outputs/extension/logs/17_l5_audit.log`.
- **Literature:** Gprin3 (GSE172484) = corticospinal ET (82% of spinal CTB+ cells EGFP+, Moya 2022); Colgalt2/Glt25d2 (GSE172484, GSE141337, GSE13379) = corticopontine ET (Moya 2022; Schmidt 2012; Doyle 2008); S100a10 (GSE202172) = L5a corticostriatal IT (Schmidt 2012; Chottekalapanda 2020); Rbp4-Cre KL100 (GSE142028) = pan-L5 IT + ET, not NP (Gerfen 2013; Tasic 2018); Scnn1a-Tg3-Cre (GSE133291) = L4 spiny stellate (Furlanis 2019), no MOp counterpart; Etv1 (GSE13379) = IT-leaning but impure (Doyle 2008 flag vascular contamination).
- **Data-driven cross-check (real-cell markers only, never imputed):** marker z (ET minus IT) +0.49 Gprin3, +0.40 Colgalt2, +0.27 Glt25d2, -0.19 S100a10, +0.07 Rbp4, -0.60 Scnn1a; centred Spearman picks L5 ET for Gprin3 and Colgalt2 with real cells, nuclei and imputed alike. All agree with the literature.
- **External marker check (2026-09-25, replaces the real-cell-derived markers as the primary check):** markers from Yao 2021 MOp Supplementary Table 6 (16 L5 IT / 29 L5 ET / 12 NP genes; `outputs/extension/metadata/yao2021/`), z-scored across the 7 external glutamatergic TRAP profiles, no model pseudobulk used. ET minus IT: Gprin3 +0.60, Colgalt2 +0.65, Glt25d2 +0.26, S100a10 -0.18, Rbp4 +0.01, Scnn1a -0.69; every call matches the tracing literature. NP set scores highest in the three whole-cortex pulldowns (flag only). Script `scripts/18_l5_mapping_audit_external_markers.py`, tables `l5_audit_extmarkers_*.tsv`, figure `L5_audit_external_markers.png`. The real-cell-derived set of script 17 is kept for reference but is not independent of the H2 real-cell ceiling.
- **Decisions encoded in `13_prepare_extension.py` / `15_extension_h2_h3.py`:** Scnn1a_L4 excluded from H2/H3 (`exclude` column in `extension_populations.tsv`); Rbp4_L5 excluded on 2026-09-25 (mixed IT+ET driver, no single matching subclass; rule: skip, do not pool) after a one-day interim as `POOL_L5_IT_ET`; Gprin3, Colgalt2, Glt25d2 stay L5 ET; S100a10 stays L5 IT. Pre-audit tables kept as `*__pre_audit_2026-09-18.tsv`; rerun job `jobs/17_rerun_15_after_mapping_audit.sh` (SLURM 11168301).
- **Effect on numbers:** uncentered H2 Spearman differs by at most 0.03 between L5 IT, L5 ET and L5/6 NP for every L5 population (`l5_audit_each_type_H2.tsv`), so the mapping changes presentation, not the H2 conclusion. Region caveat: GSE141337, GSE142028, GSE202172 and the GSE172484 SOD-negative batch pooled whole cortex; GSE172484 baseline is M1.
