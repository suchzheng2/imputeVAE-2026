# Extension: newer RNA-seq RiboTag/TRAP datasets for excitatory and glial cortical cell types

**Date:** 2026-09-18. **Model:** frozen full-data cVAE (`run/run_2026-07-18-19-40-51_570984`). **Scripts:** `scripts/12`-`16`; SLURM job 10910535 (first attempt 10910525 failed on two script bugs, fixed). **Tables/figures:** `outputs/extension/`.
Purpose: replace the role of the 2008 microarray atlas (GSE13379) with sequencing-based ribosome-associated profiles, extend H2/H3 beyond Pvalb/Sst, and test whether H2 separates ImputeVAE from baseline methods.

## 1. Overview of data used

Designs were verified from GEO sample records. Only wild-type, unstimulated/control, IP (ribosome-bound) samples were used. No SRA/FASTQ was downloaded. Checksums: `outputs/extension/data_manifest_extension.tsv`; sample table: `outputs/extension/metadata/extension_samples.tsv`; population-to-model mapping: `extension_populations.tsv`. Overview graphic of every dataset in the whole validation: `figE1_dataset_overview`.

| Dataset (year) | Tissue, assay | Population -> model cell type | Replicates used | Model genes tested |
|---|---|---|---|---|
| GSE133291 Furlanis (2019) | neocortex, RiboTag IP | CamK2a -> excitatory pool (7 CTs); Scnn1a L4 -> L5 IT (no L4 in model, flagged); PV -> Pvalb; SST -> Sst; VIP -> Vip | 4 each | 8,332-8,601 |
| GSE172484 (2022) | motor cortex M1, bacTRAP IP | Gprin3 lower L5b -> L5 ET; Colgalt2 upper L5b -> L5 ET | 2 baseline + 2-3 SOD-negative littermates (2 batches, META) | 8,535-8,713 |
| GSE202172 (2022) | cortex, bacTRAP IP | S100a10 L5a -> L5 IT | 4 (vehicle, group-housed) | 8,280 |
| GSE141337 (2019) | cerebral cortex, bacTRAP IP | Glt25d2 L5b -> L5 ET | 3 (saline) | 8,453 |
| GSE142028 (2019) | cerebral cortex, TRAP IP | Rbp4 L5 -> **excluded 2026-09-25** (pan-L5 driver labels IT and ET; no single matching subclass) | 2 | 8,104 |
| GSE141874 (2019) | cerebral cortex, TRAP IP | Htr3a interneurons -> pool of Vip, Lamp5, Sncg | 3 | 8,277 |
| GSE139684 (2019) | prefrontal cortex, bacTRAP IP | Aldh1l1 astrocytes -> Astro | 3 (unstressed controls) | 8,235 |
| GSE99791 (2017) | motor + visual cortex, RiboTag IP | GFAP astrocytes (4 months) -> Astro | 3 + 3 (2 regions, META) | 8,574 |
| GSE211162 (2023) | frontal cortex, TRAP IP | mature oligodendrocytes -> Oligo | 7 | 8,445 |
| GSE263082, GSE77243 (re-run for comparison) | see main report | Pvalb, Sst (+ Vip for GSE77243, new) | as before | 7,745-8,645 |

**Dropped or restricted, with reasons.** GSE133291 hippocampal samples (`*_W*` for CamK/SST, all Grik) and 12 `*_K*` columns that have no GEO sample record. GSE172484/GSE202172/GSE141337 input, unbound, SOD1-G93A, fluoxetine, single-housed, cocaine and yoked samples. GSE142028 dentate gyrus. GSE139684 stressed/resilient animals and non-cortical regions. GSE99791 aged (2-year) mice, cerebellum, hypothalamus, and input. GSE211162 cerebellum, spinal cord, white matter; this series supplies RPKM only (no counts), so it enters as log2(RPKM+1). No clean cortical microglia control set was added (not attempted beyond the search; optional in the brief). GSE331310 (NEX-RiboTag, 2026) was not examined.

**Processing.** Symbols harmonized (Furlanis FASTDB IDs via the GSE210071 annotation table; 22,933 mapped genes). Counts: genes with CPM >= 1 in at least the smallest group, TMM (Python implementation in `extval_utils.py`), logCPM with prior 2, within dataset. Gene universe per comparison: model genes with external mean logCPM > 0 (RPKM > 1 for GSE211162). Replicate agreement within population: median Spearman 0.93-0.99. Model profiles: per-cell-type mean log1p pseudobulks of 40,166 nuclei, their ImputeVAE imputations, and 71,117 real cells; pooled columns are unweighted means of per-CT pseudobulks. Statistics are the functions of `08_profile_improvement.py`, imported unchanged (1,000 gene bootstraps, 1,000 replicate bootstraps, 1,000 abundance-matched permutations, seed 42).

## 2. Hypotheses

- **H2 (profile improvement).** For a matched cell type, the imputed whole-cell profile is more similar to the external ribosome-associated profile than the raw nuclear profile is, and approaches the similarity achieved by real whole cells of that type.
- **H3 (translational support).** Within a cell type, genes with larger predicted nucleus-to-cell delta are more abundant on ribosomes, beyond what nuclear abundance explains.
- **H-spec (specificity).** An external population matches its own model cell type better than other cell types, and imputation does not erase that.
- **H-method.** If the learned transformation matters at cell-type resolution, ImputeVAE's imputed profiles should match external profiles better than baseline methods' imputed profiles.
- **H-platform.** The weak 2008 microarray correlations reflect the platform, so sequencing data for the same kinds of populations should give clearly higher values.

## 3. Results

### 3.1 H2: Spearman similarity to the external profile (nucleus vs imputed vs real cell)
*What it measures and why:* rank agreement of a whole expression profile with an independent ribosome-associated profile. Spearman is used because the model works in log1p UMI counts and the external data in logCPM of full-length libraries; ranks avoid unit and length-scale assumptions. The real-cell value is the ceiling any imputation could reach; `gap closed` = (imputed - nucleus) / (real cell - nucleus). Uncertainty: gene bootstrap (stability over genes) and replicate bootstrap (external biological variation), reported separately.

| Population -> model CT | nucleus | imputed | real cell | improvement [gene-boot 95% CI] | gap closed |
|---|---|---|---|---|---|
| CamK2a excitatory -> excitatory pool | 0.264 | 0.624 | 0.616 | 0.361 [0.346, 0.374] | 1.03 |
| Scnn1a L4 -> L5 IT | 0.285 | 0.619 | 0.620 | 0.334 [0.319, 0.348] | 1.00 |
| Gprin3 L5b (M1) -> L5 ET, META | 0.423 | 0.769 | 0.752 | 0.347 [0.330, 0.361] | 1.05 |
| Colgalt2 L5b (M1) -> L5 ET, META | 0.467 | 0.757 | 0.751 | 0.290 [0.276, 0.304] | 1.02 |
| Glt25d2 L5b -> L5 ET | 0.425 | 0.725 | 0.715 | 0.300 [0.285, 0.315] | 1.03 |
| S100a10 L5a -> L5 IT | 0.414 | 0.652 | 0.652 | 0.238 [0.224, 0.253] | 1.00 |
| Rbp4 L5 (excluded 2026-09-25; shown for reference, vs L5 IT+ET pool) | 0.462 | 0.726 | 0.721 | 0.263 [0.248, 0.278] | 1.01 |
| Furlanis PV -> Pvalb | 0.283 | 0.595 | 0.607 | 0.312 [0.298, 0.326] | 0.96 |
| Furlanis SST -> Sst | 0.273 | 0.604 | 0.605 | 0.330 [0.317, 0.344] | 0.99 |
| Furlanis VIP -> Vip | 0.289 | 0.541 | 0.543 | 0.252 [0.240, 0.263] | 0.99 |
| GSE77243 Vip -> Vip | 0.499 | 0.735 | 0.739 | 0.236 [0.225, 0.248] | 0.98 |
| Htr3a interneurons -> Vip/Lamp5/Sncg pool | 0.481 | 0.711 | 0.714 | 0.230 [0.218, 0.242] | 0.99 |
| Aldh1l1 astrocytes (PFC) -> Astro | 0.226 | 0.400 | 0.371 | 0.174 [0.162, 0.187] | 1.20 |
| GFAP astrocytes (MC+VC) -> Astro, META | 0.358 | 0.521 | 0.506 | 0.163 [0.152, 0.174] | 1.10 |
| Oligodendrocytes -> Oligo | 0.393 | 0.559 | 0.535 | 0.166 [0.154, 0.179] | 1.17 |

Every replicate-bootstrap interval also excludes zero (per-batch rows in `extension_H2_profile_improvement.tsv`; META rows have no replicate bootstrap, and their `n_replicates` column counts batches). Motor-cortex astrocytes alone: 0.377 -> 0.539 (real cell 0.526). Figure: `figE2_H2_all_datasets`.

### 3.2 H3: ribosome-associated abundance vs predicted delta
*What it measures and why:* Spearman between external IP abundance and the per-cell-type predicted delta; the partial Spearman given nucleus expression removes the part explained by genes simply being highly expressed; the top-10% enrichment against a null permuted within nucleus-expression deciles is the abundance-matched significance test.

| Class | rho (predicted delta) | rho (ground-truth delta) | partial rho | top-10% predicted-positive in top-20% IP vs matched null |
|---|---|---|---|---|
| Excitatory (7 populations) | 0.63-0.71 | 0.56-0.70 | 0.56-0.71 | 0.65-0.75 vs 0.19-0.23 |
| Inhibitory, new (Furlanis PV/SST/VIP, Htr3a, GSE77243 Vip) | 0.60-0.78 | 0.61-0.79 | 0.57-0.69 | 0.68-0.78 vs 0.23-0.34 |
| Glia (2 astrocyte sets, oligodendrocytes) | 0.42-0.55 | 0.41-0.53 | 0.39-0.44 | 0.57-0.64 vs 0.25-0.32 |

All permutation p = 0.001 (the floor for 1,000 permutations). For the three L5 ET-mapped populations the model's delta correlates better than the ground-truth delta (0.67-0.71 vs 0.56-0.61), consistent with the ground-truth L5 ET cell mean resting on only 161 cells. **H-platform:** the same kinds of populations on the 2008 microarray gave 0.24-0.40; sequencing gives 0.42-0.80. Layer 5b Glt25d2, the identical mouse line, goes from 0.40 (array) to 0.67 (RNA-seq). Figure: `figE3_H3_platform_effect`.

### 3.3 Specificity
*What it measures and why:* for each of 19 external populations, the rank of its matched model cell type among the 17 model cell types (plus its pool if pooled), on the 6,932 model genes expressed in all external sets. Raw Spearman is dominated by shared housekeeping structure, so a gene-centered version (each gene centered across model cell types and across external populations) isolates cell-type-specific signal.

| Profile | matched CT ranked first, raw | matched CT ranked first, gene-centered |
|---|---|---|
| raw nucleus | 5 / 19 | 10 / 19 |
| ImputeVAE imputed | 15 / 19 | 13 / 19 |
| real whole cell | 13 / 19 | 15 / 19 |

Imputation restores cell-type matching to the level of real cells. Misses (centered, imputed): CamK2a pan-excitatory (rank 4; best is L6 IT, a member of the pool), Furlanis VIP (5), Colgalt2 L5b (5; best is the excitatory pool) and Glt25d2 L5b (6). Real cells also miss CamK2a, Glt25d2, Rbp4 and S100a10, so broad or mixed driver lines are the limiting factor. Oligodendrocytes rank 13-14 on raw correlation for both imputed and real cells but first once gene-centered. Figures: `figE4_specificity_raw`, `figE4_specificity_centered`.

### 3.4 H-method: H2 by method (negative for discrimination)
*What it measures and why:* the same H2 similarity with each method's per-cell-type imputed pseudobulk (all seven baseline matrices exist under `run/*_baseline/`; row and gene order verified identical). This asks whether an external, cell-type-resolved test separates methods.

Mean Spearman of the imputed profile with the external profile:

| Class | ImputeVAE | cVAE 20% data | CT-mean | Linear reg. | scVI | scANVI | scGen | scButterfly | raw nucleus |
|---|---|---|---|---|---|---|---|---|---|
| Excitatory (7) | **0.694** | 0.688 | 0.688 | 0.665 | 0.672 | 0.678 | 0.679 | 0.678 | 0.39 |
| Inhibitory (9) | 0.689 | 0.686 | **0.693** | 0.680 | 0.679 | 0.681 | 0.683 | 0.686 | 0.41 |
| Glia (3) | 0.493 | 0.424 | 0.471 | **0.553** | 0.480 | 0.493 | 0.468 | 0.431 | 0.33 |

ImputeVAE is at or above the best of the four deep baselines in 15 of 19 populations, but the margins are 0.001-0.028; it beats linear regression in 13 of 19 and loses to it on all three glial sets (by 0.05-0.07) and on both Vip sets. CT-mean equals the real-cell profile by construction at pseudobulk level. **Conclusion: pseudobulk profile similarity to RiboTag does not discriminate methods.** Every method moves the cell-type mean profile most of the way from nucleus to cell (their Pvalb pseudobulks correlate 0.98-0.995 with ImputeVAE's), so this test validates the nucleus-to-cell shift itself, not ImputeVAE specifically. Figure: `figE5_H2_method_comparison`.

## 4. Conclusions

1. The H2/H3 result generalizes from Pvalb/Sst to excitatory neurons (pan-excitatory, L4, L5a, three L5b lines including two from motor cortex), Vip/Htr3a interneurons, astrocytes and oligodendrocytes, in nine additional sequencing datasets from at least six laboratories: imputation closes 96-120% of the nucleus-to-real-cell gap in similarity to ribosome-associated profiles.
2. Absolute similarity and H3 correlations are lower for glia (imputed 0.40-0.56; H3 0.42-0.55) than for neurons (0.54-0.78; 0.60-0.80). The real-cell ceiling is equally low for glia, so this reflects the reference (163 astrocyte and 143 oligodendrocyte nuclei; glial RiboTag contamination by neuronal RNA is also known) rather than imputation error.
3. The 2008 microarray atlas understated the relationship; it can be retired to a platform footnote.
4. Imputation restores cell-type specificity of the profiles to real-cell level (15/19 vs 5/19 for nuclei on raw correlation).
5. H2 does not separate ImputeVAE from baselines. Method discrimination has to come from tests below the cell-type mean (within-cell-type variance, per-cell structure) or from matched single-cell/single-nucleus hold-out data, not from pseudobulk external validation.
6. L5 ET: against three independent L5b TRAP datasets the imputed L5 ET profile matches slightly better than the real-cell L5 ET mean, and the predicted delta tracks TRAP abundance better than the ground-truth delta, which supports the earlier diagnosis that the L5 ET "over-prediction" is partly a noisy 161-cell reference.

## 5. Limitations and further notes

- Ribosome-associated RNA is not cytoplasm; H2/H3 are supporting evidence (see main report). Enrichment-type (IP/input) contrasts were deliberately not repeated here although GSE172484, GSE202172, GSE141337 and GSE99791 include input samples; they could extend the property-mismatch analysis.
- Full-length RNA-seq counts carry gene-length bias that 3' UMI pseudobulks do not; ranks mitigate but do not remove it. It affects nucleus, imputed and real-cell comparisons equally.
- Cell-type mapping is approximate: Scnn1a (L4) and Rbp4 (pan-L5) are excluded because the model has no matching subclass, CamK2a is a broad driver, Htr3a is a three-type pool, Colgalt2 SOD-negative samples are whole cortex rather than M1. Region differs from MOp for most sets (only GSE172484 and half of GSE99791 are motor cortex).
- Small replicate numbers (2) for Rbp4 and the GSE172484 baseline groups; GSE211162 is RPKM only; Furlanis counts are constitutive-exon counts with fractional values (rounded).
- Gap-closed values above 1 mean the imputed profile is marginally closer to the external profile than the real-cell mean; differences are 0.00-0.03 and should not be read as "better than real cells".
- Not verified: publication citations for the new series (only GEO records were read).
- Permutation p-values are floored at 0.001.

## Addendum 2026-09-24: L5 cell-type mapping audit

The tables above are the 2026-09-18 run. After Prashant's review the mapping was audited (`outputs/extension/metadata/l5_mapping_evidence.md`, `scripts/17_l5_mapping_audit.py`) and H2/H3 were rerun (SLURM 11168301; pre-audit tables kept as `*__pre_audit_2026-09-18.tsv`):
- Scnn1a L4 (GSE133291) is excluded: layer-4 spiny stellate driver, motor cortex has no L4.
- Rbp4 (GSE142028) was first re-mapped to the mean of L5 IT and L5 ET (pan-L5 IT + ET driver, Gerfen 2013: nucleus 0.462 -> imputed 0.726, real cell 0.721), then **excluded on 2026-09-25** under the rule that a population with no single matching subclass is skipped rather than pooled (job 11181236; pooled-version tables kept as `*__rbp4_pooled_2026-09-24.tsv`).
- Gprin3, Colgalt2, Glt25d2 -> L5 ET and S100a10 -> L5 IT are confirmed by projection evidence in the source papers and by real-cell marker scoring; numbers unchanged.
- Specificity summary becomes 13/17 (imputed), 5/17 (nuclei), 11/17 (real cells) after excluding Scnn1a and Rbp4 (with Rbp4 pooled it was 14/18, 5/18, 12/18; before the audit 15/19, 5/19, 13/19). H2 and H3 values of every kept population are unchanged.
- Marker cross-check redone with an external list (Yao 2021 MOp Supplementary Table 6, `outputs/extension/metadata/yao2021/`, script 18): ET minus IT +0.60 Gprin3, +0.65 Colgalt2, +0.26 Glt25d2, -0.18 S100a10, +0.01 Rbp4, -0.69 Scnn1a; every call matches the tracing literature; no model pseudobulk is used, so the check is independent of the H2 ceiling.
