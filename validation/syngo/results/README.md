# SynGO Biological Consistency Analysis

## Overview

This analysis evaluates whether ImputeVAE reproduces the cell-type-specific
nucleus-to-cell expression shifts among SynGO-annotated synaptic genes, with
stronger shifts in neuronal populations.

SynGO annotates the synaptic function and localization of proteins. This is
not a direct measurement of subcellular RNA localization. Acceptable
interpretation: "ImputeVAE reproduces the cell-type pattern of nucleus-to-cell
shifts among SynGO-annotated genes."

Because SynGO annotates synaptic protein function and localization rather than
subcellular RNA localization, this analysis establishes neurobiological
consistency rather than direct recovery of dendritically localized RNA.

## Input Files

| Input | Path | Shape | Normalization |
|-------|------|-------|---------------|
| Gene list | `data/training_gene_list_v3.csv` | 10,134 genes | N/A |
| Split manifest | `data/split_manifest.tsv` | 111,283 x 5 | N/A |
| Ortholog mapping | `data/external/mouse_human_orthologs.csv` | 20,089 x 2 | N/A |
| SynGO archive | `validation/syngo/syngo1.3_complete_data.zip` | v1.3 | N/A |
| Validation cells | h5ad (10X_cells_v2_v3_AIBS, c3A) | 14,224 x 10,134 | log1p |
| Validation nuclei | h5ad (10X_nuclei_v2_v3_AIBS, n3A) | 8,033 x 10,134 | log1p |
| ImputeVAE predictions | `run/eval_v3_splitaware/cvae_imputed_valid.npy` | 8,033 x 10,134 | log1p |

## Split Definitions

- **Training:** 60% of observations (24,099 nuclei, 42,669 cells) -- used only for expression-matched control construction
- **Validation:** 20% of observations (8,033 nuclei, 14,224 cells) -- used for all reported metrics
- **Test:** 20% -- not loaded or used
- Split is observation-level, stratified by cell type, random_state=42
- All 5 nuclei donors and 12 cell donors appear in all partitions
- Split manifest: `data/split_manifest.tsv`

## Normalization

All expression matrices (raw cell, raw nucleus, ImputeVAE cross-decoded) are on
the same log1p scale: `log1p(raw_counts)`. No CPM normalization is applied.

## Gene Alignment

- Raw cell and nuclei h5ad files are filtered to common genes intersected with
  the 10,134-gene model feature list, then sorted by gene name
- ImputeVAE predictions use the same 10,134 genes in the same sorted order
- Gene alignment verified: all three matrices share identical gene order

## Observation Alignment

- Validation nuclei identified via split manifest positional indices
- Cell type alignment between imputed and observed validation nuclei verified at 100%
- The same validation nuclei appear in cross-decoded and observed-nucleus matrices

## Ortholog Mapping

- **Source:** `data/external/mouse_human_orthologs.csv` (Ensembl BioMart, pre-existing)
- **Method:** One-to-one human-to-mouse mapping (mouse symbols unique, human has many-to-one)
- **Fallback:** Capitalization heuristic (HUMAN -> Human) for 53 genes not in the ortholog table,
  explicitly labeled as `capitalization_fallback` in the mapping table
- 1,711 genes mapped via ortholog table, 53 via capitalization fallback
- 1,437 mapped genes present in the 10,134-gene model feature space
- 72 Rpl/Rps ribosomal genes, 1,365 non-ribosomal genes
- Sensitivity analysis excluding capitalization-fallback genes: 1,244 genes,
  Pearson r = 0.982, results qualitatively unchanged

## SynGO Version and Gene Sets

- **Version:** SynGO 1.3 (March 2025)
- **genes.xlsx:** 1,789 genes (1,789 unique HGNC symbols)
- **annotations.xlsx:** 5,076 annotations (1,796 unique HGNC symbols)
- **ontologies.xlsx:** 303 terms (one row per term, comma-separated gene lists)

The discrepancy between 1,789 (genes.xlsx) and 1,796 (annotations.xlsx) unique
symbols arises because annotations.xlsx includes paralog-group entries not
listed individually in genes.xlsx. See `syngo_annotation_discrepancies.csv`.

### Ontology Structure (nested, overlapping)

Gene sets are defined from the hierarchical ontology in ontologies.xlsx.
Each term's gene list includes all child-term genes (propagated up).
Nested sets are NOT statistically independent. Subgroup bars in the main
figure should not be interpreted as independent hypothesis tests.

| Set | GO ID | Human genes | Mouse model genes | Non-ribo |
|-----|-------|-------------|-------------------|----------|
| Synapse (broad) | GO:0045202 | 1,678 | 1,346 | 1,275 |
| Presynaptic only | GO:0098793 minus GO:0098794 | 380 | 285 | 285 |
| Postsynaptic only | GO:0098794 minus GO:0098793 | 618 | 523 | 520 |
| Both pre+post | GO:0098793 intersect GO:0098794 | 466 | 379 | 312 |
| Postsynaptic density | GO:0014069 | 362 | 321 | 300 |
| Presynaptic active zone | GO:0048786 | 109 | 89 | 89 |
| Synaptic vesicle | GO:0008021 | 163 | 129 | 129 |
| Synaptic cytosol | GO:0099523 union GO:0099524 | 81 | 71 | 71 |

## Rpl/Rps Exclusion

Primary analysis excludes genes whose mouse symbol starts with `Rpl` or `Rps`
(72 genes). These ribosomal protein genes are cytoplasmic and expected to show
large positive shifts regardless of synaptic biology. A sensitivity analysis
including Rpl/Rps is also performed.

## Control Matching

- **Pool:** 8,780 model genes not in broad SynGO set and not Rpl/Rps
- **Variables:** mean log1p nuclear expression and nuclear detection rate (training nuclei only)
- **Method:** 1:1 nearest-neighbor matching without replacement, greedy by distance
- **Standardized mean differences:** mean expression = 0.027, detection rate = 0.003 (both < 0.1)
- **Result:** 1,275 matched pairs for the broad non-ribosomal SynGO set

## Shift Definitions

For gene g and cell type c:
- `delta_empirical[g,c] = mean(validation_cell[g,c]) - mean(validation_nucleus[g,c])`
- `delta_predicted[g,c] = mean(ImputeVAE_cross_decoded[g,c]) - mean(validation_nucleus[g,c])`

Positive delta = higher expression in whole-cell (observed or predicted) than in nuclei.

## Enrichment Definition and Shared-Nucleus-Term Note

For each matched SynGO/control pair p and cell type c:
- `pair_diff_empirical[p,c] = delta_empirical[SynGO_gene,c] - delta_empirical[control,c]`
- `pair_diff_predicted[p,c] = delta_predicted[SynGO_gene,c] - delta_predicted[control,c]`
- `E_empirical[c] = mean(pair_diff_empirical)` across all matched pairs
- `E_predicted[c] = mean(pair_diff_predicted)` across all matched pairs

**Shared-nucleus-term note:** Both E_empirical and E_predicted subtract the same
nuclear contrast (mean_nucleus[SynGO] - mean_nucleus[control]). This shared
subtracted term could inflate the correlation between E_empirical and E_predicted.

To address this, we decompose:
- `A_cell[c]` = cell SynGO-minus-control contrast (no nucleus)
- `A_imputed[c]` = imputed SynGO-minus-control contrast (no nucleus)
- `A_nucleus[c]` = nucleus SynGO-minus-control contrast
- Verified: E_empirical = A_cell - A_nucleus, E_predicted = A_imputed - A_nucleus

Direct agreement between A_cell and A_imputed (nucleus term removed):
- Pearson r = 0.984, Spearman rho = 0.946
- R^2 = 0.950, MAE = 0.0207
- Slope = 1.001, Intercept = 0.0199

The correlation remains high after removing the shared nuclear term, confirming
that the agreement is not an artifact of the shared subtraction.

## Bootstrap Procedure

- 2,000 bootstrap replicates resampling matched gene pairs with replacement
- 95% confidence intervals from 2.5th and 97.5th percentiles
- All reported intervals are **gene-pair bootstrap intervals** that quantify
  uncertainty over the selected gene pairs
- These intervals do **not** capture cell/nucleus sampling uncertainty, which
  is instead assessed through the rare-cell-type exclusion and donor-balanced
  sensitivity analyses

## Multiple Testing

- Benjamini-Hochberg correction applied across subgroup hypothesis tests
- Transport and translation categories treated as exploratory/positive-control

## Rare Cell Types

Cell types with < 20 validation cells or nuclei are flagged:
- Endo: 37 cells, **4 nuclei**
- Macrophage: 24 cells, **11 nuclei**
- OPC: 29 cells, **15 nuclei**

Positive point estimates were observed in all 17 cell types. However, the
gene-pair bootstrap does not address cell-sampling uncertainty for rare cell
types (Endo: 4 nuclei, Macrophage: 11 nuclei, OPC: 15 nuclei). Strong evidence
of positive enrichment should not be claimed for these cell types from the
current analysis.

## Donor-Balanced Analysis

- 5 shared donors (nuclei donors 1-5 overlap with cell donors 1-5)
- Donor-level cell-type means computed before averaging across donors
- Requires >= 5 observations per donor/modality/cell type
- Feasible for 12 cell types (Endo, L5 ET, Macrophage, OPC, Pvalb excluded)
- Cell types with only 2 shared donors (e.g., Astro) should not be described as
  strongly replicated
- Enrichment computed using the same matched SynGO/control pairs
- Results in `syngo_donor_balanced_summary.csv`

## Baseline Comparison

**PROVISIONAL:** All baseline models (CT-mean, scVI, scANVI, Linear_reg, scGen,
scButterfly) were trained on ALL data, not the training partition only. Their
predictions are aligned by positional order and verified via 100% cell-type
match, but no observation-ID column is available in baseline obs files to
confirm row identity. This comparison is therefore PROVISIONAL and should not
be used as a headline result.

## Software Versions

- Python 3.12, conda env `impute2`
- NumPy, Pandas, SciPy, scikit-learn, anndata, matplotlib, openpyxl

## Exclusions

1. Test partition not loaded or used
2. L6 IT Car3, SMC, VLMC excluded (cell-only, no nuclei)
3. Training observations used only for control matching, not for any reported metrics

## Interpretation Limitations

1. SynGO annotates protein function/localization, not RNA subcellular distribution.
   Observed shifts could reflect transcription, RNA transport, or decay differences.
2. The ortholog mapping includes a capitalization fallback for 53 genes (3.7% of
   mapped). Sensitivity analysis excluding these genes shows no qualitative change.
3. E_empirical and E_predicted share the same subtracted nuclear contrast; the
   direct A_cell vs A_imputed comparison (r = 0.984) and MAE
   (0.0207) are the non-circular calibration checks.
4. Confidence intervals are gene-pair bootstrap intervals and do not capture
   cell/nucleus sampling uncertainty.
5. Rare cell types (Endo, Macrophage, OPC) have very few validation observations;
   positive point estimates for these types should be interpreted cautiously.
6. All baseline comparisons use models trained on ALL data with positional
   alignment only (no obs_id); results are provisional.
7. Donor-balanced analysis is limited by small donor overlap (5 shared donors).
8. Nested ontology sets are not statistically independent -- subgroup results
   should be interpreted with this caveat.

## Output Files

### Tables
- `input_audit.csv` -- Input data audit
- `syngo_resource_summary.csv` -- SynGO characterization
- `syngo_evidence_summary.csv` -- Evidence type breakdown
- `syngo_annotation_discrepancies.csv` -- Gene list discrepancies
- `syngo_mouse_mapping.csv` -- Full ortholog mapping
- `syngo_filtering_flow.csv` -- Gene filtering stages
- `syngo_subgroup_counts.csv` -- Per-subgroup gene counts
- `celltype_classification.csv` -- Neuronal/non-neuronal labels
- `matched_controls.csv` -- Expression-matched control pairs
- `matching_qc.csv` -- Matching quality metrics
- `gene_celltype_shifts.csv` -- All gene x cell type shifts
- `celltype_syngo_enrichment.csv` -- Per-CT enrichment with CIs
- `bootstrap_primary_results.csv` -- Primary enrichment results
- `syngo_subgroup_results.csv` -- Subgroup enrichment results
- `syngo_recovery_metrics.csv` -- Recovery accuracy (pooled)
- `syngo_per_celltype_metrics.csv` -- Per-CT recovery accuracy
- `syngo_per_gene_metrics.csv` -- Per-gene recovery accuracy
- `syngo_sensitivity_results.csv` -- Sensitivity analyses
- `syngo_donor_sensitivity.csv` -- Donor-balanced per-gene shifts
- `syngo_donor_balanced_summary.csv` -- Donor-balanced enrichment summary
- `syngo_method_comparison.csv` -- Baseline comparison (PROVISIONAL)
- `syngo_shared_term_audit.csv` -- Shared nucleus term decomposition
- `syngo_direct_cell_imputed_agreement.csv` -- Direct A_cell vs A_imputed

### Figures (PNG + PDF)
- `syngo_validation_main` -- 3-panel main figure
- `syngo_direct_cell_imputed_agreement` -- A_cell vs A_imputed scatter
- `syngo_donor_balanced_summary` -- Donor-balanced enrichment scatter
- `syngo_baseline_comparison` -- Method comparison (PROVISIONAL)
- `syngo_matching_qc` -- Expression matching quality
- `syngo_shift_distributions` -- SynGO vs control shift distributions
- `syngo_per_ct_heatmap` -- Per-cell-type recovery accuracy
- `syngo_filtering_counts` -- Gene filtering flow
- `syngo_ribo_sensitivity` -- Rpl/Rps sensitivity comparison
- `syngo_rare_ct_sensitivity` -- Rare cell-type sensitivity
- `syngo_subgroup_overlap_upset` -- Subgroup overlap
- `syngo_evidence_composition` -- Evidence type composition
- `syngo_subgroup_detail` -- Subgroup detail
