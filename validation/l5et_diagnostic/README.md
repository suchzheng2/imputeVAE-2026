# L5 ET Diagnostic Analysis

## Purpose
Diagnose why L5 ET is a recurrent outlier in ImputeVAE SynGO and TF validation.
This analysis does NOT modify primary SynGO or TF results.

## Inputs
All primary comparisons use existing validation-only CSVs from:
- `validation/syngo/syngo_brain_validation/`
- `validation/tf/tf_brain_validation/`

Cells and nuclei are unpaired populations; mean-shift is computed from
independent cell-type-level averages.

## Key L5 ET Numbers

### SynGO
| Metric | L5 ET | L5 IT |
|--------|-------|-------|
| n_cell | 32 | 5944 |
| n_nucleus | 249 | 2320 |
| Empirical mean shift | 0.5504 | 0.8839 |
| Predicted mean shift | 1.1761 | 0.9557 |
| Prediction error | 0.6257 | 0.0718 |
| Pearson | 0.9467 | 0.9935 |
| R2 | -0.3231 | 0.9737 |
| Slope | 1.281 | 1.019 |
| Intercept | 0.4710 | 0.0552 |

### TF
| Metric | L5 ET | L5 IT (ref) |
|--------|-------|-------------|
| Prediction error | 0.4317 | 0.0323 |
| Pearson | 0.8758 | 0.9890 |
| R2 | -0.6571 | 0.9728 |

## Is L5 ET globally over-shifted or gene-specific?
- SynGO L5 ET: slope=1.281, intercept=0.4710
- SynGO L5 IT: slope=1.019, intercept=0.0552
- L5 ET fraction overpredicted: 96.2%

The high intercept (0.471) and fraction overpredicted (96.2%)
indicate a **global positive bias** rather than a few gene-specific outliers.
The model systematically over-amplifies the nucleus-to-cell shift for L5 ET.

## Profile similarity
See `syngo_l5et_profile_similarity.csv` and `tf_l5et_profile_similarity.csv`.
These show whether the L5 ET imputed profile more closely resembles L5 IT or another
neuronal population than true L5 ET whole cells.

## Sample-size caveat
L5 ET has 32 validation whole cells and 249 validation nuclei.
While not flagged by the existing <20 low-sample threshold, 32 whole cells is still
small for stable cell-type mean estimation, especially for a biologically distinct
population like L5 ET (corticospinal projection neurons).

## All-cell reference sensitivity
This analysis was run (see l5et_validation_vs_all_cells_reference.csv).
This is diagnostic only, NOT primary validation, because training cells are included.

## Conclusion
L5 ET is a recurrent calibration stress-test population. ImputeVAE preserves much
of the gene ranking but over-amplifies the nucleus-to-cell shift magnitude in L5 ET,
especially for SynGO and TF genes. This likely reflects a combination of rare-cell-type
calibration, small validation whole-cell sample size, and unpaired cell/nucleus
composition differences. The main SynGO and TF conclusions are not driven by L5 ET
because sensitivity analyses excluding L5 ET remain consistent.

## Files
- `l5et_celltype_comparison_summary.csv` — CT-level metrics for SynGO and TF
- `syngo_l5et_gene_errors.csv` / `tf_l5et_gene_errors.csv` — Gene-level L5 ET errors
- `syngo_l5et_top_overpredicted.csv` / `syngo_l5et_top_underpredicted.csv` — Top SynGO errors
- `tf_l5et_top_overpredicted.csv` / `tf_l5et_top_underpredicted.csv` — Top TF errors
- `l5et_global_bias_summary.csv` — Global bias vs gene-specific
- `syngo_l5et_profile_similarity.csv` / `tf_l5et_profile_similarity.csv` — Profile similarity
- `validation_sample_counts_by_celltype.csv` — Sample-size diagnostic
- `l5et_validation_vs_all_cells_reference.csv` — Reference sensitivity (if run)
- `l5et_diagnostic_main.png/pdf` — Main 6-panel figure
- `l5et_top_gene_errors.png/pdf` — Top gene error bar plots
- `l5et_profile_similarity.png/pdf` — Profile similarity heatmap
- `l5et_reference_sensitivity.png/pdf` — Reference sensitivity scatter (if run)
