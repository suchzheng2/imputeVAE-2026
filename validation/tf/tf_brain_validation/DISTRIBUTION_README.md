# JASPAR TF Distribution-Level Analysis

## Method
For each TF gene g and cell type c:

W_raw[g,c] = Wasserstein(raw_nucleus_log1p[g,c], raw_cell_log1p[g,c])
W_imputed[g,c] = Wasserstein(imputed_cell_log1p[g,c], raw_cell_log1p[g,c])
absolute_improvement[g,c] = W_raw[g,c] - W_imputed[g,c]
normalized_improvement[g,c] = 1 - W_imputed / W_raw (W_raw >= 0.01)

Positive absolute_improvement means imputation moved the distribution closer
to observed whole-cell expression.

## Notes
- All comparisons use log1p-normalized values (same scale as model input/output).
- Because ImputeVAE deterministic decoder outputs are dense, detection at
  threshold 0 is not very informative.

## Files
- tf_distribution_gene_celltype.csv: Per-gene per-CT Wasserstein distances
- tf_distribution_by_celltype.csv: Per-CT summary
- tf_distribution_by_group.csv: By neuronal/non-neuronal group
- tf_distribution_validation.png/pdf: Visualization
- tf_distribution_sensitivity_summary.csv: Sensitivity across CT exclusion settings
- tf_distribution_sensitivity_panel.png/pdf: Sensitivity panel
