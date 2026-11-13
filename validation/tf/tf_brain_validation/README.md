# JASPAR Mouse TF Brain Validation

## Purpose
Nuclear-function contrast analysis for ImputeVAE. Tests whether ImputeVAE creates
the same strong neuronal > non-neuronal shift for TF genes as for SynGO synaptic genes,
or whether the SynGO pattern is more specific to synaptic biology.

## Important caveats
1. TFs are not a perfect negative control; TF mRNAs can still be cytoplasmic.
2. JASPAR is a motif database; rows are motif models, not one-to-one gene records.
3. Multiple motifs can correspond to the same TF; collapsed to unique TF symbols.
4. Complex motifs split on "::" for primary set, excluded in singleton-only sensitivity.
5. All expression comparisons are validation-only, unpaired population comparisons.
6. SynGO and TFs annotate encoded protein function/class, not RNA subcellular localization.

## Gene set
- Primary TF genes (non-ribosomal, in model): 63
- Singleton-only TF genes: 61

## Mean-shift results (all CTs)
- Pearson r: 0.9039
- Spearman r: 0.8617
- R2: 0.7992
- MAE: 0.1024
- RMSE: 0.1836
- Sign agreement: 0.8880

## Neuronal vs non-neuronal
- Empirical neuronal mean shift: 0.4222
- Empirical non-neuronal mean shift: 0.1653
- Empirical diff: 0.2569 [0.1612, 0.3535]
- Predicted diff: 0.3709 [0.3081, 0.4372]

## Validation data
- Split-aware validation only (held-out)
- Imputed: run/eval_v3_splitaware/cvae_imputed_valid.npy
- 17 cell types (12 neuronal, 5 non-neuronal)

## Files
- tf_gene_metadata.csv: TF gene metadata
- tf_gene_celltype_deltas.csv: Per-gene per-CT mean-shift deltas
- tf_overall_quality.csv: Overall quality metrics
- tf_quality_by_celltype.csv: (same as tf_celltype_summary.csv)
- tf_celltype_summary.csv: Per-CT summary with CIs
- tf_neuronal_vs_nonneuronal.csv: Neuronal vs non-neuronal comparison
- tf_validation_main.png/pdf: Main validation figure
- tf_sensitivity_*.csv: Sensitivity analysis results
- tf_distribution_*.csv: Distribution-level results
