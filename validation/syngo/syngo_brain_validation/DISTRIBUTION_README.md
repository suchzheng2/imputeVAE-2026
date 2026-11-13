# SynGO Distribution-Level Biological Validation

## Question

For non-ribosomal genes encoding synaptic proteins, does cross-decoding
make the expression distribution of validation nuclei more similar to the
observed whole-cell distribution? Does the degree of distributional recovery
differ across neuronal and non-neuronal cell types?

## Relationship to Mean-Shift Analysis

This analysis uses exactly the same 1,275 non-ribosomal SynGO genes,
validation observations, gene ordering, cell-type definitions, and subgroups
as the completed mean-shift analysis (syngo_brain_validation.py). Only the
evaluation metrics differ.

## Inference Audit

- **Encoding:** Posterior mean (mu_latent). No reparameterization sampling.
- **Decoding:** Deterministic conditional mean. Final activation: LeakyReLU(0.1).
- **Draws per nucleus:** 1 (single forward pass).
- **Clipping/thresholding:** None applied to saved predictions.
- **Scale:** log1p (same as input).
- **Dense output:** YES (0.000000 exact zeros).
- **Negative values:** 977704 (0.012010).

### Implications for Distribution Recovery

Deterministic mean decoding produces a single point estimate per nucleus,
not a sample from the predictive distribution. The resulting imputed
distribution is expected to be narrower than the real whole-cell distribution
because:

1. Per-cell biological variation is not sampled.
2. The mean decoder output smooths over gene-specific noise.
3. Nearly all imputed values are nonzero, even for genes with high dropout
   in real cells.

This means detection-rate analysis at threshold=0 is uninformative for the
imputed matrix. Sensitivity analyses at higher thresholds are provided.

## Distributional Metrics

### Wasserstein-1 Distance

For each SynGO gene g and cell type c:
- W_raw[g,c] = Wasserstein-1(nucleus[g,c], cell[g,c])
- W_imputed[g,c] = Wasserstein-1(imputed[g,c], cell[g,c])
- absolute_improvement[g,c] = W_raw - W_imputed

Positive = imputation moved distribution closer to cells.

### Normalized Improvement

normalized_improvement = 1 - W_imputed / W_raw

Minimum denominator: W_raw >= 0.01.

### Detection-Rate Recovery

detection_error_reduction = |nucleus_rate - cell_rate| - |imputed_rate - cell_rate|

Prespecified thresholds (log1p scale): [0.0, 0.1, 0.5, 1.0]

### Positive-Expression Wasserstein

Wasserstein-1 computed on observations exceeding the detection threshold.
Minimum 10 positive observations required per distribution.

**NOTE:** Because the decoder output is dense, positive-expression
analysis uses threshold-dependent definitions. Results are labeled
accordingly and should be interpreted as sensitivity analyses.

## Key Results

### All
- Gene-CT pairs: 21675
- Median W_raw: 0.5660
- Median W_imputed: 0.3238
- Median absolute improvement: 0.1504
- Proportion improved: 0.601

### Neuronal
- Gene-CT pairs: 15300
- Median W_raw: 0.7934
- Median W_imputed: 0.3295
- Median absolute improvement: 0.4224
- Proportion improved: 0.747

### Non Neuronal
- Gene-CT pairs: 6375
- Median W_raw: 0.1778
- Median W_imputed: 0.3077
- Median absolute improvement: -0.0588
- Proportion improved: 0.253

### Low Sample
- Gene-CT pairs: 3825
- Median W_raw: 0.1872
- Median W_imputed: 0.2892
- Median absolute improvement: -0.0412
- Proportion improved: 0.298

### Excluding Low Sample
- Gene-CT pairs: 17850
- Median W_raw: 0.6931
- Median W_imputed: 0.3312
- Median absolute improvement: 0.2873
- Proportion improved: 0.666

## Denominator Sensitivity

- W_raw >= 0.001: n=21484.0, median_norm=0.2812
- W_raw >= 0.005: n=21459.0, median_norm=0.2828
- W_raw >= 0.010: n=21386.0, median_norm=0.2878
- W_raw >= 0.020: n=21187.0, median_norm=0.2983
- W_raw >= 0.050: n=20372.0, median_norm=0.3409
- W_raw >= 0.100: n=18906.0, median_norm=0.4032

## Interpretation

Cells and nuclei are not physically paired. These are marginal
population-distribution comparisons within cell type.

Deterministic mean decoding produces distributions that are narrower
than real whole-cell distributions. Distribution recovery should be
interpreted as a lower bound on what sampling-based decoding might achieve.

## Output Files

- `syngo_distribution_gene_celltype.csv` - All gene x CT distributional metrics
- `syngo_distribution_by_celltype.csv` - Per-CT summaries with bootstrap CIs
- `syngo_distribution_by_group.csv` - Group-level summaries
- `syngo_distribution_by_subgroup.csv` - Subgroup summaries
- `syngo_distribution_mean_relationship.csv` - Mean-shift vs distribution
- `syngo_distribution_validation.png/pdf` - 6-panel main figure
- `syngo_distribution_examples.png/pdf` - Example gene ECDFs
- `DISTRIBUTION_README.md` - This file
