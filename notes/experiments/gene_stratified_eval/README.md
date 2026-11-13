# Gene-Stratified Evaluation by Nucleus-Cell Gap Magnitude

**Date:** 2026-07-24
**Task:** Task 6 from analysis plan
**Script:** `analysis/gene_stratified_gap_eval.py`

## Question

Does the model help most on genes with the largest cell-nucleus gap, and does it avoid hurting genes with small gaps?

## Method

1. Computed the ground-truth per-gene gap = |mean(real_cell) - mean(real_nuclei)| across all cells/nuclei (log1p scale).
2. Binned genes into quartiles by gap magnitude (2,533-2,534 genes per bin).
3. Per bin, compared imputed (cVAE full-data model) vs raw nuclei against real cell CT means using MSE and Pearson correlation.

## Results

| Quartile | Gap range | N genes | Mean gap | MSE improvement | Corr (imputed) | Corr (raw nuc) | Corr gain |
|----------|-----------|--------:|---------:|:-:|:-:|:-:|:-:|
| Q1 (smallest) | [0.000, 0.166) | 2,534 | 0.076 | +70.8% | 0.904 | 0.809 | +0.082 |
| Q2 | [0.166, 0.403) | 2,533 | 0.279 | +83.8% | 0.911 | 0.819 | +0.088 |
| Q3 | [0.403, 0.739) | 2,533 | 0.558 | +89.5% | 0.930 | 0.838 | +0.088 |
| Q4 (largest) | [0.739, 4.282) | 2,534 | 1.209 | +95.6% | 0.945 | 0.817 | +0.125 |

## Interpretation

**Monotonic improvement across all quartiles:**

1. **Largest-gap genes (Q4) benefit most:** 95.6% MSE reduction and the highest absolute correlation gain (+0.125). These are genes where the cell-nucleus difference is biologically largest (ribosomal proteins, cytoplasmic mRNAs, nuclear lncRNAs) and where imputation adds the most value.

2. **Smallest-gap genes (Q1) are not harmed:** 70.8% MSE improvement and correlation gain of +0.082. The model correctly applies small or zero corrections to genes that are similar between cell and nucleus.

3. **No harm in any bin:** All quartiles show positive MSE improvement and positive correlation gain. The model does not introduce errors in the low-gap regime.

4. **The pattern is what biology predicts:** genes with the largest extraction-dependent differences (cytoplasmic mRNAs lost during nuclear isolation) are exactly where a cell-vs-nucleus imputation model should help most.

## Output files

- `gene_stratified_results.csv` — per-bin aggregate metrics
- `per_gene_detail.csv` — per-gene gap, bin assignment, MSE, and correlation
- `stratified_stats.json` — all statistics
- `gene_stratified_eval.png` — 4-panel figure (MSE improvement, correlation comparison, MSE by bin, gap distribution)
