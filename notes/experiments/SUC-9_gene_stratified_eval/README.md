# SUC-9: Gene-Stratified Evaluation by Cell-Nucleus Gap Size

**Linear issue:** [SUC-9](https://linear.app/suchen/issue/SUC-9/gene-stratified-evaluation-by-cell-nucleus-gap-size)
**Status:** Backlog (blocked by SUC-5, SUC-13)
**Priority:** Medium

## Problem

Our 0.979 gene correlation is likely dominated by ~8K housekeeping genes that barely differ between modalities. The real value of imputation is on genes with the largest cell-vs-nucleus gap (cytoplasmic RNA, immediate early genes, etc.).

## Approach

1. Compute per-gene "gap score": `abs(mean_cell - mean_nucleus)` or fold-change
2. Bin genes into quintiles by gap size
3. Re-run all metrics (CT-level + sample-level) per stratum
4. Compare VAE vs linear regression per stratum

## Scripts

- `analysis/gene_stratification.py` — existing, CT-level only
- Will be replaced by `analysis/eval/metrics/gene_stratified.py` (SUC-13)

## Notes

This is now folded into SUC-13 (eval refactor) as part of the enriched gene stratification module. The gene_stratified.py module will compute both CT-level and sample-level metrics per quintile.
