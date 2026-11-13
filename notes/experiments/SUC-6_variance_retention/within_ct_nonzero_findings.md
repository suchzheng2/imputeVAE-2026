# Within-CT Nonzero Per-Gene Analysis — Findings

**Date:** 2026-04-14
**Model:** pct5v2_cond (`run_2026-04-07-02-12-05_3779214`)
**Job:** 3592328 (COMPLETED, 11 min)
**Script:** `analysis/within_ct_nonzero_stats.py`
**Results CSV:** `notes/experiments/zero_nonzero_analysis/pct5v2_cond/within_ct_nonzero_gene_stats.csv`

## Context

The earlier zero/nonzero analysis (2026-04-09, motivated by Prashant) computed metrics by pooling all nonzero entries globally or within-CT but across all genes. This inflated Pearson (0.86) and std_ratio (1.10) because between-gene variation dominated. This follow-up computes stats **per gene within each CT** — removing both between-CT and between-gene confounding.

## Method

For each (cell type, gene) pair with >= 10 nonzero cells:
- Subset to cells where raw > 0
- Compute mean, std, var of raw and decoded
- Compute ratios (decoded / raw)

Three sources: cell reconstruction, CT mean baseline, cross-decoded (nuclei→cell).

## Key Results

### Overall (median across ~230K CT×gene pairs)

| Source | mean_ratio | std_ratio | var_ratio |
|--------|-----------|-----------|-----------|
| Cell recon | 0.40 | **0.57** | 0.33 |
| CT mean baseline | 0.32 | **0.00** | 0.00 |
| Cross-decoded | 0.41 | 0.41 | 0.16 |

### By sparsity (cell recon)

| Bucket (frac nonzero) | n_pairs | mean_ratio | std_ratio | var_ratio |
|------------------------|---------|-----------|-----------|-----------|
| <10% | 58,454 | 0.04 | 0.15 | 0.02 |
| 10-30% | 51,726 | 0.24 | 0.54 | 0.29 |
| 30-70% | 61,927 | 0.55 | 0.58 | 0.33 |
| >70% | 57,567 | 0.92 | 0.66 | 0.43 |

### By cell type (cell recon, median across genes)

| CT | n_genes | mean_ratio | std_ratio | var_ratio |
|----|---------|-----------|-----------|-----------|
| L5 ET | 11,837 | 0.73 | 0.88 | 0.78 |
| Sncg | 13,651 | 0.57 | 0.81 | 0.65 |
| Pvalb | 13,636 | 0.58 | 0.80 | 0.62 |
| Endo | 9,293 | 0.32 | 1.12 | 1.25 |
| OPC | 10,256 | 0.40 | 0.86 | 0.75 |
| Macrophage | 7,699 | 0.28 | 0.78 | 0.60 |
| Oligo | 11,882 | 0.32 | 0.70 | 0.49 |
| L6b | 14,229 | 0.52 | 0.63 | 0.39 |
| Astro | 11,205 | 0.19 | 0.55 | 0.30 |
| Sst | 15,339 | 0.43 | 0.51 | 0.26 |
| L2/3 IT | 16,146 | 0.46 | 0.42 | 0.18 |
| L5 IT | 16,291 | 0.38 | 0.43 | 0.19 |
| L6 IT | 15,859 | 0.43 | 0.40 | 0.15 |
| Vip | 15,396 | 0.30 | 0.39 | 0.15 |
| Lamp5 | 15,414 | 0.38 | 0.38 | 0.15 |
| L5/6 NP | 15,414 | 0.33 | 0.36 | 0.12 |
| L6 CT | 16,127 | 0.38 | 0.36 | 0.13 |

## Conclusions

### 1. Model captures real per-cell signal (std_ratio 0.57 vs baseline 0.00)
Not a CT-mean lookup table. The decoder does use per-cell latent variation. But it only preserves ~57% of the per-cell spread among expressed genes.

### 2. Systematic mean underestimation (mean_ratio 0.40)
The decoder compromises between zero and nonzero cells, outputting values pulled toward zero. Dense genes (>70% nonzero) get mean_ratio 0.92; sparse genes (<10%) get 0.04.

### 3. Even dense genes are compressed (std_ratio 0.66 at >70% nonzero)
The zero-compromise explains mean underestimation but not variance compression. For dense genes where zeros are rare, the decoder still loses ~34% of per-cell spread. This is the decoder bottleneck from the data flow audit.

### 4. Cross-decoded adds another layer of compression
Cross-decoded std_ratio 0.41 vs cell recon 0.57 — ~28% additional variance loss from the cross-modal path.

### 5. Previous zero/nonzero analysis was misleading
Old numbers (nonzero Pearson 0.86, std_ratio 1.10) were confounded by between-gene structure. Honest per-gene within-CT numbers are much lower.

## Corrects the earlier framing

| Claim from April 9 analysis | Correction |
|---|---|
| "Scale is preserved among expressed genes" (std_ratio 1.10) | Per-gene within-CT std_ratio is 0.57 — 43% compression |
| "Model works well on expressed genes" | Partially — captures per-cell signal but substantially compresses it |
| "Variance collapse is mostly zero-inflation artifact" | Zero-inflation explains mean underestimation, but variance compression is real even for dense genes |
| "No scale compression among expressed genes" | There is scale compression — mean_ratio 0.40 globally |

## Figures TODO

1. Per-gene scatter: raw_std vs dec_std (diagonal = perfect), colored by sparsity
2. Distribution examples: 3 genes × 1 CT at different sparsity levels (raw vs decoded histograms)
3. Heatmap: CT × sparsity bucket, colored by std_ratio
4. Nonzero scatter: raw vs decoded values for all nonzero entries, colored by CT then by gene
