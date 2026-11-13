# SUC-6 Figure Analysis & Metric Insights (2026-04-14)

## Context
Reviewed within-CT nonzero figures (job 3593514) with the goal of preparing
a summary for Prashant. This led to several insights about metrics and figure
improvements.

## Key Insight: Mean/Median Comparisons Are Misleading

From Figure 2 distribution examples, it's clear that comparing mean or median
between raw and decoded **hides the real problem**:

- The MSE decoder is optimized to predict the conditional mean, so mean
  comparisons look good by construction.
- For sparse/medium genes, the raw distribution is bimodal (zero spike +
  nonzero tail), but the decoder outputs a smooth unimodal bump near the
  expected value. Mean matches, but per-cell values are wrong.
- Median is even worse for sparse genes — raw median is exactly 0, decoded
  median is ~0.15; they're measuring different things.

**Metrics that actually capture the problem:**
- Within-CT per-cell Spearman correlation (currently 0.07 — confirms collapse)
- Per-gene std_ratio (Fig 1)
- Distribution divergence (KL/Wasserstein on per-gene within-CT distributions)
- Nonzero-conditioned metrics (separate eval on nonzero entries only)

## Three-Way Comparison: Raw Cell vs Cell Recon vs Imputed (nuc→cell)

All values are **median** across (CT, gene) pairs, **nonzero entries only**.
Sparsity bins based on raw cell frac_nonzero.
Imputed stats measured at nonzero nuclei positions (unpaired data).

| Sparsity | N pairs | Raw cell mean | Raw cell std | Recon mean | Recon std | Imputed mean | Imputed std |
|---|---|---|---|---|---|---|---|
| >70% nz | 57,209 | 1.569 | 0.545 | 1.447 | 0.360 | 1.414 | 0.231 |
| 30-70% | 56,105 | 1.016 | 0.364 | 0.567 | 0.208 | 0.480 | 0.121 |
| 10-30% | 34,979 | 0.865 | 0.260 | 0.205 | 0.102 | 0.155 | 0.049 |
| <10% nz | 34,920 | 0.795 | 0.197 | 0.036 | 0.026 | 0.018 | 0.014 |
| **Overall** | **183,213** | **0.998** | **0.360** | **0.513** | **0.212** | **0.427** | **0.114** |

### Key observations:
1. **Mean recovery** is similar for cell recon and imputed for dense genes
   (1.447 vs 1.414 vs 1.569 raw). Both collapse for sparse genes.
2. **Std compression worsens from recon → imputed**: Dense genes raw std
   0.545 → recon 0.360 → imputed 0.231. Cross-decode adds ~36% additional
   std loss on top of reconstruction loss.
3. **Imputed overall std is 0.114** — roughly 1/3 of raw (0.360). This is
   the actual output users would see.

## Min/Max as Additional Diagnostic

Added `raw_min`, `raw_max`, `dec_min`, `dec_max` columns to
`within_ct_nonzero_gene_stats.csv` (via `analysis/within_ct_nonzero_stats.py`).

**Why min/max is useful:**
- **Dynamic range clipping**: If raw max ~4.5 but decoded max ~3.0, that's
  mean-regression compressing the tails. Std alone doesn't capture this.
- **Decoder floor**: Raw min of nonzero entries should be ~0.69 (=log1p(1)),
  but decoded min may be ~0.02, confirming sub-count-level outputs.
- **Range ratio** `(dec_max - dec_min) / (raw_max - raw_min)` cleanly
  summarizes dynamic range survival.

Job **3596904** submitted to regenerate CSV with min/max columns.

## Figure Improvements Made

### Fig 1b (NEW): Std scatter colored by cell type
- Same as Fig 1 but colored by CT instead of sparsity
- Shows whether specific CTs (e.g., L2/3 IT vs Endo) cluster in different
  regions of the scatter

### Fig 2: Expanded to 4 rows
- Rows 1-2: Cell reconstruction (nonzero / zero) — as before
- Rows 3-4: Imputed nuc→cell (nonzero / zero) — NEW
- Same 3 genes (dense/medium/sparse) in L2/3 IT
- Each nonzero panel now annotated with:
  ```
  n=X (Y%)
  raw:  μ=..., σ=...
  dec:  μ=..., σ=...
  σ ratio: ...
  ```
- X-axis ranges shared between recon and imputed rows for same gene

Job **3595820** submitted to regenerate all figures.

## Figures to Send to Prashant
1. **Fig 1** — std scatter by sparsity (the gradient pattern)
2. **Fig 2** — distribution examples with stats (self-contained with annotations)
3. **Fig 3** — heatmap CT × sparsity (systematic across all CTs)
4. **Summary table** above (three-way comparison)
5. Skip Fig 4a/4b — too noisy for email

## Jobs
- **3595820** — figure regeneration (fig1, fig1b, fig2 4-row, fig3, fig4a/b)
- **3596904** — CSV regeneration with min/max columns
