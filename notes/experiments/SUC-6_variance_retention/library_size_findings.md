# Library Size Analysis Findings (2026-04-15)

**Model:** pct5v2_cond (`run_2026-04-07-02-12-05_3779214`)
**Job:** 3615438 (COMPLETED)
**Script:** `analysis/library_size_analysis.py`
**Results:** `notes/experiments/SUC-6_variance_retention/library_size/`

## Motivation

From meeting with Prashant: how much of the within-CT variance that the decoder
preserves is just library size (total UMI counts per cell) vs. gene-specific
regulation? The within-CT nonzero analysis (2026-04-14) showed std_ratio ~0.57
for dense genes — but is that real per-cell biological variation or just "big
cells get bigger outputs"?

## Key Results

### 1. Library size is almost perfectly preserved (r = 0.974)

The decoder knows each cell's total count depth and faithfully reproduces it.
Per-CT Pearson ranges from 0.74 (Macrophage, n=122) to 0.99 (L2/3 IT, n=5000).
This is not surprising — library size affects all 16K genes, so MSE strongly
rewards learning it.

### 2. Decoded variance is dominated by library size (R² = 0.96 for dense genes)

Per (CT, gene) R² of expression ~ log(library_size):

| Sparsity | Raw R² | Decoded R² | Interpretation |
|----------|--------|------------|----------------|
| >70% nz  | 0.323  | **0.957**  | Raw: libsize is 1/3 of story. Decoded: libsize is everything |
| 30-70%   | 0.116  | 0.816      | Same pattern |
| 10-30%   | 0.040  | 0.452      | Even sparse genes become libsize-dependent |
| <10%     | 0.004  | 0.166      | Decoder introduces libsize dependence that wasn't there |

The decoder doesn't just preserve library-size variation — it amplifies it
relative to everything else. The R² scatter (fig 3) shows nearly every (CT, gene)
dot above the diagonal.

### 3. After removing library size, only 17% of variation survives

std_ratio (decoded / raw) in two spaces:

| Sparsity | log1p std_ratio | Proportion std_ratio |
|----------|----------------|---------------------|
| >70% nz  | 0.550          | **0.174**           |
| 30-70%   | 0.365          | 0.144               |
| 10-30%   | 0.287          | 0.159               |
| <10%     | 0.128          | 0.056               |

The gap between 0.55 and 0.17 (dense genes) is entirely library size. Our
previous std_ratio of 0.55 was flattering — about two-thirds of the "preserved
variation" was library size scaling all genes uniformly.

### 4. Dim 31 is the library size dimension

Pearson r between each latent dim and log(library_size):
- **Dim 31: r = 0.944 (cell), 0.953 (nuclei)** — near-perfect encoding
- Next strongest: dim 14 (r = -0.64 cell), dim 24 (r = 0.48/0.59)
- Most other dims: |r| < 0.3

The encoder packs library size into a single dimension. The decoder reads it
back and scales output accordingly. This is the cheapest possible way to reduce
MSE — one dimension affects all 16K genes simultaneously.

### 5. Per-CT library size preservation

| CT | n | raw lib mean | dec lib mean | Pearson |
|----|---|-------------|-------------|---------|
| L2/3 IT | 5000 | 55,528 | 52,507 | 0.990 |
| L5 ET | 161 | 42,439 | 43,889 | 0.951 |
| Astro | 398 | 8,710 | 6,277 | 0.854 |
| Macrophage | 122 | 6,371 | 4,063 | 0.737 |

Large, well-represented CTs have near-perfect preservation. Small CTs
(Macrophage, OPC) are weaker — fewer training examples.

## Revised Understanding of Variance Collapse

The decoder is essentially computing:

```
output(cell, gene) ≈ CT_mean(gene) × f(library_size(cell))
```

This is a **1-dimensional** representation of per-cell variation. Library size
is a scalar — it scales all genes up or down together. Gene-specific variation
("this cell has unusually high gene X relative to gene Y, given its total
counts") requires the decoder to use multiple latent dims in gene-specific ways.
It doesn't.

**Why MSE makes this rational:**
- Library size: 1 latent dim × 16K genes = 16K error reductions per gradient step
- Gene-specific: 1 latent dim × 1 gene = 1 error reduction per gradient step
- With unpaired data (no matched cell to target), gene-specific variation's
  optimal prediction is the conditional mean anyway (H1). So the decoder has
  no incentive to learn it even if it could.

**Previous std_ratio numbers should be reinterpreted:**

| Metric | Old value | What it meant | Honest value |
|--------|-----------|---------------|-------------|
| std_ratio 0.55 (dense) | "55% variation preserved" | Mostly library size | 0.17 gene-specific |
| std_ratio 0.57 (overall nonzero) | "Model captures per-cell signal" | True, but only 1-dimensional | Proportion std 0.17 |

## Implications for Interventions

### What this rules out
- **Residual MSE** (already failed) — subtracting CT means doesn't help because
  library-size variation is within-CT, not between-CT
- **Larger latent** — the decoder already ignores 31 dims; adding more won't help

### What this suggests
1. **Proportion-space training** — train on counts/libsize instead of log1p.
   Removes library size as free MSE reduction, forcing the decoder to learn
   gene-specific variation. Library size can be handled separately or through
   dim 31.

2. **Frozen encoder + per-gene variance penalty** — the encoder's other 31 dims
   likely carry gene-specific info (82% within-CT variance in latent). Retrain
   decoder with explicit penalty for low per-gene output variance. Forces it to
   use dims beyond 31.

3. **Gene-specific contrastive loss** — instead of MSE across all genes, compare
   each gene's rank within a cell's profile. Invariant to library size by
   construction.

## Figures

1. `fig_libsize_scatter.png` — raw vs decoded library size per cell (r=0.974)
2. `fig_variance_decomposition.png` — R² by sparsity, raw vs decoded
3. `fig_r2_scatter_raw_vs_dec.png` — per (gene,CT) R² scatter
4. `fig_proportion_std_comparison.png` — std_ratio in log1p vs proportion space
5. `fig_latent_libsize.png` — per-dim correlation with library size
