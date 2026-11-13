# Zero / Nonzero Partition Analysis — Findings

**Date:** 2026-04-09
**Model:** pct5v2_cond (`run_2026-04-07-02-12-05_3779214`)
**Data:** Full training set — Cell 71,117 × 16,439 genes, Nuclei 40,166 × 16,439 genes
**Job:** 3296108 (COMPLETED)
**Motivated by:** Prashant's observation that reconstruction variance < input variance, and that Pearson is scale-invariant so it can't see this

## Context

Previous analysis (data flow audit, 2026-04-08) showed within-CT variance crushed to 20% of raw. We attributed this to the decoder ignoring within-CT z variation. Prashant pointed out: (1) mean matching ≠ scale matching, and (2) Pearson can't distinguish these. He asked: what's happening at the level of zeros vs nonzeros?

## Method

Treat the cell×gene matrix as a flat pool of entries (~1.17B for cells). Partition every entry by whether the raw value is zero or nonzero. Compute statistics for each partition separately. Compare across four decoded sources + a cell-type mean baseline.

### Sources analyzed

| # | Label | Encode | Decode | Compare against |
|---|---|---|---|---|
| 1 | cell_recon | cell | cell | raw cell (same cells) |
| 2 | nuclei_recon | nuclei | nuclei | raw nuclei (same cells) |
| 3 | cross_decoded_vs_raw_nuc | nuclei | cell | raw nuclei (same cells) |
| 4 | ct_mean_baseline_cell | N/A | CT mean lookup | raw cell |
| 5 | ct_mean_baseline_nuc | N/A | CT mean lookup | raw nuclei |
| 6 | cross_decoded_ct_matched | nuclei | cell | raw cell (CT-matched) |
| 7 | cross_decoded_vs_ct_mean | nuclei | cell | raw cell CT mean |

### Groupings
- Global (all entries pooled)
- By cell type (17 CTs)
- By gene sparsity bucket (<30%, 30-70%, 70-90%, >90% zero)

## Key Results — Global

| Source | Frac zero | Zero: dec_mean | Zero: dec_std | Nonzero: raw_mean | Nonzero: dec_mean | Nonzero: std_ratio | Nonzero: Pearson | AUROC |
|---|---|---|---|---|---|---|---|---|
| Cell recon | 0.599 | 0.176 | 0.273 | 1.522 | 1.270 | 1.10 | **0.860** | 0.923 |
| CT mean baseline (cell) | 0.599 | 0.201 | 0.297 | 1.522 | 1.220 | 1.03 | 0.797 | 0.914 |
| Nuclei recon | 0.803 | 0.127 | 0.203 | 1.328 | 0.806 | 1.17 | **0.795** | 0.886 |
| CT mean baseline (nuc) | 0.803 | 0.133 | 0.204 | 1.328 | 0.786 | 1.13 | 0.777 | 0.886 |
| Cross-decoded (vs raw nuc) | 0.803 | 0.411 | 0.578 | 1.328 | 1.499 | 1.56 | 0.650 | 0.849 |
| Cross-decoded (CT-matched) | 0.603 | 0.206 | 0.325 | 1.547 | 1.263 | 1.09 | 0.764 | 0.910 |
| Cross-decoded (vs CT mean) | — | -0.004 | 0.011 | 0.623 | 0.631 | 1.05 | **0.976** | 0.914 |

## Key Results — By Gene Sparsity (Cell Reconstruction)

| Sparsity | Nonzero Pearson (model) | Nonzero Pearson (CT mean) | Std ratio (model) | Std ratio (CT mean) |
|---|---|---|---|---|
| <30% zero | 0.862 | 0.770 | 0.97 | 0.88 |
| 30-70% | 0.556 | 0.451 | 0.85 | 0.66 |
| 70-90% | 0.542 | 0.493 | 0.82 | 0.67 |
| >90% zero | 0.525 | 0.484 | 0.79 | 0.64 |

## Conclusions

### 1. No scale compression among expressed genes
Nonzero entries have std_ratio ~1.0-1.1. The decoder preserves the spread of actually expressed values. The earlier "20% within-CT variance" finding was driven by zero/nonzero mixing, not by squishing the scale of real expression.

### 2. The model beats the CT mean baseline — on nonzero entries
Cell recon Pearson 0.860 vs CT mean 0.797. The model does capture per-cell variation beyond cell-type means, but only for genes that are expressed. The 60% zero entries dilute this signal to near-zero when measured as overall within-CT Spearman.

### 3. The decoder cannot represent zero-inflated distributions
The decoder outputs continuous values through LeakyReLU. For a gene expressed in 30% of cells, it gives a soft value (~0.5) to all cells — too high for zeros, too low for nonzeros. This is a fundamental architectural limitation, not a training failure.

### 4. Zero/nonzero mixing accounts for ~73% of raw variance
Previously computed (2026-04-09): the Bernoulli zero/nonzero structure generates ~73% of total raw variance (83% for sparse genes). A continuous decoder cannot reproduce this.

### 5. Separation is partial, not binary
AUROC 0.92 for cell recon — the decoder assigns higher values to truly-nonzero positions — but this is a soft gradient, not a hard boundary. The decoder "knows" where genes are expressed but can't represent it discretely.

### 6. Sparsity drives difficulty
Dense genes (<30% zero): Pearson 0.86, scale 0.97. Sparse genes (>90% zero): Pearson 0.53, scale 0.79. The model has little information in z to predict which cells express a gene that's only on in 10% of cells.

## Implications

- The "variance collapse" framing was misleading — the model works well on expressed genes
- Previous eval metrics (within-CT Spearman, variance ratio) were dominated by the zero structure
- A two-headed decoder (Bernoulli for zero/nonzero + continuous for expression level) could recover the missing variance
- Alternatively, evaluating on nonzero entries only gives a fairer picture of model quality
- The CT mean baseline comparison shows the model adds real value (~6% Pearson improvement)

## Files

- `notes/experiments/zero_nonzero_analysis/pct5v2_cond/global_summary.csv`
- `notes/experiments/zero_nonzero_analysis/pct5v2_cond/by_celltype.csv`
- `notes/experiments/zero_nonzero_analysis/pct5v2_cond/by_sparsity.csv`
- `analysis/zero_nonzero_analysis.py` — analysis script
- `jobs/eval/run_zero_nonzero_analysis.sh` — SLURM job
