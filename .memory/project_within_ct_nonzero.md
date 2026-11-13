---
name: Within-CT nonzero per-gene analysis
description: Corrected zero/nonzero analysis (2026-04-14) — per-gene within-CT metrics reveal std_ratio 0.57, mean_ratio 0.40, correcting earlier inflated numbers
type: project
---

## Within-CT Nonzero Per-Gene Analysis (2026-04-14)

Corrects the April 9 zero/nonzero analysis which pooled across genes (inflating Pearson to 0.86, std_ratio to 1.10).

**Method:** For each (CT, gene) pair, subset to nonzero cells, compute mean/std/var ratios of decoded vs raw. ~230K pairs total.

**Honest numbers (cell recon, median):**
- mean_ratio: 0.40 (systematic underestimation)
- std_ratio: 0.57 (43% compression of per-cell spread)
- var_ratio: 0.33
- CT mean baseline std_ratio: 0.00 (confirms model's 0.57 is real signal)

**By sparsity:** dense genes (>70% nz) get mean_ratio 0.92, std_ratio 0.66. Sparse (<10%) essentially lost.

**Corrects earlier claims:**
- "Scale preserved" → mean underestimated by 60%, spread compressed by 43%
- "Model works well on expressed genes" → partially, but substantial compression
- "Variance collapse is zero-inflation artifact" → zero-inflation explains mean but not variance compression

**Filed under:** SUC-6, findings in `notes/experiments/SUC-6_variance_retention/within_ct_nonzero_findings.md`
**Script:** `analysis/within_ct_nonzero_stats.py`, job 3592328
**Figures:** job 3593514 (pending), `notes/experiments/SUC-6_variance_retention/figures/`

**Why:** Prashant's question about scale vs mean needed honest per-gene within-CT answer, not confounded global pooling.
**How to apply:** Use these as the ground-truth numbers for variance retention. Previous zero/nonzero analysis numbers should not be cited.
