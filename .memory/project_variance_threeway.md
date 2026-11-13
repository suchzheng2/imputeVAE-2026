---
name: Three-way variance comparison (raw vs recon vs imputed)
description: Per-sparsity-bin stats comparing raw cell, cell reconstruction, and imputed nuc→cell — shows imputed compresses std further than recon
type: project
---

Three-way comparison of nonzero-only stats (2026-04-14), median across (CT, gene) pairs:

| Sparsity | Raw cell std | Recon std | Imputed std |
|---|---|---|---|
| >70% nz | 0.545 | 0.360 | 0.231 |
| 30-70% | 0.364 | 0.208 | 0.121 |
| 10-30% | 0.260 | 0.102 | 0.049 |
| <10% nz | 0.197 | 0.026 | 0.014 |
| Overall | 0.360 | 0.212 | 0.114 |

**Why:** Cross-decode (imputation) adds ~36% additional std loss on top of same-modality reconstruction. Imputed overall std is 1/3 of raw. This is the actual output users see.

**How to apply:** When reporting imputation quality, always show imputed stats separately from cell reconstruction. Don't use cell-recon numbers as proxy for imputation quality — imputation is measurably worse.

Source: `notes/experiments/SUC-6_variance_retention/figure_analysis_2026-04-14.md`
Data: `notes/experiments/zero_nonzero_analysis/pct5v2_cond/within_ct_nonzero_gene_stats.csv`
