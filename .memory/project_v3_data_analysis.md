---
name: V3-only data analysis results
description: Filtering cutoff analysis for v3-only cell and nuclei data — key stats, distributions, and recommended configs
type: project
---

V3-only data is clean with unimodal distributions (bimodality was v2/v3 batch effect).

**Cell (v3):** 71,183 cells × 26,852 genes, median lib size 46,148, median 7,282 genes/cell
**Nuclei (v3):** 40,446 cells × 25,871 genes, median lib size 16,099, median 3,446 genes/cell

Key findings:
- Cell filtering barely matters — data is already high quality, even 1st percentile has 7,268 UMI / 2,865 genes
- Gene filtering is the main lever: ~35-40% of genes have <1% detection rate (noise)
- Softmax concentration severe: nuclei top 10 genes = 18.1% of expression (Malat1 dominates)
- Sweet spot is Config B: det>5%, mean>0.01, lib>2000, genes>1500 → ~11-13K genes, <1% cell loss

**Why:** Need these stats to guide filtering choices for v3-only training runs.
**How to apply:** Use Config B as default filtering for v3 training. Cell filtering is nearly a no-op; focus on gene filtering to reduce softmax dimension.
