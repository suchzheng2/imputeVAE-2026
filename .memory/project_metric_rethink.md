---
name: Evaluation metric rethink
description: Proposed metric hierarchy — per-cell fidelity first, CT-level demoted. New metrics: conditional R², distribution match, variance calibration.
type: project
---

Evaluation metric rethink (2026-04-07):

**Tier 1 — Per-cell fidelity (most important)**
- Variance ratio (existing): imputed within-CT var / real cell within-CT var
- Within-CT Spearman (existing): per-gene rank preservation within each CT
- Per-gene distribution match (NEW): Wasserstein or KS test per gene per CT
- Conditional R² (NEW): R² within each CT after removing CT mean — isolates per-cell prediction quality. Model outputting CT means scores 0.
- Variance calibration plot (NEW viz): scatter of per-gene variance imputed vs real cell

**Tier 2 — Downstream utility (most meaningful)**
- Classification improvement (existing)
- DE preservation (existing)
- Coexpression (existing)
- Neighborhood preservation (NEW): kNN overlap in expression space

**Tier 3 — CT-level (demoted, necessary but insufficient)**
- Gene correlation across 17 CTs: DEMOTE — inflated, n=17 too small, blind to variance collapse
- R² across CT means: keep as secondary
- Shift scores: demote to diagnostic only

**Key visualizations:**
1. Variance calibration scatter (per-gene, imputed vs real cell variance, colored by expr quintile)
2. Within-CT distribution violins (raw cell | raw nuc | imputed, key genes)
3. Per-gene Spearman heatmap (genes × CTs)
4. Metric dashboard by expression quintile (radar/bar)
5. Classification confusion matrices (raw nuc vs imputed)

**Why:** CT-level gene correlation (0.91) looks good but hides complete variance collapse (0.085). Model is essentially a CT-mean lookup table (z ablation corr 0.995 confirms). Need metrics that expose this directly.

**How to apply:** Reorder all reports/presentations to lead with Tier 1 metrics. Implement conditional R² and distribution match in analysis/eval/metrics/. Variance calibration plot should be the first figure in any model comparison.
