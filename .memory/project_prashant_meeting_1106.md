---
name: project-prashant-meeting-1106
description: Prashant meeting 2026-11-06 (filed 2026-11-13) — covariate variance 0.002 questioned (per-gene distribution for L5 IT, cost function with/without covariate), validation "very nice", generalisation is the remaining item; action items + status + email draft in notes/meeting_2026-11-06_prashant.md
metadata:
  type: project
---

Note: `notes/meeting_2026-11-06_prashant.md`. Status of his asks (2026-11-13): per-gene L5 IT variance distribution DONE
(figures of 2026-11-06, review §6.8: fixed-library decode 0.002 = latent biology only, quantile-map decode 0.08 = old
model level, rank 0.92; distributions unimodal); cost function DONE (reconstruction identical, alignment 0.024 -> 0.007,
decode gene corr 0.90 -> 0.97, R² 0.69 -> 0.93); numerator/denominator DONE: library size is only 16 % of real within-type
variance (0.39 -> 0.28), fixed decode still 100x below; **Poisson-sampling the quantile-mapped decode gives 0.295 vs 0.391
(75 %), Spearman 0.96** -> the reduction is technical, biology compressed; make it a predict_cvae decode mode (TODO); SCORCH spatial = Allen MERFISH (single-cell, qualifies), Yale's is
human Visium/Xenium; OPEN: ask Sestan lab / lab members for human single-CELL motor cortex 10x data (email draft in
the note); European nuclei + bulk request pending.

**Why:** Prashant judges within-type variance by whether biological variability survives, not by the ratio; the
decision-relevant number is the decode accuracy gain. Next meeting week of 2026-11-16.
**How to apply:** present the L5 IT distribution figure + loss table + Xenium in-situ result; do not lead with the 0.002.
