---
name: project-sex-composition
description: v3 cell data = 6 male + 6 female libraries (sex = library), v3 nuclei all female; Xist is the top-variance gene in every cell type and imputes to a single mode
metadata:
  type: project
---

Verified 2026-09-25 from raw Xist / Eif2s3y / Uty counts (job 11182923, `notes/experiments/variance_zero_nonzero_fullmodel/variance_rank/tables/sex_composition_by_library.csv`):
- Cells (c3A): male libraries 1,2,3,8,11,12 (33,573 cells); female libraries 4,5,6,7,9,10 (37,610 cells). 47% male, similar in every cell type.
- Nuclei (n3A): all 5 libraries female.
- Y genes are absent from `training_gene_list_v3.csv` (intersection filter). Ddx3y is unreliable as a male marker (Ddx3x misassignment); use Eif2s3y or Uty.

**Why:** Xist is the highest-variance gene in real cells in all 17 types and imputes to one mode (1.9); every cell-vs-nucleus delta for X/Y genes carries a sex-composition component; library and sex are confounded in cells.
**How to apply:** exclude Xist from variance / top-N comparisons or restrict the cell reference to the six female libraries; for retraining use female cells only or add a sex covariate; do not interpret between-library effects in cells as purely technical. See [[project-variance-zero-nonzero-fullmodel]].
