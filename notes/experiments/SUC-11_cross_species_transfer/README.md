# SUC-11: Cross-Species Transfer — Apply to Human Nuclei Without Matching Cell Data

**Linear issue:** [SUC-11](https://linear.app/suchen/issue/SUC-11/cross-species-transfer-apply-to-human-nuclei-without-matching-cell)
**Status:** Backlog
**Priority:** High — critical for Nature Methods (generalizability)

## Problem

imputeVAE is trained on mouse data where paired cell + nuclei data exist. What is the strategy when applying to human nuclei data where no matching cell data exists?

## Key Question

Is the nucleus-to-cell expression difference conserved across species? If so, a mouse-trained model could transfer to human nuclei.

## Dependencies

- SUC-12 (characterize cross-species/tissue differences first)
