---
name: project-paper-status
description: Paper readiness survey 2026-10-06 — what results exist and the ranked gaps (split-aware benchmark unfinished, per-cell weakness, no downstream use case, no ablations)
metadata:
  type: project
---
Full survey in `notes/paper_status_2026-10-06.md`. No manuscript draft exists yet.

Top gaps found 2026-10-06:
- Split-aware benchmark never finished: job 8575090 crashed on a CT-mean `var < 1e-8` assert (float noise) before saving; scVI/scANVI/scGen bench jobs failed (scvi-tools missing in env), scButterfly OOM. All current benchmark numbers use all data incl. cVAE training nuclei.
- Method advantage over CT-mean/linreg exists only per-cell, where model is weak (see [[project-variance-zero-nonzero-fullmodel]]); library-size-covariate retrain not started.
- No downstream use case, no final-model ablations, one seed, Ding/Bakken/Scala holdouts not started.
- cVAE 21.9M params vs scButterfly 10.8M: parsimony cannot mean fewer params.

**Why:** user asked for a where-are-we / what's-missing review for the paper.
**How to apply:** start future paper work from the ranked list in the notes file; fix split-aware eval first.
