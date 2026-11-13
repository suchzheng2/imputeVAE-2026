---
name: project-variance-zero-nonzero-fullmodel
description: Zero/nonzero + technical-vs-biological variance diagnostic on the full-data cVAE (2026-09-18); imputed within-CT variance is ~1-D library-size axis; do not add variance loss yet
metadata:
  type: project
---

# Variance diagnostic on full-data cVAE (2026-09-18) — DONE

- Report: `notes/experiments/variance_zero_nonzero_fullmodel/README.md` (tables/, figures/ F1-F9)
- Scripts: `analysis/variance_zero_nonzero_fullmodel.py` (job 10910494), `_addon.py` (job 10910608), `_figures.py`
- Real-cell within-CT variance (log1p): 52% Poisson sampling, 25% library size, 23% biological excess (Poisson simulation; upper bound on biological)
- Imputed/total = 0.15; imputed/(total - Poisson) = 0.32; **library-size axis removed from both sides = 0.12** (the honest retention number). The 0.67 figure (imputed total vs lib-free excess) is NOT like-for-like; do not quote.
- **82% of imputed within-CT variance is explained by input nucleus library size**; participation ratio 1.1-2.1 (real cells 8-18); gene-gene correlations inflated, structure match to real cells inconsistent (0.09-0.87)
- Zeros: model ignores a nucleus's own zero (gap 0.04 log1p vs nonzero mean 1.0); 46% of imputed variance sits at nucleus-zero entries; mixing is 12% of imputed variance vs 80% nuclei, 58% cells
- Nonzero entries: var ratio imputed/nucleus 0.39; paired r 0.10 median (0.32 for >70% detected, ceiling ~0.54)
- Rare CTs (Astro, Macrophage, Oligo) collapse to CT mean (ratio 0.02-0.06)
- Supersedes April analyses ([[within-ct-nonzero]], [[zero-nonzero-analysis]]) which used the old two-stage 16K-gene model

**Why:** Prashant asked whether the missing variance is biological or technical before adding a variance loss term.
**How to apply:** Do not add a per-gene variance loss. First remove library size from the latent (explicit covariate or size-normalized input), re-run this diagnostic, check active latent dims (latent is 32, not 128). Report retention against Poisson-removed, library-size-removed target plus participation ratio.

## Latent-usage check (2026-09-18, job 10910710) — posterior collapse
- `notes/experiments/variance_zero_nonzero_fullmodel/latent_usage/README.md`, script `analysis/latent_usage_check.py`
- Only 5-6 of 32 latent dims active; 26 collapsed to prior (posterior var ~1, KL ~0). Dims 17, 4, 16, 13 = cell identity; **dim 19 = library size** (r = -0.94 within CT)
- Keeping ONLY dim 19 reproduces 95% of imputed within-CT variance; latents predict log library size with R2 0.96; per-CT effective latent dims 1.1-2.4; cells show the same collapse
- Consequence: a variance loss on the output cannot work; model is effectively "cell-type code + library size", which is why pseudobulk tests cannot separate it from CT-mean/linear regression
- Next retraining: (1) explicit library-size covariate or size-normalized input, (2) weaken/warm-up/free-bits on the JSD term (weights 0.1 with normloss collapse 26 dims), (3) 8-dim latent control; judge with active units + within-CT participation ratio, not CT-mean metrics
