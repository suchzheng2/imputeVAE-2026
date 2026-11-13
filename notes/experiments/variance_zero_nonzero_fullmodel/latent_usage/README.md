# Latent-usage check on the full-data cVAE (no retraining)

**Date:** 2026-09-18. **Script:** `analysis/latent_usage_check.py` (job 10910710, 8 min CPU), figure `analysis/latent_usage_figures.py` -> `L1_latent_usage.{png,pdf}`. Follow-up to `../README.md`.
**Why:** imputation decodes from mu only, so any within-cell-type variation in the output must come from within-cell-type variation of mu. If the latent is not used, no loss term on the output can create biological variation.
**Sanity check:** decoding the re-encoded mu with the cell flag reproduces the saved imputed matrix (max abs difference 1.4e-5).

## Result: posterior collapse. The model uses 5 to 6 of 32 latent dimensions.

| Dimension (nuclei) | Role | total var of mu | within-cell-type var | posterior var | KL (nats) | R2 with log library size (within type) |
|---|---|---|---|---|---|---|
| 17 | cell identity | 11.85 | 0.40 | 0.16 | 6.76 | 0.00 |
| 4 | cell identity | 3.46 | 0.32 | 0.87 | 1.87 | 0.00 |
| 16 | cell identity | 2.09 | 0.11 | 0.78 | 1.08 | 0.00 |
| 19 | **library size** | 1.45 | 0.88 | 0.06 | 2.96 | **0.88** (r = -0.94) |
| 13 | cell identity (weak) | 0.85 | 0.04 | 1.01 | 0.45 | 0.00 |
| 31 | marginal | 0.04 | 0.02 | 0.93 | 0.02 | 0.03 |
| other 26 | **collapsed to the prior** | < 0.01 | < 0.001 | 0.93 to 1.03 | < 0.005 each | n/a |

- Total information is 13.2 nats, of which 13.1 sit in five dimensions. The other 26 have posterior variance about 1 and KL about 0: the encoder outputs the prior regardless of input and the decoder ignores them.
- The 32 latents predict a nucleus's log library size within cell type with R2 = 0.96. Whole cells encoded with the cell flag show the same pattern (same active dimensions, dimension 19 R2 = 0.90), so both modalities share the collapsed space.
- **Ablation on the imputed output:** keeping only dimension 19 and freezing every other dimension at its cell-type mean reproduces **95%** of the imputed within-cell-type variance; dimensions 19 + 17 give 98%, five dimensions give 100%. Projecting the library-size direction out of the latent removes 85% of output variance.
- The identity dimensions 4, 16 and 13 have within-cell-type variance below their posterior noise (signal-to-noise 0.37, 0.14, 0.04), so during training the decoder could not rely on within-type differences along them. Only dimensions 19 and 17 have within-type signal above the noise (14.3 and 2.4).
- Per cell type the latent has 1.1 to 2.4 effective dimensions, and library size is 24 to 84% of within-type latent variance (exceptions: L5/6 NP, Lamp5 and Macrophage, where the dominant within-type axis is an identity dimension).

## What this means

1. The earlier findings (imputed variation about one-dimensional, 82% library size) are a direct consequence of the latent, not of the decoder or the loss on the output. This matches the April finding on the older model that one latent dimension encoded library size.
2. The model is effectively "cell-type code (4 dims) + library size (1 dim)". That is why it matches cell-type means so well in every pseudobulk test and why no pseudobulk test separates it from CT-mean or linear regression.
3. **A variance loss on the output cannot work in this state.** With no within-type information in the latent, the only way to raise output variance is to amplify the library-size dimension.
4. Notes that describe a 128-dim latent are wrong for this run: `latent_size` 5 means 32 dims, and 26 of them are unused.

## Recommended retraining experiments (not run)

Ordered by expected value; each should be judged with the variance diagnostic and this check (active units, within-type participation ratio, library-size share), not with cell-type-mean metrics, which are already saturated.
1. **Give library size to the model explicitly** (log library size as an extra input to encoder and decoder, or size-normalized input) so dimension 19 is free and output variance is no longer a library-size effect.
2. **Weaken or reshape the latent regularizer.** The divergence term (weights 0.1 with normalized losses) is strong enough to collapse 26 dimensions. Options: lower the weight by 10x to 100x, warm-up from zero, or free-bits style thresholding per dimension (no penalty below a small KL per dimension).
3. **Smaller latent as a control** (8 dims): if metrics are unchanged it confirms capacity is not the limit, regularization is.
4. Only after 1 and 2 show more active within-type dimensions: consider a structure-aware term (within-type covariance on denoised targets), not a per-gene variance term.
