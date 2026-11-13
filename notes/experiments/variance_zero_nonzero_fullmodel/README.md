# Zero / nonzero variance diagnostic on the full-data cVAE

**Date:** 2026-09-18. **Motivation:** Prashant meeting: understand why real single-cell variance is higher than imputed variance before adding a variance term to the loss. Key question: is the variance we would be matching biological or technical?
**Scripts:** `analysis/variance_zero_nonzero_fullmodel.py` (job 10910494), `analysis/variance_zero_nonzero_addon.py` (job 10910608), `analysis/variance_zero_nonzero_figures.py`. **Outputs:** `tables/`, `figures/` (PNG, PDF, `_data.csv`).
This supersedes the April analyses (`notes/experiments/zero_nonzero_analysis/`, SUC-6), which used the old two-stage 16K-gene model and compared imputed values with raw whole cells rather than with the same nuclei.

## 1. Data

| Item | Detail |
|---|---|
| Model | single-stage cVAE `run/run_2026-07-18-19-40-51_570984` (scalar conditioning, encoder + decoder, log1p input, 32-dim latent), frozen |
| Nuclei | 40,166 v3 MOp nuclei, raw counts and log1p(counts), 10,134 genes |
| Imputed | `run/cvae_full_eval/imputed_cell_from_nucleus.npy`, row-aligned with the nuclei (label agreement 100%), so every comparison nucleus vs imputed is **paired per nucleus** |
| Real cells | 71,117 v3 MOp whole cells (unpaired reference) |
| Unit of analysis | 169,406 (cell type, gene) pairs over 17 cell types, grouped by the gene's detection rate in nuclei of that cell type: <10%, 10-30%, 30-70%, >70% |

## 2. Hypotheses

- **H1 (technical share).** Most within-cell-type variance in real whole cells is count sampling noise and library-size variation, so the total-variance ratio (0.15) understates how much biological variance the model retains.
- **H2 (zeros are dropouts).** Zero entries in nuclei are mostly technical. If the model has learned this, its imputed values at nucleus-zero entries should look like its values at nonzero entries, and zero/nonzero mixing should disappear from the imputed variance.
- **H3 (nonzero entries).** Among entries where the nucleus is nonzero, imputed variance should approach the nucleus variance in excess of its technical null, and imputed values should correlate with the nucleus's own value up to the ceiling set by sampling noise.
- **H4 (retained variance is biological).** Whatever variance the imputed data retains should be gene-specific biological variation: it should not reduce to library size, should span several dimensions, and should reproduce real-cell gene-gene co-variation.

## 3. Results: metrics, why they were chosen, and figures

### 3.1 Technical vs biological variance in real cells (H1): figure F2
**Metric.** For each pair, counts were re-simulated as Poisson(mean) (sampling noise only) and as Poisson(size factor x mean) (sampling noise plus library size), and the variance of log1p was taken. Observed minus simulated variance is the biological excess. **Why:** the loss is on log1p, so the decomposition must be on log1p; simulation avoids delta-method approximations for sparse counts. Cross-check: the library-size share from the simulation (0.25) matches a direct regression on log library size (R2 0.24).

| Share of real-cell within-cell-type variance | <10% | 10-30% | 30-70% | >70% | all |
|---|---|---|---|---|---|
| Poisson sampling noise | 0.75 | 0.64 | 0.44 | 0.17 | **0.52** |
| Library-size variation | 0.11 | 0.18 | 0.30 | 0.46 | **0.25** |
| Biological excess | 0.14 | 0.18 | 0.25 | 0.37 | **0.23** |

H1 supported: three quarters of the variance a naive loss term would chase is technical. Real-cell zeros are close to what sampling predicts (median detection 0.48 observed vs 0.57 expected, F5 right).

### 3.2 Zero entries (H2): figures F1, F5
**Metric.** Law of total variance with the partition defined by the *original nucleus* entry: variance = spread among nonzero entries + spread among zero entries + zero/nonzero mixing. **Why:** it separates the Bernoulli dropout structure from spread among expressed values, which a single variance ratio cannot.

- Zero/nonzero mixing is 80% of nucleus variance and 58% of real-cell variance, but only 12% of imputed variance.
- The model predicts almost the same value whether or not the nucleus detected the gene: median gap 0.04 log1p units, against a nucleus nonzero mean of 1.0. 46% of imputed variance sits among entries where the nucleus read zero.
- H2 supported: the model treats nucleus zeros as dropouts and fills them from the rest of the profile.

### 3.3 Nonzero entries (H3): figure F4
**Metrics.** Variance ratio imputed/nucleus on the same nonzero entries; variance ratio vs real-cell nonzero entries; paired Pearson r and slope between a nucleus's value and its imputed value within (cell type, gene). **Why paired r:** variance ratios cannot tell retained per-cell information from re-drawn spread; only a paired statistic can.

| | <10% | 10-30% | 30-70% | >70% | all |
|---|---|---|---|---|---|
| var ratio imputed / nucleus (nonzero entries) | 0.11 | 0.36 | 0.53 | 0.45 | 0.39 |
| var ratio imputed / real-cell nonzero | 0.08 | 0.19 | 0.27 | 0.31 | 0.21 |
| paired r, nonzero entries | 0.04 | 0.07 | 0.15 | 0.32 | 0.10 |
| noise ceiling for r (sqrt of non-technical share) | n/a | n/a | 0.57 | 0.54 | |

For well-detected genes the model carries r = 0.32 of a possible ~0.54. Slope is 0.06 overall: a nucleus's own count for a gene barely moves that gene's output. H3 partly supported: spread on nonzero entries is 40 to 50% of the nucleus spread, but little of it is gene-specific information from that nucleus.

### 3.4 Is the retained variance biological? (H4): figures F9, F7, F6, F3, F8
**Metrics.** (a) Variance-weighted R2 of each gene on the cell's log library size; (b) participation ratio (effective number of dimensions) of within-cell-type variation over the 300 most biologically variable genes; (c) Spearman between gene-gene correlation matrices (library-size axis regressed out) of nuclei or imputed vs real cells. **Why:** a variance magnitude can be reached by scaling one axis; these three ask whether the retained variance has the right *structure*.

- **82% of imputed within-cell-type variance is explained by the input nucleus's library size** (nuclei 6%, real cells 24%).
- **Imputed variation is about one-dimensional:** participation ratio 1.1 to 2.1 across nine cell types (real cells 8 to 18, raw nuclei 18 to 56); the first component holds 64 to 98% of imputed variance.
- Gene-gene correlations are inflated (mean |r| 0.41 to 0.82 vs 0.11 to 0.23 in real cells) and match real-cell structure inconsistently: good in L5 IT (0.85) and Lamp5 (0.87), poor in L2/3 IT (0.20), Vip (0.20), Pvalb (0.09), L5/6 NP (0.09). Raw nuclei match better in seven of nine types (0.69 to 0.92).
- Like-for-like retention (F9 right): imputed/total 0.15; imputed/(total minus Poisson) 0.32; **with the library-size axis removed from both sides 0.12**. Comparing imputed total with the library-size-free biological excess gives 0.67, but that comparison is not like-for-like and should not be quoted.
- What the model does retain can be real: in L5 IT, Pcp4 is detected in 9% of nuclei yet its imputed values are bimodal like real cells (F6), so subpopulation structure is recovered from other genes. Across pairs, imputed variance tracks the real biological excess (Spearman 0.71).
- Per cell type (F8): rare types collapse to the cell-type mean (imputed/total: Astro 0.02, Macrophage 0.02, Oligo 0.06); L5 IT retains most (0.28). Endo (23 nuclei) is unreliable.

H4 not supported in general: the retained variance is dominated by one library-size-like axis.

## 4. Conclusions

1. **Do not add a variance-matching loss now.** 77% of the real-cell variance it would target is technical (52% sampling noise, 25% library size). The meeting's caution was right.
2. **The honest retention number is about 0.12, and the problem is structure, not magnitude.** The model keeps one axis of within-cell-type variation and that axis is library size. A per-gene variance penalty would most cheaply be satisfied by stretching that same axis.
3. **The model already handles zeros sensibly.** It ignores a nucleus's own zero for a gene and imputes from context, which is the correct behaviour for dropout-dominated data. No zero-inflation head is needed for that purpose.
4. **Metric change.** Report variance retention against the Poisson-removed target with the library-size axis removed from both sides, plus participation ratio and gene-gene structure agreement. Retire the total-variance ratio as a headline.

## 5. Recommended next steps (not run)

- **Take library size out of the latent.** Give log library size to encoder and decoder as an explicit covariate (as scVI does) or size-normalize the input, then re-run this diagnostic. Earlier CPM runs failed for a JSD scale reason that the normalized loss has since fixed, so this is worth retrying.
- **Latent usage check (cheap, no retraining).** Count active latent dimensions within cell type (variance of mu per dimension); imputation decodes from mu only, so within-type variation exists only where mu varies. The run has a 32-dim latent (notes elsewhere say 128).
- **If a loss term is added later,** target structure: match within-cell-type covariance or low-dimensional projections on denoised targets, not per-gene variance.
- **Rare types** need a separate remedy (sampling or weighting); they are at the cell-type mean.

## Follow-up: latent-usage check (done 2026-09-18)
See `latent_usage/README.md`. Posterior collapse: 5 to 6 of 32 latent dimensions are active, 26 sit at the prior, and one dimension (19, library size) alone reproduces 95% of the imputed within-cell-type variance. This explains sections 3.4 and 4 mechanistically.

## Caveats
Poisson is the minimal technical model; any extra technical overdispersion is counted as biological, so 0.23 is an upper bound on the biological share. Library size within a cell type is partly biological (cell size), so removing it is conservative for both sides equally. The noise ceiling for paired r is unreliable for sparse genes and is reported only for genes detected in more than 30% of nuclei. Real cells and nuclei are unpaired.

## Follow-up: variance-rank check (done 2026-09-24)
See `variance_rank/README.md` (Prashant's request: distributions and Spearman of per-gene variance). Spearman var(real cell) vs var(imputed) = 0.92 (L5 IT), 0.93 (Oligo), 0.85 pooled, above the real-vs-nucleus reference in 14/17 types; the imputed distribution is stretched (IQR ratio 1.5 to 3.6, low-variance genes compressed most), not collapsed. Ranks are kept; structure (one library-size axis) remains the problem.
