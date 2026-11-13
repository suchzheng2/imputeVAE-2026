# Variance-rank check: does imputation keep the *ordering* of genes by variance?

**Date:** 2026-09-24. **Requested by:** Prashant (Sep 2026 meeting, `notes/meeting_2026-09-24_prashant.md`).
**Script:** `analysis/variance_rank_check.py` (CPU, < 1 min; reads the per-(cell type, gene) variance table from job 10910494, no model inference).
**Outputs:** `tables/variance_rank_by_cell_type.csv`, `tables/variance_rank_by_cell_type_and_detection_bin.csv`, `figures/V1..V4` (PNG, PDF, `_data.csv`).

## Question
The within-cell-type variance of imputed cells is lower than in real cells (median ratio 0.15). Prashant's concern: is this a
uniform flattening, where imputation assigns near-identical values to every cell so that *every* gene ends up with a similar
low variance ("cheating"), or is it a scale change that keeps the same genes high-variance (compatible with denoising)?
Test: per cell type, (i) the distribution of per-gene within-type variances (real cell, raw nucleus, imputed) and
(ii) the Spearman correlation between a gene's variance in real cells and its variance in imputed cells.

Model: full-data cVAE `run/run_2026-07-18-19-40-51_570984`; 40,166 v3 MOp nuclei imputed to cells, 71,117 real v3 cells as
the unpaired reference; log1p scale; genes detected in > 1% of real cells of that type (about 10,100 genes per neuronal type).

## Result: ranks are preserved; the distribution is stretched, not collapsed

| Cell type | genes | Spearman var(real cell) vs var(imputed) | same, well-detected genes (>10%) | vs real-cell variance with Poisson + library size removed | Spearman var(real cell) vs var(raw nucleus), reference | IQR ratio imputed / real (log10 var) | median var ratio imputed / real | top-200 overlap |
|---|---|---|---|---|---|---|---|---|
| **L5 IT** | 10,133 | **0.917** | 0.905 | 0.804 | 0.722 | 2.87 | 0.19 | 0.55 |
| **Oligo** | 9,755 | **0.929** | 0.906 | 0.786 | 0.819 | 1.62 | 0.03 | 0.56 |
| L2/3 IT | 10,132 | 0.811 | 0.792 | 0.633 | 0.699 | 3.61 | 0.11 | 0.16 |
| Pvalb | 10,119 | 0.911 | 0.894 | 0.508 | 0.783 | 2.46 | 0.14 | 0.27 |
| Vip | 10,104 | 0.964 | 0.953 | 0.872 | 0.847 | 2.46 | 0.05 | 0.24 |
| Astro | 9,765 | 0.931 | 0.904 | 0.859 | 0.880 | 1.86 | 0.01 | 0.73 |
| range over 17 types | | 0.70 (L6 IT) to 0.96 (Vip) | 0.65 to 0.95 | 0.51 to 0.87 | 0.60 to 0.88 | 1.5 to 3.6 | 0.003 to 0.20 | 0.09 to 0.73 |
| pooled (cell type, gene) pairs | 169,406 | 0.850 | 0.799 | 0.709 | 0.783 | 2.59 | 0.08 | 0.25 |

Full table: `tables/variance_rank_by_cell_type.csv`. Figures: `V1` (L5 IT and Oligo: density of log10 variance for the three
sources, hexbin of real vs imputed variance with Spearman and fitted slope, same against the biological-excess variance),
`V2` (Spearman per cell type, with real-vs-nucleus as the reference bar), `V3` (ECDF of log10 variance for all 17 types),
`V4` (Spearman by detection-rate bin).

1. **The same genes are the high-variance genes.** Spearman between real-cell and imputed per-gene variance is 0.92 in L5 IT
   and 0.93 in Oligo, above the real-cell vs raw-nucleus agreement (0.72, 0.82) in 14 of 17 types. It stays at 0.80 (L5 IT)
   and 0.79 (Oligo) when the real-cell target is the biological excess (Poisson sampling and library-size variance removed),
   so the agreement is not only through the technical part of the variance.
2. **No collapse to a single low value.** The imputed variance distribution is *wider* than the real one on the log scale
   (IQR ratio 1.5 to 3.6, fitted slope of log var(imputed) on log var(real) = 2.1 in L5 IT, 1.6 in Oligo). Imputation compresses
   low-variance genes towards zero far more than high-variance genes: in L5 IT the imputed/real variance ratio is 0.02 for genes
   detected in < 10% of cells and 0.35 for genes detected in > 70%. This is the opposite of the "same value for everything"
   failure mode, which would give a tight imputed distribution and a slope near 0.
3. **Caveats.**
   - Variance tracks mean on the log1p scale, so part of the rank agreement is inherited from mean agreement (gene correlation
     0.93). After regressing log variance on log mean within each source, the residual Spearman is 0.55 (L5 IT), 0.23 (Oligo),
     0.22 pooled: the mean-independent part of the ordering is kept only partly, and mostly in the large neuronal types.
   - Within detection bins the correlation is lower (L5 IT: 0.58 for < 10%, 0.79 for 10-30%, 0.88 for 30-70%, **0.47 for > 70%**).
     For the well-detected genes in L5 IT, which carry most of the biological signal, ordering is only moderately preserved.
   - Rank preservation says nothing about *per-cell* information. The earlier diagnostic showed the imputed within-type
     variation is about one-dimensional and 82% library size (`../README.md`, `../latent_usage/README.md`). Genes can be ordered
     correctly because their loading on that one axis tracks their real variance, without the model carrying gene-specific
     per-cell variation. Prashant's test rules out uniform flattening; it does not establish that the retained variance is biological.

## Answer for the meeting
Distribution: shape preserved and stretched, scale lower (L5 IT medians: real 0.39, nucleus 0.21, imputed 0.08). Spearman on
per-gene variance: 0.92 (L5 IT), 0.93 (Oligo), 0.85 pooled, all above the raw-nucleus reference. So the variance loss is a
scale change concentrated on low-variance genes, not a collapse. The remaining problem is the one already identified: the
retained variance has the right ordering but the wrong structure (one library-size axis, posterior collapse).

## Per-gene examples (2026-09-25): what the variance numbers look like per cell

`analysis/variance_rank_gene_examples.py` (job 11181551), figure `figures/V5_gene_examples_L5IT.png`, values in `_data.csv`.
Overlaid histograms of log1p values across L5 IT real cells (29,721), raw nuclei (11,599) and the imputation of those nuclei, for
eight genes chosen from the per-gene table to span the behaviours (y axis clipped so the zero spike does not dominate).

| gene | det cell / nuc | var cell / nuc / imp | what the histogram shows |
|---|---|---|---|
| Pcp4 | 89% / 9% | 2.10 / 0.07 / 1.74 | nucleus almost empty; imputed reproduces the full 1-5 spread of real cells |
| Rplp1 | 100% / 7% | 0.40 / 0.05 / 0.20 | ribosomal, absent in nuclei; imputed matches the cell distribution, slightly narrower |
| Rorb | 94% / 77% | 1.48 / 1.72 / 0.88 | imputed keeps the main mode but loses the real-cell upper tail (4-5.5); subtype variation compressed |
| Calm2 | 100% / 89% | 0.33 / 0.48 / 0.23 | housekeeping; same mode and shape, variance ratio 0.71; library-size part kept 86%, the rest 10% (Test A) |
| Malat1 | 100% / 100% | 0.72 / 0.24 / 0.34 | shifts down from nucleus toward cell, but misses the low tail of real cells |
| Xist | 53% / 100% | 3.26 / 0.17 / 0.09 | real cells are 47% zero (mixed sex), nuclei all express; imputed is a single mode at 2. Sex is not in the nucleus input, so this is a donor-composition mismatch, not a variance failure |
| Gm49164 | 73% / 20% | 0.75 / 0.17 / 0.03 | nucleus mostly zero; imputed collapses to one value near 1, no tail |
| Gfra1 | 10% / 4% | 0.12 / 0.04 / 0.01 | sparse; imputed collapses to a spike near 0.1 |

Reading (first draft, 2026-09-25, superseded by the mechanism test below): it looked as if variance is recovered when the gene
rides on a latent axis and collapses when the nucleus carries no per-cell information for it. That was an inference from the
histograms and detection rates, not a test. Xist should be excluded from variance comparisons or the cell reference restricted
to matching sex.

## Mechanism test (2026-09-25): where does the surviving variance come from, and is the missing variance really unavailable?

`analysis/variance_rank_gene_mechanism.py` (job 11181723), figure `figures/V6_gene_mechanism_L5IT.png`, tables
`tables/gene_mechanism_L5IT_{latent,partners,partner_lists}.csv`. Same eight L5 IT genes.

**Test A, latent loading.** Imputed value of each gene regressed on the five active latent dimensions of the nuclei
(`../latent_usage/mu_nuclei.npy`; 17, 4, 16 identity; 19 library size; 13 weak). A cross-validated random forest on the five
dims gives R2 0.98 to 0.99 for every gene, as expected for a deterministic decoder. Linear R2 on dim 19 alone: Pcp4 0.80,
Rplp1 0.95, Calm2 0.97, Malat1 0.65, Gm49164 0.91, Gfra1 0.74, Rorb 0.08, Xist 0.09. So for Pcp4 and Rplp1 the "recovered"
spread is mostly the library-size axis, not a sublayer program; Rorb is the only example gene whose imputed spread sits on the
identity dimensions (17, 4, 16 jointly 0.82).

**Test B, information available in the nucleus.** For each gene, the 30 genes most correlated with it in real L5 IT cells
(partners), a signed mean-z partner score, and the same score computed in nuclei from the partners detected in >20% of nuclei.

| gene | partners -> g, real cells, library size removed (ceiling) | partners detected in nuclei | sd(nuclear score)/sd(cell score) | nuclear partners -> imputed g, library size removed |
|---|---|---|---|---|
| Pcp4 | 0.82 | 29/30 | 0.60 | 0.43 |
| Rplp1 | 0.51 | 25/30 | 0.43 | 0.04 |
| Rorb | 0.83 | 28/30 | 0.72 | 0.57 |
| Calm2 | 0.56 | 28/30 | 0.47 | 0.13 |
| Malat1 | 0.51 | 30/30 | 0.80 | 0.24 |
| Xist | 0.72 | 21/30 | 0.54 | 0.13 |
| Gm49164 | 0.68 | 24/30 | 0.48 | 0.12 |
| Gfra1 | 0.18 | 22/30 | 0.61 | 0.02 |

Without the library-size removal the nuclear-partner-to-imputed correlations are 0.6 to 0.9 for most genes, but the partner
score itself correlates with log library size at 0.7 to 0.95 in nuclei, so those numbers restate Test A.

**Corrected reading.**
1. Whether a gene's variance "survives" imputation is mostly whether its real-cell variance is library-size-driven. The model
   reproduces that part through dim 19 (Pcp4, Rplp1, Calm2, Gm49164). Only Rorb shows non-library variance carried by the
   identity dimensions, and even there the model uses 0.57 of a 0.83 ceiling.
2. ~~The nucleus does carry non-library per-cell information about Gm49164 ... model limitation. Same for Rplp1 and Calm2.~~
   **Retracted by the cause test below (same day).** The partner ceiling was computed in real cells, where the module exists;
   in nuclei the Gm49164 partners do not co-vary (mean pairwise |r| 0.08 vs 0.48 in cells), so the nuclear partner score is
   library size plus independent noise and carries no information about the module. "Information available in the nucleus"
   must be judged by module coherence *in nuclei*, not by a ceiling measured in cells. By that criterion the model-limitation
   verdict holds only where the module is coherent in nuclei (Rorb 0.26, Pcp4 0.23) and not for Gm49164 (0.08), Actb (0.08),
   Calm2 (0.16, and its residual tracks a whole-cell axis) or Rplp1 (ribosomal partners are absent from nuclei).
3. Gfra1 is the one example where the information is genuinely weak (ceiling 0.18): there the collapse is expected.
4. Pcp4 is intermediate: the nucleus carries the information (0.82), the model uses part of it (0.43), and the rest of its
   recovered spread is library size.
So the earlier sentence should read: variance collapses when the model fails to encode the non-library covariation that the
nucleus contains, which is most genes; it survives mainly where the real-cell variance was library size to begin with.
Caveat: partners are chosen on the same real cells used for the ceiling (in-sample, 29,721 cells, 30 genes, negligible), and
the nuclear partner score is a linear proxy for what a nonlinear encoder could extract, so the ceilings are lower bounds.

## Cause of the two collapse examples (2026-09-25)

`analysis/variance_rank_gene_cause.py` (job 11182186) and `analysis/variance_rank_gene_cause2.py` (job 11182242), tables
`tables/gene_cause_L5IT.csv`, `tables/gene_cause2_L5IT.csv`. Sequencing library = `batch_id` prefix (12 cell libraries, 5
nucleus libraries in L5 IT). Gene identity from Ensembl REST (GRCm39).

**Gm49164** (lncRNA, chr14:73.97-74.19 Mb, 227 kb span; MGI:6118590). The locus holds two unnamed processed pseudogenes and
Cbx3-ps6; a 21-mer comparison finds no similarity between the unnamed pseudogenes and Actb, Actg1, Gapdh, Eef1a1, Tuba1a or
Dpysl2 (positive control: Cbx3-ps6 shares 96% of its 21-mers with Cbx3), so the Actb correlation is not multimapping.
Decomposition of its real-cell variance (0.75):
- 25% between sequencing libraries (Actb 26%, Gm15810 18%; Rorb 5%, Pcp4 6%, Gfra1 0.3%). Technical.
- 57% is the detection indicator (Bernoulli share), i.e. sampling of a 73%-detected gene.
- The structured remainder is a whole-cell module (Actb r 0.71, 0.61 within library; Dpysl2, Gnb1, Gsk3b, Actr2, St13, ...),
  coherent in cells (mean pairwise |r| 0.48) and absent in nuclei (0.08; Gm49164 vs Actb in nuclei r 0.015). The module's
  score tracks a cell-enriched-minus-nucleus-enriched gene axis at 0.67 (0.74 within library), and that axis is itself 0.83
  correlated with log library size in cells and 0.16 library-explained; a whole-cell "cytoplasmic content" axis with no
  nuclear counterpart.
None of the three components is recoverable from a nucleus, so the conditional-mean output is the correct answer for this
gene. One real defect remains: the imputed Gm49164 is 91% library-size axis (Test A) while real Gm49164 is 4% library size,
so the model assigns a library-size slope the gene does not have.

**Gfra1** (10% detected in cells, 4% in nuclei). 82% of its real-cell variance is the detection indicator; library explains
0.3%; its partners (Vgf, Scg2, Ptprn, Syp, Nell2, ...; an activity / secretory module, coherent at 0.52 in cells and 0.17 in
nuclei) predict it at only 0.25 (0.21 within library and QC). Its variance is sampling noise around a low mean with a faint
module attachment. A point estimator trained with MSE returns the conditional mean, so the collapse is by construction;
reproducing the spread would require sampling from a predictive distribution, which adds noise, not information.

**General rule from the four tested genes.** Whether imputed variance can survive is decided by whether the gene's real-cell
variance rides on something the nucleus also has. Library-size variance is reproduced (dim 19). Modules coherent in nuclei
are partly reproduced (Rorb 0.57 of 0.83, Pcp4 0.43 of 0.82) and there the shortfall is the model's. Whole-cell-only modules
(Gm49164 / Actb, ribosomal, Calm2 residual), between-library effects and dropout are not reproducible and collapse correctly.
The scalar to report per gene is the partner-module coherence in nuclei, not a ceiling measured in cells.

## Sex composition of the training data (2026-09-25)

`analysis/sex_composition_check.py` (job 11182923), table `tables/sex_composition_by_library.csv`. Raw counts of Xist and the
Y-linked genes Eif2s3y, Kdm5d, Uty, Ddx3y per sequencing library (v3, doublets removed).

- **Cells (c3A): 6 male + 6 female libraries, sex = library.** Male libraries 1, 2, 3, 8, 11, 12 (Xist detected in <1% of
  cells, Eif2s3y in 82-88%, Uty in 83-87%): 33,573 cells. Female libraries 4, 5, 6, 7, 9, 10 (Xist 99.7-100%, Eif2s3y 0%):
  37,610 cells. 47% male overall, and the split is similar within every cell type (Xist detection 40-60% per type).
- **Nuclei (n3A): all 5 libraries female** (Xist 99.7-99.8%, Eif2s3y 0%, Kdm5d 0.2-0.3%).
- Ddx3y is not a clean male marker in this pipeline: female cell libraries show it "detected" in 21-43% of cells at mean 0.2-0.6
  counts, presumably Ddx3x reads assigned to Ddx3y. Use Eif2s3y or Uty for sex calls.
- The Y-linked genes are absent from `training_gene_list_v3.csv` because the list is the intersection of genes detected in >5%
  of cells *and* of nuclei, and nuclei never express them.

Consequences:
1. Xist is the only strongly sex-bimodal gene left in the model's gene set (Tsix, Jpx, Ftx are expressed in both sexes;
   X-escapee genes Ddx3x, Eif2s3x, Kdm5c, Kdm6a keep 20-38% of their variance, in line with the L5 IT median 0.19).
2. Xist is the single highest-variance gene in real cells in all 17 cell types (percentile rank 0.996-1.0), so it sits at the
   top of every top-N overlap and at the right edge of every variance distribution. Its imputed value is one mode at 1.9,
   the mean of a 47/53 male/female mixture. This should be excluded from variance comparisons, or the cell reference should
   be restricted to the six female libraries (37,610 cells), which removes the problem at the source.
3. Every cell-vs-nucleus contrast in the project (GT delta, the 45-gene nuclear lncRNA list, compartment metrics) contains a
   sex-composition component for X- and Y-linked genes. Xist's GT delta of -1.3 in L5 IT is inflated by the male half of
   the cells, though its nuclear localisation is real and confirmed externally (H4).
4. Sequencing library and sex are fully confounded in cells, so the 25% between-library variance of Gm49164 / Actb cannot be
   separated into library and sex effects with this design.
5. For retraining: use female cell libraries only, or add sex as a covariate; the current model has no way to output
   male-like cells and represents every nucleus as a half-female mixture for X-linked genes.
