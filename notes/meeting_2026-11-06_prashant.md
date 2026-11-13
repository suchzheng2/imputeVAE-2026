# Meeting with Prashant, 2026-11-06 (filed 2026-11-13)

Group meeting; the first half was another project (epsilon GC deconvolution, ROSMAP, kidney tumours) and is not
recorded here. Transcript summary of the imputeVAE part and the follow-up.

## What was discussed

1. **Library-size covariate and within-type variance.** I reported that with log library size as a covariate on
   encoder and decoder the within-type variance ratio falls from 0.07 to ~0.002, that library size moves from several
   latent dimensions to one, and that gene correlation rises from 0.90 to 0.96 with cell-type accuracy unchanged.
   Prashant's points: (a) the mean ratio over genes is meaningless if a zero-expression peak dominates it; show the
   per-gene distribution for one large type (L2/3 IT, L5 IT) with and without the covariate; (b) if library size is
   accounted for in the imputed values, think about whether the observed side needs the same treatment before the
   ratio ("numerator and denominator"); we agreed the covariate does not remove library size from the output (the
   output is still generated at a given library size), it only frees the latent from carrying it; (c) the more
   important question is whether the cost function improves with the covariate (reconstruction loss / RMSE with vs
   without); (d) within-type variance is only a limitation if biological variability is lost; if the reduction is
   technical noise it is fine. He does not yet consider it a good measure until technical and biological variance are
   separated. (e) Input and output are log1p of raw counts, raw-count generator is the goal: fine.
2. **Validation summary from the CBB RIP talk** (three categories: empirical vs predicted shift within our data;
   SynGO / TF gene sets; TRAP / RiboTag): "very positive", "this looks very nice".
3. **Generalisability is the remaining big item** before writing. Human data: the European nuclei + bulk request is
   under review (answer expected the week of 2026-11-09). Ask lab members / Nenad Sestan's group whether there is
   human single-cell (not nuclei) motor cortex 10x data (BrainSpan / PsychENCODE).
4. **SCORCH spatial**: only usable if single-cell resolution (imaging-based), not Visium spots.

## Action items and status (2026-11-13)

| item | status |
|---|---|
| Per-gene variance distribution, L5 IT, with vs without covariate | **Done 2026-11-06** (figures existed, now documented in `notes/libsize_covariate_review_2026-10-09.md` §6.8; reproducible with `analysis/benchmark/variance_distribution_ct.py`). Summary below. |
| Numerator / denominator (library size out of the real side too) | **Done 2026-11-13**, review §6.8 addendum: removing library size from real cells leaves 0.28 of 0.39 (depth is only 16 % of the real within-type variance); the fixed-library decode stays 100x below that. But decoding at the per-cell library and **Poisson-sampling the decoded counts** gives median variance 0.295 vs 0.391 real (75 %), rank Spearman 0.96: the reduction is technical (depth + sampling noise) and can be put back; the biological part remains compressed. |
| Reconstruction loss / cost function with vs without covariate | **Done 2026-11-13**, review §6.8: reconstruction loss identical (cell 0.255 vs 0.253, nuclei 0.221 vs 0.221, same normaliser); the gain is entirely in the cross-modality decode (gene corr 0.90 -> 0.97, R² 0.69 -> 0.93, gene MSE 0.030 -> 0.006) and in the alignment term (0.024 -> 0.007). Classification 0.963 vs 0.967. |
| SCORCH spatial resolution | **Answered from the scoping note** (`notes/experiments/yscorch_scoping/README.md`): the mouse SCORCH spatial set Mark meant is the Allen Institute EcoHIV + cocaine study (10x Multiome + **MERFISH**, NeMO portal) -- MERFISH is imaging-based, segmented single cells, so it qualifies; Yale's own SCORCH spatial is human (Visium DLPFC = spots, plus a Xenium DLPFC project), not mouse motor cortex. Independently, the public 10x Xenium Prime 5K mouse brain section already gives an in-situ validation of the compartment delta (Spearman -0.57 over 2,477 genes; `notes/experiments/spatial_merfish_application/README.md`). |
| Human single-cell motor cortex data | **Open** -- email draft below; to send. |
| Epsilon GC model-choice figure | other project, not tracked here |

## Per-gene variance distribution, L5 IT (answer to item 1)

`notes/experiments/libsize_covariate/armA/figures/fig_variance_distribution_L5IT.png` (+ `fig_example_gene_densities_L5IT.png`,
table `armA/tables/variance_distribution_L5IT_test.csv`), test partition, 10,133 genes, 5,945 real cells, 2,320 nuclei:

| source | median within-type variance | median per-gene ratio to real | Spearman vs real (variance rank) |
|---|---|---|---|
| real cells | 0.391 | 1 | 1 |
| raw nuclei | 0.209 | 0.70 | 0.72 |
| imputed, old model (no covariate) | 0.072 | 0.18 | 0.92 |
| imputed, covariate, fixed per-type library (ctmed) | 0.0022 | 0.007 | 0.66 |
| imputed, covariate, quantile-mapped per-cell library (label) | 0.081 | 0.21 | 0.92 |
| imputed, covariate, label-free predicted type | 0.081 | 0.21 | 0.92 |

Reading: the distribution is unimodal in every case (no zero-variance peak drives the summary), so the mean/median
tells the truth. The 0.002 number is the *fixed-library* decode, where every nucleus of a type is decoded at the same
library size, so what remains is the latent's biological variation only (and the rank correlation drops to 0.66). As
soon as each nucleus is decoded at its own (quantile-mapped) library size, the variance comes back to the old model's
level (0.08, ratio 0.21) with the same rank agreement (0.92). So: with the covariate the output variance is
*controllable* -- library-driven variance can be switched off or on -- and the old model's variance was indeed mostly
library pass-through. The gene examples (Snap25, Rpl3, Calm1, Vamp2) show the fixed-library decode as a spike at the
type mean and the quantile-map decode overlapping the real-cell density. This is Prashant's point (d): what is removed
in the fixed decode is the technical (depth) component; biological variability in the latent is what the 0.002 measures.

## Email draft: human single-cell motor cortex data

Subject: human single-cell (whole-cell) motor cortex 10x data?

Hi [name],

I am working with Prashant on a model that translates single-nucleus RNA-seq profiles into whole-cell profiles
(trained on the Allen mouse MOp paired cell / nucleus 10x data). We have validated it in mouse and applied it zero-shot
to the Allen human M1 snRNA-seq data, but we have no human *whole-cell* cortex data to check the predictions against.
Do you know of any human cortex (ideally motor cortex) 10x single-*cell* dataset, from BrainSpan / PsychENCODE or
elsewhere, including unpublished or sorted-population data? Even a small set of dissociated cells from the same region
as an snRNA-seq set would be enough for a paired test. Any pointer to who holds such data would be very helpful.

Thanks,
Suzy

## Next meeting

Week of 2026-11-16: bring the variance figure above (now with the library-regressed real cells and the Poisson-sampled decode), the loss table, the Xenium in-situ result, the SCORCH portal answer (no spatial deposited; cells / area 6 survey pending) and the human-data answer.
