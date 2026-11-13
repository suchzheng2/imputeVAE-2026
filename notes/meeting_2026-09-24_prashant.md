# Meeting Notes — Sep 2026 (with Prashant)

*Transcript summarized 2026-09-24; exact meeting date not recorded in transcript (week of 2026-09-21). Short meeting, Prashant had a hard stop.*

## Presented

1. **TRAP/RiboTag external validation** (`validation/trap_ribotag_external_validation/`)
   - Table of mouse datasets: two compartment-fractionation sets (cytosol / nucleus / whole cell) plus RiboTag / bacTRAP sets across cell types, control samples only, 2008 dataset dropped.
   - Spearman correlation of each external population's averaged profile against our raw nuclei, imputed, and real whole-cell pseudobulks, grouped to match their populations (e.g. pooled excitatory).
   - Result: imputed ≈ real whole cell, both above raw nuclei; correlations around 0.75.
2. **Zero / nonzero variance split** (`notes/experiments/variance_zero_nonzero_fullmodel/`)
   - Per cell type: median per-gene within-CT variance for nucleus nonzero entries, imputed values at those entries, real-cell nonzero entries, and imputed values at nucleus-zero entries.
   - Imputed values at zero entries carry similar spread to imputed values at nonzero entries. Neuronal types behave alike; non-neuronal types differ more.

## Prashant's feedback

### TRAP validation: promising, but fix the cell-type mapping
- "Already very promising", especially the motor-cortex dataset **GSE172484** (L5b, M1). Levels of 0.75 across very different datasets are good.
- **Do not map by name alone.** IT and ET have different functions; labels like L5a / L5b do not automatically equal L5 IT / L5 ET.
- **Drop L4.** Motor cortex has no granular layer, so `Scnn1a_L4 -> L5 IT` (GSE133291) is wrong. Skip it instead of matching to the nearest type.
- Preferred fix, in order of effort:
  1. Read each source paper for projection type: **extratelencephalic (ET)** projects to subcortex / brainstem, **intratelencephalic (IT)** stays within cortex and telencephalon. Map L5a / L5b by that evidence, not by layer name. (Use an LLM to dig through the papers.)
  2. Or compare each external L5 population against **every** L5 type in our data individually and report all, rather than committing to one match.
  3. Ideal but heavy: marker-gene-based matching.
- Be careful with region: cortex-to-cortex is fine, hippocampus is not comparable. Check dataset quality and age.
- Avoid circularity: whatever matching is used must not be chosen by the metric being reported.

### Variance: check the distribution and rank of per-gene variances, not the median
- A downward shift in variance is partly expected from denoising. The concern is that imputation "cheats" by assigning near-identical values to many cells, which would make **every** gene's variance uniformly low.
- What he wants to see, for well-represented types (**L5 IT** and **Oligo** at least):
  1. The **distribution** of per-gene within-CT variances (real cell vs imputed vs nucleus), not the median. Similar shape at a lower scale is acceptable; a tight collapsed distribution is a red flag.
  2. **Spearman correlation between per-gene variances in real cells and in imputed cells** within a cell type. The same genes should be high-variance in both even if the scale differs.
- Suchen recalled a similar check on an older model; it needs redoing on the current full model.

## Prashant's suggestions, verbatim (follow-up message)

Cell-type mapping:
- Maybe look at marker genes
- Look at each L5 individually
- Check original paper of external dataset about extra/intra-telencephalic neurons

Variance:
- Look at distribution of values for L5 IT
- Spearman on gene variance (real cell vs imputed)

## Status (2026-09-24)
- Item 1 (TRAP mapping audit): **done** — `validation/trap_ribotag_external_validation/outputs/extension/metadata/l5_mapping_evidence.md`, script 17; L4 excluded, Rbp4 pooled then excluded on 2026-09-25 (skip rule: no single matching subclass), others confirmed; H2/H3 rerun job 11168301.
- Item 2 (variance rank check): **done** — `notes/experiments/variance_zero_nonzero_fullmodel/variance_rank/README.md`; Spearman 0.92 (L5 IT), 0.93 (Oligo); distribution stretched, not collapsed.
- Item 1b (2026-09-25, Suchen: real-cell-derived markers are circular): marker check redone with an external list (Yao 2021 MOp Supp. Table 6), ET-vs-IT direction matches the literature for all six populations; `l5_mapping_evidence.md` section "External marker check".
- Item 3 (variance figure): **done** — `variance_rank/figures/V1_focus_distribution_and_rank.png`.

## Action items
1. **TRAP mapping audit**: remove the L4 → L5 IT comparison; for L5a / L5b populations (GSE172484 Gprin3 / Colgalt2, GSE202172 S100a10, GSE141337 Glt25d2, secondary etv1 / glt25d2), document projection-type evidence from the papers and either justify the ET / IT assignment or report against all L5 types. Re-run the affected H2 / H3 numbers.
2. **Variance rank check on the full model**: per cell type, per-gene variance distributions (histogram / ECDF) for real cell, imputed, nucleus, plus Spearman(var_real, var_imputed). Start with L5 IT and Oligo, then all 17. Add to `notes/experiments/variance_zero_nonzero_fullmodel/`.
3. Turn the raw variance table into a proper figure.
4. Next meeting: **Friday next week** (Wednesday not possible), likely 2026-10-02.

## Related
- `notes/experiments/trap_ribotag_external/README.md`
- `notes/experiments/variance_zero_nonzero_fullmodel/README.md` (already notes imputed variance tracks real biological excess at Spearman 0.71 across (CT, gene) pairs; the new request is the per-CT version with distributions)
