# Gerstein Perspective Review (2026-04-02)

Review of imputeVAE project through Mark Gerstein's research lens. Actionable items marked with **[TODO]**.

---

## What's Strong

- Practical infrastructure problem: translating snRNA-seq → cell-like profiles enables reuse of massive nuclei datasets (e.g., PsychENCODE 2.8M nuclei)
- Conditioned decoder + cross-recons architecture is clean and well-motivated
- 0.994 median gene-level Pearson r on cross-decode path is impressive
- MSE > NB finding is interesting and publishable in itself

---

## Key Questions & Action Items

### 1. Biological application — what does this enable?
- **[TODO]** Show one concrete downstream use case: recover cytoplasmic RNA (Fos, Arc, mito genes), improve cell-type eQTL mapping, or change a DE result
- A translator is a tool, not a finding — needs a biological payoff

### 2. Per-cell-type breakdown
- **[TODO]** Generate per-cell-type cross-decode quality table/figure (scatter or heatmap)
- Expect neurons (more cytoplasmic RNA) vs glia to behave differently
- Cell types where imputation fails reveal biology of nuclear vs cytoplasmic RNA — a finding in itself

### 3. Gene-level analysis — which genes improve?
- **[TODO]** Rank genes by cross-decode R². Identify worst-imputed genes — are they the ones with largest cell-vs-nucleus gap?
- **[CRITICAL]** If the hardest genes to impute are the ones that differ most, the tool only works where you don't need it. Must address this.
- **[TODO]** Stratify eval: top 1K most-different genes, bottom 1K housekeeping genes, show model adds value on genes that matter

### 4. Why MSE beats NB
- **[TODO]** Investigate softmax bottleneck distortion, library-size scaling
- **[TODO]** Compare gene-level R² distributions between MSE and NB — which genes does NB get wrong?
- Publishable methods insight if explained well

### 5. Cross-species transfer (mouse → human)
- **[TODO]** Plan for human validation — PsychENCODE snRNA-seq is nuclei-only, need paired cell+nuclei human data (rare)
- Options: retrain on human, or use mouse decoder as prior + fine-tune
- Path to high impact: if it works on human PsychENCODE data → consortium co-authorship

### 6. Gene filtering bias
- **[TODO]** When no-filter run (job 3115964) completes, compare: (a) Config B genes only, (b) excluded genes only, (c) all genes
- Filtering to well-detected genes removes exactly the genes most different between cell/nucleus — biases eval upward

### 7. Statistical rigor of correlation metric
- **[TODO]** Compute null distribution: permute cell-type labels, recompute gene-level r
- **[TODO]** Bootstrap 95% CI on per-gene correlations over ~20 cell types
- 20 cell types is low n for Pearson r — report confidence intervals

### 8. Benchmarking against existing methods
- **[TODO]** Compare to: scVI (cell/nucleus as batches), per-gene linear regression baseline
- **[CRITICAL]** Without benchmarks, reviewer 1 kills the paper
- totalVI if protein data available (probably not)

---

## Evaluation Concerns to Address

### Training distribution leakage
- Cross-decode eval uses same cell data as cross-recons training target
- **[TODO]** Report validation set metrics prominently (already have them — they track closely)
- Ideal: external validation on different mouse strain/brain region/batch

### Trivial vs biological signal
- **[CRITICAL]** Most genes have similar cell/nucleus expression. 0.994 corr may be dominated by ~8K unchanged genes
- **[TODO]** Stratify: show correlation on just the genes that actually differ between modalities

---

## Publication Scoping

| Target | Requirements |
|--------|-------------|
| Bioinformatics / PLOS Comp Bio | Method + benchmark on mouse AIBS + comparison to existing methods |
| Genome Biology / Genome Research | Above + human validation + downstream application (DE or eQTL improvement) |
| Nature Methods | Above + cross-tissue/species generalization + released tool/resource |

---

## Encouraged Directions

1. **PsychENCODE connection** — show it works on human consortium data
2. **Per-cell-type quality as biology** — where imputation fails = biology of nuclear vs cytoplasmic RNA
3. **Tool release** — CLI: input h5ad snRNA-seq → output imputed cell-like matrix
4. **Marker gene validation** — 10 markers per cell type, show cross-decoded matches cell pattern (the figure a biologist understands)
