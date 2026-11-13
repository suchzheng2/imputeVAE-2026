# Softmax Bottleneck Analysis (2026-03-31)

## Problem Discovery

Run `run_2026-03-26-23-28-20_500560` (conditioned decoder, delayed alignment, 300 epochs, stage1 cell init) showed a large gap between sample-level and gene-level reconstruction:

| Metric | Value |
|--------|-------|
| Sample-level corr (decoded nuclei vs raw nuclei) | 0.992 |
| Gene-level median corr (decoded nuclei vs raw nuclei) | 0.545 |
| Gene-level median corr (decoded nuclei vs raw cell) | 0.532 |

- 47% of genes have per-gene correlation < 0.5
- 12% of genes have negative correlation

### Gene correlation is strongly expression-dependent

| Nuclei expression level | Gene count | Median corr vs raw nuclei |
|------------------------|-----------|--------------------------|
| 100+ | 1 | 0.971 |
| 10-100 | 43 | 0.929 |
| 1-10 | 893 | 0.882 |
| 0.1-1 | 7,031 | 0.738 |
| 0.01-0.1 | 6,103 | 0.631 |
| 0.001-0.01 | 4,676 | 0.447 |
| <0.001 | 4,774 | 0.201 |

Stage 1 cell-only model (epoch 599) already has the same pattern:
- Median gene corr vs raw cell: 0.796
- 30% of genes < 0.5
- But proportions stay within 2x (no gene has decoded/raw ratio > 5 or < 0.2)

---

## Root Cause: Softmax Proportion Distortion

The NB decoder uses `mu = library_size * softmax(logits)` -- all ~25K genes compete in a single softmax. This is a zero-sum game.

### Evidence

**Malat1 eats the softmax:**
- Raw cell proportion: 2.43%
- Stage 1 decoded proportion: 1.97% (fine)
- Stage 3 decoded proportion: **11.19%** (hogging 8.77% of mass from other genes)

**Mass redistribution:**
- 247 genes with mean > 1 are suppressed to <10% of their true decoded/raw ratio
- 50 of those are ribosomal proteins (Rpl/Rps)
- These 247 genes hold 7.23% of raw expression but only 0.43% of decoded -> 6.81% missing
- 5,232 genes are over-represented (>2x), with total excess of 13.56%

**Stage 1 -> Stage 3 degradation:**
- 6,652 genes dropped >0.3 in per-gene correlation
- Worst drops: Rplp1 (0.925->-0.578), Rps7 (0.935->-0.548), Eef1a1 (0.972->-0.458)
- Stage 1 total decoded mean: 36,107; Stage 3: 12,565 (very different scales)

**Root mechanism:** When a few dominant genes get slightly higher logits, they steal proportion from thousands of smaller genes. With 25K genes, even small logit shifts create huge distortions. The NB loss weights highly expressed genes more, so the model optimizes their proportions first.

---

## Raw Data Properties

| Property | Cell | Nuclei |
|----------|------|--------|
| Cells | 193,824 | 116,971 |
| Genes | 26,852 | 25,871 |
| Gene mean dynamic range | 53M x | ~39M x |

### Expression concentration (cell data)

| Genes | Cumulative expression |
|-------|----------------------|
| Top 1 (Malat1) | 2.43% |
| Top 10 | 6.39% |
| Top 100 | 19.63% |
| Top 1000 | 52.61% |
| Bottom 20,000 | 13.60% |

### Gene count by expression level

| Expression level | Cell genes | Nuclei genes |
|-----------------|-----------|-------------|
| mean > 100 | 2 | ~1 |
| mean 10-100 | ~220 | ~43 |
| mean 1-10 | ~3,600 | ~900 |
| mean 0.1-1 | ~6,900 | ~7,000 |
| mean < 0.1 | ~16,000 | ~18,000 |

---

## scVI Literature Survey

### This is a known limitation

**GitHub issue YosefLab/scVI #297** (Valentine Svensson, 2019):
- URL: https://github.com/YosefLab/scVI/issues/297
- Showed that highly expressed genes dominate reconstruction loss
- "Most of the representation in the latent space will be used to model not-so-interesting housekeeping genes"
- Closed without resolution

**GitHub issue scverse/scvi-tools #729** (Saket Choudhary, 2020):
- URL: https://github.com/scverse/scvi-tools/issues/729
- Confirmed: "the negative binomial likelihood behaves differently for lowly expressed genes and highly expressed genes"
- Root cause: `log mu_NB = log l + log_softmax(nn(z))` -- library size amplifies expression-dependent bias
- Romain Lopez suggested `log mu = log l + log g + nn(z)` (gene-specific offset) but not implemented

**Svensson 2026 preprint** (bioRxiv):
- URL: https://www.biorxiv.org/content/10.64898/2026.02.11.705441v1
- Defines per-gene Pearson correlation metrics r(all) and r(nz), stratified by UMI depth
- Specific r values are in Table S2 (not fully extracted -- check supplementary directly)
- Introduces cross-correlation contrastive loss to improve low-UMI reconstruction

**scverse Discourse** (2025):
- URL: https://discourse.scverse.org/t/issue-with-scvi-vae-reconstruction-high-sensitivity-to-library-size-and-poor-rank-preservation/3883
- User reported Spearman r ~ 0.38 for rank preservation across all loss functions
- Expert confirmed: "Yes, by design" -- reconstruction quality correlates with library size

### Alternative decoder parameterizations from literature

| Approach | Parameterization | Sums to 1? | Source |
|----------|-----------------|------------|--------|
| scVI default | `mu = l * softmax(nn(z))` | Yes | scVI |
| scVI softplus | `mu = l * softplus(nn(z))` | No | Built-in option scale_activation=softplus |
| DCA | `mu = sf * clip(exp(nn(z)))` | No | Eraslan et al. 2019, Nature Comms |
| Gene offset | `mu = l * g * nn(z)` | No | Proposed in issue #729 |
| SCAR | Uses softplus_sp default | No | scvi-tools external model |
| Log-space MSE | Train on log(x+1) | No | Simpler autoencoders |

---

## Metrics Discussion

### Why per-gene Pearson correlation is insufficient

1. **Insensitive to scale** -- a gene with r=0.99 but 10x overestimated still scores great
2. **Biased by cell-type structure** -- if a gene is ON in 1 type and OFF in others, correlation is high just from binary pattern
3. **Doesn't capture downstream utility** -- can we correctly identify DEGs, marker genes?

### Better metrics (added to inline eval 2026-03-31)

| Metric | What it captures | Added to |
|--------|-----------------|----------|
| Log-space MAE | MAE(log(dec+1), log(raw+1)) -- scale errors in log space | stage3_eval.py, cond_training.py |
| Per-gene R-squared | Coefficient of determination -- penalizes both correlation and scale | stage3_eval.py, cond_training.py |
| Wasserstein distance | Distribution overlap per gene across cell types | stage3_eval.py, cond_training.py |
| Downstream task perf | DEG identification, marker genes, cell typing | TODO |

---

## Filtering Analysis (2026-03-31, updated)

Current filtering: genes in >=3 cells, cells with >=200 genes. Plots in `analysis/filtering_cutoffs/`.

### Current filtering is far too permissive

The original thresholds (min_genes=200, min_cells=3) are below the minimum values in the data -- they filter nothing:
- **Cell data:** all 193,824 cells have >1000 expressed genes (min is ~1300). Library sizes start at ~3800.
- **Nuclei data:** all 116,971 nuclei have >200 expressed genes, 99.3% have >1000. Library sizes start at ~1000.

These "QC filters" give the false impression that QC was performed when in reality 100% of data passes.

### Data distribution summary

**Cell data** (193,824 cells, 26,852 genes):

| Metric | 1st %ile | 5th %ile | 10th %ile | 25th %ile | Median |
|--------|---------|---------|----------|----------|--------|
| Library size | 3,866 | 5,845 | 7,410 | 10,973 | 17,103 |
| Genes expressed | 2,134 | 2,866 | 3,341 | 4,221 | 5,287 |

**Nuclei data** (116,971 nuclei, 25,871 genes):

| Metric | 1st %ile | 5th %ile | 10th %ile | 25th %ile | Median |
|--------|---------|---------|----------|----------|--------|
| Library size | 1,515 | 1,859 | 2,132 | 2,801 | 4,150 |
| Genes expressed | 1,018 | 1,170 | 1,300 | 1,579 | 2,048 |

Key observations:
- Nuclei have ~3x lower library sizes than cells (median 4,150 vs 17,103)
- Nuclei express ~2.5x fewer genes per cell (median 2,048 vs 5,287)
- Both distributions are bimodal (visible in histograms), likely reflecting different cell types or sequencing batches

### Gene filtering — the main lever

Gene filtering has much more impact than cell filtering. Current ~25K genes include a massive noise tail:

| Cutoff | Cell genes | Nuclei genes | Softmax reduction | Notes |
|--------|-----------|-------------|-------------------|-------|
| Current (>=3 cells) | 26,852 | 25,871 | 1x (baseline) | No meaningful filtering |
| det>1%, mean>0.01 | 16,228 | 14,195 | ~1.8x | Conservative — removes noise genes |
| det>5%, mean>0.01 | 12,659 | 9,938 | ~2.3x | Moderate — good balance |
| det>5%, mean>0.1 | 11,779 | 8,743 | ~2.7x | Aggressive — starts losing real genes in nuclei |
| det>10%, mean>0.1 | 11,084 | 7,238 | ~3.2x | Very aggressive — nuclei loses 72% of genes |

### Cell filtering — limited impact

Cell filtering barely matters for cells (data is already well-QC'd) but can significantly affect nuclei:

| Cutoff | Cell cells | Nuclei cells | Notes |
|--------|-----------|-------------|-------|
| lib>1000, genes>1000 | 193,824 (100%) | 116,194 (99.3%) | Minimal — essentially no filtering |
| lib>2000, genes>1500 | 193,174 (99.7%) | 93,171 (79.7%) | Removes low-quality nuclei tail |
| lib>5000, genes>2000 | 188,113 (97.1%) | 48,279 (41.3%) | Too aggressive for nuclei — loses 59% |

### Recommended cutoff combinations

| Config | Gene cutoff | Cell cutoff | Cell genes | Cell cells | Nuc genes | Nuc cells | Verdict |
|--------|-----------|-----------|-----------|-----------|----------|----------|---------|
| **A (recommended)** | det>1%, mean>0.01 | lib>1000, genes>1000 | 16,228 | 193,824 | 14,195 | 116,194 | Conservative; reduces softmax ~1.8x; retains all cells |
| B | det>5%, mean>0.01 | lib>1000, genes>1000 | 12,659 | 193,824 | 9,938 | 116,194 | Moderate gene filter; ~2.3x softmax reduction |
| C | det>5%, mean>0.1 | lib>2000, genes>1500 | 11,779 | 193,174 | 8,743 | 93,171 | Aggressive; loses 20% of nuclei |
| D | det>10%, mean>0.1 | lib>2000, genes>1500 | 11,084 | 193,174 | 7,238 | 93,171 | Very aggressive; nuclei lose 72% genes |
| E | det>10%, mean>0.1 | lib>5000, genes>2000 | 11,084 | 188,113 | 7,238 | 48,279 | Extreme; loses 59% of nuclei — not recommended |

### Conclusions and recommendations

1. **Gene filtering is the main knob.** Going from 25K to ~14K genes (config A) removes the noise tail that wastes softmax capacity. The removed genes (det<1%, mean<0.01) have near-zero expression and contribute to the softmax bottleneck without carrying biological signal.

2. **Cell filtering should be light.** Config A (lib>1000, genes>1000) is sufficient — it retains 99%+ of both datasets. Aggressive nuclei cell filtering (e.g., lib>5000) removes too many nuclei and biases toward high-UMI cell types.

3. **The nuclei data is the constraint.** Nuclei have fewer genes expressed, lower library sizes, and fewer genes passing detection thresholds. Any filtering must be validated against nuclei retention. Config A keeps 14,195 nuclei genes; going to det>5% drops to 9,938 — a 30% loss.

4. **Gene filtering alone won't solve the softmax bottleneck.** Even at 14K genes, the top 10 genes still hold ~19% of expression mass (nuclei) or ~7% (cells). The fundamental zero-sum issue remains. Gene filtering is complementary to decoder architecture changes (softplus, gene-specific bias), not a replacement.

5. **Recommended first step:** Regenerate data with config A (det>1%, mean>0.01, lib>1000, genes>1000), run a quick probe to measure improvement vs current 25K-gene runs. If insufficient, try config B. In parallel, prototype softplus decoder as the structural fix.

---

## Practical Options (ranked by effort)

1. **Replace softmax -> softplus in NB decoder** -- small code change in models.py: `mu_nb = library_size * F.softplus(rho)` instead of `F.softmax(rho, dim=1)`. Removes zero-sum competition. Need to handle that softplus doesn't normalize -- library size scaling needs rethinking.

2. **Stricter gene filtering** -- regenerate data.h5ad with det>1%, mean>0.01. Reduces from ~25K to ~14K genes. Less softmax competition. Visualization in `analysis/filtering_cutoffs/`.

3. **Try MSE on log1p data** -- code exists as VAE_JSD class in models.py. Use `--model "vae_js" --use_raw_data True --raw_data_normalization "log1p"`. No softmax at all. No existing runs to compare.

4. **Add gene-specific bias terms** -- add learnable per-gene log-offset to decoder so network only learns deviations from baseline expression.

5. **Gene-weighted loss** -- divide per-gene NB log-likelihood by log(1 + mean_g) to reduce dominance of HE genes.

---

## Code References

- NB decoder softmax: `imputeVAE/models.py`, VAE_NB class, `mu_nb = library_size * F.softmax(rho, dim=1)`
- MSE model: `imputeVAE/models.py`, base VAE class (line 117: `rloss = nn.MSELoss()`)
- Model selection: `train.py` lines 48-68
- Data loading: `imputeVAE/data.py`, `load_raw_data()` function
- Raw data path: `/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi/{dataset}/data.h5ad`
- Gene filtering in original pipeline: `../Cytoplasm_imputation/scvi/run_scvi.py` (min_genes=200, min_cells=3)
- Inline eval metrics: `imputeVAE/stage3_eval.py` -- `_corr`, `_log_mae`, `_r_squared`, `_wasserstein_1d`
- Cross-decode eval: `imputeVAE/cond_training.py` -- `_run_cross_decode_eval`
- Filtering visualizations: `analysis/visualize_filtering_cutoffs.py`
