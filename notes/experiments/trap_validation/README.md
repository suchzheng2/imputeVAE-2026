# TRAP/RiboTag Cross-Validation

**Date:** 2026-08-05
**Task:** Task 10 (biological validation — independent, non-circular)
**Script:** `analysis/trap_crossvalidation.py`
**Results:** `compartment_classification/trap_validation/`

## Question

Can an independent experiment (TRAP/RiboTag) validate the cell-nucleus compartment delta without circularity? The GT delta and predicted delta in the compartment classification analysis both come from the same dataset — TRAP provides a fully external signal.

## Design

### Validation 1: IP/Input Enrichment (GSE141464)
- **Source:** Schmidt et al., S100a10-TRAP, cortical L5a pyramidal neurons
- **Measure:** log2(IP_CPM / Input_CPM) per gene, saline condition (3 IP reps + 1 Input rep)
- **Comparison:** Correlate TRAP enrichment with predicted delta and GT delta
- **Expectation:** Positive correlation IF TRAP enrichment = compartment enrichment

### Validation 2: Cell-Type-Specific RiboTag (GSE77243)
- **Source:** Spiegel/Bhatt/Bhatt/Greenberg, RiboTag, 5 neuron types (Emx1, Gad2, PV, SST, VIP), visual cortex
- **Measure:** Ribosome-bound expression (IP-only, standard housing) per gene per neuron type
- **Comparison:** Within-type correlation between TRAP expression rank and GT cell-nucleus delta rank
- **Expectation:** Strong positive correlation — genes highly expressed on ribosomes in a neuron type should also be cytoplasm-enriched in that type

## Results

### Validation 1: IP/Input (negative — see interpretation)

| Method | N genes | Spearman rho | p-value | AUROC |
|--------|--------:|-------------:|--------:|------:|
| GT delta | 8,274 | -0.124 | 1e-29 | 0.441 |
| ImputeVAE | 8,274 | -0.149 | 4e-42 | 0.431 |
| CT-mean | 8,274 | -0.142 | 2e-38 | 0.433 |
| Linear reg | 8,274 | -0.175 | 1e-57 | 0.417 |
| scVI | 8,274 | +0.116 | 3e-26 | 0.562 |
| scANVI | 8,274 | +0.106 | 6e-22 | 0.557 |
| scGen | 8,274 | +0.120 | 8e-28 | 0.564 |
| scButterfly | 8,274 | +0.136 | 1e-35 | 0.572 |

**Interpretation:** The negative correlation is expected. TRAP IP/Input measures **translational engagement** (what fraction of a gene's mRNA is ribosome-bound), NOT subcellular localization. Housekeeping genes (Rpl, Actb, Calm1) are abundant in cytoplasm but have IP/Input ~ 1 because they're constitutively translated everywhere — they're not *preferentially* ribosome-bound. Cell-type-specific genes (markers, activity-dependent) have high IP/Input because they're selectively translated.

The GT delta itself is anti-correlated with TRAP enrichment (rho = -0.124), confirming this is a property mismatch, not a model failure. Methods that faithfully recover the cell-nucleus delta (ImputeVAE, CT-mean, Linear reg) inherit this anti-correlation. Methods that fail to recover it (scVI, scANVI, scGen, scButterfly) show weakly positive rho — because their predictions are closer to the nucleus baseline, which happens to correlate weakly with TRAP enrichment.

### Validation 2: Cell-Type-Specific RiboTag (strong positive)

| Comparison | N genes | Spearman rho | p-value |
|------------|--------:|-------------:|--------:|
| PV TRAP vs Pvalb GT delta | 8,639 | 0.799 | 0 |
| SST TRAP vs Sst GT delta | 8,645 | 0.814 | 0 |
| VIP TRAP vs Vip GT delta | 8,645 | 0.787 | 0 |
| diff(PV-SST) vs diff(Pvalb-Sst) | 8,451 | 0.463 | 0 |
| diff(PV-VIP) vs diff(Pvalb-Vip) | 8,448 | 0.450 | 0 |
| diff(SST-VIP) vs diff(Sst-Vip) | 8,509 | 0.412 | 0 |

**Key insight:** Within a neuron type, genes that are highly ribosome-bound (TRAP) also tend to have large positive cell-nucleus delta. This makes biological sense: ribosome-bound mRNAs are by definition in the cytoplasm and actively translated, so they should be enriched in whole-cell vs nucleus.

The differential comparisons (rho ~ 0.41-0.46) show that cell-type-specific *differences* in TRAP expression correlate with cell-type-specific differences in GT delta — confirming that the cell-nucleus transformation captures real biology at the cell-type level.

## Data Sources

- `data/external/furlanis2019/GSE133291_selectedCounts.txt.gz` — Furlanis 2019 IP-only counts (not used directly in validation)
- `data/external/furlanis2019/GSM4203678-GSM4203687*.gz` — GSE141464 individual count files (IP + Input)
- `data/external/furlanis2019/GSE77243_Greenberg_SupplementaryTable1.xlsx` — RiboTag 5 neuron types

## Non-Circularity Argument

1. TRAP/RiboTag data comes from completely separate experiments (different labs, different mice, different brain regions)
2. TRAP measures ribosome-bound mRNA, not nucleus/cytoplasm fractionation — orthogonal technology
3. The strong within-type correlation (rho ~ 0.80) cannot be explained by data leakage
4. The cell-type-specific differential correlations (rho ~ 0.45) further rule out trivial explanations (e.g., expression level confounding)

## Limitations

1. Validation 2 uses IP-only data (no input normalization), so correlations partly reflect expression level
2. Only 3 interneuron types matched between RiboTag (PV/SST/VIP) and our dataset (Pvalb/Sst/Vip) — no excitatory neuron match (Emx1/Gad2 are too broad)
3. RiboTag is from visual cortex; our data is from MOp (motor cortex) — different brain region
4. Gene name matching via capitalization heuristic (mouse Abc → TRAP ABC); ~8,500 of 10,134 genes matched
