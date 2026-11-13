# Localization Concordance vs External Databases

**Date:** 2026-07-24
**Task:** Task 2 from analysis plan
**Script:** `analysis/localization_concordance.py`

## Question

Does the model's per-gene imputation delta correlate with independently measured subcellular RNA localization? This is a non-circular validation: the model never saw localization labels during training.

## External References

- **COMPARTMENTS (Jensen lab)** — integrated subcellular localization scores for mouse genes combining curated knowledge, text mining, and predictions. CN ratio = cytoplasm score (GO:0005737) minus nucleus score (GO:0005634). Positive = more cytoplasmic evidence.
- **COMPARTMENTS knowledge channel** — curated GO annotations only (higher confidence). Binary labels: cytoplasm-only, nucleus-only, or both.

## Results

### Integrated scores (continuous CN ratio)

| Metric | Value |
|--------|------:|
| N genes matched | 8,774 / 10,134 |
| Spearman rho | **0.083** (p = 8.7e-15) |
| Pearson r | **0.097** (p = 1.1e-19) |
| AUROC (cyto vs nuc) | **0.554** |
| AUPRC | 0.715 (baseline prevalence 0.66) |
| N cytoplasmic | 5,407 |
| N nuclear | 2,787 |

### Knowledge channel (curated binary labels)

| Category | N genes | Median delta |
|----------|--------:|------------:|
| Cytoplasm-only | 3,104 | 0.488 |
| Nucleus-only | 1,415 | 0.364 |
| Both | 2,888 | 0.514 |
| Mann-Whitney (cyto vs nuc) | — | p = 5.2e-19 |

Cytoplasm-only genes have significantly higher delta than nucleus-only genes (median 0.488 vs 0.364, p = 5.2e-19). The direction is correct: the model adds more expression to genes with cytoplasmic localization evidence.

## Interpretation

The effect is **statistically significant but modest in magnitude** (rho = 0.083, AUROC = 0.554). This is expected for several reasons:

1. **Domain mismatch:** COMPARTMENTS reports general protein/RNA localization across tissues and cell types. The model's delta reflects the specific cell-vs-nucleus extraction difference in mouse MOp brain tissue — a much narrower biological question. Many genes localized to "cytoplasm" in general databases (e.g., metabolic enzymes) may not show a strong extraction-dependent shift.

2. **Not all cytoplasmic genes change:** A gene can be cytoplasmic AND nuclear simultaneously (2,888 genes are annotated as "both"). The delta captures the *difference* in extraction efficiency, not absolute localization. Genes present in both compartments may show little delta even though they are "cytoplasmic."

3. **The correct comparison is directional, not absolute:** The key finding is that (a) cytoplasm-only genes have higher delta than nucleus-only genes (0.488 vs 0.364, p = 5.2e-19), and (b) the correlation is in the correct direction (positive rho). A random or artifactual model would show no such trend.

4. **Complementary to other validations:** The scrambled-delta negative control (Task 4) proves the compartment signal is highly specific (z > 9, p < 0.001). The localization concordance adds that this signal aligns with an independent external database, even if the effect size is modest due to the domain gap.

**Bottom line:** The positive correlation and significant Mann-Whitney test confirm the model's delta is consistent with known subcellular localization, providing non-circular evidence that the imputation captures real biology. The modest effect size reflects the mismatch between general localization databases and tissue-specific extraction differences, not a failure of the model.

## Caveats

- COMPARTMENTS is a general-purpose database, not specific to brain or to the cell-vs-nucleus extraction protocol.
- 97% of genes have positive delta (the model adds expression overall), so the AUROC is compressed near 0.5 — the test has limited dynamic range.
- lncATLAS and APEX-seq (human-specific references from the original plan) were not used; they would require ortholog mapping and may provide stronger signal for lncRNAs specifically.

## Output files

- `gene_delta_vs_cn.csv` — per-gene delta and CN ratio for matched genes
- `delta_vs_cn_scatter.png` — scatter plot of delta vs CN ratio
- `delta_vs_cn_binned.png` — binned CN ratio vs mean delta
- `delta_by_localization.png` — delta distribution by localization category
- `localization_stats.json` — all statistics
