# Hallucination Test Results (2026-04-02)

## Verdict: The model IS largely collapsing to cell-type means

Both models produce outputs with dramatically reduced per-cell variance, and replacing each nucleus's z with the cell-type mean z barely changes the output. However, the conditioned decoder (no xr) retains MORE per-cell signal than cond+xr.

---

## Summary Table

| Test | MSE cond | MSE cond+xr | Interpretation |
|------|----------|-------------|----------------|
| **A1: Variance ratio** (imputed/real) | **0.115** | 0.030 | Imputed cells have ~10% (cond) or ~3% (xr) of real cell variance |
| **A2: Within-CT Spearman** | **0.112** | 0.035 | Small but positive ranking preservation |
| **A2: Frac genes rank > 0** | **0.886** | 0.648 | Most genes preserve some ranking (cond), fewer for xr |
| **B1: Ablation corr** (orig vs mean-z) | 0.998 | 0.999 | Replacing z with CT mean barely changes output |
| **B2: z between-CT variance** | 79.1% | 79.9% | ~80% of z variance is cell-type identity |

---

## Detailed Findings

### A1: Per-cell variance (imputed vs real)

**MSE cond**: variance ratio = 0.115 (imputed has 11.5% of real cell variance)
- Best: Endo (0.67), OPC (0.18), Pvalb (0.12)
- Worst: Astro (0.017), Macrophage (0.016)
- Neuronal types: 0.06–0.12 range

**MSE cond+xr**: variance ratio = 0.030 (imputed has only 3% of real cell variance!)
- Cross-recons pushes output even closer to cell-type means
- Almost all cell types < 0.05 ratio
- The cross-recons loss directly supervises toward cell-type means, compressing variation

### A2: Within-cell-type ranking preservation

**MSE cond**: median Spearman = 0.112, 88.6% of genes have positive ranking
- L2/3 IT: 0.113 (99% positive) — retains some nucleus-specific info
- Endo: 0.354 (91% positive) — best, but small sample (n=23)
- Overall: small but consistent positive signal

**MSE cond+xr**: median Spearman = 0.035, 64.8% positive
- L2/3 IT: -0.012 (42% positive) — actually LOSES ranking!
- L6 CT: -0.030 (23% positive) — anti-correlated
- Cross-recons actively erases per-cell variation for some neuronal types

### B1: z ablation

**MSE cond**: corr(original, ablated) = 0.998 — nearly identical outputs
- Relative MSE = 1.1% — replacing z with CT mean changes output by ~1%
- Exception: Endo (0.929 corr, 54.7% relative MSE) — small population, more individual variation retained

**MSE cond+xr**: corr(original, ablated) = 0.999 — virtually identical
- Relative MSE = 0.06% — the decoder almost completely ignores per-cell z variation
- Even Endo: 0.952 corr (vs 0.929 for cond) — xr compresses Endo too

### B2: z variance decomposition

Both models: ~80% of z variance is between cell types, ~20% is within.
The 20% within-CT variance exists in z but the **decoder largely ignores it** (as shown by B1).

---

## Interpretation

### What the model actually does
1. The encoder maps each nucleus to a z that encodes **cell-type identity** (~80%) plus some **per-cell variation** (~20%)
2. The decoder takes z + modality embedding and outputs **approximately the cell-type mean** for the target modality
3. A small amount of per-cell variation passes through (Spearman ~0.11 for cond, ~0.035 for xr)
4. Cross-recons supervision makes this WORSE — it pushes the output toward the cross-modal training targets, which are cell-type means

### MSE cond vs cond+xr: a tradeoff
- **MSE cond** (no xr): better per-cell signal (Spearman 0.112, variance ratio 0.115), but worse distribution-level metrics (R²=0.56)
- **MSE cond+xr**: better distribution-level (R²=0.95), but nearly erases per-cell variation (Spearman 0.035, variance ratio 0.03)

### Why the 0.992/0.998 per-cell corr is misleading
The per-cell corr metric compares each imputed nucleus to its cell-type mean in raw cells. If the model outputs something very close to the cell-type mean, this correlation will be high even without per-cell variation. The baseline (real cell vs own ct mean) is 0.843 because real cells have substantial within-CT variation. The imputed cells scoring 0.992/0.998 means they are MORE similar to the cell-type mean than real cells are — confirming mean collapse.

### Why the classification result IS genuine
Classification accuracy (94-96%) is a DISTRIBUTION-level metric — it tests whether the imputed data has the right cell-type-specific expression pattern, not whether it preserves per-cell variation. The model genuinely produces cell-like expression patterns at the cell-type level.

---

## Honest Framing for Presentation

> "Our model achieves excellent cell-type-level imputation: cross-decoded nuclei are classified as the correct cell type with 96% accuracy (vs 85% for raw nuclei), and preserve 52% of cell-type-specific DE genes. The imputation captures cell-type mean expression profiles with near-perfect fidelity (R²=0.95).
>
> However, per-cell variation is largely compressed — imputed profiles have ~3-11% of the within-cell-type variance of real cells, and within-cell-type gene rankings show modest preservation (Spearman ~0.04-0.11). The conditioned decoder without cross-recons retains more per-cell signal. Improving per-cell recovery — potentially through architectural changes or alternative training objectives — is an important direction for future work."

---

## Implications for Model Choice

For **downstream analyses that depend on cell-type-level patterns** (DE analysis, cell-type classification, marker gene analysis):
→ Use **MSE cond+xr** — best distribution-level metrics

For **analyses that need per-cell variation** (trajectory inference, correlation with other per-cell measurements):
→ Use **MSE cond** (no xr) — retains more individual signal
→ But caveat: even cond only preserves ~11% of variance

For **future improvement**:
→ Need training objective that explicitly preserves within-CT variation
→ Possible: adversarial loss on within-CT distributions, not just means
→ Possible: held-out gene reconstruction as auxiliary objective
