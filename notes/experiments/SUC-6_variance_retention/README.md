# SUC-6: Improve Per-Cell Variation Retention

**Linear:** https://linear.app/suchen/issue/SUC-6/improve-per-cell-variation-retention
**Status:** Backlog (blocked by SUC-5)
**Created:** 2026-04-06

## Goal

Increase per-cell variation in imputed nuclei profiles from ~3-11% to >30% of real cell variance, while maintaining cell-type-level metrics.

## Hypothesis

The VAE decoder collapses to cell-type means because:
1. MSE loss converges to conditional mean with unpaired data
2. ~~32-dim latent bottleneck uses ~80% capacity for cell-type identity~~ **INVALIDATED** (see below)
3. Cross-recons pushes toward cell-type means (compresses variance from 11% to 3%)

Multiple interventions targeting different aspects of this problem should recover meaningful per-cell variation.

## Latent Diagnostic (2026-04-07) — Updated Understanding

Full results: `latent_diagnostic/findings.md`

The original hypothesis #2 assumed the latent bottleneck was the problem. A per-dimension analysis of the 32-dim latent (job 3260550) showed this is wrong:

### Three-way variance decomposition of the latent

| Component | Value | Implication |
|-----------|-------|-------------|
| Within-CT variance | 82% | Encoder preserves per-cell variation well |
| Between-CT variance | 18% | Only a small fraction encodes cell-type identity |
| Inter-modality variance | 37% (14/32 dims >50%) | Encoder redundantly encodes cell-vs-nuclei in z |
| Active dims | 28/32 | Spare capacity exists |
| Posterior std | 0.69-0.82 | Healthy, not collapsed |

### Key insight: inter-modality encoding in the latent

The 8-dim modality embedding is supposed to tell the decoder "this is cell" vs "this is nuclei." But the encoder also encodes this distinction in z itself — 37% of total latent variance is inter-modality, with 14/32 dims having >50% of their variance explained by the cell-nuclei shift.

This means:
- **Within-CT variance is inflated** — some of the 82% within-CT signal is actually modality separation, not per-cell biological variation
- **The decoder may use z for modality discrimination** rather than per-cell info, since both z and embedding carry modality signal
- **3 of 4 "dead" nuclei dims** (7, 23, 27) are dead specifically because they encode inter-modality shift and nuclei sit in a narrow band

### Open question: why does the decoder ignore within-CT z?

Two competing hypotheses:

**H1: MSE→mean is the fundamental problem.** With unpaired data, MSE loss's optimal solution is the conditional mean E[cell | z]. Even if z carries per-cell variation, the decoder's best strategy is to ignore it because there's no paired cell to match against — predicting the cell-type mean minimizes MSE. The within-CT z variation is a "free rider" from the reconstruction loss on the same modality, not something the cross-modal path has any reason to use.

**H2: Inter-modality entanglement masks per-cell signal.** The decoder can't separate "this z variation means per-cell biology" from "this z variation means cell-vs-nuclei shift" because they're entangled in the same dimensions. When decoding nuclei→cell, the decoder learns to translate the modality component of z but treats the per-cell component as noise.

**How to distinguish H1 vs H2:**
- Test the **same-modality decoder path** (cell→cell reconstruction). If the cell decoder also collapses within-CT z variation to cell-type means, then H1 is dominant (MSE→mean applies even without cross-modal confound). If the cell decoder preserves per-cell variation, then something specific to the cross-modal path (H2) is the problem.
- This is already measurable: `cell recon vs raw cell` correlation from compeval. Current value is **0.869** — exactly matching the ceiling of `real cell vs own cell-type mean` (0.869). **This strongly supports H1**: even same-modality reconstruction converges to cell-type means.

### Within-CT dimensionality in gene expression space (2026-04-07, job 3261174)

| Metric | Cell | Nuclei |
|--------|------|--------|
| **32 PCs capture** | **30.7%** | **10.7%** |
| 128 PCs capture | 34.3% | 15.0% |
| PCs for 50% variance | >500 | >500 |
| PC1 alone | 22.6% | 5.7% |

**Within-CT gene expression variation is extremely high-dimensional.** 32 dims can only represent ~31% of cell within-CT variance and ~11% of nuclei within-CT variance in the 16K gene expression space. Even 500 PCs can't capture 50%. This is a fundamental information bottleneck that no loss function change alone can overcome.

**This partially rehabilitates Phase 1 (larger latent)** — but for a different reason than originally proposed. The bottleneck is not that dims are saturated with between-CT info (diagnostic disproved that). It's that 32 dims physically cannot represent enough of the high-dimensional within-CT variation. However, going from 32→128 only improves from 30.7%→34.3% for cells — diminishing returns. The variation is so spread out that even 128 dims barely helps.

### Updated root cause (revised)

Three factors contribute:

1. **H1: MSE→mean** — MSE with unpaired data optimizes for E[cell | z]. Same-modality cell recon hits cell-type-mean ceiling (0.869 = 0.869). This is the dominant factor for explaining why the decoder *ignores* z variation it does receive.

2. **Information bottleneck** — 32 dims can represent at most ~31% of within-CT cell variation. Even a perfect decoder with a perfect loss function is capped. But the steep spectrum (PC1 = 22.6%) means the top components carry structured signal — a good encoder/decoder pair should capture the most biologically meaningful variation in 32 dims.

3. **H2: Inter-modality entanglement** — secondary, may waste some of the limited 32-dim capacity on modality info the embedding should handle.

## Current Baseline

### Config B (cond no xr)
| Metric | Value |
|--------|-------|
| Variance ratio (imputed/real) | 11.5% |
| Within-CT Spearman | 0.112 |
| Frac genes rank > 0 | 88.6% |
| z ablation corr | 0.998 |
| z ablation relative MSE | 1.1% |

### PerCT_5pct (cond no xr) — latest
| Metric | Value |
|--------|-------|
| Variance ratio | 11.5% |
| Within-CT Spearman | 0.086 |
| z ablation corr | 0.999 |
| Cell recon vs raw cell | 0.869 (= ceiling) |

### Linear regression baseline (SUC-8)
| Metric | Value |
|--------|-------|
| Variance ratio | 3.30 (amplifies!) |
| Within-CT Spearman | 1.00 |

## Key Decisions

- **Do not use cross-recons (xr) initially.** Compresses variance from 11% to 3%.
- **Phase 1 (larger latent) SKIPPED.** Diagnostic proved bottleneck hypothesis wrong.
- **Focus on decoder-side interventions** — the encoder does its job, the decoder doesn't use the signal.

## Three Root Causes (revised 2026-04-07)

| Cause | Mechanism | Evidence | Severity |
|-------|-----------|----------|----------|
| **H1: MSE→mean** | MSE with unpaired data optimizes for conditional mean; decoder has no per-cell supervision | Cell→cell recon = cell-type-mean ceiling (0.869 = 0.869) | Primary |
| **H3: Modality-blind encoder** | Shared encoder can't distinguish cell-nuclei shift from per-cell variation; z mixes both; decoder gets conflicting signals at cross-decode | 37% inter-modality in z despite 8-dim embedding; alignment loss insufficient | Architectural, fundamental |
| **Info bottleneck** | 32 dims captures only 31% of within-CT cell expression variance (10.7% nuclei) | PCA: >500 PCs for 50% variance | Real but secondary (steep spectrum — top PCs carry structured signal) |

**Key insight:** H3 is not just "wasted capacity" — it actively confuses the decoder. At cross-decode time, z says "this is a nucleus" while the embedding says "output a cell." The safest decoder strategy is to ignore per-cell z variation and predict cell-type means.

**Cross-species implication:** If we can't factor out mouse-cell vs mouse-nucleus in z, we certainly can't factor out mouse vs human. Solving H3 directly enables SUC-11 (cross-species transfer).

## Experiments

### Phase 3: Residual MSE — **SUBMITTED** (2026-04-07, job 3265191)

Script: `jobs/stage2/run_s2_cond_residual_pct5v2.sh`
Base: pct5v2 stage 1 (`run_2026-04-06-21-13-13_3762768`)
Architecture: VAE_JSD_Cond, embed_dim=8, 16,439 genes (PerCT_5pct)

**Idea:** Standard MSE treats between-CT and within-CT errors equally. Since between-CT variance dominates (~89%), the decoder optimizes for cell-type means and ignores per-cell deviations. Residual MSE subtracts cell-type means from both prediction and target before computing MSE, so the gradient is purely about per-cell deviations.

**Loss:** `MSE(decoded - CT_mean, target - CT_mean)` with weight=5.0 (large because residual values are much smaller than full MSE). Annealed from epoch 100→200, after cell reconstruction stabilizes.

**What to watch:**
- Variance ratio: target >30% (from 8.5% baseline cond)
- Within-CT Spearman: target >0.2 (from 0.070 baseline)
- Classification: should stay ≥85% (from 88.7% baseline)
- DE overlap: should stay ≥40% (from 49.5% baseline)

**Risk:** If H1 (MSE→mean) is truly dominant, residual MSE won't help because the decoder still lacks paired per-cell supervision. The optimal decoder output for residual MSE with unpaired data is still E[cell_residual | z] ≈ 0. But if the encoder encodes useful per-cell info (latent diagnostic says 82% within-CT variance), the residual loss should at least amplify the decoder's sensitivity to it.

---

## Updated Plan

### Near-term (attacks H1 — loss signal)
1. **Frozen encoder experiment** — freeze trained encoder, retrain decoder with variance-aware loss. Fastest test: does the existing latent support per-cell variation with better loss? Answers H1 vs H3 dominance.
2. **Phase 2: Variance-preserving regularization** — penalize low within-CT output variance
3. **Phase 3: Residual reconstruction** — **SUBMITTED** job 3265191. Upweight per-cell deviations in loss.

### Medium-term (attacks H3 — modality entanglement)
4. **Phase 5: Adversarial modality removal** — gradient reversal to force modality-invariant z. Also directly enables cross-species transfer.
5. **Phase 1: Larger latent** — reinstated with new rationale (info bottleneck, not between-CT saturation). Test alongside other changes. Diminishing returns (32→128: 30.7%→34.3%).

### Longer-term (full solution)
6. **Phase 6: Factored latent space** — split z into z_shared (invariant) + z_modality. Most principled, enables cross-species. Most engineering.
7. **Phase 4: Read subsampling** — pseudo-paired data for per-cell supervision

## Evaluation Metrics (targets)

| Metric | Current | Target |
|--------|---------|--------|
| Variance ratio | 11.5% | >30% |
| Within-CT Spearman | 0.112 | >0.2 |
| z ablation relative MSE | 1.1% | >5% |
| Classification accuracy | 94.0% | ≥90% (no major degradation) |
| DE overlap | 46.3% | ≥40% (no major degradation) |

## References

- `latent_diagnostic/findings.md` — full diagnostic results and implications
- `phase1_larger_latent/config.md` — why Phase 1 was skipped
- `notes/hallucination_test_results.md` — full variance analysis
- `analysis/latent_diagnostic.py` — diagnostic script
- `analysis/hallucination_test.py` — evaluation script
