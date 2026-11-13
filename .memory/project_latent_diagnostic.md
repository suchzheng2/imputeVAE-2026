---
name: Latent diagnostic results
description: 32-dim latent analysis — 82% within-CT, 37% inter-modality, bottleneck hypothesis invalidated, decoder is the problem
type: project
---

Latent diagnostic (job 3260550, 2026-04-07) on pct5v2_cond model showed:

- **82% of latent variance is within-CT** — encoder preserves per-cell variation well
- **37% of total variance is inter-modality** (cell vs nuclei shift), 14/32 dims have >50% inter-mod
- 28/32 dims active; 3 of 4 dead nuclei dims are inter-modality dims (encode cell-nuclei separation)
- Posterior std healthy (0.69-0.82), not collapsed
- Cells have 4.4x more within-CT latent variance than nuclei

**Why:** Ran this before Phase 1 (larger latent) to check if bottleneck hypothesis was valid.

**How to apply:**
- Three root causes: H1 (MSE→mean, dominant), H3 (modality-blind encoder, architectural), info bottleneck (secondary)
- H3 insight: shared encoder can't separate modality from per-cell variation → z mixes both → decoder gets conflicting signals at cross-decode → defaults to cell-type means. This is architectural, not just a loss issue.
- H3 directly connects to cross-species (SUC-11): if encoder can't factor out cell-vs-nuclei, it can't factor out mouse-vs-human either
- Frozen encoder experiment is the key discriminator: if loss fix works with existing z → H1 dominant. If not → H3/bottleneck matter, need architectural changes.
- Adversarial modality removal (gradient reversal) is most promising for both variance AND cross-species
- Within-CT PCA: 32 PCs capture 31% cell, 11% nuclei. Bottleneck is real but steep spectrum means top PCs carry structured signal.
- Results in `notes/experiments/SUC-6_variance_retention/latent_diagnostic/`
