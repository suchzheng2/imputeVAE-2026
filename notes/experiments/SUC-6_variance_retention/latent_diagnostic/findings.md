# Latent Diagnostic Findings (2026-04-07)

**Job:** 3260550 (COMPLETED)
**Model:** pct5v2_cond (run_2026-04-07-02-12-05_3779214), 32-dim latent
**Script:** `analysis/latent_diagnostic.py`
**Data:** 10k subsampled cells + 10k nuclei, PerCT_5pct gene list (16,439 genes)

## Three-way variance decomposition

Each latent dimension's total variance can be decomposed into:
1. **Between-CT** — encodes cell-type identity
2. **Within-CT** — encodes per-cell variation within a cell type
3. **Inter-modality** — encodes the cell vs nuclei difference

### Key numbers

| Component | Nuclei (within-mod) | Cell (within-mod) | Pooled |
|-----------|--------------------|--------------------|--------|
| Frac between-CT | 17.8% | 17.0% | — |
| Frac within-CT | 82.2% | 83.0% | — |
| Frac inter-modality | — | — | 37.0% mean |
| Dims >50% inter-mod | — | — | 14/32 |

### Dead dims are inter-modality dims

The 4 "dead" nuclei dims (7, 21, 23, 27) tell different stories:

| Dim | NucVar | CellVar | Cell/Nuc ratio | Inter-mod frac | Interpretation |
|-----|--------|---------|---------------|----------------|----------------|
| 7 | 0.003 | 0.008 | 2.5x | **0.53** | Low var in both, but >50% of what exists is cell-nuclei difference |
| 23 | 0.007 | 0.027 | 3.6x | **0.52** | Same — mostly encoding modality, not per-cell info |
| 27 | 0.004 | 0.011 | 3.2x | **0.44** | Similar |
| 21 | 0.005 | 0.009 | 1.7x | **0.003** | Truly dead — low var, no modality signal either |

Dims 7, 23, 27 are "dead in nuclei" because they primarily encode a cell-vs-nuclei shift, and nuclei occupy a narrow region. Only dim 21 is genuinely unused.

### Per-cell variation is real but modality-confounded

The high within-CT fraction (82%) is encouraging, but 37% of total variance is inter-modality. This means within-CT variance partly reflects that cells and nuclei have different spread patterns *within* the same cell type — not just per-cell biological variation.

Cells consistently have 3-5x more latent variance than nuclei on the same dimensions. This is partly biological (more mRNA captured) and partly the encoder learning that cells and nuclei live in different regions of latent space.

## Implications for SUC-6

### Phase 1 (larger latent) — **SKIP**
- 28/32 dims active, 82% within-CT — the encoder has capacity it isn't fully using
- Dead dims are explained by inter-modality encoding, not capacity exhaustion
- Adding more dims would likely just add more inter-modality separation, not per-cell variation

### What the decoder sees
- The decoder receives z (32 dims, rich within-CT variation) + modality embedding (8 dims)
- It's reconstructing accurately at the cell-type-mean level (gene corr 0.91-0.96)
- But it produces output with var ratio 0.035-0.115 despite latent var ratio ~0.82
- **The decoder is collapsing within-CT variation to the conditional mean**

### Root cause hypothesis (updated)
The problem is not:
- ❌ Latent bottleneck (82% within-CT, 28 active dims)
- ❌ Posterior collapse (std 0.69-0.82, healthy)
- ❌ Gene list (same issue across Config B and PerCT_5pct)

The problem IS:
- ✅ **MSE loss optimizes for the conditional mean** — with unpaired data, the best MSE prediction for a nuclei input is the cell-type mean cell profile, regardless of per-cell z variation
- ✅ **Decoder has no incentive to use within-CT z signal** — MSE doesn't reward it
- ✅ The encoder preserves within-CT variation because JSD/reconstruction loss rewards it, but the cross-modal decoder path has no loss that requires per-cell fidelity

### Why does the decoder ignore within-CT z? Two hypotheses

**H1: MSE→mean.** MSE with unpaired data optimizes for E[cell | z]. The decoder's best strategy is to predict the cell-type mean regardless of per-cell z, because there's no paired target to reward deviations. Evidence: cell recon vs raw cell correlation = 0.869, exactly matching the ceiling (real cell vs own CT mean = 0.869). Even the *same-modality* path collapses to means — this is not specific to cross-modal imputation.

**H2: Inter-modality entanglement.** 37% of z variance encodes cell-vs-nuclei. The decoder can't separate "per-cell biology" from "modality shift" in z, so it treats within-CT z variation as noise when doing cross-modal decoding.

**Verdict: H1 is dominant.** The same-modality cell reconstruction already hits exactly the cell-type-mean ceiling. If entanglement (H2) were the main issue, same-modality decoding (where there's no cross-modal confound) would preserve per-cell variation — but it doesn't. MSE→mean is the fundamental problem.

**How to investigate further if needed:**
- Freeze the encoder, train a new decoder head with a variance-aware loss → if variance recovers, confirms H1 (decoder training signal is the issue, not encoder representation)
- Add an adversarial modality classifier on z to force modality invariance → if same-modality reconstruction improves its per-cell variation, H2 contributes

### Experiment idea: frozen encoder + fresh decoder (TODO — discuss before implementing)

Cleanest test of H1 vs H2 and the capacity question:
- Freeze the trained encoder (we know it encodes within-CT variation)
- Attach a fresh decoder head
- Train with a variance-aware loss (Phase 2's variance penalty or Phase 3's residual weighting)
- If per-cell variation recovers → confirms the problem is the loss signal (H1), not the representation
- If it doesn't → the 32-dim bottleneck genuinely can't carry enough info for 16K gene reconstruction
- **Advantage:** skips Stage 1 entirely, fast experiment (~hours not days)
- **Prerequisite:** within-CT PCA results — need to know if 32 dims is theoretically sufficient before testing

### Within-CT dimensionality (PCA on gene expression)

**Question:** Can 32 latent dims even represent per-cell variation across 16K genes?

The earlier diagnostic showed the encoder puts within-CT info into z, but that doesn't prove 32 floats carry *enough* information. Linear regression uses 32K params (2 per gene, each gene independent). The VAE funnels everything through 32 shared dims — 514:1 compression.

If within-CT variation is intrinsically low-dimensional (driven by a few transcriptional programs), 32 dims is sufficient. If it's high-dimensional, 32 dims physically cannot represent it and the bottleneck IS real — just not in the way we originally thought (not about between-CT vs within-CT allocation, but about total capacity).

**Diagnostic:** PCA on within-CT residuals (gene expression after subtracting cell-type means). Job 3260550-v2 measures PCs needed for 50%/70%/90%/95% of within-CT variance, for both cells and nuclei separately.

**Results (job 3261174):**

| Metric | Cell | Nuclei |
|--------|------|--------|
| Total within-CT variance | 3839 | 2521 |
| **32 PCs capture** | **30.7%** | **10.7%** |
| 64 PCs capture | 32.3% | 12.3% |
| 128 PCs capture | 34.3% | 15.0% |
| PCs for 50% variance | >500 | >500 |
| PCs for 90% variance | >500 | >500 |
| PC1 alone | 22.6% | 5.7% |

**Conclusion: Within-CT variation is extremely high-dimensional.** Even 500 PCs can't capture 50% of the variance. 32 dims captures only 30.7% for cells and 10.7% for nuclei. This is a fundamental information bottleneck — no loss function change can make the decoder reconstruct variation that the latent physically cannot represent.

However, there's a subtlety: PC1 captures 22.6% of cell within-CT variance alone. The spectrum is very steep — a few PCs capture the structured variation (transcriptional programs), and the long tail is likely stochastic/technical noise. The question is whether the biologically meaningful per-cell variation lives in the top 32 PCs (recoverable) or is spread across hundreds (not recoverable through a 32-dim bottleneck).

**This changes the strategy:** larger latent (Phase 1) may actually be needed — not because the current 32 dims are saturated with between-CT info (they're not), but because 32 dims cannot represent enough within-CT variation in expression space. The original reasoning was wrong but the intervention might still help, for a different reason.

### H3: Modality-blind encoder makes entanglement inevitable (2026-04-07)

The encoder is shared and receives no modality label. From its perspective, the systematic cell-nuclei difference is indistinguishable from per-cell biological variation — both are just "input variation." This means:

- **z inevitably encodes modality info** — not a training failure, but an architectural inevitability
- **The 82% within-CT variance is inflated** — some of it (when pooling modalities) is modality shift, not per-cell biology. (Note: our per-modality diagnostic measured within-CT separately for cells and nuclei, so those individual numbers are clean. The issue is what happens at cross-decode time.)
- **The alignment loss is the only mechanism to make z modality-invariant**, but 37% remaining inter-modality variance shows it's insufficient
- **At cross-decode time**, the decoder receives z from "nuclei territory" + cell embedding. These conflict — z says "nucleus" while embedding says "cell." The decoder resolves this by ignoring fine-grained z variation and defaulting to cell-type means.

This reframes H2 from "secondary concern" to **architecturally fundamental**: the entanglement isn't just wasting capacity, it's actively confusing the decoder during cross-modal inference.

**Critical implication for cross-species transfer (SUC-11):** If we train on mouse and apply to human, the species difference creates the same problem at a larger scale. Human nuclei will land in a different region of latent space than mouse cells — an even larger "inter-modality" shift. If the encoder can't factor out mouse-cell vs mouse-nucleus, it certainly can't factor out mouse vs human. Any solution to the modality entanglement problem directly enables cross-species transfer.

### Potential solutions to modality entanglement

**Option A: Adversarial modality removal**
- Add a discriminator D(z) → {cell, nuclei} trained to predict modality from z
- Train encoder to fool D (gradient reversal layer)
- Forces z to be modality-invariant — all modality info pushed to the embedding
- **Pro:** principled, well-studied (domain adaptation literature). Directly solves the problem.
- **Pro for cross-species:** same framework extends to species-invariant z — add species as another domain
- **Con:** adversarial training is unstable, adds hyperparameters (discriminator architecture, gradient reversal weight)
- **Con:** may strip out biological variation that correlates with modality (e.g., cytoplasmic RNA genes are both modality-specific AND biologically meaningful)

**Option B: Modality-conditioned encoder**
- Give the encoder the modality label as input (like the decoder already gets)
- Encoder can explicitly learn: "given that this is a nucleus, factor out the nuclei-specific shift"
- **Pro:** simple, symmetric with decoder design
- **Con:** at cross-species inference time, what modality label do you give for human nuclei? The encoder learned mouse-cell=0, mouse-nuclei=1. Need a way to generalize.
- **Con:** doesn't guarantee z becomes modality-invariant — encoder might just ignore the label

**Option C: Stronger/different alignment loss**
- Current: JSD between cell and nuclei latent distributions (global, one number)
- Problem: JSD matches overall distributions but doesn't enforce per-dimension alignment
- Alternative: per-dimension MMD, or optimal transport, or explicit mean+variance matching per cell type per dimension
- **Pro:** no architectural changes, just loss modification
- **Con:** stronger alignment might over-compress z (lose per-cell variation we want to keep)
- **Cross-species:** would need paired cell types across species — may not be available

**Option D: Factored latent space**
- Split z into z_shared (modality-invariant) + z_modality (modality-specific)
- Adversarial loss on z_shared (must not predict modality)
- Reconstruction uses both; cross-decode uses z_shared + target embedding
- **Pro:** explicitly separates the two sources of variation. Most principled.
- **Pro for cross-species:** z_shared could be designed as species-invariant too
- **Con:** most complex architecturally. How to decide the split (how many dims for each)?

### Assessment for cross-species generalization

Options A and D are most promising for cross-species transfer:
- They create a modality-invariant (and potentially species-invariant) latent space
- At inference on human nuclei: encode → z_shared (invariant) → decode with human-cell embedding
- Requires that biological cell types share enough structure across species for z_shared to be meaningful — likely true for conserved types (neurons, astrocytes) but less so for species-specific types

Option B is weakest for cross-species because the encoder conditioning is species-specific.

Option C could work but requires matched cell types across species for alignment.

### Recommended next steps (revised)
1. **Phase 2 (variance regularization)** — fast, attacks H1. Even if entanglement (H3) also contributes, we need to know what variance-aware loss alone can do.
2. **Frozen encoder experiment** — fast test: does the existing latent support per-cell variation with a better loss? If yes → H1 dominant, proceed with loss fixes. If no → bottleneck + entanglement matter, need architectural changes.
3. **Option A (adversarial)** as Phase 5 — if loss fixes alone don't work. Also directly enables cross-species path (SUC-11).
4. **Option D (factored latent)** — longer-term, for the cross-species paper. Most principled but most engineering.
