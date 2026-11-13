---
name: Prashant meeting 2026-04-15
description: Meeting with Prashant on variance collapse — agreed on CPM+log1p normalization, discussed ZINB loss and zero/nonzero weighting
type: project
---

Meeting with Prashant (2026-04-15) on SUC-6 variance collapse findings.

**Decisions:**
- Implement CPM normalization (counts per million + log1p) to remove library size as free MSE reduction
- Analyze zero vs non-zero entries separately in evaluation
- Consider NB or ZINB loss function instead of MSE/Gaussian assumption
- Consider weighting reconstruction loss differently for zeros vs non-zeros

**Key discussion points:**
- Model aggressively imputes zeros → outputs converge to CT mean distributions
- Cannot distinguish technical zeros from biological zeros
- Sparse genes: majority-zero rule shifts predictions toward zero (MSE→mean)
- Dense genes: more normal-like, model does better
- First histogram peak at ~0.69 = ln(2) from discrete counts
- Last latent dim (31) highly correlated with library size — CPM should fix this

**Action items for Suchen:**
1. Implement CPM + log1p normalization (priority)
2. Analyze zero vs non-zero entries separately (already done — within_ct_nonzero_findings.md)
3. Research cross-model reconstruction loss in autoencoders
4. Evaluate alternative loss functions for sparsity preservation
5. Follow-up next week on normalization results

**Why:** CPM removes library size from the data, forcing the decoder to learn gene-specific variation. ZINB explicitly models the zero-generation process instead of treating zeros as small continuous values.

**How to apply:** CPM experiment is first priority. ZINB is medium-term — we already tried NB (VAE_NB) and it had softmax bottleneck issues, but ZINB is different.
