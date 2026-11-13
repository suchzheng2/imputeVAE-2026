# Alignment Strategy Plan

## Phase 1: Latent Alignment (2026-03-19) — COMPLETED

### Problem
The cell2both (Setup 3) stage 2 run (`run_2026-03-13-00-55-52_997499`) showed that alignment JSD never successfully aligned the two modalities after 600 epochs.

### Root Causes
1. **Alignment weight too weak** (0.5) vs reconstruction (1.0-2.0) — recons losses are orders of magnitude larger than JSD, so alignment gradients get drowned out
2. **Alignment delayed** (annealed from epoch 0-200) — model falls into unaligned local minimum before alignment has any effect
3. **Batch-level JSD** (not sample-paired) gives weak signal — the alignment loss compares aggregate batch distributions, not matched samples

### Fix Applied: Increased alignment weight + reduced annealing
- **Increased alignment weight to 5.0** (was 0.5, 10x increase)
- **No annealing on alignment for Setup 3** — full weight from epoch 0
- **Linear ramp over 100 epochs for Setup 1 & 2 stage 3** — alignment from 0→5.0 over epochs 0-100
- **Equalized reconstruction weights** — both cell and nuclei recons at 1.0

### Results (2026-03-20)
Latent alignment succeeded across all 3 setups:
- Setup 1 (cell→nuclei→both): alignment_JSD=0.00079
- Setup 2 (nuclei→cell→both): alignment_JSD=0.00107
- Setup 3 (cell→both direct): alignment_JSD=0.00060 (best)

**However:** decoder output does NOT make nuclei cell-like. Shift scores are consistently negative (-0.10 to -0.13) across all setups, meaning the decoder faithfully reconstructs each modality separately. The alignment only acts in latent space — there is no decoder-level pressure.

---

## Phase 2: Decoder-Level Cross-Modality (2026-03-20) — IN PROGRESS

### Problem
Latent alignment is achieved but the decoder still produces modality-specific output. Decoded nuclei remain nuclei-like because the loss only asks `NB(decoded_nuclei, raw_nuclei)`.

### Fix Applied: Cross-modality reconstruction loss
Added a new optional loss term that compares decoded nuclei output against raw cell data:

```
cross_recons_loss = NB(rescaled_decoded_nuclei, raw_cell)
```

Library size re-scaling: recovers gene proportions from decoded nuclei (`softmax(rho)`) and re-scales with cell library size for fair NB comparison.

**Loss function is now:**
```
loss = nuclei_recons_weight * NB(decoded_nuclei, raw_nuclei)
     + cell_recons_weight * NB(decoded_cell, raw_cell)
     + nuclei_jsd_weight * nuclei_jsd
     + cell_jsd_weight * cell_jsd
     + alignment_weight * alignment_jsd
     + cross_recons_weight * NB(rescaled_decoded_nuclei, raw_cell)   ← NEW
```

Controlled by `--cross_recons_weight` (default 0.0 = disabled). Supports annealing.

### Quick Probe Results (2026-03-20) — COMPLETED

Ran 5 quick probes (100 epochs, subset_ratio=0.05) to test different weight strategies.
See `notes/training_runs.md` for full configuration and numeric results.

**Winner: cross-recons loss is the critical ingredient.**
- Probes D (cross-recons=1.0) and E (cross-recons=0.5 + low nuclei_recons=0.5) both achieve:
  - Nuclei shift near zero (-0.02 to -0.04) vs baseline -0.13
  - Decoded nuclei vs raw cell correlation 0.83-0.84 vs baseline 0.61
- Weight rebalancing alone (probes B, C) is insufficient without cross-recons
- Alignment weight can be reduced from 5.0 to 2.0-3.0 when using cross-recons

### Active Full Runs (2026-03-20)
| Job | Description | cross_recons | alignment | Script |
|-----|-------------|-------------|-----------|--------|
| 2560805 | cell→both + cross-recons | 0→0.5 over 100ep | 5.0 (no anneal) | `jobs/stage3/train_stage2_cell2both_v3_cross_recons.sh` |
| 2560882 | cell→both + anneal alignment | 0 (disabled) | 0→5.0 over 100ep | `jobs/stage3/train_stage2_cell2both_v3_anneal.sh` |

Both use cell stage1 model (928562), 600 epochs, batch_size=128, lr=1e-4, subset=0.2.

### Full Run Results (2026-03-20)
- **cross-recons=0.5, align=5.0 no anneal** (job 2560805): shift=-0.071, nuc→cell corr=0.762
- **no cross-recons, align=5.0 annealed** (job 2560882): shift=-0.123, nuc→cell corr=0.717

Cross-recons clearly helps (~2x shift improvement), but the effect is weaker at 600 epochs / larger data than in the 100ep probes.

### Alignment weight=1.0 reruns (2026-03-22) — Routes 1 & 2
Tested whether reducing alignment from 5.0 to 1.0 helps (no cross-recons):
- **Route 1** (job 2136850): shift=-0.208, nuc→cell gene corr=0.605, alignment_JSD=0.008
- **Route 2** (job 2119158): shift=-0.195, nuc→cell gene corr=0.591, alignment_JSD=0.033
- Conclusion: lower alignment doesn't help decoder. Shift is even worse than align=5.0.

### Cross-Recons Design Discussion (2026-03-23)

**Goal:** decoded nuclei gene *proportions* should match cell gene proportions.

**Current implementation:** NB cross-recons with library-size rescaling.
- Extracts gene proportions from decoded nuclei: `proportions = decoded_nuclei / nuclei_library_size`
- Rescales to cell library size: `cross_mu = cell_library_size * proportions`
- Computes NB loss against raw cell counts

**Scale matching consideration:**
- With raw count data, decoded nuclei and decoded cells are on fundamentally different scales (nuclei mean ~0.66, cell mean ~1.9) due to different library sizes
- Matching real *scale* (absolute values) would require normalized data (log1p or CPM) so both modalities are on the same scale before the model sees them
- This would require switching from VAE_NB to an MSE-based VAE — different loss function, effectively a different model
- **Decision: stay on raw data + NB model, focus on matching proportions** (not absolute scale)
- The cross-recons loss already operates at the proportion level via library-size rescaling, which is the correct approach for raw count data

**Next exploration directions for cross-recons:**
1. **Weight sweep:** cross_recons at 0.25, 0.5, 1.0, 2.0 — find the sweet spot
2. **Interaction with nuclei_recons:** 2D sweep of cross_recons x nuclei_recons (Probe E showed lowering nuclei_recons helps)
3. **Bidirectional cross-recons:** also push decoded_cell → raw_nuclei (symmetric regularizer)
4. **Cross-recons on Routes 1 & 2** (so far only tested on Route 3)
5. **Annealing schedule:** does the model benefit from establishing reconstruction before cross-recons?
6. **Alternative loss in proportion space:** MSE or cosine similarity on gene proportions instead of NB — simpler gradient, may converge faster

### Next Steps
- Run quick probes for cross-recons weight sweep and nuclei_recons interaction
- Test proportion-space loss variants (MSE on proportions, cosine similarity)

---

## Fallback Options (if current approach insufficient)

### Option C: MMD alignment
Replace JSD alignment with Maximum Mean Discrepancy — stronger gradients when distributions are far apart, ~30 lines in `divergences.py`, no architectural changes.

### Option D: GradNorm
Dynamically balance gradient norms so alignment always has meaningful influence regardless of loss magnitude.

### Option E: Two-phase with frozen decoder
1. Phase 1: Train with reconstruction + per-modality JSD (current setup)
2. Phase 2: Freeze the decoder, train only encoder with alignment loss + small reconstruction penalty
