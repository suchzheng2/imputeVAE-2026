---
name: alignment_strategy_plan
description: Plan to fix cross-modality alignment failure in stage 3 (cell2both) training - weight increase + early alignment, with fallback options
type: project
---

## Alignment Strategy Plan (2026-03-19)

The cell2both (Setup 3) stage 2 run showed that alignment JSD never successfully aligned the two modalities. Root causes identified:
1. Alignment weight (0.5) too weak vs reconstruction (1.0-2.0) — recons losses are orders of magnitude larger
2. Alignment delayed (annealed from epoch 0-200) — model falls into unaligned local minimum first
3. Batch-level JSD alignment (not sample-paired) gives weak signal

**Why:** The model optimizes reconstruction and ignores alignment because the gradient signal is too small and arrives too late.

**How to apply:** When configuring stage 3 / cell2both training scripts, use the new weight strategy below.

### Step 1 (Current): Options A + B
- **Increase alignment weight to 5.0** (was 0.5)
- **No annealing on alignment** — full weight from epoch 0
- Equalize reconstruction weights (both 1.0)
- Monitor: does alignment JSD decrease? Do decoded outputs show cross-modality structure?

### Fallback Options (if Step 1 insufficient)
- **Option C:** Replace JSD alignment with MMD (Maximum Mean Discrepancy) — stronger gradients when distributions are far apart, ~30 lines in divergences.py
- **Option D:** GradNorm — dynamically balance gradient norms so alignment always has meaningful influence
- **Option E:** Two-phase with frozen decoder — freeze decoder in phase 2, train only encoder with alignment
