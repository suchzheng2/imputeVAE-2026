# Phase 3: Residual MSE — Per-Cell Deviation Reconstruction

**Status:** Not started — RECOMMENDED as primary approach
**Prerequisite:** Frozen encoder experiment (to confirm loss change helps)

## Rationale (updated 2026-04-07)

### The MSE gradient imbalance problem

Standard MSE loss:
```
L = mean((decoded - target)^2)
```

This loss is dominated by between-CT differences because:
- Between-CT expression differences are ~10-100x larger than within-CT deviations
- MSE weights errors by magnitude: error of 2.0 → loss 4.0, error of 0.1 → loss 0.01
- The decoder allocates virtually all capacity to getting cell-type means right
- Within-CT gradients exist but are ~100x weaker than between-CT gradients

**Result:** The decoder learns a locally flat mapping in the within-CT regions of z. Small z differences (per-cell variation) produce negligible output differences. The decoder converges to cell-type means not because MSE is wrong, but because MSE doesn't weight the small within-CT signal enough relative to the large between-CT signal.

### Why residual MSE fixes this

Residual MSE explicitly isolates the within-CT component:
```
L_residual = MSE(decoded - CT_mean_target, target - CT_mean_target)
```

This term:
- **Removes between-CT signal entirely** — CT means cancel out
- **Gradient is purely about per-cell deviations** — "does the decoder reproduce THIS cell's deviation from the CT mean?"
- **Provides directional signal** — not just "add variance" but "deviate in THIS direction by THIS amount"
- **No batch-size problem** — unlike variance penalty, doesn't need per-CT batch statistics. Each cell has its own residual target.

### Combined loss
```
L_total = L_recons + λ_residual * L_residual
         = MSE(decoded, target) + λ * MSE(decoded - CT_mean, target - CT_mean)
```

- `CT_mean`: pre-computed from training data, stored as GPU tensors. Indexed by cell type label per sample.
- First term: get cell-type structure right (between-CT)
- Second term: get per-cell deviations right (within-CT)
- λ controls the balance — effectively upweights within-CT signal by λ

### Why CT_mean should be from raw training data (not model output)

Use pre-computed cell-type means from the training dataset as the reference, NOT the decoded mean. The decoded mean changes during training — creates a moving target that destabilizes optimization. The raw data mean is fixed and correct.

### What this can and cannot fix

**Can fix (H1):** The decoder ignoring within-CT z variation because MSE gradients are too weak. Residual MSE directly strengthens this signal.

**Cannot fix alone:**
- **Information bottleneck:** 32 dims can only represent ~31% of within-CT cell variance. Residual MSE can make the decoder use that 31% better, but can't recover the missing 69%.
- **H3 (modality entanglement):** At cross-decode time, z still carries modality info. Residual MSE helps the same-modality path but doesn't fix cross-modal confusion.

### Connection to other phases

- **Phase 2 (variance penalty):** Residual MSE is strictly better — provides directional signal, not just "add spread." Could combine both if needed as early-training stabilizer.
- **Frozen encoder experiment:** Test residual MSE on frozen encoder first (cheapest). If it works → train full model with residual MSE.
- **Modality-sink encoder:** Complementary — modality-sink fixes H3 (encoder), residual MSE fixes H1 (loss). Stack both for best result.
- **Phase 1 (larger latent):** Could stack — more dims to capture within-CT variation + loss that actually uses it. But diminishing returns on dims.

## Implementation

### Hyperparameters

| Parameter | Values to test |
|-----------|---------------|
| residual_weight (λ) | 1.0, 2.0, 5.0, 10.0 |
| Anneal schedule | Ramp from 0 over 100 epochs after recons stabilizes |
| CT mean source | Pre-computed from full training set (fixed) |

Note: λ needs to be relatively large because residual MSE values are much smaller than full MSE (within-CT variance << total variance).

### Code Changes

1. **`train.py`** — add args: `--residual_weight`, `--anneal_residual_weight`, `--residual_weight_start_epoch`, `--residual_weight_anneal_epoch`

2. **`imputeVAE/cond_training.py`** — pre-training setup:
   ```python
   # Pre-compute CT means (once, before training loop)
   ct_means = {}  # {cell_type_str: tensor [n_genes]}
   for ct in unique_cell_types:
       mask = cell_types == ct
       ct_means[ct] = X_train[mask].mean(dim=0)  # [n_genes]
   ```

   In training loop (after reconstruction loss):
   ```python
   # Get CT mean for each sample in batch
   batch_ct_means = torch.stack([ct_means[ct] for ct in batch_cell_types])  # [B, G]

   # Residual MSE
   decoded_residual = decoded - batch_ct_means
   target_residual = target - batch_ct_means
   residual_loss = F.mse_loss(decoded_residual, target_residual)

   loss += residual_weight * residual_loss
   ```

3. **Logging** — track residual_loss separately in epoch dumps for monitoring

### For the frozen encoder experiment

Even simpler — just retrain the decoder:
```python
# Freeze encoder
for param in model.encoder.parameters():
    param.requires_grad = False
for param in model.mu.parameters():
    param.requires_grad = False
for param in model.log_var.parameters():
    param.requires_grad = False

# Train decoder with residual MSE
# ... (same loss as above, just decoder params update)
```

## Evaluation

Compare against current baseline:

| Metric | Current (pct5v2_cond) | Target |
|--------|----------------------|--------|
| Variance ratio | 11.5% | >30% |
| Within-CT Spearman | 0.086 | >0.2 |
| z ablation relative MSE | ~1% | >5% |
| Cell recon vs raw cell | 0.869 (= ceiling) | >0.88 (break ceiling) |
| Classification | 93.3% | ≥90% (no major loss) |

**Critical success metric:** within-CT Spearman must increase, not just variance ratio. Spearman increasing = model recovering real per-cell signal. Variance ratio increasing + flat Spearman = model adding noise.

## Risk

- λ too high → decoder prioritizes residuals over cell-type structure → classification/DE degrade
- λ too low → no effect, decoder still ignores within-CT signal
- The 31% bottleneck caps improvement — even perfect residual reconstruction through 32 dims can only recover ~31% of within-CT cell variance
