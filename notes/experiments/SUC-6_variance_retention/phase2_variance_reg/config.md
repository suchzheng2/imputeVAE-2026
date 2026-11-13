# Phase 2: Variance-Preserving Regularization

**Status:** Not started
**Prerequisite:** Frozen encoder experiment (to confirm H1 dominance)

## Rationale

Directly penalize the decoder when within-CT variance of output falls below a target. Forces the decoder to maintain spread.

**Limitation:** Tells the decoder "maintain spread" but provides no *directional* signal — the decoder might add noise to satisfy the penalty rather than recovering meaningful per-cell variation. Key diagnostic: within-CT Spearman should increase, not just variance ratio.

## Why Phase 3 (Residual MSE) is Preferred

Phase 2 (variance penalty) and Phase 3 (residual MSE) both attack H1 (MSE→mean). But they work differently:

| Approach | Signal type | What it rewards | Risk |
|----------|-----------|-----------------|------|
| **Phase 2: Variance penalty** | Statistical — penalize low batch variance | Any output spread, including noise | May add noise, not signal |
| **Phase 3: Residual MSE** | Per-cell — MSE on deviations from CT mean | Output deviations that match target deviations | Meaningful variation only |

**Phase 3 is strictly better** because it gives per-cell directional gradient: "this cell's output should deviate from the CT mean in THIS direction by THIS amount." Phase 2 only says "your outputs are too similar" without saying how to fix it.

**Recommendation:** Skip Phase 2 as standalone. Use Phase 3 (residual MSE) directly. If needed, add Phase 2 as a supplement to prevent the decoder from collapsing during early training (before residual MSE gradients are strong enough).

## Loss Term (if used)

```
L_var = ReLU(target_ratio - Var(decoded|CT) / (Var(real|CT) + eps))
```

- Computed per cell type within each training batch
- Per-gene variance, then averaged across genes
- `target_ratio`: fraction of real cell variance we want to preserve (e.g., 0.5)
- Pre-compute `Var(real|CT)` once at training start

## Hyperparameters to Sweep

| Parameter | Values to test |
|-----------|---------------|
| var_reg_weight | 0.1, 0.5, 1.0 |
| var_reg_target_ratio | 0.3, 0.5 |
| Anneal start | epoch 100 (after recons stabilizes) |

## Code Changes

1. `train.py` — add args: `--var_reg_weight`, `--var_reg_target_ratio`, `--anneal_var_reg_weight`, `--var_reg_weight_start_epoch`, `--var_reg_weight_anneal_epoch`
2. `imputeVAE/stage3_eval.py` — add `compute_loader_celltype_variances(loaders, celltypes)`
3. `imputeVAE/cond_training.py` — compute variance loss in training loop, add to combined loss

## Risk

- May produce noise instead of meaningful variation
- Key diagnostic: within-CT Spearman should increase (not just variance ratio). If Spearman stays flat → model adding noise → use Phase 3 instead.
- Batch size problem: variance needs sufficient samples per CT per batch. Endo (n=23 total) is very noisy.
