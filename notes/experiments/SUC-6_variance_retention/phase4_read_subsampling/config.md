# Phase 4: Read Subsampling for Paired Data

## Rationale

The gold standard approach. Creates pseudo-paired data by subsampling reads from whole-cell data to simulate nucleus-like profiles. Enables direct per-cell supervision.

## Method

1. Take raw cell counts
2. For each cell, multinomially subsample `fraction * total_counts` reads
3. Log1p normalize the subsampled profile → "pseudo-nucleus"
4. Encode pseudo-nucleus (nuclei embedding) → decode with cell embedding → MSE vs original cell

This provides explicit per-cell training signal: "this specific pseudo-nucleus should produce this specific cell profile."

## Loss Term

```
L_paired = MSE(cross_decode(subsample(cell)), cell)
```

## Subsampling Approaches

1. **Uniform** (start here): sample 30% of reads uniformly across genes
2. **Gene-weighted**: weight by observed nucleus/cell ratio per gene — cytoplasmic genes preferentially lost
3. Fraction ~0.3 matches typical nucleus/cell library size ratio

## Hyperparameters

| Parameter | Values to test |
|-----------|---------------|
| paired_subsample_weight | 0.5, 1.0, 2.0 |
| paired_subsample_fraction | 0.3 |
| Anneal start | epoch 200 (after cell-type mapping established) |

## Code Changes

1. `imputeVAE/data.py` — add `subsample_reads(X_raw, fraction)` function; modify `myDataset` to optionally store raw counts alongside log1p
2. `imputeVAE/cond_training.py` — add paired_recons_loss term in training loop
3. `train.py` — add args for paired_subsample_weight, fraction, annealing

## Data Pipeline Consideration

Subsampling needs raw counts even when training with log1p data. Options:
- Store both raw and log1p in dataset (doubles memory)
- Store raw, normalize on-the-fly (simpler, small CPU cost) ← recommended
- Subsampling must be re-randomized each epoch (different random subsamples)

## Risk

Uniform subsampling may not capture real cell-to-nucleus gene-specific differences. Model might learn "unsubsampling" rather than true cell-nucleus translation. Evaluate by comparing to Phase 2+3 results.

## Timeline

~1-2 weeks. Only start after Phases 1-3 converge.
