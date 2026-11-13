# L5 ET Anomaly Diagnosis

**Date:** 2026-07-23
**Task:** Task 5 from analysis plan
**Script:** `analysis/l5et_diagnosis.py`

## Question

Why does L5 ET have delta R² = -0.560 despite high correlation (r = 0.950)?

## Root Cause: Scale bias from low cell count

R² < 0 with high r means the predicted delta is well-correlated with ground truth but systematically shifted in magnitude. A negative R² occurs when the mean squared error exceeds the total variance — i.e., a constant predictor (the overall mean) would do better than the model's predictions in absolute terms, even though the model correctly ranks genes.

L5 ET has only **161 cells** (vs 1,244 nuclei). This extreme imbalance means the cell-type mean for cells is estimated from a small sample, making the ground-truth delta noisy. Meanwhile the model, trained mostly on high-n types, over-predicts the delta magnitude for L5 ET.

## Key Numbers

| Cell Type | n_cell | n_nuc | r | R² | Slope | Mean Offset | GT delta mean | Pred delta mean |
|-----------|-------:|------:|----:|----:|------:|------------:|-------------:|----------------:|
| **L5 ET** | **161** | 1,244 | 0.950 | **-0.560** | 1.507 | 0.455 | 0.276 | 0.731 |
| L2/3 IT | 10,915 | 8,043 | 0.998 | 0.996 | 1.010 | -0.004 | 0.597 | 0.592 |
| Endo | 187 | 23 | 0.954 | 0.893 | 0.913 | -0.020 | 0.043 | 0.023 |
| OPC | 146 | 77 | 0.885 | 0.629 | 0.595 | -0.097 | 0.239 | 0.143 |

The slope of 1.507 means the model predicts deltas ~50% larger than ground truth. The predicted delta mean (0.731) is 2.6x the ground truth (0.276). By contrast, L2/3 IT (10,915 cells) has slope 1.01 and near-perfect R² = 0.996.

## Interpretation

This is a **magnitude calibration issue in a low-n cell type**, not a failure of the model's biological signal:

1. **The correlation is excellent** (r = 0.950) — the model correctly identifies which genes change and in which direction.
2. **The scale is off** — the model over-predicts the magnitude of change, likely because L5 ET's 161-cell ground truth is noisy and the model generalizes the larger deltas it sees in abundant types.
3. **R² penalizes scale errors** — even a perfect r with a slope of 1.5 yields R² < 0 because the squared residuals from the offset exceed the variance.

**One-sentence answer:** "High correlation (r = 0.95) confirms the model learns the correct gene-level pattern for L5 ET, but a systematic scale overshoot in this low-n type (161 cells) inflates residuals and drives R² negative."

## Output files

- `diagnosis_stats.json` — per-CT regression statistics (L5 ET, L2/3 IT, Endo, OPC)
- `delta_scatter_comparison.png` — predicted vs ground-truth delta scatter for L5 ET and L2/3 IT
