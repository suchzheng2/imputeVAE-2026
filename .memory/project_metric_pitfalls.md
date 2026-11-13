---
name: Metric pitfalls — mean/median comparisons are misleading
description: Mean and median comparisons hide variance collapse because MSE decoder is optimized to predict conditional mean; use std_ratio, per-cell Spearman, or distribution divergence instead
type: project
---

Mean/median comparisons between raw and decoded are misleading for evaluating per-cell fidelity.

**Why:** The MSE loss optimizes the decoder to predict E[x|z], so the decoded mean will match the raw mean by construction. For sparse/medium genes, raw is bimodal (zero spike + nonzero tail) but decoded is smooth unimodal near the expected value — means match while per-cell values are wrong. Median is worse for sparse genes (raw median = 0, decoded median ≈ 0.15).

**How to apply:** When evaluating model quality, prefer:
- Within-CT per-cell Spearman correlation
- Per-gene std_ratio (decoded/raw)
- Distribution divergence (KL/Wasserstein per gene per CT)
- Nonzero-conditioned metrics
- Min/max and range_ratio for dynamic range clipping

Never use gene correlation or CT-level mean comparisons alone — they look great (0.91, 0.92) while per-cell variation is severely compressed.

Motivated by: Fig 2 distribution examples showing smooth decoded vs bimodal raw.
