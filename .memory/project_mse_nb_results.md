---
name: MSE vs NB comparison results (2026-04-02)
description: Stage 2 metric comparison showing MSE dominates NB on all reconstruction metrics; cross-modal reconstruction still fails for both
type: project
---

MSE outperforms filtered NB on all Stage 2 reconstruction metrics (validation): Pearson r 0.947 vs 0.879 (cell), 0.774 vs 0.702 (cross-modal); R² 0.831 vs 0.714; log-MAE 2x lower; Wasserstein ~40% lower. Cross-modal R² is negative for both models — neither truly converts nuclei→cell yet. Nuclei shift negative for both.

**Why:** Softmax bottleneck in NB causes proportion distortion. MSE on log1p avoids this entirely.

**How to apply:** MSE is the stronger baseline. But cross-modal failure means the fundamental issue is alignment/decoder architecture, not just loss function. Next step: conditioned decoder or separate decoder heads to address decoder degradation.
