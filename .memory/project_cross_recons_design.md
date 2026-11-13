---
name: cross-recons design decisions
description: Cross-recons strategy evolution — from shared NB proportion matching to conditioned decoder with direct MSE supervision
type: project
---

Cross-recons is the critical ingredient for making decoded nuclei cell-like (probes confirmed 2026-03-20).

**Original decision (2026-03-23):** Stay on raw counts + NB model, match proportions not scale.

**Updated decision (2026-04-02):** MSE + conditioned decoder cross-recons is the current best approach.

The conditioned decoder enables a cleaner cross-recons than the shared decoder:
- Encode nuclei → z_nuc (detached from encoder grad)
- Decode z_nuc with **cell embedding** → cross-decoded output
- Loss = MSE(cross-decoded, raw_cell) in log1p space

This trains only the decoder + modality embedding, not the encoder. The encoder learns from reconstruction; cross-recons teaches the decoder how to translate nuclei latents into cell-like output using the cell head.

**Why:** Shared decoder cross-recons was indirect (same decoder, different target). Conditioned decoder explicitly switches the decoding head. MSE avoids softmax bottleneck. Detaching mu prevents encoder distortion.

**How to apply:** See `notes/design_decisions.md` for full details and `notes/training_runs.md` for the cond+xr run (job 3106445).
