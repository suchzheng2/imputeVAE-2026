---
name: delayed_alignment_experiment
description: 2026-03-26 experiment — delayed alignment for conditioned decoder to improve nuclei reconstruction quality
type: project
---

Run 337054 (conditioned decoder, 100 epochs, stage1 cell init) showed good latent alignment but nuclei reconstruction was insufficient (0.82 vs 0.95 for cell). Cross-recon (nuclei→cell head) was good (0.936), meaning decoder capacity exists but nuclei embedding can't recover nuclei-specific info lost to early alignment.

**Why:** Alignment weight (3.0 from epoch 0) pushes nuclei latents cell-like before the nuclei decoder head learns to reconstruct nuclei. The 8-dim modality embedding can't compensate.

**How to apply:** Two delayed-alignment runs submitted 2026-03-26:
- Job 2886168 (Run A): from stage1 cell model, 300 epochs, alignment delayed to epoch 100, ramp to 1.5 over 50 epochs
- Job 2886166 (Run B): from stage2 cell2nuclei model, 300 epochs, same alignment schedule
- No cross-recons, no nuclei_recons_weight boost — isolating the delayed alignment effect
- Eval every 25 epochs to track nuclei recon quality before/after alignment kicks in
- Key metric to watch: nuclei decoder correlation to raw nuclei at epoch 99 (pre-alignment) vs epoch 199+ (post-alignment)
