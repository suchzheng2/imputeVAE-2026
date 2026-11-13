---
name: prashant-meeting-2026-09-24
description: Sep 2026 Prashant meeting — TRAP validation promising but fix cell-type mapping (drop L4, ET/IT by projection evidence); variance check must show per-gene variance distributions + Spearman(var_real, var_imputed) per CT
metadata:
  type: project
---

Meeting summarized 2026-09-24 (exact date week of 2026-09-21). Full notes: `notes/meeting_2026-09-24_prashant.md`.

**TRAP/RiboTag external validation:** Prashant called it "already very promising" (imputed ≈ real cell, ~0.75 Spearman; GSE172484 M1 L5b is the best dataset). But:
- Do NOT map external populations to our cell types by name. Drop the L4 (Scnn1a, GSE133291) comparison entirely: motor cortex has no L4.
- For L5a/L5b populations, decide ET vs IT from projection evidence in the source papers (ET = extratelencephalic, projects subcortically; IT = intratelencephalic), or report against all L5 types. Marker-gene matching is the ideal.
- Cortex-only comparisons; hippocampus not comparable. Matching must not be chosen by the reported metric (circularity).

**Variance:** Median variance ratio is not enough. He wants, per well-represented CT (L5 IT, Oligo first): the distribution of per-gene within-CT variances (real vs imputed vs nucleus) and Spearman(var_real_cell, var_imputed) per gene. Same shape at lower scale = OK (denoising); uniformly tight = imputation "cheating" by assigning near-constant values.

**Why:** These are the two gates before the TRAP result goes into a paper/slide and before any variance-loss decision.

**Status 2026-09-24 (both done):** (1) L5 mapping audit: `validation/trap_ribotag_external_validation/outputs/extension/metadata/l5_mapping_evidence.md` + `scripts/17_l5_mapping_audit.py`. Literature + real-cell markers agree: Gprin3/Colgalt2/Glt25d2 = ET, S100a10 = IT, Rbp4 = IT+ET pool, Scnn1a_L4 excluded. H2 changes <= 0.03 by L5 type. Rerun of extension H2/H3 = SLURM 11168301. (2) Variance rank: `notes/experiments/variance_zero_nonzero_fullmodel/variance_rank/README.md`; Spearman var(real) vs var(imputed) 0.92 L5 IT, 0.93 Oligo, 0.85 pooled; distribution stretched (IQR ratio 1.5-3.6), not collapsed; mean-adjusted rank agreement weaker (0.55 L5 IT, 0.23 Oligo).

**How to apply:** For the next meeting show `l5_mapping_evidence.md` table and `variance_rank/figures/V1`. Open: confirm rerun 11168301 numbers in the READMEs. Next meeting Friday of the following week (likely 2026-10-02). See [[project_trap_ribotag_external]], [[project_variance_zero_nonzero_fullmodel]], [[user_prashant]].
