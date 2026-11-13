# Phase 1: Larger Latent Space

**STATUS: SKIPPED** — latent diagnostic (2026-04-07) invalidated the bottleneck hypothesis. See `../latent_diagnostic/findings.md`.

## Original Rationale (INVALIDATED)

Current 32-dim latent: ~80% (26 dims) encodes cell-type identity, leaving ~6 dims for per-cell variation. Doubling or quadrupling latent size gives proportionally more room for within-CT signal.

## Why Skip

Diagnostic (job 3260550) showed:
- **82% of latent variance is within-CT** (not 20% as assumed) — the encoder already dedicates most capacity to per-cell variation
- 28/32 dims active — the model has spare capacity it doesn't use
- 4 dead dims are explained by inter-modality encoding (3 of 4 have >44% inter-modality variance), not capacity exhaustion
- Posterior std is healthy (0.69-0.82), no collapse
- **The problem is in the decoder**, not the encoder/latent — decoder collapses within-CT variation to the conditional mean despite receiving rich per-cell z signal

## Pre-requisite: Latent Diagnostic (2026-04-07)

**Before running larger-latent experiments**, we need to understand what the current 32-dim latent is actually doing. Without this, we risk:
- Adding dims the encoder won't use (if the training signal doesn't reward encoding per-cell info)
- Not co-adjusting interacting hyperparameters (beta, embed_dim, hidden layers)
- Misattributing the problem to capacity when it may be in the loss or decoder

**Diagnostic job:** 3257234 (analysis/latent_diagnostic.py on pct5v2_cond)
**Output:** `notes/experiments/SUC-6_variance_retention/latent_diagnostic/`

### What the diagnostic measures:
1. **Per-dim total variance** — are any dims collapsed/dead (var ≈ 0)?
2. **Per-dim between-CT vs within-CT variance** — what fraction of each dim encodes cell-type vs per-cell info?
3. **Posterior std (from log_var)** — is the encoder posterior collapsed (std → 0), indicating JSD/KL is crushing the latent?
4. **# dims for 90% within-CT variance** — how concentrated is per-cell info?
5. **Cross-modality comparison** — does the encoder treat cell and nuclei latents similarly?

### Decision matrix based on diagnostic results:

| Finding | Implication | Action |
|---------|-------------|--------|
| Many dead dims (var ≈ 0) | Encoder not using capacity → more dims won't help | Fix training signal first (Phase 2/3), not latent size |
| All dims saturated, >90% between-CT | Genuine bottleneck | Proceed with larger latent |
| Posterior std ≈ 0 across dims | JSD over-regularizing | Lower beta or adjust annealing before changing latent |
| Within-CT variance exists in latent but decoder ignores it | Problem is in decoding, not encoding | Skip to Phase 3 (residual recons) |
| embed_dim (8) too small relative to larger latent | Modality signal diluted at 128-dim (6% vs current 20%) | Scale embed_dim with latent (e.g. 16 for lat-64, 32 for lat-128) |

### Parameters that interact with latent_size:

| Parameter | Current | Why it interacts |
|-----------|---------|-----------------|
| **beta** (JSD weight) | 0.1 | JSD pressure scales with latent dim — same beta on 128 dims may over-regularize |
| **embed_dim** | 8 | Currently 20% of decoder input (8/40). At lat-128: only 6% (8/136) — modality signal diluted |
| **num_internal** | [1024, 512] | Decoder first layer: 512→40 input. At lat-128: 512→136 input, may need wider layers |
| **learning_rate** | 0.0001 | More parameters → may need adjustment |
| **KL annealing schedule** | recons-only 300ep, JSD 300-400 | Larger latent may need longer annealing |

## Experiments

Pending diagnostic results. Preliminary plan:

| Experiment | latent_size | num_latent | Hidden layers | embed_dim | beta | Notes |
|------------|-------------|------------|---------------|-----------|------|-------|
| Baseline | 5 | 32 | [1024, 512] | 8 | 0.1 | Current |
| Lat-64 | 6 | 64 | [1024, 512] | 16 | 0.1 | 2x latent, scaled embed |
| Lat-128 | 7 | 128 | [1024, 512] | 16 | 0.05 | 4x latent, lower beta |

**Note:** exact configs depend on diagnostic results. If posterior is collapsed, beta sweep is higher priority than latent size.

## Code Changes

None for latent_size — `--latent_size` already accepts the exponent.
`--embed_dim` already exists. May need to verify beta interacts correctly at different latent sizes.

## Job Scripts

- TODO: create after diagnostic results inform the design

## Evaluation

Run `analysis/hallucination_test.py` + `analysis/latent_diagnostic.py` on each completed model. Compare:
- Variance ratio, within-CT Spearman, z ablation
- Per-dim variance profile (to confirm extra dims are being used)
- Reconstruction quality (gene corr, classification)

## Expected Outcome

Depends on diagnostic. If genuine bottleneck: modest improvement (11% → 20-30%). If posterior collapse or dead dims: no improvement until root cause fixed.

## Risk

- Larger latent may need more hidden layer capacity. Monitor reconstruction quality — if it degrades, try [1024, 512, 256] hidden layers.
- Changing multiple params simultaneously makes attribution harder. Consider: lat-64 with only embed_dim scaled first, then add beta adjustment if needed.
