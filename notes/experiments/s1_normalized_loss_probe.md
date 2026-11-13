# Stage 1 Normalized Loss Probe

**Date:** 2026-05-01 (started), 2026-05-08 (results + stage 2 launched)
**Goal:** Make stage 1 behave like a sensible vanilla VAE — JSD should actually affect the latent space.
**Context:** Meeting with Prashant 2026-05-01 — agreed to go back to stage 1 fundamentals.

## Motivation

Previous runs show JSD has negligible effect:
- Config B stage 1 (old, 600ep): final JSD ~0.05 with beta=0.5 annealed (starts at epoch 300)
- PerCT stage 1: JSD saturates at ln(2)=0.693 (complete non-overlap → zero gradient)
- The raw scale of recon (~0.3-1.0) vs JSD×beta (~0.05×0.1=0.005) means JSD is irrelevant
- Old Config B stage 1 had dim 5 log_var collapsed to -20 (deterministic), indicating unhealthy latent

## Approach

1. **Normalize both losses to [0,1]:**
   - Recon: divide by epoch-0 value (auto-computed after epoch 0)
   - JSD: divide by ln(2)=0.693 (its theoretical maximum)
   - Epoch 0 trains WITHOUT normalization (calibration pass), normalization kicks in from epoch 1+
   - `--clamp_recons_loss False` — old clamp at 1.0 interferes with normalization

2. **No KL annealing** — JSD active from epoch 0 so we see immediate effect

3. **Three probes:** beta=0.1 (no anneal), beta=1.0 (no anneal), beta=1.0 (anneal over 50ep)

4. **Diagnostic:** log_var vs sparsity to check if extreme values map to zeros

## Code Changes

- `imputeVAE/models.py`: `VAE_JSD.loss_function()` — added normalized loss branch (LN2 class constant)
- `imputeVAE/training.py`: auto-compute `recon_loss_norm` from epoch 0, print scaled values
- `imputeVAE/cond_training.py`: auto-compute `recon_loss_norm` from epoch 0 (cell recon) for stage 2
- `imputeVAE/staged_training.py`: same auto-compute for shared-encoder stage 3
- `train.py`: new args `--normalize_losses` (bool), `--recon_loss_norm` (float, 0=auto)
- `analysis/logvar_sparsity_diagnostic.py`: per-cell sparsity vs log_var correlation analysis

---

## Stage 1 Probes

### Jobs Submitted (2026-05-01)

| Job ID | Script | Config | Run Folder |
|--------|--------|--------|------------|
| 4153808 | `run_s1_normloss_beta01.sh` | beta=0.1, no anneal | `run/run_2026-05-01-12-01-55_1984590` |
| 4153809 | `run_s1_normloss_beta1.sh` | beta=1.0, no anneal | `run/run_2026-05-01-13-44-25_1993401` |
| 4153810 | `run_s1_normloss_beta1_anneal.sh` | beta=1.0, anneal 0→1 over 50ep | `run/run_2026-05-01-14-12-35_2525266` |
| 4153811 | `run_logvar_sparsity_diag.sh` | Diagnostic (OOM'd) | — |

Common params: Config B (10,134 genes), log1p, vae_js, internal [1024,512], latent 32, lr=1e-4, batch 128, 20% subset, `--clamp_recons_loss False`, `--normalize_losses True`, 200 epochs

### Auto-Computed Normalization Constants (epoch 0 recon loss)

| Job | recon_loss_norm |
|-----|----------------|
| 4153808 (beta=0.1) | 0.6995 |
| 4153809 (beta=1.0) | 0.7617 |
| 4153810 (beta=1.0 anneal) | 0.6455 |

### Results

| Job | Config | E199 recon | E199 JSD | E199 scaled recon | E199 JSD/ln2 | mu range | log_var range |
|-----|--------|-----------|---------|-------------------|-------------|----------|--------------|
| **4153808** | **beta=0.1 no anneal** | **0.251** | **0.080** | **0.359** | **0.115** | [-1, +1] | [-1.2, +0.6] |
| 4153809 | beta=1.0 no anneal | 0.312 | 0.014 | 0.410 | 0.020 | [-0.3, +0.5] | [-0.2, +0.1] |
| 4153810 | beta=1.0 anneal | 0.283 | 0.030 | 0.438 | 0.043 | — | — |

Convergence trajectory (beta=0.1 winner):
- E0→E10: recon drops 0.700→0.323 (fast initial learning)
- E50→E100: 0.294→0.267 (4.0% over 50 epochs)
- E100→E150: 0.267→0.256 (4.0%)
- E150→E199: 0.256→0.251 (2.0% — converging)
- Already better than old 600-epoch Config B stage 1 (recon=0.257)

### Key Findings

1. **Winner: beta=0.1 no anneal** — best reconstruction (0.251), best latent structure (clear cell-type separation in UMAP), healthy mu/log_var ranges.

2. **Higher beta over-regularizes.** beta=1.0 pushes mu→0, log_var→0 immediately, collapsing q(z|x) to N(0,1). This minimizes JSD trivially but destroys informative latent structure. Recon suffers (0.312 vs 0.251, +24%).

3. **JSD-to-prior is inherently easy to satisfy** from the start. An untrained network's latent is already close to N(0,1). The model never needs to "work hard" on JSD — the only effect of raising beta is suppressing the informative variation.

4. **The normalization works** — both losses are on comparable [0,1] scale — but the fundamental problem is that JSD-to-prior provides no useful learning signal in stage 1. It's not a scaling issue, it's that matching a standard normal prior is too easy.

5. **Latent is healthy** — no dimensions collapsed to -20 (unlike old Config B stage 1 dim 5). mu in [-1, +1], log_var in [-1.2, +0.6]. The model with `--clamp_recons_loss False` and `--normalize_losses True` produces a much better-behaved latent.

### Log_var Diagnostic (Old Config B Stage 1)

Analyzed epoch 599 dump from `run_2026-04-01-09-29-26_3250320` (job 4513202):

- **Dim 5 collapsed to -20** — mean log_var = -9.99, half the cells hit the -20 clamp
- **Dim 1 partially collapsed** — min = -15.71, mean = -1.34
- **Other 28 dims healthy** — log_var in [-1, +1]
- Only 46/275136 elements (0.01%) actually hit -20 clamp, but concentrated in dim 5
- Rare cell types (Endo, L5 ET, Macrophage, n=30) had least negative log_var
- The new stage 1 model (4153808) does NOT have this problem

---

## Stage 2 Follow-up (2026-05-08)

**Decision:** Stage 1 winner (0.251 recon at 200ep) is already better than old 600-epoch model (0.257). No need to train longer. Proceed to stage 2.

### Jobs Submitted

| Job ID | Script | Config |
|--------|--------|--------|
| 4514645 | `jobs/stage2/run_s2_cond_normloss.sh` | Cond decoder + `--normalize_losses True` |
| 4514646 | `jobs/stage2/run_s2_cond_nonorm.sh` | Cond decoder + NO normalization |

Common params:
- Init from: `run/run_2026-05-01-12-01-55_1984590/models/model.pt` (stage 1 winner)
- Config B (10,134 genes), log1p, vae_js, embed_dim=8
- 600 epochs, batch 128, lr=1e-4, 20% subset
- `--clamp_recons_loss False` (both jobs)
- beta=0.1 (same as stage 1)

Loss schedule (cascading anneal, same as best previous cond runs):
- cell_recons (1.0) — ON from epoch 0
- cell_jsd (0.1) — ON from epoch 0
- nuclei_recons (1.0) — anneals 0→1.0 over epochs 0-100
- nuclei_jsd (0.1) — delayed to epoch 100, anneals over epochs 100-200
- alignment_jsd (1.0) — delayed to epoch 200, anneals over epochs 200-300
- cross_recons — disabled (0.0)

For normalized losses: `recon_loss_norm` auto-computed from epoch 0 cell recon loss.

### Normloss Stage 2 Results (job 4514645, COMPLETED 4h 34m)

**Run folder:** `run/run_2026-05-08-00-23-35_3604390`
**recon_loss_norm auto-set:** 0.253344 (from epoch 0 cell recon)

Cross-imputation metrics (final epoch 599, validation set):
| Metric | Value |
|--------|-------|
| cross_dec gene corr vs raw_cell | 0.954 |
| cross_dec shift | +0.195 |
| cross_dec R² vs raw_cell | 0.768 |
| 5x5 corr: cross_dec vs raw_cell | 0.989 |
| nuclei_recons | 0.223 |
| cell_recons | 0.256 |

UMAPs: Clean cell-type clusters, good modality mixing after alignment anneal (epoch 200+).

### No-Norm Stage 2 (job 4514646, RUNNING as of 2026-05-08)

**Run folder:** `run/run_2026-05-08-05-21-27_3694365`
At epoch 169/600. Pending completion for A/B comparison.

---

## Within-Cell-Type Variance Analysis (2026-05-08)

**Critical finding: variance collapse is NOT solved.**

The total-variance ratio (cross_dec/baseline across all cells) is ~0.94, but this is misleading — it's dominated by between-cell-type variance (which is easy to capture). The real test is within-cell-type per-gene variance.

### Results

| Source | Mean within-CT per-gene var |
|--------|----------------------------:|
| Baseline cell (S1 decoded, e0) | 0.0864 |
| Baseline nuc (S1 decoded, e0) | 0.0177 |
| Normloss decoded cell (e599) | 0.0885 |
| Normloss decoded nuc (e599) | 0.0195 |
| **Normloss cross_dec (e599)** | **0.0393** |
| Old cond decoded cell (e599) | 0.1011 |
| Old cond decoded nuc (e599) | 0.0189 |
| **Old cond cross_dec (e599)** | **0.0352** |

### Variance Ratios (cross_dec / baseline cell)

| Model | Overall | Best CT | Worst CT |
|-------|--------:|---------|----------|
| **Normloss** | **0.454** | OPC (1.09), Endo (3.21) | Macrophage (0.04), Vip (0.27) |
| **Old cond** | **0.407** | OPC (1.14), Endo (3.18) | Macrophage (0.06), Astro (0.11) |

Per-cell-type breakdown:

| Cell Type | n | Baseline cell var | Normloss xd var | Normloss ratio | Old cond ratio |
|-----------|---|------------------:|----------------:|---------------:|---------------:|
| L5 ET | 30 | 0.1894 | 0.0540 | 0.285 | 0.157 |
| Sncg | 30 | 0.1795 | 0.0488 | 0.272 | 0.207 |
| Pvalb | 44 | 0.1468 | 0.0540 | 0.368 | 0.303 |
| Endo | 14 | 0.0340 | 0.1093 | 3.213 | 3.183 |
| L5 IT | 1391 | 0.1182 | 0.0578 | 0.489 | 0.419 |
| L2/3 IT | 964 | 0.1077 | 0.0345 | 0.321 | 0.289 |
| Sst | 202 | 0.1064 | 0.0369 | 0.347 | 0.314 |
| L6b | 66 | 0.0954 | 0.0430 | 0.450 | 0.452 |
| Lamp5 | 114 | 0.0785 | 0.0266 | 0.339 | 0.320 |
| L6 CT | 955 | 0.0705 | 0.0302 | 0.428 | 0.411 |
| L6 IT | 261 | 0.0663 | 0.0416 | 0.627 | 0.574 |
| L5/6 NP | 180 | 0.0665 | 0.0312 | 0.469 | 0.526 |
| Oligo | 30 | 0.0570 | 0.0174 | 0.305 | 0.362 |
| Vip | 131 | 0.0571 | 0.0152 | 0.266 | 0.183 |
| OPC | 30 | 0.0521 | 0.0569 | 1.091 | 1.144 |
| Astro | 30 | 0.0285 | 0.0095 | 0.332 | 0.111 |
| Macrophage | 30 | 0.0153 | 0.0006 | 0.036 | 0.057 |

### Interpretation

1. **Same-modality reconstruction preserves variance perfectly** (ratio ~1.0). The problem is specific to cross-imputation.

2. **~55% of within-CT variance is lost** in cross-imputation for both models. The normloss model is marginally better (45% retained vs 41%).

3. **Cross_dec at epoch 0 already has low within-CT var** (0.013, only 15% of baseline cell). Stage 2 training *improves* it from 15% → 45%, so the alignment training is actually helping, not hurting.

4. **Cell types with few samples** (Macrophage n=30, Astro n=30) tend to have worst variance retention. Exception: OPC (n=30) retains well.

5. **Endo anomaly:** Endo has ratio >3 — likely because it only has 14 samples and the cross-decoded variance is inflated by the small sample size / modality embedding noise.

6. **The variance collapse is in the cross-decode operation itself**, not in the training. This suggests the modality embedding doesn't carry enough information to translate per-cell variation, only the cell-type-level mean shift.

### Plots

See `notes/experiments/s1_normalized_loss_probe/`:
- `within_ct_variance_absolute.png` — bar chart of absolute variance by cell type
- `within_ct_variance_ratio.png` — ratio (cross_dec / baseline cell) by cell type
- `within_ct_variance_scatter.png` — scatter: baseline vs cross_dec per cell type
