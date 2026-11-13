# Conditioning Variants + CPM Normloss Experiment

**Date:** 2026-05-10
**Goal:** Test CPM normalization under the normalized-loss framework, and compare three conditioning approaches for the modality-conditioned decoder/encoder.

## Background

### CPM + Normloss fixes JSD failure
Previous CPM experiments (2026-04-16) failed because CPM data range (0-14) caused encoder mu to explode (~500), making JSD saturate at ln(2) with zero gradient. The normalized loss framework (recon/epoch0 + JSD/ln2) fixes this without needing soft constraints or mu clamping.

- CPM normloss stage 1 (job 4521167): recon_loss_norm=5.375, final JSD=0.064 (healthy)
- Log1p normloss stage 1 (job 4153808): recon_loss_norm=0.700, final JSD=0.080

### Three conditioning approaches
1. **Learned embedding (baseline):** `nn.Embedding(2, 8)` — decoder only. Current best approach.
2. **Scalar conditioning (traditional cVAE):** Fixed scalar (1=cell, 0=nuclei) concatenated to latent before decoder. No learned parameters for the modality signal.
3. **Full cVAE (encoder + decoder):** Same scalar conditioning applied to BOTH encoder input and decoder input. Encoder first layer expanded from input_size to input_size+1.

## Code Changes

### `models_cond.py`
- `VAE_JSD_Cond`: added `scalar_cond` and `cond_encoder` flags
  - `scalar_cond=True`: embed_dim=1, no nn.Embedding, `_get_modality_emb()` returns fixed 1.0/0.0
  - `cond_encoder=True`: encoder rebuilt with `input_size + embed_dim` first layer, modality signal concatenated to input in `forward()` and `decode_cross()`
  - `load_pretrained_partial()`: handles both encoder.0.weight and decoder.0.weight expansion (zero-init conditioning columns)
- `VAE_NB_Cond`: added `scalar_cond` (no `cond_encoder` yet — only used with vae_js currently)

### `train.py`
- `--scalar_cond True/False` (default False)
- `--cond_encoder True/False` (default False)

### `cond_training.py`
- Passes `scalar_cond` and `cond_encoder` to model constructor

## Loss Schedule Rationale

### Decoder-only conditioning (learned embed + scalar cond)
Same cascading anneal as previous best runs:
- cell_recons (1.0) — ON from epoch 0
- cell_jsd (0.1) — ON from epoch 0
- nuclei_recons (1.0) — anneals 0→1.0 over epochs 0-100
- nuclei_jsd (0.1) — delayed to epoch 100, anneals over 100-200
- alignment_jsd (1.0) — delayed to epoch 200, anneals over 200-300

Rationale: encoder is modality-blind (trained on cells only in S1), needs gradual nuclei introduction.

### Full cVAE (encoder + decoder conditioning)
All reconstruction and JSD terms ON from epoch 0, only alignment annealed:
- cell_recons (1.0) — ON from epoch 0
- cell_jsd (0.1) — ON from epoch 0
- nuclei_recons (1.0) — ON from epoch 0
- nuclei_jsd (0.1) — ON from epoch 0
- alignment_jsd (1.0) — anneals 0→1.0 over epochs 50-100

Rationale: the encoder receives an explicit modality signal, so it knows from the start whether it's processing cell or nuclei data. No need for gradual introduction. Alignment still benefits from a short delay (50 epochs) so each modality establishes its latent representation before forcing overlap.

## Stage 1 Models (shared across conditions)

| Normalization | Run Folder | Job | recon_loss_norm | Final recon | Final JSD |
|---------------|-----------|-----|-----------------|-------------|-----------|
| log1p | `run/run_2026-05-01-12-01-55_1984590` | 4153808 | 0.700 | 0.251 | 0.080 |
| CPM | `run/run_2026-05-08-13-22-30_3907651` | 4521167 | 5.375 | 2.167 | 0.064 |

## Stage 2 Jobs

| Job | Name | Norm | Conditioning | Anneal | Script |
|-----|------|------|-------------|--------|--------|
| 4514645 | s2_cond_nl | log1p | learned embed 8-dim, dec-only | cascading | `jobs/stage2/run_s2_cond_normloss.sh` |
| 4598108 | s2_cpm_nl | CPM | learned embed 8-dim, dec-only | cascading | `jobs/stage2/run_s2_cond_cpm_normloss.sh` |
| 4598109 | s2_cpm_xr | CPM | learned embed 8-dim + cross-recons, dec-only | cascading | `jobs/stage2/run_s2_cond_xr_cpm_normloss.sh` |
| 4598175 | s2_sc_nl | log1p | scalar (1/0), dec-only | cascading | `jobs/stage2/run_s2_scalar_cond_normloss.sh` |
| 4598176 | s2_sc_cpm | CPM | scalar (1/0), dec-only | cascading | `jobs/stage2/run_s2_scalar_cond_cpm_normloss.sh` |
| 4598336 | s2_fcvae | log1p | scalar (1/0), enc+dec | all-on + align@50 | `jobs/stage2/run_s2_full_cvae_normloss.sh` |
| 4598337 | s2_fcvae_c | CPM | scalar (1/0), enc+dec | all-on + align@50 | `jobs/stage2/run_s2_full_cvae_cpm_normloss.sh` |

Common params: Config B (10,134 genes), vae_js, internal [1024,512], latent 32, lr=1e-4, batch 128, 20% subset, beta=0.1, 600 epochs, `--normalize_losses True`, `--clamp_recons_loss False`.

## Reference Baselines

| Model | Gene corr | Shift | Within-CT var ratio |
|-------|-----------|-------|---------------------|
| log1p normloss learned embed (4514645) | 0.954 | +0.195 | 0.454 |
| log1p no-norm learned embed (4514646) | pending | pending | pending |
| Old cond (pre-normloss) | ~0.963 | — | 0.407 |

## Single-Stage cVAE (added 2026-05-11)

The two-stage approach (S1 on cells, S2 joint) is a legacy of the modality-blind encoder design.
With full cVAE (encoder + decoder conditioned), the model has the modality signal from the start —
no reason to pretrain on one modality first.

### Setup
- Train from scratch on both cell and nuclei simultaneously
- No `--stage1_model_run` (no pretrained weights)
- ALL losses ON from epoch 0, including alignment (no annealing at all)
- Otherwise identical params to the two-stage full cVAE runs

### Code Change
- `cond_training.py`: `stage1_model_run` now optional. When omitted, skips `load_pretrained_partial`
  and prints "Training from scratch".

### Jobs

| Job | Name | Norm | Script |
|-----|------|------|--------|
| 4607333 | ss_cvae | log1p | `jobs/stage2/run_single_stage_cvae_normloss.sh` |
| 4607334 | ss_cvae_c | CPM | `jobs/stage2/run_single_stage_cvae_cpm_normloss.sh` |

## What to Compare

1. **CPM vs log1p**: Does CPM normalization improve gene correlation or variance retention?
2. **Scalar vs learned embed**: Does a simpler 1-dim signal work as well as 8-dim learned embedding?
3. **Full cVAE vs decoder-only**: Does encoder conditioning help? Especially for variance retention.
4. **Anneal schedule**: Does the full cVAE's simpler schedule converge faster?
5. **Single-stage vs two-stage**: Does training from scratch match or beat pretrained initialization?

## Results (updated 2026-05-12)

### All completed runs (600 epochs, validation set)

| Job | Config | Gene corr | Shift | 5x5 xd↔cell | R² |
|-----|--------|-----------|-------|-------------|-----|
| 4514645 | log1p, learned emb 8d, dec-only | **0.954** | +0.195 | **0.989** | 0.768 |
| 4607333 | **log1p, single-stage cVAE** | 0.932 | **+0.211** | — | **0.792** |
| 4598336 | log1p, scalar, enc+dec (2-stage) | 0.919 | +0.209 | 0.988 | 0.494 |
| 4598175 | log1p, scalar, dec-only | 0.902 | +0.216 | 0.977 | 0.097 |
| 4598108 | CPM, learned emb 8d, dec-only | 0.888 | +0.190 | 0.966 | 0.475 |
| 4598109 | CPM, learned emb 8d + xr, dec-only | 0.783 | +0.150 | — | 0.314 |
| 4598176 | CPM, scalar, dec-only | 0.796 | **-0.175** | — | -0.855 |

### Still running / pending (as of 2026-05-12)

| Job | Config | Status | Notes |
|-----|--------|--------|-------|
| 4598337 | CPM, scalar, enc+dec (2-stage) | RUNNING (restarted after preemption) | Was at e181 before preemption |
| 4607334 | CPM, single-stage cVAE | PENDING (preempted 3x on scavenge_gpu) | Reached e347 before last preemption |

### Key Findings

**1. log1p dominates CPM across all conditioning variants.**
- Best log1p (learned emb): gene corr 0.954, R² 0.768
- Best CPM (learned emb): gene corr 0.888, R² 0.475
- CPM + scalar dec-only is a complete failure (negative shift = imputation goes wrong direction)
- CPM + cross-recons (xr) also underperforms (gene corr 0.783)

**2. Single-stage cVAE is the best new architecture.**
- Highest R² (0.792 valid) and highest shift (+0.211)
- Gene corr 0.932 slightly below learned emb baseline (0.954)
- Trains from scratch — no need for stage 1 pretrain
- Simplest setup: all losses ON from epoch 0, no annealing

**3. Two-stage full cVAE underperforms single-stage.**
- 2-stage (4598336): gene corr 0.919, R² 0.494
- Single-stage (4607333): gene corr 0.932, R² 0.792
- Pretrained weights from modality-blind stage 1 may actually hurt

**4. Scalar conditioning works but learned embedding is better for gene corr.**
- Learned 8-dim emb: 0.954 gene corr
- Scalar dec-only: 0.902 gene corr
- Scalar enc+dec: 0.919 (2-stage), 0.932 (single-stage)
- The extra degrees of freedom in learned embedding help fine-grained gene reconstruction

**5. Encoder conditioning helps substantially.**
- Dec-only scalar: R² 0.097
- Enc+dec scalar (2-stage): R² 0.494
- Enc+dec scalar (single-stage): R² 0.792
- Encoder knowing modality allows it to separate modality-specific from shared variation

**6. Cross-recons (xr) hurts CPM.** CPM + xr (0.783) is worse than CPM without xr (0.888). The decoder already struggles with CPM's larger data range; forcing cross-reconstruction adds instability.
