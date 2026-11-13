# CPM + log1p Experiment (2026-04-16)

## Motivation
Library size analysis (2026-04-15) showed decoder dedicates latent dim 31 to library size
and gets most "variance" from scaling CT means by library size. After proportion normalization,
true per-cell std_ratio drops from 0.55 to 0.17.

CPM normalization removes library size *before* the model sees the data, forcing the model
to learn actual gene-specific variation instead of the library size shortcut.

Agreed with Prashant (2026-04-15 meeting) to try CPM + log1p as next experiment.

## Plan
1. **Stage 1**: Cell-only pretraining, CPM + log1p, PerCT_5pct (16,439 genes)
   - Script: `jobs/stage1/run_s1_cell_js_perct5_cpm.sh`
   - Same hyperparams as pct5v2 (beta=0.1, slow anneal)
2. **Stage 2 cond**: Conditioned decoder, no cross-recons
   - Script: `jobs/stage2/run_s2_cond_js_perct5_cpm.sh`
   - Updated to match v2 JSD weights (0.1, not 0.5)
3. **Stage 2 cond+xr**: Conditioned decoder + cross-recons
   - Script: `jobs/stage2/run_s2_cond_xr_js_perct5_cpm.sh`
4. **Eval**: Standard eval framework, compare with log1p models post-hoc

## Key difference from log1p runs
- `--raw_data_normalization cpm` instead of `log1p`
- `--clamp_recons_loss False` (CPM MSE > 1, clamp would distort)
- Everything else identical to pct5v2

## Stage 1 Results

- **First attempt**: job 3674870 (failed/superseded)
- **Second attempt**: job 3964307 — COMPLETED (4h 29m, 600 epochs)
- **Run folder**: `run/run_2026-04-23-23-59-34_1163351`

| Epoch | Train Recons | Train JSD | Valid Recons | Valid JSD |
|-------|-------------|-----------|-------------|-----------|
| 0     | 3.94        | 0.0       | —           | —         |
| 100   | 1.58        | 0.0       | 0.277       | 0.0       |
| 200   | 1.53        | 0.0       | 0.282       | 0.0       |
| 300   | 1.49        | 0.0       | 0.277       | 0.0       |
| 400   | 1.55        | 0.693     | 0.279       | 0.120     |
| 500   | 1.59        | 0.693     | 0.277       | 0.120     |
| 600   | 1.67        | 0.693     | 0.281       | 0.120     |

**JSD stuck at ln(2) = 0.693** (maximum value). Same behavior as log1p pct5v2 stage 1
(which also has JSD stuck at 0.693, recons clamped to 1.0). This is a shared issue:
beta=0.1 is too small relative to reconstruction loss for JSD to have any effect in stage 1.
Not CPM-specific — log1p model worked fine in stage 2 despite this.

Considered normalizing recons by feature dimension (`/num_genes`), but MSELoss already
uses `reduction='mean'` (divides by B*G). Extra `/G` over-corrects (recons → 0.00009,
JSD would dominate completely). Decided to proceed to stage 2 as-is.

## Stage 2 (in progress)

Submitted 2026-04-28:
- **cond** (no cross-recons): job 4061786
- **cond+xr** (cross-recons): job 4061787
- Both use `--clamp_recons_loss False`
- Stage 1 model: `run/run_2026-04-23-23-59-34_1163351/models/model.pt`

## Eval framework

CPM models registered in `analysis/eval/registry.py`:
- `cpm_pct5_cond` — run_folder TBD (job 4061786)
- `cpm_pct5_cond_xr` — run_folder TBD (job 4061787)

Data loader updated to support `normalization='cpm'` parameter (CPM + log1p).
Cache key now includes normalization to avoid cross-contamination.

After jobs complete: grep run folder from logs, fill in registry, run eval.

## Stage 2 Results

### Job status
- **cond** (job 4061786): COMPLETED (8h 53m), run `run/run_2026-04-28-22-16-14_990383`
- **cond+xr** (job 4061787): CANCELLED (scavenge preemption at ~1h, no epochs completed)

### CPM cond inline eval (train set, cross-decode nuclei→cell)

| Epoch | Shift | Gene Corr | Log MAE | R² |
|-------|-------|-----------|---------|-----|
| 0 | +0.040 | 0.478 | 0.297 | -1.48 |
| 100 | -0.119 | 0.724 | 0.225 | -0.73 |
| 350 | -0.097 | 0.710 | 0.214 | -0.62 |
| 599 | -0.031 | 0.703 | 0.232 | -0.73 |

**Gene corr 0.703 vs log1p cond baseline 0.963 — substantially worse.**

### Critical finding: JSD stuck at ln(2) throughout all 600 epochs

All three JSD terms (nuclei, cell, alignment) remained at exactly **0.693147** from epoch 0
to epoch 600. The model trained reconstruction but the latent space has no structure.

| Epoch | nuclei_JSD | cell_JSD | alignment_JSD |
|-------|-----------|----------|---------------|
| 0 | 0.6931 | 0.6931 | 0.6931 |
| 100 | 0.6931 | 0.6931 | 0.6931 |
| 600 | 0.6931 | 0.6931 | 0.6931 |

Compare with log1p S2 cond which started at JSD 0.117-0.255 and trained down to 0.02-0.04.

---

## Diagnosis: Why JSD is stuck (2026-04-29)

### Symptoms
- JSD = ln(2) = 0.693147 (maximum possible value), never moves
- Reconstruction trains normally (loss decreases from ~2.8 to ~2.2)
- Same stage 1 architecture and hyperparams as log1p — only normalization changed

### Root cause: CPM input scale → mu too large → JSD gradient = 0

**Chain of causation:**

1. **CPM data range is ~3x larger than log1p**
   - log1p(raw_counts): typical range 0–5, mean nonzero ~1.5
   - log1p(CPM): typical range 0–14, mean nonzero ~7
   - Both apply log1p, but CPM first scales by ×10⁶ → larger pre-log values

2. **Encoder activations scale with input**
   - Encoder is Linear+LeakyReLU stacked — essentially linear in input magnitude
   - CPM input ~3x larger → encoder output h ~3x larger
   - mu = W_mu @ h + b_mu → mu ~3x larger

3. **Probed the trained CPM S1 model with real data:**
   - mu range: **[-513, +457]**, log_var clamped to ±20 (raw values 100-200+)
   - For comparison, log1p S2 cond: mu range [-16, +26], log_var mean -1.9

4. **JSD MC estimator saturates at ln(2) when q and p don't overlap**
   - q(z|x) = N(mu≈50, σ≈exp(10)=22026) — posterior centered far from origin
   - p(z) = N(0,1) — standard normal prior
   - P(prior assigns any mass near mu=50) ~ exp(-50²/2) ≈ 10⁻⁵⁴⁴ — below float32
   - MC estimator samples z_q ~ q, z_p ~ p; they never overlap → JSD = ln(2)

5. **Gradient of JSD w.r.t. mu is zero when no overlap**
   ```
   jsd_q = 0.5 * E_{z~q}[log q(z) - log m(z)]
   When p(z) ≈ 0:  log m(z) ≈ log q(z) - log 2
   => jsd_q ≈ 0.5 * log(2)  — a constant, d/d(mu) = 0
   ```

6. **Stage 1 JSD was already stuck** — but for log1p this was survivable because
   stage 2 started with mu small enough that JSD gradient was nonzero (mu ~5-10 range).
   For CPM, stage 2 inherits mu ~50-500 → immediately stuck from epoch 0.

### Why increasing beta would NOT help
beta multiplies the JSD gradient. When gradient = 0: beta × 0 = 0. No effect.

### Why clamping mu would NOT help
Clamping mu in the loss function: d(clamped_mu)/d(raw_mu) = 0 outside clamp range.
The gradient chain is broken — encoder weights receive no signal.
Clamping in the forward pass has the same problem (reconstruction gradients also killed).

### Key comparison: log_var weight column sums
After stage 2, the log_var weights show whether JSD had any effect:
- log1p S2 cond: log_var weight ||W|| = **2.78** (shrank 46% from S1's 5.14)
- CPM  S2 cond: log_var weight ||W|| = **3.81** (essentially unchanged from S1's 3.81)

JSD pulled the log1p model's log_var weights toward zero. It had zero effect on CPM.

---

## Proposed Fix: Soft Constraint (L2 penalty on mu)

### Why it works where JSD/clamping fails

Soft constraint adds directly to the loss:
```
loss += lambda * mean(mu²) + lambda * mean(log_var²)
```

Gradient: `d(lambda * mu²)/d(mu) = 2 * lambda * mu`

This is **never zero** — and grows with mu, so the further mu drifts, the stronger
the pull back. Contrast with JSD which saturates to zero.

| Approach | Gradient at mu=50 |
|---|---|
| JSD (beta=0.1) | ≈ 0 (saturated) |
| JSD (beta=100) | ≈ 0 (still saturated) |
| Clamp mu | = 0 (chain broken) |
| Soft constraint (lambda=0.01) | = 2 × 0.01 × 50 = 1.0 |
| Soft constraint (lambda=0.1)  | = 2 × 0.1 × 50 = 10.0 |

### Code change
Added to `VAE_JSD.loss_function()` in `imputeVAE/models.py` (was previously only in VAE_L1):
```python
if getattr(args, "use_soft_constraint", False):
    mu_penalty = args.soft_constraint_lambda * torch.mean(mu ** 2)
    logvar_penalty = args.soft_constraint_lambda * torch.mean(log_var ** 2)
    loss = loss + mu_penalty + logvar_penalty
```

### Probe runs (2026-04-29)
Starting from CPM S1 model, running 100 epochs of stage 2 cond with:
- **job 4112282**: lambda=0.01 — moderate penalty, recons should survive
- **job 4112283**: lambda=0.1  — strong penalty, may hurt recons but mu will move

**What to look for**: Does nuclei_JSD drop below 0.693 within 20-30 epochs?
Scripts: `jobs/probe/run_probe_cpm_softcon_l001.sh`, `run_probe_cpm_softcon_l01.sh`

### Lambda selection rationale
- CPM recons loss ~1.5-2.2
- Soft constraint penalty = lambda × mean(mu²)
- If mu ~ 10 after pulling back: lambda=0.01 → penalty=1.0 (same order as recons) ✓
- If mu ~ 50 initially: lambda=0.01 → penalty=25 (dominates early, shrinks as mu shrinks) ✓
- lambda=0.1: aggressive, may collapse reconstruction if mu is pulled too fast
