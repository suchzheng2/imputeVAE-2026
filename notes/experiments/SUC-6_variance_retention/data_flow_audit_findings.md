# Data Flow Audit Findings (2026-04-08)

**Model:** pct5v2_cond (`run_2026-04-07-02-12-05_3779214`)
**Gene list:** PerCT_5pct (16,439 genes)
**Data:** Cell 18,759 samples, Nuclei 17,792 samples, 17 CTs
**Script:** `analysis/data_flow_audit.py`
**Results:** `audit_results/pct5v2_cond/`

## Variance Flow Summary

| Stage | Mean gene var | Within-CT | Between-CT | Within frac |
|-------|--------------|-----------|------------|-------------|
| Raw cell | 0.3125 | 0.2263 | 0.0863 | **72.4%** |
| Raw nuclei | 0.1752 | 0.1492 | 0.0261 | 85.1% |
| Decoded cell | 0.1455 | 0.0461 | 0.0997 | **31.6%** |
| Decoded nuclei | 0.0348 | 0.0068 | 0.0280 | 19.6% |
| Cross-decoded | 0.1024 | 0.0223 | 0.0801 | 21.8% |

## Key Findings

### 1. The decoder inverts the within/between variance ratio

Raw cells: 72% within-CT. Decoded cells: 32%. The model is not just collapsing to means — it actively **amplifies between-CT signal while crushing within-CT signal**. Between-CT variance in decoded output (0.0997) is actually *higher* than in raw data (0.0863), meaning the decoder is over-separating cell types.

### 2. Same-modality reconstruction is already the main bottleneck

Cell→cell recon retains only **20.4%** of within-CT variance. This is not a cross-modal problem — H1 (MSE→mean) is confirmed as the dominant cause. The decoder ignores per-cell z variation even when reconstructing the same modality.

### 3. Cross-decode adds ~35% additional loss on top

Median additional_var_loss (cross/same-mod) = 0.65. So cross-decode makes it worse, but same-mod recon is already 80% of the problem.

### 4. Nuclei reconstruction is catastrophically worse

- Nuclei within-CT var ratio per gene (median): **0.014** (vs 0.104 for cell)
- Nuclei per-gene correlation (mean): 0.209 (vs 0.438 for cell)
- The model barely reconstructs any per-cell variation for nuclei

### 5. Massive modality asymmetry in raw data

- Library size (log1p sum): cell 9411 ± 3293, nuclei 4138 ± 1212 (**2.3x ratio**)
- 93% of genes (15,368/16,439) have higher expression in cells
- Nuclei have lower variance across the board

### 6. Latent space: within-CT variance is dwarfed by between-CT

| Component | Cell | Nuclei |
|-----------|------|--------|
| Total latent var | 2.50 | 2.46 |
| Within-CT | 0.21 | 0.10 |
| Between-CT | 0.94 | 0.87 |

Despite earlier diagnostic showing 82% within-CT proportion, the *absolute* within-CT variance (0.21) is 4.5x smaller than between-CT (0.94). The encoder compresses within-CT relative to between-CT.

### 7. Inter-modality fraction is only 2.1% (lower than expected)

The earlier diagnostic (different model, Config B) showed 37% inter-modality. This pct5v2_cond model has much better modality separation — only 2.1% inter-modality, max dim at 20.9%. The modality embedding may be doing its job better here.

### 8. Per-gene reconstruction quality

- Cell recon: mean gene corr 0.438, median var ratio 0.20
- Nuc recon: mean gene corr 0.209, median var ratio 0.05
- Cross-decode: median var ratio vs cell 0.13

### 9. Per-sample correlation (decoder output vs raw)

- Cell: 0.857 mean (min 0.333) — good profile-level recon
- Nuclei: 0.756 mean — weaker

## Revised Understanding

The variance collapse has three layers:

1. **Encoder compression** — within-CT variance in latent (0.21) vs between-CT (0.94) means the encoder puts 4.5x more information into cell-type identity than per-cell variation. Not necessarily wrong (between-CT is the larger signal in raw data too), but it sets a ceiling.

2. **Decoder ignoring within-CT z** (dominant) — even the within-CT info that exists in z is not used by the decoder. Same-modality cell recon loses 80% of within-CT variance. This is the MSE→mean effect (H1).

3. **Cross-modal additional loss** (secondary) — cross-decode loses another ~35% on top. This is the modality-switching penalty.

## Implications for Interventions

- **Residual MSE failed** because it still optimizes E[cell_residual | z] ≈ 0 with unpaired data
- **The key target is layer 2**: make the decoder actually use within-CT z variation
- A **variance-preserving penalty** on decoder output could work: directly penalize when within-CT output variance is too low relative to latent within-CT variance
- The **encoder is not the bottleneck** for cell data — it preserves proportional within-CT info. But for nuclei, latent within-CT is only 0.10, which may be too low.
- The between-CT *amplification* (0.0863 raw → 0.0997 decoded) is interesting — the model over-separates cell types. This could be a JSD divergence side effect.

## Files

- `audit_results/pct5v2_cond/A1_gene_level.csv` — per-gene raw statistics
- `audit_results/pct5v2_cond/C1_gene_level_recon.csv` — per-gene reconstruction quality
- `audit_results/pct5v2_cond/D1_gene_level_cross.csv` — per-gene cross-decode quality
- `audit_results/pct5v2_cond/E2_variance_flow_summary.csv` — aggregate flow
- `audit_results/pct5v2_cond/plots/` — 10 figures
