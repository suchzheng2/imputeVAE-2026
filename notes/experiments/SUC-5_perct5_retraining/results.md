# SUC-5: Results

## Config B Baseline (for comparison)

From `notes/comprehensive_eval_results.md`:

| Metric | MSE cond | MSE cond+xr |
|--------|----------|-------------|
| Classification accuracy | 94.0% | 95.8% |
| DE overlap (overall) | 46.3% | 52.1% |
| Per-gene corr → raw cell | 0.941 | 0.979 |
| Coexpression | 0.916 | 0.915 |

Non-neuronal DE (Config B, cond+xr):
- Astro: 0.04, Endo: 0.02, Oligo: 0.06, OPC: 0.06, Macrophage: 0.07

## PerCT_5pct Results

Gene list: `data/gene_list_perct_5pct.csv` (16,439 genes)

### Stage 1 Training

- Completed 600 epochs
- Final loss: ~1.346 (recons ~1.0, JSD ~0.693)

### Stage 2 Training

- **Cond (no xr):** run `run_2026-04-07-02-12-05_3779214` — completed
- **Cond+xr:** run `run_2026-04-07-02-12-05_3551465` — completed
- Comprehensive eval: job 3250927 (COMPLETED)

### Comprehensive Eval (compeval job 3250927)

| Metric | pct5v2_cond | pct5v2_cond+xr |
|--------|-------------|----------------|
| Gene corr → raw cell | 0.909 | 0.963 |
| Gene R² | 0.699 | 0.893 |
| Cell recon gene corr | 0.964 | 0.973 |
| Classification accuracy | 93.3% | 94.2% |
| Classification Δ vs raw nuc | +10.6% | +11.5% |
| DE overlap | 0.495 | 0.547 |
| Coexpression | 0.929 | 0.898 |
| Per-cell: imputed vs ct mean | 0.995 | 0.998 |
| Per-cell: imputed vs raw nuc | 0.626 | 0.620 |
| Per-cell: cell recon vs raw cell | 0.869 | 0.869 |
| [Ceiling] real cell vs ct mean | 0.869 | 0.869 |

### Hallucination Tests (job 3250913)

| Metric | pct5v2_cond | pct5v2_cond+xr |
|--------|-------------|----------------|
| A1 variance ratio | 0.115 | 0.035 |
| A2 within-CT Spearman | 0.086 | 0.032 |
| A2 frac genes positive rank | 0.828 | 0.687 |
| B1 ablation corr | 0.999 | 0.999 |

Both models still show severe variance collapse and near-1.0 ablation correlation (hallucination).
cond (no xr) retains ~3x more variance than cond+xr, consistent with Config B findings.

### Comparison: PerCT_5pct vs Config B

| Metric | pct5 cond | pct5 cond+xr | ConfigB cond | ConfigB cond+xr |
|--------|-----------|--------------|--------------|-----------------|
| Gene corr | 0.909 | 0.963 | 0.941 | 0.979 |
| Classification | 93.3% | 94.2% | 94.0% | 95.8% |
| DE overlap | 0.495 | 0.547 | 0.463 | 0.521 |
| Coexpression | 0.929 | 0.898 | 0.916 | 0.915 |
| Variance ratio | 0.115 | 0.035 | — | — |

**Conclusions:**
- PerCT_5pct **improves DE overlap** (54.7% vs 52.1%) — the larger gene set captures more DE genes
- Cell-type-level metrics (gene corr, classif) are slightly lower than Config B, likely because more genes = harder task
- Variance collapse persists regardless of gene list — confirms this is an architecture/training issue, not a gene selection issue
- cond+xr still compresses variance ~3x more than cond alone
