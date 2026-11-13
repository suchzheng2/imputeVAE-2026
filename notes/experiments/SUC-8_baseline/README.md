# SUC-8: Baseline Comparison — Linear Regression + scVI

**Linear issue:** [SUC-8](https://linear.app/suchen/issue/SUC-8)
**Status:** In Progress
**Priority:** High — blocks SUC-6 (variance retention)

## Goal

Establish baseline methods to compare against the VAE. Without baselines, we can't quantify the VAE's value-add or diagnose whether variance collapse is a VAE-specific problem or inherent to the unpaired setting.

## Baselines

### 1. Per-gene linear regression (this experiment)

For each gene, fit `cell_expr[g] = a[g] * nuc_expr[g] + b[g]` using cell-type means as anchor points (19 CTs). Then apply to individual nuclei.

Key insight: this preserves per-cell variation through the linear transform (slope != 0), unlike the cell-type mean baseline which collapses by definition.

### 2. scVI transfer (stretch goal)

Train scVI on cell data, use as reference for nuclei imputation.

## Key Questions

1. Does linear regression also collapse to cell-type means? (variance ratio)
2. Does it preserve within-CT ranking better than the VAE? (A2 Spearman)
3. How does classification accuracy compare?
4. If linear regression outperforms VAE on variance → our training is the bottleneck
5. If it also collapses → the problem is fundamental to unpaired setting

## Scripts

- `analysis/baseline_linear_regression.py` — main analysis script
- `jobs/eval/run_baseline_linreg.sh` — SLURM submission
