# SUC-8: Linear Regression Baseline Results

**Date:** 2026-04-06
**Job:** 3245355 (COMPLETED)
**Gene list:** PerCT_5pct (16,439 genes)
**Method:** Per-gene linear regression: `cell_expr = a * nuc_expr + b`, fit on cell-type means (19 CTs), applied to individual nuclei

## Results

| Metric | Linear Regression | Description |
|--------|------------------|-------------|
| **variance_ratio** | **3.30** | Imputed variance / real cell variance (>1 = amplifies) |
| **within_ct_spearman** | **1.00** | Per-gene rank preservation within cell types |
| **within_ct_frac_pos** | **0.95** | Fraction of genes with positive Spearman |
| **gene_corr_linreg** | 0.80 | Cell-type mean gene correlation |
| **gene_r2_linreg** | 0.64 | Gene-level R² |
| **clf_linreg_acc** | 0.93 | Logistic regression classification accuracy |
| **clf_linreg_f1** | 0.92 | Classification F1 |
| **clf_raw_nuc_acc** | 0.83 | Raw nuclei classification (reference) |
| **clf_ctmean_acc** | 1.00 | Cell-type mean baseline classification |
| **percell_imp_vs_rawnuc** | 0.80 | Per-cell corr: imputed vs raw nuclei |
| **percell_realcell_ceiling** | 0.87 | Per-cell corr: real cell ceiling |
| **percell_imp_vs_cellmean** | 0.62 | Per-cell corr: imputed vs cell-type mean |
| **coexp_linreg** | 0.82 | Gene-gene coexpression preservation |
| **de_overlap_linreg** | 0.46 | DE gene overlap (top 100) |

## Key Conclusions

1. **Variance collapse is NOT fundamental** — linear regression preserves (even amplifies) per-cell variation with variance ratio 3.3 vs VAE's 0.037
2. Linear regression achieves 93% classification (vs VAE's ~96%) — modest cell-type-level tradeoff
3. Gene correlation 0.80 is solid but below VAE's 0.91 — room for the VAE to add value
4. **The VAE should target**: variance ratio ≥ 1.0, within-CT Spearman > 0.5, while maintaining its cell-type-level advantage

## Comparison with VAE (pct5_cond_xr, broken stage 1)

| Metric | Linear Reg | VAE pct5_cond_xr |
|--------|-----------|-----------------|
| Variance ratio | 3.30 | 0.037 |
| Within-CT Spearman | 1.00 | 0.044 |
| Gene correlation | 0.80 | 0.91 |
| B1 ablation corr | N/A | 0.998 (hallucinating) |

The VAE wins on cell-type means but completely fails on per-cell variation. This is partly due to the broken stage 1 (collapsed at epoch 400). A working VAE should aim to match linear regression's per-cell metrics while leveraging its nonlinear capacity for better cell-type-level translation.
