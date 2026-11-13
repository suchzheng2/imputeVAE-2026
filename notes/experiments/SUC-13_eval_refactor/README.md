# SUC-13: Refactor Evaluation Framework — Modular, 5-Source, Reusable

**Linear issue:** [SUC-13](https://linear.app/suchen/issue/SUC-13/refactor-evaluation-framework-modular-5-source-reusable)
**Status:** In Progress
**Priority:** High — blocks SUC-6, SUC-9
**Labels:** Improvement

## Problem

Evaluation code is fragmented across 5+ standalone scripts with:
- Duplicated metric logic between inline and post-training eval
- Hardcoded model presets in each script
- Inconsistent 4-source vs 5-source (missing cross_decoded in inline)
- No gene-stratified sample-level metrics
- Data loaded redundantly across scripts

## 5 Sources (conditioned decoder)

| # | Source | Description | Role |
|---|--------|-------------|------|
| 1 | raw_cell | Ground truth cell expression | Target |
| 2 | raw_nuclei | Ground truth nuclei expression | Input |
| 3 | decoded_cell | encode cell → decode cell embedding | Reconstruction check |
| 4 | decoded_nuclei | encode nuclei → decode nuclei embedding | Reconstruction check |
| 5 | cross_decoded | encode nuclei → decode **cell embedding** | **The imputation** |

## Current Overlap Map

| Metric | Inline (stage3_eval) | Post-training | Duplicated? |
|--------|---------------------|---------------|-------------|
| Gene correlation (CT means) | Tier 1 | comprehensive_eval | YES |
| R² | Tier 1 | comprehensive_eval | YES |
| Shift scores | Tier 1 | comprehensive_eval | YES |
| Wasserstein | Tier 1 | — | no |
| Heatmaps | Tier 2 | — | no |
| Correlation matrix | Tier 2 | — | no |
| UMAPs | Tier 2 | — | no |
| scIB | Tier 2 | — | no |
| Classification | — | comprehensive_eval | no |
| DE preservation | — | comprehensive_eval | no |
| Coexpression | — | comprehensive_eval | no |
| Per-cell correlation | — | comprehensive_eval | no |
| Variance ratio | — | hallucination_test | no |
| Within-CT Spearman | — | hallucination_test | no |
| z ablation | — | hallucination_test | no |
| Gene stratification | — | gene_stratification | no |

## Architecture

```
analysis/eval/
  __init__.py
  metrics/              # Layer 1: Pure functions (arrays → numbers)
    ct_level.py         #   gene_corr, r2, shift, classification, DE, coexpression
    sample_level.py     #   variance_ratio, within_ct_spearman, per_cell_corr
    latent.py           #   z_ablation, z_variance_decomposition
    gene_stratified.py  #   per-quintile breakdown (ct_level + sample_level)
  data.py               # Layer 2: Load, filter, cache by gene list
  inference.py           # Layer 2: Model → 5-source dict
  registry.py            # Layer 2: Model configs (single source of truth)
  runner.py              # Layer 3: CLI orchestration (post-training) + SLURM
  report.py              # Layer 4: Cross-model comparison tables
```

### Key design principles

1. **Pure metric functions** — no I/O, no model loading. Both inline and post-training call the same functions.
2. **5-source as standard** — every metric works with standardized 5-source dict.
3. **Model registry** — add new model = add one dict entry, run eval immediately.
4. **Data cached by gene list** — load once, reuse across models.
5. **Linear regression is NOT a model** — handled separately as baseline.

### CLI usage

```bash
# List models
python -m analysis.eval.runner --list

# Evaluate a single model (all metrics)
python -m analysis.eval.runner --model pct5v2_cond --eval all

# Evaluate specific metrics
python -m analysis.eval.runner --model pct5v2_cond --eval ct_level,sample_level

# Evaluate multiple models
python -m analysis.eval.runner --model pct5v2_cond,pct5v2_cond_xr --eval all

# Generate SLURM job script
python -m analysis.eval.runner --model pct5v2_cond --eval all --slurm
```

## Implementation Phases

### Phase 1 (DONE — 2026-04-07)
Built all modules: metrics/, runner.py, registry.py, data.py, inference.py, report.py.
Submitted pct5v2 eval jobs (3257230 cond, 3257231 cond+xr).

### Phase 2
Refactor stage3_eval.py inline to call shared metrics/ via inline.py.
Upgrade to 5-source.

### Phase 3
Deprecate old standalone scripts (thin wrappers for backwards compat).

## Scripts

- `analysis/eval/` — new framework
- Current scripts being replaced:
  - `analysis/comprehensive_eval.py`
  - `analysis/hallucination_test.py`
  - `analysis/gene_stratification.py`
  - `imputeVAE/stage3_eval.py` (inline portion)
