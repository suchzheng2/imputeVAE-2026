# Cross-Dataset Validation Experiments

## Goal
Test whether the v3-trained cVAE generalizes to unseen data:
1. Different protocol (v2 batch from same AIBS dataset)
2. Different tissue/species (external datasets)

## Model Under Test
- **Best full-data cVAE**: `run/run_2026-07-18-19-40-51_570984`
- Architecture: single-stage cVAE, scalar modality embedding, enc+dec, log1p, normloss
- Trained on: v3 batch (n3A nuclei, c3A cells), 10,134 genes, 100% data

## Experiment 1: v2 Batch (Protocol Transfer)

**Rationale**: AIBS dataset has both 10x v2 and v3 batches. Model trained on v3 only.
Testing on v2 shows whether the learned cell-nucleus transform is protocol-specific.

- Script: `analysis/cross_dataset_eval.py`
- Job script: `jobs/eval/run_cross_dataset_v2.sh`
- Output: `run/cross_dataset_eval/v2_batch/` and `run/cross_dataset_eval/v3_batch/` (reference)

### v2 Batch Details
- Nuclei (n2A): 8 donors, ~8K-11K nuclei each
- Cells (c2A): same donors, similar counts
- Same tissue (mouse MOp), same cell types
- Key difference: 10x Chromium v2 chemistry (shorter reads, lower sensitivity)

### Results (job 7758417, completed 2026-07-20)

- v2: 76,525 nuclei, ~100K cells, 17 common CTs (incl. L6 IT Car3)
- v3: 40,166 nuclei, 71,117 cells, 17 common CTs

| Metric | v3 (trained) | v2 (unseen) | Notes |
|--------|:-:|:-:|-------|
| Gene corr (median) | **0.927** | 0.593 | CT-level metric drops significantly |
| R² (median) | **0.798** | -0.026 | Model predictions biased for v2 CT means |
| Overall corr | **0.980** | 0.872 | All CT-gene pairs pooled |
| QQ gene corr (median) | **0.944** | 0.894 | Label-free — holds up well |
| Wasserstein (median) | 0.279 | **0.263** | Label-free — slightly better on v2 |
| Variance ratio (median) | **0.158** | 0.077 | Less variance preserved on v2 |
| Shift score (mean) | **0.084** | 0.162 | Larger shift error on v2 |
| Per-cell corr (median) | **0.996** | 0.954 | Still high on v2 |

### Per-CT Breakdown (v2)

| Cell Type | n_imp | n_cell | gene_corr |
|-----------|------:|-------:|:---------:|
| L2/3 IT | 18,607 | 11,119 | 0.970 |
| L5 IT | 20,272 | 42,647 | 0.970 |
| L5 ET | 2,977 | 515 | 0.955 |
| L6 CT | 12,704 | 22,981 | 0.949 |
| L6b | 1,329 | 2,337 | 0.957 |
| Sst | 2,799 | 7,107 | 0.955 |
| Vip | 1,538 | 8,082 | 0.954 |
| Pvalb | 3,532 | 4,976 | 0.952 |
| L6 IT | 7,580 | 9,489 | 0.951 |
| L5/6 NP | 2,246 | 4,118 | 0.942 |
| Lamp5 | 1,609 | 6,071 | 0.939 |
| Sncg | 189 | 1,352 | 0.935 |
| L6 IT Car3 | 293 | 646 | 0.910 |
| Astro | 415 | 440 | 0.826 |
| OPC | 69 | 123 | 0.841 |
| Endo | 21 | 141 | 0.778 |
| Oligo | 320 | 393 | 0.775 |
| Macrophage | 25 | 104 | 0.503 |

### Interpretation

**CT-level metrics drop but label-free metrics hold up:**
- Gene corr (0.927→0.593) and R² (0.798→-0.03) are CT-mean-based metrics. The drop
  reflects v2-vs-v3 chemistry differences in capture efficiency, especially at low
  expression levels. CT means shift between protocols.
- QQ gene corr (0.944→0.894) and Wasserstein (0.279→0.263) are label-free distributional
  metrics. These remain strong, showing the overall expression distributions are reasonable.

**Neuronal types transfer well, rare/glial types don't:**
- All neuronal types maintain gene_corr > 0.93 on v2
- Rare types (Macrophage n=25, OPC n=69, Endo n=21) and glial types (Oligo, Astro) drop
- Likely causes: (1) v2 has lower sensitivity for low-expressed genes, (2) rare types
  have too few samples for stable CT means, (3) glial biology may differ more across protocols

**Implications:**
- The model learns a reasonable cell-nucleus transform that partially generalizes across protocols
- Protocol-specific batch effects cause CT-mean shifts that hurt CT-level metrics
- For cross-protocol use, either protocol-aware conditioning or fine-tuning would help
- Label-free metrics suggest the core biological signal transfers

## Planned External Datasets

| Dataset | Species | Tissue | Protocol | Access |
|---------|---------|--------|----------|--------|
| Lake 2023 | Human | Kidney | 10x v3 | GSE183279 / CELLxGENE |
| Yao 2021 BICCN | Mouse | Motor cortex (MOp) | 10x v3 | NeMO dat-ch1nqb7 |
| Denisenko 2020 | Mouse | Kidney | 10x v3 | GEO |
| Madissoon 2023 | Human | Lung | 10x v3.1 | GEO |
| Kumar 2023 | Human | Breast | 10x | GEO |
| Bosse 2024 | Human | Lung (paired) | 10x | Zenodo |

### Challenges for External Datasets
- Gene name mapping (human orthologs vs mouse genes)
- Different gene sets (need intersection with training genes)
- Different cell type annotations (need mapping or label-free metrics only)
- Different normalization / count depths

## Update 2026-10-23: the v2 drop was a library-scale shift (covariate model closes it)

Re-run through the same script (`analysis/cross_dataset_eval.py`, now with `--target_libsize` for `libsize_cond`
runs; job array 13783381, `jobs/eval/run_cross_dataset_v2_libsize.sh`). Paper model reproduces July exactly
(gene corr 0.593, R² -0.03). The library-size covariate model (arm A, `run/run_2026-10-16_libsize_armA_jsd0.1_seed1`)
decoded at the **v2-cell per-type median library** gives gene corr **0.927**, R² **0.723**; quantile-mapped onto v2
cells **0.935 / 0.748**, Wasserstein 0.247 (best of all). Decoded at the v3 training-cell library it is 0.880 / -4.8
(wrong scale), at the nucleus's own library 0.604 / -1.3 (the paper model's behaviour). Glia and rare types recover
too (Macrophage 0.50 -> 0.85, Oligo 0.78 -> 0.86, OPC 0.84 -> 0.92). Full table: `notes/libsize_covariate_review_2026-10-09.md` §6.3.
Conclusion: the transformation transfers across chemistry; only the output scale was wrong, and it is now an
explicit parameter (target whole-cell RNA content per type). Open: a label-free scale mode for nuclei-only data.
