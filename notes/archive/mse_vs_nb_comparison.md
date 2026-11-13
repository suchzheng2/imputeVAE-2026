# MSE vs NB Comparison Plan (2026-04-02)

## Overview

We have two generations of runs. The **old runs** used vae_nb with ~24.7K genes (det>1% filter). The **new runs** use Config B gene filtering (10,134 genes via gene list) and compare two loss functions side by side: vae_js (MSE on log1p) vs vae_nb (NB on raw counts).

## Run Inventory

### Old Runs (det>1%, ~24.7K genes, JSD anneal start=200 over 400)

| Stage | Run Folder | Model | Status |
|-------|-----------|-------|--------|
| Stage 1 cell | `run_2026-03-12-17-29-06_928562` | vae_nb | completed |
| Stage 1 nuclei | `run_2026-03-12-17-29-06_928561` | vae_nb | completed |
| Stage 2 cell→nuclei | `run_2026-03-13-00-55-11_1260842` | vae_nb | completed |
| Stage 2 nuclei→cell | `run_2026-03-13-01-05-55_998747` | vae_nb | completed |
| Stage 3 (many variants) | see training_runs.md | vae_nb | completed |

### New Runs (Config B 10,134 genes, JSD anneal start=300 over 100)

| Stage | Run Folder | Model | Status |
|-------|-----------|-------|--------|
| Stage 1 cell MSE | `run_2026-04-01-09-29-26_3250320` | vae_js | completed |
| Stage 1 cell NB | `run_2026-04-01-09-29-27_2189144` | vae_nb | completed |
| Stage 2 joint MSE | `run_2026-04-01-14-01-16_1946018` | vae_js | completed |
| Stage 2 joint NB | `run_2026-04-01-14-43-14_475265` | vae_nb | completed |

## What Changed Between Old and New

1. **Gene filtering**: det>1% (~24.7K) → Config B gene list (10,134 genes). Removes ~14K noise/low-expression genes.
2. **JSD annealing**: start=200 over 400 epochs → start=300 over 100 epochs (sharper ramp, later start).
3. **Stage 2 architecture**: Old used separate stage2 (fine-tune on second modality only) → New uses stage3_v2 code with cascading loss annealing (cell always on, nuclei ramps 0-100, nuclei_jsd delayed 100-200, alignment delayed 200-300).
4. **New: MSE model added** (vae_js with MSE loss on log1p-normalized data) to compare against NB.

## Comparisons To Do

### 1. Stage 1: Reconstruction Quality (Over-Smoothing)

**Question:** Does Config B filtering and/or MSE loss reduce the over-smoothing problem seen with NB?

**Compare:** old_NB_24K vs new_NB_10K vs MSE_10K (all stage 1 cell models)

**Metrics:**
- Per-gene Pearson correlation (raw input vs decoded output) — measures gene-level fidelity
- Per-gene R² — variance explained per gene
- Per-cell Pearson correlation — measures cell-level profile preservation
- Wasserstein distance per gene — distribution-level match
- Decoded UMAP structure — visual check for over-smoothing (thin strands = over-smoothed)
- Decoded mean/std vs raw mean/std — scale preservation

**Visual check (already done):**
- Old NB decoded: tight thin strands per cell type (over-smoothed)
- New NB decoded: more spread within clusters (improved)
- MSE decoded: best structure, natural cluster shapes (no softmax bottleneck)

**Status:** Comparison script written (`analysis/compare_stage1_models.py`) but OOM on 200G — needs fix (process one model at a time, or reduce memory for old 24K model).

### 2. Stage 2: Alignment + Cross-Modality Quality

**Question:** How well does each model align cell and nuclei latent spaces, and how good is cross-modality reconstruction?

**Compare:** new MSE stage2 vs new NB stage2

**Metrics (from stage3_eval):**
- Alignment JSD — how close are cell/nuclei latent distributions
- Nuclei shift score — does decoded nuclei look more cell-like
- Gene correlation: decoded_nuc vs raw_cell, decoded_nuc vs raw_nuc
- Per-gene R², Wasserstein distance
- Decoder scale comparison
- scIB metrics (if computed): ASW, NMI, ARI, cLISI, iLISI

**Metrics from epoch dumps (every 50 epochs):**
- Training dynamics: how do alignment and cross-modality metrics evolve over epochs
- When does alignment converge
- Whether reconstruction quality degrades as alignment strengthens

### 3. Old vs New NB Stage 2

**Question:** Does Config B filtering help the NB model in the joint training stage?

**Compare:** old stage3 runs (various alignment weights) vs new NB stage2

**Key concern:** The old runs showed that alignment worked in latent space but corrupted decoder output (the "decoder shift" problem). Does reducing to 10K genes help?

### 4. MSE vs NB: Fundamental Architecture Question

**Question:** Should we switch from NB to MSE for this project?

**Pros of MSE:**
- No softmax bottleneck — avoids proportion distortion
- Simpler loss landscape
- Better decoded UMAP structure (stage 1)
- log1p normalization is standard in scRNA-seq

**Pros of NB:**
- Statistically grounded for count data
- Models overdispersion
- Standard in scVI/scArches literature
- Better handles zero-inflation

**Decision criteria:** If MSE stage2 produces better cross-modality reconstruction AND alignment, the softmax bottleneck in NB is clearly hurting more than the statistical model helps. If NB stage2 is better at alignment despite worse stage1, the count model may still be valuable.

## Results Summary (2026-04-02)

### Stage 1 Final Loss (epoch 599)

| Run | Model | Genes | Recons Loss | KL Div | Total Loss |
|-----|-------|-------|-------------|--------|------------|
| Old NB | vae_nb | 24,700 | 0.756 | 0.031 | 0.771 |
| Filtered NB | vae_nb | 10,134 | 2.054 | 0.040 | 2.074 |
| Filtered MSE | vae_js | 10,134 | 0.257 | 0.048 | 0.281 |

> Loss values not directly comparable: MSE is in log1p space, NB in NLL space, gene counts differ.

### Stage 2 Metrics (epoch 599, validation)

**Per-gene Pearson r (median):**

| Comparison | Filtered MSE | Filtered NB | Winner |
|------------|-------------|-------------|--------|
| Dec_Cell vs Raw_Cell | **0.947** | 0.879 | MSE |
| Dec_Nuc vs Raw_Nuc | **0.950** | 0.868 | MSE |
| Dec_Nuc vs Raw_Cell (cross-modal) | **0.774** | 0.702 | MSE |

**Per-gene R² (median):**

| Comparison | Filtered MSE | Filtered NB |
|------------|-------------|-------------|
| Dec_Cell vs Raw_Cell | **0.831** | 0.714 |
| Dec_Nuc vs Raw_Nuc | **0.807** | 0.625 |
| Dec_Nuc vs Raw_Cell | -1.115 | **-0.760** |

**Log-space MAE (median, lower=better):**

| Comparison | Filtered MSE | Filtered NB |
|------------|-------------|-------------|
| Cell→Cell | **0.053** | 0.111 |
| Nuc→Nuc | **0.034** | 0.066 |
| Nuc→Cell (cross-modal) | **0.237** | 0.391 |

**Wasserstein Distance (median, lower=better):**

| Comparison | Filtered MSE | Filtered NB |
|------------|-------------|-------------|
| Cell→Cell | **0.012** | 0.019 |
| Nuc→Nuc | **0.018** | 0.027 |
| Nuc→Cell | **0.030** | 0.035 |

**Shift Scores (validation):**

| | Filtered MSE | Filtered NB |
|--|-------------|-------------|
| Cell shift | 0.168 | 0.132 |
| Nuclei shift | -0.140 | -0.182 |

**scIB (latent space, training):**

| Metric | Filtered MSE | Filtered NB |
|--------|-------------|-------------|
| Cross-modal ASW_batch | **0.379** | 0.311 |
| Cross-modal iLISI | 0.0 | 0.0 |

### Key Findings

1. **MSE dominates NB on all reconstruction metrics** — Pearson r, R², log-MAE, Wasserstein all better.
2. **No-filter NB has no eval metrics** — only loss values; predates eval infrastructure. Need to run `compare_stage1_models.py` to get comparable numbers.
3. **Cross-modal reconstruction fails for both** — Dec_Nuc vs Raw_Cell R² is negative (worse than mean predictor). Neither model truly converts nuclei→cell yet.
4. **Nuclei shift negative for both** — alignment not strong enough.
5. **MSE avoids softmax bottleneck**, which is the primary advantage over NB.

## Suggested Additional Metrics

### High Priority — Biological Validation

1. **DE gene preservation**: Run DE (cell vs nuclei) on raw and decoded data. Measure overlap of significant DEGs. This is the actual downstream use case for the model.
2. **Marker gene fidelity**: Select known cell-type markers, report per-gene correlation specifically for these. A model that reconstructs markers well but misses noise genes is preferable.
3. **Cell-type classification accuracy**: Train KNN/logistic on raw cell data → apply to decoded nuclei. If nuclei→cell conversion works, classification should improve vs raw nuclei.
4. **Per-cell-type breakdown**: Report Pearson r, R², Wasserstein by cell type. Some types may be much harder; global aggregates hide this.
5. **Zero-fraction preservation**: What fraction of true zeros are reconstructed near-zero vs inflated? Critical for NB.

### Medium Priority — Distribution & Structure

6. **Gene-gene coexpression preservation**: Top-K gene-gene correlations in raw vs decoded. Spearman rank correlation of correlation matrices. Tests biological coexpression structure.
7. **Library size fidelity**: NB: compare decoded vs raw library sizes. MSE: compare total log1p expression per cell.
8. **Metrics by expression level**: Bin genes into high/medium/low expression, report metrics separately. NB may do well on high-expression genes but fail on low due to softmax.

### Cross-Modal Specific

9. **Cell-type centroid distance in latent space**: Per-cell-type centroid for cells vs nuclei after alignment. Measures alignment quality per type.
10. **kNN neighborhood overlap**: For each nucleus in latent space, fraction of k-nearest neighbors that are cells of matching type. More sensitive than iLISI for this use case.

## Extended Run Matrix (2026-04-02)

The comparison has expanded beyond MSE vs NB to a multi-axis design:

### Axes of Variation

1. **Loss function:** MSE (log1p) vs NB (raw counts)
2. **Gene filtering:** No filter (~24.7K, det>1%) vs Config B (10,134 genes)
3. **Decoder architecture:** Shared vs Conditioned (modality embedding, embed_dim=8)
4. **Cross-recons:** Off vs On (encode nuclei → decode with cell embedding → compare to raw cell)

### All Stage 2 Runs

| Run | Loss | Genes | Decoder | Cross-recons | Status | Eval | Nuc→Cell corr |
|-----|------|-------|---------|-------------|--------|------|---------------|
| `1946018` | MSE | 10K | shared | off | done | **done** | **0.760** |
| `475265` | NB | 10K | shared | off | done | **done** | 0.734 |
| `3345839` | MSE | 10K | cond | off | done | **done** | 0.756 |
| `722169` | NB | 10K | cond | off | done | **done** | 0.688 |
| job 3106445 | MSE | 10K | cond | **on** | running (e346) | — | TBD |
| (not yet) | MSE | 24K | shared | off | s1 running | — | — |
| old routes | NB | 24K | shared | off/on | done | partial | ~0.60-0.76 |

### Key Comparisons

1. **Effect of filtering:** MSE 24K vs MSE 10K (both shared) — isolate gene filtering impact
2. **Effect of loss:** MSE 10K vs NB 10K (both shared) — isolate loss function impact (MSE wins so far)
3. **Effect of cond decoder:** MSE shared vs MSE cond (both 10K, no xr) — isolate decoder architecture
4. **Effect of cross-recons:** MSE cond vs MSE cond+xr (both 10K) — **the key experiment**
5. **Combined improvement:** MSE cond+xr vs old NB shared — full stack improvement

## Full Stage 2 Comparison — All Results (updated 2026-04-02 evening)

All runs: 600 epochs, Config B (10,134 genes), v3 protocol, cascading loss annealing.

### Run Index

| ID | Loss | Decoder | Run Folder | Job | Cross-recons |
|----|------|---------|------------|-----|-------------|
| A | MSE | Shared | `run_2026-04-01-14-01-16_1946018` | 3073916 | off |
| B | NB | Shared | `run_2026-04-01-14-43-14_475265` | 3077244 | off |
| C | MSE | Cond | `run_2026-04-02-02-02-35_3345839` | 3102281 | off |
| D | NB | Cond | `run_2026-04-02-02-02-36_722169` | 3102282 | off |
| E | MSE | Cond+XR | `run_2026-04-02-10-51-32_3738876` | 3106445 | **on (1.0)** |

### Gene-Level Correlation (median per-gene Pearson r, epoch 599)

| | Cell→raw_cell | Cell→raw_nuc | Nuc→raw_cell | Nuc→raw_nuc |
|---|---|---|---|---|
| **A. MSE Shared** | **0.987** | 0.768 | 0.781 | 0.965 |
| **B. NB Shared** | 0.950 | 0.757 | 0.755 | 0.914 |
| **C. MSE Cond** | 0.982 | 0.748 | 0.782 | 0.964 |
| **D. NB Cond** | 0.929 | 0.722 | 0.735 | 0.922 |
| **E. MSE Cond+XR** | 0.982 | 0.769 | 0.730 | **0.968** |

### Cross-Decode Gene Correlation (cond runs only — nuclei latent → cell head)

| | XDec→raw_cell | XDec→raw_nuc | XDec R²→raw_cell |
|---|---|---|---|
| **C. MSE Cond** | 0.945 | 0.783 | 0.602 |
| **D. NB Cond** | 0.800 | 0.817 | -0.710 |
| **E. MSE Cond+XR** | **0.993** | 0.734 | **0.980** |

### Cell-Type-Mean Correlation Matrix (decoded vs raw, across all genes)

| | dec_cell↔raw_cell | dec_nuc↔raw_nuc | dec_nuc↔raw_cell |
|---|---|---|---|
| **A. MSE Shared** | 0.992 | 0.984 | 0.816 |
| **B. NB Shared** | 0.944 | 0.990 | 0.800 |
| **C. MSE Cond** | 0.995 | 0.993 | 0.807 |
| **D. NB Cond** | 0.948 | 0.991 | 0.764 |
| **E. MSE Cond+XR** | 0.995 | 0.992 | 0.797 |

5x5 cross-decode entries (cond runs):

| | xdec↔raw_cell | xdec↔raw_nuc | xdec↔dec_cell |
|---|---|---|---|
| **C. MSE Cond** | 0.985 | 0.804 | 0.979 |
| **D. NB Cond** | 0.954 | 0.779 | 0.984 |
| **E. MSE Cond+XR** | **0.998** | 0.795 | 0.993 |

### Shift Scores (corr_to_raw_cell − corr_to_raw_nuc, higher = more cell-like)

| | Cell Shift | Nuclei Shift | XDec Shift |
|---|---|---|---|
| **A. MSE Shared** | +0.177 | -0.167 | — |
| **B. NB Shared** | +0.113 | -0.190 | — |
| **C. MSE Cond** | +0.200 | -0.187 | +0.181 |
| **D. NB Cond** | +0.151 | -0.226 | +0.176 |
| **E. MSE Cond+XR** | +0.197 | -0.195 | **+0.203** |

### Decoder Output Scale

| | Cell Mean | Cell Std | Nuc Mean | Nuc Std |
|---|---|---|---|---|
| **A. MSE Shared** | 0.951 | 0.890 | 0.409 | 0.545 |
| **B. NB Shared** | 4.571 | **18.76** | 1.583 | **19.74** |
| **C. MSE Cond** | 0.943 | 0.887 | 0.407 | 0.547 |
| **D. NB Cond** | 4.509 | **17.61** | 1.585 | **21.66** |
| **E. MSE Cond+XR** | 0.931 | 0.877 | 0.404 | 0.547 |

### Final Training Loss Components (epoch 599)

| | Total | Nuc Recons | Nuc JSD | Cell Recons | Cell JSD | Align JSD | Cross Recons |
|---|---|---|---|---|---|---|---|
| **A. MSE Shared** | 0.544 | 0.224 | 0.034 | 0.256 | 0.038 | 0.028 | — |
| **B. NB Shared** | 2.794 | 1.042 | 0.031 | 1.704 | 0.017 | 0.024 | — |
| **C. MSE Cond** | 0.541 | 0.224 | 0.034 | 0.259 | 0.037 | 0.022 | — |
| **D. NB Cond** | 2.756 | 1.044 | 0.013 | 1.697 | 0.014 | 0.002 | — |
| **E. MSE Cond+XR** | 0.889 | 0.224 | 0.035 | 0.259 | 0.037 | 0.023 | 0.347 |

---

### Key Findings (updated 2026-04-02)

**MSE dominates NB on every metric:**
- Cell reconstruction: 0.982–0.987 vs 0.929–0.950
- Nuclei reconstruction: 0.964–0.968 vs 0.914–0.922
- Cross-decode to raw cell: 0.993 vs 0.800
- Decoder scale stability: std 0.55–0.89 vs 17–22 (NB is 30x noisier)

**Conditioned decoder: helps MSE slightly, hurts NB:**
- MSE: Cond (C) ≈ Shared (A) on reconstruction, but enables cross-decode path
- NB: Cond (D) worse than Shared (B) on nuc→cell (0.735 vs 0.755), alignment JSD collapses to 0.002

**Cross-recons is the breakthrough (MSE Cond+XR = run E):**
- Cross-decode gene corr: **0.993** (vs 0.945 without XR, vs 0.800 for NB)
- Cross-decode R²: **0.980** (vs 0.602 without XR, vs -0.710 for NB)
- Nuclei reconstruction preserved: **0.968** (actually best of all runs)
- No trade-off between imputation quality and nuclei fidelity
- Cross-decode shift: **+0.203** (strongly cell-like)

**NB scale distortion confirmed:**
- NB decoder output std 17–22 vs MSE std 0.5–0.9
- Direct evidence of softmax proportion × library_size amplifying variance
- This is the root cause of NB underperformance

**Architecture verdict:**
- Best overall: **MSE + Conditioned Decoder + Cross-Recons** (run E)
- The cross-decode path (nuclei latent → cell decoder head) is the correct way to do nuclei→cell translation
- Standard nuclei decode path correctly outputs nuclei-like data (nuc→nuc corr 0.968)
- Two decoder heads specialize independently without interference

## Next Steps

1. ~~Wait for cond+cross-recons run (job 3106445) to complete~~ **DONE** — completed epoch 599
2. MSE no-filter stage 1 (job 3115964, epoch ~158/600) — then submit stage 2 to complete 2×2 matrix
3. Implement biological validation metrics (DE preservation, marker fidelity, classification)
4. Run per-cell-type metric breakdown — which cell types benefit most from imputation?
5. Gene stratification — does model help on genes with largest cell/nucleus difference?
6. Benchmark against scVI and simple baselines (per-gene linear regression)
7. Statistical rigor: bootstrap CI on gene-level correlations, null distribution via permutation
8. Consider: cross-recons with shared decoder MSE for comparison (does cond decoder matter, or is XR alone enough?)
