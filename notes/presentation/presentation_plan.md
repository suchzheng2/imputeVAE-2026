# Lab Meeting Presentation Plan (2026-04-03)

## Key Discoveries During Planning

### 1. Cross-decode already works (no cross-recons needed)

The conditioned decoder (MSE) achieves **0.949 corr to raw cell** via cross-decode
(encode nuclei → decode with cell embedding), with **positive shift (+0.18)** and **R²=0.54**.
This is dramatically better than the shared decoder's 0.760 corr / -0.14 shift / R²=-1.1.

### 2. Cross-recons improves further: 0.989 corr, R²=0.97

Job 3106445 (MSE cond + cross-recons) completed. Cross-decode results at epoch 599:
- Corr to raw cell: **0.989** (vs 0.949 without xr)
- R² to raw cell: **0.968** (vs 0.537 without xr)
- Shift: **+0.198** (vs +0.180 without xr)
- Wasserstein: **0.010** (vs 0.031 without xr)

Cross-recons is a clear improvement, especially for R² and Wasserstein.

### 3. Gene filtering removes ALL non-neuronal marker genes

Config B (det>5% in both modalities + MitoCarta exclusion) keeps 10,134 genes but **removes**:
- Aqp4 (astrocyte), Aldh1l1 (astrocyte)
- Mog (oligodendrocyte), Pdgfra/Cspg4 (OPC)
- Flt1/Pecam1 (endothelial)
- Ctss/Tmem119 (microglia/macrophage)

These are cell-type-specific markers for rare types (~1-5% of cells each), so they fail
the 5% global detection rate threshold. **This is a real problem** for a brain single-cell
audience — we can't validate imputation of non-neuronal markers if they're filtered out.

Genes that ARE present: Slc17a7 (excitatory), Gad1/Gad2 (inhibitory), Snap25/Syt1 (pan-neuronal),
Rbfox3 (neuronal), Cx3cr1 (microglia), Mbp (mature oligo — this one made it through).

### 4. Job 3106445 completed, 3115964 still running

- **3106445** (MSE cond+xr): COMPLETED, epoch 600/600, run folder `run_2026-04-02-10-51-32_3738876`
- **3115964** (MSE no-filter s1): still running, epoch ~204/600, won't be ready

---

## Updated Comparison Table (validation, epoch 599, cross-decode mode)

| Model | Corr to raw cell | R² to raw cell | Shift | Wasserstein |
|-------|-------------------|----------------|-------|-------------|
| MSE shared (no cross-dec) | 0.760 | -1.115 | -0.14 | 0.030 |
| MSE cond (cross-dec, no xr) | **0.949** | 0.537 | +0.18 | 0.031 |
| MSE cond+xr (cross-dec) | **0.989** | **0.968** | **+0.20** | **0.010** |
| Raw nuclei means vs raw cell means | 0.821 | — | — | — |
| Real cell vs own ct mean (baseline) | — | — | — | — |

**BASELINE COMPUTED:** Real cell vs own cell-type mean per-cell corr = 0.842.
Raw nuclei means vs raw cell means per-gene corr = 0.821.

The cross-decode metrics (0.949/0.989) are per-gene corr on cell-type means (17 data points).
Per-cell imputed nuc vs ct mean: MSE_shared = 0.793 (below baseline 0.842 — not good enough).
Waiting for cond/cond+xr per-cell results from comprehensive eval (job 3128807).

---

## Per-Celltype Results (cross-decode, validation, epoch 599)

### MSE cond+xr (job 3106445)

| Cell Type | Corr to Cell | Corr to Nuclei | Shift |
|-----------|-------------|----------------|-------|
| L6b | 0.999 | 0.758 | +0.241 |
| Pvalb | 0.999 | 0.764 | +0.236 |
| Sst | 0.999 | 0.764 | +0.235 |
| Sncg | 0.997 | 0.768 | +0.229 |
| L6 IT | 0.998 | 0.771 | +0.227 |
| L2/3 IT | 0.999 | 0.780 | +0.219 |
| L5 IT | 0.999 | 0.786 | +0.213 |
| Lamp5 | 0.999 | 0.793 | +0.206 |
| Macrophage | 0.970 | 0.769 | +0.201 |
| L5/6 NP | 0.998 | 0.804 | +0.194 |
| L5 ET | 0.997 | 0.808 | +0.190 |
| Oligo | 0.992 | 0.803 | +0.190 |
| Vip | 0.998 | 0.820 | +0.178 |
| Endo | 0.982 | 0.815 | +0.167 |
| OPC | 0.994 | 0.845 | +0.150 |
| Astro | 0.967 | 0.886 | +0.081 |

Notes:
- Neuronal types (L2/3 IT, Pvalb, Sst, etc.) have near-perfect corr (0.997-0.999)
- Non-neuronal types (Astro, Endo, Macrophage) have lower corr (0.967-0.982)
- Astro has lowest shift (+0.081) — cell/nuclei already most similar for astrocytes?
- All shifts positive — model consistently produces more cell-like output

---

## Gene Filtering Problem — Discussion

### The issue
Config B uses 5% global detection rate. Cell-type-specific markers for rare populations
(astrocytes ~5%, OPC ~3%, endothelial ~2%, microglia ~1%) fail this threshold because
they're only expressed in a small fraction of total cells.

### Options
1. **Lower threshold to 1%** — keeps ~17.5K genes including most markers, but adds noise
2. **Use per-cell-type detection rate** — keep genes detected in >5% of ANY cell type
3. **Union of Config B + curated marker list** — add known markers back explicitly
4. **Just acknowledge it** — our model works on the gene set we have, markers are future work

### For the talk
Best approach: **acknowledge the limitation honestly**, show which markers are present/absent,
note this means we can validate neuronal imputation well but non-neuronal validation is limited.
A quick analysis showing total variance captured by Config B genes would strengthen this.

---

## Revised Talk Structure (30-45 min)

1. **Problem** (5 min): Nuclei-seq misses cytoplasmic RNA; can we impute cell expression from nuclei?
   - Show cell vs nuclei expression differences (known biology)
   - Allen Brain Atlas dataset: 17 cell types, 10K genes

2. **Approach** (5-8 min): VAE with modality-conditioned decoder
   - Shared encoder → aligned latent space
   - Modality embedding in decoder → cell mode vs nuclei mode
   - 3-stage training: pretrain cell → joint training → alignment
   - Brief: why MSE not NB (one slide, softmax bottleneck)

3. **Key Result** (10 min): Cross-decode enables imputation
   - Encode nuclei → decode with cell embedding → cell-like output
   - 0.989 corr to raw cell (with cross-recons), 0.949 without
   - Per-cell-type breakdown: all positive shift, neuronal types near-perfect
   - Comparison to baseline (cell-type mean) — **TO GENERATE**

4. **Ablation** (5-8 min): What matters and what doesn't
   - Conditioned decoder vs shared: huge difference (0.949 vs 0.760)
   - Cross-recons helps R² dramatically (0.97 vs 0.54) but corr already good
   - Alignment alone (shared decoder) insufficient
   - NB vs MSE: MSE wins, brief mention

5. **Limitations & Next Steps** (5 min):
   - Gene filtering removes non-neuronal markers — need to address
   - Per-cell variation (R² without xr only 0.54)
   - Need biological validation: DE preservation, marker fidelity
   - Cross-recons vs no-xr for downstream tasks

## Must-Generate Analyses

### Priority 1 (blocking for talk)
- [ ] Cell-type mean baseline — compute and compare
- [ ] Summary figure: comparison table (shared vs cond vs cond+xr)

### Priority 2 (strengthens talk)
- [ ] Gene-level analysis: per-gene R² histogram, highlight specific markers
- [ ] Check which markers from Allen Brain Atlas taxonomy ARE in Config B
- [ ] Variance captured by Config B genes vs full set

### Priority 3 (nice to have)
- [ ] Training dynamics plot (cascading annealing)
- [ ] Latent UMAP before/after alignment
- [ ] Gene dot plot for present markers
