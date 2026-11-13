# Lab Meeting Presentation — Outline v3
# Date: 2026-04-03, 30-45 min + discussion

## Narrative Arc

**Problem** → **Approach** → **It works** → **How good, really?** → **Gene filtering fix** → **Next steps**

---

## Section 1: The Problem (4 min, 3 slides)

### Slide 1 — Title

1. **Title**: "Imputing Cell Expression from Single-Nucleus RNA-seq with a Modality-Conditioned VAE"
2. **Content**:
   - Presenter name, lab, date
3. **Visual**: Clean title slide with cell/nucleus schematic

---

### Slide 2 — snRNA-seq Is Ubiquitous but Incomplete

1. **Title**: Single-Nucleus Sequencing Misses Cytoplasmic RNA
2. **Content**:
   - PsychENCODE: 2.8M nuclei from 388 brains (Gerstein, *Science* 2024)
   - Nuclei lack cytoplasmic mRNA: activity genes (*Fos*, *Arc*), secreted factors
   - Population-level cell-nucleus correlation $r \approx 0.82$ — systematic gap
3. **Visual**: Schematic of cell showing nuclear vs cytoplasmic RNA pools; scatter plot of nucleus-vs-cell mean expression with gap highlighted

---

### Slide 3 — Can We Computationally Recover Cell Expression?

1. **Title**: Predicting Whole-Cell Profiles from Nuclei Alone
2. **Content**:
   - Allen Brain Atlas: 17 cell types, ~71K cells + ~40K nuclei (unpaired)
   - No matched cell-nucleus pairs — learn translation from populations
   - Goal: nucleus profile $\rightarrow$ VAE $\rightarrow$ cell-like profile
3. **Visual**: Dataset overview table (cell types, sample sizes); diagram: nucleus expression → ? → cell expression

---

## Section 2: Approach (5-7 min, 4 slides)

### Slide 4 — VAE Primer

1. **Title**: Variational Autoencoder in 60 Seconds
2. **Content**:
   - Encoder: 10K genes $\rightarrow$ 32-dim latent $z$; decoder: $z \rightarrow$ reconstructed expression
   - Loss: $\mathcal{L} = \mathcal{L}_{\text{recon}} + \beta \cdot D_{\text{JSD}}$
   - MSE on log1p-normalized data (outperformed negative binomial — softmax bottleneck)
3. **Visual**: Simple VAE block diagram: input → encoder → $z$ → decoder → output

---

### Slide 5 — Key Idea: Modality-Conditioned Decoder

1. **Title**: One Decoder, Two Modes — via Learned Modality Embedding
2. **Content**:
   - Shared encoder maps both modalities to same latent space
   - Decoder input: $[z \,||\, e_{\text{mod}}]$ where $e_{\text{mod}}$ is a learned 8-dim embedding
   - The trick: encode nucleus → decode with **cell** embedding → cell-like output
3. **Visual**: Architecture diagram showing encoder → $z$ → concat with modality embedding → decoder. Highlight the embedding swap arrow for cross-decode

---

### Slide 6 — Cascading Loss Annealing Prevents Decoder Degradation

1. **Title**: Staged Training: Losses Activate Sequentially
2. **Content**:
   - Stage 1: pretrain on cell data only (600 epochs)
   - Stage 2: cell recon always on; nuclei recon → cross-recon → alignment ramp sequentially
   - Simultaneous losses degraded decoder in earlier attempts
3. **Visual**: Gantt-style timeline: epoch 0–100 nuclei recon ramps, 100–200 cross-recon + JSD, 200–300 alignment

---

### Slide 7 — Three Model Variants Compared

1. **Title**: Shared vs Conditioned vs Conditioned + Cross-Recons
2. **Content**:
   - Shared decoder: single decoder, no modality awareness (baseline)
   - Conditioned: adds modality embedding to decoder input
   - Cond + cross-recons: adds $\mathcal{L}_{\text{xr}}$: encode nuclei, decode as cell, compare to real cell
3. **Visual**: 3-column comparison table: architecture, loss terms, key difference

---

## Section 3: Results — It Works (10-12 min, 6 slides)

### Slide 8 — Conditioned Decoder Is the Key Unlock

1. **Title**: Modality Embedding Enables Cross-Modal Translation
2. **Content**:
   - Shared: $r = 0.79$, $R^2 = -1.1$ (worse than predicting the mean)
   - Conditioned: $r = 0.94$, $R^2 = 0.56$
   - + cross-recons: $r = 0.98$, $R^2 = 0.95$
3. **Visual**: Grouped bar chart — per-gene correlation and $R^2$ across 3 models. Dramatic jump from shared → conditioned

---

### Slide 9 — Classification: 85% → 96% with Imputation

1. **Title**: Imputed Nuclei Classify Better Than Raw Nuclei
2. **Content**:
   - Logistic regression trained on real cells; tested on raw vs imputed nuclei
   - Raw nuclei: 85.1% → Imputed (cond+xr): **95.8%** (+10.7%)
   - Consistent improvement across 11/17 neuronal cell types
3. **Visual**: Bar chart: raw nuclei vs 3 models' accuracy. Optional: confusion matrix for best model

---

### Slide 10 — DE Recovery Exceeds Raw Nuclei

1. **Title**: Cross-Decode Recovers More DE Genes Than Raw Nuclei
2. **Content**:
   - Top-100 Wilcoxon DE overlap: imputed **52%** vs raw nuclei 47% (+5.1%)
   - 11/17 types improved; Vip +40%, L6 IT +35%, Lamp5 +30%
   - Reconstruction ceiling is 55% — cross-decode at 52% nearly matches it
4. **Visual**: Paired bar chart per cell type (imputed blue vs raw nuclei gray), sorted by delta. Dashed line for recon ceiling. Non-neuronal types annotated "gene filtering"

---

### Slide 11 — Model Adds Most Value on the Hardest Genes

1. **Title**: Imputation Quality Scales with Cell-Nucleus Divergence
2. **Content**:
   - Genes stratified by cell-nucleus expression gap into quintiles
   - Most-different genes (Q5): median $r = 0.78$, 46% above $r > 0.8$
   - Cell-type markers (Gad2, Plp1, Slc17a7): $r > 0.99$; IEGs remain hardest
3. **Visual**: `gene_stratification_overview.png` Panel B (bar chart by cell-nuc difference quintile) + Panel D (top 1K vs bottom 1K). Or: `expression_vs_diff_heatmap.png`

---

### Slide 12 — Training Dynamics Track Cascading Schedule

1. **Title**: Cross-Decode Quality Improves as Each Loss Activates
2. **Content**:
   - Correlation and shift improve in stages matching loss activation
   - Cross-recon onset (epoch 100) drives the largest quality jump
   - Final metrics stable by epoch 400
3. **Visual**: `cross_dec_gene_corr_over_time.png` and `cross_dec_shift_over_time.png` with vertical lines annotating loss activation epochs

---

### Slide 13 — Latent Space Aligns; Decoded Output Is Cell-Like

1. **Title**: UMAPs Confirm Alignment and Cross-Decode Quality
2. **Content**:
   - Encoded: cell types cluster cleanly, modalities overlap
   - Decoded: cross-decoded nuclei overlap with real cell profiles
   - Early → late epoch progression shows alignment converging
3. **Visual**: 2×2 UMAP grid (early vs late epoch × encoded vs decoded), colored by cell type + modality. Use `e599_valid_encoded.png`, `e599_valid_decoded.png`

---

## Section 4: How Good Is It Really? (5-6 min, 2 slides)

### Slide 14 — Reconstruction Ceiling: Cross-Decode Is Nearly Lossless

1. **Title**: Cross-Modal Translation Nearly Matches Same-Modality Reconstruction
2. **Content**:
   - Same-modality cell recon: $r = 0.84$, classif 93%, DE 0.55, coexpr 0.98
   - Cross-decode: classif 96%, DE 0.52 — within 3% of ceiling
   - Non-neuronal DE near-zero even in reconstruction — gene filtering, not model
3. **Visual**: Table: metric × {recon ceiling, cross-decode, raw nuclei}. Highlight how close cross-decode gets to ceiling. Annotate: "gap = gene filtering, not model"

---

### Slide 15 — Honest: Imputation Operates at Cell-Type Resolution

1. **Title**: Per-Cell Variance Is a General VAE Bottleneck
2. **Content**:
   - Within-CT variance: recon retains ~24%, cross-decode retains ~8%
   - $z$ ablation: replacing $z$ with CT mean changes output $<1\%$
   - Still useful: classification, DE, coexpression are CT-level analyses
3. **Visual**: `summary_MSE_cond.png` (4-panel hallucination test) or simplified bar: variance retained at each stage (raw → recon → cross-decode)

---

## Section 5: Gene Filtering Fix (3-4 min, 2 slides)

### Slide 16 — Global Detection Threshold Removes Non-Neuronal Markers

1. **Title**: 5% Global Filtering Drops 29 of 59 Cell-Type Markers
2. **Content**:
   - Config B (10,134 genes): removes Aqp4, Mog, Pdgfra, Flt1, C1qa, ...
   - Rare types (~1-5% of cells) fail global threshold
   - Directly explains near-zero DE for Astro, Endo, OPC, Oligo, Macrophage
3. **Visual**: `dotplot_present_markers.png` — cell type × marker presence/absence matrix

---

### Slide 17 — Fix: Per-Cell-Type Filtering Recovers 58/59 Markers

1. **Title**: PerCT 5% Filtering: 16K Genes, 93% Variance, Nearly All Markers
2. **Content**:
   - Keep gene if detected in >5% of **any** cell type in **either** modality
   - 16,439 genes; 58/59 markers recovered (only *Aldh1l1* missing)
   - Retraining in progress — expect non-neuronal DE to improve substantially
3. **Visual**: Comparison table: Config B vs PerCT 5% (genes, markers, variance%). Before/after marker recovery highlight

---

## Section 6: Path Forward (4-5 min, 3 slides)

### Slide 18 — Improving Per-Cell Variation

1. **Title**: Three Approaches to Recover Within-Cell-Type Variation
2. **Content**:
   - Larger latent (32 → 64/128 dims): more capacity for per-cell signal
   - Variance-preserving loss: $\mathcal{L}_{\text{var}} = \max(0, \tau - \text{Var}_{\text{imp}} / \text{Var}_{\text{real}})$
   - Read subsampling: create pseudo-paired data for per-cell supervision
3. **Visual**: Diagram showing 3 approaches as parallel branches, all feeding into improved model

---

### Slide 19 — Toward a Paper: Biological Validation

1. **Title**: What Does This Enable? — Validation Roadmap
2. **Content**:
   - Benchmark vs scVI batch-correction and per-gene linear regression
   - Biological test: recover cytoplasmic markers (Fos, Arc) from nuclei data
   - Human transfer: retrain on PsychENCODE or human Allen Brain Atlas
3. **Visual**: Roadmap timeline: current → mouse benchmark → human validation → publication

---

### Slide 20 — Summary

1. **Title**: Three Takeaways
2. **Content**:
   - Modality-conditioned VAE: $r = 0.98$, +11% classification, +5% DE over raw nuclei
   - Cross-decode nearly matches reconstruction ceiling; variance loss is general VAE property
   - Clear next steps: PerCT gene filtering, per-cell variation, human validation
3. **Visual**: Summary metrics table (3 models × key metrics) + reconstruction ceiling comparison

---

## Slide Count: 20 slides, ~32-38 min + discussion

---

## Available Figures

| Figure | Source | Slide |
|--------|--------|-------|
| Hallucination 4-panel (cond) | `notes/hallucination_tests/summary_MSE_cond.png` | 15 |
| Hallucination 4-panel (cond+xr) | `notes/hallucination_tests/summary_MSE_cond_xr.png` | 15 |
| z variance decomposition | `notes/hallucination_tests/B2_z_variance_MSE_cond.png` | 15 |
| Gene strat overview (cond) | `run/run_2026-04-02-02-02-35_3345839/evaluation/gene_stratification/gene_stratification_overview.png` | 11 |
| Gene strat overview (cond+xr) | `run/run_2026-04-02-10-51-32_3738876/evaluation/gene_stratification/gene_stratification_overview.png` | 11 |
| Expression × diff heatmap | `run/run_2026-04-02-10-51-32_3738876/evaluation/gene_stratification/expression_vs_diff_heatmap.png` | 11 |
| Top/bottom 30 genes | `run/run_2026-04-02-10-51-32_3738876/evaluation/gene_stratification/top_bottom_genes_crossdecode.png` | 11 |
| IEG/marker detail | `run/run_2026-04-02-10-51-32_3738876/evaluation/gene_stratification/ieg_marker_detail.png` | 11 |
| Marker dotplot | `notes/marker_gene_analysis/dotplot_present_markers.png` | 16 |
| Raw data UMAP | `notes/marker_gene_analysis/umap_configB_raw.png` | 3, 13 |
| Correlation heatmap | `run/.../evaluation/correlation/epoch_599/4x4_avg_correlation_heatmap.png` | 8 |
| Shift bar charts | `run/.../evaluation/correlation/epoch_599/4x4_decoded_*_shift_scores_bar.png` | 8 |
| Per-CT correlation grid | `run/.../evaluation/correlation/epoch_599/4x4_per_celltype_correlation_grid.png` | 8 |
| Cross-decode corr over time | `run/.../inline_eval/cross_dec_gene_corr_over_time.png` | 12 |
| Cross-decode shift over time | `run/.../inline_eval/cross_dec_shift_over_time.png` | 12 |
| UMAPs (epoch 599) | `run/.../umap/e599_valid_encoded.png`, `e599_valid_decoded.png` | 13 |
| UMAPs (early epoch) | `run/.../umap/e0_valid_encoded.png`, `e0_valid_decoded.png` | 13 |
| Classification reports | `notes/comprehensive_eval/classification_report_MSE_*.csv` | 9 |
| Gene filtering comparison | `notes/gene_filtering_analysis/strategy_comparison.csv` | 17 |
| Summary metrics table | `notes/comprehensive_eval/summary_table.csv` | 8, 20 |
| Recon eval DE (per-CT) | `notes/recon_eval/de_recon_MSE_*.csv` | 14, 16 |

## Figures to Generate

| Priority | Figure | Slide | Effort |
|----------|--------|-------|--------|
| **P1** | Architecture diagram (cond decoder + cross-decode) | 5 | Manual/draw |
| **P1** | Summary bar chart: classification (raw nuc vs 3 models) | 9 | Script |
| **P1** | Paired bar chart: per-CT DE (imputed vs raw nuc + recon ceiling) | 10 | Script |
| **P1** | Grouped bar chart: $r$ and $R^2$ across 3 models | 8 | Script |
| **P2** | Training timeline/Gantt (cascading losses) | 6 | Manual/draw |
| **P2** | Recon ceiling vs cross-decode comparison table/figure | 14 | Script |
| **P3** | Confusion matrix for best model | 9 | Script |
| **P3** | Variance retention waterfall (raw → recon → cross-decode) | 15 | Script |

## Anticipated Questions

1. **"How does this compare to scVI?"** → Planned benchmark, not done yet. scVI treats modality as batch; our approach explicitly models the translation.
2. **"Show me genes that actually matter"** → Slide 11: most-different genes have highest imputation quality. Markers $r > 0.99$.
3. **"Pseudobulk DE?"** → Our DE is cell-level Wilcoxon; pseudobulk needs donor replicates not available here.
4. **"Path to human?"** → Slide 19: PsychENCODE or human Allen Brain Atlas.
5. **"32 latent dims enough?"** → Part of per-cell variation plan (Slide 18).
6. **"How much does the VAE lose?"** → Slide 14: recon retains 93% classif, 0.55 DE. Cross-decode is 0.52 — almost no additional loss.
7. **"Is variance collapse your model's problem?"** → No — general VAE property. Recon retains only 22-26%. Motivates Slide 18 directions.
8. **"What about IEGs — the ultimate cytoplasmic genes?"** → Slide 11: IEGs are the weakest category ($r \approx 0.1\text{–}0.6$). Partially recovered but still the frontier.
