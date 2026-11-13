# Lab Meeting Presentation — Revised Outline v2
# Date: 2026-04-03, 30-45 min + discussion

## Narrative Arc

**Problem** → **Why it's hard** → **Our approach** → **It works (with evidence)** → **How good is it really? (honest assessment)** → **What broke and how we fixed it** → **What's next (toward a paper)**

The story: single-nucleus RNA-seq is everywhere (PsychENCODE: 2.8M nuclei) but misses cytoplasmic RNA. We built a VAE that translates nucleus → cell-like expression. It genuinely improves cell-type classification (+11%) and DE recovery (+5%). Cross-decode nearly matches same-modality reconstruction quality (DE: 0.52 vs 0.55 ceiling), showing the translation is almost lossless. We rigorously tested what it learns — it imputes at cell-type resolution, and per-cell variance loss is a general VAE property (reconstruction retains only 22-26%). We diagnosed a gene filtering problem affecting non-neuronal types, have a fix in progress, and have a clear path toward biological validation and a paper.

---

## Section 1: The Problem (4-5 min, 3 slides)

### Slide 1 — Title
- Project title, name, date
- One-line summary: "Imputing cell expression from single-nucleus RNA-seq using a modality-conditioned VAE"

### Slide 2 — snRNA-seq is powerful but incomplete
- Why nuclei? Frozen tissue, clinical samples, large-scale consortia
- Concrete scale: PsychENCODE 2.8M nuclei from 388 human brains (Gerstein co-author, Science 2024)
- What's missing: cytoplasmic mRNA — activity-dependent genes (Fos, Arc), secreted factors, mitochondrial genes
- Cell-vs-nucleus gap: per-gene correlation ~0.82 at population level
- **Visual**: schematic of cell with nuclear vs cytoplasmic RNA pools, or scatter of nuc-vs-cell mean expression with gap highlighted

### Slide 3 — Can we computationally recover cell expression from nuclei?
- The question: given a nucleus expression profile, predict what the whole-cell profile would look like
- Dataset: Allen Brain Atlas mouse visual cortex, 17 cell types, ~71K cells + ~40K nuclei (v3 chemistry), unpaired
- Challenge: no matched cell-nucleus pairs — must learn the translation from population-level correspondence
- **Visual**: dataset summary (cell types, sample sizes). Simple diagram: nucleus profile → ? → cell profile

---

## Section 2: Approach (5-7 min, 4 slides)

### Slide 4 — VAE refresher (1 min)
- Encoder compresses high-dimensional expression → low-dimensional latent z
- Decoder reconstructs expression from z
- Trained to minimize reconstruction error + KL regularization
- **Visual**: simple VAE diagram (input → encoder → z → decoder → output)

### Slide 5 — Key idea: modality-conditioned decoder
- Shared encoder: both cells and nuclei map to same latent space
- Decoder receives z + modality embedding (learned 8-dim vector: cell=0, nucleus=1)
- Same decoder, different "mode" — learns modality-specific expression patterns
- **The trick**: encode a nucleus → decode with the CELL embedding → cell-like output
- **Visual**: architecture diagram. Emphasize the embedding swap with an arrow/highlight

### Slide 6 — Training strategy: staged with cascading losses
- Stage 1: pretrain encoder+decoder on cell data only (600 epochs)
- Stage 2: joint training — introduce nuclei data with cascading loss annealing:
  - Epoch 0: cell reconstruction always on
  - Epoch 0-100: nuclei reconstruction ramps up
  - Epoch 100-200: cross-reconstruction + nuclei JSD ramp up
  - Epoch 200-300: alignment JSD ramps up
- Why cascading: prevents decoder degradation (earlier attempts with simultaneous losses failed — showed at previous lab meeting)
- **Visual**: timeline/Gantt showing loss components activating sequentially

### Slide 7 — Three models compared
- MSE shared decoder (baseline): single decoder, no modality awareness
- MSE conditioned decoder: shared encoder + modality embedding, no cross-recons loss
- MSE conditioned + cross-recons: adds explicit cross-modal supervision
- Also tested NB loss — MSE on log1p wins (brief mention, 1 sentence: softmax bottleneck in NB distorts proportions)
- **Visual**: simple table with 3 models × key architectural differences

---

## Section 3: Results — It Works (10-12 min, 6 slides) ← core of the talk

### Slide 8 — Conditioned decoder is the key unlock
- Shared decoder: corr to real cell = 0.788, R² = -1.1 (worse than predicting the mean!)
- Conditioned decoder: corr = 0.941, R² = 0.56
- + cross-recons: corr = 0.979, R² = 0.95
- The modality embedding is what makes cross-modal translation possible
- **Visual**: bar chart of per-gene correlation and R² across 3 models. Dramatic jump from shared → conditioned

### Slide 9 — Classification: raw nuclei 85% → imputed nuclei 96%
- Train logistic regression on real cell data, predict cell type of:
  - Raw nuclei: 85.1% accuracy
  - Imputed nuclei (cond+xr): 95.8% accuracy (+10.7%)
- The imputed data has BETTER cell-type signal than raw nuclei
- Per-cell-type: consistent improvement across neuronal types
- **Visual**: bar chart (raw nuc vs 3 models). Optionally: confusion matrix for best model showing clean diagonal

### Slide 10 — DE gene recovery exceeds raw nuclei
- Top-100 DE genes (one-vs-rest Wilcoxon): imputed overlap 52% vs raw nuclei 47%
- Per-cell-type: 11/17 types improved, some dramatically (Vip +40%, L6 IT +35%, Lamp5 +30%)
- 5 non-neuronal types near-zero → lead-in to gene filtering discussion (Slide 16)
- Important context: same-modality reconstruction DE ceiling is 0.53-0.56 — cross-decode at 0.52 is nearly matching the ceiling, showing the model translates almost as well as it reconstructs
- Coexpression: imputed 0.915 vs raw nuclei 0.818
- **Visual**: paired bar chart per cell type — imputed (blue) vs raw nuclei (gray), sorted by improvement. Add reconstruction ceiling line. Highlight failing non-neuronal types in red with annotation "gene filtering"

### Slide 11 — Which genes improve? (preemptive answer to "is the correlation trivial?")
- NOTE: this analysis may need to be generated — check if we have per-gene R² data
- Strategy: stratify genes by cell-vs-nucleus expression difference
  - Genes with large cell-nuc gap: these are the ones that MATTER for imputation
  - Genes with small gap: easy to predict, inflate overall metrics
- Show that the model adds value specifically on genes that differ between modalities
- If we can't generate this in time: show specific marker genes (Slc17a7, Gad1, Snap25, Mbp) and their per-gene imputation quality from existing correlation data
- **Visual**: scatter or boxplot of per-gene R² stratified by cell-nuc difference quintile. Or: gene-level examples with expression barplots

### Slide 12 — Training dynamics
- Show how cross-decode quality improves as loss components come online
- Correlation and shift improve in stages matching the cascading schedule
- Demonstrates that the training strategy is working as designed
- **Visual**: `cross_dec_gene_corr_over_time.png` and `cross_dec_shift_over_time.png`. Annotate when each loss activates

### Slide 13 — Latent space and UMAPs
- Encoded: cell types cluster cleanly, cell and nuclei overlap (alignment works)
- Decoded: cross-decoded nuclei produce cell-like outputs
- Show progression: early epoch (poor alignment) → late epoch (good alignment)
- **Visual**: 2×2 grid: encoded (early vs late) or encoded vs decoded, colored by cell type and modality. Use `e599_valid_encoded.png` and `e599_valid_decoded.png`

---

## Section 4: How Good Is It Really? (5-6 min, 2 slides)

### Slide 14 — Reconstruction sanity check: the model preserves signal faithfully
- Before evaluating cross-decode (the imputation task), check: does the VAE preserve information during same-modality reconstruction?
- All 3 models reconstruct equally well — conditioned decoder and cross-recons don't hurt:
  - Cell recon per-cell corr: 0.842-0.843 (all models identical)
  - Cell recon classification: 93% (vs 100% raw — small loss through bottleneck)
  - Cell recon DE overlap: 0.53-0.56 (ceiling for what cross-decode can achieve)
  - Cell recon coexpression: 0.977-0.981
- **Key point**: the reconstruction ceiling for DE is ~0.55, and our cross-decode achieves 0.52 — nearly matching the ceiling. The model is doing almost as well at cross-modal translation as it does at same-modality reconstruction.
- Non-neuronal DE is near-zero even in same-modality cell reconstruction (Astro 0.07, Endo 0.06) — proves this is a gene filtering issue, not a cross-decode failure
- **Visual**: table comparing reconstruction ceiling vs cross-decode performance; highlight how close cross-decode gets

### Slide 15 — Honest assessment: what the model learns
- Variance test: imputed cells have 3-11% of real cell variance within each cell type
- But this is a general VAE property! Same-modality reconstruction also retains only 22-26% of cell variance
  - Cross-decode retains ~60-70% of the reconstruction variance (not 100%, but not nothing)
- z ablation: replacing each nucleus's z with cell-type mean z changes output by <1% → model operates at cell-type resolution
- **Interpretation**: the model learns an accurate mapping from cell-type identity → cell-type expression profile. Per-cell variation is partially encoded but weakly decoded — a general VAE bottleneck issue, not specific to our cross-modal approach.
- **Framing**: this is honest, and it's still genuinely useful — classification, DE, and coexpression are cell-type-level analyses
- **Visual**: `summary_MSE_cond.png` (4-panel figure) + variance comparison: raw → recon → cross-decode showing progressive loss

---

## Section 5: Gene Filtering Discovery (3-4 min, 2 slides)

### Slide 16 — Gene filtering removes non-neuronal markers
- Config B (5% global detection threshold): 10,134 genes, but removes 29/59 brain cell-type markers
- Missing: Pvalb, Aqp4, Mog, all OPC markers, all endothelial markers, most microglia markers
- Why: cell-type-specific markers for rare populations (~1-5% of cells) fail global detection threshold
- This directly explains near-zero DE gene overlap for Astro (0.04), Endo (0.02), Oligo (0.06), OPC (0.06), Macrophage (0.07)
- Confirmed by recon eval: even same-modality cell reconstruction has Astro DE = 0.07, Endo = 0.06 — the model simply can't find DE genes that aren't in its gene set
- This is a data filtering artifact, not a model failure — and we have a fix (next slide)
- **Visual**: `dotplot_present_markers.png` — which markers are in/out. Or table: cell type × markers present/absent

### Slide 17 — Fix: per-cell-type filtering, retraining in progress
- PerCT_5pct: keep gene if detected in >5% of ANY cell type in EITHER modality
- Result: 16,439 genes, 58/59 markers recovered (only Aldh1l1 still missing), 93% variance captured
- Stage 1 retraining submitted and running (job 3134209)
- Expected impact: non-neuronal DE should improve dramatically
- **Visual**: strategy comparison table (genes, markers, variance). Before/after marker recovery

---

## Section 6: Path Forward (4-5 min, 2-3 slides)

### Slide 18 — Improving per-cell variation (technical roadmap)
- Priority approaches:
  1. Larger latent space (currently 32-dim, ~80% encodes cell-type identity → try 64/128)
  2. Variance-preserving regularization (penalize when imputed variance << real variance)
  3. Read subsampling for pseudo-paired data (subsample cells → simulate nuclei → train with per-cell supervision)
- These are complementary — can combine
- **Visual**: bullet list or simple diagram of 3 approaches

### Slide 19 — Toward a paper: biological validation
- What Gerstein will ask: "what does this ENABLE?"
- Planned validation:
  1. Benchmark vs baselines: scVI (treat modality as batch), per-gene linear regression
  2. Gene-level stratification: show model adds value on genes with largest cell-nuc gap (the hard/useful ones)
  3. Specific biological example: recover cytoplasmic markers (Fos, Arc, mito genes) from nuclei
  4. Human validation: retrain/fine-tune on human Allen Brain Atlas or PsychENCODE paired data
- Publication target: Bioinformatics/PLOS Comp Bio (method + mouse benchmark), with path to Genome Biology if human validation works
- **Visual**: roadmap diagram or timeline

### Slide 20 — Summary
- Three takeaways:
  1. Modality-conditioned VAE enables cross-modal imputation: nuc → cell with 0.979 correlation, +11% classification, +5% DE — and cross-decode nearly matches the same-modality reconstruction ceiling (0.52 vs 0.55 DE)
  2. Honest assessment: imputation operates at cell-type resolution. Per-cell variance loss is a general VAE property (recon retains ~24%), not specific to cross-modal translation. This is already useful for classification, DE, and coexpression.
  3. Clear path forward: better gene filtering (in progress, recovers 58/59 markers), per-cell variation improvement (3 approaches), biological validation, human transfer
- **Visual**: clean summary table with key numbers + reconstruction ceiling comparison

---

## Total: 20 slides, ~32-38 min + discussion

## Available Figures (already generated)

| Figure | Source | For slide |
|--------|--------|-----------|
| Hallucination 4-panel (cond) | `notes/hallucination_tests/summary_MSE_cond.png` | 14 |
| Hallucination 4-panel (cond+xr) | `notes/hallucination_tests/summary_MSE_cond_xr.png` | 14 |
| z variance decomposition | `notes/hallucination_tests/B2_z_variance_MSE_cond.png` | 14 |
| Marker dotplot | `notes/marker_gene_analysis/dotplot_present_markers.png` | 16 |
| Raw data UMAP | `notes/marker_gene_analysis/umap_configB_raw.png` | 3 or 13 |
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

| Priority | Figure | For slide | Effort |
|----------|--------|-----------|--------|
| **P1** | Architecture diagram (conditioned decoder + cross-decode) | 5 | Manual/draw |
| **P1** | Summary bar chart: classification accuracy (raw nuc vs 3 models) | 9 | Script, quick |
| **P1** | Paired bar chart: per-CT DE overlap (imputed vs raw nuc) | 10 | Script, quick |
| **P1** | Summary bar chart: R² across 3 models | 8 | Script, quick |
| **P2** | Per-gene R² stratified by cell-nuc difference | 11 | Script, moderate — need per-gene data |
| **P2** | Training timeline/Gantt (cascading losses) | 6 | Manual/draw |
| **P3** | Confusion matrix for best model | 9 | Script, quick if we have predictions |
| **P3** | Conceptual diagram: CT-level vs per-cell imputation | 15 | Manual/draw |

## Anticipated Questions from Gerstein / Lab

1. "How does this compare to scVI treating cell/nucleus as a batch effect?" → Planned benchmark, not done yet
2. "Show me genes where this actually matters — the ones that differ most" → Slide 11 (need to generate or use specific examples)
3. "What about pseudobulk DE?" → Our DE is cell-level Wilcoxon; pseudobulk needs donor replicates we don't have in this setup. Acknowledge limitation.
4. "Path to human?" → Slide 19: retrain on human Allen Brain Atlas or PsychENCODE
5. "Is 32 latent dims enough for 16K genes?" → Part of per-cell variation improvement plan
6. "What changed since your last presentation?" → Staged training works (UMAPs now good), conditioned decoder is the key insight, have systematic evaluation
7. "How much does the VAE itself lose during reconstruction?" → Slide 14: same-modality recon retains 0.842 per-cell corr, 93% classification, 0.55 DE overlap. Cross-decode is 0.52 DE — almost no additional loss from the cross-modal translation itself.
8. "Is the variance collapse a problem specific to your model?" → No — it's a general VAE bottleneck property. Even same-modality reconstruction retains only 22-26% of within-CT variance. This motivates the larger latent / variance regularization directions.
