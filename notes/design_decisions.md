# Design Decisions

## Cross-Recons: Raw Counts + Proportion Matching (2026-03-23)

**Decision:** Stay on raw counts + NB model, match proportions not scale.

**Context:** Cross-recons is the critical ingredient for making decoded nuclei cell-like (probes confirmed 2026-03-20). The question arose whether switching to normalized data (log1p/CPM) would make it easier to match both scale and proportion.

**Why stay on raw counts:**
- Matching absolute scale would require switching to normalized data (log1p/CPM) + MSE-based VAE — effectively a different model
- The current NB cross-recons already handles proportions via library-size rescaling, which is the right level for raw count data
- Nuclei and cell library sizes differ ~3x (decoder scale: nuclei mean ~0.66, cell mean ~1.9), so absolute scale matching is impossible without normalization
- VAE_NB is statistically principled for scRNA-seq count data

**How this applies:**
- All cross-recons experiments should use the existing NB proportion-matching loss
- If results plateau, consider adding proportion-space variants (MSE on softmax proportions, cosine similarity) as lighter alternatives
- Do NOT switch the base model away from raw counts + NB without a strong reason

## Cross-Recons Goal: Distribution-Level Matching (2026-03-23)

**Goal:** Decoded nuclei should resemble cells at the *distribution* level, not the individual cell level. Even within a cell type, expression profiles vary — nuclei should shift toward the cell distribution but retain their own individual signatures/variation.

**Implications:**
- `nuclei_recons` must NOT be zero — it preserves sample-level diversity and nuclei-specific variation
- `cross_recons` should be moderate — enough to shift the population distribution but not collapse all nuclei into identical cell-like profiles
- This is fundamentally a distribution matching problem, not a point-wise matching problem
- The right evaluation metrics are UMAP overlap + scIB scores (iLISI, ASW, cLISI), not just shift/correlation

**Evaluation priorities:**
1. UMAP: visual overlap between decoded nuclei and decoded/raw cells
2. scIB metrics: iLISI (integration LISI — do modalities mix?), ASW (average silhouette width), cLISI (cell-type LISI — do cell types still separate?)
3. Shift scores and gene correlation (existing metrics)

**Key balance:** cross_recons pulls nuclei→cell distribution, nuclei_recons preserves individual variation. The ratio between these two is the main knob.

## Decoder Degradation in Stage 3 — Diagnosis (2026-03-24)

**Problem:** Stage 3 joint training degrades decoder output quality. UMAPs of decoded output lose clear cell type separation, even though the encoded latent space retains good clusters and modality alignment.

**Evidence:**
- Stage 1 cell-only model: both encoded and decoded UMAPs show excellent cell type separation
- F-series probes (stage 3): encoded UMAPs still show good cell type clusters + modality mixing, but decoded UMAPs are washed out — structure degraded, clusters less distinct
- Reconstruction losses (cell ~0.76, nuclei ~0.52) are not improving with weight tuning
- This holds across all F-series weight configurations (cross_recons 0.5–2.0, nuclei_recons 0.5–1.0)

**Root cause:** A single shared decoder serving competing objectives (cell_recons + nuclei_recons + cross_recons + JSD) is being pulled in too many directions. It compromises by producing "average-ish" outputs that satisfy all losses partially but none well.

**Proposed solutions (in priority order):**
1. **Modality-conditioned decoder** — add a modality embedding to the decoder input so it can learn modality-specific adjustments while sharing most parameters. Lightweight, preserves shared architecture.
2. **Separate decoder heads** — split decoder into modality-specific branches (at least final layers). More capacity, cleaner gradients. Heavier change.
3. **Remove JSD in stage 3** — alignment loss already regularizes the latent space; JSD=0.5 on top may over-compress. Quick test.
4. **Two-phase stage 3** — Phase 1: reconstruct both modalities (no alignment/cross-recons) to stabilize decoder. Phase 2: gradually introduce alignment + cross-recons.
5. **Freeze encoder after alignment converges** — once latent alignment is achieved (~epoch 50), freeze encoder and let decoder focus on reconstruction.

**Decision:** Pursue option 1 (modality-conditioned decoder) and option 4 (two-phase training) first.

## Validation Evaluation Added (2026-03-23)

Added validation set evaluation to stage3 training loop:
- Valid loss computed every epoch → `valid_epoch_loss/loss.txt`
- Valid inline eval (shift, gene corr, decoder scale, UMAPs, correlation) at eval epochs → `valid_inline_eval/`
- Valid UMAPs saved as `umap/e{N}_valid_*.png`
- Train vs valid overlay loss curves generated at end of training
- Uses train raw_means as reference for valid eval (not valid raw_means) — makes train vs valid metrics directly comparable since the only difference is the input data

## Switch to V3-Only Data (2026-03-31)

**Decision:** Train all models on v3 chemistry only, dropping v2 data entirely.

**Context:** Analysis of the combined v2+v3 data (`analysis/check_v2v3_batch_effect.py`) showed strong bimodal distributions in both library size and genes-per-cell, caused by the v2/v3 chemistry batch effect. V2 cells have ~3-5x lower UMI counts than v3, creating a confounding variable that the model would need to disentangle alongside the cell/nuclei difference.

**Evidence (from `analysis/filtering_cutoffs_v3/`):**
- V3-only data is clean: unimodal, roughly log-normal distributions for both cell and nuclei
- Cell (v3): 71,183 cells × 26,852 genes, median lib size 46K, median 7,282 genes/cell
- Nuclei (v3): 40,446 cells × 25,871 genes, median lib size 16K, median 3,446 genes/cell
- No dead/empty droplet tail in either modality — data is already high-quality
- Cell filtering barely matters (even 1st percentile has 7,268 UMI / 2,865 genes)

**How this applies:**
- All new training runs use `--protocol v3` to filter to v3 chemistry
- Previously trained models (Routes 1-3, probes A-F) also used v3 (--batch c3A/n3A), so they're still valid baselines
- V2 data is not discarded from disk, just excluded from training

## Config B Gene Filtering (2026-03-31)

**Decision:** Use detection_rate > 5% + mean_expression > 0.01 + exclude_mitocarta for gene filtering (Config B).

**Context:** Previous runs used detection_rate_threshold=1.0 which kept ~24.7K genes. Analysis showed ~35-40% of genes have <1% detection rate — pure noise that inflates the softmax dimension without contributing biological signal.

**Config B results (v3 only):**
- Cell: 13,658 genes → 51% of original, removes ~13K noise genes
- Nuclei: 11,048 genes → 43% of original
- Cell loss: <1% of cells filtered (99.8% survive at genes>1500, lib>2000)
- Nuclei loss: <1% of cells filtered (99.1% survive)

**Why Config B over A or C:**
- Config A (det>1%): too permissive, keeps 17.4K genes — still includes many noise genes
- Config C (det>5%, mean>0.1): too aggressive, only 8.7K nuclei genes — risks losing biological signal
- Config B is the sweet spot: halves the gene dimension while losing <1% of cells in either modality

**How this applies:**
- All new runs use `--detection_rate_threshold 5.0 --exclude_mitocarta True`
- Reduces softmax dimension from ~25K to ~11-14K genes
- Note: this changes the gene set, so new models are NOT directly comparable to old ones in absolute metrics — compare only within the same filtering regime

## MSE vs NB Reconstruction Loss Comparison (2026-03-31)

**Decision:** Run both vae_js (MSE on log1p) and vae_nb (NB on raw counts) with identical architecture, schedule, and filtering to determine which loss formulation works better.

**Context:** All previous runs used vae_nb with NB reconstruction loss on raw counts. The NB model is statistically principled for count data, but the softmax decoder creates a zero-sum bottleneck where dominant genes (Malat1, ribosomal) eat probability mass and suppress low-expression genes. An MSE loss on log1p data avoids this bottleneck entirely — each gene is reconstructed independently.

**What we're testing:**
- vae_js (MSE): `--model vae_js`, data is log1p-normalized, `--recons_loss_type 0`
- vae_nb (NB): `--model vae_nb`, raw counts, `--use_raw_data True --raw_data_normalization raw`
- Same architecture (1024→512→32 latent), same annealing schedule, same Config B filtering
- Same 2-stage setup: stage1 cell → stage2 joint (cell+nuclei) with cascading loss annealing

**Training schedule (same for both):**
- Stage 1 (cell only, 600 epochs): recons-only for 300 epochs, JSD anneal 300-400, both 400-600
- Stage 2 (joint, 600 epochs): cell always on; nuclei_recons ramps 0-100; nuclei_jsd delayed 100, ramps 100-200; alignment delayed 200, ramps 200-300

**Why this matters:**
- If MSE matches or beats NB, we can avoid the softmax bottleneck entirely
- If NB is clearly better, the bottleneck needs architectural fixes (which we've explored via conditioned decoder)
- The Config B filtering (halving gene count) should help NB more than MSE since it reduces the softmax dimension

## Cross-Recons for Conditioned Decoder (2026-04-02)

**Decision:** Add cross-reconstruction loss to conditioned decoder training, using the cleaner formulation enabled by modality embeddings.

**Context:** Two separate problems were identified:
1. **Decoder degradation** (shared decoder pulled by competing losses) → fixed by conditioned decoder
2. **No direct nuclei→cell supervision** (latent alignment doesn't translate to decoder output) → cross-recons was the key ingredient in Probe D/E/F-series

Previous cross-recons (shared decoder, NB) compared decoded nuclei (rescaled) to raw cells — a somewhat indirect signal since the decoder doesn't know about modalities. The conditioned decoder enables a cleaner cross-recons: encode nuclei, decode with cell embedding, compare to raw cell.

**Implementation:**
- `decode_cross_train(mu_source, target_id, ...)`: takes pre-computed mu from source forward pass, detaches it, decodes with target embedding. Gradients flow only through decoder + modality embedding, NOT back through encoder.
- MSE loss: `mse_loss(cross_decoded, cell_data)` — both in log1p space
- NB loss: NB log-likelihood of raw cell counts given cross-decoded mu (rescaled to cell library size)
- Weight annealing: cross_recons delayed to epoch 100 (after nuclei_recons ramps up), annealed over 100 epochs

**Why detach mu:** The encoder should learn representations that are good for reconstruction, not be distorted by cross-modal supervision. The cross-recons gradient should only teach the decoder how to translate a nuclei-like latent into cell-like output using the cell embedding.

**Key question:** Will this produce better shift scores than (a) conditioned decoder without cross-recons, and (b) shared decoder with cross-recons?

---

## Post-Training Eval Fixes for Gene List + Cond Decoder (2026-04-02)

**Problem:** Three bugs prevent post-training evaluation from working correctly on new runs:

### Bug 1: Gene alignment broken for Config B (gene_list) runs
- `eval_utils.load_raw_data()` filters raw data by `common_genes.csv` (25,802 genes) + `detection_rate_threshold` + `exclude_mitocarta`
- Config B training uses `--gene_list data/training_gene_list_v3.csv` → 10,134 genes
- In arguments.json: `gene_list="data/training_gene_list_v3.csv"`, `detection_rate_threshold=1.0`, `exclude_mitocarta=False` (gene_list skips those)
- Eval loads 25,802 raw genes, decoded has 10,134 → positional alignment `decoded[:, :10134]` maps to wrong raw genes
- **Fix:** Read `gene_list` from arguments.json and filter raw data to those exact genes in `load_raw_data()`

### Bug 2: Conditioned decoder runs not detected as stage3
- Eval code checks only `fine_tune_stage3_v2` to dispatch 3x3 (stage1) vs 4x4 (stage3) analysis
- Conditioned decoder runs set `fine_tune_cond_stage3=True` but `fine_tune_stage3_v2=False`
- **Fix:** Check both flags: `is_stage3 = run_args.get('fine_tune_stage3_v2', False) or run_args.get('fine_tune_cond_stage3', False)`

### Bug 3: Protocol/batch not passed to eval raw data loading
- `eval_utils.load_raw_data()` supports `protocol` and `batch` params, but `evaluate.py` and `eval_correlation.py` don't pass them from arguments.json
- Training uses `--protocol v3 --batch n3A` → only v3 cells in training data
- Eval loads ALL cells (v2+v3) → raw reference means include v2 batch effect
- **Fix:** Pass `protocol` and `batch` from arguments.json to raw data loading

### Affected runs
| Run type | Gene align | Stage detect | Protocol filter | Status |
|----------|-----------|-------------|-----------------|--------|
| NB old (~24.7K, no gene_list) | OK | OK | OK (v3 only already) | **Works** |
| MSE stage1 new (Config B) | **BROKEN** | OK (stage1) | Missing | **Needs fix** |
| MSE stage2 new (Config B) | **BROKEN** | OK (stage3_v2=True) | Missing | **Needs fix** |
| NB stage1/2 new (Config B) | **BROKEN** | OK | Missing | **Needs fix** |
| Cond decoder stage2 | **BROKEN** | **BROKEN** | Missing | **Needs fix** |
