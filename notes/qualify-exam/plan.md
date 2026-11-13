# Pre-Qual and Qualifying Exam Plan

## Context

Based on:
- Chang Lu's pre-qual (2 aims, 5 pages) and full prospectus (1 aim, 21 pages with results)
- ISMB 2026 reviews (6 reviewers, key gaps: baselines, circularity, generalization, novelty)
- Current project progress (multiple model variants, eval framework, CPM/normloss/conditioning experiments)
- Gerstein lab perspective (biological application, cross-species, gene stratification)

## Timeline

- **Pre-qual: TOMORROW**
- Full qualifying prospectus: ~2-3 months after pre-qual
- Qualifying exam: TBD

---

## Proposed Structure: 2 Aims

### Aim 1: ImputeVAE — Learning Structured Transformations Between Single-Nucleus and Single-Cell Expression

**Core thesis**: The nucleus-to-cell expression difference is a structured, gene-specific, learnable transformation — not random noise or a simple batch effect. A VAE with cross-modal decoding can learn this transformation and produce cell-like profiles from nuclei-only data.

**What you have (completed work)**:
- VAE architecture: shared encoder + conditioned decoder + cross-reconstruction
- MSE > NB finding (softmax bottleneck analysis)
- Cascading loss annealing for stable joint training
- Config B → PerCT_5pct gene filtering evolution
- Cell-type classification 95.8%, DE overlap improvement, gene correlation 0.96+

**What you need to add (directly from ISMB review gaps)**:

#### 1a. Proper baselines (addresses R1, R4, R6)
- **scVI as batch-correction baseline** — treat cell/nuclei as batches, show it fails because the difference is structured, not stochastic (SUC-8, scVI part still TODO)
- **Improved linear regression** — transcriptome-wide (not cell-type means), properly implemented
- **Harmony/Scanorama** — standard batch correction, show they also fail
- Framing: "existing methods assume stochastic noise; we show the transformation is systematic"

#### 1b. Address evaluation circularity (addresses R1, R5)
- **Held-out cell type evaluation** — train on N-1 cell types, evaluate on the held-out type → proves the model learns generalizable transformation, not just memorized alignment
- **Per-cell metrics** — within-cell-type correlation, variance retention (SUC-6 work)
- **Gene-stratified evaluation** — show model works on genes with LARGE cell-nuc gap, not just housekeeping genes (SUC-9, Gerstein concern #3)

#### 1c. Ablation study (addresses R4, R6)
- JSD vs KL divergence for latent alignment
- Conditioned vs shared decoder
- With/without cross-reconstruction
- Scalar vs learned modality embedding
- CPM vs log1p normalization
- You already have most of these runs — just need to present them in a table

#### 1d. Variance preservation (addresses R5, Gerstein concern)
- Current status: models collapse to cell-type means (3-11% variance retained)
- Normalized loss experiments (ongoing)
- Full cVAE (encoder + decoder conditioning) experiments (ongoing)
- This is an honest, open challenge — good for a qual to show you understand the limitation and have a plan

**Framing for the qual**: This aim is a complete methods paper. The ISMB abstract was a snapshot; the qual version includes rigorous baselines, ablations, proper evaluation, and honest discussion of variance collapse.

---

### Aim 2: Generalization and Biological Validation of Cross-Modal Translation

**Core thesis**: The learned nucleus-to-cell transformation generalizes across tissues and species, and produces biologically meaningful imputed profiles that improve downstream analyses.

This aim addresses the "so what?" question and the generalization concerns from reviewers and Gerstein.

#### 2a. Cross-tissue characterization (SUC-12)
- Characterize how the nucleus-cell difference varies across brain regions, tissues, species
- Use public datasets (Allen Brain, PsychENCODE, Human Cell Atlas)
- Key question: is the transformation conserved enough to transfer, or tissue-specific?
- This is a biological finding in itself — publishable even if transfer fails

#### 2b. Cross-species transfer (SUC-11, addresses R4)
- Train on mouse AIBS → apply to human brain nuclei
- Options: direct transfer, fine-tuning with limited human paired data, zero-shot on shared orthologs
- PsychENCODE connection (2.8M human nuclei) — high-impact if it works
- Even negative results are informative (the transformation is species-specific → biology)

#### 2c. Biological validation / downstream application (SUC-10, addresses Gerstein concern #1)
- **Concrete use case**: recover cytoplasmic RNA signatures (Fos, Arc, mitochondrial genes) from nuclei
- **Cell-type eQTL improvement**: imputed profiles → better power for eQTL mapping?
- **DE analysis improvement**: show that imputed nuclei recover cell-type DE genes missed by raw nuclei
- **Per-cell-type quality as biology**: where imputation fails = biology of nuclear vs cytoplasmic RNA (Gerstein concern #2)

#### 2d. Tool release
- CLI: input snRNA-seq h5ad → output imputed cell-like expression matrix
- Pretrained models for mouse brain (and human if transfer works)
- Needed for Nature Methods-level publication

**Framing for the qual**: This aim transforms imputeVAE from a method into a biological resource. It tests whether the learned transformation is a universal property of the cell-nucleus relationship or dataset-specific.

---

## How This Maps to ISMB Review Concerns

| ISMB Concern | Where Addressed |
|-------------|-----------------|
| Weak baselines | Aim 1a (scVI, linear reg, Harmony) |
| Circular evaluation | Aim 1b (held-out cell types, per-cell metrics) |
| Single dataset | Aim 2a-b (cross-tissue, cross-species) |
| Incremental novelty | Aim 1 framing (structured transformation thesis) + Aim 2 (generalization) |
| No ablations | Aim 1c (full ablation table) |
| Variance collapse | Aim 1d (ongoing work, honest limitation + plan) |
| No biological validation | Aim 2c (downstream use case) |

## How This Maps to Gerstein Concerns

| Gerstein Concern | Where Addressed |
|-----------------|-----------------|
| Biological application | Aim 2c (cytoplasmic RNA recovery, eQTL) |
| Per-cell-type breakdown | Aim 1b, 2c |
| Gene-level analysis (hardest genes) | Aim 1b (gene-stratified eval) |
| Cross-species transfer | Aim 2b |
| Benchmarking | Aim 1a |

## How This Maps to Existing Linear Issues

| Linear Issue | Qual Section | Status |
|-------------|-------------|--------|
| SUC-5 (PerCT_5pct) | Aim 1 (gene filtering) | Done |
| SUC-6 (Variance) | Aim 1d | In Progress (CPM/normloss/cVAE experiments) |
| SUC-7 (Publication) | Overall qual → paper pipeline | Backlog |
| SUC-8 (Baselines) | Aim 1a | Linear reg done, scVI TODO |
| SUC-9 (Gene-stratified eval) | Aim 1b | Blocked by SUC-13 |
| SUC-10 (Bio validation) | Aim 2c | Backlog |
| SUC-11 (Cross-species) | Aim 2b | Backlog |
| SUC-12 (Cross-tissue) | Aim 2a | Backlog |
| SUC-13 (Eval refactor) | Aim 1 infrastructure | Phase 1 done |

---

## Pre-Qual Proposal Plan (~5-6 pages, following Chang's format)

### Outline

1. **Introduction** (~0.5 page)
   - snRNA-seq is practical but systematically different from scRNA-seq
   - Existing methods treat this as batch effect (wrong assumption)
   - We propose learning the structured transformation

2. **Background** (~1 page)
   - Cell vs nucleus biology (cytoplasmic RNA, library size, detection)
   - Existing approaches: batch correction (scVI, Harmony), pseudobulk aggregation
   - Why they fail: the difference is gene-specific and structured, not stochastic
   - VAEs for single-cell data (scVI, totalVI, LDVAE)

3. **Aim 1: Framework** (~1.5 pages)
   - Architecture (conditioned decoder, cross-reconstruction)
   - Training pipeline (3-stage cascading annealing)
   - Preliminary results: classification, DE overlap, gene correlation
   - Planned: baselines, ablations, held-out cell type eval, variance work

4. **Aim 2: Generalization** (~1.5 pages)
   - Cross-tissue/species characterization and transfer
   - Biological validation (cytoplasmic RNA recovery, downstream use case)
   - Tool release plan

5. **References** (~1 page)

### Key figures for pre-qual
- Architecture diagram (encoder → shared latent → conditioned decoder)
- UMAP showing raw nuclei vs imputed vs real cells
- Bar chart: classification / DE overlap across model variants + baselines
- Gene-stratified correlation (if available by then)

---

## Priority Order for Experiments Before Pre-Qual

1. **scVI baseline** (Aim 1a) — highest impact, directly addresses strongest review criticism
2. **Complete conditioning experiments** (Aim 1d) — already running, just need to wait + analyze
3. **Held-out cell type eval** (Aim 1b) — new experiment but straightforward
4. **Ablation table** (Aim 1c) — mostly compilation of existing runs
5. **Gene-stratified eval** (Aim 1b) — needs SUC-13 eval framework

Items 1-4 should be achievable before pre-qual. Item 5 can be listed as "planned."
Aim 2 work is mostly "proposed" for the qual — pre-qual just needs the plan, not results.
