# Pre-Qual Slide Outline (~38 slides + 5 backup, 45 min)

Placeholders for figures marked with [FIGURE]. Gray italic text = speaker notes / talking points.

---

## Opening (3 slides)

### Slide 1: Title
- Title: "ImputeVAE: Translating Single-Nucleus to Single-Cell Expression Profiles via Cross-Reconstructive Variational Autoencoders"
- Suchen Zheng
- Pre-Qualifying Exam, [Date]
- Gerstein Lab · Computational Biology and Biomedical Informatics · Yale University

### Slide 2: Committee
- List each member: Name, Department, Role (Chair, Advisor, etc.)

### Slide 3: Outline
- Background (slides 4–10)
- Aim 1: Cross-reconstructive VAE framework (slides 11–25)
  - Architecture, training, preliminary results
- Aim 2: Cross-species generalization & biological validation (slides 26–34)
  - Cross-dataset, cross-species, human application
- Timeline & summary (slides 35–38)

---

## Background (7 slides, ~8 min)

### Slide 4: snRNA-seq is now dominant
**Key message:** Most new brain datasets are nuclei-only — and that's a practical necessity, not a choice.
- **Why nuclei?** Frozen, archived, postmortem tissue → cells lyse during dissociation → only nuclei survive
- [FIGURE: simple 2-panel schematic — Left: fresh tissue → dissociate → intact cells (scRNA-seq); Right: frozen/postmortem tissue → dissociate → cells lyse, nuclei survive (snRNA-seq)]
- **Scale of nuclei-only data:**
  - PsychENCODE: 2.8M human brain nuclei
  - Human Cell Atlas brain, SEA-AD (Alzheimer's), ROSMAP
  - CZ CELLxGENE: 169M+ cells across 1,550 datasets, large fraction nuclei-only
- **The problem:** all these datasets are locked out of analyses designed for whole-cell data
- Cite: Ding et al. 2020, Hu et al. 2023
- *Talking point: for brain, snRNA-seq is effectively the only option for most samples — frozen surgical tissue, postmortem donors, biobank archives. The question isn't whether to use nuclei, it's what to do with the data we get from them.*

### Slide 5: But nuclei ≠ cells
**Key message:** The differences are structured, not random.
- [FIGURE: cartoon of a cell — nucleus contains nuclear RNA, cytoplasm contains mRNAs (Fos, Arc, ribosomal) that are missing from snRNA-seq]
- Three key properties of the gap:
  - **Gene-specific**: cytoplasmic transcripts (Fos, Arc) depleted; housekeeping genes unchanged
  - **Cell-type-dependent**: neurons have far more cytoplasmic mRNA (dendritic transport, local translation) than glia → bigger gap
  - **Directional**: cytoplasmic genes are systematically *lower* in nuclei, not randomly scattered
- Cite: Bakken et al. 2018, Ding et al. 2020

### Slide 6: The gap varies by cell type
**Key message:** One global correction can't fix this.
- [FIGURE: slide6_ct_gap_bar.png — Pearson r (cell vs nuclei mean expression) per cell type, neuronal=blue, glial=green]
- Neurons: high cytoplasmic mRNA traffic (dendritic transport, local translation in axons/dendrites) → large fraction of transcriptome missed by snRNA-seq → lower correlation
- Glia: less active cytoplasmic mRNA localization → smaller gap → higher correlation
- Correlation ranges from r=0.58 (OPC) to r=0.86 (Vip) — each cell type has a different cell-nuclei relationship
- This cell-type dependence is why a single global correction can't work
- *Talking point: if you apply one correction factor, you either under-correct neurons or over-correct glia. The transformation must be cell-type-aware.*

### Slide 7: Current approach 1 — Batch correction
**Key message:** These tools assume the wrong kind of variation.
- Harmony, scVI, Scanorama: learn a shared latent space, remove variation correlated with the batch/modality label
- Core assumption: batch variation is **stochastic and global**
- Nucleus-cell gap violates this — it's gene-specific, cell-type-dependent, directional
- [FIGURE: conceptual 2-panel diagram — Left: what batch correction assumes (random scatter between batches), Right: what nuc-cell actually looks like (structured, directional shift)]
- Result: under-correct (structured patterns remain) or over-correct (remove real biological signal)
- Cite: Korsunsky 2019, Lopez 2018

### Slide 8: Current approach 2 — Pseudobulk
**Key message:** Captures means but loses per-cell information.
- Average expression within each cell type → compute per-gene correction factors
- Works for cell-type-level analyses (e.g., pseudobulk DE)
- Three limitations:
  - Discards all within-cell-type variation (cell states, gradients, bursting)
  - Cannot produce per-cell imputed profiles
  - Assumes linear relationship between nuc and cell expression
- Cite: Crowell et al. 2020
- *Talking point: this is our simplest baseline — linear regression does surprisingly well at the mean level*

### Slide 9: Our key insight
**Key message:** Frame the problem differently.
- Two-line comparison (make this visually prominent):
  - **Batch correction**: "make these look the same" → removes differences
  - **ImputeVAE**: "learn the mapping from one to the other" → models differences
- The nucleus-to-cell gap is not noise — it is a structured biological transformation driven by RNA localization
- Goal: given a nucleus expression profile, generate what the whole-cell profile would have looked like
- *Talking point: this reframing is the core intellectual contribution*

### Slide 10: Specific Aims
**Key message:** Clean overview of both aims.
- **Aim 1**: Develop a cross-reconstructive VAE framework that learns the structured transformation from snRNA-seq → scRNA-seq profiles
  - Train on paired mouse brain data, evaluate with multiple metrics
- **Aim 2**: Test cross-species generalization and demonstrate biological utility
  - Cross-dataset (Ding et al.), cross-species (macaque), human application (PsychENCODE)
- *Talking point: Aim 1 builds the tool, Aim 2 asks a biological question — is RNA localization conserved?*

---

## Aim 1: Framework (6 slides, ~8 min)

### Slide 11: Section divider — "Aim 1: A Cross-Reconstructive VAE Framework"

### Slide 12: Architecture overview
**Key message:** Three components, each with a clear purpose.
- [FIGURE: architecture diagram — MUST HAVE. Show:
  - Cell data and nucleus data both entering the same encoder
  - Latent space z with JSD alignment arrows between cell-type clusters
  - Conditioned decoder receiving z + modality indicator
  - Normal reconstruction paths (cell→cell, nuc→nuc)
  - Cross-reconstruction path HIGHLIGHTED: nuc encoded → decoded with cell condition → compared to real cell]
- Walk through the three components:
  1. **Shared encoder** → common latent space (cell-type identity preserved across modalities)
  2. **Conditioned decoder** → modality indicator lets decoder specialize per modality
  3. **Cross-reconstruction** → encode nuc, decode as cell, compare to real cells (direct supervision)

### Slide 13: Why a conditioned decoder?
**Key message:** This is an ablation result — show the evidence.
- Problem: a shared (unconditioned) decoder must simultaneously:
  - Reconstruct cells accurately
  - Reconstruct nuclei accurately
  - Cross-reconstruct nuclei → cells
  - These objectives compete → degraded performance
- [FIGURE: side-by-side UMAP — Left: shared decoder output (washed out, poor cluster separation), Right: conditioned decoder output (clean, well-separated clusters)]
- Solution: concatenate a modality indicator to decoder input → decoder knows which modality to produce
- *Talking point: this is one of our ablation findings — the conditioned decoder was necessary for the model to work*

### Slide 14: Two-stage training with cascading annealing
**Key message:** We can't throw all losses in at once — they compete.
- **Stage 1**: Pretrain on cell data only
  - Reconstruction loss + JSD divergence (weight annealed 0→1)
  - Purpose: establish a well-structured latent space before adding complexity
- **Stage 2**: Joint training on both modalities
  - [FIGURE: horizontal timeline/Gantt showing cascading activation:
    - Cell recons: ON from epoch 0 (always)
    - Nuclei recons: ramp 0→1 over epochs 0–100
    - Nuclei JSD: ramp 0→1 over epochs 100–200
    - Alignment: ramp 0→1 over epochs 200–300
    - Cross-recons: ramp 0→1 over epochs 100–200
    - Each stays on at full weight once activated]
- *Talking point: without cascading, nuclei losses interfere with cell reconstruction early on*

### Slide 15: Cross-reconstruction — the key ingredient
**Key message:** This is what makes ImputeVAE different from standard VAE integration.
- Mechanism: encode nucleus → decode with cell condition → MSE against real cell expression
- This provides **direct supervision** for the nucleus→cell translation
- Without it: model reconstructs each modality separately, but nothing forces the cross-modal mapping to be correct
- With it: the model explicitly learns "what should nucleus X look like as a cell?"
- [FIGURE: schematic — nucleus sample → encoder → z → decoder(cell condition) → output → loss arrow to real cell sample]
- *Talking point: this is the single most impactful design choice — ablation shows it gives the largest metric improvement*

### Slide 16: Training data
**Key message:** Describe the dataset clearly.
- Allen Brain Cell Atlas — mouse cortical tissues
- 10X Chromium v3 chemistry
- 71,000 cells + 40,000 nuclei
- 17 cell types (excitatory/inhibitory neurons, astrocytes, oligodendrocytes, microglia, etc.)
- 16,439 genes (PerCT_5pct: keep genes detected in ≥5% of cells in any cell type)
  - Previous global filtering (Config B, 10,134 genes) missed 29/59 known cell-type markers
  - PerCT_5pct recovers 58/59
- *Talking point: gene filtering matters — a per-cell-type threshold captures cell-type-specific markers that global thresholds miss*

---

## Aim 1: Preliminary Results (8 slides, ~10 min)

### Slide 17: Section divider — "Aim 1: Preliminary Results"

### Slide 18: UMAP — raw nuclei vs imputed vs real cells
**Key message:** Visual proof that imputation works.
- [FIGURE: 3-panel UMAP, all colored by cell type:
  - Left: raw nuclei (shifted from cells, modality gap visible)
  - Center: cross-decoded / imputed profiles
  - Right: real cells
  - Imputed profiles should overlap with real cells, NOT raw nuclei]
- One sentence: "Imputed profiles cluster with real cells, not with raw nuclei"

### Slide 19: Cell-type classification accuracy
**Key message:** Quantitative improvement.
- [FIGURE: bar chart with 4 bars, Y-axis starting from 0 (not 80%!):
  - Raw nuclei: 85.1%
  - Linear regression baseline: 94.7%
  - ImputeVAE (cond, no xr): 93.7%
  - ImputeVAE (cond+xr): **95.8%**]
- Highlight: cond+xr beats linear regression baseline
- Cross-reconstruction adds +2.1% over conditioned decoder alone
- *Talking point: Y-axis from 0 is important — reviewers flagged misleading axis in our ISMB submission*

### Slide 20: DE gene recovery
**Key message:** Imputed profiles recover cell-like differential expression patterns.
- Method: compute top DE genes (cell type vs rest) in imputed profiles, compare overlap with DE genes from real cells
- [FIGURE: grouped bar chart — DE overlap % for raw nuclei, linear regression, cond, cond+xr]
- Key numbers: cond+xr DE overlap 54.7% (best), +5.1% over raw nuclei
- Gene-level correlation: Pearson r = 0.963

### Slide 21: Coexpression structure
**Key message:** Not just means — the gene-gene correlation structure is preserved.
- [FIGURE: 5×5 coexpression heatmaps side by side:
  - Real cells
  - Cross-decoded (imputed)
  - Raw nuclei
  - Correlation between cross-decoded and real cells: 0.989]
- *Talking point: this shows the model captures gene-gene relationships, not just individual gene means*

### Slide 22: Ablation summary
**Key message:** Each design choice matters — show the evidence.
- [FIGURE: table with columns: Model variant | Classification | DE overlap | Gene corr | Shift score]
- Rows: shared decoder / conditioned / cond+xr / MSE vs NB / scalar vs learned embed
- Highlight the two key findings:
  1. **Cross-reconstruction** is the single largest improvement
  2. **Conditioned decoder** prevents degradation (shared decoder is worse than no decoder conditioning)
- MSE dominates NB on every metric (NB's softmax bottleneck suppresses low-expression genes)
- *Talking point: save MSE vs NB detail for backup if they ask — just state the conclusion here*

### Slide 23: Known limitation — variance collapse
**Key message:** Be upfront about the main weakness.
- Cell-type-level metrics are strong, but within-cell-type variance collapses:
  - Cond+xr retains only **3%** of within-CT variance
  - Cond (no xr): **11%**
- [FIGURE: Left — violin plots per cell type showing variance in real cells vs imputed. Right — scatter plot of per-cell z vs output showing near-flat relationship]
- Hallucination test: replace each cell's z with its cell-type mean z → output changes by <0.1%
- Interpretation: the decoder ignores per-cell latent information and outputs the cell-type average
- *Talking point: this is the central tension — cross-recons drives the best cell-type metrics AND the worst variance collapse*

### Slide 24: Ongoing work on variance
**Key message:** We have multiple strategies and early progress.
- Normalized loss: within-CT variance ratio improved from 11% → **45%**
- CPM normalization: removes library-size as a shortcut the model can exploit
- Full conditional VAE: condition the encoder too (not just decoder)
- Single-stage training from scratch: avoid inheriting pretrained biases
- [FIGURE: table showing ongoing experiment configs and early results if available]
- *Talking point: variance is the main open problem — I have a clear plan with multiple approaches*

### Slide 25: Aim 1 next steps
**Key message:** Concrete plan the committee can hold you to.
- **Baselines**: scVI and Harmony treating modality as batch — test whether batch correction under-performs on this structured transformation
- **Held-out cell type**: train on N-1 cell types, test on held-out → does the model generalize beyond training cell types?
- **Gene-stratified eval**: stratify genes by nuc-cell gap magnitude → model should help most on genes with the largest differences
- **Variance**: complete ongoing experiments, identify best strategy

---

## Aim 2: Cross-Species Generalization (8 slides, ~10 min)

### Slide 26: Section divider — "Aim 2: Cross-Species Generalization & Biological Validation"

### Slide 27: The question
**Key message:** This aim asks a biological question, not just an engineering one.
- The nucleus-cell gap reflects RNA localization biology
- If RNA localization programs are conserved across species → model transfers
- If they're not → that's a finding about the evolution of RNA compartmentalization
- Either outcome is scientifically informative
- [FIGURE: simple diagram — mouse brain (model trained) → arrow with "?" → primate brain → arrow with "?" → human brain]

### Slide 28: Expression conservation across species
**Key message:** There's reason to think partial transfer is plausible.
- Orthologous genes preserve cell-type-specific expression patterns across mammals
- Core marker genes and pathway membership are conserved
- But expression magnitudes can differ 2–5×
- Implication: *which* genes are affected by the nuc-cell gap may be conserved, but *how much* may differ
- Cite: Hodge et al. 2019, Bakken et al. 2021
- *Talking point: this is why we test three strategies — direct transfer tests "which", fine-tuning adjusts "how much"*

### Slide 29: Available paired datasets
**Key message:** What data exists for validation.
- [FIGURE: table or visual showing data landscape:
  | Dataset | Species | Region | Protocol | Type |
  |---------|---------|--------|----------|------|
  | Allen Brain Cell Atlas | Mouse | Cortex | 10X v3 | **Training data** |
  | Ding et al. 2020 | Mouse | Cortex | 10X v2, Drop-seq, Smart-seq2 | **Cross-dataset test** |
  | Bakken et al. 2021 | Macaque | dLGN | SMART-Seq v4 | **Cross-species test** |
  | PsychENCODE, HCA, etc. | Human | Cortex+ | Various | **Application (no paired)** |]
- Key constraint: Bakken et al. is the **only** paired primate brain dataset
- Human brain: nuclei only (postmortem) — no paired data exists → this is why we need ImputeVAE

### Slide 30: Experiment 2a — Cross-dataset generalization (mouse cortex)
**Key message:** First test robustness within the same species and tissue.
- Apply Allen Brain-trained model to Ding et al. 2020 mouse cortex data
- Same tissue, **different lab**, **multiple protocols** (10X v2, Drop-seq, Smart-seq2)
- Tests robustness to:
  - Laboratory effects
  - Sample preparation differences
  - Sequencing protocol variation
- This establishes a baseline: how well does the model handle technical variation before we add species as a variable?
- Cite: Ding et al. 2020
- *Talking point: if it fails on same-tissue different-protocol, cross-species is premature*

### Slide 31: Experiment 2b — Cross-species transfer (mouse → macaque)
**Key message:** Three strategies that test a biological hypothesis.
- Using macaque dLGN paired data from Bakken et al. 2021
- Three strategies:
  1. **Direct transfer (zero-shot)**: apply mouse model to macaque nuclei via orthologous genes, no retraining
  2. **Fine-tuning**: initialize from mouse weights, run Stage 2 on macaque paired data
  3. **Retrain from scratch**: train entirely on macaque data (upper bound, but small dataset → may overfit)
- [FIGURE: flowchart showing the three paths from mouse model → macaque evaluation]
- Note: macaque data is from dLGN (thalamus), not cortex — confounds species with region
  - *If committee asks: "We can't fully separate species from region with available data. But the three-strategy comparison tells us whether transfer learning helps, regardless of the confound."*

### Slide 32: What each outcome tells us
**Key message:** Every result has a biological interpretation.
- [FIGURE: table — make this visually clean:
  | Outcome | Biological interpretation |
  |---------|--------------------------|
  | Direct transfer works | RNA localization is universal across mammals — same genes depleted in cytoplasm |
  | Direct fails, fine-tuning works | Shared structure exists but magnitude differs — species-specific tuning needed |
  | Only retraining works | Transformation is species-specific — different RNA export/localization programs |]
- All three outcomes are publishable findings
- *Talking point: this is what makes this aim more than engineering — it tests a biological hypothesis about conservation*

### Slide 33: Experiment 2c — Application to human brain nuclei
**Key message:** The real use case, evaluated through biological plausibility.
- No paired human brain data → can't compute accuracy directly
- Three indirect evaluation metrics:
  1. **Cytoplasmic RNA recovery**: Fos, Arc, ribosomal genes should increase; lncRNAs should not change
     - These are verifiable predictions that don't require paired ground truth
  2. **Cell-type marker improvement**: known markers should become more distinct in imputed profiles
  3. **Downstream DE on PsychENCODE**: do imputed profiles change disease-associated DE gene detection?
     - Use PsychENCODE reference DE lists (ASD, Bipolar, Schizophrenia, Aging) as ground truth
- Will also characterize systematic failures → genes/cell types that impute poorly point to species- or tissue-specific RNA localization
- *Talking point: "the absence of paired human data is both our limitation and our motivation — if it existed, imputation would be unnecessary"*

### Slide 34: Aim 2 next steps
**Key message:** Concrete and achievable.
- Obtain and preprocess macaque dLGN paired data (Bakken et al., GEO: GSE182211)
- Implement ortholog gene mapping (mouse ↔ macaque ↔ human)
  - Characterize gene loss at each mapping step (how many of 16K survive? are key markers preserved?)
- Run three-strategy transfer experiment on macaque
- Design biological plausibility evaluation pipeline for human nuclei

---

## Closing (4 slides, ~5 min)

### Slide 35: Summary
- **Aim 1**: Learn the structured nucleus → cell transformation
  - Cross-reconstruction + conditioned decoder = key ingredients
  - Strong cell-type-level results; per-cell variance is the main open problem
- **Aim 2**: Test whether the transformation is conserved across species
  - Cross-dataset → cross-species → human application
  - All outcomes (transfer works / partially / fails) are scientifically informative
- One-liner: **Not batch correction — structured biological transformation**

### Slide 36: Proposed Timeline
- [FIGURE: Gantt chart or timeline with:
  - Summer 2026: Aim 1 baselines (scVI, Harmony), variance experiments
  - Fall 2026: Aim 2a cross-dataset eval, begin macaque data preprocessing
  - Spring 2027: Aim 2b cross-species transfer, qualifying exam
  - Summer–Fall 2027: Aim 2c human application (PsychENCODE)
  - 2028: Manuscript preparation and submission]

### Slide 37: Reading list
- Key papers for committee (if required by Yale CBB):
  - Bakken et al. 2018 (PLoS ONE) — nucleus vs cell comparison
  - Ding et al. 2020 (Nat Biotech) — systematic method comparison
  - Lopez et al. 2018 (Nat Methods) — scVI
  - Hodge et al. 2019 (Nature) — human vs mouse cortex cell types
  - Bakken et al. 2021 (eLife) — cross-species dLGN

### Slide 38: Thank you / Questions
- Acknowledgements: Mark Gerstein, Gerstein Lab members, committee members
- Funding if applicable

---

## Backup slides (5 slides)

### Backup 1: Loss function math
- MSE reconstruction: L_recons = ||x - x̂||²
- JSD alignment: JSD(q_cell(z|ct) || q_nuc(z|ct)) estimated via Monte Carlo sampling
- Cross-reconstruction: L_xr = ||x_cell - Decoder(Encoder(x_nuc), cond=cell)||²
- Total loss = Σ λ_i(epoch) · L_i where λ_i ramps 0→1 per the cascading schedule

### Backup 2: Full training hyperparameters
- Architecture: input → 1024 → 512 → 32-dim latent → 512 → 1024 → output
- Modality embedding: dim=8 (learned) or scalar (1=cell, 0=nuclei)
- Optimizer: Adam, LR=1e-4
- Batch size: 128
- Stage 1: 300 epochs, Stage 2: 600 epochs
- Activation: LeakyReLU
- Training time: ~8-12 hours on single GPU

### Backup 3: MSE vs Negative Binomial deep dive
- NB: models counts with learned dispersion, output through softmax → proportions × library size
- Softmax creates zero-sum: if one gene's proportion goes up, others must go down
- This suppresses low-expression genes — the model "spends" its probability budget on dominant genes
- MSE on log1p: each gene is an independent regression target, no competition
- [FIGURE: per-gene reconstruction error comparison, MSE vs NB — show NB failing on low-expression genes]

### Backup 4: Variance collapse deep dive
- Hallucination test details: for each cell, replace z with cell-type centroid z → decode → compare to original decode
  - Pearson correlation between original and hallucinated output: 0.998–0.999
  - This means <0.1% of the output depends on per-cell latent information
- Within-CT variance ratio by model:
  - Shared decoder: 0.03
  - Conditioned (no xr): 0.115
  - Conditioned + xr: 0.035
  - Normalized loss: 0.454
- Within-CT Spearman correlation (per-cell gene ranking): 0.035–0.112
- Key tension: cross-recons drives the best cell-type metrics AND the worst variance collapse
- Ongoing experiments: normloss, CPM, full cVAE, single-stage — table of latest results

### Backup 5: ISMB review responses
- Map each major reviewer concern to how it's addressed:
  - "Weak baselines" → Aim 1 next steps: scVI, Harmony baselines
  - "Single dataset" → Aim 2: cross-dataset (Ding), cross-species (macaque)
  - "No ablation" → Slide 22: systematic ablation table
  - "Circular evaluation" → Aim 1: held-out cell type evaluation
  - "Variance collapse" → Slides 23-24: acknowledged + active work
  - "Figure quality" → Will redo all figures for qual/paper
