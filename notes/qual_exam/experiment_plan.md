# Qual Exam Experiment Plan

Experiments needed to close the gap between "model reproduces held-out cells" and
"the delta is real cytoplasmic biology with a downstream consequence."

Model under test: full-data cVAE (`run/run_2026-07-18-19-40-51_570984`)
Existing delta: `notes/experiments/delta_analysis/`

---

## Tier 1 — Validation & confound control (reanalysis, no retraining)

### 1. Localization concordance (lncATLAS + RNALocate + APEX-seq)
- **What**: Correlate per-gene delta with independently measured cytoplasmic:nuclear ratios
- **Data needed**: lncATLAS CN-RCI scores (https://lncatlas.crg.eu/), RNALocate v2 labels (http://www.rnalocate.org/), APEX-seq compartment gene lists (Fazal 2019 supplementary)
- **Analysis**:
  - Spearman correlation: delta vs CN-RCI across all mappable genes
  - ROC/AUC: delta sign predicts "cytoplasmic" vs "nuclear" label
  - Fisher's exact: added genes enriched in APEX cytosol/ER? Removed genes in nuclear/nucleolar?
- **Expected result**: Positive correlation; Malat1/Neat1 at nuclear extreme; AUC >> 0.5
- **Critique it closes**: "Global ribosomal vector" + "circular validation" — none of these refs derive from Allen training data
- **Effort**: Medium (download DBs, mouse-human ortholog mapping)
- **Status**: NOT STARTED

### 2. Dissociation / ambient artifact audit
- **What**: Show added-gene set is NOT enriched for dissociation stress modules
- **Data needed**:
  - van den Brink 2017 IEG/stress gene module (~100 genes: Fos, Jun, Hspa1a/b, etc.)
  - Marsh 2022 glial activation artifact gene list
- **Analysis**:
  - Overlap: are IEG/stress genes in our top positive-delta set?
  - Enrichment test (hypergeometric) of stress module in added genes
  - Check Fos, Jun, Arc, Hspa1a specifically — what are their deltas?
- **Expected result**: Stress genes NOT enriched in top added set (ribosomal/synaptic dominate)
- **Extended version**: Re-run on CellBender/SoupX-corrected data — Malat1/Neat1 depletion survives ambient correction
- **Critique it closes**: "Is your delta biology or dissociation+soup?" — the #1 attack
- **Effort**: Low (gene list overlaps on existing delta) / High (CellBender reprocessing)
- **Status**: NOT STARTED

### 3. Technical covariate regression
- **What**: Show delta is not explained by gene length, GC content, or mean expression
- **Data needed**: Gene annotations (length, GC%) from Ensembl/biomaRt
- **Analysis**:
  - Regress delta ~ gene_length + GC_content + mean_expression
  - Report residual variance — does delta carry signal beyond technical covariates?
- **Expected result**: Low R² for technical model; residual delta still correlates with localization
- **Critique it closes**: Genome Research 2024 "sampling not biology"
- **Effort**: Low
- **Status**: NOT STARTED

### 4. Intron-retention control
- **What**: Test whether removed set is just unspliced/intron-retained transcripts
- **Data needed**: Intron retention rates from Xia 2019 MERFISH (1,488 nuclear-enriched genes, many intron-retained)
- **Analysis**:
  - Separate delta into spliced vs unspliced components (if velocyto/scVelo annotations available)
  - Or: check overlap of negative-delta genes with Xia 2019 intron-retained gene list
  - Show delta carries signal beyond intron content
- **Critique it closes**: "You just learned to strip introns"
- **Effort**: Medium
- **Status**: NOT STARTED

### 5. Scrambled-delta negative control
- **What**: Show metrics have power and don't pass on noise
- **Analysis**:
  - Permute gene labels in imputed output → recompute all metrics
  - Shuffle cell-type assignments → recompute
  - Identity baseline: feed nuclei through without transformation
- **Expected result**: All metrics crash on permuted data
- **Effort**: Low
- **Status**: NOT STARTED

### 6. TRAP gene set overlap
- **What**: Test that added genes are enriched in ribosome-bound (translated, cytoplasmic) sets
- **Data needed**: TRAP/RiboTag gene lists from Heiman 2008 / Doyle 2008 (24 CNS types), Sheehan 2025 (glial cortex)
- **Analysis**: GSEA or hypergeometric overlap per cell type
- **Critique it closes**: "Is the delta actually cytoplasmic RNA, or just high-expression genes?"
- **Effort**: Medium (find processed gene lists from publications)
- **Status**: NOT STARTED

---

## Tier 2 — Orthogonal data validation + baselines

### 7. Patch-seq "closes the gap" test (Scala 2021)
- **What**: Show imputation moves nucleus profiles toward whole-cell Patch-seq profiles
- **Data needed**: Scala 2021 Patch-seq data (BICCN/NeMO, mouse MOp, same taxonomy)
- **Analysis**:
  - For each matched cell type: distance(raw_nucleus, patch_seq_mean) vs distance(imputed, patch_seq_mean)
  - Expected: imputation reduces distance
  - Per-gene: genes Patch-seq detects but snRNA-seq depletes should have positive delta
- **Critique it closes**: "Validate against an orthogonal assay in the same tissue"
- **Effort**: Medium-high (download, preprocess, map cell types)
- **Status**: NOT STARTED

### 8. Expanded baseline panel
- **What**: Beat interpretable baselines, not just scVI and CT-mean
- **Baselines to add**:
  - Global mean-shift (add fixed delta vector to all nuclei)
  - Per-gene linear regression (cell = a*nuc + b, per CT or global)
  - scGen-style latent arithmetic (mean_cell_latent - mean_nuc_latent, add to each nucleus)
  - scANVI with modality label
  - scButterfly (current SOTA cross-modal translation)
- **Existing**: linear reg already in `run/linreg_baseline/` — check/add to comparison
- **Critique it closes**: "Isn't this just scGen?" + "Did you beat a linear baseline?"
- **Effort**: Medium (scGen/scANVI straightforward; scButterfly needs setup)
- **Status**: PARTIAL (linear reg exists)

### 9. Ablation study
- **What**: Show modality condition and alignment loss do distinct jobs
- **Ablations**:
  - Remove modality condition → outputs collapse (two modalities indistinguishable)
  - Remove alignment term → cell-type structure degrades
  - Remove normloss → variance collapses (already shown: 0.158 → ~0.04)
- **Critique it closes**: "Why doesn't alignment just batch-correct?" + "Is the architecture necessary?"
- **Effort**: Medium (retrain 2-3 ablation models)
- **Status**: PARTIAL (normloss ablation exists from factorial experiment)

---

## Tier 3 — Simulation & per-cell ground truth

### 10. scDesign3 simulation sandbox
- **What**: Plant a known nucleus→cytoplasm transformation, confirm model recovers it
- **Analysis**:
  - Simulate paired nuc/cyto with scDesign3 (planted gene-specific deltas)
  - Train cVAE, measure recovery of planted delta
  - **Negative control**: identity transformation → model reports ~no change
- **Critique it closes**: "Do your metrics have statistical power?"
- **Effort**: Medium-high
- **Status**: NOT STARTED

### 11. SINC-seq per-cell test
- **What**: Test per-cell prediction against paired nuclear+cytoplasmic RNA (K562)
- **Data**: Abdelmoez 2018 SINC-seq (GEO)
- **Caveat**: K562 not neurons — validates principle, not neuron-specific
- **Critique it closes**: "Is there any per-cell ground truth?"
- **Effort**: Medium
- **Status**: NOT STARTED

---

## Tier 4 — Cell-type biology & generalization

### 12. Cell-type-resolved delta biology
- **What**: Show delta differs by cell type in biologically sensible ways
- **Analysis**:
  - Already have per-CT delta correlations (0.989 median)
  - Add: neurons gain more synaptic/cytoplasmic content than glia
  - Subtype comparison: L5 ET vs L2/3 IT delta profiles
  - Meg3 case study: nuclear in glia, cytoplasmic in neurons (already found)
- **Critique it closes**: "Is this a class-level average or cell-type-specific?"
- **Effort**: Low (mostly reanalysis of existing results)
- **Status**: PARTIAL (per-CT delta exists, Meg3 analysis done)

### 13. Cross-species / cross-tissue transfer
- **What**: Test if nucleus→cell shift is conserved across species
- **Data**: Bakken cross-species matched data, or human kidney (Lake 2023)
- **Prerequisite for**: Disease reanalysis (#14)
- **Effort**: High
- **Status**: NOT STARTED (v2 protocol transfer done as first step)

---

## Tier 5 — The "so what" experiment

### 14. PsychENCODE disease DE reanalysis
- **What**: Does imputing cytoplasm change a disease conclusion?
- **Data**: PsychENCODE snRNA-seq (SCZ, BD, AD cohorts)
- **Analysis**:
  - Impute cytoplasmic layer for all nuclei
  - Redo cell-type pseudobulk DE (disease vs control)
  - Do cytoplasmic/synaptic/ribosomal genes that nuclear-only DE misses now surface?
- **Impact**: Difference between Nat Methods (clean method) and Nat Neuroscience (mechanism)
- **Requires**: Cross-species model (#13) or human training data
- **Effort**: Very high
- **Status**: NOT STARTED

---

## Execution order

### Phase 1: Exam defense (1-2 weeks, reanalysis only)
1. Dissociation artifact audit (#2) — LOW effort, high return
2. Scrambled-delta negative control (#5) — LOW effort
3. Technical covariate regression (#3) — LOW effort
4. Cell-type-resolved delta (#12) — LOW effort, partially done
5. Localization concordance (#1) — MEDIUM effort, highest-impact single figure

### Phase 2: Orthogonal validation (2-4 weeks)
6. TRAP gene set overlap (#6)
7. APEX-seq compartment check (part of #1)
8. Intron-retention control (#4)
9. Expanded baselines (#8)

### Phase 3: Strongest validation (1-2 months)
10. Patch-seq validation (#7)
11. Ablation study (#9)
12. scDesign3 simulation (#10)

### Phase 4: Paper-level
13. Cross-species transfer (#13)
14. Disease reanalysis (#14)

---

## Scripts to create

| Experiment | Script | Input | Output |
|-----------|--------|-------|--------|
| #1 | `analysis/localization_concordance.py` | delta CSV + lncATLAS + RNALocate | Spearman r, AUC, figure |
| #2 | `analysis/dissociation_artifact_audit.py` | delta CSV + IEG gene list | enrichment stats, gene-level table |
| #3 | `analysis/technical_covariate_check.py` | delta CSV + gene annotations | regression stats |
| #5 | `analysis/scrambled_delta_control.py` | imputed .npy + obs | permuted metrics |
| #6 | `analysis/trap_overlap.py` | delta CSV + TRAP gene lists | enrichment per CT |
| #7 | `analysis/patchseq_validation.py` | imputed + Scala 2021 data | distance reduction per CT |
