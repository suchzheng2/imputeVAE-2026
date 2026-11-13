# Validation Datasets — Literature Search

For ISMB 2026 poster. Goal: identify datasets to validate VAE imputation model.

---

## Category A: Matched Single-Cell vs Single-Nucleus Datasets

These are the most direct validation — same tissue profiled with both scRNA-seq and snRNA-seq.

### A1. Bakken et al. 2018 — Mouse Visual Cortex (BEST MATCH)
- **Paper**: "Single-nucleus and single-cell transcriptomes compared in matched cortical cell types" (PLOS ONE)
- **Species**: Mouse
- **Region**: Primary visual cortex (VISp), layer 5
- **Platform**: Smart-Seq v4 (NOT 10X — full-length, deeper per cell)
- **Samples**: 463 matched nuclei + 463 whole cells
- **Cell types**: 11 neuronal types (9 excitatory, 2 inhibitory)
- **Data**: GEO GSE123454; GitHub: https://github.com/AllenInstitute/NucCellTypes
- **Key findings**:
  - Cells detect ~11,000 genes vs nuclei ~7,000 genes
  - 8,614 genes significantly higher in cells (>1.5-fold, FDR<0.05)
  - Immediate early genes (Fos, Egr1, Arc) up to 10x higher in cells (dissociation artifact)
  - Nuclear mRNA proportion varies by cell type (20-50%)
- **Relevance**: Matched cell/nuclei from mouse cortex = direct ground truth for our model. Smart-Seq gives deeper coverage per cell. Different platform from our training data (10X) = true external validation.
- **Limitation**: Small sample (463 cells), Smart-Seq not 10X, only layer 5 VISp

### A2. Ding et al. 2020 — Systematic Benchmark (Mouse + Human Brain)
- **Paper**: "Systematic comparison of single-cell and single-nucleus RNA-sequencing methods" (Nature Biotechnology)
- **Species**: Mouse (+ human cell lines, PBMCs)
- **Region**: Brain cortex
- **Platform**: 7 methods compared including 10x Chromium, Smart-seq2, DroNc-seq
- **Samples**: 36 libraries across 6 experiments
- **Data**: Single Cell Portal SCP424, SCP425, SCP426
- **Key findings**: First systematic comparison of sc vs sn methods on same tissue
- **Relevance**: Multi-method comparison on brain tissue. 10X data available. Can test if our model generalizes across platforms.

### A3. Allen Brain Cell Types Database
- **Overview**: Comprehensive single-cell and single-nucleus RNA-seq from mouse brain
- **Datasets**:
  - Mouse Whole Cortex and Hippocampus 10x: 1.1M cells, multiple cortical areas + hippocampus (GEO GSE185862)
  - Whole Mouse Brain 10Xv2: 1.7M cells
  - Whole Mouse Brain 10Xv3: 2.3M cells
- **Platform**: 10X Genomics (v2 and v3) + Smart-Seq
- **Data access**: https://alleninstitute.github.io/abc_atlas_access/intro.html (AWS S3, CC BY NC 4.0)
- **Relevance**: Our training data already comes from AIBS. Other brain regions (hippocampus, visual cortex) from the same source = ideal for testing cross-region generalization.
- **Key question**: Does the atlas include BOTH cell and nuclei from the same regions? Need to check which regions have matched modalities.

---

## Category B: Nuclear vs Cytoplasmic RNA Fractionation

These provide ground truth for which genes are nuclear-enriched vs cytoplasmic-enriched.

### B1. Buckley et al. 2016 — Mouse Brain Nuclear/Cytoplasmic RNA
- **Paper**: "A comprehensive analysis of cell-type specific nuclear RNA from neurons and glia of the brain" (PMC4996761)
- **Species**: Mouse
- **Region**: Brain (neurons and glia)
- **Method**: Nuclear/cytoplasmic fractionation + RNA-seq
- **Relevance**: Direct ground truth for validating our delta analysis — genes our model increases should be cytoplasm-enriched, genes it decreases should be nucleus-enriched.
- **Data**: Need to check GEO for accession

### B2. Price et al. 2020 — Human Cortex Nuclear/Cytoplasmic
- **Paper**: "Characterizing the nuclear and cytoplasmic transcriptomes in developing and mature human cortex" (Genome Research)
- **Species**: Human
- **Region**: Frontal cortex (fetal + adult)
- **Method**: Bulk RNA-seq of nuclear and cytoplasmic fractions
- **Key findings**: Differential expression differs between fractions; transcript types and DE genes change based on fraction
- **Relevance**: Cross-species validation — human cortex nuclear/cytoplasmic differences should parallel mouse

### B3. Moutinho et al. 2021 — Human Brain Nuclear/Cytosolic
- **Paper**: "Characterization of the nuclear and cytosolic transcriptomes in human brain tissue" (Scientific Reports)
- **Species**: Human
- **Region**: Brain tissue
- **Method**: Bulk RNA-seq of nuclear and cytosolic fractions
- **Data**: Nature Scientific Reports (open access)
- **Relevance**: Provides gene-level nuclear/cytosolic enrichment scores to validate against our delta analysis

### B4. ENCODE Nuclear/Cytoplasmic Fractionation
- **Data**: GEO GSE30567 — polyA+ and polyA- RNA-seq from nuclear and cytoplasmic fractions
- **Cell lines**: GM12878, HepG2, K562, etc. (NOT brain tissue)
- **Relevance**: Large-scale reference for nuclear vs cytoplasmic RNA, though cell lines not brain-specific

---

## Category C: Other Tissue / Cross-Species

### C1. Thrupp et al. 2020 — Human Microglia sc vs sn
- **Paper**: "Single-Nucleus RNA-Seq Is Not Suitable for Detection of Microglial Activation Genes in Humans" (Cell Reports)
- **Species**: Human
- **Data**: GEO GSE137444 (single cells), GSE153807 (single nuclei)
- **Key finding**: ~1% of genes depleted in nuclei vs whole cells; microglial activation genes specifically lost
- **Relevance**: Shows cell-type-specific gene loss in nuclei — our model should capture this

### C2. Whole Mouse Brain Atlas (Yao et al. 2023)
- **Paper**: "A high-resolution transcriptomic and spatial atlas of cell types in the whole mouse brain" (Nature)
- **Data**: ~4M cells, 5,322 clusters, 34 classes
- **Platform**: 10X v2 + v3
- **Relevance**: If we can access data from regions not in our training set, can test cross-region generalization

---

## Recommended Validation Strategy

### Priority 1: Direct validation (for ISMB poster)
1. **Bakken 2018 (GSE123454)**: Download matched cell/nuclei from mouse VISp. Run nuclei through our model, compare imputed values to actual cell expression. Compute gene-level correlation, top-gene overlap with real cell-nucleus differences.
2. **Nuclear/cytoplasmic gene lists (Buckley, Moutinho)**: Get published nuclear-enriched and cytoplasmic-enriched gene lists. Check overlap with our delta analysis increased/decreased genes. Report enrichment statistics.

### Priority 2: Cross-region generalization
3. **Allen Brain Atlas other regions**: Check if hippocampus or visual cortex data has both cell and nuclei. If so, apply model trained on motor cortex to new regions.

### Priority 3: Cross-species
4. **Ding 2020 (SCP424-426)**: If human cortex data has matched cell/nuclei, test cross-species transfer.

---

## Data Download TODO
- [ ] GSE123454 — Bakken 2018 mouse VISp matched cells/nuclei
- [ ] SCP424/425/426 — Ding 2020 benchmark (check which has brain 10X data)
- [ ] Nuclear/cytoplasmic gene lists from Buckley 2016 and Moutinho 2021
- [ ] Check Allen Brain Atlas for other regions with both modalities
- [ ] GSE185862 — Allen Whole Cortex + Hippocampus 10x

## References
- Bakken et al. (2018) PLOS ONE. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0209648
- Ding et al. (2020) Nature Biotechnology. https://www.nature.com/articles/s41587-020-0465-8
- Buckley et al. (2016) PMC4996761. https://pmc.ncbi.nlm.nih.gov/articles/PMC4996761/
- Moutinho et al. (2021) Scientific Reports. https://www.nature.com/articles/s41598-021-83541-1
- Price et al. (2020) Genome Research. https://genome.cshlp.org/content/30/1/1.full
- Thrupp et al. (2020) Cell Reports. https://www.sciencedirect.com/science/article/pii/S2211124720311785
- Yao et al. (2023) Nature. https://www.nature.com/articles/s41586-023-06812-z
- Allen Brain Cell Atlas. https://alleninstitute.github.io/abc_atlas_access/intro.html
