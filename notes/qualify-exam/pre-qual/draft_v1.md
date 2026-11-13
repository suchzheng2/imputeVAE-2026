# Preliminary Qualifying Proposal

Suchen Zheng

[Date TBD]

## Aim 1: A Cross-Reconstructive VAE Framework for Translating Single-Nucleus to Single-Cell Expression Profiles

### Introduction

Single-nucleus RNA sequencing (snRNA-seq) has become the dominant method for profiling gene expression in tissues that are difficult to dissociate into intact cells, including archived, frozen, and postmortem samples. Unlike single-cell RNA-seq (scRNA-seq), which captures the full cellular transcriptome, snRNA-seq measures only nuclear RNA, systematically missing cytoplasmic transcripts such as immediate early genes and other localized mRNAs. These differences are not random noise — they are structured, gene-specific, and cell-type-dependent, reflecting the biology of RNA localization and nuclear export (Bakken et al., 2018; Ding et al., 2020). As a result, expression profiles from nuclei and whole cells can diverge substantially, complicating cross-study comparisons and limiting the reuse of large nuclei-only datasets for analyses originally designed around whole-cell data.

Current computational approaches treat the nucleus-cell difference as a batch effect, applying integration methods like Harmony (Korsunsky et al., 2019) and scVI (Lopez et al., 2018) to align the two modalities. But because the differences are systematic and gene-specific rather than stochastic, batch correction can either under-correct (missing structured patterns) or over-correct (removing real biology). Instead of removing the difference, we propose to learn the transformation from nucleus to cell expression and use it to generate cell-like profiles from nuclei-only data.

### Background

**The nucleus-cell expression gap.** Multiple studies have documented systematic differences between snRNA-seq and scRNA-seq profiles from the same tissue (Bakken et al., 2018; Ding et al., 2020). Nuclei yield fewer detected transcripts, show depletion of cytoplasmic mRNAs (e.g., immediate early genes such as *Fos* and *Arc*), and enrichment of nuclear-retained long non-coding RNAs. Importantly, these effects vary by cell type — neurons, with extensive cytoplasmic processes, show larger nucleus-cell gaps than compact glial cells. This cell-type-dependence makes simple global correction inadequate.

**Existing approaches.** Current methods fall into two categories. Batch-correction tools (Harmony, scVI, Scanorama) assume the variation between modalities is stochastic and globally structured, which does not hold for the gene-specific, cell-type-dependent nucleus-cell gap. Pseudobulk approaches (Crowell et al., 2020) compute per-gene correction factors averaged within cell types, capturing mean shifts but discarding all within-cell-type variation.

**Variational autoencoders for single-cell data.** VAEs have been widely adopted for single-cell analysis (Lopez et al., 2018; Gayoso et al., 2021; Xu et al., 2021), using probabilistic encoder-decoder architectures to learn low-dimensional latent representations. However, existing VAE frameworks are designed to remove variation between conditions, not to learn a directional transformation from one modality to another.

### Proposed Framework

We propose ImputeVAE, a VAE-based framework that learns to translate single-nucleus expression profiles into cell-like profiles through three key components: (1) a shared encoder that maps both modalities into a common latent space, with alignment losses that encourage same-cell-type distributions to overlap, (2) a conditioned decoder that enables modality-specific reconstruction from the shared latent, and (3) a cross-reconstruction objective that directly supervises the nucleus-to-cell translation.

[Figure 1: Architecture diagram — shared encoder, latent space with alignment, conditioned decoder with modality embedding, cross-reconstruction path]

**Shared encoder with aligned latent space.** Both cell and nucleus data pass through the same encoder into a common latent space. An alignment loss pushes same-cell-type distributions from both modalities together so that cell-type identity is maintained across modalities.

**Conditioned decoder.** The decoder takes the latent vector along with a modality indicator, letting it produce modality-specific outputs while sharing most parameters.

**Cross-reconstruction.** During training, we encode nucleus data and decode it with the cell modality condition, then compare the output to real cell profiles from the same cell type. This directly supervises the nucleus-to-cell translation rather than hoping that latent alignment alone will produce the right mapping.

**Two-stage training.** We first pretrain on cell data only to get a well-structured latent space, then do joint training on both modalities with loss terms introduced gradually to avoid competing gradients. At inference, we encode new nucleus data and decode with the cell condition.

### Dataset

We use paired cell and nucleus data from the Allen Brain Cell Atlas (mouse cortical tissues, 10X Chromium v3 chemistry; 71K cells, 40K nuclei, 17 cell types, 16,439 genes). Preliminary results on this dataset demonstrate improved cell-type classification and differentially expressed gene recovery over raw nuclei.

### Next Steps

- Compare against baseline methods (scVI, Harmony) that treat the modality difference as a batch effect.
- Evaluate generalization via held-out cell types and gene-stratified analysis.
- Assess preservation of within-cell-type variation at the per-cell level.

---

## Aim 2: Cross-Species Generalization and Biological Validation

### Introduction

Does the learned transformation generalize beyond the training dataset? The nucleus-cell gap reflects RNA localization biology, which varies by gene and cell type. If these localization programs are conserved across species, a model trained on mouse data should at least partially transfer to primate brain. If not, that itself tells us something about how RNA compartmentalization has diverged across species. Aim 2 tests this through cross-species transfer experiments and applies the model to large-scale human nuclei datasets.

### Background

**Available paired datasets.** Paired single-cell and single-nucleus data from the same brain tissue is rare, particularly for primates. For mouse, the Allen Brain Cell Atlas provides extensive paired cortical data (our Aim 1 training set). For primates, Bakken et al. (2021) profiled both single-cell and single-nucleus transcriptomes from the dorsal lateral geniculate nucleus (dLGN) of macaque brain (3 donors, SMART-Seq v4), alongside mouse dLGN data using the same protocol. This dataset provides the only available paired primate brain data suitable for validating cross-species transfer. For human brain, paired data is effectively unavailable — postmortem tissue yields only nuclei, and fresh dissociation requires surgical samples that are extremely rare. This absence of paired human brain data is precisely the motivation for ImputeVAE.

**Cross-tissue and cross-species variation.** The nucleus-cell expression gap varies across tissues (Ding et al., 2020) and likely across species, reflecting differences in cell morphology, RNA metabolism, and nuclear export machinery. Neurons, with their long axons and dendrites, harbor substantially more cytoplasmic RNA than compact glial cells. Whether these cell-type-dependent differences are conserved across species is unknown. Comparative transcriptomic studies have shown that orthologous genes largely preserve cell-type-specific expression patterns across mammals, with core marker genes and pathway membership conserved even when expression magnitudes differ (Hodge et al., 2019; Bakken et al., 2021). This suggests that which genes are affected by the nucleus-cell gap may also be conserved, even if the magnitude of the effect differs across species.

**Large-scale nuclei-only resources.** Datasets such as PsychENCODE (2.8M human brain nuclei), the Human Cell Atlas, and many disease cohorts are nuclei-only due to practical constraints. If ImputeVAE can produce reliable cell-like profiles from these datasets, they could be reanalyzed with methods that were originally designed for whole-cell data.

### Proposed Experiments

#### 2a. Cross-dataset generalization within mouse cortex

Before testing cross-species transfer, we first ask whether the transformation generalizes to independent datasets from the same tissue. Ding et al. (2020) profiled mouse cortex with multiple protocols (10X v2, Drop-seq, Smart-seq2, and others) alongside both single-cell and single-nucleus preparations, providing paired ground truth from a different laboratory than our training data. We evaluate the Allen Brain-trained model on these held-out datasets to test whether the learned transformation is robust to differences in laboratory, sample preparation, and sequencing protocol.

#### 2b. Cross-species transfer to macaque

Using the macaque dLGN paired data from Bakken et al. (2021), we test three transfer strategies:

1. **Direct transfer (zero-shot):** Apply the mouse-trained model to macaque nuclei using orthologous genes, without any retraining. This tests whether the nucleus-to-cell transformation is conserved across species.

2. **Fine-tuning:** Initialize from mouse-trained weights and run Stage 2 joint training on macaque paired data (both macaque cells and nuclei). The hypothesis is that the mouse model already captures most of the transformation structure, and fine-tuning adapts to species-specific differences in gene expression scale and cell type composition.

3. **Retrain from scratch:** Train entirely on macaque paired data as an upper bound. The macaque dLGN dataset is small (SMART-Seq, plate-based), so this may overfit, making the comparison with fine-tuning informative about the value of mouse pretraining.

The comparison between these three strategies directly tests a biological hypothesis:

| Outcome | Biological interpretation |
|---------|--------------------------|
| Direct transfer works | Nucleus-to-cell transformation is conserved — RNA localization biology is universal across mammals |
| Direct transfer fails, fine-tuning works | Transformation is partially conserved — shared structure exists but requires species-specific tuning |
| Only retraining works | Transformation is species-specific — different RNA export/localization programs per species |

#### 2c. Application to human brain nuclei

Paired human brain cell and nucleus data does not exist at scale — postmortem tissue yields only nuclei. We therefore evaluate human application through indirect biological plausibility metrics rather than direct accuracy:

1. **Cytoplasmic RNA recovery:** Known nuclear-depleted transcripts (immediate early genes: Fos, Arc; ribosomal protein genes) should increase in imputed profiles relative to raw nuclei. Known nuclear-enriched transcripts (lncRNAs, chromatin-associated RNAs) should remain unchanged. These expectations are verifiable without paired ground truth.

2. **Cell-type marker improvement:** Imputed profiles should show stronger expression of known cell-type marker genes, improving cell-type classification consistency against literature-derived references.

3. **Downstream analysis:** Apply ImputeVAE to PsychENCODE human brain nuclei and evaluate whether imputed profiles change DE gene detection or cell-type deconvolution results relative to raw nuclei.

The absence of paired human brain data limits direct validation but is also the reason this tool is needed. We will also look at which genes and cell types impute poorly, since systematic failures likely point to species- or tissue-specific RNA localization patterns.

### Next Steps

- Obtain and preprocess macaque paired data for the cross-species transfer experiment.
- Implement ortholog gene mapping across species.
- Design and execute the biological plausibility evaluation on human brain nuclei.

---

## References

Bakken, T. E., et al. (2018). Single-nucleus and single-cell transcriptomes compared in matched cortical cell types. PLoS ONE, 13(12), e0209648.

Hodge, R. D., et al. (2019). Conserved cell types with divergent features in human versus mouse cortex. Nature, 573(7772), 61–68.

Bakken, T. E., van Velthoven, C. T., Menon, V., et al. (2021). Single-cell and single-nucleus RNA-seq uncovers shared and distinct axes of variation in dorsal LGN neurons in mice, non-human primates, and humans. eLife, 10, e64875.

Ding, J., et al. (2020). Systematic comparison of single-cell and single-nucleus RNA-sequencing methods. Nature Biotechnology, 38(6), 737–746.

Gayoso, A., et al. (2021). Joint probabilistic modeling of single-cell multi-omic data with totalVI. Nature Methods, 18(3), 272–282.

Korsunsky, I., et al. (2019). Fast, sensitive and accurate integration of single-cell data with Harmony. Nature Methods, 16(12), 1289–1296.

Lopez, R., et al. (2018). Deep generative modeling for single-cell transcriptomics. Nature Methods, 15(12), 1053–1058.

Xu, C., et al. (2021). Probabilistic harmonization and annotation of single-cell transcriptomics data with deep generative models. Molecular Systems Biology, 17(1), e9620.
