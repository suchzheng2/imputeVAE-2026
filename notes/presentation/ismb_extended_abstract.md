# imputeVAE: Translating Single-Nucleus to Single-Cell Expression Profiles via Cross-Reconstructive Variational Autoencoders

**Authors:** [TBD — confirm author list and affiliations]

---

## Introduction

Single-nucleus RNA sequencing (snRNA-seq) has become the method of choice for profiling transcriptomes from frozen and archived tissues, enabling large-scale brain atlases including the Allen Brain Cell Atlas (Yao et al., 2023) and PsychENCODE (Wang et al., 2018). Unlike single-cell RNA sequencing (scRNA-seq), which requires fresh tissue dissociation, snRNA-seq works on banked specimens and is compatible with high-throughput droplet platforms at scale. However, nuclei systematically differ from whole cells: they capture predominantly nuclear-retained transcripts, exhibit lower total library sizes, and show biased detection rates for cytoplasmic and activity-dependent genes such as immediate early genes (Bakken et al., 2018; Ding et al., 2020).

These systematic differences mean that downstream analyses — cell-type classification, differential expression testing, gene co-expression networks — can yield discordant results depending on the input modality. Existing computational approaches treat this as a batch effect (e.g., scVI; Lopez et al., 2018) or apply per-gene linear transformations. Batch correction does not model the directional, biology-driven nature of cell-nucleus differences, while per-gene regression ignores gene-gene covariance structure.

We present imputeVAE, a variational autoencoder framework that explicitly learns to translate nucleus expression profiles into cell-equivalent profiles. Our approach uses a conditioned decoder with learned modality embeddings and a cross-reconstruction training objective that directly optimizes the nucleus-to-cell translation pathway. We show this architecture substantially outperforms both raw nuclei and linear regression baselines on cell-type-level metrics, and we provide a zero/nonzero partition analysis that reframes how continuous decoders on sparse single-cell data should be evaluated.

## Methods

imputeVAE uses a symmetric encoder-decoder architecture (input → 1024 → 512 → 32-dim latent → 512 → 1024 → output) trained with MSE reconstruction on log1p-normalized expression and JSD divergence regularization. Two key components enable cross-modal translation:

**Conditioned decoder.** A learned modality embedding (dim 8) is concatenated to the latent z before decoding, enabling modality-specific output from shared parameters. This prevents competing reconstruction gradients from degrading decoder performance when training on both modalities jointly.

**Cross-reconstruction.** Nuclei are encoded into z and decoded with the *cell* embedding, with MSE computed against real cell expression (cell-type-matched, since data is unpaired). This directly optimizes the nucleus-to-cell translation pathway.

Training proceeds in two stages: (1) cell-only reconstruction (600 epochs) to establish a well-structured latent space, followed by (2) joint training (600 epochs) with cascading loss annealing that sequentially introduces nuclei reconstruction, JSD, latent alignment, and cross-reconstruction losses.

**Data.** Allen Brain Cell Atlas (Yao et al., 2023), 10X v3 chemistry: 71,117 cells and 40,166 nuclei across 17 cell types, 16,439 genes (per-cell-type 5% detection filtering). **Evaluation.** Cross-decoded nuclei are assessed against real cells via cell-type classification accuracy, top-100 DE gene overlap, gene-level Pearson correlation, coexpression preservation, and zero/nonzero partition analysis. Baselines: raw nuclei and per-gene linear regression.

## Results

### Model comparison

Cross-decoded nuclei from the full model (cond + xr) closely resemble real cells in UMAP space, with cell-type clusters overlapping between cross-decoded and real cell populations (Fig. 1B). Quantitatively, cross-decoded nuclei achieve 94.1% cell-type classification accuracy — a +10.6 percentage point improvement over raw nuclei (Fig. 1C, Table 1). Per-gene Pearson correlation of cell-type means reaches 0.963 (R² = 0.893), and cross-decoded expression shifts 16.4% closer to real cell profiles relative to raw nuclei (mean shift score +0.164). Gene-gene coexpression structure is also well preserved: Spearman correlation between gene-gene correlation matrices (top 500 variable genes) is 0.898, compared to 0.819 for raw nuclei.

**Table 1.** Performance across model variants (PerCT 5% gene set, 16,439 genes).

| Variant | Classification | DE Overlap | Gene Corr | Coexpression |
|---------|:---:|:---:|:---:|:---:|
| Raw nuclei | 83.5% | 45.0% | 0.803 | 0.819 |
| Linear regression | 93.0% | 46.0% | 0.800 | 0.820 |
| VAE cond (no xr) | 88.7% | 49.5% | 0.909 | 0.929 |
| **VAE cond + xr** | **94.1%** | **54.6%** | **0.963** | **0.898** |

Cross-reconstruction is the critical ingredient, adding +5.4pp in classification and +5.1pp in DE overlap over the conditioned decoder alone (Fig. 1C). The full model outperforms linear regression on gene correlation (0.963 vs 0.800) and DE overlap (54.6% vs 46.0%), while matching it on classification. Linear regression preserves per-cell rank order perfectly (within-CT Spearman 1.0) but cannot model nonlinear or gene-gene relationships in the cell-nucleus mapping.

### Zero/nonzero partition analysis

Standard evaluation metrics aggregate across all gene×cell entries, but in sparse single-cell data, 60–80% of entries are zeros. We partition entries by whether the raw expression is zero or nonzero and evaluate each subset independently.

**Table 2.** Zero/nonzero partition statistics for cell reconstruction (same-modality, not cross-decoded).

| Partition | Pearson | Std ratio | CT mean baseline Pearson |
|-----------|:---:|:---:|:---:|
| Nonzero entries | 0.860 | 1.10 | 0.797 |
| Zero entries (decoder mean) | — | — | — |
| Overall (including zeros) | ~0.35 | 0.20 | — |

Among actually expressed entries (nonzero), the model achieves Pearson 0.860 with std ratio 1.10 — the decoder preserves the scale of real expression values. This exceeds the cell-type mean baseline (0.797), confirming the model captures genuine per-cell variation beyond cell-type identity. The dramatic drop in overall metrics is driven by zero entries: the continuous decoder outputs small positive values (~0.18) where the true value is zero, because LeakyReLU activations cannot produce exact zeros.

Zero-inflation accounts for approximately 73% of total raw variance. A continuous decoder fundamentally cannot reproduce this discrete structure. This is not a training failure but an architectural property shared by all continuous single-cell decoders.

**By gene sparsity** (cell reconstruction, nonzero entries):

| Sparsity bucket | Nonzero Pearson | Std ratio | CT mean Pearson |
|-----------------|:---:|:---:|:---:|
| Dense (<30% zero) | 0.862 | 0.97 | 0.770 |
| Moderate (30–70%) | 0.556 | 0.85 | 0.451 |
| Sparse (70–90%) | 0.542 | 0.82 | 0.493 |
| Very sparse (>90%) | 0.525 | 0.79 | 0.484 |

The model consistently outperforms the cell-type mean baseline across all sparsity levels, with the largest advantage on dense genes (+9.2pp Pearson). The decoder also achieves AUROC 0.923 for distinguishing zero vs. nonzero positions — it "knows" where genes are expressed but represents this as a soft gradient rather than a discrete boundary.

## Discussion

**Practical utility.** imputeVAE enables reuse of large nuclei atlases (PsychENCODE, BRAIN Initiative, Human Cell Atlas) for analyses originally benchmarked on single-cell data. The 94.1% classification accuracy and 54.6% DE overlap demonstrate that cross-decoded nuclei are substantially more cell-like than raw nuclei across multiple biologically meaningful metrics.

**Evaluation insight.** The zero/nonzero partition analysis has implications beyond this specific model. Standard metrics for continuous single-cell decoders — variance ratio, within-cell-type correlation — are dominated by zero-inflation structure that no continuous decoder can reproduce. Evaluating on nonzero entries separately provides a fairer and more informative assessment of model quality. We recommend this partitioned evaluation become standard practice for generative models of single-cell expression.

**Limitations.** Data is unpaired (cells and nuclei from different specimens), preventing per-cell ground truth evaluation. Current training and evaluation use mouse visual cortex from a single atlas; generalization to other tissues and species remains to be tested. The continuous decoder cannot represent the discrete zero/nonzero structure of sparse single-cell data.

**Future directions.** A two-headed decoder architecture (Bernoulli gate for zero/nonzero + continuous head for expression level) could recover the missing variance structure. Cross-species transfer to human nuclei data would demonstrate broader applicability. Direct comparison with scVI as a modality-translation baseline is planned.

## References

1. Yao, Z. et al. A high-resolution transcriptomic and spatial atlas of cell types in the whole mouse brain. *Nature* 624, 317–332 (2023).
2. Lopez, R. et al. Deep generative modeling for single-cell transcriptomics. *Nature Methods* 15, 1053–1058 (2018).
3. Wang, D. et al. Comprehensive functional genomic resource and integrative model for the human brain. *Science* 362, eaat8464 (2018).
4. Bakken, T. E. et al. Single-nucleus and single-cell transcriptomes compared in matched cortical cell types. *PLoS ONE* 13, e0209648 (2018).
5. Ding, J. et al. Systematic comparison of single-cell and single-nucleus RNA-sequencing methods. *Nature Biotechnology* 38, 737–746 (2020).
6. Grønbech, C. H. et al. scVAE: Variational auto-encoders for single-cell gene expression data. *Bioinformatics* 36, 4415–4422 (2020).

---

*Figure 1 (multi-panel, see separate file):*
- **(A)** Architecture schematic — encoder → shared latent z → conditioned decoder with cell/nuclei modality embedding; cross-decode path highlighted
- **(B)** UMAP visualization — raw nuclei, cross-decoded nuclei, raw cells (colored by cell type)
- **(C)** Model comparison bar chart — classification accuracy and DE overlap across 3 variants + linear regression baseline
- **(D)** Zero/nonzero analysis — nonzero Pearson correlation vs. cell-type mean baseline, stratified by gene sparsity
