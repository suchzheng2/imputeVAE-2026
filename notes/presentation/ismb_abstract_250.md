# ISMB 2026 Non-Proceedings Abstract (250 words)

**Title:** imputeVAE: Translating Single-Nucleus to Single-Cell Expression Profiles via Cross-Reconstructive Variational Autoencoders

**Authors:** [TBD]

---

Single-nucleus RNA sequencing (snRNA-seq) enables transcriptomic profiling from frozen tissues, driving large-scale atlases such as the Allen Brain Cell Atlas and PsychENCODE. However, nuclei systematically differ from whole cells — capturing less cytoplasmic RNA with biased gene detection. Translating nucleus profiles into cell-equivalent expression would unlock these massive datasets for cell-level analyses.

We present imputeVAE, a variational autoencoder that translates between single-nucleus and single-cell expression. A shared encoder maps both modalities into a common latent space, and a conditioned decoder receives a learned modality embedding for modality-specific reconstruction. A cross-reconstruction objective — encoding nuclei and decoding with the cell embedding — explicitly trains the translation pathway. Two-stage training with cascading loss annealing uses MSE on log1p-normalized data, outperforming negative binomial likelihood due to softmax bottleneck effects. The model is trained on Allen Brain Cell Atlas data (71,117 cells, 40,166 nuclei, 16,439 genes, 17 cell types).

Cross-decoded nuclei achieve 94.1% cell-type classification (+10.6 percentage points over raw nuclei), gene-level Pearson correlation of 0.963, and DE gene overlap of 54.6% (+9.6pp), outperforming per-gene linear regression on all cell-type-level metrics. Ablation confirms both conditioned decoder and cross-reconstruction are necessary.

Partitioning evaluation by zero versus nonzero expression reveals the model preserves scale among expressed genes (nonzero Pearson 0.86, std ratio 1.0) — apparent variance collapse is a zero-inflation artifact inherent to continuous decoders on sparse data, not a training failure. This reframes evaluation of continuous generative models for single-cell expression.
