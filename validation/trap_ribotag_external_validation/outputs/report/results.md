# Results

## H1: direct compartment concordance

### GSE210071 (cultured cortical cells, triptolide 3 h, paired cultures), 8,158 genes

| Contrast | Method | rho [95% CI] | AUROC | top-10% pos OR | bottom-10% neg OR | matched null | perm p |
|---|---|---|---|---|---|---|---|
| cytosol - nucleus, Ctl | ImputeVAE (ALL) | 0.384 [0.366, 0.402] | 0.802 | 3.4 | 4.6 | -0.019 | 0.001 |
| cytosol - nucleus, Ctl | ground truth | 0.370 [0.351, 0.387] | 0.784 | 3.2 | 3.9 | -0.019 | 0.005 |
| whole cell - nucleus, Ctl | ImputeVAE (ALL) | 0.412 [0.393, 0.430] | 0.822 | 2.2 | 4.6 | -0.005 | 0.001 |
| whole cell - nucleus, Ctl | ground truth | 0.398 [0.379, 0.414] | 0.783 | 2.3 | 4.2 | -0.006 | 0.005 |
| cytosol - nucleus, Bic | ImputeVAE (ALL) | 0.371 [0.351, 0.389] | 0.803 | 3.1 | 4.3 | -0.031 | 0.005 |
| whole cell - nucleus, Bic | ImputeVAE (ALL) | 0.376 [0.357, 0.392] | 0.802 | 0.9 | 5.5 | -0.027 | 0.005 |

Neuron-pooled delta: rho 0.394 / 0.424. Median-centered sign agreement on reliable genes: 0.72 / 0.78. Per-culture replicate rho 0.37 to 0.40. Filters: rho 0.33 (all filters combined) to 0.41. Other methods on cytosol - nucleus: CT-mean 0.377, linear 0.371, scVI 0.392, scANVI 0.400, scGen 0.407, scButterfly 0.391 (AUROC 0.77 to 0.80 vs 0.80 ImputeVAE); on whole - nucleus AUROC scVI family 0.72 to 0.76 vs 0.82.

### GSE121199 (NAc D1/D2 MSNs, male, limma-voom), 9,565 genes, neuron-pooled delta

| Contrast | rho [95% CI] | AUROC | matched null | ground truth rho |
|---|---|---|---|---|
| whole cell - nuclear, subtype-averaged | 0.272 [0.251, 0.293] | 0.731 | -0.055 | 0.271 |
| whole cell - nuclear, D2 | 0.500 [0.484, 0.516] | 0.819 | 0.079 | 0.499 |
| whole cell - nuclear, D1 | -0.092 [-0.107, -0.071] | 0.420 | -0.181 | -0.093 |
| RiboTag - nuclear (triangulation) | 0.572 [0.557, 0.586] | 0.814 | 0.010 | 0.573 |
| RiboTag - whole cell (enrichment-type) | 0.569 [0.553, 0.579] | 0.829 | 0.018 | 0.570 |

D1 nuclear FACS libraries: 4 of 6 below 5 M reads, within-group Spearman 0.84 (D2: 0.97). Including `wc_D1_M_1_2` changes rho by < 0.01. On the subtype-averaged contrast the scVI family scores higher (0.36 to 0.37) than ImputeVAE (0.27) because it is less affected by the D1 anomaly; on D2 alone ImputeVAE 0.50 vs scVI family 0.31 to 0.33.

### H4 nuclear controls
45-gene MOp nuclear set: 38 present in GSE121199, 89.5% externally nuclear (mean log2FC -0.59 vs +0.06 all genes, Mann-Whitney p < 1e-3); in GSE210071 only Xist survives filtering (cytosol - nucleus -7.07). Xist, Meg3, Snhg11 nuclear in GSE210071 (-7.1, -3.3, -2.9); Malat1, Meg3, Snhg11, Kcnq1ot1 nuclear in GSE121199 whole-cell-vs-nuclear (-0.44, -0.76, -0.75, -0.83) and RiboTag-vs-nuclear (-1.6, -1.9, -2.3, -2.3). Model deltas: Xist -1.21, Malat1 -0.83, Snhg11 -0.05, Meg3 +0.84, Kcnq1ot1 +0.8 (Meg3 and Kcnq1ot1 discordant, as pre-registered). Neat1, Pvalb absent from the model gene set.

## H2: profile improvement (Spearman with external RiboTag profile)

| Dataset | CT | Batch (n reps) | nucleus | imputed | real cell | delta_similarity [gene boot] [replicate boot] |
|---|---|---|---|---|---|---|
| GSE263082 | Pvalb | batch1 (3) | 0.555 | 0.741 | 0.741 | 0.186 [0.173, 0.198] [0.179, 0.189] |
| GSE263082 | Pvalb | batch2 (2) | 0.470 | 0.723 | 0.728 | 0.253 [0.239, 0.268] [0.232, 0.268] |
| GSE263082 | Pvalb | batch3 (3) | 0.223 | 0.565 | 0.575 | 0.342 [0.327, 0.358] [0.328, 0.355] |
| GSE263082 | Pvalb | META (8) | 0.432 | 0.733 | 0.738 | 0.302 [0.287, 0.316] |
| GSE263082 | Sst | batch2 (4) | 0.552 | 0.737 | 0.741 | 0.185 [0.171, 0.197] [0.166, 0.203] |
| GSE263082 | Sst | batch4 (3) | 0.253 | 0.565 | 0.565 | 0.311 [0.298, 0.324] [0.309, 0.313] |
| GSE263082 | Sst | META (7) | 0.445 | 0.728 | 0.730 | 0.283 [0.270, 0.297] |
| GSE77243 | Pvalb | (3) | 0.504 | 0.776 | 0.783 | 0.271 [0.258, 0.284] [0.241, 0.288] |
| GSE77243 | Sst | (3) | 0.496 | 0.778 | 0.781 | 0.282 [0.269, 0.294] |

Gap to the real-cell ceiling closed by 97 to 100%. Unchanged after excluding ribosomal/mitochondrial genes and on the validation-split pseudobulks. Specificity (GSE263082 META): imputed Pvalb vs Pvalb RiboTag 0.749, vs Sst RiboTag 0.710; imputed Sst vs Sst 0.732, vs Pvalb 0.702 (real cells 0.755/0.699 and 0.735/0.705). Differential: predicted delta(Pvalb) - delta(Sst) vs RiboTag Pvalb - Sst rho 0.375 (ground truth 0.314; vs batch-adjusted limma log2FC 0.266, ground truth 0.366).

## H3: translational support (baseline RiboTag abundance vs predicted delta)

| Dataset | CT | rho pred | rho GT | rho nucleus expr | partial (given nuc expr) | top-10% pred-positive in top-20% RiboTag | matched null | p |
|---|---|---|---|---|---|---|---|---|
| GSE263082 | Pvalb META | 0.743 | 0.748 | 0.432 | 0.691 | 0.746 | 0.239 | 0.001 |
| GSE263082 | Sst META | 0.747 | 0.755 | 0.445 | 0.687 | 0.741 | 0.245 | 0.001 |
| GSE77243 | Pvalb | 0.786 | 0.797 | 0.504 | 0.718 | 0.756 | 0.257 | 0.001 |
| GSE77243 | Sst | 0.804 | 0.812 | 0.496 | 0.742 | 0.777 | 0.260 | 0.001 |

Per-batch rho 0.65 to 0.71. Decile curves are monotonic (fig6).

## Secondary datasets (enrichment-type; property mismatch expected)
GSE174303 WT CA1: TRAP - input rho 0.240 (CB) and 0.160 (ML) for the excitatory-pooled delta (matched null 0.05, p = 0.001; ground truth 0.238 / 0.160); ML-minus-CB enrichment difference -0.040 (uncorrelated); input ML - CB -0.258. GSE13379 cortex bacTRAP abundance vs matched-CT delta: astro 0.275, L5a/L5 IT 0.346, L5b/L5 ET 0.397, L6/L6 CT 0.360, Cmtm5 oligo 0.294, Olig2 0.237, Cck 0.313, Pnoc 0.345, Cort/Sst 0.352 (matched null 0.00 to 0.05, all p = 0.001; partials 0.23 to 0.40; ground truth within 0.02). IP/UB ratios: -0.27 (Etv1) to +0.08 (astro), matching the GSE141464 finding.
