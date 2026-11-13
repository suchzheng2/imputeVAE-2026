# Human M1 Global Imputation QC Report

**Date:** 2026-08-17
**Script:** `validation/human_m1_transfer/scripts/10_global_imputation_qc.py`

## 1. Input Files and Sample Sizes

| Input | Path | Shape/Size |
|-------|------|-----------|
| Nucleus matrix | `processed/human_m1_nuclei_mouse_model_order.npz` | 76533 x 10134 |
| Imputed matrix | `inference/human_m1_imputed_cell_log1p.npy` | 76533 x 10134 |
| Mapped genes | `processed/human_m1_available_gene_mask.npy` | 8215 / 10134 |
| Metadata | `inference/human_m1_inference_obs.csv` | 76533 rows |

## 2. Subsampling

- Method: balanced by `mapped_mouse_like_class`, max 500 per class
- Seed: 42
- Sampled: **7889** nuclei from 20 mouse-like classes
- Both raw nucleus and corresponding imputed profile included for each sampled nucleus

## 3. Gene Selection

- Mapped genes: 8215
- HVG method: top 2000 by variance in raw nucleus matrix (sampled cells)
- Standardized before PCA

## 4. PCA/UMAP

- PCA: 50 components, explained variance: 0.788
- UMAP: computed (n_neighbors=30, min_dist=0.3)

## 5. Centroid Shift Summary

### Top 5 broad classes by centroid shift distance:
- **vlmc_pericyte_smc**: shift=39.996, mean_delta=0.3495, n=40 ⚠️
- **excitatory_neuron**: shift=38.565, mean_delta=0.4754, n=3827
- **microglia_macrophage**: shift=38.181, mean_delta=0.3145, n=108 ⚠️
- **inhibitory_neuron**: shift=36.081, mean_delta=0.4362, n=2567
- **endothelial**: shift=32.805, mean_delta=0.2810, n=64 ⚠️

### Top 5 mouse-like classes by centroid shift distance:
- **L6_IT_Car3**: shift=48.279, mean_delta=0.5531, n=327
- **Pvalb**: shift=46.112, mean_delta=0.5674, n=500
- **L5_ET**: shift=45.133, mean_delta=0.4632, n=500
- **L6_CT**: shift=43.872, mean_delta=0.5349, n=500
- **L6b**: shift=43.637, mean_delta=0.4935, n=500

### Bottom 3 mouse-like classes by centroid shift distance:
- **Oligo**: shift=26.281, mean_delta=0.1945, n=500
- **OPC**: shift=23.845, mean_delta=0.2042, n=283
- **Astro**: shift=22.505, mean_delta=0.1453, n=500

## 6. Cell Identity Preservation

### Broad class:
| Class | n | Raw self-assign | Imputed → original | Drop |
|-------|---|----------------|-------------------|------|
| vlmc_pericyte_smc | 40 | 0.750 | 0.000 | 0.750 |
| microglia_macrophage | 108 | 0.880 | 0.000 | 0.880 |
| endothelial | 64 | 0.875 | 0.156 | 0.719 |
| astrocyte | 500 | 0.908 | 0.342 | 0.566 |
| inhibitory_neuron | 2567 | 0.829 | 0.521 | 0.308 |
| oligodendrocyte | 500 | 0.906 | 0.764 | 0.142 |
| opc | 283 | 0.954 | 0.908 | 0.046 |
| excitatory_neuron | 3827 | 0.894 | 0.956 | -0.062 |

### Mouse-like class (summary):
- Median imputed → original rate: **0.161**
- Min: 0.000 (L5/6_NP)
- Max: 0.795 (OPC)

## 7. Program Score Shifts (Broad Class)

| Class | SynGO | TF | Ribosomal | Nuclear lncRNA |
|-------|-------|-----|-----------|----------------|
| astrocyte | 0.3480 | 0.0372 | 0.5586 | 0.8459 |
| endothelial | 0.6153 | 0.1250 | 0.8027 | 1.3786 |
| excitatory_neuron | 0.7729 | 0.3175 | 1.3794 | 0.6777 |
| inhibitory_neuron | 0.7773 | 0.2047 | 1.4634 | 1.0056 |
| microglia_macrophage | 0.6918 | 0.1715 | 0.8856 | 1.5689 |
| oligodendrocyte | 0.4097 | 0.1101 | 0.7553 | 0.8456 |
| opc | 0.3928 | 0.0406 | 0.7670 | 0.8550 |
| vlmc_pericyte_smc | 0.7384 | 0.1924 | 0.9873 | 1.5308 |

Gene counts: SynGO=1236, TF=60, Ribosomal=70, Nuclear_lncRNA=3

## 8. Distributional Delta Summary (Broad Class)

| Class | n_nuclei | Mean delta | Median delta | % positive | % large (>0.5) | IQR |
|-------|---------|-----------|-------------|-----------|---------------|-----|
| inhibitory_neuron | 23992 | 0.4562 | 0.3525 | 87.9% | 37.5% | 0.5785 |
| excitatory_neuron | 48536 | 0.4562 | 0.3563 | 82.8% | 39.5% | 0.6635 |
| vlmc_pericyte_smc | 40 | 0.3495 | 0.1982 | 88.4% | 23.5% | 0.4128 |
| microglia_macrophage | 108 | 0.3145 | 0.1676 | 88.3% | 20.8% | 0.3751 |
| endothelial | 64 | 0.2810 | 0.1501 | 86.9% | 18.5% | 0.3390 |
| opc | 283 | 0.2042 | 0.1399 | 83.5% | 12.7% | 0.2729 |
| oligodendrocyte | 2942 | 0.1936 | 0.1186 | 81.9% | 13.0% | 0.2718 |
| astrocyte | 568 | 0.1476 | 0.0768 | 78.0% | 8.8% | 0.2075 |

## 9. Low-Sample Warnings

Classes with fewer than 200 sampled nuclei:
- broad_class: **endothelial** (n=64)
- broad_class: **microglia_macrophage** (n=108)
- broad_class: **vlmc_pericyte_smc** (n=40)
- mouse_like_class: **Endo** (n=64)
- mouse_like_class: **Micro** (n=108)
- mouse_like_class: **Sst_Chodl** (n=67)
- mouse_like_class: **VLMC** (n=40)

## 10. Mouse-Human Delta Projection (Part F)

Figure saved: `human_m1_mouse_human_delta_projection.png/pdf`

## 11. Interpretation and Caveats

1. **No human whole-cell ground truth.** Allen Human M1 data are nuclei only. We cannot validate whether imputed profiles match actual human whole-cell expression.
2. **Zero-shot cross-species transfer.** The model was trained exclusively on mouse MOp data.
3. **Ortholog-restricted genes only.** Only 8215/10134 model genes have detected one-to-one human orthologs. Unmapped genes are zero-filled and excluded from all analyses here.
4. **Cell-type mapping is approximate** at the fine-subclass level (human M1 subclasses named to match mouse taxonomy).
5. **SynGO annotates protein function/location, not RNA localization.** The SynGO score shift reflects that genes encoding synaptic proteins are more cytoplasmically enriched in neurons, but the causal link is protein function, not direct RNA transport.
6. **These are global QC/plausibility checks, not direct human validation.**

## 12. Summary

Global QC shows whether the frozen mouse-trained model shifts human M1 nuclei toward cell-like profiles while preserving cell identity and biological program structure. Because no human whole-cell profiles are available, these analyses support cross-species plausibility rather than direct human accuracy.

## Output Files

```
figures/
  human_m1_global_pca_umap.{png,pdf}
  human_m1_centroid_arrows.{png,pdf}
  human_m1_identity_preservation.{png,pdf}
  human_m1_program_score_shifts.{png,pdf}
  human_m1_distributional_delta_summary.{png,pdf}
  human_m1_mouse_human_delta_projection.{png,pdf}
results/
  human_m1_global_embedding.csv
  human_m1_centroid_shift_summary.csv
  human_m1_identity_preservation.csv
  human_m1_program_score_shifts.csv
  human_m1_distributional_delta_summary.csv
reports/
  HUMAN_M1_GLOBAL_QC_REPORT.md
```
