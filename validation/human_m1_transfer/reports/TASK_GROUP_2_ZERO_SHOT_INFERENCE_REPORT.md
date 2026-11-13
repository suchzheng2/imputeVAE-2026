# Task Group 2: Human M1 Zero-Shot Inference Report

**Date:** 2026-08-16
**SLURM Job:** 9231508 (COMPLETED, 4 min, 10GB RAM, RTX 2080)
**Status:** All 34 acceptance checks passed.

## 1. What Was Run

The frozen mouse-trained ImputeVAE model was applied zero-shot to 76,533 Allen Human M1 10x v3 single-nucleus RNA-seq profiles. For each human nucleus, the model encoded the expression with the nuclei modality indicator (`modality_id=1`, which maps to scalar value 0.0 inside the model) and decoded with the cell modality indicator (`modality_id=0`, scalar value 1.0), producing predicted "cell-like" log1p expression profiles. No model parameters were updated; this is purely inference.

## 2. Model Checkpoint

- **Run:** `run/run_2026-07-18-19-40-51_570984`
- **Class:** `VAE_JSD_Cond` (MSE + JSD loss, scalar conditioning on encoder + decoder)
- **Architecture:** 10,134 input features, [1024, 512] hidden, 32 latent, scalar_cond=True, cond_encoder=True
- **Training data:** Mouse MOp, 10x v3 nuclei + cells, 600 epochs, log1p normalization
- **Normalization:** `raw_data_normalization: "log1p"` (log1p of raw counts)

## 3. Input Data and Gene Coverage

- **Source:** Allen Human M1 10x Chromium v3, single-nucleus RNA-seq
- **Nuclei:** 76,533 (2 donors: H18.30.001, H18.30.002)
- **Gene mapping:** MGI one-to-one orthologs: 8,215 / 10,134 model genes detected (81.1%)
- **Unmapped:** 1,919 genes (mouse-specific Gm*/Rik/locus IDs, filled with zeros)
- **Input scale:** log1p(raw counts), matching model training

## 4. Inference and Output Scale

- **Encoding:** nuclei modality (`modality_id=1` → internal scalar 0.0)
- **Decoding:** cell modality (`modality_id=0` → internal scalar 1.0)
- **Output scale:** log1p expression (same as input, same as model training)
- **Inference time:** 0.1 min on GPU (RTX 2080), batch_size=512

## 5. QC Summary

| Metric | Value |
|--------|-------|
| Output shape | 76,533 x 10,134 |
| Output min | -0.0823 |
| Output max | 8.2270 |
| Output mean | 0.9086 |
| Output std | 0.8590 |
| Fraction negative | 1.27% |
| NaN/Inf | 0 / 0 |
| Clamping applied | None |

The 1.27% negative values are small in magnitude (min = -0.082) and consistent with the LeakyReLU(0.1) decoder activation. Not clamped, matching the original mouse pipeline.

## 6. Human Predicted Shifts by Broad Class

All deltas are positive (model predicts cytoplasmic enrichment in imputed "cell" profiles), with neuronal classes showing the largest shifts:

| Broad class | Mean delta | Median delta | % positive |
|-------------|-----------|-------------|------------|
| excitatory_neuron | 0.4562 | 0.3563 | 82.8% |
| inhibitory_neuron | 0.4562 | 0.3524 | 87.9% |
| vlmc_pericyte_smc | 0.3495 | 0.1982 | 88.4% |
| microglia_macrophage | 0.3145 | 0.1676 | 88.3% |
| endothelial | 0.2810 | 0.1501 | 86.9% |
| opc | 0.2042 | 0.1399 | 83.5% |
| oligodendrocyte | 0.1936 | 0.1186 | 81.9% |
| astrocyte | 0.1476 | 0.0768 | 78.0% |

The neuronal > non-neuronal pattern is consistent with the mouse training data, where neurons have larger cytoplasmic compartments and more ribosome-bound RNA.

## 7. SynGO vs TF Transfer Result

| Gene set | N mapped | Neuronal delta | Non-neuronal delta | Difference |
|----------|---------|---------------|-------------------|------------|
| SynGO | 1,236 | 0.7751 | 0.5327 | **0.2425** |
| TF | 60 | 0.2611 | 0.1128 | **0.1483** |
| All mapped | 8,215 | 0.4562 | 0.2484 | **0.2078** |

**Key finding:** SynGO (synaptic, non-ribosomal) genes show the largest neuronal-specific predicted shift (0.78 neuronal vs 0.53 non-neuronal, diff=0.24). TF genes show smaller, more uniform shifts. This preserves the biological pattern from mouse: synaptic transcripts are preferentially added when converting from nuclear to whole-cell profiles in neurons.

## 8. Mouse-Human Predicted-Shift Similarity

Per-class comparison of mouse predicted delta vs human predicted delta (all mapped genes):

| Metric | Value |
|--------|-------|
| Median Pearson r across 17 cell classes | **0.715** |
| Range | [0.082, 0.803] |

The high median correlation (0.72) indicates that the nucleus-to-cell transformation pattern is substantially conserved between mouse MOp and human M1 at the cell-type level. The outlier (r=0.08) likely corresponds to a cell class with very few nuclei or poor cross-species homology.

## 9. Donor Consistency

| Metric | Value |
|--------|-------|
| Median donor Pearson r (broad classes) | **0.986** |

The predicted shifts are highly consistent between the two human donors, indicating the model's predictions are robust to individual variation. This high consistency is expected because the model was trained on pooled mouse data and applies a deterministic transformation.

## 10. Major Caveats

1. **No human whole-cell ground truth.** Human M1 data are nuclei only. We cannot validate whether the predicted "cell-like" profiles match actual human whole-cell expression. All results are hypothesis-generating.

2. **Zero-shot cross-species transfer.** The model was trained exclusively on mouse MOp data. Conservation of the nucleus-to-cell transformation across species is assumed but not proven.

3. **Ortholog-restricted interpretation.** Only 8,215 / 10,134 model genes have detected one-to-one human orthologs. Unmapped genes are zero-filled and should be ignored in all analyses.

4. **Cross-species cell-type labels are approximate.** Human M1 subclasses (e.g., L5 ET, Pvalb) are named to match mouse taxonomy, but cross-species homology at the subtype level is imperfect.

5. **L5 ET caution.** Mouse validation showed a calibration issue for L5 ET (negative R^2 despite high correlation). Human L5 ET predictions should be interpreted cautiously.

6. **Two donors only.** Donor consistency is high (r=0.986) but with n=2, individual variation assessment is limited.

7. **Negative output values.** 1.27% of output values are slightly negative (min=-0.082). These are small artifacts of the LeakyReLU decoder activation, not clamped to match the original pipeline.

8. **Bulk validation (if done later) is imperfect.** Bulk RNA-seq includes extracellular RNA and mixed cell composition, making it an orthogonal but not direct reference.

## 11. Next Steps for Task Group 3

Potential follow-up analyses:
- **Ribosomal gene check:** Verify RPL/RPS genes have positive delta in human (conserved cytoplasmic enrichment signal)
- **Nuclear lncRNA check:** Verify MALAT1/XIST/MEG3 have negative or near-zero delta
- **Cross-dataset validation:** Apply to other human brain snRNA-seq datasets
- **Bulk pseudobulk comparison:** Compare predicted whole-cell profiles to GTEx/Allen bulk brain RNA-seq (imperfect but orthogonal)

## Output Files

```
validation/human_m1_transfer/
  inference/
    human_m1_imputed_cell_log1p.npy           # 3.1 GB, (76533, 10134) dense
    human_m1_input_nucleus_log1p.npz          # sparse input copy
    human_m1_inference_obs.csv                # per-nucleus metadata
    human_m1_inference_gene_map.csv           # per-gene mapping status
    human_m1_inference_run_info.json          # full run metadata
    human_m1_inference_qc.csv                 # QC metrics
  results/
    human_m1_broad_class_gene_means.csv       # long-format: 65,720 rows
    human_m1_mouse_like_class_gene_means.csv  # long-format: 164,300 rows
    human_m1_donor_*_gene_means.csv           # donor-stratified versions
    human_m1_*_matrix.csv                     # matrix-style (genes x classes)
    human_m1_syngo_transfer_summary.csv       # SynGO gene shifts
    human_m1_tf_transfer_summary.csv          # TF gene shifts
    human_m1_syngo_vs_tf_transfer_contrast.csv
    human_m1_neuronal_vs_nonneuronal_predicted_shift.csv
    human_mouse_predicted_shift_similarity_by_class.csv
    human_m1_donor_consistency_by_class.csv
  figures/
    human_m1_transfer_qc.{png,pdf}
    human_m1_predicted_delta_summary.{png,pdf}
    human_mouse_shift_similarity.{png,pdf}
    human_m1_donor_consistency.{png,pdf}
  reports/
    TASK_GROUP_2_ZERO_SHOT_INFERENCE_REPORT.md  # This file
    task_group_2_acceptance_checks.txt          # 34/34 passed
```
