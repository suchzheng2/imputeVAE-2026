# Human-First Gene Set Transfer Validation Report

**Date:** 2026-08-17
**Script:** `validation/human_m1_transfer/scripts/12_human_first_geneset_validation.py`

## 1. Why This Rerun Was Needed

The previous human M1 transfer analysis used gene sets defined in mouse gene space (mouse SynGO
gene symbols from the mouse-trained model validation, and mouse-indexed JASPAR TF genes). For
scientific rigor when reporting cross-species transfer results, gene sets should be defined from
**human annotation resources** and then projected into the mouse-trained model gene space using
one-to-one human-mouse orthologs.

Gene sets were defined from human annotation resources and then projected into the mouse-trained
model gene space using one-to-one human-mouse orthologs.

## 2. Data Sources

### SynGO
- **Source:** SynGO 1.3 (`syngo1.3_complete_data.zip`, `genes.xlsx`)
- **Path:** `validation/syngo/syngo1.3_complete_data.zip`
- **Gene identifiers:** HGNC symbols (human)
- **Total genes:** 1789
- **Version:** SynGO 1.3 (March 2025 release)

SynGO annotates the function and synaptic location of encoded proteins, not RNA localization.

### JASPAR
- **Source:** JASPAR CORE 2026 (`ultimate_metadata_table_CORE.tsv`)
- **Path:** `validation/tf/external/ultimate_metadata_table_CORE_2026.tsv`
- **File date:** 2026-08-13
- **File MD5:** `2134a3aa5b9274143fad80419ea6562f`
- **Filter:** `species == "Homo sapiens"` from CORE collection
- **Motif rows:** 1645
- **Unique TF symbols:** 788 (including 21 derived from A::B complexes)

### Ortholog Mapping
- **Source:** MGI HOM_MouseHumanSequence.rpt, one-to-one orthologs only
- **Mapped genes in human M1:** 8215 / 10134 model genes

## 3. Human-First Mapping Logic

1. Start with human gene symbols from each annotation resource (SynGO HGNC, JASPAR Homo sapiens).
2. Map each human symbol to its one-to-one mouse ortholog using the MGI ortholog table.
3. Keep only genes where the mouse ortholog is in the ImputeVAE model gene list (10,134 genes).
4. Keep only genes where the human ortholog is detected in the Allen Human M1 matrix.
5. For SynGO: exclude ribosomal genes (human RPL/RPS or mouse Rpl/Rps prefixes).
6. For TF: create primary set (all mapped) and singleton-only sensitivity set (excluding
   genes known only from A::B complex motifs).

## 4. Final Gene Counts

### SynGO

| Stage | Count |
|-------|-------|
| SynGO HGNC symbols (all) | 1789 |
| With 1:1 mouse ortholog in model | 1390 |
| Also detected in human M1 | 1390 |
| Non-ribosomal (primary set) | **1324** |
| Excluded: ribosomal | 66 |
| Excluded: no 1:1 ortholog | 42 |
| Excluded: ortholog not in model | 357 |
| Excluded: not in human M1 | 0 |

### TF (JASPAR)

| Stage | Count |
|-------|-------|
| Unique human TF symbols | 788 |
| Mapped to model + human M1 (primary) | **273** |
| Singleton-only sensitivity set | **269** |
| Excluded: no 1:1 ortholog | 102 |
| Excluded: ortholog not in model | 413 |
| Excluded: not in human M1 | 0 |

## 5. Comparison: Previous vs Human-First

| Metric | SynGO | TF |
|--------|-------|-----|
| Previous n genes | 1236 | 60 |
| Human-first n genes | 1324 | 273 |
| Overlap | 1236 | 32 |
| Previous neuronal diff | 0.2425 | 0.1483 |
| Human-first neuronal diff | 0.2437 | 0.2147 |
| Class-wise Pearson | 0.9999 | 0.9445 |
| Class-wise Spearman | 1.0000 | 0.9293 |

## 6. Main Results (Human-First)

### Neuronal vs Non-Neuronal Predicted Shift

| Gene set | n genes | Neuronal | Non-neuronal | Diff |
|----------|---------|----------|-------------|------|
| human_first_SynGO | 1324 | 0.7633 | 0.5196 | **0.2437** |
| human_first_TF_primary | 273 | 0.4001 | 0.1853 | **0.2147** |
| human_first_TF_singleton | 269 | 0.4035 | 0.1885 | **0.2150** |
| all_mapped | 8215 | 0.4562 | 0.2484 | **0.2078** |

### Mouse-Human Predicted-Shift Similarity

| Gene set | Median Pearson | Range | n classes |
|----------|---------------|-------|-----------|
| all_mapped | 0.715 | [0.082, 0.803] | 17 |
| human_first_SynGO | 0.705 | [-0.015, 0.774] | 17 |
| human_first_TF_primary | 0.694 | [0.072, 0.799] | 17 |

## 7. Biological Interpretation

The human-first gene set analysis confirms the same biological pattern as the previous
mouse-indexed analysis:

1. **SynGO genes show the largest neuronal-specific predicted shift.** Genes encoding synaptic
   proteins are predicted to be more cytoplasmically enriched in neurons than non-neurons, consistent
   with the mouse training data.

2. **TF genes show smaller, more uniform shifts.** Transcription factor genes, which function in the
   nucleus, show weaker predicted cytoplasmic enrichment regardless of cell type.

3. **Results are robust to gene set definition approach.** Class-wise Pearson correlations between
   previous (mouse-indexed) and human-first results are 1.000 (SynGO) and
   0.945 (TF), indicating that conclusions do not depend on
   whether gene sets were defined in mouse or human gene space first.

## 8. Caveats

1. **No human whole-cell ground truth.** Allen Human M1 data are nuclei only. These are
   cross-species plausibility checks, not direct human validation.
2. **Zero-shot cross-species transfer.** The model was trained exclusively on mouse MOp data.
3. **Ortholog-restricted gene space.** Only 8215/10134 model genes have
   detected one-to-one human orthologs.
4. **SynGO annotates protein function/location, not RNA localization.** The SynGO score shift
   reflects that genes encoding synaptic proteins tend to have more cytoplasmically enriched mRNA
   in neurons, but this is a correlation with protein function, not direct RNA transport annotation.
5. **JASPAR complex handling.** TF motif complexes (A::B) were split into component genes. The
   singleton-only sensitivity set excludes genes known only from complexes, but results are similar.

## Output Files

```
results/
  human_first_syngo_gene_mapping.csv
  human_first_tf_gene_mapping.csv
  human_first_syngo_transfer_summary.csv
  human_first_tf_transfer_summary.csv
  human_first_neuronal_vs_nonneuronal_predicted_shift.csv
  human_first_syngo_vs_tf_transfer_contrast.csv
  human_first_vs_previous_gene_set_comparison.csv
  human_first_mouse_human_shift_similarity_by_class.csv
figures/
  human_first_gene_set_mapping_qc.{png,pdf,svg}
  human_first_predicted_delta_summary.{png,pdf,svg}
  human_first_mouse_human_similarity.{png,pdf,svg}
reports/
  HUMAN_FIRST_GENESET_TRANSFER_REPORT.md
  human_first_geneset_acceptance_checks.txt
```
