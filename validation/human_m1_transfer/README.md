# Human M1 Zero-Shot Transfer Pilot

**Date:** 2026-08-16
**Status:** Task Group 2 complete. Inference run, summaries generated, all 34 acceptance checks passed.

## Purpose

Prepare all inputs needed to run the frozen mouse-trained ImputeVAE model on Allen Human M1 10x v3 single-nucleus data. This is a zero-shot, ortholog-restricted cross-species transfer pilot.

## Data Source

**Allen Human M1 10x** (Allen Institute for Brain Science)
- URL: https://portal.brain-map.org/atlases-and-data/rnaseq/human-m1-10x
- Technology: 10x Chromium Single Cell 3' Reagent Kit **v3**, single-nucleus RNA-seq
- Nuclei: **76,533** from 2 postmortem human M1 specimens
- Donors: H18.30.001 (42,728 nuclei), H18.30.002 (33,805 nuclei)
- Cell types: **127** transcriptomic clusters, 20 subclasses
- Expression matrix: raw counts (introns + exons), 50,281 genes

### Files Downloaded

| File | Size | Description |
|------|------|-------------|
| metadata.csv | 23 MB | Per-nucleus annotations (class, subclass, cluster, donor, layer) |
| matrix.csv | 7.2 GB | Raw count matrix (76,533 nuclei x 50,281 genes) |
| trimmed_means.csv | 30 MB | Per-cluster trimmed means (log2 CPM) |
| medians.csv | 26 MB | Per-cluster median expression |
| tsne.csv | 4.6 MB | t-SNE coordinates |
| dend.json | 408 KB | Taxonomy dendrogram |

Checksums: `external/allen_human_m1_10x/file_manifest.csv`

## Ortholog Mapping

**Source:** MGI HOM_MouseHumanSequence.rpt (Jackson Laboratory)
**Method:** One-to-one orthologs only, filtered to mouse model gene list, ambiguous mappings removed.

| Metric | Value |
|--------|-------|
| Mouse model genes | 10,134 |
| Raw human-mouse ortholog rows | 24,590 |
| One-to-one ortholog pairs | 18,782 |
| Model genes with 1:1 human ortholog | 8,433 |
| Human orthologs detected in Allen matrix | **8,215** |
| Percent model genes mapped | **81.1%** |

### Unmapped genes (1,919 of 10,134)

- 669 Gm* predicted genes (mouse-specific)
- 343 Rik RIKEN cDNA clones (mouse-specific)
- 689 other (locus IDs like AC*, some with MGI mapping gaps)
- 218 orthologs exist but human gene absent from Allen matrix

**Known MGI gaps:** Gapdh and Neat1 are real 1:1 orthologs but MGI groups them with pseudogene/retrotransposon copies, breaking the 1:1 filter. These are noted but not manually rescued (preserving systematic mapping).

**Key genes confirmed mapped:** Malat1, Actb, Rpl3, Rps3, Snap25, Meg3, Eef1a1.

## Cell-Type Harmonization

Allen Human M1 annotations mapped to broad classes and mouse-like subclasses:

| Broad class | Nuclei | Mouse-like classes |
|-------------|-------:|--------------------|
| excitatory_neuron | 48,536 | L2/3_IT, L5_IT, L6_CT, L6b, L6_IT, L5/6_NP, L5_ET, L6_IT_Car3 |
| inhibitory_neuron | 23,992 | Pvalb, Sst, Vip, Lamp5, Sncg, Sst_Chodl |
| oligodendrocyte | 2,942 | Oligo |
| astrocyte | 568 | Astro |
| opc | 283 | OPC |
| microglia_macrophage | 108 | Micro |
| endothelial | 64 | Endo |
| vlmc_pericyte_smc | 40 | VLMC |

All 127 clusters mapped with high confidence (Allen subclass labels directly match mouse taxonomy).

## Matrix Preprocessing

1. Loaded raw counts from Allen matrix.csv (76,533 x 50,281)
2. Extracted 8,215 columns matching one-to-one ortholog genes
3. Placed into 10,134-column model-gene-order matrix (zeros for unmapped genes)
4. Applied `log1p(raw_counts)` normalization — matching ImputeVAE training (`raw_data_normalization: "log1p"`)
5. Saved as sparse CSR matrix (scipy npz)

### QC Summary

| Metric | Value |
|--------|-------|
| Nuclei | 76,533 |
| Model genes | 10,134 |
| Transfer genes (mapped + detected) | 8,215 |
| Percent mapped | 81.1% |
| Median genes detected per nucleus | 3,791 |
| Median total counts per nucleus | 16,351 |
| Fraction zero in model matrix | 0.6454 |
| Non-zero entries | 274,994,678 |
| Expression scale | Raw counts |
| Normalization | log1p |

## Files Generated

```
validation/human_m1_transfer/
  README.md                          # This file
  orthologs/
    human_mouse_orthologs_raw.csv    # All MGI human-mouse ortholog pairs (24,590)
    human_to_mouse_one_to_one_orthologs.csv  # 1:1 only (18,782)
    human_to_mouse_model_overlap.csv # 1:1 + in model gene list (8,433)
  external/allen_human_m1_10x/
    metadata.csv                     # Allen metadata
    matrix.csv                       # 7.2 GB raw count matrix
    trimmed_means.csv                # Per-cluster trimmed means
    medians.csv                      # Per-cluster medians
    tsne.csv                         # t-SNE coordinates
    dend.json                        # Taxonomy
    file_manifest.csv                # Checksums and download info
  processed/
    human_m1_nuclei_mouse_model_order.npz  # log1p sparse matrix (76,533 x 10,134)
    human_m1_nuclei_raw_counts.npz   # Raw counts sparse matrix
    human_m1_available_gene_mask.npy # Boolean mask (10,134), True for mapped genes
    human_m1_nuclei_obs.csv          # Per-nucleus metadata with mapped classes
    human_m1_celltype_mapping.csv    # 127-cluster to broad/fine class mapping
    human_m1_gene_mapping_for_model_order.csv  # Per-gene mapping status
  reports/
    ortholog_mapping_summary.csv
    allen_human_m1_metadata_summary.csv
    allen_human_m1_celltype_counts.csv
    allen_human_m1_donor_counts.csv
    celltype_mapping_summary.csv
    human_m1_input_qc.csv
    human_m1_input_qc.png
    human_m1_input_qc.pdf
  scripts/
    01_build_ortholog_map.py
    02_characterize_metadata.py
    03_prepare_model_matrix.py
    04_qc_figures.py
    run_prepare_matrix.sh            # SLURM job (job 9231013, COMPLETED)
```

## Caveats

1. **Human M1 data are nuclei only.** There is no matched human whole-cell ground truth in this dataset. We cannot compute cell-nucleus delta or validate human imputation accuracy.
2. **Zero-shot cross-species transfer.** The model was trained exclusively on mouse MOp data. Cross-species performance depends on conservation of the nucleus-to-cell expression relationship.
3. **Ortholog-restricted gene space.** Only 8,215 of 10,134 model genes have detectable one-to-one human orthologs. Unmapped genes (mostly mouse-specific Gm*/Rik) are filled with zeros. Model predictions on these positions should be ignored.
4. **Fine cell-type labels are approximate.** Human M1 subclasses (e.g., L5 ET, Pvalb) are named to match mouse, but cross-species homology at the subtype level is imperfect.
5. **This task prepares inputs only.** It does not validate human whole-cell imputation. Validation requires either: (a) a matched human cell/nucleus dataset, or (b) comparison of predicted delta patterns against known biology (ribosomal, lncRNA directions).
6. **Two donors only.** Both are from the same brain region (M1) but individual variation is not assessable with n=2.
