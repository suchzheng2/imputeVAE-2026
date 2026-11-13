---
name: project-scorch-scoping
description: SCORCH data on the group share (2026-10-23) — human snRNA-seq 655k nuclei PFC/INS/VST HIV vs CTR with cortical + MSN labels, no mouse spatial data found; candidate nuclei-only application
metadata:
  type: project
---
`notes/experiments/yscorch_scoping/README.md`. `/gpfs/gibbs/pi/gerstein/yscorch_data_share/seurat_combined_v3.RDS` = human snRNA-seq Seurat v5 (654,918 nuclei, 33 donors, INS/PFC/VST, HIV 375k / CTR 280k, `consensus_annotation_v3` 37 labels incl. L2/3 IT, L5 IT, L6 IT/CT, Pvalb, Sst, Vip, Lamp5, Astro, Oligo, OPC, Micro, D1/D2 MSN subtypes). No spatial slot; no mouse spatial SCORCH data anywhere under the group dir (ask the advisor where it lives). Reading it needs `module load R/4.4.1-foss-2022b R-bundle-Bioconductor/3.19-foss-2022b-R-4.4.1` (the `-bare` module lacks Seurat), ~3 min, < 120 GB.

- 2026-11-06 lookup: the mouse spatial set the advisor means is most likely the **Allen Institute SCORCH project
  (U01DA056003, Zeng/Kenny/Volsky): EcoHIV + cocaine mice, Multiome + MERFISH**, data via the NeMO SCORCH portal
  (`scorch-portal.nemoarchive.org`, JS; browse by hand). Yale's own Y-SCORCH spatial transcriptomics is **human** (PFC/VST/INS)
  and its analysis tree lives on `cantor.yale.edu:/banach2/SCORCH/` (not the HPC). Region match 2026-11-06: Allen SCORCH = PFC, striatum, habenula (no MOp); public MOp-covering alternatives (ABC MERFISH 500 genes, Zhuang MERFISH 1,122 genes, 10x Xenium 5K mouse brain with per-transcript nuclear flag) listed in the README.

**Why:** advisor proposed a SCORCH application (mouse brain spatial); the human nuclei-only set is the realistic downstream application (storyline gap 1) and the spatial set, if found, is the in-situ validation of the compartment delta.
**How to apply:** use the human-M1 ortholog mapping and the covariate model with an explicit human whole-cell library target. See [[project-libsize-covariate]], [[human-m1-transfer]].
- 2026-11-13: user has access to the private SCORCH portal (private-scorch-portal.nemoarchive.org); checklist for the Allen MERFISH release in the scoping README 'Portal access' section.
- 2026-11-13 RESULT: the SCORCH portal holds NO spatial data at all (Technique facet over all 17 studies = sequencing only); Allen mouse MERFISH and Yale spatial are not deposited. Portal use for us = mouse PFC EcoHIV x cocaine snRNA-seq (Zeng, processed npz/h5 outputs) as a nuclei-only application. The in-situ validation stays on the public Xenium section.
