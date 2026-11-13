---
name: project-spatial-application
description: Spatial application 2026-11-06/13 — Xenium Prime 5K mouse brain (on palmer scratch, hippocampal-level section, no MOp) chosen over MERFISH atlases; in-situ nuclear fraction vs GT delta Spearman -0.57 / AUROC 0.90 (2026-11-13); next = subclass label transfer + gene-filling test
metadata:
  type: project
---

Record: `notes/experiments/spatial_merfish_application/README.md` (gates 1-2, panel files, per-gene normalised GT delta
`gt_delta_per_gene_test_normalised.csv` = cell-minus-nucleus log1p(CP10k), type-balanced).

- Design (chat 2026-11-06): spatial cells are the truth set, never model input. Readouts: (1) hold-out panel genes,
  fill into spatial cells from each reference (raw nuclei / imputed labelled + label-free / real cells ceiling /
  nuclei + per-type mean shift) via kNN and Tangram, score per gene, stratify by compartment bias; (2) label transfer;
  (3) in-situ agreement of the delta; plus per-molecule nuclear fraction vs predicted delta (validation).
- Allen SCORCH mouse data (Zeng, EcoHIV + cocaine, Multiome + MERFISH) = PFC / striatum / habenula, no MOp; not needed.
- Gate result: Yao 500 / Zhuang 1,122 MERFISH panels = 217 / 426 shared genes, zero ribosomal / lncRNA / marker sets,
  21-27 cyto-enriched -> unusable for the compartment demonstration. Xenium Prime 5K mouse brain coronal FF (13.3 GB
  public bundle on cf.10xgenomics.com, 63 k cells) = 2,508 shared genes, delta spread equal to genome, 387 cyto-enriched,
  616 SynGO, no ribosomal (panel design), `overlaps_nucleus` per transcript -> the dataset to use.
- 2026-11-13 scoping (README last section): bundle on palmer scratch (purge after 60 days idle); section is hippocampal
  level (~bregma -1.5 to -2), NO motor cortex (RSP/S1/AUD/TEa) -> subclass-level use only; no labels; 2,214 shared genes
  detected. **In-situ validation:** per-gene Xenium nuclear fraction vs GT delta Spearman -0.57 (2,477 genes), AUROC
  0.90, cyto-enriched median 0.51 vs nuclear-enriched 0.68 (`scoping/figures/fig_xenium_nuclear_fraction_vs_delta.png`).
  Next: subclass label transfer, gene-filling test, model-predicted delta vs in-situ fraction per type.

**Why:** the paper lacks a downstream application on nuclei-only data; a nuclei reference fills spatial data with
nuclear expression levels, and the imputed reference should fix that for cytoplasm-enriched genes.
**How to apply:** build on the Xenium set, not the MERFISH atlases; stratify every result by `delta_norm`; include the
per-type mean-shift baseline (standing conclusion: external tests are matched by predicted type + mean shift).
