# SUC-5: PerCT_5pct Gene List Retraining

**Linear:** https://linear.app/suchen/issue/SUC-5/perct-5pct-gene-list-retraining
**Status:** In Progress
**Created:** 2026-04-02

## Goal

Fix near-zero DE overlap for non-neuronal cell types caused by Config B gene filtering.

## Hypothesis

Config B (10,134 genes, 5% global detection) removes cell-type-specific markers that are only highly expressed in small populations. Switching to PerCT_5pct (5% detection in ANY cell type in EITHER modality) recovers these markers and should dramatically improve non-neuronal DE.

## Background

Config B removes 29/59 known brain cell-type markers:
- Pvalb (Pvalb neurons), Sncg (Sncg neurons)
- Aqp4 (astrocytes), Mog (oligodendrocytes)
- All OPC markers (Pdgfra, Cspg4, Olig1/2)
- All endothelial markers (Flt1, Cldn5, Pecam1)
- Most microglia markers (C1qa/b/c, Ctss, Tmem119)

This directly causes: Astro DE=0.04, Endo DE=0.02, Oligo DE=0.06.

PerCT_5pct recovers 58/59 markers (only Aldh1l1 missing).

| Strategy | Genes | Markers | Cell Var% | Nuc Var% |
|----------|-------|---------|-----------|----------|
| Config B | 10,134 | 30/59 | 85.0% | 90.6% |
| PerCT 5% | 16,439 | 58/59 | 93.1% | 93.9% |

Gene filtering analysis: `notes/gene_filtering_analysis/`
Gene list file: `data/gene_list_perct_5pct.csv`
