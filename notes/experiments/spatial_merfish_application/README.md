# Spatial (MERFISH / Xenium) application scoping

Started 2026-11-06. Idea (chat, 2026-11-06): use our imputed nuclei as the *reference* for imaging-based spatial data
of the same region (primary motor cortex): (1) hold out panel genes and fill them into spatial cells from each reference
(raw nuclei / imputed / real cells ceiling / nuclei + per-type mean shift), (2) label transfer, (3) in-situ agreement of
the compartment delta. The spatial data are the truth set and are never model input (encoder needs the 10,134-gene
nucleus vector). Full step list in the chat record of 2026-11-06; to be copied here once the design is fixed.

## Gate 1: panel overlap (2026-11-06)

Panels from the ABC Atlas S3 manifest (release 20240330): `MERFISH-C57BL6J-638850` (Yao 2023, 550 genes incl. blanks)
and `Zhuang-ABCA-1` (Zhang 2023, 1,122 genes). Files: `overlap_<panel>.txt`, `overlap_<panel>_delta.csv`,
`gt_delta_per_gene_test_normalised.csv` (per-gene cell-minus-nucleus delta on log1p(CP10k), type-balanced mean over the
14 benchmark types, test partition), `panel_overlap_2026-11-06.csv`.

| | Yao 500 | Zhuang 1,122 |
|---|---|---|
| genes in our 10,134-gene list | 217 | 426 |
| ribosomal (73) / nuclear lncRNA (45) / cytoplasmic markers (18) on panel | 0 / 0 / 0 | 0 / 0 / 0 |
| overlap genes with normalised delta > +0.3 (cytoplasm-enriched) | 21 | 27 |
| overlap genes with delta < -0.3 (nuclear-enriched) | 1 | 2 |
| overlap genes with abs(delta) <= 0.1 (neutral) | 142 | 307 |
| sd of delta on panel vs genome-wide | 0.17 vs 0.25 | 0.15 vs 0.25 |

Reading: the atlas panels are cell-type marker panels. Two thirds of their genes are compartment-neutral, every curated
compartment gene set is absent, and only 21-27 cytoplasm-enriched genes are scoreable (Nfix, Rgs4, Slc17a7, Slc38a1,
Gad2, Vgf, ...). A hold-out gene-filling test on these panels would mostly test neutral genes, where no reference should
differ. These panels are fine for label transfer (readout 2) and a weak version of readout 1, NOT for the
compartment-specific demonstration. Also: 333 / 696 panel genes are not in our 10,134-gene list although they are in
the 25,803 common genes (Aqp4, Acta2, Pvalb, Rpl13a ... removed by the Config B filter: detection rate > 5 % in both
modalities + mean expression > 0.01 + MitoCarta excluded), so the model cannot supply them either.

Next gate: the 10x Xenium Prime 5K mouse pan-tissue panel (5,006 genes; per-transcript nuclear flag). If it carries the
ribosomal / synaptic / lncRNA sets, it is the dataset for both the gene-filling application and the in-situ validation.

## Gate 2: Xenium Prime 5K mouse brain panel (2026-11-06) -- PASSES

Dataset: 10x "Xenium Prime 5K Fresh-Frozen Mouse Brain Coronal" (Xenium Onboard Analysis 3.0.0), public, bundle
`Xenium_Prime_Mouse_Brain_Coronal_FF_outs.zip` = 13.3 GB (2024-07-31) at
`https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_Prime_Mouse_Brain_Coronal_FF/`; 63 k cells, median 1,088
transcripts per cell; outputs include `transcripts` (per-molecule x, y, z, cell id, `overlaps_nucleus`), `cells`,
`cell_feature_matrix`, segmentation masks, morphology images. Panel file `xenium_prime5k_mouse_brain_panel_genes.txt`
(5,006 genes, design mAtlas_v1), per-gene delta of the overlap in `overlap_xenium_prime5k_delta.csv`.

| | Xenium 5K |
|---|---|
| genes in our 10,134-gene list | **2,508** (2,498 not: 219 MitoCarta, the rest Config B-filtered) |
| ribosomal (73) / nuclear lncRNA (45) / cytoplasmic markers (18) | 0 / 1 (Xist) / 4 (Ywhaz, Vamp2, Ywhag, Calm?) |
| SynGO genes in our list (1,346) | **616** |
| overlap genes with delta > +0.3 / < -0.3 / neutral (abs <= 0.1) | **387 / 23 / 1,343** |
| sd of delta on the overlap vs genome-wide | 0.25 vs 0.25 (representative) |

Top cytoplasm-enriched on the panel: Ywhaz 1.77, Vamp2 1.74, Ywhag 1.63, Ndrg4, Rab6a, Sncb, Atp6v1a, Rab3a, Sv2a;
nuclear-enriched: Xist -1.48, Ctnna3, Ahi1, Ube3a, Fgfr2, Kcnq1. Ribosomal genes are excluded from the 5K panel by
design, so the ribosome result is not testable in situ; the synaptic / vesicle program (SynGO 616) is, and that is the
biology the paper leads with.

Reading: the shared gene space is 2,508 genes with the full genome-wide spread of compartment bias, 387 cytoplasm-
enriched genes to score and 616 SynGO genes -- enough for the stratified gene-filling test AND for the per-molecule
in-situ nuclear-fraction validation (same dataset, `overlaps_nucleus`). This is the dataset to use.

Open items before building: (a) which cortical area the coronal section covers (bregma level; MOp is present from
about +2.5 to -1.0 mm, hippocampal-level sections are S1 / RSP territory) -- determine from the morphology image or
layer markers after download; cross-area cortex is acceptable (subclass-level types, same caveat as Bakken V1);
(b) no cell-type labels ship with the dataset -> label transfer from our MOp reference (and from the Allen WMB
taxonomy via the 10x cell clusters) is the first analysis step and doubles as readout 2; (c) 13.3 GB bundle must go
to scratch, not the project quota (4089 / 4096 GiB).

Status 2026-11-13: (c) done -- the bundle is at `/vast/palmer/scratch/gerstein/sz477/xenium_prime5k_mouse_brain/`
(zip 13,315,451,681 bytes = server size, downloaded 2026-11-06; palmer scratch purges after 60 days of no access, so
touch or re-download before late January 2027). Only the small files are extracted so far (`cell_feature_matrix.h5`,
`cells.parquet`, `cells.csv.gz`, `gene_panel.json`, `experiment.xenium`, `metrics_summary.csv`, `analysis/` =
10x clustering / PCA / UMAP / diffexp); `transcripts.parquet` (needed for `overlaps_nucleus`) and the morphology
images are still inside the zip. (a) and (b) not started.

## Scoping on the extracted data (2026-11-13)

Scripts `analysis/spatial/xenium_scoping.py` (expression matrix, 10x graph clusters, marker table),
`xenium_marker_map.py` (figure), `xenium_nuclear_fraction.py` (per-molecule aggregation, job 15979936, 2 min on
`scavenge`), `xenium_nuclear_fraction_figure.py`. Outputs in `scoping/` (`xenium_stats.json`, `cluster_markers.csv`,
`cell_obs_with_cluster.csv.gz`, `shared_genes.txt`, `gene_nuclear_fraction.csv`, `cell_nuclear_fraction.csv`,
`gene_nuclear_fraction_vs_gt_delta.csv`, `figures/`). Env note: `impute2` needs `LD_LIBRARY_PATH=$CONDA_PREFIX/lib`
on the login node for matplotlib / scipy (system libstdc++ too old); `pyarrow 20` was installed into `impute2`
(pyarrow >= 21 needs NumPy 2); scanpy is import-broken in `impute2` and `abc_env`, the scripts use h5py + scipy.

**Data:** 63,013 cells with >= 50 transcripts (median 1,091 transcripts, 656 genes per cell); 5,006 panel genes, 2,508
in our list, 2,214 of them detected in >= 5 % of cells; a cell carries a median 862 counts on the shared genes.
Per-molecule table: 188.5 M rows, overall nuclear fraction of assigned q20 molecules 0.557.

**(a) Section level / area** (`figures/fig_xenium_region_layer_markers.png`; flat left edge = midline, cortex wraps the
bottom and right edges, so y increases dorsally): the 10x clusters give a full caudoputamen (cluster 9, 2,909 cells,
Ppp1r1b), thalamus (cluster 7, 2,965) and a dentate gyrus + CA (cluster 17, 1,347, Prox1) -> hippocampal level, about
bregma -1.5 to -2.0 mm. MOp ends near -1.0 mm, so **this section has no motor cortex**; the cortical sheet is RSP /
S1 / AUD / TEa plus piriform. The Rorb (L4) band is visibly stronger along the lateral arc than dorsomedially, but
per-cell detection among Slc17a7+ cells is 0.36 (dorsomedial box) vs 0.31 (lateral arc), so marker-based area calls
are not reliable at 1 k transcripts per cell. Consequence: the reference use is subclass-level only (the Bakken V1
caveat), and the label transfer must be done at subclass level with the depth-robust type classifier, not by area.

**(b) Labels:** none shipped; the 10x graph clustering (27 clusters) separates non-neuronal classes (Oligo, Astro,
Micro, OPC, Endo, pericyte) and regions but lumps cortical excitatory neurons (clusters 2, 4, 10, 11; layer markers
average < 0.05 log CP100 per cluster). Label transfer at subclass level is the first build step, unchanged.

**In-situ validation of the compartment delta (new, `figures/fig_xenium_nuclear_fraction_vs_delta.png`):** for the
2,477 shared genes with >= 200 molecules, the in-situ nuclear fraction of a gene's molecules (`overlaps_nucleus`)
anti-correlates with our type-balanced ground-truth cell-minus-nucleus delta: Spearman -0.57 (p 1e-215), Pearson
-0.52; median nuclear fraction 0.68 for predicted nuclear-enriched genes (delta < -0.3, n 22), 0.63 neutral (n 1,317),
0.51 cytoplasm-enriched (delta > 0.3, n 387); AUROC of the delta for the low-nuclear-fraction quartile 0.90. The
extremes agree: Rab3a, Aldoa, Tpi1, Vamp2, Ywhaz (delta 1.0-1.8) have the lowest nuclear fractions (0.42-0.45); Cbx1,
Rsrp1, Neat1 the highest (0.86-0.92). This is an orthogonal, imaging-based confirmation of the delta the model learns,
independent of any dissociation protocol; it uses the ground-truth delta, so the next step is the same test with the
*model-predicted* delta per type (imputed minus input nuclei) and per cortical subclass once labels exist.

Next build steps (unchanged from the 2026-11-06 design, now with the section facts): (1) subclass label transfer with
`analysis/external/type_classifier.py` on the 2,508-gene space (check calibration on 1 k-transcript cells), (2) the
gene-filling test with the imputed-nuclei reference vs raw nuclei vs real cells, stratified by `delta_norm`, (3) the
per-type model-predicted delta vs per-type in-situ nuclear fraction.
