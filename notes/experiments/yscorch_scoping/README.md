# SCORCH data scoping (2026-10-23)

Advisor suggestion: apply ImputeVAE to "Y-SCORCH" mouse brain single-cell spatial data. What is on the group share:

| Location | Content | Relevant? |
|---|---|---|
| `/gpfs/gibbs/pi/gerstein/yscorch_data_share/seurat_combined_v3.RDS` (9.3 GB, mj648, 2026-06-22) | **Human** snRNA-seq Seurat v5 object: 654,918 nuclei x 35,574 genes, 88 samples / 33 donors, regions INS 234,779 / PFC 234,206 / VST 185,933, condition HIV 374,850 / CTR 280,068; reductions pca/harmony/tsne; **no spatial slot**; labels `consensus_annotation_v3` (37): Oligo 208k, L2/3 IT 79k, Astro 77k, OPC 43k, Micro 43k, L5 IT 26k, L4 IT 23k, Pvalb 19k, Vip 19k, Sst 18k, L6 IT 13k, Endo 10k, D1/D2 MSN subtypes (Shell/OT, Matrix, Striosome, NUDAP, ICj, Hybrid), Lamp5, L6 CT, L6b, L6 IT Car3, Chandelier, L5/6 NP, Sncg, L5 ET, Pax6, Sst Chodl, PC, VLMC, SMC, Immune, Macrophages. Median 2,869 UMI / 1,564 genes per nucleus. | Yes: nuclei-only human disease dataset = storyline application 1b/1d. Cortical labels match the training types; MSNs are an unseen-type test. |
| `/gpfs/gibbs/pi/gerstein/scorch/data/` (xy264, 2024) | earlier export of the same project: `data_sce_{pfc,ins,vst}.rds`, dense CSV count matrices (30 GB PFC), metadata with `cell_types_level0`, sex, race, PMI; `all_new_PFC/` ArchR snATAC project | superseded by the Seurat object |
| `/gpfs/gibbs/pi/gerstein/scorch/metadata/` | clinical_info.xlsx (113 subjects), updated_metadata.xlsx (282 rows: HIV status, CD4, ART, toxicology, opioid status) | covariates for a DE application |
| mouse brain single-cell **spatial** data | **not found** anywhere under `/gpfs/gibbs/pi/gerstein` (searched xenium / merfish / visium / spatial; only human Spatial_PEC2 Visium DLPFC) | ask the advisor where it lives |

Inspection script: `inspect_rds.R` / `inspect_rds.sh` (job 13783942; needs `module load R/4.4.1-foss-2022b R-bundle-Bioconductor/3.19-foss-2022b-R-4.4.1`, the `-bare` module has no Seurat).

## Why single-cell spatial would fit (if the mouse data exists)
Imaging-based spatial assays (Xenium / MERFISH / CosMx) segment whole cells in situ, so they are whole-cell,
dissociation-free, and assign each transcript to nucleus or cytoplasm. Two uses:
1. **In-situ validation of the compartment delta**: per gene x type nuclear fraction from transcript coordinates vs the
   model's predicted delta. No current validation is per-type, in-tissue, and dissociation-free.
2. **Application: snRNA-seq -> spatial integration.** Map spatial whole cells onto raw-nucleus vs imputed references;
   score label-transfer accuracy, per-type correlation, mapping confidence. If imputation improves mapping, that is the
   downstream use the paper lacks.
Risks: panel size (overlap with the 10,134 genes and especially with ribosomal / nuclear-lncRNA / SynGO sets), region
(striatal MSNs unseen in MOp training), segmentation bleed in dense tissue (use rank-based metrics within type), scale
(use the covariate model's explicit library target).

## What the human object enables now
- Nuclei-only disease application: impute PFC nuclei (cortical labels present) with mouse weights via the human-M1
  ortholog map (8,215 genes), run HIV-vs-CTR DE per type on raw vs imputed, ask whether cytoplasmic programs
  (ribosome, synapse, mitochondria) become detectable; compare with published whole-cell HIV brain signatures.
- Unseen-type behaviour on striatal MSNs (continuous-identity argument).
Caveats: cross-species zero-shot (no human paired truth), 10x multiome chemistry (use the quantile-map library mode with
a human whole-cell reference scale, or a label-free global ratio once implemented).

## Where the mouse spatial data could be (lookup 2026-11-06)

Advisor says SCORCH has mouse brain spatial transcriptomics. Not on the HPC: a name search over `/gpfs/gibbs/pi/gerstein`
and `/gpfs/gibbs/project/gerstein` (xenium / merfish / visium / spatial / cosmx / slide-seq / scorch) finds only the human
Spatial_PEC2 Visium/Xenium DLPFC project, unrelated Visium lymph-node and Slide-seq runs, `yc774/project/scorch_HIV`
(HIV-read detection in Y-SCORCH PFC fastqs) and `qiu.mouse.sc.data` (GEO GSE186069 / GSE228590, not spatial). The
Y-SCORCH analysis itself lives on the lab server, not the cluster: `scorch/data/readme.txt` copies
`cantor.yale.edu:/banach2/SCORCH/analysis/240604-combinedAnalysisAll/merged_data/` (Junchen) and `/data/chang/` (ATAC).
That tree is the first place to look for the Yale spatial data (ssh from the login node was not run from this session).

Consortium-level facts (web, 2026-11-06):
- SCORCH overview paper: Ament et al., Mol Psychiatry 2024, PMID 38879719, PMC11609103 (open access).
- Funded-project list (`scorch.igs.umaryland.edu/data-and-protocols.php`): the only project listing "spatial
  transcriptomics" is **Y-SCORCH (UM1DA051410, Yale/Spudich): human PFC, ventral striatum, insular cortex; snRNA-seq,
  scATAC-seq, spatial transcriptomics** -- human, not mouse. Mouse projects: Mount Sinai/Kenny (U01DA053629, EcoHIV mice,
  snRNA-seq) and **Allen Institute (U01DA056003, Zeng / Kenny / Volsky; NIH RePORTER 10454708)**: EcoHIV + intravenous
  cocaine self-administration mice, 10x Multiome (RNA + ATAC) **and MERFISH** "to confirm and validate snRNA-seq and
  Multiome findings and generate spatially resolved maps of cells impacted by cocaine and/or HIV". That is the mouse
  brain spatial dataset the advisor most likely means. No preprint found yet (searched 2025-2026).
- Data route: public SCORCH data at the NeMO SCORCH portal `https://scorch-portal.nemoarchive.org` (JavaScript portal,
  browse by hand; filter by grant U01DA056003 / modality MERFISH), restricted data via dbGaP; NeMO main portal
  `https://portal.nemoarchive.org`. Whether the Allen MERFISH release is already public is not visible without the portal.

What to ask the advisor: (a) does "SCORCH mouse spatial" mean the Allen EcoHIV/cocaine MERFISH (consortium data, NeMO)
or Yale's own spatial run (which the project list says is human); (b) if Allen: is there a consortium data-share path
before public release (the Yale site is a consortium member); (c) if Yale: where on cantor/banach2 it sits and which
platform (Visium = spots, no per-cell nucleus/cytoplasm; Xenium/MERFISH = segmented cells, usable for the in-situ delta).
Fit for us (both cases): imaging-based (MERFISH/Xenium) = in-situ nuclear-vs-cytoplasmic validation + snRNA-to-spatial
mapping application; Visium = pseudobulk-level deconvolution test only. MERFISH panel size (typically 500-1,000 genes)
limits the overlap with our 10,134-gene list; check ribosomal / nuclear-lncRNA / SynGO coverage first.

## Region match (2026-11-06)

Our paired training data are AIBS primary motor cortex (MOp). The Allen SCORCH project (RePORTER abstract) profiles
**prefrontal cortex, corpus striatum and habenula** in EcoHIV + cocaine mice -- no motor cortex. PFC is the only
cortical region, so a SCORCH test would be a cross-area transfer at the subclass level (our 17 types exist in PFC;
PFC has no L4), the same situation as the Bakken 2018 V1 hold-out (delta r 0.58, ceiling 0.92). Striatum and habenula
are unseen types (MSNs, habenular neurons) = the N-1 identity-collapse regime.

Public mouse spatial datasets that DO cover MOp, from the same Zeng lab or vendor:
| dataset | platform / panel | cells | MOp | per-transcript compartment | cell-type labels |
|---|---|---|---|---|---|
| Allen ABC Atlas MERFISH-C57BL6J-638850 (Yao 2023) | MERFISH, 500 genes | 4.0 M, 59 coronal sections, CCF-registered | yes (parcellation) | cell-level matrices only on ABC (molecule data not confirmed) | yes, WMB taxonomy |
| Zhuang ABCA-1..4 (Zhang 2023 Nature) | MERFISH, 1,122 genes | 9 M, 147 + 66 coronal, 26 sagittal sections, CCF-registered | yes | ABC tutorial mentions molecule coordinates (verify) | yes, mapped to WMB taxonomy |
| 10x Xenium Prime 5K fresh-frozen mouse brain hemisphere | Xenium, 5,006-gene pan-tissue panel | 63 k cells, 1 coronal hemisphere | depends on section level (check anatomy) | yes: transcripts table with `overlaps_nucleus` flag | none (label-transfer from our MOp reference) |
For the in-situ nucleus-vs-cytoplasm validation the Xenium 5K set is the best fit (about half of our 10,134 genes on
the panel, per-molecule nuclear flag, segmented whole cells); the MERFISH atlases give cell-type-resolved whole-cell
profiles for the snRNA-to-spatial mapping application with a 500 / 1,122-gene overlap. None of this needs SCORCH access.

## Portal access (2026-11-13)

Access granted to the private SCORCH portal `https://private-scorch-portal.nemoarchive.org/`. First look: filter by
grant U01DA056003 (Allen / Zeng) and modality MERFISH; check (1) whether the MERFISH release carries per-molecule
tables with a nuclear flag or only cell x gene matrices, (2) panel gene list vs our 10,134 genes (ribosomal / SynGO /
nuclear-lncRNA coverage, as for the Xenium gate), (3) regions (PFC / striatum / habenula expected, no MOp), (4) whether
the matching 10x Multiome snRNA-seq of the same animals is in the same release (nuclei input for the model), (5) cell
type labels and taxonomy. Record the answers here before downloading anything to the project quota.

Result (2026-11-13, portal browsed by the user): the Technique facet over all 17 studies (64,198 files, 17,098 samples,
330 TB) lists only sequencing assays (10x Multiome RNA/ATAC, Chromium 3' v2-v4, cell hashing, Flex, immune profiling,
ATAC-seq, Nanopore, PacBio). **No MERFISH / Visium / Xenium / spatial files are deposited anywhere in the portal**, so
neither the Allen mouse MERFISH (U01DA056003_zeng: 7,982 files = 4,630 Multiome RNA + 1,952 Multiome ATAC + 1,268
Chromium v4 + 132 Flex; regions prefrontal (infralimbic, prelimbic), cortical subplate, amygdala, dorsal striatum) nor
the Yale Y-SCORCH spatial run (UM1DA051410_spudich, 5,969 files) is available there. Spatial, if it exists, must be
obtained directly from the Allen / Yale groups. What the portal does offer us: mouse PFC snRNA-seq with EcoHIV x
cocaine conditions and processed per-sample outputs (`gex_sparse_counts.npz`, cell barcodes, cell/gene metrics,
CellBender `posterior.h5`) = the mouse nuclei-only disease application, cross-area at subclass level (no MOp).

## Portal survey brief (2026-11-13)

A local browser agent is surveying the private portal with a five-task brief (whole-cell libraries under Specimen
Type = cells, 8,904 files, incl. "microglial cell" 640; human Brodmann area 6 nuclei, 2,749 files; any bulk RNA-seq;
donor matching / manifest export; the Zeng mouse study's regions and processed file types). The brief text is in the
chat record of 2026-11-13; paste the agent's report into `portal_survey_2026-11-13.md` in this directory.
