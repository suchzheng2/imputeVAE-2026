# Task: External nucleus-to-whole-cell validation with compartment fractionation and TRAP/RiboTag data

**Created:** 2026-09-17
**Status:** Complete (2026-09-18). All phases run with 1,000-iteration final statistics. Results in README.md and outputs/report/.
**Location:** `validation/trap_ribotag_external_validation/`
**Origin:** Codex draft written off-cluster, reviewed and revised by Claude against the actual repo on 2026-09-17

---

## 0. Revision notes (what changed from the Codex draft, and why)

Read this section first. Each item below was verified against the repo or GEO on 2026-09-17.

1. **All draft input paths were wrong for this cluster.** `data/external_validation/`, `scripts/`, `outputs/`, `external_validation_dataset_exploration.md`, `prashant_followup_next_steps_plan.md` and the `/Users/suzy/...` paths do not exist here. Section 3 lists the real repo inputs.
2. **GSE210071 is not an untreated compartment dataset.** Every cytosol/nucleus/whole-cell sample (GSM6416109 to GSM6416126) was treated with triptolide for 3 h, a Pol II transcription-initiation inhibitor, before "unstimulated" or "bicuculline 1 h". The DRB series (GSM6416127 to GSM6416135, GPL24247) has no compartment split. There is no inhibitor-free compartment sample in this series. The draft called the unstimulated samples "control"; they are "triptolide + unstimulated". This must be stated as the leading caveat for H1 (nascent nuclear RNA is depleted, so nuclear-vs-cytosol ratios of short-lived nuclear transcripts are distorted).
3. **Pre-registered nuclear controls corrected.** In this project's own ground truth (`notes/experiments/nuclear_lncrna_analysis/all_genes_delta.csv`), `Meg3` has a positive global delta (+0.84; nuclear only in glia, cytoplasmic in neurons, see `notes/experiments/fractionation_validation/README.md`) and `Snhg11` is near zero (-0.05). `Kcnq1ot1` was also previously found to have the wrong expected sign. `Neat1` is not in the 10,134-gene model set. New controls: `Malat1` and `Xist` as global nuclear controls; the 45-gene data-driven nuclear set (`notes/experiments/nuclear_lncrna_analysis/proposed_nuclear_lncrna_list.txt`) as the set-level nuclear control; `Meg3` pre-registered as a cell-type-dependent case (expect negative in Astro/Oligo, positive in neurons); `Snhg11` and `Neat1` reported as "not a control / absent".
4. **This repo already has TRAP and fractionation results that the new work must build on, not repeat.** `notes/experiments/trap_validation/README.md` (GSE141464 IP/Input, GSE77243 RiboTag IP-only) and `notes/experiments/fractionation_validation/README.md` (GSE73391 Buckley, GSE110727 Moutinho) established two facts that change how the new datasets are framed:
   - TRAP **IP/Input enrichment** is anti-correlated with the cell-minus-nucleus delta (rho about -0.12 to -0.15 for both the ground-truth delta and ImputeVAE). It measures translational engagement, not compartment. Any IP-vs-input or IP-vs-unbound contrast (GSE13379, GSE174303, GSE121199 RiboTag-vs-whole-cell) is expected to be weak or negative and is a property-mismatch check, not a validation failure.
   - TRAP **IP-only abundance** within a cell type correlates strongly with that cell type's delta (rho about 0.80 for PV, SST, VIP in GSE77243). GSE263082 is the cross-lab replication of exactly that design and is the priority dataset.
   - Every contrast must report the **ground-truth delta** next to the ImputeVAE delta so the reader can separate "model recovers the MOp delta" from "the MOp delta relates to the external measurement".
5. **Delta units.** Project deltas are differences of mean log1p(counts) per gene (nuclei-level or cell-type pseudobulk), not log2 fold changes. External contrasts are log2 ratios of CPM or FPKM. Rank statistics (Spearman) are the primary comparison; slopes are reported but are not unit-comparable and must not be interpreted as calibration.
6. **Per-cell-type predicted deltas must be built, not located.** The draft's "full_delta_matrix.csv" was made on the Mac. On the cluster the per-CT ImputeVAE delta and the nucleus/imputed pseudobulks are not saved anywhere (the compartment classification script computes them in memory only). Script `01` in Section 8 builds them from `run/cvae_full_eval/imputed_cell_from_nucleus.npy`.
7. **GSE13379 raw tar removed from the download list.** `GSE13379_RAW.tar` is 672 MB of CEL files. Use the series matrix (GCRMA values, 15.6 MB) plus the GPL1261 annotation. Every other required file is under 7 MB.
8. **GSE263082 design spelled out from GEO** (27 GSMs; GSM8185308 does not exist; Batch5 count file has no GSM linked to it). See Section 6C.
9. **GSE121199 design spelled out from GEO.** Whole-cell FACS and nuclear FACS are male only (D1: 8 WC, 6 nuclear; D2: 8 WC, 5 nuclear by sample titles; the GEO summary says 9/7 for WC, so reconcile against matrix columns). RiboTag has 6 male + 5 female per subtype. Medium spiny neurons have **no matched cell type** in the MOp model; Section 6B specifies which pooled delta to use.
10. **Tooling.** No R on PATH; R 4.3.2 with Bioconductor 3.18 exists as a module but Lmod refuses to load it while the auto-loaded `miniconda` module is present. `pydeseq2`, `GEOparse`, `gseapy`, `mygene`, `adjustText` are not in `impute2`. Section 9 gives the R-first path with a Python fallback. Do not pip-install into `impute2`.
11. **Repo conventions applied.** Scripts, data, and outputs live under this validation folder (as in `validation/human_m1_transfer/` and `validation/gse219278_sorted_population_validation/`). SLURM jobs use `--partition=scavenge`, `conda activate impute2`, logs to `jobs/logs/slurm-%j_<name>.{out,err}`. A summary README goes to `notes/experiments/trap_ribotag_external/README.md` and one row to the table in `notes/experiments/README.md`. Memory files must be written to both `~/.claude/projects/.../memory/` and `.memory/`.
12. **Downloads must run on the login node.** Internet access from compute nodes is not guaranteed; the login node reaches `ftp.ncbi.nlm.nih.gov` (verified). BioMart queries likewise.
13. **Starting benchmarks (Section 13) come from the Mac exploration** and cannot be reproduced here. They remain sanity checks only.

---

## 1. Role and scope

You are working as a computational biology research agent in:

```
/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE
```

Implement a rigorous, reproducible external validation of ImputeVAE's predicted transformation from single-nucleus to whole-cell RNA expression, using biochemical compartment fractionation and ribosome-associated (TRAP/RiboTag) datasets that the model never saw. Complete the work end to end: audit and download data, construct metadata, preprocess, run statistics, make figures, write a report.

Do not stop after proposing code. Run the analyses on SLURM, inspect outputs, debug failures, and report what the data support. Preserve existing files. Do not overwrite anything outside this validation folder except the two notes files and the memory files named in Section 12.

Read before writing code:

- `CLAUDE.md` (session sync rules, conventions)
- `notes/experiments/trap_validation/README.md` and `analysis/trap_crossvalidation.py`
- `notes/experiments/fractionation_validation/README.md` and `analysis/fractionation_validation.py`
- `compartment_classification/README.md`
- `validation/syngo/scripts/syngo_brain_validation.py` (`load_data` shows how to load raw nuclei, the split-aware imputed matrix, and compute per-CT pseudobulks)
- `analysis/export_cvae_imputed.py` (how the full 40,166-nucleus imputed matrix was produced)
- `validation/gse219278_sorted_population_validation/outputs/reports/gse219278_validation_report.md` (report style)

---

## 2. Scientific background

ImputeVAE (single-stage conditional VAE, `run/run_2026-07-18-19-40-51_570984`) encodes a nucleus with the nucleus flag and decodes with the cell flag to produce an imputed whole-cell profile. The central quantity is

```
predicted_delta(gene, cell_type) = mean_log1p(imputed_whole_cell) - mean_log1p(observed_nucleus)
ground_truth_delta(gene, cell_type) = mean_log1p(real_whole_cell) - mean_log1p(observed_nucleus)
```

Positive delta: predicted more abundant in whole-cell RNA than in nuclear RNA. Negative: predicted nuclear-enriched.

The qualifying-exam committee asked for validation using data that independently distinguish nuclear and cytoplasmic/whole-cell RNA, including spatial/FISH data and ribosome-associated profiling. This task addresses the compartment and ribosome-associated parts. Do not describe RiboTag/TRAP as literal cytoplasmic ground truth: it measures ribosome-associated RNA, a related but distinct contrast. MERFISH/FISH validation is a separate future task.

What is already established in this repo (do not re-derive; cite):

| Prior result | Where | Number |
|---|---|---|
| Compartment classification vs MOp ground truth | `compartment_classification/` | AUROC 0.991 |
| Fractionation: Buckley 2016 mouse brain CN ratio vs delta | `compartment_classification/fractionation_validation/` | rho 0.44 |
| Fractionation: Moutinho 2021 human FC | same | rho 0.40 |
| TRAP IP-only abundance vs per-CT delta (GSE77243) | `compartment_classification/trap_validation/` | rho 0.80 PV, 0.81 SST, 0.79 VIP |
| TRAP IP/Input enrichment vs delta (GSE141464) | same | rho -0.15 (property mismatch, GT delta also negative) |

---

## 3. Existing project inputs (cluster paths)

| Input | Path | Notes |
|---|---|---|
| Model checkpoint and args | `run/run_2026-07-18-19-40-51_570984/{models/model.pt,arguments.json}` | Not needed unless re-running inference |
| Imputed matrix, all 40,166 nuclei | `run/cvae_full_eval/imputed_cell_from_nucleus.npy` (40166 x 10134, float32, log1p scale, 1.6 GB) | Row order = filtered nuclei order in `analysis/export_cvae_imputed.py` |
| Nucleus labels for that matrix | `run/cvae_full_eval/nucleus_obs.csv` | column `cell_type` |
| Imputed matrix, validation split only | `run/eval_v3_splitaware/cvae_imputed_valid.npy` + `valid_nuc_obs.csv` (8,033 nuclei) | Used by SynGO/TF validations; use as sensitivity |
| Raw nuclei counts | `/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/Cytoplasm_imputation/scvi/10X_nuclei_v2_v3_AIBS/data.h5ad` | filter `meta_batch_id` endswith `n3A`, gene list, log1p; follow `export_cvae_imputed.py` |
| Raw cells counts (for GT delta) | `.../10X_cells_v2_v3_AIBS/data.h5ad`, suffix `c3A` | Only needed if rebuilding GT per-CT pseudobulk |
| Gene list (10,134) | `data/training_gene_list_v3.csv` (no header) | |
| Common cell types (17) | `data/common_cell_types.csv` | Astro, Endo, L2/3 IT, L5 ET, L5 IT, L5/6 NP, L6 CT, L6 IT, L6b, Lamp5, Macrophage, OPC, Oligo, Pvalb, Sncg, Sst, Vip |
| Ground-truth delta, global | `compartment_classification/gt_delta_global.csv` | |
| Ground-truth delta, per CT | `compartment_classification/gt_delta_per_ct.csv` | genes x 17 CTs |
| Predicted global delta, 8 methods | `compartment_classification/predicted_deltas_global.csv` | columns Raw_nuclei, ImputeVAE, CT-mean, Linear_reg, scVI, scANVI, scGen, scButterfly |
| Global delta with lncRNA flags | `notes/experiments/nuclear_lncrna_analysis/all_genes_delta.csv` | columns gene, cell_mean, nuc_mean, delta (this is the ground-truth delta) |
| 45-gene nuclear set | `notes/experiments/nuclear_lncrna_analysis/proposed_nuclear_lncrna_list.txt` | |
| Per-CT delta tables (20% dec-only model, older) | `notes/experiments/delta_analysis/tables/<CT>_delta.csv` | Do NOT use for the primary analysis; wrong model |
| Ensembl mouse ID to symbol | `data/external/fractionation/ensembl_mouse_id2name.csv` | reuse for GSE121199 |
| Mouse-human orthologs | `data/external/mouse_human_orthologs.csv` | not needed (all datasets are mouse) |
| Prior TRAP data | `data/external/furlanis2019/` (GSE141464 counts, GSE77243 xlsx) | for cross-lab comparison with GSE263082 |
| pybiomart cache | `.pybiomart.sqlite` (repo root) | BioMart only from login node |

**Model pseudobulk matrices for H2 must be built (script 01):** per-CT mean log1p for observed nuclei and imputed cells over the same nuclei, for (a) all 40,166 nuclei (primary; more cells for rare types) and (b) the 8,033 validation-split nuclei (sensitivity, matches SynGO/TF convention). Verify row alignment by checking that `nucleus_obs.csv` cell-type labels match the filtered h5ad in order, and that the matrix shape is (40166, 10134). Save the sources and aggregation rule in the output README.

---

## 4. Hypotheses

### H1: Direct compartment concordance (GSE210071 primary; GSE121199 whole-cell vs nuclear)

Genes with positive predicted delta should have positive observed cytosol-minus-nucleus or whole-cell-minus-nucleus log ratios; negative predicted delta should be nuclear-enriched externally.

Primary tests: Spearman correlation with bootstrap CI; robust regression slope with CI (reported, not interpreted as calibration); AUROC of predicted delta for classifying externally cytosol/whole-cell-enriched vs nuclear-enriched genes; enrichment of top predicted-positive genes among externally cytosol-enriched genes and of top predicted-negative among nuclear-enriched. Report ground-truth delta and CT-mean baseline on every contrast.

### H2: Profile improvement after imputation (GSE263082 Pvalb and Sst; secondary GSE121199 pooled neurons)

For matched cell types the imputed profile should resemble the external RiboTag profile more closely than the observed nuclear profile does:

```
delta_similarity = cor(imputed_pseudobulk, external_profile) - cor(nucleus_pseudobulk, external_profile)
```

Paired gene bootstrap (stability across genes) and external replicate bootstrap (biological uncertainty), labeled separately. Spearman primary, Pearson secondary. Also report cor(real_cell_pseudobulk, external) as the ceiling.

### H3: Cell-type-specific translational support (GSE263082)

In Pvalb and Sst separately, genes with large positive predicted delta should be over-represented among highly expressed RiboTag genes, and RiboTag abundance rank should correlate with predicted delta (replicating the GSE77243 design). Supporting evidence, not a compartment test. Do not merge Pvalb and Sst before estimating effects; pool only in a final meta-summary if heterogeneity permits.

### H4: Nuclear noncoding RNA controls

Pre-registered: `Malat1`, `Xist` (global nuclear controls); the 45-gene nuclear set (set-level control); `Meg3` (cell-type-dependent: negative in Astro/Oligo, positive in neurons; in mixed cultures and interneuron RiboTag expect the neuronal sign). Report `Snhg11` (project delta near zero) and `Neat1` (absent from model gene set) as non-controls with explicit missingness. Do not substitute datasets or genes when one is absent.

### H5: Robustness to abundance and gene class

Concordance should survive abundance-matched permutation nulls and sensitivity exclusions: low-expression genes, mitochondrial (`mt-`), ribosomal protein (`Rpl*`/`Rps*`, not `Rps6k*`), immediate-early/dissociation module (list in `notes/experiments/dissociation_audit/`), and the nuclear ncRNA class from `compartment_classification/gene_classes.csv`.

---

## 5. Phase 1: Data acquisition and integrity audit

Data dir: `validation/trap_ribotag_external_validation/data/<GSE>/`. Run downloads on the **login node**. Atomic downloads (temp name, `gzip -t`, SHA-256, then move). Record size, checksum, URL, date, and pre-existing vs new in `outputs/data_manifest.tsv`. Do not download SRA/FASTQ. Do not download `GSE13379_RAW.tar`.

| Dataset | File | URL (ftp.ncbi.nlm.nih.gov/geo/...) | Size |
|---|---|---|---|
| GSE210071 | `GSE210071_FPKM_with_annotations.txt.gz` | `series/GSE210nnn/GSE210071/suppl/` | 6.5 MB |
| GSE210071 | `GSE210071-GPL17021_series_matrix.txt.gz`, `GSE210071-GPL24247_series_matrix.txt.gz` | `series/GSE210nnn/GSE210071/matrix/` | small |
| GSE121199 | `GSE121199_all_counts.txt.gz` | `series/GSE121nnn/GSE121199/suppl/` | 2.5 MB |
| GSE121199 | `GSE121199_series_matrix.txt.gz` | `series/GSE121nnn/GSE121199/matrix/` | small |
| GSE263082 | `GSE263082_NS046_raw-counts_gt40.txt.gz`, `GSE263082_Max_NS68_raw.txt.gz`, `GSE263082_Batch3_RawGeneCounts.txt.gz`, `GSE263082_Batch4_RawGeneCounts.txt.gz`, `GSE263082_Batch5_RawGeneCounts.txt.gz` | `series/GSE263nnn/GSE263082/suppl/` | < 1 MB each |
| GSE263082 | `GSE263082_series_matrix.txt.gz` | `series/GSE263nnn/GSE263082/matrix/` | small |
| GSE174303 | `GSE174303_counts_matrix.txt.gz` | `series/GSE174nnn/GSE174303/suppl/` | 1.2 MB |
| GSE174303 | `GSE174303_series_matrix.txt.gz` | `series/GSE174nnn/GSE174303/matrix/` | small |
| GSE13379 | `GSE13379_series_matrix.txt.gz` | `series/GSE13nnn/GSE13379/matrix/` | 15.6 MB |
| GSE13379 | `GPL1261.annot.gz` | `platforms/GPL1nnn/GPL1261/annot/` | for probe-to-symbol |

GEO pages: `https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=<GSE>`. Text-format sample records: append `&targ=self&form=text&view=brief` to a GSM query; this is the reliable way to build metadata programmatically without GEOparse.

---

## 6. Phase 2: Sample metadata (verified designs)

One tidy table per dataset plus `metadata/all_samples.tsv`. Required fields: GSM, expression column name, dataset, species, tissue/region, cell type, assay/compartment, genotype, treatment, sex, replicate, batch/source file, platform, include_primary (bool), exclusion_reason. Derive labels from GEO sample records, never from column order alone. Fail loudly if a selected sample has no matrix column or a column has no metadata.

### 6A. GSE210071 (Mazille et al. 2022, cultured mouse primary cortical cells; verify citation)

Verified from the downloaded files (2026-09-17). 27 samples, one FPKM table with 59,489 rows (FASTDB IDs, 23,504 unique symbols, many unannotated loci). GPL17021 compartment series (GSM6416109 to GSM6416126) has matrix columns `S{1,2,4}_{N,C,W}_{Ctl,Bic}`: culture IDs S1/S2/S4 correspond to rep1/rep2/rep3 and **pair the three compartments within a culture**, so a paired design is supported. **All 18 compartment samples received triptolide 3 h.**

| Compartment | Triptolide + unstimulated (`Ctl`) | Triptolide + bicuculline 1 h (`Bic`) |
|---|---|---|
| Cytosol (`C`) | GSM6416109-111 | GSM6416118-120 |
| Nucleus (`N`) | GSM6416112-114 | GSM6416121-123 |
| Whole Cell Extract (`W`) | GSM6416115-117 | GSM6416124-126 |

GPL24247 DRB series (GSM6416127 to GSM6416135) = columns `Ctrl_/BicS_/BDNF_ x N7/N8/N10`, no compartment split; excluded from compartment contrasts. FPKM, not counts; nonzero FPKM quantiles 5% = 0.005, 25% = 0.03, median = 0.27.

Gene presence by symbol: `Malat1`, `Neat1`, `Kcnq1ot1` **absent**; `Xist`, `Meg3`, `Snhg11`, `Snap25`, `Camk2n1`, `Tmsb4x`, `Sst`, `Actb` present. Unannotated FASTDB entries overlap the Malat1/Neat1 loci (e.g. GSMG0065070 chr19:5801982-5835942, N 8.6 vs C 2.2 FPKM); report as a locus-level footnote only, never as Malat1. Caveat text: embryonic mixed culture, transcription blocked for 3 h before fractionation.

### 6B. GSE121199 (Kronman et al. 2019, nucleus accumbens D1/D2 MSNs; verify citation)

Verified. 49 GSMs, 49 matrix columns named `{wc,nuclear,ribo}_{D1,D2}_{M,F}_<rep>`; Ensembl gene IDs (52,637 rows), raw counts. Whole-cell and nuclear samples are male only. Column-level resolution of the GEO title/summary discrepancy:

- `wc_D2_M_3_L8` is a technical re-sequencing of `wc_D2_M_3` (Spearman 0.995, identical marker profile): **sum counts** into one D2 sample. D2 whole-cell n = 6.
- `wc_D1_M_1_2` is D1 by markers (Drd1/Tac1/Pdyn high, Drd2/Penk/Adora2a low) and correlates 0.983 with `wc_D1_M_1`: treated as a possible second library of D1 rep 1, **excluded from primary**, used in sensitivity. D1 whole-cell n = 8.
- GEO titles (D1 rep1-8, D2 rep1-8) are therefore not a 1:1 map of the whole-cell block; GSMs are assigned in series order within block and the two extra columns are flagged in the metadata.
- Nuclear: D1 x6, D2 x5 (`nuclear_D2_M_{1,2,3,5,7}`). RiboTag: D1 6M + 5F, D2 6M + 5F; female RiboTag excluded from primary.

Primary: male only; WC vs nuclear within D1 and within D2 with subtype in the design matrix. RiboTag vs nuclear as assay triangulation. RiboTag vs WC as an enrichment-type contrast (expected weak/negative per Section 0.4).

**No matched MOp cell type.** MSNs are GABAergic projection neurons. Compare against (i) the neuron-pooled predicted delta (mean over the 12 neuronal CTs), primary, and (ii) the inhibitory-pooled delta (Pvalb, Sst, Vip, Lamp5, Sncg), sensitivity. Say explicitly that no cell-type-matched comparison exists.

### 6C. GSE263082 (Pvalb-Cre and Sst-Cre RiboTag, cortex + hippocampus pooled, P60)

Verified: **34 GSMs** (GSM8185307, 8185309 to 8185341; GSM8185308 does not exist), not the 27 stated in the first draft. `!Sample_description` gives the matrix column ID and the source file. IP-only (no input), raw counts.

| Batch / file | Rows | Samples (column IDs) |
|---|---|---|
| Batch1 `NS046_raw-counts_gt40` | 24,371 symbols | PV saline rep1,3,4 (NS46_S1,S5,S6); PV PTZ x4 (S2,S4,S7,S8) |
| Batch2 `Max_NS68_raw` | 11,808 symbols (pre-filtered) | SST saline x4 (NS68_M1-M4); SST PTZ x3 (M5,M6,M8); PV saline x2 (M9,M10) |
| Batch3 `Batch3_RawGeneCounts` | 53,465 Ensembl + 'Symbol | PV CFC x3 (1,3,6); PV homecage x3 (2,4,5); PV Nr4a1-cKO homecage x3 (7,9,12) |
| Batch4 `Batch4_RawGeneCounts` | 53,465 | SST homecage x3 (1,4,6); SST CFC x3 (2,3,5) |
| Batch5 `Batch5_RawGeneCounts` | 53,465 | SST Nr4a1-cKO homecage x3 (91,92,93) |

Primary baseline subset (wild type, unstimulated): **Pvalb** = Batch1 saline (3) + Batch2 saline (2) + Batch3 homecage (3), batch-stratified; **Sst** = Batch2 saline (4) + Batch4 homecage (3), batch-stratified. Exclude PTZ, CFC, and both Nr4a1-cKO groups from primary; PTZ/CFC serve as sensitivity ("does concordance survive stimulation"). Batch2 is pre-filtered to 11,808 genes and Batch1 to genes with >40 counts, so the gene universe must be intersected per contrast and reported. Never concatenate normalized matrices across batches and ignore batch; estimate within batch, then meta-summarize.

### 6D. GSE174303 (Hale et al. 2021, CA1 pyramidal TRAP, cell body vs neuropil; verify citation)

Verified: 48 GSMs, 48 matrix columns. GEO titles use `{CB,NP}_{CreNeg,KO,WT}_{RNAseq,TRAP}_R{2-5}`; the matrix uses `ML` for NP and `input` for RNAseq. Gene symbols plus Entrez IDs, 24,100 rows, raw counts. Primary: WT only, TRAP vs input within CB and within ML, replicates R2-R5 as blocking factor. Enrichment-type contrast; frame per Section 0.4. The CB-vs-ML difference in TRAP enrichment is the exploratory piece (dendritic mRNAs).

### 6E. GSE13379 (Doyle et al. 2008 bacTRAP atlas, Affymetrix GPL1261)

Verified: 107 samples, 45,103 probesets, GCRMA values on **linear scale** (log2 before use), normalized within replicate groups. Cortex populations with both TRAP and unbound: aldh1l1 astrocytes (lines JD130, JD133), cck (GM391), cmtm5 oligodendrocytes (JD307), etv1 L5a (TS88), glt25d2 L5b (DU9), ntsr1 L6 (TS16), olig2 (JD97), pnoc (GM64); cort (GM130) has TRAP only. Each has 3 TRAP and 1 unbound (astrocytes 2+2), so IP-vs-unbound has n = 1 unbound per population and cannot support replicate-based inference; use TRAP abundance (3 reps) per population for the abundance-vs-delta test and IP/UB only descriptively. Lowest priority; may be deferred, and say so in the report.

---

## 7. Phase 3: Gene identifier harmonization

Primary identifier: current mouse gene symbol; keep original IDs. Strip Ensembl version suffixes. Map Ensembl IDs with `data/external/fractionation/ensembl_mouse_id2name.csv` first; if coverage is poor, query BioMart via `pybiomart` on the login node and save the mapping under `outputs/gene_mapping/`. GSE263082 files that embed Ensembl-symbol pairs can serve as a second local map; validate agreement. Sum raw counts across duplicate symbols before normalization; for FPKM or microarray, take the median across duplicate rows/probes and document it. GSE13379 probes map via `GPL1261.annot.gz`; never analyze probe IDs as genes. Save an audit (input rows, mapped, unmapped, duplicated, final unique, overlap with the 10,134 model genes) per dataset.

---

## 8. Scripts and outputs

```
validation/trap_ribotag_external_validation/
  TASK.md                      (this file)
  README.md                    (status, results summary, file index; written at the end)
  data/<GSE>/                  (downloaded files, never modified)
  scripts/
    00_audit_and_download.py   (login node)
    01_build_model_pseudobulk.py   (SLURM jobs/01_*.sh, 96G)
    02_build_metadata.py
    03_harmonize_genes.py
    04_preprocess_gse210071.py
    05_preprocess_gse121199.R      (edgeR TMM + limma-voom; SLURM jobs/05_06_*.sh)
    06_preprocess_gse263082.R
    07_primary_concordance.py      (H1, H4, H5 on GSE210071 and GSE121199)
    08_profile_improvement.py      (H2, H3 on GSE263082 and GSE77243)
    09a_preprocess_gse174303.R
    09_secondary_datasets.py       (GSE174303, GSE13379)
    10_make_figures.py
    11_summarize_results.py
    extval_utils.py                (Python TMM/OLS fallback helpers)
  jobs/                        (sbatch wrappers: --partition=scavenge; logs in <project>/jobs/logs/)
  outputs/
    data_manifest.tsv
    metadata/  gene_mapping/  harmonized/  qc/  normalized/  model_pseudobulk/  tables/  figures/  logs/  report/
```

Every script runs from the project root without manual editing. Every figure has a plotted-data CSV. Every table carries enough identifiers to reproduce it. Record package versions and seeds in `outputs/logs/session_info.txt`.

---

## 9. Statistical framework and tooling

**Normalization.** Tier A (preferred): edgeR TMM + limma-voom or edgeR quasi-likelihood for count contrasts. Load with `module purge && module load R-bundle-Bioconductor/3.18-foss-2022b-R-4.3.2` inside the SLURM script; miniconda must not be loaded in the same shell (Lmod refuses otherwise). Run a 5-minute smoke job first that prints `requireNamespace` for edgeR, limma, DESeq2. Tier B (fallback if the module cannot be made to work within one attempt): Python in `impute2` with a hand-written TMM (document it), log-CPM with prior count 1 via `numpy`, and replicate-level moderated t via `statsmodels` OLS per gene. Do not pip-install into `impute2`; if a separate env is needed (`pydeseq2`), create `extval` and record it. FPKM (GSE210071): `log2(FPKM + eps)` with eps chosen from the nonzero distribution and stated. Microarray (GSE13379): use supplied GCRMA values, check log scale.

**Per contrast report:** n model genes, n external genes, n overlapping tested; Spearman and Pearson with 95% bootstrap CI; robust slope with CI; sign agreement on genes with reliable nonzero external effect (|log2FC| > 0.5 and FDR < 0.1) and separately on high-confidence predicted shifts (|delta| > 0.25); AUROC for predicted-positive and predicted-negative sets; top 20 concordant and discordant genes; replicate-bootstrap and gene-bootstrap stability; abundance-matched permutation p; before/after each sensitivity filter. Always include columns for ground-truth delta and CT-mean alongside ImputeVAE.

**Abundance-matched null.** Bin genes (10 deciles) by observed nuclear pseudobulk expression in the matched MOp cell type (or the pooled nucleus for MSN/culture), permute within bins, 1,000 iterations for final numbers (200 during development), seed 42.

**Multiple testing.** Benjamini-Hochberg within each hypothesis family; mark primary vs exploratory tests in every table.

**Pseudo-replication.** Biological samples define uncertainty for external contrasts; gene bootstrap addresses gene-level stability. Label both; never substitute one for the other.

---

## 10. Showcase gene panel

Pre-registered: nuclear `Malat1`, `Xist`; cell-type-dependent `Meg3`; neuronal cytoplasmic `Snap25`, `Camk2n1`, `Tmsb4x`; markers `Pvalb`, `Sst`; reported-only `Snhg11`, `Neat1` (absent), `Kcnq1ot1`. Project global deltas for reference: Xist -1.21, Malat1 -0.83, Snhg11 -0.05, Sst +0.19, Meg3 +0.84, Snap25 +1.88, Camk2n1 +3.77, Tmsb4x +4.28.

For each dataset, a tidy table: gene, predicted delta (relevant CTs), predicted nucleus and imputed pseudobulk, GT delta, external effect, mean external expression, replicate uncertainty, status (concordant/discordant/missing), caveat. Keep `Camk2n1` whatever it shows; the Mac exploration flagged it as potentially discordant in GSE210071 and that needs re-checking here. `Malat1` was reported absent from the GSE210071 FPKM table in the Mac exploration; confirm and report rather than impute.

---

## 11. Figures (PDF + PNG, plotted-data CSV each)

1. Design schematic: what each dataset measures vs what the model predicts (compartment vs ribosome-associated, explicitly labeled).
2. QC: library size / detected genes, PCA or correlation heatmap, per primary dataset, colored by compartment/assay, condition, batch.
3. H1 GSE210071: predicted delta vs C-minus-N and W-minus-N (triptolide + unstimulated), with GT delta panel.
4. H1 GSE121199: predicted (neuron-pooled) delta vs WC-minus-nuclear and RiboTag-minus-nuclear, male-only.
5. H2 GSE263082: nucleus-to-RiboTag vs imputed-to-RiboTag correlation for Pvalb and Sst with paired uncertainty; real-cell ceiling shown.
6. H3 enrichment: RiboTag abundance across predicted-delta quantiles, Pvalb and Sst separately; comparison to GSE77243 (prior result) on the same axes.
7. Showcase gene panel.
8. Sensitivity forest plot: Spearman or delta-similarity across datasets, subsets, and filters, with enrichment-type contrasts visually separated from compartment-type contrasts.

---

## 12. Report and bookkeeping

Write `outputs/report/{external_validation_report.md, executive_summary.md, methods.md, results.md, limitations.md}`. The executive summary answers: (1) does ImputeVAE recover the direction of nuclear-to-whole-cell shifts in independent compartment data; (2) does imputation move Pvalb and Sst profiles closer to external RiboTag profiles; (3) what replicates across datasets and across the prior GSE77243/GSE141464/Buckley results; (4) which genes or contexts fail; (5) what can be claimed now vs what needs MERFISH/FISH or paired human data.

Language: "concordant with", "supports", "improves similarity to". Never call RiboTag whole-cell ground truth. Distinguish cultured embryonic cortical cells under triptolide, adult NAc MSNs, pooled adult cortex/hippocampus interneurons, and CA1. No mouse-to-human generalization.

Bookkeeping at completion: `README.md` in this folder; `notes/experiments/trap_ribotag_external/README.md` in the style of `notes/experiments/trap_validation/README.md`; one row in the "Analysis Tasks" table of `notes/experiments/README.md`; a memory file `project_trap_ribotag_external.md` written to both memory dirs with a pointer in both `MEMORY.md` files; SLURM job IDs recorded in the README.

---

## 13. Starting benchmarks (from the Mac exploration; sanity checks only)

Simple pseudobulk Spearman correlations obtained off-cluster with the older delta matrix. Use only to catch gross parsing or contrast-direction errors; do not tune toward them.

- GSE210071 cytosol vs nucleus: about 0.31; whole cell vs nucleus: about 0.30
- GSE121199 whole cell vs nucleus: about 0.32; RiboTag vs nucleus: about 0.48
- GSE263082 Pvalb RiboTag abundance vs delta: about 0.34; Sst: about 0.36 (note GSE77243 gave 0.80 with the per-CT delta; a large gap here should prompt checking which delta and which normalization were used)
- GSE174303 WT cell-body TRAP vs input: about 0.06; neuropil: about -0.13 (expected per Section 0.4)

---

## 14. Acceptance criteria

- All source files audited, checksummed, and listed in `data_manifest.tsv`; nothing outside the list downloaded.
- Metadata explicit and reconciled with matrix columns for all five datasets; Batch5 of GSE263082 and the WC count discrepancy of GSE121199 resolved and documented.
- Model pseudobulks and per-CT predicted/GT deltas built from the full-data model with row alignment verified.
- Gene mapping quantified per dataset.
- H1 complete for GSE210071 (with triptolide caveat) and GSE121199; H2 and H3 complete for GSE263082 Pvalb and Sst; H4 and H5 complete on every primary contrast.
- Every contrast reports GT delta and CT-mean next to ImputeVAE; enrichment-type contrasts are labeled as such.
- Uncertainty, matched nulls, sensitivity analyses, and BH rules implemented; final numbers use 1,000 iterations.
- Figures have plotted-data tables and were visually inspected.
- Scripts run from the project root; SLURM jobs on scavenge with logs in `jobs/logs/`.
- Report distinguishes direct compartment validation from ribosome-associated supporting evidence; negative and discordant findings retained.
- Run log lists commands, versions, warnings, job IDs, and unresolved blockers; notes and memory updated.
