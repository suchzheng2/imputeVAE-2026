# Layer-5 driver lines: projection identity from the source literature

**Date:** 2026-09-24. Reason: Prashant (Sep 2026 meeting) asked that external L5 TRAP populations not be mapped to L5 IT / L5 ET by
name. ET = extratelencephalic (Allen "L5 PT" until Yao 2021; projects to thalamus, pons, medulla, spinal cord); IT =
intratelencephalic (corticostriatal / corticocortical). Quotes were pulled from the full texts (Europe PMC / PMC) or GEO SOFT
records. Companion data-driven check: `scripts/17_l5_mapping_audit.py`, tables `outputs/extension/tables/l5_audit_*.tsv`.

| Dataset | Line | Paper | Tissue sampled for TRAP | Stated layer | Projection identity | Key evidence | Confidence | Mapping used |
|---|---|---|---|---|---|---|---|---|
| GSE172484 | Gprin3 bacTRAP | Moya et al. 2022 Cell Rep, 10.1016/j.celrep.2022.110556 (PMID 35320722) | M1 motor cortex (~P40, "M1 cortex sub-dissected") | lower L5b | **ET, corticospinal** | "In Gprin3-bacTRAP mice, 82% of spinal CTB+ cells were EGFP+"; "94% of Gprin3 EGFP+ cells were co-labeled with CTIP2 ... confirming the PT identity" | high | **L5 ET** (unchanged) |
| GSE172484 | Colgalt2 (= Glt25d2) bacTRAP | same | M1 for the baseline batch; the SOD-negative batch is **whole cortex** | upper L5b | **ET, corticopontine** (9% of spinal CTB+ cells; 64% of pons-projecting UL5b cells) | "Colgalt2 cells represent the majority of pons-projecting cells in UL5b"; "79% ... co-labeled with CTIP2" | high | **L5 ET** (unchanged) |
| GSE202172 | S100a10 (p11) bacTRAP | Chottekalapanda et al. 2020 Mol Psychiatry, 10.1038/s41380-019-0473-6; tracing in Schmidt et al. 2012 Cell, 10.1016/j.cell.2012.03.038 | whole cortex (hippocampi and striata removed) | L5a | **IT, corticostriatal** | "consistent with ... intratelencephalic corticostriatal neurons"; Schmidt 2012: "72 +/- 5% of the cortical neurons labeled by injecting CTb into CPu were also EGFP-L10a+" | high | **L5 IT** (unchanged) |
| GSE141337 | Glt25d2-TRAP (DU9) | no publication in GEO; line characterised in Doyle 2008 Cell (PMID 19013282), Schmidt 2012, Groh 2010 Cereb Cortex, Kim 2015 Neuron | whole cerebral cortex | L5b | **ET** (pons, thalamus, spinal cord) | Schmidt 2012: "Glt25d2 expressing pyramidal cells are long-distance projection neurons that reside in layer 5b and send axons to pons, thalamus, and spinal cord" | high for ET; no paper, whole-cortex | **L5 ET** (unchanged) |
| GSE142028 | Rbp4-Cre (KL100) x Rpl10a-EGFP | no publication in GEO; Gerfen et al. 2013 Neuron (PMID 24360541), Harris et al. 2019 Nature, Tasic 2018 | whole cerebral cortex (dentate-gyrus samples not used) | pan-L5 | **mixed IT + ET, not NP** | Gerfen 2013: "pan layer 5 line ... expression in PT and IT corticostriatal neurons"; Tasic et al. 2018 Nature 563:72, 10.1038/s41586-018-0654-5 (PMID 30382198, PMC6456269), Extended Data Fig. 6 legend, panel b: "the L5 near-projecting types were missed owing to reliance on Rbp4-cre_KL100 to isolate L5 cells. We now find that, in contrast to pan-neuronal (Snap25-IRES-cre) or pan-glutamatergic Cre lines (Slc17a7-IRES-cre), Rbp4-cre_KL100 labels all L5 types except L5 near-projecting (Extended Data Fig. 8)" (VISp and ALM, not MOp; verified against the PMC author manuscript 2026-09-25) | high (mixed) | **EXCLUDED** (2026-09-25; was L5 IT, then POOL_L5_IT_ET on 2026-09-24). Rule: a line with no single matching subclass is skipped, not pooled. |
| GSE13379 | Etv1 (Er81) bacTRAP TS88 | Doyle et al. 2008 Cell, 10.1016/j.cell.2008.10.029 | whole cortex | L5a | **IT-leaning (corticostriatal), impure** | Doyle 2008: Etv1 line projects "to ... striatum"; but "the EGFP-L10a transgene may also be expressed in circulating cells in the CNS vasculature" | medium | L5 IT (secondary dataset, unchanged; caveat noted) |
| GSE13379 | Glt25d2 bacTRAP DU9 | same | whole cortex | L5b | **ET** ("pons/spinal cord") | Doyle 2008: "5b (Glt25d2) ... pons/spinal cord" | high | L5 ET (unchanged) |
| GSE133291 | Scnn1a-Tg3-Cre RiboTag | Furlanis et al. 2019 Nat Neurosci, 10.1038/s41593-019-0465-5 | **whole neocortex**, P25-28 | L4 | **L4 spiny stellate / star pyramid; no MOp counterpart** | "Scnn1a-cre for spiny stellate and star pyramid layer 4 cells"; MOp is agranular (Yao 2021 Nature) | high | **EXCLUDED** (was L5 IT) |

## Data-driven cross-check (`17_l5_mapping_audit.py`, real-cell markers only, never imputed data)

Marker z-scores within each external profile (mean z of the marker set; ET minus IT > 0 = ET-like), data-derived markers = top 40
genes specific to each L5 type among the 7 excitatory types in the real-cell pseudobulk; canonical markers = the verified panel listed below (Tshz2 alone for NP, so the NP canonical score is noisy):

| Population | data-derived: ET minus IT | canonical: ET minus IT | best set (data-derived) | centred Spearman, best type (real cell / nucleus / imputed) | literature | agree |
|---|---|---|---|---|---|---|
| Gprin3_L5b | +0.49 | +0.56 | L5 ET | L5 ET / L5 ET / L5 ET | ET | yes |
| Colgalt2_L5b | +0.40 | +0.32 | L5 ET | L5 ET / L5 ET / L5 ET | ET | yes |
| Glt25d2_L5b | +0.27 | +0.71 | L5 ET | L5/6 NP (0.05) / L5 ET / L5/6 NP (0.04); all near zero | ET | markers yes; profile too weak to discriminate |
| S100a10_L5a | -0.19 | -0.22 | L5 IT | L5/6 NP (0.17) / L5 IT / L5/6 NP (0.19) | IT | markers yes; profile IT/NP |
| Rbp4_L5 | +0.07 | +0.52 | L5 ET (0.39) vs L5 IT (0.32) | L2/3 IT / L5 ET / L5 ET | mixed IT + ET | consistent with a mixture |
| Scnn1a_L4 | -0.60 | -1.27 | L5 IT | L5 IT / L5/6 NP / L5 IT | L4, no counterpart | excluded |

Uncentered H2 Spearman differs by at most 0.03 between L5 IT, L5 ET and L5/6 NP for every population
(`l5_audit_each_type_H2.tsv`), so the choice of L5 type does not change the H2 conclusion (imputed ~ real cell > nucleus);
the audit is about presenting the mapping correctly, not about the numbers.

## Marker genes with a verified source (present in the 10,134-gene space)
- L5 ET: Fezf2, Bcl11b/Ctip2 (Harris & Shepherd 2015; Yao 2021 Cell), Bcl6 (Tasic 2018), Npnt (non-medulla-projecting ET) and Slco2a1 (medulla-projecting ET) (Yao 2021 Nature, MOp). Fam84b (Tasic 2018) is not in the gene space. Npr3 is a Cre line, not a verified marker; dropped.
- L5 IT: Rorb (Yao 2021 Nature: "Rorb for L4/5 IT and L5 IT"), Slc30a3 (pan-IT, Tasic 2018), Cux2 (Tasic 2018), Rspo1 (MOp L4/5 IT, Yao 2021 Nature). Deptor / Otof not verified as L5 IT; dropped.
- L5/6 NP: Tshz2 (Tasic 2018; Yao 2021). Slc17a8, Trhr, Sla2, Rapgef3 (Tasic 2018) not in the gene space. Nxph1 unverified; dropped.

Caveats: several TRAP datasets pooled whole cortex, so region matching to MOp is approximate (Prashant: cortex-to-cortex is
acceptable, hippocampus is not). Etv1 is also expressed in L5 ET and L5/6 NP (Yao 2021), so it is not an IT marker.

## Terminology check: do the source papers themselves say ET / IT? (2026-09-25)

Full texts were re-read (PMC XML via NCBI efetch; Furlanis 2019 via Europe PMC) and searched for "extratelencephalic",
"intratelencephalic", "telencephalic", "pyramidal tract" and "PT". Only one paper uses the word "intratelencephalic";
none uses "extratelencephalic". The others use the older synonym **pyramidal tract (PT)** or name the projection target
directly. PT = ET is the standard equivalence (Allen "L5 PT" until Yao 2021 renamed it "L5 ET"; Gerfen 2013 defines PT
cells by collaterals to thalamus, subthalamic nucleus, superior colliculus, pons and spinal cord, which is the ET definition).

| Paper | Line(s) | "intra/extratelencephalic" used? | Terminology actually used |
|---|---|---|---|
| Chottekalapanda 2020 (GSE202172) | S100a10 | **yes, once**: "These properties are consistent with the electrophysiological parameters previously reported for intratelencephalic corticostriatal neurons in adult mice" | corticostriatal, layer 5a |
| Gerfen 2013 (GSE142028) | Rbp4 KL100 | no; defines PT vs IT explicitly, spelled "intertelencephalic (IT)" | "PT corticostriatal" vs "IT corticostriatal"; Rbp4 KL100 listed as subtype "PT/IT" and shown in Fig. 2 as "expression in PT and IT corticostriatal neurons" |
| Moya 2022 (GSE172484) | Gprin3, Colgalt2 | no | "L5b pyramidal tract-projecting (PT) neurons"; corticospinal (Gprin3), corticopontine (Colgalt2); CTIP2 co-labeling "confirming the PT identity of both populations" |
| Schmidt 2012 (tracing for S100a10 and Glt25d2) | S100a10, Glt25d2 | no | corticostriatal (CStr) vs corticopontine; S100a10 reporter shows "no labeling in the thalamus or pyramidal tract" and "little overlap" with corticospinal cells |
| Doyle 2008 (GSE13379) | Etv1, Glt25d2 | no | one mention of "corticostriatal"; otherwise layer and target descriptions only |
| Furlanis 2019 (GSE133291) | Scnn1a | no | "Scnn1a-cre for spiny stellate and star pyramid layer 4 cells" |

Implication: the ET / IT assignments in the table above are a translation of projection-target evidence (retrograde tracing
from pons / spinal cord / striatum, CTIP2 staining) into the Yao 2021 nomenclature used by our MOp labels, not a
relabelling of L5a / L5b. The two populations whose original mapping rested on the layer name alone (Scnn1a "L4",
Rbp4 "pan-L5") were the ones changed; the four that kept their assignment did so because tracing supports it.

## External marker check with Yao 2021 MOp markers (2026-09-25) — replaces the real-cell-derived marker set as the primary check

Reason: the "data-driven" markers above come from this repo's real-cell pseudobulk, which is also the H2 "real whole cell"
ceiling, so that check was not independent of a reported number. This check uses only an external marker list and the
external TRAP profiles; no model pseudobulk is loaded.

**Marker source:** Yao et al. 2021 Nature 598:103 (10.1038/s41586-021-03500-8), Supplementary Table 6 (pairwise conserved DE
genes between the 116 MOp consensus types). Download: Supplementary Tables 1-9 zip at
https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-03500-8/MediaObjects/41586_2021_3500_MOESM4_ESM.zip
(from https://www.nature.com/articles/s41586-021-03500-8, "Supplementary information"). Local copy of the glutamatergic rows,
derived fraction table and provenance: `outputs/extension/metadata/yao2021/`. A gene is a marker of L5 IT / L5 ET / L5/6 NP if
it is higher in that subclass's clusters than in the other two L5 subclasses' clusters in >= 80% of cluster pairs and higher
than all other glutamatergic clusters in >= 50% of pairs: 16 / 29 / 12 genes (16 / 25 / 11 in the model gene space).
L5 ET: Gprc5b, Bcl6, Cacna1h, Tcerg1l, Pou3f1, Npr3, Vat1l, Syt2, Trpc4, ... L5 IT: Cpne9, Slc30a3, Ntng2, Dkkl1, Inhba, Kcnc2, ...
L5/6 NP: Nxph1, Unc13c, Tshz2, Grp, Vwc2l, ... (Fezf2 is not an ET marker under this rule because Yao's data show it shared with NP.)

**Scoring** (`scripts/18_l5_mapping_audit_external_markers.py`, log `outputs/extension/logs/18_l5_audit_external_markers.log`):
on the 7,562 genes expressed in all 7 external glutamatergic TRAP profiles (CamK2a, Scnn1a, Colgalt2, Gprin3, S100a10, Glt25d2,
Rbp4) and in the model gene space, each gene is z-scored ACROSS the external profiles; a set score is the mean z of its markers.
Decision = sign of (ET score - IT score) with a 0.1 margin. The within-profile z of script 17 is reported alongside.
Tables `outputs/extension/tables/l5_audit_extmarkers_{scores,gene_z,summary,yao2021_sets}.tsv`; figure `outputs/extension/figures/L5_audit_external_markers.png`.

| Population | literature | cross-population z: IT / ET / NP | ET minus IT | call | agrees | within-profile ET minus IT |
|---|---|---|---|---|---|---|
| Gprin3_L5b | ET | -0.12 / **0.47** / 0.00 | +0.60 | L5 ET | yes | +0.28 |
| Colgalt2_L5b | ET | 0.10 / **0.75** / -0.20 | +0.65 | L5 ET | yes | +0.33 |
| S100a10_L5a | IT | -0.46 / -0.64 / 0.10 | -0.18 | L5 IT | yes | -0.08 (indeterminate) |
| Glt25d2_L5b | ET | 0.03 / 0.29 / 0.54 | +0.26 | L5 ET | yes | +0.22 |
| Rbp4_L5 | mixed IT+ET | 0.09 / 0.10 / 0.46 | +0.01 | indeterminate | consistent with a mixture; **excluded from H2/H3 on 2026-09-25** | +0.08 |
| Scnn1a_L4 | L4, excluded | 0.09 / -0.60 / -0.61 | -0.69 | L5 IT-like (L4 IT genes Cpne9, Slc30a3, Dkkl1 high) | n/a | -0.53 |

Conclusion: with an external marker list the ET-vs-IT direction matches the tracing literature for every population, and the
mapping decisions above are unchanged. Caveats: (i) the NP set scores highest in the three whole-cortex pulldowns (S100a10,
Glt25d2, Rbp4), driven by Tshz2, Camk2d and Vwc2l; no external population is NP, so this is reported as a flag, not used for
assignment; (ii) cross-population z over 7 profiles from 6 labs is confounded by dataset (S100a10 is low on every L5 set),
which is why only the ET-IT contrast, not the absolute level, is interpreted; (iii) the within-profile z is confounded by
gene abundance instead, and gives the same signs (S100a10 falls inside the margin).
