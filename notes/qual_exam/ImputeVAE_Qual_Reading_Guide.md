# ImputeVAE — Qualifying Exam Reading Guide

**Suchen Zheng · Gerstein Lab, Yale (CBB)**
Project: **ImputeVAE** — a conditional VAE that translates single-nucleus RNA-seq (snRNA-seq) into single-cell RNA-seq (scRNA-seq), predicting the cytoplasmic transcriptome from nuclear RNA. Trained on matched mouse motor cortex (MOp), Allen Brain Cell Atlas, 10x Chromium v3. **Not batch correction — a directed nucleus→cell biological shift.**

---

## How to use this guide

For each of the five exam themes you get: (1) the **missing papers** — recent (2022–2025) and foundational — with verified citations and links, in a **suggested reading order**; (2) **key findings** to be ready to discuss; (3) an explicit **connection to ImputeVAE**; and (4) a **likely committee question** per paper. Themes close with the **datasets** worth using and the **critiques** each theme arms you against.

Papers you already have are *not* re-listed as missing, but they're slotted into the reading order so you can see where new reading connects to old.

Read the two synthesis sections at the end before your exam: **Papers that span multiple themes** (the connective tissue a committee loves to probe) and **The hardest critiques + how Gerstein will attack this** (your project's real vulnerabilities, and the framing your PI specifically cares about).

> ### ⚠️ Three things to fix / confirm before you finalize
> 1. **Disambiguate your "Yao 2021."** The genuinely missing flagship is the **BICCN 2021 MOp census** (*Nature* 598:86–102, `s41586-021-03950-0`) — distinct from the **Yao 2021 transcriptomic MOp atlas** (`s41586-021-03500-8`, the 7-platform integration). Confirm which one is already on your list and add the other. Both are below (Theme 1).
> 2. **A few author lists were CAPTCHA-blocked during verification** — every one is flagged inline. Titles/journals/years/DOIs are confirmed; pull the author lists from the PMID before citing by name.
> 3. **The "sampling not biology" counter-argument** (Genome Research 2024) is the single sharpest external attack on your thesis — it's in Theme 5. Have your answer cold.

---
---

# Theme 1 — Matched Cell vs Nucleus Datasets

Survey of matched scRNA/snRNA datasets: their biases, which cell types are gained/lost, and how protocol (10x v2 vs v3, plate vs droplet) shapes comparisons. This theme's deepest payload for you: **your "cells" are not a neutral ground truth**, and part of the nucleus→cell shift you learn is technical (dissociation stress, ambient RNA), not cytoplasmic biology.

## Reading order

1. **Oh et al. 2022** — cleanest short demo that sc vs sn give different *cell-type proportions*, and why. Frames the "cells aren't neutral" problem.
2. **van den Brink et al. 2017** — foundational: dissociation itself induces a stress/immediate-early gene artifact.
3. **Marsh et al. 2022** — the brain-specific, deeper version: enzymatic dissociation fabricates "activated" glial/microglial signatures.
4. **BICCN 2021 (flagship census)** — the MOp consortium capstone; slot your existing **Yao 2021 atlas, Yao 2023, Hodge 2019, Bakken 2018** around it.
5. **Eraslan et al. 2022 (GTEx snRNA-seq)** — cross-tissue proof that snRNA-seq loses immune cells and carries less dissociation signature; slot **Ding 2020** here.
6. **Young & Behjati 2020 (SoupX)** — ambient-RNA contamination model.
7. **Fleming et al. 2023 (CellBender)** — the deep-learning ambient-RNA successor (a VAE — your closest methodological cousin); slot **Denisenko 2020, Thrupp 2020** nearby.

## The missing papers

### BICCN 2021 — flagship MOp cell census
BRAIN Initiative Cell Census Network. *A multimodal cell census and atlas of the mammalian primary motor cortex.* **Nature** 598, 86–102 (2021). DOI [10.1038/s41586-021-03950-0](https://www.nature.com/articles/s41586-021-03950-0) · [PMC8494634](https://pmc.ncbi.nlm.nih.gov/articles/PMC8494634/)

- **Key findings.** Consortium capstone integrating transcriptomics, chromatin, methylation, spatial, morphology, ephys and connectivity into a consensus MOp taxonomy conserved across mouse, marmoset, human. Establishes cross-platform, cross-modality *correspondence* of cell types — the reference for "how many cell types are in MOp and how well do platforms agree."
- **Connection.** The parent ecosystem your ABC-Atlas 10x-v3 training set is drawn from; defines your ~17-type scaffold and documents that platform choice changes which types appear and how cleanly they separate. Legitimizes "matched MOp" but warns that "matched" is a consortium-level alignment, not identical cells.
- **Likely committee question.** *"Your alignment term pulls matched cell types together — but BICCN shows cell-type boundaries themselves shift by platform. How do you know your labels are consistent enough across cell and nucleus that the alignment term isn't papering over a labeling artifact?"*

### Eraslan et al. 2022 — GTEx single-nucleus cross-tissue
*Single-nucleus cross-tissue molecular reference maps toward understanding disease gene function.* **Science** 376, eabl4290 (2022). DOI [10.1126/science.abl4290](https://www.science.org/doi/10.1126/science.abl4290) · [PMC9383269](https://pmc.ncbi.nlm.nih.gov/articles/PMC9383269/)

- **Key findings.** 209K nuclei, 8 tissues, 4 snRNA methods, benchmarked against scRNA-seq. sn/sc annotations correspond well, but **tissue-dissociation signatures are markedly elevated in scRNA-seq and largely absent in snRNA-seq**; snRNA recovers fragile/adherent types scRNA loses, while under-sampling immune cells.
- **Connection.** The strongest evidence that the shift you learn is **partly a dissociation-stress artifact on the cell side**, not purely cytoplasm-vs-nucleus biology. Supplies candidate stress-gene signatures to check you're *not* hallucinating into your prediction; also gives out-of-domain tissues to test generalization.
- **Likely committee question.** *"When your model predicts cytoplasm from a nucleus, is it recovering real cytoplasmic biology or injecting an artifactual stress signature the cell side happens to carry? How would you separate those?"*

### Marsh et al. 2022 — dissociation-induced glial artifacts
*Dissection of artifactual and confounding glial signatures by single-cell sequencing of mouse and human brain.* **Nature Neuroscience** 25, 306–316 (2022). DOI [10.1038/s41593-022-01022-8](https://www.nature.com/articles/s41593-022-01022-8)

- **Key findings.** Warm enzymatic dissociation generates strong *ex vivo* activation artifacts specifically in glia — microglia acquire an artifactual "activated" transcriptome mistakable for disease state. snRNA-seq and cold-protease protocols suppress it.
- **Connection.** Your whole-cell "ground truth" for microglia/glia may be contaminated by dissociation-activation artifacts. Any glial "cytoplasm" delta or GO term should be cross-checked against this artifact list; also explains part of the microglial loss in nuclei that skews your matched composition.
- **Likely committee question.** *"Thrupp says snRNA under-detects microglial activation; Marsh says scRNA over-detects it as an artifact. For microglia, which modality is your ground truth, and does your per-type performance drop there?"*

### van den Brink et al. 2017 — dissociation-induced gene expression
*Single-cell sequencing reveals dissociation-induced gene expression in tissue subpopulations.* **Nature Methods** 14, 935–936 (2017). DOI [10.1038/nmeth.4437](https://www.nature.com/articles/nmeth.4437) · [PMC5665481](https://pmc.ncbi.nlm.nih.gov/articles/PMC5665481/)

- **Key findings.** The foundational short paper: warm enzymatic dissociation induces a reproducible module of stress/immediate-early genes (Fos, Jun, Hspa1a/b) absent in vivo. Established the practice of flagging/regressing a "dissociation module."
- **Connection.** Defines the concrete gene list contaminating the cell (cytoplasmic) side of your training pairs. Because your log1p targets include these genes, your decoder can learn to "add" them as if genuine nuclear→cytoplasmic upregulation. **This is the source of your poster's Fos/Arc landmine.**
- **Likely committee question.** *"Are dissociation IEGs among the genes your model most confidently adds going nucleus→cell? If so, is R²=0.798 partly rewarding you for reproducing a protocol artifact?"*

### Oh et al. 2022 — sc vs sn cell-type distribution
*Comparison of cell type distribution between single-cell and single-nucleus RNA sequencing…* **Experimental & Molecular Medicine** 54, 2128–2134 (2022). DOI [10.1038/s12276-022-00892-z](https://www.nature.com/articles/s12276-022-00892-z)

- **Key findings.** In matched samples, sc and sn recover similar diversity but very different *proportions*: sn enriches adherent/fragile cells, sc enriches immune cells and shows stress/apoptosis signatures; sn composition better matched H&E and bulk.
- **Connection.** Your cleanest citation for "cells are not neutral ground truth" and that sc/sn differ in *composition*, not just per-gene expression — motivating a check on whether rare types (microglia) are so depleted on one side that the alignment term is under-constrained.
- **Likely committee question.** *"If cell-type proportions differ between your nuclei and cells, your pairing is at the cluster/pseudobulk level, not single-cell. How does compositional mismatch bias both your cell-type-mean baseline and your alignment loss?"*

### Young & Behjati 2020 — SoupX (ambient RNA)
*SoupX removes ambient RNA contamination from droplet-based single-cell RNA sequencing data.* **GigaScience** 9(12), giaa151 (2020). DOI [10.1093/gigascience/giaa151](https://academic.oup.com/gigascience/article/9/12/giaa151/6049831) · [PMC7763177](https://pmc.ncbi.nlm.nih.gov/articles/PMC7763177/)

- **Key findings.** Formalizes "the soup": every droplet carries cell-free ambient mRNA dominated by highly-expressed genes of abundant types; SoupX estimates and subtracts it. Ambient is *worse in snRNA-seq* because nuclei prep releases cytoplasmic mRNA into suspension.
- **Connection.** A critical confound in *both* directions: ambient in your nucleus input can carry cytoplasmic transcripts (ribosomal, etc.) that leak in — so some "cytoplasmic" signal may be soup, and some you "predict" may be echoing soup already present.
- **Likely committee question.** *"Ribosomal and Malat1/Neat1 genes dominate ambient soup. Was ambient RNA removed before training? If not, how is your recovery of ribosomal genes / depletion of Malat1/Neat1 not a soup effect?"*

### Fleming et al. 2023 — CellBender
*Unsupervised removal of systematic background noise from droplet-based single-cell experiments using CellBender.* **Nature Methods** 20, 1323–1335 (2023). DOI [10.1038/s41592-023-01943-7](https://www.nature.com/articles/s41592-023-01943-7)

- **Key findings.** A deep generative (VAE-style) model that jointly infers the ambient profile, per-droplet contamination, and true counts, outperforming SoupX; widely adopted for snRNA-seq where ambient is severe.
- **Connection.** Your closest methodological cousin — a VAE on the *same* droplet data — so the committee will expect you to know it. The recommended upstream step to make your mapping about biology, and a natural point of contrast for your cVAE's assumptions.
- **Likely committee question.** *"CellBender already uses a VAE to separate ambient background from true content. How is your cVAE conceptually different, and would applying it upstream change your headline correlations — especially the cross-protocol v2 case?"*

## Datasets to consider

- **Allen/BICCN MOp multi-platform** (Yao 2021 atlas; BICCN 2021) — matched sc/sn across **10x v2, 10x v3, SMART-seq v4**; the natural home for a *controlled* v2-vs-v3 test of your cross-protocol drop (same types on all platforms). Via NeMO / ABC Atlas.
- **GTEx snRNA-seq (Eraslan 2022)** — 8 non-brain tissues; out-of-domain generalization test + curated dissociation-signature gene sets.
- **Ding 2020 (yours)** — matched sc/sn across many protocols; protocol-confound ablation.
- **Oh 2022** — small clean paired sc/sn outside brain for a composition-bias sanity check.
- **Marsh 2022 / van den Brink 2017 gene modules** — not training data but essential *gene lists* to overlay on your delta.

## Critiques this theme arms you against

1. **"Cells" are not neutral ground truth.** Whole-cell cortex is biased toward easily-dissociated cells and against fragile neurons; nuclei over-represent neurons and lose microglia/immune. A perfect nucleus→cell map reproduces that bias, not corrects it. Frame your target as *"10x-v3 whole-cell cortex composition,"* not *"the true cytoplasm."*
2. **Part of the learned shift is a dissociation artifact.** Audit your delta's GO enrichment against IEG/stress and glial-activation gene sets — show you're not rewarded for reproducing a protocol signature.
3. **Ambient RNA is an unmodeled confound on both ends.** Without upstream CellBender/SoupX, "recovers ribosomal, depletes Malat1/Neat1" is confounded by differential soup. Likely a top-2 attack.
4. **Protocol confounds are entangled with the nucleus/cell axis.** Your v3→v2 result (QQ corr 0.894 holds, gene corr 0.593 drops) is the signature of a model that learned a *distributional* shift robust to chemistry but a *cell-type-conditional* map that is partly v3-specific. State this honestly.
5. **"Not batch correction" — prove the negative.** Dissociation artifacts + ambient soup are exactly what batch models remove; be ready to argue your directed shift is *separable* from those nuisances (the gene-set audits are your evidence).

---
---

# Theme 2 — Alternative Cytoplasmic RNA Profiling (Patch-Seq, TRAP/RiboTag)

The orthogonal-validation theme. Your strongest defense of the delta is a *physical* measurement of cytoplasmic RNA the model never saw. Three axes: **Patch-seq** (aspirated neuronal cytoplasm), **TRAP/RiboTag** (ribosome-bound = translated, cytoplasmic mRNA per genetic cell type), and **subcellular fractionation** (nuclear-vs-cytoplasmic partition of the same cell).

## Reading order

1. **Lipovsek 2021** — orients the theme: what patch-seq measures, its biases, why aspirate ≠ whole cell.
2. **Camuñas-Soler 2024** — modern framing of coupling transcriptomics to phenotype; situates cytoplasmic profiling methods.
3. **Sanz 2009 (RiboTag)** — the foundational cytoplasmic-capture method you're missing (you have TRAP, not RiboTag).
4. **Sheehan 2025** — recent cell-type TRAP/RiboTag atlas in mouse cortex; your validation template.
5. **Parallel TRAP + total RNA-seq (2020)** — computes ribosome engagement = TRAP/total, the direct analog of your delta.
6. **Abdelmoez 2018 (SINC-seq)** — separates nuclear & cytoplasmic RNA from the *same single cell*; the most direct test of your partition.
7. **Glock 2021** — soma/dendrite/axon translatomes; validates the *synaptic/dendritic* arm of your delta.

## The missing papers

### Lipovsek et al. 2021 — "Patch-seq: Past, Present, and Future"
**J. Neurosci.** 41(5):937–946 (2021). DOI [10.1523/JNEUROSCI.1653-20.2020](https://www.jneurosci.org/content/41/5/937) · [PMC7880286](https://pmc.ncbi.nlm.nih.gov/articles/PMC7880286/)

- **Key findings.** Consensus review defining patch-seq (whole-cell patch-clamp ephys + morphology + transcriptomics by aspirating a recorded neuron's cytoplasm). Catalogs pitfalls: RNA degradation, low yield vs dissociated scRNA-seq, neuropil/nuclear contamination, reduced gene detection; sets QC best practices.
- **Connection.** The methodological charter for your patch-seq validation. Aspirate is *cytoplasm-biased* (nucleus largely stays in tissue), so genes patch-seq-detectable but snRNA-depleted are candidate "true cytoplasmic" transcripts — if your added genes overlap that set for the matched MOp type, that's orthogonal support. Also flags a confound: neuropil contamination can *inflate* apparent synaptic signal.
- **Likely committee question.** *"Patch-seq still captures some nuclear RNA and contaminating neuropil. How do you know the transcripts you add are genuinely cytoplasmic rather than patch-seq contamination you happen to reproduce?"*

### Camuñas-Soler 2024 — integrating transcriptomics with cellular phenotypes
**Biophysical Reviews** 16:89–107 (2024). DOI [10.1007/s12551-023-01174-2](https://link.springer.com/article/10.1007/s12551-023-01174-2)

- **Key findings.** Reviews methods pairing single-cell transcriptomics with functional phenotype (patch-seq centerpiece), including computational integration and cross-modal prediction; discusses what whole-cell aspirate vs nucleus vs dissociated cell each captures.
- **Connection.** Positions your cVAE within the "predict one modality from another" landscape and gives vocabulary to justify snRNA→scRNA as a legitimate cross-modal task warranting a *generative* model over a fixed export factor.
- **Likely committee question.** *"Why a conditional VAE rather than a regression or a mechanistic export/decay model? What does the latent space buy you?"*

### Sanz et al. 2009 — RiboTag
**PNAS** 106(33):13939–13944 (2009). DOI [10.1073/pnas.0907143106](https://www.pnas.org/doi/10.1073/pnas.0907143106) · [PMC2728999](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2728999/)

- **Key findings.** Cre-dependent HA-tagged Rpl22 knock-in lets you IP ribosomes (and bound, actively-translated mRNA) from genetically-defined cell types in native tissue — the cytoplasmic, ribosome-engaged pool without dissociation.
- **Connection.** Closes your TRAP/RiboTag method gap (you have bacTRAP, not the genetic RiboTag now more common in mouse cortex). If your delta is real cytoplasmic recovery, added genes for a type should be enriched among that type's RiboTag-positive genes and *exclude* nuclear-retained lncRNAs.
- **Likely committee question.** *"TRAP/RiboTag measure translation, not steady-state cytoplasmic abundance. You predict total cytoplasmic RNA — which genes would the two disagree on?"*

### Sheehan et al. 2025 — glial circadian TRAP atlas
**Nature Neuroscience** 28:2366–2379 (2025). DOI [10.1038/s41593-025-02067-1](https://www.nature.com/articles/s41593-025-02067-1)

- **Key findings.** TRAP/RiboTag isolation of cell-type ribosome-bound mRNA from astrocytes and microglia in mouse cortex across circadian time, aging, and amyloid pathology; robust cell-type translatome recovery and context-dependent reprogramming. (Co-authored by Dougherty, whose 2010 TRAP paper you cite.)
- **Connection.** A current reference translatome in your **same tissue and species** — matched cell-type ribosome-bound gene sets as "cytoplasmic ground truth." Glia are a useful negative-control-style check (your model should not massively rebalance glial ribosomal content).
- **Likely committee question.** *"Your training is neurons in MOp. If you validate against glial TRAP, are you testing cytoplasmic recovery or just cell-type identity? What's the neuron-specific TRAP reference?"*

### Parallel human-neuron TRAP + total RNA-seq (2020)
*Quantification of mRNA ribosomal engagement in human neurons using parallel TRAP and RNA sequencing.* **STAR Protocols** (2020). PMID 33364619 · [PMC7753199](https://pmc.ncbi.nlm.nih.gov/articles/PMC7753199/) · [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2666166720302161)
> ⚠️ Title/journal/year/PMID/PMC confirmed; **author list CAPTCHA-blocked — pull from PMID 33364619** before citing by name.

- **Key findings.** Runs TRAP and total RNA-seq *in parallel on the same human neuron samples*, computing per-gene ribosome engagement as the TRAP-to-total ratio.
- **Connection.** The closest published analog to your delta: your (cell − nucleus) parallels their (TRAP − total) — both isolate a cytoplasm/translation-weighted residual. Gives you a validated analysis framework (engagement ratios, normalization, which gene classes move) to mirror on your delta and cite as precedent.
- **Likely committee question.** *"Is your delta a cytoplasmic signal, or is it just capturing highly-expressed housekeeping genes? Show the analog of a TRAP/total engagement plot for your prediction."*

### Abdelmoez et al. 2018 — SINC-seq
**Genome Biology** 19:66 (2018). DOI [10.1186/s13059-018-1446-9](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-018-1446-9)

- **Key findings.** Microfluidic separation + sequencing of the **nuclear and cytoplasmic RNA of the same single cell**. In K562, ~16% nuclear / ~84% cytoplasmic; identifies nuclear-retained introns and nuclear-enriched transcripts; N:C correlation tracks cell state. (NanoSINC-seq, *Sci Adv* 2021, extends to isoform level.)
- **Connection.** The single most direct orthogonal test of the *partition* your model exploits — empirically defines, per cell, which transcripts are nuclear-retained vs cytoplasmic. Check your removed set (Malat1, Neat1, nuclear lncRNAs) matches SINC-seq's nuclear class and your added set the cytoplasmic class. **Caveat to raise first:** public data are K562, not neurons — validates the *principle*, not neuron-specific identities.
- **Likely committee question.** *"SINC-seq shows N:C ratios are cell-state-dependent and transcript-specific. Your model learns one mapping — how do you handle a transcript whose export ratio differs across neuronal states?"*

### Glock et al. 2021 — translatome of soma, dendrites, axons
**PNAS** 118(43):e2113929118 (2021). DOI [10.1073/pnas.2113929118](https://www.pnas.org/doi/10.1073/pnas.2113929118) · [PMC8639352](https://pmc.ncbi.nlm.nih.gov/articles/PMC8639352/)

- **Key findings.** Compartment-resolved translatomes: a large, compartment-specific pool of locally translated mRNAs; dendrites/axons enriched for synaptic, cytoskeletal, and translation-machinery transcripts distinct from the somatic translatome.
- **Connection.** Your delta is enriched for **synaptic** transcripts, yet nuclei are somatic. Glock provides the independent reference for which transcripts are peripherally localized/translated — overlap argues you're recovering genuinely cytoplasmically-exported mRNA, not noise. Also flags that dissociation/patch-seq may under-sample dendritic RNA.
- **Likely committee question.** *"Nuclei are somatic, yet your delta is synaptic/dendritic. Are you predicting cytoplasmic export or subcellular localization — and can a steady-state nucleus even contain the information to recover dendritically-localized mRNA?"*

## Datasets for validation

- **Allen Institute Patch-seq (mouse cortex / MOp)** — Gouwens 2020 & Scala 2021 (both yours) via [Allen Brain Map](https://celltypes.brain-map.org/) and BICCN. **Best-matched orthogonal set** — same species, overlapping region, same taxonomy. Build "patch-seq-detected vs snRNA-depleted" gene panels per t-type.
- **BICCN MOp multimodal atlas** (Nature 2021; NeMO Archive) — includes the patch-seq subset for your exact region.
- **Patch-seq ephys+morphology (DANDI Archive)** — NWB-format recordings if you want cell-identity conditioning.
- **RiboTag/TRAP references (GEO/SRA)** — Sanz 2009, Doyle/Heiman bacTRAP, Sheehan 2025 glial atlas. Build per-type ribosome-bound gene sets. *(Copy accessions from each paper's Data Availability — don't invent GEO IDs.)*
- **Glock 2021 compartment translatome (GEO, Schuman lab)** — for the synaptic-enrichment check.
- **SINC-seq / NanoSINC-seq (GEO)** — empirical single-cell N:C partition (K562).

## Critiques this theme arms you against

- **Patch-seq is low-throughput and biased** — neuropil contamination can add the very synaptic transcripts you're validating → risk of *circular* validation. (Lipovsek 2021, Camuñas-Soler 2024.)
- **TRAP/RiboTag measure translation, not steady-state cytoplasmic abundance** — over-represents highly-translated genes, under-represents stored/repressed mRNAs and many lncRNAs. Supportive, not sufficient. (Sanz 2009; parallel-TRAP 2020.)
- **N:C ratios are transcript- and state-specific** — a single learned mapping may not generalize across states/rare types. (Abdelmoez 2018.)
- **Somatic nucleus vs peripheral localization** — does the nucleus even carry information to recover localized mRNA? (Glock 2021.)
- **Reference-species/region matching** — SINC-seq (K562), human-neuron TRAP, glial atlas are off-tissue/off-cell-type for principle-level validation; lead with **mouse-cortex Patch-seq and TRAP** for the primary test.

---
---

# Theme 3 — RNA Localization in Brain

The ground-truth biology that lets you argue your delta is real: cytoplasmic/ribosomal/synaptic transcripts *added*, nuclear-retained lncRNAs (Malat1, Neat1) *removed*. Spans nuclear-retention mechanisms, genome-wide subcellular atlases, and neuron-specific dendritic/synaptic localization.

## Reading order

1. **Bahar Halpern 2015** — *why* nuclear retention exists (noise buffering); your delta models a real quantity, not dropout.
2. **Fox 2018 (paraspeckles)** — the mechanistic "why" for Neat1 retention (and, by extension, Malat1).
3. **Roundtree 2017 (YTHDC1/m6A)** — the molecular export gate separating retained from exported transcripts.
4. **Xia 2019 (MERFISH subcellular)** — imaging ground truth: which genes are nuclear- vs cytoplasm-enriched, in situ.
5. **Fazal 2019 (APEX-seq)** — genome-wide biochemical compartment map; your added/removed sets should track it.
6. **Zappulo 2017 (neurite localization)** — RNA localization (not protein transport) drives the neurite proteome.
7. **Tushev 2018 (3′UTR isoforms)** — resolution check: localization is isoform-specific, stress-testing your gene-level delta.
8. **Perez 2021 (single-neuron subcellular)** — closest analog: soma vs dendrite transcriptomes of single neurons, cell-type-specific.

## The missing papers

### Bahar Halpern et al. 2015 — nuclear retention buffers transcriptional noise
**Cell Reports** 13(12):2653–2662 (2015). DOI [10.1016/j.celrep.2015.11.036](https://www.cell.com/cell-reports/fulltext/S2211-1247(15)01351-0) · [PMC4700052](https://pmc.ncbi.nlm.nih.gov/articles/PMC4700052/)

- **Key findings.** A substantial, gene-specific fraction of spliced polyadenylated mRNAs sits higher in the nucleus than cytoplasm in intact tissue (~30% of genes in β-cells, ~13% in liver); retained transcripts dwell for hours; modeling shows retention buffers cytoplasmic noise *without* changing mean cytoplasmic levels.
- **Connection.** Foundational justification that nucleus-vs-cytoplasm difference is a real, quantifiable, gene-specific quantity — exactly your delta. Explains why some mRNAs (not just lncRNAs) are cytoplasm-underrepresented, so a nonzero gene-varying delta is expected.
- **Likely committee question.** *"If retention only buffers noise but preserves the mean cytoplasmic level, why should your model see a systematic nucleus→cytoplasm shift rather than just a variance change?"* (Answer: retained genes have elevated *nuclear steady-state fraction*; snRNA samples that inflated pool, so the mean N:C ratio differs per gene.)

### Fox, Nakagawa, Hirose & Bond 2018 — paraspeckles / Neat1
**Trends in Biochemical Sciences** 43(2):124–135 (2018). DOI [10.1016/j.tibs.2017.12.001](https://www.cell.com/trends/biochemical-sciences/abstract/S0968-0004(17)30224-4) · [PubMed](https://pubmed.ncbi.nlm.nih.gov/29289458/)

- **Key findings.** Neat1 is the essential architectural scaffold of paraspeckles — membraneless nuclear bodies built by liquid–liquid phase separation with NONO/SFPQ; the RNA is by construction nuclear-immobile.
- **Connection.** Direct mechanistic ground truth for why **Neat1** is nuclear-retained. Pair with Malat1 (nuclear-speckle-associated, triple-helix-stabilized). Their strong depletion in your predicted cytoplasm is biologically *mandatory*, not a quirk — your single strongest "why" for the headline removed genes.
- **Likely committee question.** *"Malat1 and Neat1 are your poster-child removed genes — what physically keeps each in the nucleus, and is the mechanism the same?"* (Neat1 = paraspeckle scaffold; Malat1 = speckle-retained, no canonical export, 3′ triple-helix.)

### Roundtree et al. 2017 — YTHDC1 / m6A licenses nuclear export
**eLife** 6:e31311 (2017). DOI [10.7554/eLife.31311](https://elifesciences.org/articles/31311) · [PMC5648532](https://pmc.ncbi.nlm.nih.gov/articles/PMC5648532/)

- **Key findings.** The nuclear m6A reader YTHDC1 promotes export of m6A-methylated mRNAs (via SRSF3/NXF1); knockdown prolongs nuclear residence. m6A is an active determinant of export kinetics.
- **Connection.** A molecular mechanism for the export step your delta summarizes at steady state — reframes nucleus→cytoplasm as regulated and sequence/modification-dependent, so per-gene export differences predict per-gene deltas. Useful if you test whether your top "added" genes are m6A/export-enriched.
- **Likely committee question.** *"Your model has no notion of m6A — could gene-specific export rates confound your delta?"* (Answer: the delta is a steady-state N:C summary integrating export rate; you don't claim to infer mechanism, but export biology is *why* it's gene-specific.)

### Xia et al. 2019 — MERFISH subcellular compartmentalization
**PNAS** 116(39):19490–19499 (2019). DOI [10.1073/pnas.1912459116](https://www.pnas.org/doi/10.1073/pnas.1912459116) · [PMC6765259](https://pmc.ncbi.nlm.nih.gov/articles/PMC6765259/)

- **Key findings.** Single-molecule imaging of ~10,000 genes identified **1,488 nuclear-enriched genes** — enriched for lncRNAs, mRNAs, and especially **intron-retaining transcripts** (unspliced RNA can't be exported) — and ~1,000 ER-enriched genes; nuclear enrichment is cell-cycle-dynamic.
- **Connection.** Orthogonal imaging ground-truth list of nuclear- vs cytoplasm-enriched genes to benchmark your removed/added sets against. The **intron-retention** point matters: nuclear-retained transcripts often carry retained introns that snRNA captures but scRNA doesn't — a direct predictor of negative delta.
- **Likely committee question.** *"Is your delta partly just intron-retention / unspliced RNA trapped in the nucleus? Have you separated that from genuine cytoplasmic localization?"*

### Fazal et al. 2019 — APEX-seq atlas of subcellular RNA localization
**Cell** 178(2):473–490.e26 (2019). DOI [10.1016/j.cell.2019.05.027](https://www.cell.com/cell/fulltext/S0092-8674(19)30555-0)

- **Key findings.** Proximity-labeling map of RNA across nine subcellular locations (nucleus, nucleolus, lamina, pore, cytosol, ER, mito…) in living cells; reveals sequence rules and nuclear vs cytosolic partitioning.
- **Connection.** The genome-wide "which compartment does each RNA live in." Removed genes should map to nuclear/lamina/pore; added genes to cytosol/ER (ribosomal, secretory, synaptic). The paper to cite for "my delta respects a known compartment map" — directly answers the "is it a global ribosomal vector?" critique.
- **Likely committee question.** *"APEX-seq was done in cell lines, not neurons — how do you know the assignments transfer to mouse cortical neurons?"* (Have Perez 2021 / Zappulo 2017 ready.)

### Zappulo et al. 2017 — RNA localization drives the neurite proteome
**Nature Communications** 8:583 (2017). DOI [10.1038/s41467-017-00690-6](https://www.nature.com/articles/s41467-017-00690-6)

- **Key findings.** Subcellular RNA-seq + ribosome profiling + proteomics of soma vs neurites shows the neurite-enriched *proteome* is explained better by local **mRNA localization** than protein transport — transcripts are shipped to processes and translated locally.
- **Connection.** Establishes, in neurons, that a large cytoplasmic/process-localized mRNA population exists and matters — the biological identity of your "added" cytoplasmic/synaptic transcripts, and evidence the recovery is neuronally meaningful, not generic.
- **Likely committee question.** *"You don't resolve dendrites — how does dendritic localization enter your cell profile at all?"* (Answer: scRNA captures the whole cytoplasm including dendritic mRNA; these are exactly the transcripts absent from the nucleus → positive delta.)

### Tushev et al. 2018 — alternative 3′UTRs set localization
**Neuron** 98(3):495–511.e6 (2018). DOI [10.1016/j.neuron.2018.03.030](https://www.sciencedirect.com/science/article/pii/S0896627318302368) · [PubMed](https://pubmed.ncbi.nlm.nih.gov/29656876/)

- **Key findings.** Genes generate multiple 3′UTR isoforms whose usage sets subcellular localization, stability, plasticity; longer 3′UTR isoforms often preferentially localize to neuropil. Localization is an isoform-level, cis-encoded property.
- **Connection.** A resolution caveat and stress test: localization can differ between isoforms of the *same gene*, so a gene-level delta averages over isoforms. Acknowledge the limit; note as future work, or show the delta still tracks gene-level cytoplasmic bias.
- **Likely committee question.** *"Localization is 3′UTR-isoform-specific — isn't a gene-level model blind to the actual localization signal?"* (Answer: gene-level N:C bias is still real and dominant for lncRNAs/intron-retained; isoform resolution is a stated limitation.)

### Perez et al. 2021 — single-neuron soma vs dendrite transcriptomes
**eLife** 10:e63092 (2021). DOI [10.7554/eLife.63092](https://elifesciences.org/articles/63092) · [PubMed](https://pubmed.ncbi.nlm.nih.gov/33404500/)

- **Key findings.** Mechanically separates and sequences soma vs dendrites of individual mouse hippocampal neurons (incl. cell-type-specific GABAergic interneurons), yielding compartment-resolved single-neuron transcriptomes and a partly cell-type-specific dendritic mRNA catalog.
- **Connection.** The closest *experimental* analog to your computational task — an empirically measured soma/dendrite split at single-neuron, cell-type resolution. Validate your added dendritic/cytoplasmic genes against it; its cell-type specificity motivates your conditional (cell-type-aware) design.
- **Likely committee question.** *"Is your delta cell-type-specific, and does it match measured dendritic transcriptomes like Perez 2021 for the corresponding cortical types?"*

**Optional add-ons (verified):** Glock et al. 2021 (PNAS, compartment translatomes — also in Theme 2); **Fazal et al. 2024**, "Mechanistic insights into the basis of widespread RNA localization," *Nat Cell Biol* 26:1088–1099, DOI [10.1038/s41556-024-01444-5](https://www.nature.com/articles/s41556-024-01444-5) — recent mechanistic synthesis for the "recent" bucket.

## Databases / datasets

- **[lncATLAS](https://lncatlas.crg.eu/)** — cytoplasmic/nuclear relative concentration (CN-RCI) for lncRNAs. **Gives the nuclear-retention score for Malat1 and Neat1** directly.
- **[RNALocate v2](http://www.rnalocate.org/)** — curated subcellular localization for tens of thousands of RNAs, human + mouse.
- **APEX-seq data (Fazal 2019)** — nuclear vs cytosol/ER lists to benchmark the sign of your delta.
- **MERFISH subcellular (Xia 2019)** — the 1,488 nuclear-enriched (incl. intron-retained) gene list.
- **Dendritic/neuropil transcriptomes** — Cajigas 2012 (yours), Tushev 2018, Perez 2021, and the **Middleton/Farris** dendritic-isoform catalog (*BMC Biology* 2019, [10.1186/s12915-019-0630-z](https://bmcbiol.biomedcentral.com/articles/10.1186/s12915-019-0630-z)).
- **ENCODE subcellular fractionation RNA-seq** — bulk N:C ratio per gene; a "does my delta correlate with measured N:C fractionation?" figure.
- **Allen Brain Cell Atlas MERFISH (Yao/Zhang 2023, Nature 624)** — your own data source; in-situ subcellular signal for cell-type-specific N:C checks in MOp.

## Critiques this theme arms you against

1. **"Global ribosomal vector, or gene-specific localization?"** → Fazal + Xia compartment maps and lncATLAS: show the delta's rank correlates with independently-measured per-gene N:C ratios, and is negative *specifically* for paraspeckle/speckle lncRNAs and intron-retained genes — not a uniform shift.
2. **"Retention is cell-state/cell-cycle dependent — is your delta state-confounded?"** → Concede; argue sn/sc are matched steady-state populations; test delta stability across states/types.
3. **"Localization is isoform-specific (Tushev) — a gene-level model can't capture it."** → Concede resolution limit; show gene-level N:C bias dominates for the driving classes.
4. **"Export biology (m6A) is an unobserved confound."** → Frame delta as a steady-state readout integrating export kinetics.
5. **"Compartment atlases are cell lines, not neurons."** → Corroborate with Zappulo, Perez, Glock, ABC-Atlas MERFISH.
6. **"Neuron vs glia differ in N:C ratio — does your conditional model handle that?"** → The cell-type conditioning is motivated precisely by these documented differences.

---
---

# Theme 4 — Multi-Modal Single-Cell Integration Methods

The framing theme. Most integration methods treat cell/nucleus as a **batch effect to remove**; yours learns a **directed biological transformation**. You must articulate the distinction crisply — and know both the methods you *contrast* with and the neighbors your method *resembles*.

## Reading order

**Group A — batch-removal / integration (the contrast: modality as nuisance to erase)**
1. **Xu 2021, scANVI** — read first; literally "conditional scVI with a label," the naive baseline a committee will propose against you.
2. **Luecken 2022, scIB benchmark** — the metrics bible (kBET, iLISI, ASW…); where you learn why mixing metrics are the *wrong* objective for translation.
3. **Heumos 2023, best practices** (framing) — field consensus that integration = removing technical variation.

**Group B — directed-transformation / generative-prediction (the neighbors: a learned biological map)**
4. **Lotfollahi 2019, scGen** — the key analogy: latent vector arithmetic (δ = mean_perturbed − mean_control), the discrete precursor of your learned modality shift.
5. **Lotfollahi 2023, CPA** — replaces scGen's hand-computed δ with *learned, conditioned* perturbation embeddings — structurally close to your scalar-conditioned cVAE.
6. **Bunne 2023, CellOT** — learns an explicit transport map control→perturbed; "directed transformation" in its purest form.
7. **Wu 2021, BABEL** — cross-modality translation (ATAC↔RNA) via shared encoder/decoder; your architectural template across omic layers.
8. **Cao 2024, scButterfly** — current SOTA cross-modality translation (dual VAEs + alignment); your most direct methodological neighbor to benchmark against.
9. *(optional)* **Yang/Zhang 2022, Polarbear** — semi-supervised cross-modality translation; the "scarce paired data" precedent.

## The missing papers

### GROUP A — batch-removal (the contrast)

**Xu et al. 2021 — scANVI.** *Probabilistic harmonization and annotation of single-cell transcriptomics data with deep generative models.* **Mol Syst Biol** 17(1):e9620 (2021). DOI [10.15252/msb.20209620](https://pmc.ncbi.nlm.nih.gov/articles/PMC7829634/)
- **Key findings.** Semi-supervised extension of scVI: adds cell-type labels as a latent variable while treating batch as a covariate to integrate out; harmonizes datasets and transfers annotations. The canonical "condition scVI on a covariate."
- **Connection — CONTRAST (the sharpest).** scANVI conditions on batch/label to *remove* their effect and mix cells; you condition on modality to *apply* a directed nucleus→cell shift and *generate* the counterpart. Same mechanism, opposite objective.
- **Likely committee question.** *"Why isn't your method just scANVI/scVI with a modality label? What makes yours a transformation rather than a correction?"*

**Luecken et al. 2022 — scIB benchmark.** *Benchmarking atlas-level data integration in single-cell genomics.* **Nature Methods** 19(1):41–50 (2022). DOI [10.1038/s41592-021-01336-8](https://pubmed.ncbi.nlm.nih.gov/34949812/)
- **Key findings.** Benchmarks 16 methods across 13 tasks; formalizes the two-axis evaluation — *batch-mixing* (kBET, iLISI, PCR, graph connectivity) vs *bio-conservation* (ARI, NMI, ASW, cLISI), combined 40/60. Defines "good integration" = mix batches while preserving cell type.
- **Connection — EVALUATION.** The paper that defines the metrics you must argue are *inappropriate as your objective*. High iLISI/kBET reward making nucleus and cell indistinguishable — for you, failure.
- **Likely committee question.** *"How do you evaluate success if not with standard integration metrics? Show your metric isn't rewarding memorization or mode collapse."* (Track: held-out matched-cell reconstruction, correlation, MMD to the true scRNA distribution, DE-gene recovery.)

**Heumos et al. 2023 — best practices.** *Best practices for single-cell analysis across modalities.* **Nat Rev Genetics** 24(8):550–572 (2023). DOI [10.1038/s41576-023-00586-w](https://pmc.ncbi.nlm.nih.gov/articles/PMC10066026/)
- **Key findings.** Field-consensus review codifying "data integration" as removal of batch/technical variation to build atlases.
- **Connection — FRAMING.** Cite to define crisply the paradigm your directed-transformation framing departs from. For positioning, not a competitor.
- **Likely committee question.** *"Where does your method sit in the standard pipeline — is it integration, or something else?"* (Neither correction nor label transfer; a generative cross-condition predictor.)

### GROUP B — directed-transformation (the neighbors)

**Lotfollahi, Wolf & Theis 2019 — scGen.** *scGen predicts single-cell perturbation responses.* **Nature Methods** 16(8):715–721 (2019). DOI [10.1038/s41592-019-0494-8](https://pubmed.ncbi.nlm.nih.gov/31363220/)
- **Key findings.** VAE + latent vector arithmetic: δ = (mean latent stimulated − mean latent control), added to control latents to predict perturbed state; generalizes to held-out types/species. Biologically meaningful transformations ≈ linear directions in a structured latent space.
- **Connection — NEIGHBOR (foundational analogy).** Your modality shift *is* scGen's δ, but learned and conditioned rather than a fixed difference-of-means. Present your method as "scGen's latent arithmetic generalized from a fixed vector to a learned conditional map." Your cell-type-mean baseline is essentially the fixed-δ approximation — beating it is the empirical point.
- **Likely committee question.** *"How is your cVAE different from scGen's latent arithmetic — couldn't you just compute a mean nucleus→cell δ and add it? What does learned conditioning buy you?"*

**Lotfollahi et al. 2023 — CPA.** *Predicting cellular responses to complex perturbations in high-throughput screens.* **Mol Syst Biol** 19(6):e11517 (2023). DOI [10.15252/msb.202211517](https://pubmed.ncbi.nlm.nih.gov/37154091/)
- **Key findings.** Compositional Perturbation Autoencoder disentangles a basal latent from *learned additive* perturbation/covariate embeddings (drug, dose, line), composing embeddings for unseen combinations; supports OOD prediction, dose interpolation, uncertainty.
- **Connection — NEIGHBOR (closest architectural cousin).** CPA is "scGen's δ made a learned, conditioned embedding added in latent space" — structurally what your scalar-modality-conditioned cVAE does. Frame yours as a two-condition CPA specialized to the nucleus↔cell axis.
- **Likely committee question.** *"Why a single scalar modality condition rather than a learned embedding à la CPA? Does that limit you to two modalities, and would a CPA-style embedding generalize better?"*

**Bunne et al. 2023 — CellOT.** *Learning single-cell perturbation responses using neural optimal transport.* **Nature Methods** 20(11):1759–1768 (2023). DOI [10.1038/s41592-023-01969-x](https://pmc.ncbi.nlm.nih.gov/articles/PMC10630137/)
> ⚠️ A *different* same-issue paper ("Neural optimal transport predicts perturbation responses…", `s41592-023-01968-y`) sits back-to-back. CellOT = **-01969-x**. Don't cross them.
- **Key findings.** Learns an explicit deterministic **transport map** unperturbed→perturbed via input-convex neural nets (Monge OT), predicting single-cell responses for held-out patients/perturbations; maps *individual* cells, not just means.
- **Connection — NEIGHBOR (purest "directed transformation").** CellOT *learns a map between two distributions* rather than mixing them — precisely your claim (nucleus distribution → cell distribution). Cite as prior art that learned transport, not batch alignment, is the right formalism; position your cVAE as the amortized-generative counterpart.
- **Likely committee question.** *"OT gives an explicit, mass-preserving map with guarantees. Why a VAE? What does your alignment term correspond to in OT terms, and do you have any identifiability guarantee OT gives for free?"*

**Wu et al. 2021 — BABEL.** *BABEL enables cross-modality translation between multiomic profiles at single-cell resolution.* **PNAS** 118(15):e2023070118 (2021). DOI [10.1073/pnas.2023070118](https://pubmed.ncbi.nlm.nih.gov/33827925/)
- **Key findings.** Modality-specific encoders/decoders around a *shared latent*, trained so any encoder's latent decodes into any modality — enabling RNA↔ATAC prediction at single-cell resolution.
- **Connection — NEIGHBOR (architectural template).** BABEL's shared-latent, cross-decoder design is the multi-omic analog of your shared encoder/decoder; you translate *within* RNA between two sampling modalities, where matched correspondence and a directed shift are more defensible than across ATAC/RNA. Justifies your shared-encoder/decoder choice.
- **Likely committee question.** *"BABEL already translates modalities with shared encoders/decoders — what's novel beyond applying it to nucleus vs cell? Why is nucleus→cell well-posed when the two aren't independent measurements of the same cell?"*

**Cao et al. 2024 — scButterfly.** *scButterfly: a versatile single-cell cross-modality translation method via dual-aligned variational autoencoders.* **Nature Communications** 15:2973 (2024). DOI [10.1038/s41467-024-47418-x](https://pmc.ncbi.nlm.nih.gov/articles/PMC10998864/)
> ⚠️ First authors + bibliographic fields confirmed; full author list beyond first authors not exhaustively verified — flag before citing by name.
- **Key findings.** Two modality-specific VAEs + a **dual alignment** scheme (translator + discriminator/alignment losses) to translate RNA/ATAC/protein; reports SOTA translation, robustness in paired and unpaired settings, downstream utility.
- **Connection — NEIGHBOR (most direct competitor).** Its dual-VAE + alignment-loss structure closely mirrors your shared encoder/decoder + latent-alignment term. Benchmark against it (or at least cite as SOTA) and sharpen why your *single-model, scalar-conditioned, asymmetric/directed* design differs from scButterfly's symmetric bidirectional translation.
- **Likely committee question.** *"scButterfly already does aligned dual-VAE translation with strong results — what does your architecture add, and have you compared against it rather than only scVI and a cell-type-mean baseline?"*

**Yang/Zhang 2022 — Polarbear** *(optional).* *Multimodal Single-Cell Translation and Alignment with Semi-Supervised Learning.* J Comput Biol / RECOMB 2022. DOI [10.1089/cmb.2022.0264](https://pmc.ncbi.nlm.nih.gov/articles/PMC9700358/)
- **Key findings.** Trains modality-specific AEs on large unpaired single-assay data + a smaller co-assayed set semi-supervised to learn RNA↔ATAC translation, beating paired-only training.
- **Connection — NEIGHBOR (secondary).** Precedent for training a translator when matched cell↔nucleus pairs are scarce. Include only for the semi-supervised angle.
- **Likely committee question.** *"Do you have matched nucleus–cell pairs? If not, how is the directed map supervised — and if unpaired, what stops it learning an arbitrary bijection?"*

## The key argument you must be able to make

**Batch-effect removal vs directed biological transformation — two different objectives.**

- **Batch removal (scVI, scANVI, Harmony, Seurat, Scanorama):** cell/nucleus is a *nuisance covariate*; goal is a latent space where the two modalities are *indistinguishable* — the signal is marginalized out. Success = maximal mixing subject to preserving type. Symmetric, non-directional — you can't ask it to "produce the cell version of this nucleus."
- **Directed transformation (yours; scGen, CPA, CellOT, BABEL, scButterfly):** nucleus→cell is a *biological map to learn and apply*; the modality signal is the thing modeled, retained, conditioned on, and used to *generate*. Success = accurate reconstruction of the true scRNA profile from a snRNA input.

**Why the metrics differ — the crux.** Integration metrics (kBET, iLISI, batch-ASW; Luecken 2022) reward *making the modalities inseparable* — a method that maximizes iLISI has, by construction, **erased the nucleus→cell difference you're trying to reproduce.** One sentence to have ready: *"Mixing metrics reward mode collapse of the two modalities — precisely the signal my method must preserve and transform, so integration benchmarks measure the wrong thing for a translation task."* The right evaluation is **predictive**: held-out target-modality reconstruction fidelity (per-gene correlation, MMD/Wasserstein, DE-gene concordance, cell-type-resolved accuracy) — the scGen/CPA/CellOT lineage, not scIB.

Your two baselines are the *right* comparison set: **scVI = the batch-removal null; cell-type-mean = the fixed-global-δ / no-cell-state-dependence null.** Beating both shows you need (a) directional conditioning and (b) a learned, state-dependent shift.

## Critiques this theme arms you against

- **"Isn't this just scGen?"** → scGen adds a *single fixed* difference-of-means vector; your decoder applies a *learned, conditioned, state-dependent* shift. The cell-type-mean baseline you beat *is* the fixed-δ approximation.
- **"Isn't this just conditional scVI/scANVI with a modality covariate?"** → Same mechanism, opposite use: scVI conditions to integrate the covariate *out*; you condition to *drive generation* along it. Your model must output a *different, cell-like* profile — a query scVI's batch label is never trained to answer.
- **"Why doesn't the alignment term collapse the two modalities (destroying the transformation)?"** → The sharpest technical critique. Argue alignment pulls *shared biological structure* (identity, state) together while the modality condition + reconstruction likelihood preserve/route modality-specific signal through the decoder. **Ablation:** remove the condition → outputs collapse; keep it → directed shift survives.
- **Identifiability / well-posedness.** Nucleus and cell aren't independent measurements of the same molecules, so "translation" is a *distributional* map, not a per-molecule bijection. With limited/no pairs, what constrains it to the biologically correct map vs an arbitrary aligning bijection? Name your supervision signal and any uniqueness assumption (CellOT's OT uniqueness; Polarbear's paired semi-supervision).

---
---

# Theme 5 — Validation of Predicted Cytoplasmic Transcriptome

The synthesis theme. You have **no per-cell paired nucleus+cytoplasm ground truth** for human brain (and little for mouse). This assembles the rare paired datasets, orthogonal cytoplasm/translation assays, localization databases, and the imputation-benchmarking literature that separates rigorous validation from circular.

## Reading order

1. **Bahar Halpern 2015** — nuclear retention is real and gene-specific (the biology your model learns). *(Also Theme 3.)*
2. **Bakken 2018 (yours)** — canonical matched sn/sc in cortex; what a "validated" nucleus→cell relationship looks like at cell-type level.
3. **Abdelmoez 2018 (SINC-seq)** — the one method giving *per-cell* paired nuclear+cytoplasmic RNA. *(Also Theme 2.)*
4. **Fazal 2019 (APEX-seq)** — subcellular atlas; orthogonal localization signal. *(Also Theme 3.)*
5. **RNALocate v2 (Cui 2022) + lncATLAS (Mas-Ponte 2017)** — the two localization DBs, read as a pair.
6. **Scala 2021 (yours)** — Patch-seq in *mouse MOp*: orthogonal whole-cell transcriptomes, same taxonomy. *(Also Theme 2.)*
7. **Heiman 2008 (TRAP, yours)** — cell-type translated (cytoplasmic-proxy) mRNA.
8. **Hou 2020** — imputation-benchmarking; the circularity/held-out critique manual. Read before finalizing metrics.
9. **Song 2024 (scDesign3)** — simulation with known ground truth for negative controls. Read last.

## The missing papers/resources

### Abdelmoez et al. 2018 — SINC-seq *(the per-cell ground truth)*
**Genome Biology** 19:66 (2018). DOI [10.1186/s13059-018-1446-9](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-018-1446-9)
- **What it provides.** Physically separated + separately sequenced nuclear and cytoplasmic RNA of the *same individual cell*; gene-specific N:C correlations carrying physiological state. NanoSINC-seq (*Sci Adv* 2021) adds isoform resolution.
- **Connection — the strongest counter to "no per-cell ground truth."** The one modality where paired nucleus→cytoplasm prediction can be scored *per cell*. Propose SINC-seq (or an equivalent split-compartment assay) as the gold-standard held-out benchmark for per-cell error — directly addressing the aggregation/circularity critique. (Caveat: K562, not brain.)
- **Likely committee question.** *"You evaluate on distributions and means — is there any assay that lets you check a single cell's predicted cytoplasm? Why not benchmark on it?"*

### RNALocate v2.0 — Cui et al. 2022
**Nucleic Acids Research** 50(D1):D333–D339 (2022). DOI [10.1093/nar/gkab825](https://academic.oup.com/nar/article/50/D1/D333/6374157) · DB [rnalocate.org](http://www.rnalocate.org/)
- **What it provides.** Manually curated + predicted RNA subcellular localization, >210K entries, many RNA types, human + mouse, with experimental-evidence annotation.
- **Connection.** Turn it into a binary/graded cytoplasm-vs-nucleus label per gene, then test that your delta's sign predicts the label (ROC/PR on "is this cytoplasmic"). Converts "makes sense" into a measurable concordance statistic.
- **Likely committee question.** *"How well does the added-gene set match an independent curated localization database — quantitatively, with an AUC, not anecdotes?"*

### lncATLAS — Mas-Ponte et al. 2017
**RNA** 23(7):1080–1087 (2017). DOI [10.1261/rna.060814.117](https://rnajournal.cshlp.org/content/23/7/1080) · [PMC5473142](https://pmc.ncbi.nlm.nih.gov/articles/PMC5473142/) · DB [lncatlas.crg.eu](https://lncatlas.crg.eu/)
- **What it provides.** Per-gene **cytoplasmic-to-nuclear relative concentration (CN-RCI)** for lncRNAs (+ protein-coding references) across human cell lines, from ENCODE fractionation — a *continuous* localization coordinate.
- **Connection.** The cleanest quantitative test: added genes should skew cytoplasmic (high CN-RCI), removed genes nuclear (low CN-RCI). Correlate your per-gene delta against CN-RCI — a monotonic relationship is direct, independent evidence you move transcripts the right way. Especially strong for the lncRNA fraction (Malat1/Neat1).
- **Likely committee question.** *"lncRNAs are famously nuclear — does your model correctly leave them out of the predicted cytoplasm, per lncATLAS CN-RCI?"*

### Fazal et al. 2019 — APEX-seq *(orthogonal, non-sequencing)*
**Cell** 178(2):473–490.e26 (2019). DOI [10.1016/j.cell.2019.05.027](https://www.cell.com/cell/fulltext/S0092-8674(19)30555-0)
- **Connection.** Independent proximity-labeling ground truth for *where* a transcript lives. Added genes should enrich APEX cytosol/ER; removed genes the nuclear/nucleolar fractions. Because it's a completely different technology (not derived from Allen snRNA), concordance is strong *non-circular* validation.
- **Likely committee question.** *"Your localization signal could be baked into training. What orthogonal, non-sequencing measurement confirms the added genes are genuinely cytoplasmic?"*

### Hou et al. 2020 — imputation benchmarking *(the critique manual)*
**Genome Biology** 21:218 (2020). DOI [10.1186/s13059-020-02132-x](https://link.springer.com/article/10.1186/s13059-020-02132-x)
- **Key findings.** Benchmarks scRNA-seq imputation and warns it can **introduce spurious gene–gene correlations and inflate downstream signal**, sometimes worse than no imputation; emphasizes held-out (masked-entry) evaluation and biologically-orthogonal references over self-consistency.
- **Connection.** Directly informs the risk that your GO-enrichment-of-the-delta and QQ metrics look good because the model *manufactured* structure. Preempt with (a) a **no-transformation / identity baseline** and simple linear baselines, (b) **held-out cell types and genes**, (c) a **scrambled-delta negative control**.
- **Likely committee question.** *"Imputation is notorious for inventing correlations — how do you know your delta's GO enrichment isn't an artifact of the model smoothing data toward itself?"*

### Song et al. 2024 — scDesign3 *(known-ground-truth sandbox)*
**Nature Biotechnology** 42(2):247–252 (2024). DOI [10.1038/s41587-023-01772-1](https://www.nature.com/articles/s41587-023-01772-1) · [PMC11182337](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11182337/)
- **Key findings.** Unified simulator generating realistic single-cell/spatial data with *known* generating parameters + a principled model-assessment framework (goodness-of-fit, calibration).
- **Connection.** Build a sandbox: simulate paired nucleus/cytoplasm with a *specified* transformation, train the cVAE, confirm it recovers the planted delta — and, crucially, reports *no change* when the truth is identity (negative control). This is how you show your distributional metrics have power and don't pass on noise.
- **Likely committee question.** *"On simulated data where you know the true nucleus→cytoplasm map, does your model recover it — and correctly report 'no change' when the truth is identity?"*

### Supporting resources (cite as needed)

- **⚠️ The sharpest external attack — "sampling, not biology":** *Differences in molecular sampling and data processing explain variation among single-cell and single-nucleus RNA-seq experiments.* **Genome Research** 34(2):179–188 (2024). DOI [10.1101/gr.278253.123](https://genome.cshlp.org/content/34/2/179.full). Argues sc/sn differences are largely *technical sampling*, not biology — you must address this head-on. *(First-author name not independently verified — check before citing by author.)*
- **Siletti et al. 2023** — *Transcriptomic diversity of cell types across the adult human brain.* **Science** 382:eadd7046. DOI [10.1126/science.add7046](https://www.science.org/doi/10.1126/science.add7046). The definitive human brain atlas — **nucleus-only**, which is *why* your predictor matters for human tissue. Use to frame motivation and the "no human cytoplasm ground truth" gap.
- **Eraslan et al. 2022** — GTEx snRNA-seq (also Theme 1); another large nucleus-only human resource underscoring the scarcity of matched human cytoplasm.

## A concrete validation menu (a slide you can present)

| # | Strategy | Data / resource | What it tests | Independence |
|---|----------|-----------------|---------------|--------------|
| 1 | **Per-cell paired benchmark** | SINC-seq / NanoSINC-seq | Per-cell nucleus→cytoplasm error (not just means) | Fully independent modality |
| 2 | **Matched cell-type reference** | Bakken 2018 matched sc/sn | Delta reproduces known sc−sn differential genes | Independent dataset |
| 3 | **Patch-seq overlap** | Scala 2021 mouse M1 | Predicted cytoplasm moves *toward* whole-cell, same region+taxonomy | Orthogonal assay |
| 4 | **TRAP overlap** | Heiman 2008 + cell-type TRAP | Added genes enriched in translated/cytoplasmic pool | Orthogonal assay |
| 5 | **Localization (continuous)** | lncATLAS CN-RCI | Per-gene delta correlates with cyto:nuclear ratio | Independent (ENCODE fractionation) |
| 6 | **Localization (label)** | RNALocate v2, APEX-seq | AUC for "added = cytoplasmic, removed = nuclear" | Independent DB / proximity labeling |
| 7 | **Nuclear-retention prior** | Bahar Halpern 2015 | Removed genes match known nuclear-retained set | Independent biology |
| 8 | **Held-out cell types & genes** | your own Allen data | Generalization, not memorization | Internal, leakage-controlled |
| 9 | **Cross-protocol** (done) | 10x v3 vs other chemistries | Robustness to technical variation | Internal |
| 10 | **Simulation / negative control** | scDesign3 | Recovers planted delta; null when truth = identity | Ground truth by construction |
| 11 | **Circularity / baseline audit** | Hou 2020 framework | Delta beats identity + linear baselines; scrambled-delta control | Methodological guardrail |

**Priority for the exam:** lead with **5–7 and 10–11** (feasible now, non-circular, no scarce data); position **1–4** as "gold-standard experiments I would run/propose"; use **8–9** as the internal-rigor floor.

## Critiques this theme arms you against

- **"There is no per-cell ground truth."** → SINC-seq as the one per-cell paired assay; simulation where per-cell truth is known. Concede that on real brain you validate at cell-type resolution, and justify that as the honest ceiling.
- **"Your evaluation is circular."** → Externally-derived references never in training: lncATLAS, RNALocate, APEX-seq, TRAP, Bahar Halpern. Concordance can't be an artifact of the Allen snRNA the model learned from.
- **"Distributional metrics can pass while per-cell predictions are wrong."** → Held-out per-gene/per-cell scoring + scDesign3 negative controls (metrics *fail* on scrambled deltas) + identity/linear baselines.
- **"Biology or protocol/sampling artifact?"** → Answer the Genome Research 2024 paper head-on: show the delta aligns with *biological* retention/localization axes, not technical covariates (length, GC, depth), and generalizes across chemistries.
- **"Does it generalize to human / disease?"** → Human brain is overwhelmingly nucleus-only (Siletti 2023, Eraslan 2022) — the *motivation* — but matched human cytoplasm is scarce; be explicit that human validation is currently cell-type-level + localization DBs, and propose human SINC-seq/Patch-seq as the definitive future test.

---
---

# Synthesis 1 — Papers that span multiple themes

These are the connective-tissue papers a committee uses to test whether you see your project whole. Know each one's *double duty*.

- **Bakken 2018** *(Themes 1 & 5, yours)* — matched sn/sc cortical comparison. Theme 1: defines the biases; Theme 5: the descriptive gene-list target your delta must reproduce. Expect *"what does a generative model add beyond Bakken's lists?"*
- **SINC-seq / Abdelmoez 2018** *(Themes 2, 3 & 5)* — per-cell nuclear/cytoplasmic split. The rare place all three converge: orthogonal cytoplasmic assay (2), empirical N:C partition biology (3), and the only per-cell validation (5).
- **Fazal 2019 APEX-seq** *(Themes 3 & 5)* — subcellular atlas as both localization ground truth (3) and non-circular validation (5).
- **Scala 2021** *(Themes 2 & 5, yours)* — Patch-seq in *your exact region/taxonomy*; orthogonal cytoplasmic profiling (2) and a clean cell-type validation target (5).
- **Heiman 2008 / TRAP** *(Themes 2 & 5, yours)* — translated cytoplasmic mRNA; orthogonal assay (2) and validation (5).
- **Glock 2021** *(Themes 2 & 3)* — compartment translatomes; validates the synaptic arm of the delta from both the profiling-method and localization-biology sides.
- **Bahar Halpern 2015** *(Themes 3 & 5)* — nuclear retention as the *reason* nucleus≠cytoplasm (3) and the retention-prior validation (5).
- **Eraslan 2022 GTEx** *(Themes 1 & 5)* — sc/sn biases + dissociation signatures (1); large nucleus-only human resource motivating validation (5).
- **lncATLAS / RNALocate** *(Themes 3 & 5)* — localization biology (3) and the quantitative concordance test (5).
- **CellBender (Fleming 2023)** *(Themes 1 & 4)* — ambient-RNA confound (1) *and* a generative-VAE neighbor on the same data (4).

**One-line map to recite:** *Themes 1 and 3 supply the two competing explanations for my delta — technical (dissociation/ambient) vs biological (localization/retention); Themes 2 and 5 supply the orthogonal data to adjudicate between them; Theme 4 defines why my method is a directed transformation rather than the batch removal that would erase the very signal Themes 1–3 are arguing about.*

---

# Synthesis 2 — The hardest critiques, and how Gerstein will attack this

Your five poster "landmines" already anticipate the field-level attacks (no per-cell ground truth; global-ribosomal-vector; Fos/Arc dissociation artifact; "so what"; glia-biased cells). Below are the **three deepest cross-cutting critiques** plus **the specific lens Gerstein brings** — because a Gerstein-lab qual is not just "is the method sound," it's "does this connect to variant → regulation → expression → network → disease."

### The three deepest cross-cutting critiques

1. **Biology vs technical confound (Themes 1 + 5).** The single most dangerous line: your delta could be *dissociation stress + differential ambient RNA + sampling*, not cytoplasmic biology. The Genome Research 2024 "sampling not biology" paper is the sharp form. **Your defense is convergent orthogonal evidence:** the delta aligns with lncATLAS CN-RCI, RNALocate labels, APEX-seq compartments, and nuclear-retention priors — none of which are dissociation artifacts — while *not* enriching for IEG/stress modules (van den Brink, Marsh). Run the artifact-gene-set audit *before* the exam so the answer is a figure, not a promise.
2. **Circularity of validation (Theme 5).** If the localization signal is in the training data, GO-enrichment of the delta is self-fulfilling. **Defense:** externally-derived references + a scrambled-delta negative control + scDesign3 identity-truth control + identity/linear baselines (Hou 2020).
3. **Identifiability / "why doesn't alignment collapse the modalities" (Theme 4).** Without per-molecule pairs, what pins the map to the biologically correct one? **Defense:** the ablation (remove the modality condition → outputs collapse), the directed asymmetry vs scGen/scButterfly, and the OT-uniqueness / semi-supervision framing (CellOT, Polarbear).

### The Gerstein lens — questions to expect from your PI specifically

Gerstein evaluates brain single-cell work against the chain **variant → regulation → expression → network → disease → drug target.** Your method sits at "expression," so he will push it *outward* in both directions:

- **Cell-type granularity (his Q1).** *"You have 17 types — are the ones that matter resolved? Excitatory vs inhibitory vs glia are not interchangeable, and your delta should differ between them. Is your nucleus→cell shift cell-type-specific in a way that respects known subtype biology (L2/3 IT vs L5 ET), or is it a class-level average?"* Have the per-cell-type delta correlation (0.918 vs scVI 0.807) and a subtype breakdown ready.
- **"So what, at population scale?" (his Q6 + the publication table).** *"Recovering cytoplasm is a means. What's the disease question? Does imputing cytoplasmic content change which genes come out of a PsychENCODE disease-DE analysis — and does it change any cell-type-specific eQTL or heritability-enrichment result?"* Your poster already gestures at this (PsychENCODE disease DE); make it concrete. This is what moves the project from *Nat Methods* (a clean method) toward *Nat Neuroscience* (a mechanism).
- **Multi-omics / regulation (his Q3).** *"You predict cytoplasmic RNA — but the regulatory story lives in chromatin. Could you condition on, or predict toward, snATAC/Multiome? Does the cytoplasmic gap even matter for a regulatory-network readout, which is nuclear anyway?"* Good answer: the *nuclear* signal is what carries regulatory/eQTL information, so your model's value is orthogonal — restoring the *functional/effector* (cytoplasmic, translated, synaptic) layer that nuclear-only data loses. That reframing is exactly his language.
- **Interpretable ML + simpler baselines (his Q7).** *"Is the cVAE interpretable, and did you beat an interpretable linear baseline before going deep?"* Your cell-type-mean and scVI baselines partly answer this; be ready to add a linear/global-δ baseline (your poster's own TODO) so "the deep model is necessary" is demonstrated, not asserted.
- **Tool / resource (his Q8).** *"What does a wet-lab neuroscientist get — a browser, a table, a pipeline they can run on their own snRNA-seq? Is the code on GitHub, data on Synapse/GEO?"* Have a one-line answer: ImputeVAE as a released tool that takes any brain snRNA-seq (of which there are millions of nuclei, PsychENCODE alone 2.8M) and restores an estimated cytoplasmic layer for downstream DE/marker analysis.
- **Publication framing.** He'll ask where it sits. Honest placement: **methods-tier** now (novel directed-transformation method + benchmark), climbing toward **Nat Neuroscience-tier** *if* you land a disease-mechanism result (cytoplasmic recovery changes a real PsychENCODE DE/eQTL conclusion) at population scale. Naming this yourself signals you've scoped it.

**The meta-point to internalize:** your method is currently framed as *fidelity* (how well do I reproduce the cell?). Gerstein will keep pushing it toward *consequence* (what downstream brain-disease conclusion changes because you did?). Every theme above has a "so what" answer — rehearse the chain from your delta to a PsychENCODE disease-DE or eQTL result, and the exam is yours.

---

## Verification notes

All primary citations were verified against journal/DOI or PMC pages (title, journal, year, DOI). Flags to resolve before you cite by author name: the **Yao 2021 disambiguation** (Theme 1); the **parallel human-neuron TRAP** author list (Theme 2, PMID 33364619); the **scButterfly** full author list (Theme 4); and the **Genome Research 2024 "sampling not biology"** first author (Theme 5). PubMed was intermittently CAPTCHA-blocked during research, so those four author lists should be confirmed from the PMID/DOI — every title, journal, year, and DOI is confirmed.

---

## Appendix — Additional Papers (from initial reading list, not in main guide)

These papers were in the original reading list and provide useful supplementary coverage. Organized by theme.

### Theme 1 additions

- **Habib 2017** — "Massively parallel single-nucleus RNA-seq with DroNc-seq." *Nature Methods*. Early snRNA-seq method with matched comparisons — historical context for the field.
- **Krishnaswami 2016** — "Using single nuclei for RNA-seq to capture the transcriptome of postmortem neurons." *Nature Protocols*. Foundational protocol paper — why nuclei work on frozen/postmortem tissue.
- **Madissoon 2023** — "scRNA-seq assessment of the human lung, spleen, and esophagus tissue stability after cold preservation." *Genome Biology*. Human lung, 10x v3.1 — potential external validation dataset.
- **Wu 2019** — "Advantages of Single-Nucleus over Single-Cell RNA Sequencing of Adult Kidney." *JASN*. Documents cell-type-specific biases in dissociation-based protocols (kidney).

### Theme 2 additions

- **Cadwell 2016** — "Electrophysiological, transcriptomic and morphologic profiling of single neurons using Patch-seq." *Nature Biotechnology*. Original Patch-seq method paper.
- **Dougherty 2010** — "The Disruption of Celf6, a Gene Identified by Translational Profiling of Serotonergic Neurons, Results in Autism-related Behaviors." *J Neurosci*. TRAP applied to serotonergic neurons — example of cell-type-specific translational profiling.
- **Shigeoka 2016** — "Dynamic Axonal Translation in Developing and Mature Visual Circuits." *Cell*. TRAP applied to axonal compartment — directly measures locally translated mRNA in visual cortex.
- **Heiman 2014** — "Cell type-specific mRNA purification by translating ribosome affinity purification (TRAP)." *Nature Protocols*. Updated TRAP protocol reference.

### Theme 3 additions

- **Chen & Bhargava 2013** — "Nuclear retained lncRNAs." *Methods Mol Biol*. Review of nuclear retention mechanisms: chromatin association, paraspeckle formation, nuclear body tethering.
- **Palazzo & Lee 2015** — "Non-coding RNA: what is functional and what is junk?" *Frontiers in Genetics*. Nuclear vs cytoplasmic RNA distribution context.
- **Ainsley 2014** — "Functionally diverse dendritic mRNAs rapidly associate with ribosomes following a novel experience." *Nature Communications*. Activity-dependent translation at synapses — relevant to synaptic enrichment in your delta.
- **Holt & Bhatt 2014** — "Subcellular mRNA Localization in Animal Cells and Why It Matters." *Science*. Broad review of RNA transport mechanisms.
- **Clemson 1996 / later Neat1 work** — Established Neat1/Malat1 as nuclear-retained lncRNAs (see Fox 2018 in main guide for the mechanistic update).
- **Zeisel 2015** — "Cell types in the mouse cortex and hippocampus revealed by single-cell RNA-seq." *Science*. Early atlas showing cell-type-specific expression programs.
- **Mo 2015** — "Epigenomic Signatures of Neuronal Diversity in the Mammalian Brain." *Neuron*. Cell-type-specific epigenomics and differential lncRNA regulation.

### Theme 4 additions

- **Stuart & Butler 2019** — "Comprehensive Integration of Single-Cell Data." *Cell*. Seurat v3 anchoring — the dominant non-deep-learning integration approach.
- **Gayoso 2022** — "A Python library for probabilistic analysis of single-cell omics data." *Nature Biotechnology*. scvi-tools ecosystem (totalVI, MultiVI) — the software framework your scVI baseline uses.
- **Cao & Gao 2022** — "GLUE: A flexible framework for multi-omics data integration." *Nature Biotechnology*. Graph-linked unified embedding — a non-VAE multi-modal method.
- **Gong 2021** — "Cobolt: Integrative analysis of multimodal single-cell sequencing data." *Genome Biology*. Multi-modal VAE alternative.
- **Amodio 2019** — "Exploring single-cell data with deep multitasking neural networks." *Nature Methods*. MAGAN — GAN-based cross-modal translation (contrast with your VAE approach).
- **Yang 2021** — "Multi-domain translation between single-cell imaging and sequencing data using autoencoders." *Nature Communications*. Cross-modal VAE translation precedent.

### Theme 5 additions

- **Scala 2021** — "Phenotypic variation of transcriptomic cell types in mouse motor cortex." *Nature*. Patch-seq in mouse MOp — **same brain region as training data**. Strongest orthogonal validation source (also in Theme 2).
- **Berg 2021** — "Human neocortical expansion involves glutamatergic neuron diversification." *Nature*. Allen Institute Patch-seq in **human** cortex — most promising human validation dataset.
