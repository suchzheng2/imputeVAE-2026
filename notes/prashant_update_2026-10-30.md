# ImputeVAE: status, storyline, and gaps (update for Prashant, 2026-10-30)

Prepared for restarting weekly meetings. Sources: `notes/paper_storyline_2026-10-23.md`,
`notes/benchmark_review_2026-10-09.md` §5, `notes/libsize_covariate_review_2026-10-09.md` §6,
`notes/experiments/bakken2018_holdout/README.md`, `notes/experiments/cross_dataset_ding/README.md`.
All benchmark numbers are from the split-aware held-out test partition unless marked.

## 1. Since the September meeting

Your two requests from that meeting are done:

| Request | Result |
|---|---|
| TRAP/RiboTag: map L5 populations by projection evidence, not by name; drop L4 | Every L5 population mapped from the source paper (ET vs IT); L4 removed; Rbp4 pool excluded (no single matching subclass); external marker check (Yao 2021 MOp) agrees for all six. H2 re-run: imputed 0.73 vs raw nuclei 0.43, ceiling 0.74. Holds over 19 populations incl. excitatory and glia. |
| Variance: per-gene distribution and rank for L5 IT and Oligo, not the median | Distribution is stretched to a lower scale, not collapsed. Spearman(var real, var imputed) = 0.92 (L5 IT), 0.93 (Oligo). |

Four further results, in order of importance:

1. **Benchmark redone on a proper held-out split with seed replicates (Oct 9-12).** Every method on the same
   train/valid/test manifest, three cVAE seeds. Label-free: ImputeVAE 0.900 ± 0.002 gene correlation vs scVI /
   scANVI 0.86; energy distance 6.8 vs 9.2 (floor 0.6); within-type coexpression 0.79 vs 0.72. With labels:
   cell-type mean 0.981 (the ceiling), scGen 0.975, scANVI with per-type library 0.962. The earlier all-data
   comparison is retired.
2. **Cause of the low within-type variance found and removed (Oct 16-23).** The latent uses 5 of 32 dimensions,
   and one dimension (library size) carries 95 % of the imputed within-type variance. Adding library size as an
   explicit covariate (encoder + decoder) takes its share of imputed variance from 85 % to 0 %; what remains sits on
   2-3 cell-identity axes. Lowering the JSD weight 10x changes nothing, and 8- or 4-dim latents reproduce the
   result, so this is objective-driven, not capacity-driven. Consequence: the per-cell variance claim is withdrawn
   and replaced by a characterisation of what a nucleus can tell you about its cytoplasm at 10x depth.
3. **Label-free decode at the labelled ceiling (Oct 23-27).** With the covariate model, decoding at the library
   of the type predicted from the model's own latent gives 0.962 / R² 0.91 / energy 2.6 with no label at inference
   (labelled 0.970; no labels anywhere 0.921). On unseen v2 chemistry: 0.90 label-free (0.93 labelled) with a
   depth-robust reference classifier for the type (v3 accuracy 0.982). Residual v2 error is Pvalb -> Sst and
   L6 IT <-> L5 IT confusion, not depth.
4. **Independent matched validation (Oct 26).** Bakken 2018 (GSE123454, VISp SMART-seq, 461 nuclei / 445 cells,
   11 matched clusters): direction of the nucleus-to-cell change predicted at the split-half ceiling (sign
   agreement 0.94), gene-level magnitude r 0.58 pooled / 0.73 composition-free vs a reliability ceiling of 0.92.
   Ding 2020 (GSE132044) turned out to have nuclei only (four snRNA-seq chemistries + bulk): imputation moves each
   chemistry's pseudobulk toward the whole-tissue bulk (Pearson 0.27-0.56 -> 0.58-0.65). **Caveat on both:** adding
   the per-type training delta to the nucleus scores the same. On external data the model equals "predict the
   type, add that type's mean shift".

Also: CBB research-in-progress talk scheduled for November; slides and script exist
(`notes/presentation/CBB_RIP_2026-11/`).

## 2. What the evidence supports today

| Claim | Evidence | Strength |
|---|---|---|
| Best label-free nucleus-to-cell translator | 0.90 (0.96 with predicted-type decode) vs scVI/scANVI 0.86; energy distance; coexpression; 3 seeds | solid |
| With labels, simple methods reach the ceiling | CT-mean 0.981, scGen 0.975 | solid, and it bounds the method claim |
| The learned delta is compartment biology and cell-type-specific | ribosomal 100 % sign, nuclear lncRNA 100 % (45 genes), SynGO vs TF controls, neuron/glia ratio 3-5x, scrambled control z = 40 | solid |
| Validated non-circularly | fractionation-seq rho 0.44 (mouse) / 0.40 (human); TRAP/RiboTag 0.43 -> 0.73 (ceiling 0.74), 19 populations | solid, but CT-mean and per-gene regression reach the same numbers |
| Transfers | Bakken 2018 sign at ceiling; Ding 4 chemistries; v2 0.90; human M1 zero-shot (mouse-human r 0.72, donor r 0.99); sorted human populations; kidney core | solid for mouse, plausibility-level for human (no paired truth) |
| Per-cell fidelity | within-type variance is library-size pass-through; latent is 2-3 identity axes + RNA content | claim withdrawn; now a finding about the data |
| Downstream payoff | none yet | **missing** |

The pattern: everything validated is validated at the cell-type level, and at that level the type mean plus a
per-gene map is as good. The model's measurable edge is (i) no labels at inference, (ii) a continuous identity
code, (iii) population-level realism, (iv) chemistry robustness through an explicit output scale.

## 3. Storyline

**Recommended: biology paper with the model as the instrument.** "The nuclear-to-whole-cell transcriptome
transformation is cell-type-specific, low-dimensional and conserved; here is a learned, validated map of it, and
what it changes in nuclei-only atlases."

Findings in order: (1) a cell-type-resolved compartment delta for 10k genes, validated by biochemistry and
ribosome profiling; (2) the transformation is low-rank: type identity plus RNA content explain what a nucleus
profile says about its cytoplasm at 10x depth, and per-nucleus information beyond that is below the sampling
ceiling; (3) conserved to human and partly across tissues; (4) applied to a nuclei-only atlas it recovers
cytoplasmic biology and changes a downstream result. **Item 4 does not exist yet and is the paper.**

Figures: 1 the transformation; 2 model + split-aware benchmark with both groups and the ceiling; 3 non-circular
validation; 4 what a nucleus can and cannot tell you (latent usage, covariate result, paired-r ceiling, per-type
difficulty); 5 generalisation (v2, Bakken, Ding, human M1, kidney); 6 application.

The alternative, a methods paper ("ImputeVAE imputes cells from nuclei"), does not survive the split-aware
benchmark at Nature level: label-using baselines sit at the ceiling and reviewers will ask what a 22M-parameter
model adds over "type mean + regression". It fits Bioinformatics / PLOS CB / Genome Biology with careful framing.

## 4. Gaps, ranked by what they block

1. **No downstream application.** The one result that decides the journal. Candidates:
   (a) a nuclei-only disease DE study re-analysed on imputed profiles; the group's SCORCH object (human snRNA-seq,
   655k nuclei, 33 donors, PFC / insula / ventral striatum, HIV vs control, 37 labels covering all our cortical
   types) is on disk and ready to run with the human-M1 ortholog mapping;
   (b) marker-based annotation of nuclei improving after imputation, on external data;
   (c) a PsychENCODE-scale run with a consortium co-author.
   Ding 2020's weak form of (a) is done and matched by the per-type delta, so cytoplasmic gene recovery alone is
   not a headline.
2. **Thin unique capability.** One panel showing where label-free + continuous identity matters: held-out cell
   type (train on N-1 types), subtype gradients within L2/3 IT or L5 IT, mis-annotated nuclei, mixtures.
3. **Ablations on the final model.** Latent width done; still missing: no conditioning, no JSD, N-1 type,
   donor-held-out split. Cheap, scripts exist.
4. **Sex covariate.** Cells are 6 male + 6 female libraries, nuclei all female; Xist is the top-variance gene in
   every type and imputes to one mode. Not built; would reuse the library-size covariate path.
5. **Human paired truth.** None exists publicly for brain. Honest route: application (a) with mouse-trained weights
   and explicit caveats.
6. **Packaging.** No manuscript draft, no frozen model description, no weights/CLI release. The covariate model
   is the one to freeze.

## 5. Proposed order and what to decide together

1. Freeze the covariate model as the reported model (decide now).
2. Pick the application: SCORCH HIV-vs-control DE on imputed profiles is the only candidate with data in hand.
   Needs a decision on scope and on whether a disease result from mouse-trained weights is defensible for the
   target journal.
3. Run the cheap ablations (N-1 type, no JSD, no conditioning) in the background.
4. Start the manuscript skeleton from the figure list above; the RIP talk forces a first draft of the narrative
   by November anyway.

Questions for the meeting: is Option B the right frame for the lab; which application would you find convincing;
is a Nature-family target realistic without human paired data, or should we aim at Genome Biology / Nature
Communications and finish sooner.
