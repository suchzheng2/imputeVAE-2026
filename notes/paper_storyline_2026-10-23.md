# Step back: storyline, main results, conclusion, and gaps for a Nature-family genomics submission (2026-10-23)

Written while the library-size covariate arms were being evaluated (latent-usage results in; benchmark and
variance tables pending). Builds on `paper_status_2026-10-06.md` (figure inventory, gap list A1-C4),
`benchmark_review_2026-10-09.md` §5 (split-aware numbers), `libsize_covariate_review_2026-10-09.md`,
`gerstein_perspective_review.md` (April). Numbers below are the split-aware test-partition ones unless stated.

## 1. What the evidence actually supports today

| Claim | Evidence | Strength |
|---|---|---|
| A label-free cVAE learns the nucleus-to-cell translation and is the best label-free method | gene corr 0.900 ± 0.002 vs scVI/scANVI label-free 0.86; energy distance 6.8 vs 9.2 (floor 0.6); within-type coexpression 0.79 vs 0.72; 3 seeds | solid |
| With cell-type labels, simple methods reach the ceiling | CT-mean 0.981 (ceiling), scGen 0.975, scANVI + per-type library 0.962; cVAE 0.90 | solid, and it bounds the method claim |
| The learned delta is compartment biology and is cell-type-specific | ribosomal 100 % sign, nuclear lncRNA 100 % (45-gene list), SynGO vs TF controls, neuron/glia ratio 4-5x, scrambled control z = 40 | solid |
| It is validated non-circularly | fractionation-seq rho 0.44 (mouse) / 0.40 (human); TRAP/RiboTag H2 0.43 -> 0.73 with ceiling 0.74, holds over 19 populations | solid, but CT-mean / linreg reach the same numbers |
| It transfers | human M1 zero-shot (SynGO 0.24, mouse-human r 0.72, donor r 0.99), sorted human populations (SynGO delta 1.8x genome-wide), kidney conserved core | plausibility-level, no paired human truth |
| Per-cell fidelity | within-type variance 0.15 of real = library-size pass-through; latent uses 5 of 32 dims; covariate model: library share 0 %, within-type variance falls 11x, 2-3 identity axes remain | the per-cell claim is **gone**; this is now a characterisation of the data, not a feature |
| Downstream payoff | none | **missing** |

The pattern: everything that is validated is validated at the cell-type level, and at that level a per-gene
linear map or the type mean is as good. The model's measurable edge is (i) no labels at inference, (ii) a
continuous identity code (gradients inside a type, mixtures, unannotated nuclei), (iii) population-level realism
(energy distance). None of these has yet been turned into a result a biologist cares about.

## 2. Storyline options

**Option A, methods paper ("ImputeVAE imputes whole-cell profiles from nuclei").** This is what the ISMB abstract
promised. It does not survive the split-aware benchmark at Nature level: label-using baselines sit at the
ceiling, the per-cell claim has been withdrawn, and reviewers will ask what the 22M-parameter model adds over
"type mean + per-gene regression". Journal fit: Bioinformatics / PLOS CB / Genome Biology with strong framing.

**Option B, biology paper ("the nuclear-to-whole-cell transcriptome transformation is cell-type-specific,
low-dimensional and conserved; here is a learned map of it and what it changes in nuclei-only atlases").**
The model is the instrument, not the finding. Findings: (1) a cell-type-resolved compartment delta for 10k
genes, validated by biochemistry and ribosome profiling; (2) the transformation is low-rank: a handful of
latent directions (type identity, RNA content) explain what a nucleus profile can say about its cytoplasm at
10x resolution, and individual-nucleus information beyond that is below the sampling ceiling (paired r 0.32 of
a 0.54 ceiling for well-detected genes); (3) conserved to human and partly across tissues; (4) **applied to a
nuclei-only atlas it recovers cytoplasmic biology that nuclei miss and changes a downstream result**. Item 4
does not exist yet and is the paper. Journal fit: Nature Genetics / Nature Communications / Genome Biology if
item 4 is a real PsychENCODE-scale result; Nature Methods only if the tool and generalisation (item 3,
independent matched datasets) are strong.

**Option C, resource paper ("cell-like profiles for N million brain nuclei").** Needs human paired training
data or a convincing zero-shot argument plus a consortium use. Not reachable alone from this repo.

Recommendation: **Option B**, with the method section honest about label-free translation and the latent
result turned into a finding ("the nucleus determines the cell-type-level cytoplasm; not the single-cell
cytoplasm"). The covariate model is the one to report: explicit output scale, no library-size artefact.

## 3. Main results as figure claims under Option B

1. **Fig 1, the transformation.** Cell-type-resolved nucleus-to-cell delta from paired AIBS MOp data; what
   moves (ribosome, synapse, cytoplasm) and what moves the other way (nuclear lncRNA, Gm*); neuron vs glia.
2. **Fig 2, the model and its benchmark.** Label-free cVAE; split-aware table with the ceiling and both groups
   (label-free vs label-using); ablations on the final architecture (conditioning, JSD, latent width, N-1 type).
3. **Fig 3, non-circular validation.** Fractionation-seq, TRAP/RiboTag across 19 populations (imputed = real
   whole cell = ceiling), sorted human populations.
4. **Fig 4, what a nucleus can and cannot tell you about its cytoplasm.** Latent usage (5 dims), library-size
   covariate result (within-type variance is RNA content), paired-r ceiling analysis, per-type "difficulty"
   (which types / genes are least predictable and why: compartment biology, not model failure).
5. **Fig 5, generalisation.** Unseen chemistry (v2; must be explained, see gap 3), independent matched sc/sn
   datasets (Ding 2020, Bakken 2018), human M1 zero-shot, kidney core.
6. **Fig 6, application.** Nuclei-only dataset -> imputed -> a result that changes (candidates below).

## 4. Gaps, ranked by what they block

1. **No downstream application (A3).** The paper's reason to exist under Option B. Candidates, cheapest first:
   (a) cytoplasmic/activity gene recovery in nuclei-only data: Fos/Arc/Egr1, mitochondrial and ribosomal
   programs, cell-type-specific cytoplasmic markers (Plp1, Mt1) in an independent snRNA-seq set, checked
   against the matching scRNA-seq of the same tissue (**Ding 2020 has no cells, only bulk; its weak form is done
   2026-10-26, see gap 4, and the per-type delta matches it, so 1a is not the headline**); (b) a nuclei-only DE study
   re-analysed on imputed profiles (does a disease/condition DE list change toward what whole-cell data show?);
   (c) marker-based cell-type annotation of nuclei improves after imputation (classification 0.73 -> 0.96 is
   already in the benchmark; make it a figure on external data); (d) PsychENCODE-scale run (76k human M1
   nuclei already done; scale to a consortium set and co-author).
2. **Thin unique capability.** Must show where label-free + continuous identity matters: held-out cell type
   (N-1 training), subtype gradients within L2/3 IT or L5 IT, unannotated or mis-annotated nuclei, mixtures.
   One of these should be a panel, not a sentence.
3. **Cross-chemistry (v2): CLOSED 2026-10-23.** The covariate model decoded at the v2-cell library reaches gene corr
   0.93 / R² 0.72 on unseen chemistry (paper model 0.59 / -0.03). The transformation transfers; the output scale is an
   explicit parameter. Becomes a Fig 5 panel and a selling point of the covariate model. **Update, same day:** the
   label-free scale modes exist; one chemistry-wide ratio gives 0.84-0.86 on v2 and 0.87 on the v3 benchmark (label
   0.93 / 0.97) because the cell/nucleus RNA-content ratio is type-specific (1.2x glia, 2.5-3.7x neurons). Reading the
   type from the model's own latent (majority vote over the 50 nearest training nuclei, 97 % accurate) recovers
   **0.962 / R² 0.91 / energy 2.6 with no label at inference** (v3 test), level with scANVI-with-labels; with no
   labels anywhere 0.921. Fig 2 therefore becomes: label-free ImputeVAE (covariate, predicted-type decode) 0.962 vs
   scVI/scANVI label-free 0.86, and label-using scGen 0.975 / CT-mean 0.981 as the ceiling; the old 0.900 row is the
   no-covariate ablation. On v2 the label-free number is 0.89 (type prediction drops to 77 % across chemistry), which
   is the remaining cross-chemistry gap and is a type-assignment problem, not a scale problem. **Update 2026-10-27:**
   the depth-robust reference classifier (gap 4) lifts v2 type assignment to 84 % and the label-free number to
   **0.90** (label 0.93; job 14189131; libsize review §6.4a). What remains is Pvalb -> Sst (56 %) and L6 IT <-> L5 IT
   confusion specific to v2 input; a reference that includes v2 nuclei would be the next step if the number matters.
4. **Independent matched sc/sn dataset (B1): DONE 2026-10-26, with a correction.** Ding 2020 (GSE132044) cortex has
   *nuclei only* (four snRNA-seq chemistries + bulk; GEO series design), so it is not a matched set. The matched
   hold-out is **Bakken 2018 (GSE123454)**: mouse VISp, SMART-seq, 461 nuclei / 445 cells, 11 matched clusters
   (`notes/experiments/bakken2018_holdout/README.md`). Result: direction of the nucleus->cell change predicted at
   the split-half ceiling (sign agreement 0.94 = ceiling; ribosomal / cytoplasmic 100 %), gene-level magnitude
   r 0.58 pooled / 0.73 composition-free vs a reliability ceiling of 0.92, per cluster 0.43-0.80, label-free type
   assignment correct (Pvalb, Sst, L5 ET incl. the Chrna6 cluster, L6 CT, NP). **But the per-type training delta
   added to the nucleus scores the same (0.56 / 0.59)**: on external data the model equals "predict the type, add
   its mean shift". Full regeneration is the wrong cross-platform mode (0.41-0.48, emits a 10x-like profile);
   the model's delta applied to the observed nucleus is the mode to report. Ding 2020 becomes the nuclei-only
   chemistry test (`notes/experiments/cross_dataset_ding/README.md`): imputation moves the pseudobulk of 10x v2,
   DroNc-seq, sci-RNA-seq and Smart-seq2 nuclei toward the matched whole-tissue bulk (Pearson 0.27-0.56 ->
   0.58-0.65; predicted delta agrees 0.82 across chemistries), again matched by the per-type training delta.
   Fig 5 panels: Bakken predicted-vs-empirical delta with ceiling and baseline; Ding four-chemistry pseudobulk.
   **New tool:** a depth-robust reference classifier for the training type (`analysis/external/type_classifier.py`,
   v3 validation 0.982) replaces the latent kNN vote off-chemistry (the vote assigns 14-23 % L6b on 2-6 k UMI
   nuclei); re-run for the v2 label-free number 2026-10-27: 0.89 -> 0.90, type 77 % -> 84 % (gap 3).
5. **Ablations and controls on the final model (A4).** Latent width done 2026-10-23 (2 active dims suffice for 15/17
   types at gene corr 0.957; 1 dim keeps 5 types; the identity code is 2-4 dimensional, review §6.7), no
   conditioning, no JSD, N-1 type, donor-held-out split. Seeds exist (3). Statistical rigour is otherwise in place.
   **Done 2026-11-06** (`notes/experiments/ablations_final_model/README.md`, Results): JSD removable (0.964 vs 0.970,
   label-free unchanged 0.962); alignment is the one loss term that matters (0.947 / R² 0.87 / clf 0.92 without it);
   encoder conditioning worth ~0.005. Unseen type: delta interpolated from neighbours at r 0.93 (L5 IT) / 0.85-0.87
   (Pvalb) but identity collapses onto the nearest seen type (classifier recall 1 % / 0 %; vote files L5 IT as L2/3 IT +
   L6 IT, Pvalb as Sst). Unseen nuclei donor: no cost at the labelled decode, 0.01-0.015 label-free. Architecture stays
   frozen; N-1 + donor rows go to Fig 5 beside Bakken / v2. A library (batch) covariate run was added the same day
   (README, last section) as a supplementary check, not a change of the reported model. **Read 2026-11-13:** it costs
   0.014 gene corr (0.956 vs 0.970) and the identity of cell-rare types (L5 ET recall 0 %, filed as L6 CT) and weakens
   the latent alignment; but a female / male target-library decode reproduces the real sex difference (Xist 3.6 vs 0.0,
   genome-wide r 0.66 with the real female-minus-male vector). Supplementary only; the one usable panel is the
   sex-placement decode.
6. **Human.** Zero-shot only. **On disk (2026-10-23):** the group's SCORCH object (`/gpfs/gibbs/pi/gerstein/yscorch_data_share/seurat_combined_v3.RDS`)
   is human snRNA-seq, 654,918 nuclei, 33 donors, PFC / insula / ventral striatum, HIV vs control, 37 consensus labels that
   include every cortical type in our training set plus striatal MSN subtypes; no spatial slot. It is the nuclei-only
   disease application (1b/1d) ready to run with the human-M1 ortholog mapping. The mouse spatial SCORCH data the advisor
   mentioned is not in the group share; see `notes/experiments/yscorch_scoping/README.md`. For Nature Genetics a human paired set (even small: GSE219278 is sorted
   populations, not paired) or a consortium application is needed. Paired human cell/nucleus brain data are rare;
   the honest route is application 1d with mouse-trained weights and explicit caveats.
7. **Packaging (C1-C4).** No manuscript draft, no canonical model description, no CLI / weights release. The
   covariate model changes the architecture description again; freeze it now.

## 5. Conclusion for the planning meeting

- The method result is real but bounded: best label-free translator; at the cell-type level, labels plus
  RNA content close the gap. Say this in the paper rather than let a reviewer say it.
- The biology result is the publishable core: a validated, conserved, cell-type-specific compartment map, and a
  quantitative statement of what a nucleus can tell you about its cytoplasm.
- The missing piece for a Nature-family journal is one convincing downstream use on nuclei-only data. Without
  it the paper lands in Genome Biology / Bioinformatics territory regardless of how clean the benchmark is.
- Order of work: (1) v2 re-test with the covariate model and the latent-width control (done 2026-10-23);
  (2) independent matched evaluation (done 2026-10-26 on Bakken 2018; Ding 2020 nuclei-only done the same day);
  (3) application 1b/1d (SCORCH human snRNA-seq, disease DE on imputed profiles) is now the only candidate for a
  result that changes; (4) N-1 type ablation; (5) re-run the v2 label-free number with the reference classifier
  (done 2026-10-27: 0.90, type 84 %); (6) freeze the model and draft. **Standing conclusion after (2): every external test so far is matched by
  "predicted type + that type's mean nucleus->cell shift"; the paper's method claim must be the label-free,
  chemistry-robust *application* of a validated cell-type-level compartment map, and the biology / resource
  (the map itself, Fig 1 + Fig 3 + Fig 5) is the contribution.**
