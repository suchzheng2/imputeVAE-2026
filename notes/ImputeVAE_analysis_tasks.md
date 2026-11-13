# ImputeVAE — Analysis Task List (for Claude Code)

Ordered by urgency. Phase 1 = no retraining, reanalysis of existing outputs (do first). Phase 2 = external-dataset validation (inference only, no retrain). Phase 3 = retraining experiments (only if time before exam).

## Repo orientation (read first)

- **Model:** single-stage conditional VAE `VAE_JSD_Cond`, scalar modality conditioning (1=cell, 0=nuclei) on **encoder + decoder**, MSE + JSD loss, log1p-normalized input, 128-dim latent, 10,134 genes (**Config B** gene list).
- **Best checkpoint:** `run/run_2026-07-18-19-40-51_570984/models/model.pt` (args in `arguments.json`).
- **Imputation path:** encode nuclei with nuclei scalar → decode with **cell** scalar → imputed cell-like expression.
- **Key existing outputs:**
  - Imputed matrix: `run/cvae_full_eval/imputed_cell_from_nucleus.npy` (40166 × 10134)
  - Nucleus obs/metadata (incl. cell-type labels): `run/cvae_full_eval/nucleus_obs.csv`
  - Full eval JSON: `run/cvae_full_eval/eval/eval_cvae_full.json`
  - Per-gene delta: `notes/experiments/nuclear_lncrna_analysis/all_genes_delta.csv`
  - Per-CT delta tables: `notes/experiments/delta_analysis/tables/<CT>_delta.csv`
  - Top-100 gene lists: `notes/experiments/delta_analysis/top100_{increased,decreased}_genes.txt`
- **Existing scripts:** `analysis/eval/runner.py` (modular eval CLI), `analysis/baseline_linear_regression.py`, `analysis/cross_dataset_eval.py`, `analysis/export_cvae_imputed.py`, `analysis/go_analysis.py`, `analysis/find_nuclear_lncrnas.py`.
- **Conventions:** save new outputs under `notes/experiments/<task_name>/`, write a short `README.md` per experiment (match existing style), append headline numbers to `notes/experiments/baseline_comparison.md`. Gene symbols are **mouse** (Allen MOp).
- **Delta definition:** `delta[g] = mean(imputed_cell[:,g]) − mean(input_nuclei[:,g])` on log1p scale. Positive = model adds (cytoplasmic); negative = model removes (nuclear). Confirm the exact column name in `all_genes_delta.csv` before using.

---



# PHASE 1 — Reanalysis, no retraining (do immediately)



## Task 1 — Run the linear-regression baseline and add it to the comparison table  ⏱ ~1–2 h

**Why (urgency):** The committee's methods member (Cho) will ask "did you beat a simple interpretable baseline." The script exists but there are no numbers in the table.
**Steps:**

1. Run `analysis/baseline_linear_regression.py` on the same train/val split as the cVAE full eval. It should fit, per gene, `cell_mean ~ a + b·nucleus` across the 17 cell types (per-gene α,β), then apply per-nucleus: `pred_cell[n,g] = a_g + b_g · x_nuc[n,g]`.
2. Run it through the **same eval pipeline** used for the other baselines (`analysis/eval/runner.py` or `analysis/evaluate_baseline_incremental.py`) so metrics are computed identically: gene corr, R², delta corr, classification acc, coexpression, variance ratio, QQ, Wasserstein, DE overlap, per-cell corr.
3. Save `run/linreg_baseline/eval/eval_linreg.json`.

**Output:** add a "Linear regression" column to the head-to-head table in `notes/experiments/baseline_comparison.md`.
**Interpretation to record:** linear = the "global fixed-δ / scGen-analog" baseline. Expect cVAE to win on variance, coexpression, distributional metrics; note where linear is competitive (likely gene corr / CT-means).

## Task 2 — Localization concordance vs external databases (non-circular validation)  ⏱ ~1 day  ★ highest ROI

**Why (urgency):** Directly answers the "is the delta real biology or curve-fitting / circular?" heat, using references the model never saw. Runs on existing delta.
**Inputs:** `notes/experiments/nuclear_lncrna_analysis/all_genes_delta.csv`.
**Steps:**

1. **Get external localization refs:**
  - **lncATLAS** CN-RCI table (cytoplasmic:nuclear ratio per gene; positive = cytoplasmic) from [https://lncatlas.crg.eu/](https://lncatlas.crg.eu/) (download the full CN-RCI CSV).
  - **RNALocate v2** ([http://www.rnalocate.org/](http://www.rnalocate.org/)): pull per-gene subcellular localization; reduce to a binary/graded cytoplasm-vs-nucleus label.
  - (Optional) **APEX-seq** (Fazal 2019 supplementary): nuclear vs cytosol/ER gene lists.
2. **Ortholog mapping:** map mouse gene symbols → human (pybiomart / MGI homology table). Keep the intersection; log how many of 10,134 map.
3. **Concordance stats:**
  - Spearman correlation between per-gene delta and lncATLAS CN-RCI. Expect **positive**.
  - AUC/ROC: label genes cytoplasmic (CN-RCI>0) vs nuclear (CN-RCI<0); use delta as the classifier score. Report AUROC + PR-AUC.
  - Repeat the AUC using RNALocate labels (covers mRNAs, not just lncRNAs).
  - Confirm the extreme cases: Malat1, Neat1, Xist sit at the nuclear (negative-delta, low-CN-RCI) end.
4. **Save:** `notes/experiments/localization_concordance/` — a scatter (delta vs CN-RCI), ROC curve, a stats JSON, and `README.md`.

**Interpretation:** a monotonic positive delta↔CN-RCI relationship and AUROC ≫ 0.5 = independent, non-circular evidence the model moves transcripts in the biologically correct direction. This is the core "not a global ribosomal vector, not circular" figure.

## Task 3 — Dissociation / IEG confound audit  ⏱ ~2–4 h

**Why (urgency):** The Fos/Arc landmine — nuclei lack dissociation-induced immediate-early genes, so the model could be rewarded for reproducing a protocol artifact. Preempt it with a number.
**Steps:**

1. Define a **dissociation/stress gene module** (van den Brink 2017 + Marsh 2022): `Fos, Fosb, Jun, Junb, Jund, Egr1, Egr2, Arc, Npas4, Hspa1a, Hspa1b, Ier2, Ier3, Dusp1, Zfp36, Btg2, Nr4a1` (+ Marsh microglial ex-vivo-activation set). Put the list in the experiment folder.
2. For each module gene, report its delta and its **percentile rank** among all 10,134 gene deltas.
3. Test whether the module is **enriched among top-increased genes** (hypergeometric test vs background). You *want* NOT-enriched (or weakly).
4. Cross-check the top-100 increased list (`top100_increased_genes.txt`) for module members.

**Output:** `notes/experiments/dissociation_audit/` — table of module gene deltas + ranks, enrichment p-value, `README.md`.
**Interpretation:** if the added-gene signal is dominated by ribosomal/cytoplasmic/synaptic (it is — Rpl/Rps/Eef1a1/Actb) and the stress module is NOT top-ranked, you can state the delta is biology, not dissociation. Report honestly whichever way it comes out.

## Task 4 — Scrambled-delta negative control  ⏱ ~2–3 h

**Why (urgency):** Proves your GO/compartment/localization metrics have statistical power (the Hou-2020 circularity answer). Cheap.
**Steps:**

1. Permute the gene labels on the delta vector (N=1000 permutations).
2. Recompute: (a) GO enrichment p-values for "top increased" (via `go_analysis.py`), (b) the compartment sign-correctness metrics (ribosomal/cytoplasmic/nuclear-lncRNA), (c) the Task-2 localization Spearman/AUC.
3. Build null distributions; report the real value's percentile / empirical p vs the permuted null.

**Output:** `notes/experiments/negative_control/` — null-distribution plots for each metric, empirical p-values, `README.md`.
**Interpretation:** real metrics should sit far in the tail (p<0.001); scrambled deltas should collapse to chance. This is the slide that kills "you manufactured the structure."

## Task 5 — Diagnose the L5 ET anomaly (delta R² = −0.560)  ⏱ ~2–3 h

**Why (urgency):** This negative R² sits in your own per-CT table (`eval_cvae_full.json → delta_gt.per_ct_corr`) despite r=0.950. A committee will spot it. Understand it before they do.
**Steps:**

1. Pull L5 ET: predicted vs ground-truth delta per gene; scatter it.
2. Check n (training + eval cells for L5 ET), and whether a few high-leverage genes or a scale/offset mismatch drive the negative R² (R²<0 with high r ⇒ systematic slope/intercept bias, not lack of correlation).
3. Compare to the healthy CTs (L2/3 IT) to isolate the cause. Check cell count (Finding 6 — is L5 ET low-n or imbalanced?).

**Output:** `notes/experiments/l5et_diagnosis/` — scatter, n's, a one-paragraph explanation, `README.md`.
**Interpretation:** most likely low-n or a magnitude-calibration issue. Have a one-sentence answer ready ("high correlation, but a scale offset in a low-n type inflates the residual").

## Task 6 — Gene-stratified evaluation by nucleus–cell gap magnitude  ⏱ ~3–4 h

**Why (urgency):** Sharp, expected result: the model should help most on large-gap genes and not hurt small-gap genes. Already on your Aim-1 next-steps.
**Steps:**

1. Compute the ground-truth per-gene gap = |mean(real_cell) − mean(real_nuclei)| (from raw data).
2. Bin genes into quartiles by gap.
3. Per bin, compute the improvement of imputed vs raw-nuclei (gene corr and/or per-gene error) against real cell.

**Output:** `notes/experiments/gene_stratified_eval/` — bar/line plot of improvement vs gap quartile, `README.md`.
**Interpretation:** monotonic — biggest improvement in the top gap quartile; near-zero change (not harm) in the bottom. That's the "model does work where the biology says it should" figure.

---



# PHASE 2 — External-dataset validation (inference only, no retrain)

> **Shared ingest recipe for all Phase-2 tasks:** (a) subset the external dataset to the Config B 10,134-gene space (intersect on gene symbol; log how many map; fill missing with 0 or drop — be explicit); (b) apply the **same log1p normalization** as training; (c) harmonize cell-type labels to the 17 training CTs (map/rename; drop types with no correspondence); (d) load the checkpoint and run the imputation path (encode w/ nuclei scalar → decode w/ cell scalar). Reuse `analysis/export_cvae_imputed.py` and `analysis/cross_dataset_eval.py` as templates. Write ingest as a reusable helper.



## Task 7 — Extend Ding 2020 cross-dataset eval to all protocols  ⏱ ~1 day

**Status 2026-10-26: DONE as a nuclei-only test** — GSE132044 cortex has no whole cells (4 snRNA-seq chemistries + bulk). See `notes/experiments/cross_dataset_ding/README.md`.

**Why (urgency):** You already did v2 (Finding 7). Adding Drop-seq + Smart-seq2 shows robustness across chemistries — a clean, low-risk win.
**Steps:**

1. Obtain Ding 2020 mouse cortex matched sc/sn for **10x v2, Drop-seq, Smart-seq2** (GEO GSE132044).
2. For each protocol: ingest per the recipe, run inference, run the full eval pipeline.
3. Save `run/cvae_full_eval/eval/eval_cvae_full_ding_<protocol>.json`.

**Output:** a small table (gene corr, delta corr, QQ, Wasserstein) × 3 protocols in `notes/experiments/cross_dataset_ding/README.md`.
**Interpretation:** distributional metrics (QQ, Wasserstein) should hold; CT-mean metrics may soften on the plate-based Smart-seq2. Report honestly — this is your cross-protocol generalization story.

## Task 8 — Bakken 2018 matched sc/sn held-out test (the "simulation replacement")  ⏱ ~2–3 days  ★ core

**Status 2026-10-26: DONE** — GSE123454 (VISp SMART-seq, 11 matched clusters): sign at ceiling, delta r 0.58 vs ceiling 0.92, equal to the per-type training delta. See `notes/experiments/bakken2018_holdout/README.md`.

**Why (urgency):** The strongest non-circular test of the *transformation*: an independently generated matched sc/sn dataset in a DIFFERENT cortical region (VISp) and toward human. Does the predicted delta reproduce Bakken's *measured* nucleus-vs-cell differences?
**Steps:**

1. Obtain Bakken 2018 matched sc/sn (mouse VISp; and the human cortical set if available — **confirm accession from the paper's Data Availability / Allen Institute portal**; do not guess).
2. Ingest per the recipe. Compute Bakken's **empirical** delta = mean(real_cell) − mean(real_nucleus) per matched cell type.
3. Run the model on Bakken nuclei → predicted delta.
4. Correlate predicted delta vs Bakken empirical delta (overall + per matched CT). Also check gene-class agreement: long genes / lncRNAs down, translation/ribosomal up.
5. If a human matched subset exists, repeat (this is your only near-term human transformation check).

**Output:** `notes/experiments/bakken2018_holdout/` — scatter(s) predicted vs empirical delta, per-CT correlation table, `README.md`.
**Interpretation:** high predicted↔empirical delta correlation on a non-training dataset = the transformation generalizes and is not memorized. This is your headline external-validation slide.

## Task 9 — Scala 2021 Patch-seq gap-closing (orthogonal cytoplasmic, same region)  ⏱ ~2 days

**Why (urgency):** Patch-seq aspirates cytoplasm in your exact region (mouse MOp). Does imputation move the profile TOWARD the whole-cell Patch-seq mean?
**Steps:**

1. Obtain Scala 2021 mouse MOp Patch-seq (NeMO / Allen / GEO — confirm accession). Ingest per recipe; map to the shared t-type taxonomy.
2. Per matched t-type, compute the distance from (a) raw nucleus mean → Patch-seq mean, and (b) imputed mean → Patch-seq mean. Use correlation distance and/or Euclidean on shared genes.
3. Test that (b) < (a) across t-types (paired test).

**Output:** `notes/experiments/scala_patchseq/` — per-t-type before/after distance plot, paired-test p-value, `README.md`.
**Interpretation:** imputation should reduce the distance to the orthogonal whole-cell reference. Caveat in README: Patch-seq neuropil contamination, low n → validate at pseudobulk/t-type level, not per cell.

## Task 10 — TRAP/RiboTag overlap (orthogonal cytoplasmic)  ⏱ ~4–6 h

**Why:** Ribosome-bound = cytoplasmic/translated. Are the added genes enriched in a cell-type TRAP set?
**Steps:**

1. Obtain a mouse-cortex TRAP/RiboTag gene set (e.g., Sheehan 2025 glia; a neuronal bacTRAP set — confirm GEO accession from paper).
2. For matched cell types, test whether the top-increased genes are enriched in that type's TRAP-positive set (hypergeometric / GSEA).

**Output:** `notes/experiments/trap_overlap/` — enrichment table, `README.md`.
**Interpretation:** added genes over-represented in TRAP = independent support for the ribosome/translation axis. Caveat: TRAP = translation, not steady-state; population-level.



## Task 14 — Lake 2023 kidney: conserved-transformation test (Aim 2, no model inference) ⏱ ~0.5–1 day ★ high value, low risk

**Why (urgency):** Tests whether the nucleus→cytoplasm shift the model learns in mouse brain is a *conserved* biological axis — using an independent MATCHED human dataset (different tissue AND species). **No model inference, no retraining** — you compute the *empirical* cell−nucleus delta directly from matched data, so it is fully non-circular. Answers Aim 2's central question cheaply. Do NOT run the brain-trained model on kidney (out-of-domain, unfair).

**Data — Lake et al. 2023 human kidney, 10x 3′ v3, donor-paired (CELLxGENE v1.5 h5ad, raw counts in** `raw.X`**, shared** `donor_id`**):**

- Cells (225,177): `https://datasets.cellxgene.cziscience.com/08648660-9898-4644-a9a1-ff3a159a0041.h5ad`
- Nuclei (304,989): `https://datasets.cellxgene.cziscience.com/02d5db96-d67e-45f2-b556-b6bd99753665.h5ad`
- Per-library matrices alternative (GEO): `GSE183276` (scCv3, cells) / `GSE183277` (snCv3, nuclei). **Fix the spreadsheet accessions to these** (catalog had GSE199711/GSE183273 — those are not the matched-v3 pair).

**Steps:**

1. Load both h5ads; use `raw.X` counts; apply the SAME log1p normalization as the mouse pipeline.
2. Restrict to shared donors (intersect `donor_id`; the catalog confirms heavy overlap — e.g. 31-10001, 29-10006, 33-10005/6, 32-10003/34, 29-10008/10/12/13).
3. Pseudobulk cell vs nucleus per cell type (and global); compute empirical delta = mean(cell) − mean(nucleus) per gene (log1p).
4. **Conserved-signature test:** are ribosomal genes (`RPL`*/`RPS*`) positive-delta, and `MALAT1`/`NEAT1` negative-delta, in human kidney too? Report ranks/percentiles (mirror the Task-3 audit style).
5. **Cross-tissue comparison:** map human→mouse orthologs; compare the top-increased / top-decreased gene *classes* to your mouse-brain delta. Expect ribosomal + nuclear-lncRNA SHARED; synaptic genes mouse-brain-ONLY (kidney has no synapses). Log how many genes map.
6. GO on human-kidney top-increased: expect ribosome / cytoplasm / translation, NOT synapse.

**Output:** `notes/experiments/conserved_transformation/` — empirical delta table, ribosomal/lncRNA rank plot, human-kidney-vs-mouse-brain class-overlap figure, `README.md`.

**Interpretation:** a conserved CORE (ribosomal export ↑, nuclear-lncRNA retention ↓) across tissue + species, plus tissue-specific components (synaptic = brain-only) = evidence the model learns a generalizable biological axis. Either outcome is informative (Aim 2 framing: conserved → transfers; divergent → RNA-compartment tissue-specificity finding).

**Caveats to state:** cross-tissue → cell-type composition differs, so compare at the gene/class level, not per-CT identity; ortholog mapping drops some genes; this validates the biological PREMISE, not that the model runs on kidney. Feeds the deck's "conserved transformation" (Aim 2) slide.

## Task 15 — scANVI baseline (fair counterfactual imputation)  ⏱ ~half day  ★ closes the #1 methods question
**Why (urgency):** scANVI is the closest architectural cousin (scVI + cell-type labels). A committee will ask "isn't this just scANVI?" — an empirical head-to-head turns a conceptual answer into a number. Adds one decisive column to the baseline table.
**Reuse:** same data / split / gene set (Config B, 10,134) / eval pipeline as `analysis/baseline_scvi.py`. Give scANVI the cell-type labels (semi-supervised = its intended, *stronger* setup — a fair fight).

**Setup + train (scvi-tools):**
```python
import scvi, numpy as np
scvi.settings.seed = 0
# adata: combined cells + nuclei; raw counts in layer "counts";
# obs has "modality" in {"cell","nucleus"} and "cell_type"
scvi.model.SCANVI.setup_anndata(
    adata, labels_key="cell_type", unlabeled_category="Unknown",
    batch_key="modality", layer="counts")
model = scvi.model.SCANVI(adata)          # default hyperparams (do NOT tune to lose)
model.train(max_epochs=400)
```
(Equivalent alt: train `scvi.model.SCVI` with `batch_key="modality"`, then `SCANVI.from_scvi_model(scvi_model, unlabeled_category="Unknown", labels_key="cell_type")`.)

**Impute = the counterfactual decode (the direct analog of ImputeVAE's embedding swap):**
```python
nuc = adata[adata.obs["modality"] == "nucleus"].copy()
imp = model.get_normalized_expression(
    nuc, transform_batch="cell",          # decode nuclei "as if" they were cells
    library_size=1e4, return_numpy=True)   # match the CP10K scale your pipeline uses
# then post-process IDENTICALLY to baseline_scvi.py (e.g. np.log1p) so scANVI and scVI are directly comparable
imp = np.log1p(imp)
```
- `transform_batch="cell"` must match the exact `modality` category string. This is scVI/scANVI's built-in counterfactual — it's the fair version of "isn't transform_batch the same as your swap?"

**Eval hookup:** run `imp` through the SAME pipeline as every other baseline (`analysis/eval/runner.py` / `evaluate_baseline_incremental.py`) → save `run/scanvi_baseline/eval/eval_scanvi.json`. Add a "scANVI" column to the head-to-head table in `notes/experiments/baseline_comparison.md`, next to scVI.

**Fair-comparison guardrails (important — do NOT rig it):**
- Give it the labels (semi-supervised). Use default hyperparameters. Use its intended `transform_batch` counterfactual. A clean win on a fair fight is worth infinitely more than a rigged one.
- Match scANVI's output post-processing exactly to the scVI baseline's, so the comparison is like-for-like.

**Expected outcome (report honestly either way):** scANVI ≈ scVI on *expression-fidelity* metrics (poor gene corr, likely negative R²) — label supervision preserves cell-type structure but still cannot *add* the missing cytoplasm (the structural limit). It may beat scVI on **classification** because of the labels — if so, say so, then note classification isn't the imputation task; expression fidelity is, and that's where it fails.
**Output:** `run/scanvi_baseline/eval/eval_scanvi.json` + updated `baseline_comparison.md` + one line for the deck: "scANVI (trained semi-supervised, decoded nuclei→cells via transform_batch): gene corr [X] vs cVAE 0.93."
**Caveat:** scANVI default latent is small (10) vs your 128 — if you want the strongest possible baseline, also run it at n_latent≈30–128 and report the best; don't let a small-latent handicap be the reason you win.

# Task 16 — scButterfly benchmark (closest competitor; parsimony argument)  ⏱ ~1–1.5 days  ★ high value, moderate risk
**Why (urgency):** scButterfly (Cao et al. 2024, Nat Commun) is your *closest prior art* — same aligned-VAE cross-modality translation paradigm. Your defense is **parsimony**: a shared encoder/decoder + scalar condition should match scButterfly's dual-VAE + translator + adversarial design *on your same-feature-space task*. That claim is only credible if you show it empirically. This is the single most important missing baseline; run it before the exam if at all feasible.

**Repo / API:** `BioX-NKU/scButterfly` (pip: `scButterfly`). **Confirm the exact API from the repo tutorials** — do not trust remembered signatures; the workflow is roughly `Butterfly() → load_data → data_preprocessing → (augmentation) → construct_model → train_model → test_model`.

**Two real complications (read before starting):**
1. **scButterfly is RNA↔ATAC by default** (different feature spaces; its ATAC decoder uses a per-chromosome `chrom_list`). Your task is RNA(nucleus)→RNA(cell), **same** gene space. Configure **both modalities as RNA-type** (no ATAC/`chrom_list` branch) so both sides are gene-expression decoders over the 10,134 Config-B genes. This is the main integration effort.
2. **No true single-cell pairs** (destroying the cell to get the nucleus). scButterfly's translator trains on paired cells. Construct **cell-type-matched pseudo-pairs** (match each nucleus to a randomly drawn cell of the same type) — this is the direct analog of ImputeVAE's cell-type alignment, so it's a *fair* signal (neither method gets true pairs). If scButterfly exposes an **unpaired mode**, use that instead and note which you used.

**Protocol:**
1. Two AnnData: `rna_nuc` (modality A) and `rna_cell` (modality B), **same 10,134 genes, same normalization** as your pipeline; same train/val split as the other baselines.
2. Build cell-type-matched pseudo-pairs (document the matching); train scButterfly nucleus→cell with **default hyperparameters**.
3. Predict on **held-out nuclei** → imputed cell profiles.
4. Post-process **identically to your other baselines** (match `baseline_scvi.py`); run through the SAME eval pipeline → `run/scbutterfly_baseline/eval/eval_scbutterfly.json`; add a "scButterfly" column to `notes/experiments/baseline_comparison.md`.

**What to report (the framing is different from scVI/scANVI — this one you may not crush):**
- The honest goal is **competitive, not necessarily better**. If cVAE **matches or beats** scButterfly with far fewer parameters and no adversarial training → that *is* the parsimony win. Report **parameter count and training cost** for both ("competitive with ~N× fewer parameters, no discriminators").
- If scButterfly wins on some metric, say so and reframe: the contribution is problem formulation + parsimony + biological validation, and a simpler model that's *within range* of SOTA on a same-space task is the point. Do NOT tune scButterfly to lose.

**Output:** `eval_scbutterfly.json` + updated table + a deck line: "cVAE matches scButterfly (gene corr [X] vs [Y]) with a shared encoder/decoder + scalar condition — no dual VAEs, translator, or adversarial training."

**⚠️ Risk & fallback:** unlike scANVI (a drop-in via scvi-tools), scButterfly is a separate package built for a different task — expect real integration friction (RNA↔RNA reconfig, pairing). **If it can't be made to run cleanly before the exam,** the honest fallback is: (a) benchmark **scGen** instead — same-feature-space, trivial to run, and a legitimate representative of the directed-transformation family — and (b) present scButterfly **conceptually + cited**, with the head-to-head clearly labeled as in-progress/future work. A cited comparison you're honest about beats a rushed, broken benchmark.


---



# PHASE 3 — Retraining experiments (only if time remains)



## Task 11 — Held-out cell-type generalization (N−1)  ⏱ ~1–2 days (retrain)

Train leaving one cell type out; test imputation on the held-out type. Repeat for a few representative types (one neuron subtype, one glia). Report whether delta direction/biology holds on unseen types.
**Output:** `notes/experiments/heldout_celltype/` — per-held-out-type metrics, `README.md`.

## Task 12 — Likelihood ablation: MSE vs NB/ZINB decoder  ⏱ ~1–2 days (retrain)

Retrain a variant with a negative-binomial (or ZINB) decoder on counts instead of MSE-on-log1p. Compare all eval metrics, especially **variance ratio** (NB may recover more within-CT variance). Preempts Cho's "why MSE on count data."
**Output:** `notes/experiments/likelihood_ablation/` — metric table MSE vs NB, `README.md`.

## Task 13 — Alignment-term (JSD) ablation + collapse check  ⏱ ~1–2 days (retrain)

Train variants: (a) remove the modality conditioning, (b) remove the JSD alignment term. Show (a) → outputs collapse / modalities indistinguishable; (b) → cell-type structure degrades. This is the direct answer to "why doesn't your alignment just batch-correct the modalities away."
**Output:** `notes/experiments/alignment_ablation/` — metrics + latent UMAPs per variant, `README.md`.

---



## Suggested execution order for a tight timeline

Do **1 → 2 → 3 → 4 → 5** first (all Phase 1, no retrain, ~1 week, closes the heated topics). Then **7 → 8 → 9** (external validation, the credibility core). **6** slots in anywhere. Phase 3 only if the exam is >2 weeks out.

## Global acceptance criteria

Every task: (1) writes outputs under `notes/experiments/<task>/`, (2) includes a `README.md` with method + headline numbers + honest caveats, (3) appends its headline metric to `notes/experiments/baseline_comparison.md`, (4) logs how many genes/cell-types mapped when an external dataset is involved (silent gene-dropping is the #1 pitfall).