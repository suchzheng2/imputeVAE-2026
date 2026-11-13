---
name: project-ablations-final-model
description: Ablations on the final library-size covariate model (2026-11-04/06, DONE) — JSD optional, alignment essential, N-1 type delta r 0.93/0.85 with identity collapse, donor hold-out ~0 cost; library (batch) covariate run (results read 2026-11-13) costs 0.014 gene corr and L5 ET identity, but sex falls out of named-library decodes (Xist) -> supplementary only
metadata:
  type: project
---

Launched 2026-11-04 as the "needs no decision" item from [[project-prashant-update-1030]]. Record and reading guide:
`notes/experiments/ablations_final_model/README.md`; run tracker entry in `notes/training_runs.md`.

- Base = arm A (`run/run_2026-10-16_libsize_armA_jsd0.1_seed1`), see [[project-libsize-covariate]]. Six arms, one
  change each, folders `run/run_2026-11-04_ablation_{nojsd,noalign,deccond,n1_L5IT,n1_Pvalb,donor5}_seed1`.
- New flags: `train.py --exclude_cell_types`, `--holdout_donors <modality>:<donor>`; applied by
  `imputeVAE/data.py apply_train_exclusions` AFTER the stratified split (train/valid only) so the manifest test set is
  unchanged. Filtering before the split would shift other types' rows (sklearn draws per class in sequence).
- New script `analysis/benchmark/eval_by_donor.py` (per-nuclei-donor CT-level metrics; validated against
  `ct_level_robust` of the main table).
- Expected: training ~12 h per arm, two at a time on r813u29n03; predict + evaluate (~5 h) follow automatically.
- Side finding: on the 11 neuronal types alone, no-covariate cvae_seed1 gene corr 0.64 vs arm A 0.92; the covariate's
  gain is concentrated in neuron-vs-neuron structure.

- 2026-11-05: training done for all arms. deccond preempted at the instant of exit broke the dependency chain; label-free
  decodes of the exclusion arms had a kNN-reference leak (predict_cvae used the cache's unfiltered train partition; type
  vote named the held-out type) -> `apply_run_exclusions` added to `analysis/benchmark/predict_cvae.py`, arms 4-6
  re-decoded (15559684), evaluate 15559685. 9.4 GB of debug logs `jobs/logs/slurm-14861675_{1,2,3,6}_cvae_ablation.out`
  await deletion by the user (quota 4089/4096 GiB).

- 2026-11-06: **library (batch) covariate** built on request (user chose it over a sex covariate; library subsumes sex
  in cells). `--library_cond` flags in train.py; embedding + UNKNOWN token with dropout in models_cond.py; vocab/remap
  helpers in cond_training.py; `--target_library` in predict_cvae. Chain smoke 15562731 -> train 15562732 -> predict
  15562733 -> evaluate 15563057; outputs `run/benchmark/test/libcond_*`, `results_libcond/`. README section
  "Library (batch) covariate" has the reading guide and decision rule.

- **Results 2026-11-06** (evaluate 15559685; README "Results"): nojsd 0.964 (label-free 0.962 unchanged) -> JSD optional;
  noalign 0.947 / R² 0.87 / clf 0.92 -> alignment is the loss term that matters; deccond 0.963 (-0.005, -1 pt clf).
  N-1: held-out delta r 0.93 (L5 IT) / 0.85-0.87 (Pvalb), classifier recall 1 % / 0 %, vote files L5 IT as L2/3 IT 45 % +
  L6 IT 44 %, Pvalb as Sst 96 %; neighbours unaffected. donor5: 0 cost at ctmed, -0.01/-0.015 label-free for that donor
  (donor 5 is the hard donor for every model). Decision: architecture stays frozen; N-1 + donor rows -> Fig 5.

- **Library covariate results read 2026-11-13** (README "Results" under the library section): labelled 0.956 / R² 0.893
  vs arm A 0.970 / 0.927, label-free 0.959 vs 0.962, clf 0.927 vs 0.967; cell-rare types lose identity (L5 ET recall
  0.00, nearest centroid L6 CT 98 %; Endo -> Macrophage, Oligo -> OPC; 74-113 training cells over 12 libraries);
  nuclei-vs-cell latent purity worse (0.78 vs 0.64). Positive: female (4-c3A) vs male (1-c3A) target-library decodes
  give Xist 3.6 vs 0.0 and genome-wide r 0.655 with the real female-minus-male vector. Verdict: supplementary only;
  usable panel = sex-placement decode. Possible fix if ever needed: decoder-only library + UNKNOWN dropout 0.5 (not planned).

**Why:** paper Fig 2 needs ablations on the frozen architecture; scripts existed, only the exclusion flags were missing.
**How to apply:** when the chain finishes, read `run/benchmark/test/results_ablations/summary.md` and `by_donor/by_donor.md`
per the README's "how to read" section, then write results into the README and the storyline note. For the N-1 arms the
label-free number of record is the `kqmap` decode (kNN-type cannot name an unseen type).
