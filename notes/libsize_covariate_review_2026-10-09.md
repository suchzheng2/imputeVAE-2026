# Library-size covariate retrain: what has been done, what the evidence says, and the plan (2026-10-09)

Status: **nothing has been trained with an explicit library-size covariate yet.** `imputeVAE/models_cond.py`
has no such input; the only library-size handling in the code base is the multiplicative scale factor of the
NB models (`mu_nb = library * softmax(rho)`), which the paper model (`VAE_JSD_Cond`, MSE on log1p) does not use.
The item has been "next" since 2026-09-18 (`notes/paper_status_2026-10-06.md`, gap A2).

## 1. Chronology of what has been done

| Date | What | Finding | Where |
|---|---|---|---|
| 2026-04-07 | Latent diagnostic on the old two-stage pct5v2 cond model (16,439 genes, job 3260550) | 28/32 latent dims active, posterior std 0.7-0.8 (no collapse then); 82% of latent variance within-CT; decoder, not encoder, blamed for mean-regression | `notes/experiments/SUC-6_variance_retention/latent_diagnostic/`, memory `project_latent_diagnostic.md` |
| 2026-04-15 | Library-size analysis on the same model | one latent dim (31) encoded library size; after proportion normalisation the per-cell std ratio fell from 0.55 to 0.17, i.e. most "retained variance" was library-size scaling. Prashant agreed to try CPM input | memory `cpm_experiment.md` |
| 2026-04-16 to 04-29 | **CPM + log1p input** (size-normalised input; stage 1 job 3964307, stage 2 cond job 4061786) | stage-2 gene corr 0.703 vs 0.963 for log1p. Root cause: CPM values are ~3x larger, encoder is near-linear, mu reached ±500, the Monte-Carlo JSD saturated at ln 2 with zero gradient. Soft L2 constraint on mu proposed; probe jobs 4112282/4112283 launched, outcome never written up (see §1a) | `cpm_experiment.md` |
| 2026-05-28 | Normalised-loss factorial (16 runs, log1p vs CPM × scalar/emb8 × dec/encdec × 1/2-stage) | **log1p > CPM again** on cross-modal correlation; one CPM config collapsed. Only CT-level metrics were compared, not variance structure | `notes/experiments/session_2026-06-12_results.md`, `design_decisions.md` |
| 2026-07-18 | Paper model trained (`run_2026-07-18-19-40-51_570984`): log1p, scalar cond on enc+dec, normloss, JSD weights 0.1/0.1/alignment 1.0, 32-dim latent | | |
| 2026-09-18 | Variance diagnostic on the paper model | real-cell within-CT variance is 52% Poisson / 25% library size / 23% biological; **82% of imputed within-CT variance is explained by the input nucleus library size**; participation ratio 1.1-2.1 vs 8-18 in real cells; like-for-like retention 0.12. Decision: no variance loss; remove library size from the latent first | `notes/experiments/variance_zero_nonzero_fullmodel/README.md` |
| 2026-09-18 | Latent-usage check (job 10910710) | **posterior collapse**: 5-6 of 32 dims active; dim 19 = library size (r = -0.94 within CT) alone reproduces 95% of imputed within-CT variance; the rest is 4 cell-identity dims. Model is "cell-type code + library size" | `.../latent_usage/README.md` |
| 2026-09-24 | Variance-rank check (Prashant's request) | per-gene variance ranks preserved (Spearman 0.92-0.93), distribution stretched not collapsed. Magnitude is not the issue; structure is | `.../variance_rank/README.md` |
| 2026-10-09 | Split-aware benchmark (this session) | on held-out test nuclei the cVAE's within-CT Spearman against its own input nucleus is 0.11, two-sample AUROC vs real cells 0.97 (floor 0.51), energy distance 7.5 (floor 0.6, CT-mean 25). Confirms the per-cell weakness independently of the diagnostic | `notes/benchmark_review_2026-10-09.md` |

### 1a. The April soft-constraint probes
Jobs 4112282 and 4112283 (`jobs/probe/run_probe_cpm_softcon_l0*.sh`) left one-line logs: they never ran past
start-up, and nothing was written up. The soft L2 constraint on mu was therefore **never tested**. It is not
needed for the covariate plan below (the log1p input keeps mu in range), but the record should say so.

## 2. What the evidence says about the two ways to remove library size

**Size-normalised input (CPM) has been tried twice and lost both times** on cell-type-level metrics (April:
JSD saturation; May: factorial with normalised losses, still worse, one collapse). Two caveats: (i) the comparison
was on CT-level metrics, which the September work shows are saturated and insensitive to the thing we care about;
(ii) CPM also removes the information the decoder needs to put the output on a cell library scale, and nothing
in that pipeline put it back. So CPM is not disproven for the *variance-structure* question, but it is the
weaker candidate and has an awkward inference story.

**Explicit covariate (scVI-style) has never been tried here.** It keeps the log1p input the model is known to
train well on, adds one observed scalar to encoder and decoder, and makes the output library scale an explicit
inference-time choice instead of a side effect of latent dimension 19. This is the recommended route, and it is
what both September reports asked for.

**The regulariser is the second half of the problem.** 26 of 32 dims sit at the prior. With normalised losses the
per-modality JSD terms (weight 0.1 each, divided by ln 2) and the alignment JSD (weight 1.0) are enough to collapse
them. Removing library size from the latent frees dim 19 but does not, by itself, make the encoder put biology
into the other 26. Expect the first covariate-only run to show fewer, not more, active dims unless the
regulariser is also relaxed. Hence two arms (below).

## 3. Proposed experiment (2 to 3 runs, 12 h GPU / 400 GB each)

### Model change (`VAE_JSD_Cond`, new flag `--libsize_cond True`)
- `s = (log(total raw counts over the 10,134 genes) - m) / sd`, with `m`, `sd` fixed from the training
  partition (both modalities pooled; the modality scalar already carries the modality offset). Stored in
  `arguments.json` so inference uses the same constants.
- Encoder input: `[x, modality_scalar, s]`; decoder input: `[z, modality_scalar, s]`. Everything else unchanged.
- Training: `s` is the sample's own library size (reconstruction stays well-posed).
- Cross-decoding (nuclei -> cell) gets an explicit `target_libsize` argument:
  - `ct_median_cell` (**primary for the diagnostic**): the median training-cell library of the nucleus's cell
    type. Within a type every output then shares one scale, so any within-type variation in the output is
    latent-driven. This is the clean test of "is there biology in z".
  - `quantile_map` (for actual imputation): map the nucleus's library-size quantile within its type onto the
    training-cell library distribution of that type. Keeps per-cell size variation, on the cell scale.
  - `source` (ablation only): the nucleus's own library; reproduces the current behaviour.
- The MSE objective is unchanged. Nothing here forces more variance; it only stops variance from being a
  library-size artefact.

### Regulariser arms
| Arm | Change | Purpose |
|---|---|---|
| A | covariate only, JSD weights as in the paper model (0.1 / 0.1 / 1.0) | isolates the covariate effect |
| B | covariate + per-modality JSD weights 0.01 (alignment stays 1.0) | tests whether the collapse is regulariser-driven |
| C (optional) | covariate + free-bits on an analytic per-dimension KL (floor 0.05 nats/dim) replacing the per-modality MC-JSD | the "right" fix if B helps but destabilises alignment |

Seed-replicate control: the two paper-config seeds submitted today (job 12646907) give the run-to-run noise of
active-dim counts and variance metrics, so arm A vs paper is interpretable.

### How to judge (all exist, two need a `--run_dir` argument)
1. `analysis/latent_usage_check.py`: active dims (posterior var < 0.9 or KL > 0.05 nats), library-size R² per
   dim (should be ~0 for all dims now), per-CT effective dims. **Primary readout.**
2. `analysis/variance_zero_nonzero_fullmodel.py`: library-size share of imputed variance (target: from 82% to
   near the real-cell 24% or below), participation ratio (target: move from 1-2 toward 8-18), like-for-like
   retention, gene-gene structure agreement.
3. `analysis/benchmark/predict_cvae.py` + `evaluate_all.py` on the test partition: within-CT Spearman vs input,
   energy distance and two-sample AUROC vs real cells, and the CT-level numbers as a guard (gene corr should
   stay near 0.90; a drop below linreg_shrunk's 0.78 would be a failure).
4. Variance-rank Spearman (should stay ~0.9).

Decision rule after the runs: if arm A or B raises active within-type dims above ~8 and the library-size share
falls below 30% without losing CT-level accuracy, the per-cell framing of the paper survives and arm C / a
structure-aware loss becomes worth pursuing. If not, the paper is framed around the cell-type-level translation
plus biology (paper_status gap A2, second branch), and the covariate model is still the better-specified model to
report because its output scale is explicit.

## 4. Things to watch
- `load_raw_data` loads the whole dense h5ad (41 + 24 GB); the 400 GB job is unavoidable without reworking the
  training loader. Not in scope today.
- Library size for `s` must be computed on the 10,134 benchmark genes (same as inference), not on all 26k genes.
- `decode_cross` is used by every evaluation script (`export_cvae_imputed.py`, `eval_splitaware.py`,
  `predict_cvae.py`, inline eval). The new argument needs a default (`ct_median_cell` requires cell-type labels,
  so default to `source` for backward compatibility and pass the choice explicitly in the benchmark scripts).
- The scalar modality flag and `s` are both scalars; keep them as two separate inputs, do not fold them.

## 5. Implementation and submission (2026-10-16)

Implemented as planned in §3; default-off, the paper checkpoint still loads strictly (smoke-tested).

| Piece | Where | What |
|---|---|---|
| Covariate | `imputeVAE/models_cond.py` `VAE_JSD_Cond(libsize_cond=True)` | `s = (log1p(total counts over the 10,134 genes) - mean) / std`, recovered from the log1p input by `expm1(x).sum()`; appended to the encoder input (after the modality scalar) and to the decoder input. `mean`/`std` are registered buffers (saved in `model.pt`), set by `set_libsize_stats`. `forward` uses the sample's own library; `decode_cross(..., target_log_libsize=)` takes a float (per-type median) or a per-sample tensor; `None` = source library. `decode_cross_train` requires the target library when the covariate is on. New `encode_mu` helper. |
| Training | `imputeVAE/cond_training.py` | `compute_libsize_stats` over the TRAIN loaders (both modalities pooled for mean/std; per-type medians and 0..100 % quantiles for cells and nuclei) -> `<run>/libsize_stats.json`; inline cross-decode eval (train and valid) decodes at the per-type median training-cell library (`--libsize_target ct_median_cell`, or `source`). |
| Resume | `cond_training.py`, `train.py --run_name` | fixed run folder; `run_state.json` (last checkpoint epoch + the epoch-0 `recon_loss_norm`) written with every checkpoint; `--continue_training` restores the normaliser and truncates `loss.txt` past the checkpoint. Motivation: the seed-2 replicate was preempted 9 times and restarted from scratch each time (37 GB of junk folders, see `training_runs.md`). |
| Flags | `train.py` | `--libsize_cond`, `--libsize_target {ct_median_cell,source}`, `--run_name` |
| Inference | `analysis/benchmark/predict_cvae.py --target_libsize {auto,ct_median_cell,quantile_map,source}` | `ct_median_cell` (default for covariate models) uses the eval nucleus's type label and the training-cell median from `libsize_stats.json`; `quantile_map` maps the nucleus's within-type library quantile (vs training nuclei) onto the training-cell quantiles of that type; `source` = ablation. `jobs/benchmark/predict_cvae.sh` reads `CVAE_TARGET_LIBSIZE`. |
| Jobs | `jobs/stage2/run_single_stage_cvae_libsize.sh`, array **13445566** | `_1` = arm A (JSD 0.1/0.1), `_2` = arm B (JSD 0.01/0.01), both seed 1, alignment 1.0, 600 epochs, dumps off. Folders `run/run_2026-10-16_libsize_armA_jsd0.1_seed1`, `run/run_2026-10-16_libsize_armB_jsd0.01_seed1`. Matched control = seed-1 replicate `run_2026-10-09-19-13-44_611868` (benchmarked as `cvae_seed1`). |

Diagnostics are parametrised (2026-10-16), defaults = paper run, so the September numbers reproduce unchanged:
- `analysis/export_cvae_imputed.py --run_dir <run> --output_dir <dir> --target_libsize ct_median_cell` (all-data
  imputed matrix for the two scripts below; writes `export_args.json`);
- `analysis/latent_usage_check.py --run_dir <run> --out <dir> --imputed <dir>/imputed_cell_from_nucleus.npy
  [--target_libsize ct_median_cell|source]` (encoder/decoder inputs now built with the covariate; the ablations decode
  every nucleus at its type's median training-cell library);
- `analysis/variance_zero_nonzero_fullmodel.py --imputed <...> --out <dir>`.
Arm C (free bits) is not implemented.

Recipe once job 13445566 finishes (per arm, GPU not required except export):
1. `CVAE_RUN_DIR=run/run_2026-10-16_libsize_armA_jsd0.1_seed1 CVAE_OUT=run/benchmark/test/cvae_libsize_armA CVAE_TARGET_LIBSIZE=ct_median_cell sbatch --export=ALL jobs/benchmark/predict_cvae.sh`
   (repeat with `quantile_map` -> `cvae_libsize_armA_qmap` and `source` -> `cvae_libsize_armA_source`), then `jobs/benchmark/evaluate_all.sh`.
2. `python analysis/export_cvae_imputed.py --run_dir <run> --output_dir run/libsize_armA_eval --target_libsize ct_median_cell` (loads the 24 GB nuclei h5ad; ~60 GB job).
3. `python analysis/latent_usage_check.py --run_dir <run> --out notes/experiments/libsize_covariate/armA/latent_usage --imputed run/libsize_armA_eval/imputed_cell_from_nucleus.npy`
4. `python analysis/variance_zero_nonzero_fullmodel.py --imputed run/libsize_armA_eval/imputed_cell_from_nucleus.npy --out notes/experiments/libsize_covariate/armA/tables`
Compare against `cvae_seed1` (benchmark) and the September paper-model tables.

Seed-replicate control (from the benchmark, §5a of `benchmark_review_2026-10-09.md`): across the paper run and
seeds 1-2 the variance ratio is 0.151 ± 0.009 and within-CT Spearman vs input 0.114 ± 0.001, so arm A has to move
the variance ratio by more than ~0.03 or the library-size share (82 %) by far more than that to count as an effect.
Same-seed GPU nondeterminism alone changes within-type per-gene variances by r 0.84-0.99 between attempts.

## 6. Results (2026-10-23)

Both arms trained to completion on 2026-10-17 (arm A job 13446475_1, arm B 13445566_2, ~12 h each on r813u29n03 A100s,
600 epochs, final valid inline cross-decode gene corr 0.989 / 0.987). Evaluated 2026-10-23: benchmark predictions
13779998-13780003, `evaluate_all` 13780004 (pending at the time of writing; fill in §6.4), diagnostics 13780170 + addon
13783881, cross-chemistry test 13783381. All imputed matrices below are decoded at the per-type median library
(`ct_median_cell`) unless stated, i.e. within-type output variation is latent-driven only.

### 6.1 Latent usage (`notes/experiments/libsize_covariate/arm{A,B}/latent_usage/`)

| | paper model | arm A (JSD 0.1) | arm B (JSD 0.01) |
|---|---|---|---|
| dims with KL > 0.05 nats (nuclei) | 5 | 4 | 6 |
| dims with within-type var > 0.01 | 6 | 4 | 3 |
| dims with within-type SNR > 1 | 2 (19 = library, 17 = identity) | 2 (22, 24; both identity) | 1 (16; identity) |
| R² of log library from all latents, within type | 0.96 | 0.70 | 0.74 |
| share of output within-type variance removed by projecting out the library direction | **0.85** | **0.002** | **-0.002** |
| output within-type variance, sum over genes | 683 | 61.5 | 62.2 |
| latent participation ratio, pooled within type | 2.9 | 3.0 | 1.2 |
| per-type library share of within-type latent variance (median) | 0.59 | 0.04 | 0.03 |
| ablation: dims needed for 99 % of output within-type variance | 3 (19, 17, 4) | 3 (22, 24, 15) | 3 (16, 4, 28) |

Reading. The covariate did exactly what it was meant to do: the library-size dimension is gone (no dimension has
R² > 0.01 with library size; projecting the library direction out of the latent no longer changes the output).
What is left of within-type output variation is **11x smaller** and rides on 2 to 3 cell-identity axes (the dims
that separate types also vary a little inside a type: continuous identity, e.g. the L2/3 IT - L4/5 IT gradient). Arm
B (10x weaker per-modality JSD) activates more dims in total (11 with total var > 0.01 vs 4) but no more
*within-type* dims (3 vs 4); its single dominant dim 16 carries 92 % of the residual. So the collapse is not
regulariser-driven: with the scale explained away, the MSE objective has nothing left to put in the latent beyond
identity. The encoder still carries library information (R² 0.70, spread over many low-variance dims) that the
decoder does not use.

### 6.2 Variance structure (`.../arm{A,B}/tables/`, diagnostic scripts of 2026-09-18)

| | paper | arm A | arm B |
|---|---|---|---|
| median within-type var ratio imputed / real cell | 0.077 | 0.0021 | 0.0015 |
| library-size R² per (type, gene), imputed (median) | 0.84 | 0.025 | 0.025 |
| participation ratio of imputed within-type variation (median over 9 types; real cells 8-18) | 1.44 | 1.66 | 1.56 |
| share of imputed within-type variance in the real-cell top-10 PC subspace | 0.94 | 0.71 | 0.83 |
| gene-gene correlation structure vs real cells, Spearman (median over 9 types; raw nuclei 0.80) | 0.33 | 0.48 | 0.67 |

Reading. At fixed library the output is almost a point per type (var ratio 0.002), still one- to two-dimensional.
The small residual is *better structured* than the paper model's (gene-gene Spearman 0.48 / 0.67 vs 0.33): what the
identity axes carry is biological covariation, just very little of it. The decision rule of §3 is not met (needed
> ~8 active within-type dims and library share < 30 % *with retained variance*): library share is 0 % but the
variance went with it.

### 6.3 Cross-chemistry (v2 nuclei -> v2 cells, 76,525 nuclei, 17 types; `analysis/cross_dataset_eval.py`, job 13783381; `run/cross_dataset_eval/v2_batch_*`)

| decode library | gene corr (median) | R² (median) | overall corr | QQ corr | Wasserstein | var ratio | shift |
|---|---|---|---|---|---|---|---|
| paper model (no covariate; July result reproduced) | 0.593 | -0.03 | 0.872 | 0.894 | 0.263 | 0.077 | 0.162 |
| arm A, nucleus's own library (`source`) | 0.604 | -1.30 | 0.850 | 0.865 | 0.270 | 0.027 | 0.254 |
| arm A, v3 *training*-cell per-type median | 0.880 | -4.84 | 0.937 | 0.699 | 0.490 | 0.054 | 0.380 |
| arm A, **v2-cell per-type median** | **0.927** | **0.723** | 0.963 | 0.789 | 0.304 | 0.023 | 0.076 |
| arm A, **quantile map onto v2 cells** | **0.935** | **0.748** | 0.965 | 0.867 | **0.247** | 0.161 | 0.073 |

Per type (v2-cell library): every neuronal type 0.95-0.98; Astro 0.84, Oligo 0.84-0.86, OPC 0.91, Endo 0.86-0.89,
Macrophage 0.85 (paper model: 0.78 / 0.78 / 0.84 / 0.78 / 0.50). The v3 test-partition benchmark number for the
same model family is 0.90, so on unseen chemistry the covariate model is *at* its in-distribution accuracy.

Reading. **Gap B2 is closed: the cross-chemistry drop was a library-scale shift, nothing else.** The paper model
decodes v2 nuclei (small libraries) into small-library "cells"; the covariate model decodes the same latent at
whatever scale it is told and gets the type means right. The information it needs is the target chemistry's
whole-cell RNA content per type, which is a *parameter of the application*, not something a nucleus can supply:
with the v3 training-cell libraries (3x larger than v2 cells) R² is -4.8. The quantile map is the mode to use for
imputation (keeps per-cell size variation on the target scale; best Wasserstein). For nuclei-only datasets a
label-free variant (one global cell/nucleus ratio per chemistry, applied to the nucleus's own library, as done
for the label-free scVI decode in the benchmark) should be added and tested on v2; not done yet.

### 6.4 Benchmark on the v3 test partition (job 13780004; label-free scale decodes 13819587/8 + eval 13819596; 2026-10-23)

Guard passed (every decode >= 0.78). Full tables in `run/benchmark/test/results/` (30 method dirs) and
`run/benchmark/test/results_libfree_scale/` (label-free scale question only). Test partition, 8,034 nuclei, 17 types;
floors: energy 0.62, two-sample AUROC 0.51. "Label" = the eval nucleus's cell-type label is used to pick the decode library.

| decode | label at inference | gene corr | R² | delta r | var ratio | W1 HVG | energy | within-CT Spearman | clf acc | DE overlap (logFC) | coexp L5 IT | neuron/glia |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| paper model, seed 1 (no covariate) | no | 0.903 | 0.682 | 0.932 | 0.145 | 0.357 | 6.2 | 0.115 | 0.967 | 0.824 | 0.795 | 4.8 |
| **arm A, ct_median_cell** | yes | **0.970** | **0.927** | 0.982 | 0.019 | 0.530 | 19.5 | 0.019 | 0.967 | 0.864 | 0.420 | 4.5 |
| **arm A, quantile_map** | yes | **0.970** | **0.930** | 0.982 | 0.275 | **0.279** | **2.6** | 0.116 | 0.966 | **0.857** | 0.789 | 4.6 |
| arm A, source (nucleus's own library) | no | 0.839 | -0.404 | 0.786 | 0.052 | 0.830 | 82.3 | 0.111 | 0.963 | 0.776 | 0.732 | 6.4 |
| arm A, global_shift (one ratio 2.73 per chemistry) | no | 0.869 | 0.718 | 0.927 | 0.101 | 0.407 | 8.3 | 0.115 | 0.966 | 0.852 | 0.760 | 3.8 |
| arm A, global_quantile_map (pooled) | no | 0.843 | 0.586 | 0.894 | 0.177 | 0.357 | 6.1 | 0.116 | 0.966 | 0.842 | 0.781 | 6.3 |
| arm A, knn_cell_median (K=50 nearest training cells in latent) | no | 0.912 | 0.775 | 0.949 | 0.059 | 0.528 | 13.8 | 0.035 | 0.966 | 0.844 | 0.684 | 4.1 |
| **arm A, knn_quantile_map** (kNN nuclei rank -> kNN cells) | no | **0.921** | **0.809** | 0.955 | 0.263 | 0.424 | **5.6** | 0.102 | 0.966 | 0.852 | 0.774 | 4.5 |
| arm A, knn_type_median (type predicted from 50 nearest training nuclei, 97 % acc.) | no (training labels only) | 0.959 | 0.904 | 0.975 | 0.021 | 0.528 | 19.3 | 0.020 | 0.967 | 0.862 | 0.419 | 3.9 |
| **arm A, knn_type_quantile_map** (same, per-type quantile map) | no (training labels only) | **0.962** | **0.909** | 0.976 | 0.269 | **0.279** | **2.6** | 0.116 | 0.966 | 0.860 | 0.792 | 3.9 |
| arm B, ct_median_cell | yes | 0.961 | 0.909 | 0.975 | 0.015 | 0.533 | 19.5 | 0.017 | 0.970 | 0.840 | 0.418 | 4.3 |
| arm B, quantile_map | yes | 0.960 | 0.906 | 0.974 | 0.268 | 0.293 | 3.3 | 0.117 | 0.969 | 0.835 | 0.819 | 4.4 |
| scGen (3 seeds) | yes | 0.975 | 0.771 | 0.980 | 0.075 | 0.369 | 15.9 | 0.053 | 0.946 | 0.864 | 0.442 | 3.6 |
| scANVI + per-type library (3 seeds) | yes | 0.962 | 0.719 | 0.970 | 0.074 | 0.387 | 16.4 | 0.048 | 0.930 | 0.809 | 0.442 | 3.0 |
| scVI label-free (3 seeds) | no | 0.857 | 0.433 | 0.910 | 0.151 | 0.341 | 9.2 | 0.134 | 0.922 | 0.796 | 0.691 | 2.5 |
| CT-mean (ceiling) | yes | 0.981 | 0.956 | 0.990 | 0 | 0.625 | 25.1 | n/a | 1.000 | 0.905 | n/a | 4.7 |

Reading.
1. **Given the per-type cell library, the covariate model sits in the label-using group at the ceiling**: gene corr 0.970
   (scGen 0.975, CT-mean 0.981) and **R² 0.93, the best of any method bar CT-mean** (scGen 0.77, scANVI 0.72): the
   per-gene scale is right, not only the ranking. Arm B (JSD 0.01) is 0.01 lower everywhere; JSD 0.1 stays.
2. **`quantile_map` is the best row in the whole benchmark on every within-type / population metric**: energy 2.6
   (floor 0.6; next best is the paper model at 6.2), Wasserstein 0.279, DE overlap 0.640 (top-100) / 0.857 (logFC),
   coexpression 0.789 (= paper model), variance ratio 0.275 (closest to 1 of all methods, but it is imposed from the
   nucleus's library rank and is still a one-axis quantity, §6.2). Classification 0.966; OPC recall 0.40 -> 0.73/0.80,
   Sncg 0.71 -> 0.82 (per-type recall table).
3. **Both of those rows use the label.** The label-free decodes of the same model are *below* the paper model on gene
   corr: source 0.839, pooled quantile map 0.843, one global ratio 0.869 (R² 0.72 is still above the paper model's 0.68
   and far above label-free scVI's 0.43). Cause: the per-type cell/nucleus library ratio in the training data ranges
   from 1.2x (Astro, Endo, L5 ET) to 3.7x (Lamp5); all other neurons 2.5-3.3x, Oligo/OPC 2.7-3.1x. A single 2.7x
   overshoots glia and L5 ET, and across-type gene patterns (what gene corr measures) absorb the error while within-type
   profile shape does not (per-type profile corr is unchanged). The paper model without the covariate had learned this
   per-type ratio implicitly in its latent (§6.1), which is where its 0.03 advantage over label-free covariate decodes
   comes from.
4. So the covariate model cleanly separates *what* (identity, from the latent) from *how much* (RNA content, which
   must be supplied). Three sources for "how much": the label (0.970), a chemistry-wide ratio (0.869), or the model's
   own identity code. Two label-free readings of the latent were tested (eval 13822258, `results_libfree_scale/`):
   (a) the K=50 nearest training *cells* in the aligned latent (96 % share the nucleus's type) supply the library:
   `knn_cell_median` 0.912 / R² 0.78, `knn_quantile_map` **0.921 / 0.81, energy 5.6**: the first fully label-free
   decode that beats the paper model on every headline metric (0.903 / 0.68 / 6.2) and label-free scVI by 0.06.
   Its library is a *local* one (r 0.39 with the type median; neighbours' libraries vary along within-type latent
   gradients), which is why it stays 0.05 short of the labelled decode. (b) The type itself can be predicted from the
   50 nearest training *nuclei* (majority vote: 97 % accuracy on the test nuclei, per-type library r 0.98 with the
   truth); `knn_type_median` / `knn_type_quantile_map` use training labels only, none at inference (eval 13822366;
   vote agreement 0.99, predicted-type accuracy 0.972 on the test nuclei). **Result: 0.959 / R² 0.90 and
   0.962 / 0.91 / energy 2.6**, i.e. within 0.01 of the labelled decode on every metric, level with scANVI (0.962,
   which *does* use the label at inference) and above labelled scVI (0.958) and scGen's R² (0.77). This is the
   reported nuclei-only decode: `knn_type_quantile_map`. The honest ladder for the paper, all one model: label at
   inference 0.970; label-free with training labels 0.962; label-free with no labels anywhere 0.921 (kNN cells);
   one scalar per chemistry 0.869; nucleus's own library 0.839.

### 6.4a Label-free scale on unseen chemistry (v2; jobs 13819589_5-7, 13822257_8-10; `run/cross_dataset_eval/v2_batch_armA_*`)

| decode library (arm A, v2 nuclei -> v2 cells) | label | gene corr | R² | overall corr | QQ corr | Wasserstein | var ratio | shift |
|---|---|---|---|---|---|---|---|---|
| paper model (no covariate) | no | 0.593 | -0.03 | 0.872 | 0.894 | 0.263 | 0.077 | 0.162 |
| v2-cell per-type median | yes | 0.926 | 0.723 | 0.963 | 0.789 | 0.304 | 0.023 | 0.075 |
| quantile map onto v2 cells (per type) | yes | 0.935 | 0.747 | 0.965 | 0.867 | 0.247 | 0.161 | 0.073 |
| pooled quantile map onto v2 cells | no | 0.859 | 0.580 | 0.951 | 0.867 | 0.244 | 0.132 | 0.091 |
| nucleus library + global v2 shift (ratio 3.84) | no | 0.842 | 0.550 | 0.946 | 0.871 | 0.259 | 0.095 | 0.094 |
| nucleus library + global v3 *training* shift (ratio 2.73; no v2 cells used) | no | 0.805 | 0.288 | 0.938 | 0.867 | 0.269 | 0.075 | 0.127 |
| latent kNN onto unlabelled v2 cells (median / quantile map) | no | 0.833 / 0.849 | 0.21 / 0.34 | 0.940 / 0.944 | 0.880 / 0.902 | 0.259 / 0.232 | 0.083 / 0.223 | 0.134 / 0.123 |
| latent kNN onto v3 training cells + global v2 shift | no | 0.838 | 0.560 | 0.946 | 0.860 | 0.254 | 0.098 | 0.093 |
| predicted type (kNN over v3 training nuclei; 77 % acc. on v2) -> v2 per-type cell median | no (v2 cell labels for the table) | 0.888 | 0.649 | 0.955 | 0.791 | 0.302 | 0.037 | 0.084 |
| predicted type -> v3 per-type median + global v2 shift (no v2 label anywhere) | no | 0.814 | 0.494 | 0.942 | 0.762 | 0.303 | 0.039 | 0.096 |
| **reference-classifier type (`pca_lr`, 84 % acc. on v2; 2026-10-27, 14189131_13) -> v2 per-type cell median** | no (v2 cell labels for the table) | **0.903** | 0.656 | 0.958 | 0.754 | 0.298 | | |
| reference-classifier type -> v3 per-type median + global v2 shift (no v2 label anywhere; 14189131_14) | no | 0.833 | 0.505 | 0.946 | 0.724 | 0.301 | | |

Reading: the label-free covariate decodes on unseen chemistry are 0.84-0.86 (paper model 0.59), i.e. most of the
cross-chemistry gain needs no label; the per-type remainder (0.86 -> 0.93) is the same per-type-ratio effect as in
the v3 benchmark. Per-type profile corr is identical between pooled and per-type decodes (neurons 0.94-0.98, Astro 0.86,
Oligo 0.82, Macrophage 0.84). The chemistry ratio itself differs (v3 2.7x, v2 3.8x) and does not transfer unchanged
(0.805 with the v3 ratio), so one scalar per chemistry has to be stated, from a handful of unlabelled target cells or
from the literature. **With the kNN modes (13822257, 13822365):** the predicted type is only 77 % accurate on v2 nuclei
(97 % on v3; the latent of unseen-chemistry nuclei sits less cleanly on the training nuclei), so the predicted-type
decode reaches 0.888 (label 0.926); latent-cell kNN modes 0.83-0.85 do not beat the pooled map here. The cross-chemistry
gap for a nuclei-only user is therefore 0.86-0.89 vs 0.93, and it is a type-prediction problem, not a scale problem:
a better cross-chemistry type assignment (e.g. a reference-mapping step on the nuclei themselves) would close it.

**Reference classifier on v2 (2026-10-27, job 14189131_13-14; `cross_dataset_eval.py --type_predictor pca_lr`, the
log-CPM / 3,000 HVG / PCA 50 / logistic-regression classifier of `analysis/external/type_classifier.py` fitted on the
v3 training nuclei, v3 validation accuracy 0.983).** Type accuracy on v2 nuclei 0.77 -> 0.84, gene corr 0.888 -> 0.903
(label 0.926) with the v2 per-type median, 0.814 -> 0.833 with the v3 median + global shift; R² 0.65 / 0.51; QQ corr
drops slightly (0.79 -> 0.75). Half of the remaining gap is closed. Per-type gene corr is unchanged except Oligo
(0.78 -> 0.84). What is still misassigned (crosstab `run/cross_dataset_eval/v2_batch_armA_lrtype_v2/`): **Pvalb -> Sst
56 %** (3,532 nuclei; the largest error and not a depth effect, since Pvalb/Sst are 0.97-0.99 on v3), L6 IT <-> L5 IT
(0.48 / 0.45 split, 7,580 nuclei), L6 IT Car3 -> L5 IT 0.53, L5 ET -> L6 CT 0.25, and the small types (Sncg -> Vip,
Macrophage -> OPC). So the v2 label-free number for the paper is **0.90 (0.93 with the label)**, and the residual is
a handful of systematic cross-chemistry confusions between neighbouring neuron types, which a classifier trained with
v2 nuclei in the reference (or batch-corrected PCs) would be expected to remove. `predicted_types_*.csv` and the
crosstab are saved per run.

### 6.7 Latent-width control (8 / 4 dims; runs `run/run_2026-10-23_libsize_armA_latent{8,4}_seed1`, job 13784373; eval 13823078 -> `run/benchmark/test/results_latent_width/`; diagnostics 13822273_3/4 + 13823088 -> `notes/experiments/libsize_covariate/latent{8,4}/`)

Arm A configuration (covariate, JSD 0.1/0.1, seed 1) with `--latent_size 3` (8 dims) and `2` (4 dims); 600 epochs,
11.4 / 11.3 h on an A100. Decoded at the per-type cell library (label) unless stated.

| | arm A (32 dims) | 8 dims | 4 dims |
|---|---|---|---|
| final training loss (normalised) / nuclei recons / cell recons | 0.958 / 0.221 / 0.253 | 0.950 / 0.224 / 0.262 | 0.946 / 0.230 / 0.279 |
| active latent dims (KL > 0.05 nats; within-type var > 0.01) | 4; 4 | 2; 2 | 1; 1 |
| latent participation ratio, pooled within type | 3.0 | 1.7 | 1.0 |
| R² of log library from all latents, within type | 0.70 | 0.21 | 0.19 |
| output within-type variance, sum over genes (ct_median_cell) | 61.5 | 9.5 | 3.4 |
| gene-gene Spearman of imputed vs real within type (median of 9 types; raw nuclei 0.80) | 0.48 | 0.19 | 0.02 |
| share of imputed within-type variance in the real-cell top-10 PCs (median) | 0.71 | 0.54 | 0.22 |
| **gene corr / R² (ct_median_cell)** | **0.970 / 0.927** | 0.957 / 0.902 | 0.910 / 0.802 |
| gene corr / R² / energy (quantile_map) | 0.970 / 0.930 / 2.6 | 0.956 / 0.900 / 3.0 | 0.913 / 0.815 / 16.6 |
| gene corr (knn-free pooled quantile map, no label) | 0.843 | 0.829 | 0.779 |
| classification accuracy of imputed cells | 0.967 | 0.935 | 0.557 |
| types with recall < 0.5 | none (OPC 0.73) | L5 ET 0.00, Sncg 0.00 (OPC 0.53) | all but L5 IT, L6 CT, Macrophage, Sst, Vip |
| coexpression within L5 IT (ct_median / quantile_map) | 0.42 / 0.79 | 0.23 / 0.74 | 0.19 / 0.58 |

Reading. The model uses even fewer dimensions than it is given: 8 -> 2 active, 4 -> 1 active, and in each case the
within-type residual shrinks further (61 -> 9 -> 3) and loses its gene-gene structure (0.48 -> 0.19 -> 0.02). The
per-type decode is remarkably tolerant: 2 latent dims still give gene corr 0.957 / R² 0.90 and 0.935 classification,
i.e. a two-dimensional identity code places 15 of 17 types correctly, with the two small, intermediate types (L5 ET,
Sncg) merged into neighbours. One dimension (4-dim run) orders the types along a line and only five remain
separable (accuracy 0.56), yet the gene-level correlation is still 0.91 because the large types sit at distinct
positions on that line. So (i) capacity is not what limits within-type variation: the 32-dim model already has 28
idle dims and shrinking the budget only removes identity resolution, never adds within-type structure; (ii) the
cell-type-level translation needs 2-4 identity dimensions, which is the "low-rank transformation" statement for the
paper (Fig 4); (iii) 32 dims stays the reported configuration (best identity resolution; same within-type behaviour).
Loss decreased with fewer dims only because the normalised JSD terms are larger (0.066 / 0.075 vs 0.023), not
because reconstruction improved (it got worse: 0.224 / 0.230 vs 0.221 nuclei).

### 6.5 Decision and next steps (executed 2026-10-23 unless marked)

1. **The per-cell framing is withdrawn.** Frame the paper around the cell-type-level translation + biology (paper_status
   gap A2, second branch); report the covariate model as the model (explicit output scale; chemistry-robust).
   The latent finding becomes a result: a nucleus determines its type-level cytoplasm plus RNA content, not its
   individual cytoplasm, at 10x resolution. See `notes/paper_storyline_2026-10-23.md`.
2. **Arm C (free bits) is dropped**: arm B shows the collapse is not regulariser-driven.
3. **Latent-width control**: job 13784373 (`jobs/stage2/run_single_stage_cvae_libsize_latent.sh`; 8 and 4 dims, arm A
   config, seed 1) **trained to completion 2026-10-23** (11.4 h / 11.3 h on r813u29n03; final loss 0.950 / 0.946 vs
   0.958 for 32 dims; folders `run/run_2026-10-23_libsize_armA_latent{8,4}_seed1`). Benchmark decodes 13822181-8
   (`cvae_libsize_latent{8,4}[_qmap|_source|_gqmap]`), full table 13822189 -> 13822272, latent-usage / variance
   diagnostics 13822273 (`notes/experiments/libsize_covariate/latent{8,4}/`). **Results §6.7**: 8 dims -> 2 active,
   gene corr 0.957 / clf 0.935 (L5 ET, Sncg lost); 4 dims -> 1 active, 0.910 / 0.557; within-type structure shrinks
   with width. Capacity is not the limit; 32 dims stays.
   Expectation: 8 dims unchanged at the type level; 4 dims tests whether identity needs more than 4. An ablation, not a fix.
4. **Label-free scale mode: done 2026-10-23** (`predict_cvae.py --target_libsize global_shift | global_quantile_map |
   knn_cell_median | knn_quantile_map`; `cross_dataset_eval.py --target_libsize global_*_target | global_shift_train |
   knn_*_target`; `compute_libsize_stats` now also writes pooled medians/quantiles). Results §6.4 / §6.4a: one global
   ratio gives 0.869 on v3 and 0.84-0.86 on v2 (label 0.97 / 0.93); the latent-kNN modes close the gap without an
   inference label: `knn_type_quantile_map` 0.962 / R² 0.91 / energy 2.6 on v3 (reported nuclei-only decode),
   `knn_quantile_map` 0.921 with no labels anywhere; on v2 0.89 / 0.85 (type prediction 77 % across chemistry).
   **Update 2026-10-27:** with the reference classifier (`--type_predictor pca_lr`) v2 is 0.90 / 0.83 (type 84 %);
   see §6.4a. Remaining error is Pvalb -> Sst and L6 IT <-> L5 IT confusion on v2.
5. Applications now that the model is chemistry-robust: Ding 2020 matched sc/sn holdout; human SCORCH snRNA-seq
   (655k nuclei, PFC/INS/VST, HIV vs control; cortical labels match the training types; see
   `notes/experiments/yscorch_scoping/`) as the nuclei-only application.

## Figure (2026-11-06)

`notes/experiments/libsize_covariate/figures/fig_v2_cross_chemistry_ladder.png`: the v2 decode-library ladder of §6.3 / §6.4a (gene corr and R² per decode mode, label vs label-free, v3 label-free benchmark as the dashed line). Values are the table numbers of §6.4a.

### 6.8 Per-gene variance distribution and cost function, with vs without the covariate (figures 2026-11-06, written 2026-11-13)

Asked by Prashant (meeting 2026-11-06, `notes/meeting_2026-11-06_prashant.md`): the per-gene distribution instead of the
mean ratio, for one large type, with and without the covariate; and whether the cost function improves.

Figures `notes/experiments/libsize_covariate/armA/figures/fig_variance_distribution_L5IT.png` (density of log10
within-type per-gene variance, L5 IT, test partition, plus hexbin of imputed vs real per gene) and
`fig_example_gene_densities_L5IT.png` (per-cell densities of Snap25, Rpl3, Calm1, Malat1, Pcp4, Fezf2, Vamp2, Xist);
table `armA/tables/variance_distribution_L5IT_test.csv`; script `analysis/benchmark/variance_distribution_ct.py`
(re-created 2026-11-13, the original was a scratch script). Medians of the within-type variance: real cells 0.391, raw
nuclei 0.209, old model 0.072 (ratio 0.18, Spearman vs real 0.92), covariate at a fixed per-type library 0.0022 (0.007,
0.66), covariate with per-cell quantile-mapped library 0.081 (0.21, 0.92), label-free predicted type 0.081 (0.21, 0.92).
All distributions are unimodal; no zero-variance peak. Reading: the covariate makes the library-driven variance a decode
choice -- off at a fixed library (0.002 = latent biology only), on at the per-cell library (back to the old model's
level with the same gene ranking) -- which confirms that the old model's within-type variance was library pass-through.

Cost function, run `run_2026-10-09-19-13-44_611868` (no covariate, seed 1) vs `run_2026-10-16_libsize_armA_jsd0.1_seed1`
(covariate), both 600 epochs, normalised losses with the same epoch-0 normaliser (0.5032 vs 0.5033, so the values are
directly comparable; raw MSE = value x 0.503):

| term (epoch 599) | no covariate | covariate (arm A) |
|---|---|---|
| train total | 0.993 | 0.958 |
| nuclei reconstruction | 0.2214 | 0.2208 |
| cell reconstruction | 0.2548 | 0.2527 |
| alignment JSD | 0.0243 | 0.0074 |
| nuclei / cell JSD (weight 0.1) | 0.037 / 0.040 | 0.023 / 0.024 |
| valid total | 1.028 | 0.970 |
| valid cell reconstruction | 0.268 | 0.256 |

Reconstruction is unchanged (< 1 %); the covariate lowers the alignment and prior terms, i.e. the latent no longer has
to encode depth. The benchmark gain is in the cross-modality decode (split-aware test partition, labelled decode): gene
corr 0.900 -> 0.970, R² 0.688 -> 0.927, gene MSE 0.030 -> 0.006, delta r 0.93 -> 0.98, classification 0.963 -> 0.967;
label-free 0.962 / 0.909. Answer to Prashant: the cost function does not get worse, and the imputation accuracy
improves by 7 pts gene corr / 24 pts R², which is the real argument for the covariate.

**Addendum 2026-11-13 (Prashant's "numerator and denominator" and "raw count generator" points; same script, two new
series in the table / figure, previous figure kept as `fig_variance_distribution_L5IT_2026-11-06.png`):**

| source (L5 IT, test) | median var | ratio to real | Spearman vs real | ratio / Spearman vs real with library size regressed out |
|---|---|---|---|---|
| real cells | 0.391 | 1 | 1 | 1.20 / 0.75 |
| real cells, log library size regressed out per gene | 0.276 | 0.84 | 0.75 | 1 / 1 |
| imputed, covariate, fixed per-type library | 0.0022 | 0.007 | 0.66 | 0.009 / 0.50 |
| imputed, covariate, quantile-mapped per-cell library | 0.081 | 0.21 | 0.92 | 0.25 / 0.54 |
| imputed, covariate, quantile map **+ Poisson sampling of the decoded counts** | **0.295** | **0.75** | **0.96** | 0.90 / 0.63 |

(1) Correcting the denominator does not rescue the fixed-library decode: taking library size out of the real cells
removes only 16 % of their median variance (0.39 -> 0.28; the 2026-09-18 decomposition had 25 % library, 52 % Poisson,
23 % biology), and the fixed decode stays 100x below the depth-free real variance with rank agreement 0.50. The model
outputs expected values, so it never generates the sampling noise that is half of the real within-type variance. (2)
Making the output a count generator -- decode at the per-cell library size, then draw Poisson counts from the expected
counts -- reproduces the real distribution to 75 % of the median with the best rank agreement of any decode (0.96),
and a width close to the real one (IQR of log10 var 0.41 vs 0.30). The 25 % that is still missing is the compressed
biological part (the fixed-library decode, 0.002, is what the latent contributes). Answer to Prashant: the variance
reduction is technical (depth and sampling noise), both of which can be put back explicitly; the biological
within-type variability is compressed, and that is the honest limitation to state. This Poisson-sampled decode is the
"raw cell count generator" he asked for and should become a decode mode of `predict_cvae` (not done).

Per-gene view (2026-11-13): `armA/figures/fig_gene_densities_L5IT_10genes.png` (script `analysis/benchmark/gene_density_ct.py`;
Pcp4, Snap25, Rplp1, Rorb, Fezf2, Calm2, Malat1, Xist, Gm49164, Gfra1; six series = real cells, nuclei, old model,
covariate fixed library, covariate per-cell library, + Poisson). Reading: for abundant cytoplasm-enriched genes
(Snap25, Rplp1, Calm2) the per-cell-library decode already overlaps the real density (var 0.27-0.32 vs 0.32-0.40) and
the fixed decode is a spike (0.01); Poisson sampling is what restores the discrete zero / one / two-count structure of
sparse genes (Fezf2 det 37 % vs real 31 %, Gfra1 12 % vs 10 %) and adds nothing at high counts (Malat1 0.49 either way);
Pcp4's broad distribution survives even the fixed decode (0.88) so it is latent structure, not depth; what stays
compressed is the long tail / zero peak of Rorb (0.9-1.0 vs 1.50) and Gm49164 (0.30 vs 0.76), and Xist remains the
sex-average single mode.

Other-type markers and other categories in L5 IT (2026-11-13, `gene_densities_L5IT/ex_*.png`, 22 extra genes via
`gene_density_ct.py --extra`; the per-type folders `gene_densities_<type>/` also carry nine rule-chosen type-specific
genes `ct_*.png`, overview `fig_gene_densities_<type>_overview.png`). Question (Suzy): does the model, which never sees
the type label, predict different types alike on the same genes? Answer: no at the type level. For 12 markers of other
types (Cux2, Syt6, Foxp2, Tshz2, Bcl11b, Sst, Vip, Lamp5, Gad1, Mbp, Plp1, Cx3cr1) the per-cell-library decode puts the
L5 IT mean within 0.1 log units of the real L5 IT mean (e.g. Sst 0.14 vs 0.15 with 4.38 in real Sst cells; Syt6 0.03 vs
0.04 vs 2.29 in L6 CT; Plp1 0.33 vs 0.35 vs 6.49 in Oligo) -- the ambient / low level of the real cells, nowhere near the
owning type; Bcl11b is the one slight over-call (0.32 vs 0.17). Immediate early genes (Arc 1.55 vs 1.54, Egr1 3.49 vs
3.41, Npas4 0.40 vs 0.42) are restored from near-zero nuclei (Arc 0.10): the model reproduces the dissociation-induced
IEG shift of whole cells faithfully, which is a caveat to state (it is a protocol signature, not biology of the cell).
What is compressed is within-type: the mean decode gives small non-zero values in 97-100 % of cells where real cells
detect the marker in 4-25 % (Poisson sampling restores detection: Syt6 3 vs 4 %, Foxp2 18 vs 24 %, Sst 14 vs 19 %, Mbp
76 vs 76 %), and the upper tail is pulled to the mean (95th percentile: Cux2 2.44 vs 3.09, Npas4 0.74 vs 1.79, Foxp2
0.38 vs 1.10). So the refined version of the hypothesis holds at the sub-type level: within L5 IT the Cux2-high
(L2/3-like) end of the IT continuum and the strongly IEG-activated cells are flattened toward the type mean. Type
identity is resolved; within-type gradients and states are under-resolved -- the posterior-collapse picture (2-6 active
latent dims) seen from the gene side.

### 6.9 Per-gene distribution closure (2026-11-13): a replacement for the variance ratio

Motivation (Suzy, 2026-11-13): the variance ratio under-sells decodes that already sit on the real distribution (Snap25,
Calm2, Rplp1 look right but score 0.5-0.7), and the benchmark's headline is a ratio of mean variances that a few
high-variance genes dominate. Script `analysis/benchmark/distribution_closure_ct.py` (L5 IT, 50 min single core);
outputs `armA/tables/distribution_closure_L5IT_{summary,per_gene}.csv*`, figure `armA/figures/fig_distribution_closure_L5IT.png`.
Definition per gene: W1 = 1-D Wasserstein distance on log1p between per-cell distributions; floor = W1 between two
random halves of the real cells; closure = 1 - (W1(imputed, real) - floor) / (W1(nuclei, real) - floor), clipped to
[-1, 1]; genes with W1(nuclei, real) - floor < 0.05 (no shift to recover) are excluded (9,005 of 10,134 kept).

| series | stratum (real-cell detection) | n | median closure | frac > 0.8 | frac worse than nuclei | median KS | median var ratio |
|---|---|---|---|---|---|---|---|
| old model | > 70 % | 4,344 | 0.72 | 0.35 | 0.04 | 0.20 | 0.34 |
| old model | 30-70 % | 3,552 | -0.15 | 0.00 | 0.59 | 0.49 | 0.13 |
| old model | 10-30 % | 1,047 | -1.00 | 0.00 | 0.98 | 0.77 | 0.06 |
| old model | all | 9,005 | 0.32 | 0.17 | 0.37 | 0.32 | 0.21 |
| covariate, per-cell library | > 70 % | 4,344 | 0.75 | 0.41 | 0.04 | 0.19 | 0.40 |
| covariate, per-cell library | all | 9,005 | 0.36 | 0.20 | 0.36 | 0.32 | 0.23 |
| covariate + Poisson | > 70 % | 4,344 | **0.92** | **0.85** | 0.01 | 0.18 | 0.84 |
| covariate + Poisson | 30-70 % | 3,552 | 0.73 | 0.32 | 0.04 | 0.31 | 0.72 |
| covariate + Poisson | 10-30 % | 1,047 | 0.59 | 0.14 | 0.04 | 0.20 | 0.63 |
| covariate + Poisson | < 10 % | 62 | 0.82 | 0.55 | 0.00 | 0.07 | 0.61 |
| covariate + Poisson | all | 9,005 | **0.82** | **0.56** | 0.03 | 0.25 | 0.77 |

Reading. (1) For well-detected genes the mean decodes already close 72-75 % of the nuclei-to-cell distance although
their variance ratio is 0.34-0.40: location and shape are right, only the sampling width is missing -- this is the
discrepancy Suzy saw in the gene panels, now quantified. (2) For genes detected in 10-70 % of cells the mean decodes are
*worse than the raw nuclei* by distribution distance (closure < 0 for 59-98 % of them): a mean output puts a small
positive value in every cell where both nuclei and cells are mostly zero, so the imputed histogram is further from the
real one than the nucleus histogram was. The variance ratio could not show this. (3) Poisson sampling of the per-cell-
library decode fixes both: median closure 0.82 over all genes, 0.92 with 85 % of genes above 0.8 for well-detected
genes, and only 3 % of genes worse than nuclei. The residual (W1 still ~5x the split-half floor, KS 0.18 for abundant
genes) is the compressed within-type spread of §6.8. (4) Variance ratio and closure agree once sampling is in (only 7 of
9,005 genes have closure > 0.9 with ratio < 0.6), so the two metrics disagree only for mean-valued outputs.
Proposal: report closure (median, fraction > 0.8, fraction worse than nuclei) by detection stratum on the Poisson-
sampled decode as the within-type headline; keep the variance ratio by expression quartile (§6.8) as the secondary
table; add both to `evaluate_all.py` and the sampled decode to `predict_cvae.py` (not done).
