# Ablations on the final (library-size covariate) model

Started 2026-11-04. Item 3 of `notes/prashant_update_2026-10-30.md` §5 and item 5 (A4) of
`notes/paper_storyline_2026-10-23.md`; they need no decision from the meeting, so they run in the background.
Latent width was already done 2026-10-23 (`notes/libsize_covariate_review_2026-10-09.md` §6.7). Seeds exist (3).

## Base and arms

Base = arm A of the covariate retrain (`run/run_2026-10-16_libsize_armA_jsd0.1_seed1`, job 13446475_1): covariate
on encoder + decoder, per-modality JSD 0.1/0.1, alignment 1.0, seed 1, 600 epochs, log1p, Config B genes.
Each arm changes exactly one thing (`jobs/ablation/run_ablations_final_model.sh`, array **14861675**, tasks 1-6):

| task | arm | change | question |
|---|---|---|---|
| 1 | `nojsd` | per-modality JSD weights 0/0 | is the prior-matching regulariser needed at all? (arm B at 0.01 changed nothing) |
| 2 | `noalign` | alignment weight 0 | is cross-modality latent alignment needed for translation, or does the conditioning carry it? |
| 3 | `deccond` | `--cond_encoder False` | does the encoder need the modality signal (decoder-only conditioning = the earlier design)? |
| 4 | `n1_L5IT` | `--exclude_cell_types "L5 IT"` | held-out type with neighbours (L2/3 IT, L6 IT): interpolation case; 2,320 test nuclei |
| 5 | `n1_Pvalb` | `--exclude_cell_types Pvalb` | held-out type whose neighbour (Sst) is the known confusion pair; 520 test nuclei |
| 6 | `donor5` | `--holdout_donors nuclei:5` | nuclei from an unseen animal (7,594 nuclei leave train/valid); cells have 12 other donors |

Run folders: `run/run_2026-11-04_ablation_<arm>_seed1`. Resumable (`--run_name` + `run_state.json`).

## Implementation (2026-11-04)

- `train.py --exclude_cell_types <comma list>` and `--holdout_donors <modality>:<donor>[,...]`.
- `imputeVAE/data.py apply_train_exclusions`, called in `create_dataloaders_by_cell_type` on the TRAIN and VALID
  partitions only, **after** `balanced_split_by_cell_type`. The stratified split draws per class in sequence from
  `random_state=42`, so filtering before it would move other types' rows between partitions and leak the manifest's
  test set into training. Filtering after it leaves the test partition identical to `data/split_manifest.tsv` and the
  benchmark cache, and the held-out observations are scored there. Donor = prefix of `meta_batch_id`
  (`<donor>-<modality><version><sample>`), the field `data_split.py` uses for the manifest's donor column.
  Unit-tested on synthetic obs (type removal, donor removal scoped to the right modality, both, bad spec raises).
- `analysis/benchmark/eval_by_donor.py`: CT-level metrics per nuclei donor on the types with >= 20 eval nuclei in every
  donor (11 types) plus a pooled row. Validated: its pooled 14-type numbers equal `ct_level_robust` in
  `run/benchmark/test/results/benchmark_full.json` (arm A 0.963 / R² 0.906; cvae_seed1 0.855 / 0.480).

## Evaluation chain (submitted with dependencies)

1. `jobs/ablation/predict_ablations.sh` array **14861676** (`aftercorr` on 14861675): per arm, four decodes of the test
   nuclei -> `run/benchmark/test/abl_<arm>_{ctmed,qmap,ktqmap,kqmap}` = ct_median_cell (label), quantile_map (label),
   knn_type_quantile_map (reported label-free), knn_quantile_map (no labels anywhere).
2. `jobs/ablation/evaluate_ablations.sh` **14861677** (`afterok` on 14861676): `evaluate_all` over the 24 ablation dirs +
   controls (`cvae_libsize_armA`, `_qmap`, `_knntypeq`, `_knnq`, `cvae_seed1`) -> `run/benchmark/test/results_ablations/`;
   then `eval_by_donor` for the donor arm + controls -> `results_ablations/by_donor/`.

## How to read each arm

- `nojsd`, `noalign`, `deccond`: the headline rows are `ctmed` (label, compare 0.970 / R² 0.927) and `ktqmap`
  (label-free, compare 0.962 / 0.91 / energy 2.6). For `noalign` also look at `classification` accuracy and `latent`
  energy: without alignment the kNN-type decode may break even if the labelled decode holds.
- `n1_L5IT`, `n1_Pvalb`: per-type columns of the held-out type in `per_ct_shift.csv`, `per_ct_delta_pearson.csv`,
  `per_ct_energy.csv`, `per_ct_clf_recall.csv` (recall of the held-out type under the cell-trained classifier) versus the
  same columns for arm A. Label-free decodes (`ktqmap`, `kqmap`) use the kNN reference WITHOUT the held-out type (see 2026-11-05
  fix below): the type vote can only name a neighbouring training type, so `ktqmap` shows which type the model files the
  unseen nuclei under and `kqmap` is the no-label-anywhere number; both are legitimate numbers of record for the unseen
  type. `ctmed`/`qmap` use the eval label but the per-type library stats fall back to the median of type medians
  (predict_cvae prints a WARNING naming the type).
- `donor5`: `results_ablations/by_donor/by_donor.md`, donor 5 column vs donors 1-4 and vs arm A's donor 5 column.
  Baseline spread across donors for arm A (ctmed) is 0.917-0.922 on the 11 common types, so a drop > 0.01 is a donor
  effect. Note from the validation run: arm A's label-free `knntypeq` decode already scores donor 5 lower (R² 0.45 vs
  0.72-0.80), so type prediction is harder for donor 5 even when it is in training.

## Side observation from validating `eval_by_donor.py` (2026-11-04)

On the 11 neuronal types alone the no-covariate model (`cvae_seed1`) reaches gene corr 0.64 / R² < 0 while arm A keeps
0.92 / 0.79; the 0.90 vs 0.97 gap in the main table is dominated by the glial/rare types that carry most between-type
variance. The covariate matters most for within-class (neuron vs neuron) structure. Worth a sentence in the Fig 2 text.

## Status

- 2026-11-04: training array 14861675 submitted on `pi_gerstein_gpu,scavenge_gpu` (`--gpus=2`, 400G, GPU guard);
  r813u29n03 had 3 healthy GPUs and 887 GB free at submission, so two arms run at a time (~12 h each).
- 2026-11-04 10:55: all six arms running (tasks 1, 6 on r813u29n03 A100s; 2 on an A100-40GB, 3, 5 on RTX A5000s, 4 on
  r107u27n01 via scavenge_gpu). Task 4 was preempted at 10:38 after its epoch-0 checkpoint, requeued itself and resumed at
  epoch 1 (so it skipped the epoch-0 inline eval and is ~30 epochs ahead). Timing from arm A's `loss.txt`: training
  ~18 s/epoch (11 s for the N-1 arms, less data), inline eval at every 50th epoch ~40 min CPU-bound -> ~12 h per arm.
  Exclusion counts on real data (task 4, N-1 L5 IT): nuclei train 6,959 / 24,099 removed, valid 2,320 / 8,033; cells
  train 17,832 / 42,669, valid 5,944 / 14,224 (valid count = manifest test count, as expected from the 20/20 split).
  Note: with `--debug True` each stdout log grows to ~3 GB (arm A's was 3.1 GB); delete `jobs/logs/slurm-14861675_*` after
  the runs are recorded.
- 2026-11-05: all six arms reached epoch 599 (`model.pt`, `run_state.json`); wall 10-15 h for the full arms. Three arms
  were preempted and resumed (task 3 deccond 3x, task 4 n1_L5IT 4x, task 5 n1_Pvalb 3x; the last attempts of 4 and 5 found
  the finished checkpoint and exited in minutes). **Chain break:** deccond's final attempt was preempted at the exact
  moment of exit (log ends with "Done" at 00:10:53, sacct says PREEMPTED), so its `aftercorr` predict task never ran and
  evaluate 14861677 was cancelled (dependency never satisfied). Predict for deccond rerun as **15559184_3** (done).
  **Leak in the label-free decodes of the exclusion arms:** `predict_cvae` built the kNN reference (type vote, neighbour
  libraries) from the benchmark cache's full train partition, which still contains the held-out type / donor (the
  exclusion is applied inside train.py after the split). The kNN-type vote therefore named L5 IT for the held-out L5 IT
  nuclei (type accuracy 0.948; the WARNING about missing library stats fired in `ktqmap` mode, which is how it showed).
  Fix: `predict_cvae.apply_run_exclusions` drops the run's `exclude_cell_types` / `holdout_donors` from `nuc_train` /
  `cell_train` before any kNN step (counts verified against the training log: 6,959 L5 IT nuclei, 17,832 L5 IT cells;
  nuclei donor 5 = 4,579 train rows); `arguments.json` of each decode now records `train_exclusions` and
  `knn_reference`. The `ctmed`/`qmap` decodes were unaffected (they use `libsize_stats.json`, written from the filtered
  training data). Decodes of arms 4-6 rerun as **15559684_[4-6]**, evaluate resubmitted as **15559685**
  (`afterok:15559184:15559684`). The four debug stdout logs (`slurm-14861675_{1,2,3,6}_cvae_ablation.out`, 9.4 GB,
  group quota at 4089/4096 GiB) still need deleting by hand; everything they contain is in the run folders.

## Results (evaluate 15559685, 2026-11-06; `run/benchmark/test/results_ablations/summary.md`)

Controls: arm A `ctmed` 0.970 / R² 0.927 / clf 0.967; arm A `ktqmap` (label-free) 0.962 / 0.909 / energy 2.59.
Seed spread of arm A is 0.002 gene corr (3 seeds), so differences below ~0.005 are noise.

| arm | ctmed gene corr / R² | ktqmap gene corr / R² / energy | clf acc | delta r | reading |
|---|---|---|---|---|---|
| arm A (base) | 0.970 / 0.927 | 0.962 / 0.909 / 2.59 | 0.967 | 0.982 | |
| nojsd | 0.964 / 0.913 | 0.962 / 0.914 / 2.77 | 0.970 | 0.978 | per-modality JSD is not needed; drop it or keep 0.1, nothing changes |
| noalign | 0.947 / 0.866 | 0.944 / 0.866 / 2.78 | 0.917 | 0.957 | alignment is needed: -0.02 gene corr, -0.06 R², -5 pt classification, two-sample AUROC 0.99 |
| deccond | 0.963 / 0.912 | 0.965 / 0.917 / 2.59 | 0.956 | 0.972 | encoder modality signal is worth ~0.005 and 1 pt classification; small but above seed spread |
| n1_L5IT | 0.957 / 0.891 | 0.959 / 0.899 / 3.22 | 0.699 | 0.971 | global table barely moves (other types untouched); see held-out rows below |
| n1_Pvalb | 0.961 / 0.901 | 0.963 / 0.914 / 2.99 | 0.905 | 0.969 | idem |
| donor5 | 0.969 / 0.921 | 0.970 / 0.931 / 2.72 | 0.956 | 0.980 | an unseen animal costs nothing at the labelled decode; see per-donor table |

**Held-out type (N-1).** Per-type columns, held-out type vs the same type in arm A:

| | delta r | energy (ktqmap) | W1 hvg (ktqmap) | clf recall of the type | kNN-type vote files the test nuclei as |
|---|---|---|---|---|---|
| L5 IT, arm A | 0.997 | 1.10 | 0.21 | 0.99 | L5 IT |
| L5 IT, held out (2,320 nuclei) | 0.93 | 12.5 | 0.47 | 0.01 | L2/3 IT 45 %, L6 IT 44 %, L5 ET 10 % |
| Pvalb, arm A | 0.983 | 4.2 | 0.33 | 0.96 | Pvalb |
| Pvalb, held out (520 nuclei) | 0.85 (ctmed) / 0.87 (ktqmap) | 29 | 0.54 | 0.00 | Sst 96 % |

So the cell-vs-nucleus *delta* of an unseen type is interpolated from its neighbours with r 0.93 (L5 IT, two IT
neighbours) and 0.85-0.87 (Pvalb, one neighbour that is the known confusion pair), but the *identity* collapses onto
the nearest seen type: the cell-trained classifier recognises 1 % / 0 % of the imputed nuclei as their true type and
the distributional metrics (energy, W1) are 3-10x worse. Neighbouring types are unaffected (L2/3 IT, L6 IT, Sst delta r
>= 0.994). Reported label-free number for the N-1 arms is `ktqmap` (the vote names the neighbour type, which is the
honest behaviour after the 2026-11-05 reference fix); `kqmap` is already unreliable for L2/3 IT in arm A (energy 23)
and is not a number of record here. Paper sentence: the model transfers the compartment shift to an unseen type
through its neighbours but cannot invent the type; a label or a seen neighbour is required for type identity.

**Held-out nuclei donor (donor 5).** `results_ablations/by_donor/by_donor.md`, 11 neuronal types, donor 5 column vs
donors 1-4 (arm A donor spread 0.917-0.922 at ctmed):

| decode | arm A donor 5 (seen) | donor5 arm, donor 5 (unseen) | arm A donors 1-4 | donor5 arm, donors 1-4 |
|---|---|---|---|---|
| ctmed (label) gene corr / R² | 0.920 / 0.795 | 0.927 / 0.811 | 0.917-0.922 / 0.79-0.80 | 0.923-0.925 / 0.80-0.81 |
| ktqmap (label-free) | 0.880 / 0.446 | 0.868 / 0.394 | 0.916-0.931 / 0.72-0.80 | 0.916-0.934 / 0.74-0.81 |
| qmap (label, per-cell scale) | 0.890 / 0.465 | 0.874 / 0.383 | 0.919-0.929 / 0.73-0.81 | 0.918-0.925 / 0.75-0.80 |

An unseen animal costs nothing at the labelled type-median decode and 0.01-0.015 gene corr (0.05-0.08 R²) in the
per-cell-scale decodes of that donor. Donor 5 is the hard donor for every model (its label-free R² is 0.4-0.45 even
when it is in training), so the hold-out effect is small relative to the donor's own difficulty. Generalisation to a new
animal of the same protocol holds; this is the within-chemistry complement of the Bakken / v2 tests.

**Decisions for Fig 2 / methods.** (1) Keep the architecture as frozen (covariate on encoder + decoder, alignment 1.0);
the JSD term can be reported as optional. (2) Report the alignment ablation as the one loss term that matters. (3) The
N-1 and donor rows go to the generalisation figure (Fig 5) next to Bakken and v2: unseen donor ~0 cost, unseen type =
delta interpolated at r 0.85-0.93 with identity collapse. Storyline note updated (A4).

## Library (batch) covariate (added 2026-11-06)

Asked for after the sex-covariate discussion (cells = 6 male + 6 female libraries, nuclei all female; library and sex
are confounded in cells, so a library condition subsumes sex). One run on the arm A config + `--library_cond True`
(`jobs/ablation/run_library_cond.sh`, run `run/run_2026-11-06_libcond_seed1`).

Design (`imputeVAE/models_cond.py`, `imputeVAE/cond_training.py`, `train.py`, `analysis/benchmark/predict_cvae.py`):
- `nn.Embedding(library_capacity + 1, library_dim)` of the sequencing library (`meta_batch_id`, e.g. `4-c3A`; one
  vocabulary over both modalities, 17 libraries = 12 cell + 5 nuclei, saved to `<run>/library_vocab.json`, ids 1..N
  sorted by modality then name so a resumed run rebuilds the same mapping). `library_dim 4`, `library_capacity 32`
  (fixes the state_dict shape before the data are loaded).
- Appended to the encoder input and the decoder input next to the modality scalar and the library-size scalar
  (scVI-style: the encoder sees the batch so the latent can be batch-free; the decoder gets the target batch).
- Row 0 = UNKNOWN library. During training each sample's id is replaced by 0 with p = `library_dropout 0.1`, so the
  model learns a marginal "average library" embedding; it is the decode target of record (`--target_library unknown`)
  and what unseen libraries (external data) get on the encoder side.
- Loader batches already carry `batch_id`, but `myDataset` factorises per cell-type dataset, so
  `build_library_vocab` / `build_library_remaps` map local codes to the global ids per loader.
- Inference (`predict_cvae`): the eval nucleus's own library goes to the encoder (via (modality, donor) of the
  benchmark cache obs); `--target_library unknown | mean_cell | <name>` picks the output library. kNN modes encode the
  training reference with its own library ids.

Jobs: smoke test **15562731** (CPU, 10 % data, 2 epochs, all decode modes) -> training **15562732** (afterok) ->
predict **15562733** (afterany; `run/benchmark/test/libcond_{ctmed,qmap,ktqmap,kqmap}` at UNKNOWN,
`libcond_ctmed_meancell`, `libcond_ctmed_lib4F` (female 4-c3A), `libcond_ctmed_lib1M` (male 1-c3A)) ->
evaluate **15563057** (`results_libcond/`; also latent_library_mixing + eval_by_donor).
Smoke test passed 2026-11-06 01:02 (45 min, peak RSS 340 GB on 10 % data: vocabulary 17 = 12 cell + 5 nuclei, 2 epochs,
inline eval, all six decode variants through predict_cvae); training released to the GPU queue.
Arm A baseline of the latent-mixing check (`results_libcond/latent_mixing_armA_only/`): within-type normalised excess
library purity 0.027 (cells) / 0.002 (nuclei) -- the latent is already nearly library-free, so expect the covariate to
show up in the decodes (female/male target library, Xist) rather than in the latent; nuclei-vs-cell kNN purity 0.64.

How to read: (1) `libcond_ctmed` / `libcond_ktqmap` vs arm A 0.970 / 0.962: does the batch condition change the
benchmark at all? (2) latent library-mixing (to add: kNN library purity / silhouette by library in the latent, arm A vs
libcond) = is the latent library-free? (3) Xist in `libcond_ctmed_lib4F` vs `libcond_ctmed_lib1M`: a sex-specific
decode falls out of the library condition if the embedding learned sex. (4) `eval_by_donor` on the nuclei side.
Decision rule: it does not replace the frozen model unless (1) moves by more than the seed spread (0.002) or (2)/(3)
give a figure panel; otherwise it is a supplementary ablation.

### Results (evaluate 15563057, read 2026-11-13; `run/benchmark/test/results_libcond/summary.md`)

Training 15562732 ran the full 600 epochs (A100-40GB, three requeues; the last attempt resumed at epoch 599 and
exited in 10 min). Training is not the difference: final validation loss 0.971 vs arm A 0.970, train loss 0.948 vs
0.958, inline shift at epoch 599 cell +0.184 / nuclei -0.190 vs +0.189 / -0.188.

| decode (target library) | gene corr / R² | delta r | clf acc | energy | arm A same decode |
|---|---|---|---|---|---|
| `libcond_ctmed` (UNKNOWN) | **0.956 / 0.893** | 0.965 | 0.927 | 19.8 | 0.970 / 0.927, 0.982, 0.967 |
| `libcond_ctmed_meancell` | 0.952 / 0.873 | 0.964 | 0.917 | 20.7 | |
| `libcond_ctmed_lib4F` (female 4-c3A) | 0.951 / 0.852 | 0.956 | 0.916 | 22.0 | |
| `libcond_ctmed_lib1M` (male 1-c3A) | 0.940 / 0.834 | 0.954 | 0.885 | 21.3 | |
| `libcond_qmap` | 0.958 / 0.898 | 0.965 | 0.925 | 3.30 | 0.970 / 0.930, 0.982, 0.966, 2.62 |
| `libcond_ktqmap` (label-free) | **0.959 / 0.899** | 0.965 | 0.924 | 3.34 | 0.962 / 0.909, 0.976, 0.966, 2.59 |
| `libcond_kqmap` | 0.883 / 0.566 | 0.914 | 0.928 | 23.0 | 0.921 / 0.809 |

(1) **Benchmark:** -0.014 gene corr / -0.034 R² at the labelled decode and -0.003 / -0.010 label-free, -4 pt
classification; 7x the seed spread, in the wrong direction. The loss is concentrated in the types that are rare in
*cells* (profile corr to real cells, arm A -> libcond): Endo 0.949 -> 0.827, L5 ET 0.981 -> 0.938, Oligo 0.983 -> 0.924,
Macrophage 0.964 -> 0.919; the abundant neuronal types are unchanged (L2/3 IT, L5 IT, L6 CT, L6 IT >= 0.997 in both).
**L5 ET loses its identity:** classifier recall 0.988 -> 0.000 under every target library, and a nearest-centroid
check (correlation to the 18 real-cell type means) files 98 % of the imputed L5 ET nuclei as L6 CT (arm A: 99 % L5 ET);
Endo goes 60 % Macrophage, Oligo 21 % OPC. These types have 74-113 training cells spread over 12 libraries (~8 per
library), so the per-library decoder and the UNKNOWN token (trained on 10 % of samples) never see enough of them
(nuclei side: 746 L5 ET). The type x library confound shows up where expected: Pvalb cells exist only in libraries
9-12 (220 cells, none in 1-8), so decoding as library 1 drops Pvalb recall to 0.29 and Sncg to 0.00; the UNKNOWN
decode keeps them (0.95 / 0.71).
(2) **Latent:** unchanged within modality (within-type normalised excess library purity, cells 0.030 vs 0.027, nuclei
0.002 vs 0.002 -- arm A's latent was already library-free); the nuclei-vs-cell kNN purity got *worse* (0.78 vs 0.64,
silhouette 0.16 vs 0.03): giving the encoder the library weakens the cross-modality alignment.
(3) **Sex falls out of the library embedding (positive result):** mean log1p Xist per type, real female cells 2.4-3.6,
real male cells 0.00; `lib4F` decode 2.8-3.8, `lib1M` decode -0.1 to +0.2, UNKNOWN 1.3-2.6 (= the mixed-sex mean, same
as arm A 1.0-3.0). Over all 10,134 genes the decode difference (lib4F - lib1M) correlates 0.655 with the real
female-minus-male cell difference; the top genes agree (Xist 3.57 vs 3.47, Actb, Gm49164, Arc, Eef1a1 (negative) on
both lists). So a named-library decode places the output in a sex / batch, which the frozen model cannot do.
(4) **By donor** (11 neuronal types): `libcond_ctmed` 0.886 flat across nuclei donors vs arm A 0.917-0.922; donor 5
label-free 0.844 vs 0.880 -- the same -0.03 everywhere, no donor-specific effect.

**Decision (2026-11-13):** the library covariate does not replace the frozen model (rule (1) fails by 7x the seed
spread; rare cell types lose identity; alignment weakens). It is a supplementary ablation with one usable panel: the
Xist / sex-placement decode (item 3) as the demonstration that batch can be made an explicit decode target. If the
batch covariate is ever wanted in the reported model, the fix to try is decoder-only conditioning (keep the encoder
blind, so the alignment stays) with a higher UNKNOWN dropout (0.5) so the marginal profile of rare types is trained;
not planned. Storyline A4 and `notes/training_runs.md` updated.
