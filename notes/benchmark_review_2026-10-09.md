# Benchmark review and redo (2026-10-09)

Goal: redo the baseline benchmark carefully. This note records (1) what was wrong with how the
comparison was done so far, (2) what each metric actually measures and which ones we keep, drop or
add, (3) the protocol of the redo, and (4) the results once the jobs finish.

Code: `analysis/benchmark/` (data layer, one script per method, `evaluate_all.py`).
Jobs: `jobs/benchmark/submit_all.sh`. Outputs: `run/benchmark/<partition>/<method>/`,
tables in `run/benchmark/<partition>/results/`.

---

## 1. How the benchmark was done until now, and what was wrong

The numbers in `notes/experiments/baseline_comparison.md` (cVAE gene corr 0.927, R² 0.798, delta corr
0.950, variance ratio 0.158; scVI 0.386 / -0.67; linreg 0.822; scButterfly 0.786 / -0.87) came from
`analysis/evaluate_baseline_incremental.py` run on **all** 40,166 n3A nuclei with **all** 71,117 c3A
cells as reference. Problems, in order of severity:

1. **Baselines were fitted on all data; the cVAE was not.** CT-mean, linear regression, scVI, scANVI
   and scGen were fit on all cells and nuclei and then scored on the same nuclei. The cVAE was trained
   on its 60% train split and scored on all nuclei, i.e. 60% of its score came from its own training
   nuclei. Neither side is clean, and the two sides are not comparable. (Audit:
   `run/eval_v3_splitaware/audit_report.md`, 2026-08-06.)
2. **The August fix never ran to completion.**
   - `analysis/eval_splitaware.py` (job 8575090) computed every cVAE metric, then died on
     `assert var < 1e-8` for CT-mean (actual 2.6e-6, float32 accumulation) before saving a single JSON.
   - `analysis/benchmark/*` (jobs 8578258-62): CT-mean / linreg / scButterfly were OOM-killed at 64 GB
     because the raw h5ad files are **dense int64** (41 GB cells, 24 GB nuclei) and `read_h5ad` loads
     them whole; scVI / scANVI / scGen failed with "scvi-tools not installed" because the job scripts
     activated `impute2` instead of `scvi2`.
   - Worse, `analysis/benchmark/data_split.py` drew its **own** split (per-cell-type shuffle,
     `RandomState(42)`), which is **not** the split `balanced_split_by_cell_type` (two nested
     `sklearn.train_test_split`, `random_state=42`) used to train the cVAE. Had those jobs succeeded,
     the baselines' "train" set would have overlapped the cVAE's held-out set.
3. **Scale mismatch penalised the count-based baselines.** scVI / scANVI / scGen were decoded at a
   fixed library size of 1e4 and scButterfly was trained on `normalize_total(1e4)+log1p`, whereas the
   reference and the cVAE live on `log1p(raw counts)` with a median v3 cell library of ~15-30k UMI.
   Correlations are unaffected, but R², gene MSE and Wasserstein were. A large part of scVI's
   R² = -0.67 and scButterfly's Wasserstein 0.99 is this offset, not model quality.
4. **CT-mean was reported as "trivially 1.0" on cell-type-level metrics.** With all data it is the
   reference itself. With a split it becomes the right object: the sampling-noise *ceiling* that any
   method can reach on cell-type-level metrics (train-cell means vs held-out cell means).
5. **Some metrics were uninformative or mis-specified.**
   - "Per-cell corr 0.996": Pearson of a log1p profile with its type centroid is dominated by the mean
     expression vector; every method except linreg scores > 0.96. Dropped.
   - Wasserstein / QQ were computed on nuclei pooled across cell types, so they mostly measured
     cell-type composition differences between the nucleus and cell datasets. Now computed within
     cell type.
   - Nuclear-lncRNA sign test went through a flawed 5-gene list (see baseline_comparison.md); the
     45-gene data-driven list is used, but it was *derived from all data* (GT delta < -0.2), so it is a
     consistency check, not an independent validation.
   - Delta sign accuracy over all gene × CT pairs is inflated by near-zero deltas whose sign is noise;
     we add a |delta| > 0.1 version and an AUROC.
6. **No uncertainty.** Single seed everywhere, no CIs. Differences of 0.01 were discussed as real.
7. **Design-selection leakage (documented, not fixable without retraining).** Hyperparameters and
   design choices of the cVAE (June/July factorial) were compared with all-data metrics. Test-partition
   observations never entered a gradient step or an epoch-monitoring step, but they did inform the
   choice of architecture. The split is also per observation, so all 5 nucleus donors and 12 cell
   donors appear in every partition; a donor-held-out run requires retraining and is listed as
   follow-up.
8. **Gene list** (Config B, 10,134 genes) was selected on all data (detection rate > 5%). Documented;
   not changed because it would invalidate the checkpoint. Low impact (threshold is data-independent).

## 2. What the task is, and what a metric can and cannot measure

Input: a nucleus profile with cell-type label known. Output: a cell-like profile. There is **no
paired ground truth per nucleus**; the truth exists only as a *distribution* of real cells per cell
type. Any metric is therefore one of:

| Level | Question | Reference | Who can score well |
|---|---|---|---|
| A. cell-type mean | Is the mean imputed profile per type right? | held-out cell means | anything with the labels (CT-mean is the ceiling) |
| B. within-type distribution | Does the cloud of imputed profiles look like the cloud of real cells of that type? | held-out cells of that type | only per-cell models; CT-mean fails by construction |
| C. per-cell information | Does the output still carry the input nucleus's individual signal? | the input nucleus itself | identity (raw nuclei) is perfect; CT-mean is zero |
| D. downstream utility | Does a downstream analysis on imputed data recover what it recovers on real cells? | held-out cells | mixed |
| E. biology | Does the learned delta have the expected compartment signs? | prespecified gene sets | consistency check |

The method advantage we claim lives in B and C (A is matched by CT-mean; D is largely A).
The redo therefore makes B and C first-class and reports A with its ceiling.

## 3. Metric set for the redo (`analysis/benchmark/evaluate_all.py`)

### A. Cell-type-level (reference = eval-partition cells; K = 17 common cell types)
- `gene_corr`: per-gene Pearson across the K type means, median over genes (+ `pct_above_0.8`).
  Also reported as fraction of the CT-mean ceiling and on the 14 "robust" types with ≥ 20 obs/modality.
- `r2`: per-gene R² across type means; `gene_mse`.
- `profile_corr_to_cell` and `shift` = corr(imp, cell) - corr(imp, nuc) per type across genes.
- `delta`: predicted (imp mean - eval nuc mean) vs empirical (eval cell mean - eval nuc mean) over all
  gene × type pairs: Pearson, Spearman, R², sign accuracy (all, and |emp| > 0.1), AUROC, per-type Pearson.
  Note: for raw nuclei the predicted delta is identically 0, so delta metrics are undefined (NaN).

### B. Within-type distribution (reference = eval cells of the same type; types with ≥ 20 in both)
- `variance_ratio`: mean over genes of var(imp)/var(cell), median over types (kept for continuity).
- `std_ratio`: median over genes of sd(imp)/sd(cell), less dominated by the few high-variance genes.
- `wasserstein`: per-gene W1 between the imputed and real distributions **within type**, nucleus-count
  weighted over types; median over all genes and over the 500 HVGs (variance in training cells).
- `qq_corr`: per-gene QQ correlation within type (HVGs).
- `energy_distance` and `two_sample_auroc` (random-forest classifier, 5-fold CV) in the 50-PC space of
  training cells, per type (≤ 1000 vs 1000 subsample), with a **real-vs-real floor** (two disjoint halves
  of eval cells). AUROC 0.5 = imputed cells indistinguishable from real cells of that type; 1.0 =
  trivially separable. Classifier choice matters: a collapsed point mass at the centre of the real cloud
  is not linearly separable (logistic regression gave CT-mean 0.55 in a first pass) and in 50-D the
  centroid is the nearest neighbour of almost every real cell, so a kNN classifier also returns ~0.5;
  tree splits isolate it immediately.
  These are the holistic per-cell metrics the old table lacked.

### C. Per-cell information retention
- `within_ct_spearman_vs_input`: per gene within type, Spearman between imputed and input nucleus
  values (2000 random genes). Raw nuclei = 1, CT-mean = undefined/0. Measures whether per-nucleus
  ordering survives.
- `variance_rank_spearman`: Spearman between per-gene variance vectors of imputed and real cells
  within type (the "stretched, not collapsed" check from the September variance work).

### D. Downstream utility
- `classification`: multinomial logistic regression on 50 PCs fit on **training** cells, applied to
  imputed eval nuclei (accuracy, macro-F1, per-type recall). Raw nuclei give the floor.
- `de_overlap_top100`: one-vs-rest Wilcoxon markers (vectorised rank-sum, tie-corrected) per type on
  eval cells vs on imputed; fraction of the top-100 up-regulated markers recovered. The rank-sum
  statistic rewards low within-type dispersion, so a zero-variance method (CT-mean) scores poorly
  (0.22 in a first pass) for a reason unrelated to marker identity; `de_overlap_top100_logfc` ranks by
  mean in-group minus out-group difference instead and is dispersion-insensitive. Report both.
- `coexpression`: Spearman between upper-triangular gene-gene correlation matrices (500 HVGs), pooled
  and **within the largest type** (pooled coexpression is mostly cell-type structure).

### E. Compartment biology
- Sign agreement of mean delta for ribosomal (73), cytoplasmic (18) and nuclear-lncRNA (45) genes;
  neuronal-marker delta in neurons vs glia (ratio).

### Uncertainty
- 200 bootstrap resamples of eval nuclei and eval cells within type for the headline numbers (gene
  corr median, R² median, delta Pearson, delta sign accuracy, variance ratio median, shift mean).
  Percentile intervals of the bootstrap distribution; because resampling adds sampling noise to both
  sides of a correlation, the interval sits slightly below the point estimate (seen on synthetic data:
  0.996 vs [0.992, 0.994]). Use the width, not the position, and compare methods on the same resamples.
- 3 seeds for scVI, scANVI, scGen (mean ± sd across seeds in `benchmark_summary_by_method.csv`);
  1 seed for scButterfly (2 h/run). cVAE: 3 runs since 2026-10-12 (paper run, unseeded, + seeds 1 and 2; see §5a).

### Dropped
- per-cell centroid correlation (uninformative), pooled Wasserstein / QQ (composition-confounded),
  DE overlap on pooled nuclei vs the old all-data references.

## 4. Protocol of the redo

| Step | Script | Partition used |
|---|---|---|
| Split | `data/split_manifest.tsv` (regenerated 2026-08-06 from `balanced_split_by_cell_type`, seed 42) | — |
| Cache | `analysis/benchmark/build_cache.py`: h5py block reads of the manifest rows and Config-B columns → float32 raw counts in `cache/benchmark/` (train/valid/test × cell/nuc) | all three, kept separate |
| CT-mean, raw nuclei | `train_ctmean.py` | fit train cells → eval nuclei |
| linreg, linreg_shrunk | `train_linreg.py` (closed-form per-gene OLS on train type means; shrunk variant adds the per-nucleus deviation at unit gain so slopes > 1 do not amplify noise) | fit train → eval |
| scVI / scANVI / scGen | `train_scvi.py` etc. in env `scvi2`, seeds 0-2, early stopping on scvi's internal 10% of the train partition; decode at the per-type median library of training cells | fit train → eval |
| scButterfly | `train_scbutterfly.py` on log1p(raw), seed 19193 | fit train → eval |
| ImputeVAE | `predict_cvae.py`, checkpoint `run/run_2026-07-18-19-40-51_570984/models/model.pt` (trained on the same train partition) | eval |
| Evaluate | `evaluate_all.py --eval_partition test` (and `valid` as a secondary check) | eval + train cells for PCA/classifier |

**Reported partition = test** (8,034 nuclei, 14,224 cells). The validation partition was used for
epoch-level loss monitoring of the cVAE (no checkpoint selection) and is reported as a secondary
table. Rare types in test: Endo 5 nuclei / 37 cells, Macrophage 11 / 24, OPC 15 / 29; these are in
the 17-type tables but excluded from the "robust" (≥ 20) tables and from the within-type metrics.

Fairness decisions, in one place:
- Every method sees exactly the same training observations and the same 10,134 genes.
- Every method's output is compared on `log1p(raw)`; count models are decoded at the per-type median
  training-cell library size (not 1e4), scButterfly is trained in log1p(raw) space.
- Cell-type labels are available to every method at inference time (CT-mean, linreg, scGen, scANVI
  and the pseudo-pairing of scButterfly use them; the cVAE and scVI do not). The cVAE is therefore
  the *least* label-dependent entrant.
- No method is tuned on the eval partition.

## 5. Results (test partition, 8,034 held-out nuclei; job 12641011; full tables in `run/benchmark/test/results/`)

Mean ± sd over 3 seeds where available. CT-mean is the ceiling for block A. Floors: energy distance real-vs-real
0.62, two-sample AUROC real-vs-real 0.51.

| | CT-mean | **ImputeVAE** | scGen | scANVI | scVI | scButterfly | Linreg shrunk | Linreg | Raw nuclei |
|---|---|---|---|---|---|---|---|---|---|
| **A. gene corr (median)** | 0.981 | 0.898 | 0.975 ± 0.001 | 0.962 ± 0.005 | 0.958 ± 0.004 | 0.966 | 0.777 | 0.759 | 0.760 |
| A. fraction of ceiling | 1.00 | 0.92 | 0.99 | 0.98 | 0.98 | 0.99 | 0.79 | 0.77 | 0.78 |
| A. R² (median) | 0.956 | 0.723 | 0.771 ± 0.012 | 0.719 ± 0.027 | 0.703 ± 0.011 | 0.827 | 0.573 | 0.523 | -0.862 |
| A. delta Pearson | 0.990 | 0.930 | 0.980 | 0.970 | 0.966 | 0.948 | 0.843 | 0.814 | n/a |
| A. delta sign acc (abs > 0.1) | 0.994 | 0.974 | 0.985 | 0.982 | 0.980 | 0.909 | 0.967 | 0.966 | 0 |
| **B. variance ratio** | 0 | **0.147** | 0.075 | 0.074 | 0.076 | 0.041 | 0.693 | 2.99 | 0.699 |
| B. std ratio (median gene) | 0 | **0.305** | 0.169 | 0.162 | 0.166 | 0.132 | 0.816 | 1.234 | 0.816 |
| B. Wasserstein within CT, HVG | 0.625 | 0.365 | 0.369 | 0.387 | 0.387 | 0.500 | **0.296** | 0.340 | 0.651 |
| B. energy distance (floor 0.62) | 25.1 | **7.5** | 15.9 ± 0.7 | 16.4 ± 0.2 | 16.6 ± 0.7 | 12.2 | 19.7 | 12.6 | 128.6 |
| B. two-sample AUROC (RF) | 1.000 | 0.999 | 0.998 | 0.998 | 0.998 | 1.000 | 0.999 | 0.980 | 1.000 |
| **C. within-CT Spearman vs input** | n/a | **0.114** | 0.053 | 0.048 | 0.051 | -0.005 | 1.0 | 1.0 | 1.0 |
| C. variance-rank Spearman | 0.649 | **0.845** | 0.677 | 0.669 | 0.672 | 0.843 | 0.735 | 0.601 | 0.723 |
| **D. classification acc** | 1.000 | 0.960 | 0.946 | 0.930 | 0.925 | 0.974 | **0.978** | 0.923 | 0.729 |
| D. DE top-100 overlap (logFC) | 0.905 | 0.821 | **0.864** | 0.809 | 0.801 | 0.803 | 0.686 | 0.648 | 0.621 |
| D. coexpression within L5 IT | n/a | **0.790** | 0.442 | 0.442 | 0.431 | 0.664 | 0.655 | 0.655 | 0.655 |
| E. neuron/glia marker ratio (CT-mean 4.69) | 4.69 | **4.05** | 3.64 | 2.97 | 2.98 | undefined (glia delta -0.01) | 1.22 | 1.22 | n/a |
| params (M) | 0 | 21.9 | 13.2 | 13.4 | 13.2 | 10.8 | 0.02 | 0.02 | 0 |
| train time | 1 s | 12.3 h | 11 min | 36 min | 10 min | 57 min | 1 s | 1 s | 0 |

Bootstrap 95% intervals for ImputeVAE: gene corr [0.849, 0.918], R² [0.62, 0.78], delta r [0.90, 0.94],
variance ratio [0.134, 0.159]. Seed sd of the scVI family is 0.001-0.005 on gene corr, so the gaps in block A
are far outside noise.

### 5a. cVAE seed replicates (job 12686636, 2026-10-12; added 2026-10-16)

Two seed replicates of the paper configuration (`--seed 1`, `--seed 2`; job 12646907; runs `_611868`, `_450293`)
were cross-decoded on the test partition and the whole table re-run. ImputeVAE now has n = 3 like the scvi family
(mean ± sd over the three runs; the paper run is unseeded):

| metric | paper run | seed 1 | seed 2 | mean ± sd | nearest competitor |
|---|---|---|---|---|---|
| gene corr (median) | 0.898 | 0.903 | 0.900 | 0.900 ± 0.002 | scVI 0.958 ± 0.004, scANVI label-free 0.865 ± 0.002 |
| R² (median) | 0.723 | 0.682 | 0.660 | 0.688 ± 0.032 | scVI 0.703 ± 0.011 |
| delta Pearson | 0.930 | 0.932 | 0.929 | 0.930 ± 0.002 | |
| variance ratio | 0.147 | 0.145 | 0.161 | 0.151 ± 0.009 | scVI label-free 0.151 ± 0.003 |
| energy distance | 7.5 | 6.2 | 6.6 | 6.8 ± 0.7 | scVI label-free 9.2 ± 0.4 |
| within-CT Spearman vs input | 0.114 | 0.115 | 0.114 | 0.114 ± 0.001 | scVI label-free 0.134 |
| classification acc | 0.960 | 0.967 | 0.964 | 0.963 ± 0.004 | linreg_shrunk 0.978 |
| coexpression within L5 IT | 0.790 | 0.795 | 0.792 | 0.792 ± 0.003 | scanvi label-free 0.717 ± 0.008 |
| neuron/glia marker ratio | 4.05 | 4.79 | 7.27 | 5.4 ± 1.7 | truth 4.69 |

Reading: every §5 conclusion survives. Gene corr, delta Pearson, within-CT Spearman and coexpression are stable to
±0.003; the paper run is not a lucky draw. R² (±0.03) and energy distance (±0.7) are the noisy ones, and the
neuron/glia marker ratio is unstable (4.1 to 7.3) and should be reported with the sd or dropped. Variance ratio
0.151 ± 0.009 vs label-free scVI 0.151 ± 0.003 confirms the tie.

Same-seed nondeterminism: seed 2 was preempted nine times and trained to completion three times; two complete
attempts (`_450293`, used above, and `_3752550`) agree on gene means (r 0.9996) and per cell (r 0.9987) but differ
by up to 5.5 log1p units in single entries, and their within-type per-gene variances correlate at 0.99 (median
type) down to 0.84 (worst type). `--seed` pins initialisation and shuffling only; per-cell metrics carry this
floor of run-to-run noise regardless of the seed.

### 5b. Library-size covariate model (job 13780004, 2026-10-23; full account in `libsize_covariate_review_2026-10-09.md` §6.4)

Same model family with the log library size as an explicit encoder+decoder covariate (`--libsize_cond True`, arm A =
paper hyperparameters, seed 1). At inference the output library is a parameter; what it is set to decides which group
of the table the model belongs to:

| decode library | label used | gene corr | R² | var ratio | W1 HVG | energy | within-CT Spearman | clf | coexp L5 IT |
|---|---|---|---|---|---|---|---|---|---|
| per-type training-cell median | yes | 0.970 | 0.927 | 0.019 | 0.530 | 19.5 | 0.019 | 0.967 | 0.420 |
| per-type quantile map (nucleus rank -> cell library) | yes | 0.970 | 0.930 | 0.275 | 0.279 | 2.6 | 0.116 | 0.966 | 0.789 |
| **type predicted from the 50 nearest training nuclei in the latent (97 % acc.), per-type quantile map** | no (training labels only) | **0.962** | **0.909** | 0.269 | 0.279 | 2.6 | 0.116 | 0.966 | 0.792 |
| K nearest training cells in the latent, quantile map (no labels anywhere) | no | 0.921 | 0.809 | 0.263 | 0.424 | 5.6 | 0.102 | 0.966 | 0.774 |
| one global cell/nucleus ratio (2.73x) | no | 0.869 | 0.718 | 0.101 | 0.407 | 8.3 | 0.115 | 0.966 | 0.760 |
| pooled quantile map | no | 0.843 | 0.586 | 0.177 | 0.357 | 6.1 | 0.116 | 0.966 | 0.781 |
| nucleus's own library | no | 0.839 | -0.404 | 0.052 | 0.830 | 82.3 | 0.111 | 0.963 | 0.732 |
| (paper model, seed 1, for reference) | no | 0.903 | 0.682 | 0.145 | 0.357 | 6.2 | 0.115 | 0.967 | 0.795 |

With the label it joins the label-using group at the ceiling (gene corr 0.970 vs scGen 0.975; R² 0.93 is the best
non-trivial value in the table) and the quantile-map decode is the best row of the whole benchmark on every
within-type / population metric (energy 2.6, floor 0.6). Without the label it needs the cell/nucleus RNA-content ratio
from somewhere; one chemistry-wide ratio gives 0.869 (R² 0.72), above label-free scVI (0.857 / 0.43) but 0.03 below
the paper model, because the ratio is type-specific (1.2x glia and L5 ET, 2.5-3.7x other neurons). Reading the ratio
from the model's own latent fixes this without an inference label: the type predicted by majority vote over the 50
nearest training nuclei (97 % accurate) gives **0.962 / R² 0.91 / energy 2.6**, level with scANVI (which uses the
label at inference) and above labelled scVI; with no labels anywhere (K nearest training *cells* supply the library)
0.921 / 0.81 / 5.6, still above the paper model on every metric. **The paper's label-free row is therefore the
covariate model with the predicted-type quantile-map decode (0.962), and the "no labels anywhere" row is 0.921.** The
ImputeVAE 0.900 row becomes the no-covariate ablation. Rows `cvae_libsize_arm{A,B}[_qmap|_source|_gshift|_gqmap|_knn|
_knnq|_knntype|_knntypeq]` in `run/benchmark/test/results*/` (final full table: job 13822272).

### What changed relative to the old table, and why

1. **The cVAE's cell-type-level lead is gone.** Old all-data table: cVAE 0.927 vs scVI 0.386 gene corr. Held-out,
   with the count models decoded at a sensible library size: scVI 0.958, scANVI 0.962, scGen 0.975, scButterfly
   0.966, cVAE 0.898. The old scVI/scANVI/scGen numbers were an artefact of decoding at a fixed library of 1e4
   (median v3 neuron library is 35-51k, glia 5-12k): with one library for every type, the between-type RNA-content
   difference that drives much of the across-type gene correlation was erased. The old scButterfly numbers were
   the normalize_total(1e4) scale mismatch.
2. **Fairness, resolved with a label-free decode (job 12660720, 21 method dirs).** The per-type median library
   handed to scVI/scANVI/scGen at decode time comes from the cell-type *label* and from real training cells of that
   type. CT-mean, linreg, scGen (per-type shift vectors) and scANVI (labels in training) are label-using by
   construction; the cVAE uses no labels anywhere. Re-decoding the saved scVI / scANVI models label-free (their own
   inferred nucleus library x one global cell/nucleus ratio of 2.73; `redecode_scvi_libfree.py`) gives:

   | | ImputeVAE | scVI label-free | scANVI label-free | scANVI per-type library |
   |---|---|---|---|---|
   | gene corr | **0.898** | 0.857 ± 0.007 | 0.865 ± 0.002 | 0.962 |
   | R² | **0.723** | 0.433 ± 0.023 | 0.458 ± 0.024 | 0.719 |
   | delta Pearson | **0.930** | 0.910 | 0.913 | 0.970 |
   | energy distance | **7.5** | 9.2 ± 0.4 | 9.0 ± 0.4 | 16.4 |
   | classification | **0.960** | 0.922 | 0.929 | 0.930 |
   | coexpression within L5 IT | **0.790** | 0.691 | 0.717 | 0.442 |
   | neuron/glia marker ratio (truth 4.69) | **4.05** | 2.48 | 2.41 | 2.97 |
   | variance ratio | 0.147 | 0.151 | 0.150 | 0.074 |
   | std ratio | 0.305 | 0.304 | 0.307 | 0.162 |
   | within-CT Spearman vs input | 0.114 | **0.134** | 0.133 | 0.048 |
   | variance-rank Spearman | 0.845 | 0.852 | **0.862** | 0.669 |

   So the per-type library accounted for essentially all of scANVI's cell-type-level gain (0.865 -> 0.962), and
   label-free to label-free the cVAE leads on every accuracy and population-level metric. The three
   within-type *magnitude* metrics (variance ratio, std ratio, per-nucleus Spearman) are **tied** between the cVAE
   and label-free scVI: both pass the nucleus's library size through to the output, and that is where their
   within-type variance comes from. This is independent confirmation of the September diagnosis, from a different
   model.

3. **The cVAE's genuine advantage is within-type structure (blocks B and C), exactly where the September
   diagnostics said the model is weakest in absolute terms.** Energy distance to real cells 7.5 vs 12-17 for every
   other learned method (floor 0.6); twice the retained variance (0.15 vs 0.07); within-type coexpression 0.79
   vs 0.43-0.66; variance-rank Spearman 0.85 (tied with scButterfly); per-nucleus information retained 0.11 vs 0.05
   for the scVI family and 0 for scButterfly. All of these are small absolute numbers: every learned method is
   trivially separable from real cells by a random forest (AUROC ≥ 0.98), and raw nuclei keep 0.70 of the variance
   that the best learned method keeps 0.15 of.
4. **Trivial baselines are competitive on single metrics.** Linreg-shrunk (type-level prediction + the nucleus's
   own deviation) wins classification (0.978) and within-type Wasserstein (0.296) but is far from real cells in
   multivariate space (energy 19.7). This is the "CT-mean plus noise" failure mode that pooled per-gene metrics
   cannot see and energy distance can.
5. **Metric sanity.** Wilcoxon DE overlap penalises zero-dispersion outputs (CT-mean 0.22, linreg-shrunk 0.23
   vs 0.91 / 0.69 by logFC): report the logFC version. The random-forest two-sample test saturates for all methods
   and therefore does not rank them; energy distance does. scButterfly's neuron/glia ratio is undefined because it
   gives glia a zero neuronal-marker shift where real glia have +0.53.

### Validation-partition check (job 12640616; `run/benchmark/valid/results/`)
Same methods on the 8,033 validation nuclei (test / valid): CT-mean gene corr 0.981 / 0.982; ImputeVAE 0.898 / 0.915,
R² 0.723 / 0.762, energy 7.5 / 6.3; scGen 0.974 / 0.975; scANVI 0.957 / 0.968; scVI 0.953 / 0.957; scButterfly
0.966 / 0.959; linreg-shrunk 0.777 / 0.778. Rankings are identical on every headline metric. ImputeVAE is the one
method that scores noticeably better on validation than on test (+0.017 gene corr, +0.04 R²), consistent with the
validation partition having been used for its epoch-level monitoring; the test numbers are the ones to report.
Label-free variants on validation (job 12676334) agree with test: scVI label-free 0.859 / R² 0.41 / energy 8.2, scANVI label-free 0.868 / 0.43 / 8.1, vs ImputeVAE 0.915 / 0.76 / 6.3; variance ratio and per-nucleus Spearman again tied (0.15-0.17, 0.13).

### Consequence for the paper
Two defensible statements, and one that is no longer defensible:
- **Defensible:** among methods that use no cell-type labels at inference (ImputeVAE, scVI, scANVI decoded
  label-free), ImputeVAE is the most accurate at the cell-type level (gene corr 0.898 vs 0.86, R² 0.72 vs 0.45),
  produces the most cell-like population of imputed profiles (energy distance 7.5 vs 9.0, floor 0.6), keeps the
  most within-type gene-gene structure (0.79 vs 0.72), and best preserves cell-type-specific biology
  (neuron/glia ratio 4.05 vs 2.4; truth 4.69).
- **Defensible:** if labels are allowed at inference, a per-type library size alone lifts scANVI to 0.962 and
  scGen reaches 0.975, within 0.01-0.02 of the CT-mean ceiling (0.981); ImputeVAE at 0.898 is 0.08 below. The
  table must show both groups, and the label-using group should be framed as "label + real-cell RNA content
  given", not as competing translators.
- **Not defensible:** "ImputeVAE recovers per-cell variation." Its within-type variance (0.15 of real) is the same
  as label-free scVI's and, per the September diagnostics, is library-size variation. The library-size covariate
  retrain (`notes/libsize_covariate_review_2026-10-09.md`) is the experiment that decides whether a per-cell
  claim can ever be made; until then the paper's per-cell story is "population-level realism" (energy distance),
  not individual-cell fidelity.

## 6. Follow-ups this redo does not cover
- Donor-held-out split (requires retraining the cVAE; all donors are in every partition now).
- ~~Seed replicates of the cVAE (3 × 12 h GPU).~~ Done 2026-10-12, §5a.
- ~~Library-size covariate arms A/B (job 13445566, submitted 2026-10-16) -> `run/benchmark/test/cvae_libsize_*` once trained.~~ Done 2026-10-23, §5b.
- Latent-width ablation of the covariate model (8 / 4 dims; runs of 2026-10-23) -> `cvae_libsize_latent{8,4}*`, full table rerun 13822272.
- External matched sc/sn holdouts (Ding 2020, Bakken 2018) — a different question (generalisation).
