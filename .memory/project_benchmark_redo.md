---
name: project-benchmark-redo
description: Split-aware benchmark redo started 2026-10-09 — new analysis/benchmark pipeline (manifest split, h5py cache, test partition, per-CT library decode, within-CT metrics, bootstrap CIs); jobs and where results land
metadata:
  type: project
---
Review + protocol: `notes/benchmark_review_2026-10-09.md`. Code: `analysis/benchmark/` (data_split.py, build_cache.py,
train_*.py, predict_cvae.py, evaluate_all.py). Jobs: `jobs/benchmark/submit_all.sh [test|valid] [cache_jobid]`.
Outputs: `run/benchmark/<partition>/<method>[_seedN]/imputed_eval.npy`, tables in `run/benchmark/<partition>/results/`.

Key facts found 2026-10-09:
- Old `analysis/benchmark/data_split.py` used its OWN per-CT shuffle split, not the cVAE's sklearn split -> would have leaked. Now reads `data/split_manifest.tsv`.
- Raw h5ad are dense int64 (41 GB cells / 24 GB nuclei); `read_h5ad` loads whole -> all 64 GB jobs OOM'd. Cache in `cache/benchmark/` built with h5py block reads (job ~32 GB).
- scvi-tools lives in conda env `scvi2` (not impute2); scButterfly code in impute2.
- Report on TEST partition (8,034 nuclei / 14,224 cells); valid was used for cVAE epoch monitoring.
- Count models decoded at per-CT median training-cell library (old 1e4 penalised R2/W1); scButterfly trained on log1p(raw).
- CT-mean(train) = sampling-noise ceiling for CT-level metrics, no longer "trivially 1.0".
- Bad GPU nodes seen 2026-10-09 (CUDA init / busy / ECC): r105u17n01, r106u40n01, r106u37n01, r202u17n01 -> use --exclude.
- cVAE seed replicates DONE (job 12646907 -> eval 12686636, 2026-10-12; `benchmark_review` §5a): 3 runs give gene corr 0.900 ± 0.002, R2 0.688 ± 0.032, variance ratio 0.151 ± 0.009 (= label-free scVI 0.151 ± 0.003), energy 6.8 ± 0.7, within-CT Spearman 0.114 ± 0.001, clf 0.963 ± 0.004; neuron/glia ratio unstable (4.1-7.3). Every §5 conclusion holds. Seed 2 folder used = `run_2026-10-09-19-36-35_450293` (job preempted 9x, restarted from scratch each time; ~37 GB junk folders listed in training_runs.md, quota 4087/4096 GiB). Same seed, two complete attempts: per-cell r 0.9987 but within-type variance corr down to 0.84 -> seed does not pin GPU nondeterminism.
- **FINAL test-partition results (job 12641011, 2026-10-09)**: cVAE gene corr 0.898 vs scGen 0.975 / scANVI 0.962 / scVI 0.958 / scButterfly 0.966 (ceiling CT-mean 0.981). The old scVI 0.386 was a decode-at-1e4 library artefact. cVAE leads only on within-type structure: energy distance 7.5 vs 12-17 (floor 0.6), var ratio 0.15 vs 0.07, within-CT coexp 0.79 vs 0.44-0.66, within-CT Spearman vs input 0.11 vs 0.05. All learned methods separable from real cells by RF (AUROC>=0.98). linreg_shrunk wins classification 0.978.
- Label-free re-decode done (job 12660720): scVI_libfree gene corr 0.857, scANVI_libfree 0.865, R2 0.43-0.46, energy 9.0-9.2 -> cVAE (0.898 / 0.72 / 7.5) leads label-free to label-free on accuracy, energy distance, classification, within-CT coexp, neuron/glia ratio. BUT variance ratio (0.15), std ratio (0.30), within-CT Spearman vs input (0.11-0.13) are TIED with label-free scVI: both are nucleus-library pass-through. Per-type library accounted for all of scANVI's gain (0.865 -> 0.962).
- Validation partition (jobs 12640616/12676334) reproduces every test ranking; cVAE +0.017 gene corr on valid (epoch-monitoring optimism) -> report test.
- Paper framing: two groups (label-free vs label-using at inference). Defensible: best label-free method + most cell-like population. Not defensible: per-cell variation recovery (library-size artefact).
- Job scripts: `set -e` must come AFTER `source ~/.bashrc` (bashrc returns nonzero -> silent FAILED at 00:00:00).

**Why:** user asked to redo benchmarking carefully and review the metrics.
**How to apply:** use this pipeline for all future baseline numbers; old table in baseline_comparison.md is all-data and not comparable.
