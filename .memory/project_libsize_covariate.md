---
name: project-libsize-covariate
description: Library-size covariate retrain (paper gap A2) — DONE 2026-10-23: labelled decode at the benchmark ceiling group (0.970 / R2 0.93); label-free predicted-type decode (kNN over training nuclei, 97 % acc.) 0.962 / 0.91 / energy 2.6 = the reported nuclei-only number; no labels anywhere 0.921; v2 label-free 0.89; latent width: 2 active dims suffice (0.957), 1 dim keeps 5 types
metadata:
  type: project
---
Record: `notes/libsize_covariate_review_2026-10-09.md` (§3 plan, §5 implementation, **§6 results**: 6.1 latent usage, 6.2 variance, 6.3 v2 cross-chemistry, 6.4 benchmark, 6.4a label-free v2, 6.5 decisions); storyline `notes/paper_storyline_2026-10-23.md`; benchmark review §5b.

- Runs: arm A `run/run_2026-10-16_libsize_armA_jsd0.1_seed1` (13446475_1), arm B `..._armB_jsd0.01_seed1` (13445566_2); control `cvae_seed1` (`run_2026-10-09-19-13-44_611868`); latent-width `run/run_2026-10-23_libsize_armA_latent{8,4}_seed1` (13784373, done 2026-10-23).
- Latent: library share of imputed within-type variance 85 % -> 0 %; within-type variance 11x lower on 2-3 identity dims; collapse is objective-driven (arm B same) -> arm C dropped.
- **Benchmark (13780004)**: decode at per-type cell library (uses label) gene corr 0.970 / R² 0.927 (scGen 0.975, CT-mean 0.981); `quantile_map` decode best in the whole table on energy (2.6, floor 0.6), W1, DE overlap, coexp 0.79. Label-free decodes: source 0.839, pooled quantile 0.843, one global ratio 2.73x 0.869 / R² 0.72 (paper model 0.903 / 0.68; scVI label-free 0.857 / 0.43). Reason: per-type cell/nucleus ratio 1.2x (Astro, Endo, L5 ET) to 3.7x (Lamp5).
- **v2 cross-chemistry**: label 0.93 / R² 0.72 (paper 0.59 / -0.03); label-free pooled 0.86 / 0.58; v3 ratio does not transfer (v2 ratio 3.8x) -> state one scalar per chemistry.
- Decode modes live in `analysis/benchmark/predict_cvae.py --target_libsize` (ct_median_cell, quantile_map, source, global_shift, global_quantile_map, knn_cell_median, knn_quantile_map) and `analysis/cross_dataset_eval.py` (…_target, global_shift_train, knn_train_shift_target). Label-free kNN = K=50 nearest training cells in the aligned latent.
- **Label-free ladder (v3 test, one model, §6.4)**: label at inference 0.970; `knn_type_quantile_map` (type from 50 nearest training nuclei, 97 % acc.; training labels only) **0.962 / R² 0.91 / energy 2.6** = reported nuclei-only decode; `knn_quantile_map` (no labels anywhere) 0.921 / 0.81; global ratio 0.869; own library 0.839; paper model 0.903. Results in `run/benchmark/test/results_libfree_scale/`; final full table job 13822272 -> `results/`.
- v2 label-free: predicted type 0.888 (type prediction 77 % across chemistry), pooled 0.859, kNN cells 0.83-0.85; gap is type assignment, not scale.
- **Latent width (§6.7)**: 8 dims -> 2 active, 0.957 / clf 0.935 (L5 ET, Sncg merged); 4 dims -> 1 active, 0.910 / clf 0.56; within-type residual and its gene-gene structure shrink with width (0.48 -> 0.19 -> 0.02). Capacity is not the limit; 32 dims stays.
- Decision: per-cell framing withdrawn; paper at cell-type level + biology; covariate model with the predicted-type quantile-map decode = reported model (0.962 label-free); the old 0.900 ImputeVAE row = no-covariate ablation. Open: cross-chemistry type assignment (v2 77 %), Ding 2020 holdout, SCORCH application.

**Why:** both September diagnostics said imputed within-type variance was library size; the covariate route separates identity from RNA content and makes the output scale explicit.
**How to apply:** always state the decode library and whether it used the label; never claim per-cell variation. See [[project-benchmark-redo]], [[project-variance-zero-nonzero-fullmodel]], [[project-scorch-scoping]].
