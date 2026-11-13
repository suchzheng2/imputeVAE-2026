---
name: training runs pointer
description: Pointer to master training run inventory — see notes/training_runs.md for full details
type: reference
---

The authoritative run inventory is at `notes/training_runs.md` in the repo. It contains all runs, model paths, parameters, job statuses, and results.

Key runs as of 2026-04-02:
- Stage 1 cell models: old NB 24K (928562), MSE 10K (3250320), NB 10K (2189144), MSE 24K (3106245 running)
- Stage 2 shared: MSE 10K (1946018), NB 10K (475265)
- Stage 2 cond: MSE 10K (3102281/run 3345839), NB 10K (3102282/run 722169)
- Stage 2 cond+xr: MSE 10K (3106445 running) — key experiment combining cond decoder + cross-recons

Also see `notes/mse_vs_nb_comparison.md` for the extended run matrix and comparison plan.
