---
name: Eval scripts should not request GPUs
description: Eval framework (analysis.eval.runner) runs on CPU — SLURM scripts must not request --gpus to avoid 0% utilization cancellation
type: feedback
---

Eval scripts (analysis.eval.runner) must NOT request `--gpus` in SLURM headers. The eval framework runs entirely on CPU (inference is <3s, rest is numpy/sklearn). Requesting a GPU triggers Yale HPC's 0% GPU utilization policy, which auto-cancels the job after ~1 hour.

**Why:** Job 3264971 was cancelled by HPC automated policy for 0% GPU utilization despite having completed its work. The eval finished but the job sat idle with an allocated GPU.

**How to apply:** When creating or editing eval job scripts (`jobs/eval/*.sh`), never include `--gpus=1`. Use CPU-only partitions (e.g. `--partition=pi_gerstein` without `_gpu`, or `day`). Training jobs still need GPUs.
