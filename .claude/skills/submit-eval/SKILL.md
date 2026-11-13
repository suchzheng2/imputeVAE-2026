---
name: submit-eval
description: Create and submit a SLURM eval job for a completed training run. Use when a training job finishes and needs post-training evaluation.
argument-hint: "<run_folder>"
---

# Submit Eval

Create and submit a post-training evaluation job for a completed training run.

## Input

$ARGUMENTS should be the run folder name (e.g. `run_2026-04-02-10-51-32_3738876`) or a job ID to look up.

## Steps

### 1. Identify the run

- If a job ID is given, find the run folder from the log: `grep "run_" jobs/logs/slurm-{JOBID}_*.out | head -1`
- Verify the run folder exists under `run/`
- Read `run/{folder}/arguments.json` to get model type, gene list, data path, architecture params

### 2. Check existing eval scripts

Look at existing eval scripts in `jobs/eval/` for templates. Key patterns:
- MSE shared decoder: `analysis/evaluate.py` with appropriate args
- MSE conditioned decoder: needs `--fine_tune_cond_stage3 True`
- NB models: needs `--raw_data_normalization raw`

### 3. Create the eval script

Write a new SLURM script to `jobs/eval/eval_{short_name}.sh` with:
- Partition: `pi_gerstein` (not scavenge — eval should not be preempted)
- Memory: 200G (eval is lighter than training)
- Time: 2h
- Conda env: `impute2`
- Delete `__pycache__` before running (prevent stale bytecode)
- Run the appropriate evaluation script

### 4. Submit

```bash
sbatch jobs/eval/eval_{short_name}.sh
```

### 5. Record

Print the job ID and remind to update `notes/training_runs.md` with the eval job info.

## Important

- Always delete `__pycache__` before submitting — stale bytecode has caused failures
- Use `pi_gerstein` partition for eval (not scavenge_gpu) to avoid preemption
- Check that the model.pt file exists in the run folder before submitting
