---
name: check-jobs
description: Check status of running SLURM jobs, show epoch progress and recent loss values. Use when the user asks about job status, training progress, or running experiments.
argument-hint: "[job_id]"
---

# Check Jobs

Check the status of running and recent SLURM jobs for this project.

## Steps

### 1. List running jobs

```bash
squeue -u sz477 --format="%.18i %.30j %.8T %.10M %.6D %R"
```

### 2. For each running job, check progress

Find the corresponding log file in `jobs/logs/` and show the last few completed epochs:

```bash
grep "Epoch.*completed" jobs/logs/slurm-{JOBID}_*.out | tail -5
```

Also check for errors:

```bash
tail -5 jobs/logs/slurm-{JOBID}_*.err
```

### 3. If a specific job ID is given as $ARGUMENTS

Focus on that job only. Show:
- Job status from `sacct`
- Last 5 completed epochs from the log
- Current loss components from the last epoch line
- Any errors from stderr
- The run folder (grep for `run_` path in stdout)

### 4. For completed jobs

If the job is no longer running, use `sacct` to show final status:

```bash
sacct -j {JOBID} --format=JobID,JobName%30,State,Elapsed,MaxRSS -n
```

### 5. Summary

Print a concise table of all jobs with:
- Job ID, name, state, elapsed time
- Current epoch / total epochs
- Latest loss value
- Run folder path
