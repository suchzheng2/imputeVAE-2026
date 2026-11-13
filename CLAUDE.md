# imputeVAE

## HPC Session Sync Rules

This project runs on a cluster where `~/` (including `~/.claude/`) is wiped between sessions. The repo copies are the source of truth.

### On session start — run these syncs immediately:

1. **Memory:** Copy all files from `.memory/` (repo root) to `~/.claude/projects/-gpfs-gibbs-pi-gerstein-sz477-cell-v-nucleus-imputeVAE/memory/`
2. **Settings:** Copy `.claude/settings.local.json` (repo root) to the active project settings location so permissions take effect

### On every update:

3. **Memory files:** When any system memory file is created or updated, immediately write it back to BOTH the system dir AND repo `.memory/`
4. **Settings:** If `.claude/settings.local.json` changes, write back to repo `.claude/`
5. **Notes:** All project notes go in `notes/` directory (version-controlled)

---

VAE-based imputation for single-cell RNA-seq: learning to translate between cell and nucleus expression profiles.

## Environment

- **Conda env:** `impute2` (main), `abc_env` (scib metrics only)
- **Platform:** Yale HPC (SLURM), partitions: `pi_gerstein`, `scavenge_gpu` (faster but preemptible)
- **Python:** 3.12, PyTorch + CUDA

## Project Structure

```
train.py                  # Main entry point — parses args, dispatches to training loops
imputeVAE/                # Core package
  models.py               # VAE_JSD (MSE loss), VAE_NB (negative binomial loss)
  models_cond.py           # Conditioned decoder variant (modality embedding)
  training.py              # Stage 1 training loop (train_loop_ABC)
  staged_training.py       # Stage 2/3 joint training (fine_tune_stage3_v2)
  cond_training.py         # Conditioned decoder training loop
  data.py                  # Data loading, filtering, protocol splitting, dataloader creation
  stage3_eval.py           # Inline eval: shift scores, gene corr, Wasserstein, scIB
  testing.py               # Post-training evaluation functions
  divergences.py           # JSD, Wasserstein distance implementations
  visualization.py         # UMAP plotting
  plot_loss.py             # Loss curve plotting
  fine_tuning.py           # Legacy fine-tuning (stage2 old style)
  utils.py                 # GPU utils, helpers
data/                      # Gene lists, cell type lists, metadata
  training_gene_list_v3.csv  # Config B gene list (10,134 genes)
  Mouse.MitoCarta.csv        # Mitochondrial genes to exclude
  common_genes.csv           # Shared genes between cell/nuclei datasets
jobs/                      # SLURM job scripts
  stage1/, stage2/, stage3/  # Organized by training stage
  probe/                     # Quick probe experiments
  eval/                      # Evaluation job scripts
  logs/                      # SLURM stdout/stderr (slurm-{jobid}_{name}.out/.err)
run/                       # Training output (one folder per run)
  run_YYYY-MM-DD-HH-MM-SS_{id}/
    arguments.json           # Saved hyperparameters
    models/model.pt          # Final model checkpoint
    epoch_dump/              # Per-epoch data (train_epoch_N.json, valid.json)
    train_epoch_loss/loss.txt  # Epoch-level loss CSV
    valid_loss/              # Validation loss
    umap/                    # UMAP plots (eN_train_encoded.png, etc.)
analysis/                  # Analysis and evaluation scripts
  benchmark/               # Split-aware baseline benchmark (manifest split, h5py cache, evaluate_all.py)
                           #   results: run/benchmark/{test,valid}/results/; review: notes/benchmark_review_2026-10-09.md
notes/                     # Project notes and experiment tracking
  training_runs.md           # Master run inventory with all jobs, params, results
  mse_vs_nb_comparison.md    # Current comparison plan
  design_decisions.md        # Architecture decisions
cache/                     # Cached dataloaders (by data path); cache/benchmark/ = float32 raw counts per split (4.3 GB)
```

## Key Commands

```bash
# Submit a job
sbatch jobs/run_s1_cell_vae_js_v3.sh

# Check job status
sacct -j JOBID --format=JobID,JobName%20,State,Elapsed -n

# Check progress (last completed epoch)
grep "Epoch.*completed" jobs/logs/slurm-JOBID_*.out | tail -3

# Check for errors
tail -20 jobs/logs/slurm-JOBID_*.err
```

## Training Pipeline

1. **Stage 1:** Train on single modality (cell or nuclei). Reconstruction + JSD divergence with annealing.
2. **Stage 2 (stage3_v2 in code):** Joint training on both modalities with cascading loss annealing — cell always on, nuclei_recons ramps first, then nuclei_jsd, then alignment.
3. **Evaluation:** UMAPs, gene correlation, shift scores, Wasserstein distance, scIB metrics.

## Models

- **vae_js (VAE_JSD):** MSE reconstruction on log1p-normalized data. No softmax bottleneck.
- **vae_nb (VAE_NB):** Negative binomial reconstruction on raw counts. Outputs log-rates through softmax → proportion distortion risk.
- **ConditionalVAE:** Shared encoder/decoder with modality embedding. Used in `cond_training.py`.

## Data

- Raw data dirs: `10X_cells_v2_v3_AIBS/`, `10X_nuclei_v2_v3_AIBS/` (outside this repo)
- Protocol filtering via `--protocol v3` and `--batch c3A` (or `n3A` for nuclei)
- Gene filtering: `--gene_list data/training_gene_list_v3.csv` (Config B, 10,134 genes)
- Legacy filtering: `--detection_rate_threshold 5.0 --exclude_mitocarta True`

## Important Conventions

- **Run folders** are auto-generated: `run/run_YYYY-MM-DD-HH-MM-SS_{random}/`
- **Always use `--gene_list`** for new runs (not detection_rate_threshold) to ensure consistent gene sets
- **Delete `__pycache__/`** after code changes before submitting jobs — stale bytecode has caused failures
- **scavenge_gpu** is faster to schedule but can be preempted; use `--continue_training` to resume
- **Bad GPU nodes (seen 2026-10-09):** r105u17n01, r106u40n01, r106u37n01, r202u17n01 reject a second CUDA process or have ECC faults; add `--exclude=`
- **Broken GPU on r813u29n03 (pi_gerstein_gpu, seen 2026-10-16):** device 0000:31:00.0 fails in nvidia-smi and training silently falls back to CPU. On that node request `--gpus=2` and select a healthy GPU by UUID (guard block in `jobs/stage2/run_single_stage_cvae_libsize.sh`). Always check the log for a GPU line before trusting a run.
- **400G GPU jobs:** use `--partition=pi_gerstein_gpu,scavenge_gpu`; scavenge_gpu alone can wait days for a 1 TB node
- **Job scripts:** put `set -e` AFTER `source ~/.bashrc; conda activate` (bashrc returns nonzero and kills the job silently)
- **Baseline numbers:** quote only the split-aware benchmark (`run/benchmark/test/results/`), never `notes/experiments/baseline_comparison.md` (all-data, superseded)
- **Notes go in `notes/`**, not in code comments. `training_runs.md` is the master run tracker.
- All scripts are relative to project root, not `imputeVAE/` subdirectory
