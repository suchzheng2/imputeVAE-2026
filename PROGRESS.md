# imputeVAE — Progress Log

## Current Status (2026-03-13)

### Stage 1: Independent VAE_NB Training (v3 only) — COMPLETED

| Modality | Job ID | Data | Batch/Protocol | Status | Run Folder | Final Loss (total / recon / JSD) |
|----------|--------|------|----------------|--------|------------|----------------------------------|
| Cell | 928562 | `10X_cells_v2_v3_AIBS/` | c3A / v3 | **DONE** (Mar 13 00:06) | `run_2026-03-12-17-29-06_928562` | 0.771 / 0.756 / 0.031 |
| Nuclei | 928561 | `10X_nuclei_v2_v3_AIBS/` | n3A / v3 | **DONE** (Mar 13 00:06) | `run_2026-03-12-17-29-06_928561` | 0.462 / 0.457 / 0.009 |

**Data sizes**: Cells: 71,117 samples (v3 only), Nuclei: 116,691 samples (v3 only), 24,713 genes

**Hyperparameters** (same for both):
- Model: `vae_nb`, architecture: `[24713 → 1024 → 512 → 32]`
- 600 epochs, batch_size=128, lr=1e-4, 20% subset
- KL annealing: beta=0 (epochs 0-199), ramp to 0.5 (200-399), full (400-599)
- exclude_mitocarta=True, raw counts (no normalization), use_zinb=False

**Outputs**: Both runs include `model.pt`, loss curves, UMAPs, epoch dumps, evaluation directory

**Job scripts**: `jobs/train_stage1_cell_v3.sh`, `jobs/train_stage1_nuclei_v3.sh`

**Previous Stage 1 attempts** (superseded):
| Modality | Job ID | Status | Notes |
|----------|--------|--------|-------|
| Cell | 1851621 | Superseded | Earlier attempt, replaced by 928562 |
| Nuclei | 1851622 | Superseded | Earlier attempt, replaced by 928561 |

### Stage 2: Cross-Modality Fine-Tuning (READY TO RUN)

| Direction | Script | Description |
|-----------|--------|-------------|
| Cell → Nuclei | `jobs/train_stage2_cell2nuclei_v3.sh` | Load cell model, fine-tune on nuclei data |
| Nuclei → Cell | `jobs/train_stage2_nuclei2cell_v3.sh` | Load nuclei model, fine-tune on cell data |

**Strategy**: Load pretrained model from other modality → NB-recon only (epochs 0-199) → anneal JSD (200-399) → full JSD (400-599)

**Action needed**: Fill in model paths in stage 2 scripts:
- Cell Stage 1 run folder: `run_2026-03-12-17-29-06_928562`
- Nuclei Stage 1 run folder: `run_2026-03-12-17-29-06_928561`

### Stage 2 / Joint Training Outcomes (2026-03-13)

| Setup | Job ID | Script | Objective | Status |
|-------|--------|--------|-----------|--------|
| Setup 1 / Stage 2 | `1894703` | `jobs/train_stage2_cell2nuclei_v3.sh` | Initialize from cell stage-1 model, then train on nuclei only with nuclei reconstruction + annealed nuclei JSD | **COMPLETED** (Mar 13) |
| Setup 2 / Stage 2 | `1894705` | `jobs/train_stage2_nuclei2cell_v3.sh` | Initialize from nuclei stage-1 model, then train on cell only with cell reconstruction + annealed cell JSD | **COMPLETED** (Mar 13) |
| Setup 3 / Stage 2 | `1894704` | `jobs/train_stage2_cell2both_v3.sh` | Initialize from cell stage-1 model, then train jointly on nuclei+cell batches with nuclei reconstruction + nuclei JSD + cell reconstruction + cell JSD + alignment JSD | **TIMEOUT** at epoch 370 (Mar 13) |

**Run folders**
- Setup 1 / Cell -> Nuclei: `run_2026-03-13-00-55-11_1260842`
- Setup 2 / Nuclei -> Cell: `run_2026-03-13-01-05-55_998747`
- Setup 3 / Cell -> Both: `run_2026-03-13-00-55-52_997499`

**Live outcome summary**
- Setup 1 finished successfully in 2:33:55 and saved `models/model.pt`.
- Setup 2 finished successfully in 4:27:50 and saved `models/model.pt`.
- Setup 3 timed out after 6:00:28, but saved `models/model.pt` and `train_loss/epoch_370_train_loss.txt`.
- `squeue` is now empty for the stage-2 and dependent follow-up job IDs listed below.

**Dependent follow-up jobs**
- `1894859` depends on stage-2 job `1894703`:
  - parse the completed run folder from `jobs/logs/slurm-1894703_stage2_cell2nuclei_v3.out`
  - run `analysis/evaluate.py` on the stage-2 nuclei run
  - submit Setup 1 / Stage 3 using the resulting stage-2 model as initialization
  - immediately submit a dependent stage-3 evaluation job after the stage-3 training job is created
- `1894861` depends on stage-2 job `1894705`:
  - parse the completed run folder from `jobs/logs/slurm-1894705_stage2_nuclei2cell_v3.out`
  - run `analysis/evaluate.py` on the stage-2 cell run
  - submit Setup 2 / Stage 3 using the resulting stage-2 model as initialization
  - immediately submit a dependent stage-3 evaluation job after the stage-3 training job is created
- `1894860` depends on stage-2 job `1894704`:
  - parse the completed run folder from `jobs/logs/slurm-1894704_stage2_cell2both_v3.out`
  - run `analysis/evaluate.py` on the direct joint-training run
  - do **not** submit stage 3, since Setup 3 already uses the joint objective in stage 2

**Dependent job outcomes**
- `1894859` (`post-s2-c2n`) failed immediately with exit code `13:0`.
- `1894861` (`post-s2-n2c`) failed immediately with exit code `13:0`.
- `1894860` (`post-s2-c2both`) was cancelled after the parent joint job timed out.
- The `.err` files for the failed post-stage2 jobs are empty, so the failure likely occurred in shell control flow or environment setup before Python emitted a traceback.

**Resume plan for Setup 3**
- Added resume script: `jobs/train_stage2_cell2both_v3_continue_from_370.sh`
- Resume source checkpoint: `run/run_2026-03-13-00-55-52_997499/models/model.pt`
- Resume run folder: `run/run_2026-03-13-00-55-52_997499`
- Clean restart epoch: `371` (epoch 370 checkpoint already exists)

**Notes on intent**
- **Setup 1** is the main "make nuclei more cell-like" transfer path: stage 1 learns cell structure, stage 2 adapts that representation to nuclei counts.
- **Setup 2** is the reverse-direction control: it tests whether starting from nuclei and adapting to cells behaves differently.
- **Setup 3** skips the separate nuclei-only adaptation stage and goes straight from the cell model into the joint objective, to test whether direct alignment is enough.
- In all setups, `beta` controls the per-modality latent JSD term with annealing; in the joint objective, `alpha` controls the cross-modality alignment JSD term.

### Training Plan Going Forward (2026-03-13)

**Primary goal**: make nuclei profiles more cell-like without destroying cell reconstruction quality.

**Current staged design**
- **Stage 1**: single-modality training with reconstruction + annealed per-modality JSD.
- **Stage 2**: initialize from the other modality's stage-1 model, then train on the target modality only with reconstruction + annealed per-modality JSD.
- **Joint stage** (`fine_tune_stage3_v2`): train on both modalities together with:
  - nuclei reconstruction
  - nuclei JSD
  - cell reconstruction
  - cell JSD
  - alignment JSD

**Loss controls now available**
- `beta`: annealed weight for per-modality JSD terms
- `alpha`: weight for cross-modality alignment JSD
- `cell_recons_weight`: independent weight for cell reconstruction
- `cell_jsd_weight`: independent weight for cell JSD
- `nuclei_recons_weight`: independent weight for nuclei reconstruction
- `nuclei_jsd_weight`: independent weight for nuclei JSD

**Recommended default for joint training**
- Use the cell side as the anchor:
  - `cell_recons_weight = 2.0`
  - `cell_jsd_weight = 1.0`
  - `nuclei_recons_weight = 1.0`
  - `nuclei_jsd_weight = 1.0`
- Anneal `beta` as before for per-modality JSD.
- Start `alpha` later than reconstruction training so alignment does not destabilize early adaptation:
  - current job-script default: `anneal_alpha=True`, start at epoch 200, ramp over 200 epochs

**Planned comparison runs**
- Run A: `cell_recons_weight=2.0`, `alpha=0.5` with delayed alpha annealing
- Run B: `cell_recons_weight=3.0`, `alpha=0.5` if cell reconstruction still degrades
- Run C: lower `alpha` if alignment damages reconstruction, or raise `alpha` only after cell reconstruction remains stable

**Interpretation guide**
- If cell reconstruction degrades: increase `cell_recons_weight` and/or delay lower `alpha`
- If nuclei does not move enough toward cell: raise `alpha` gradually rather than raising `beta`
- If both modalities over-regularize: lower JSD weights or reduce learning rate in the joint stage

### Future Workflow Improvement

- Evaluation is currently orchestrated by separate SLURM follow-up jobs after training finishes.
- Future direction: make evaluation part of the training workflow itself, so each training run can automatically trigger `analysis/evaluate.py` at the end or expose evaluation as an integrated post-training phase.
- Preferred end state:
  - training code writes all metadata needed for evaluation
  - optional `--run_evaluation_after_train` flag or equivalent
  - stage 1 / stage 2 / stage 3 can all finish by automatically launching evaluation on their own run folder
  - less dependence on external job-chaining shell scripts

### Active-Run Safety Note (2026-03-13)

- We identified a mismatch between the current joint-training dump format and the evaluation stack:
  - joint runs currently save `epoch_dump/train.json`
  - joint runs store modality-specific keys such as `cell_decoded`, `nuclei_decoded`, `cell_encoded`, `nuclei_encoded`
  - many evaluators were originally written for single-modality dumps with `encoded` / `decoded` and epoch-specific filenames
- To avoid affecting the currently running/submitted training jobs, we did **not** change training output format yet.
- Safe fix applied now: evaluation code was made backward-compatible with both formats:
  - fallback support for `epoch_dump/train.json`
  - modality-aware loading for joint dumps
  - current active runs can be evaluated without touching training code
- Deferred until current run chain completes:
  - standardize joint training dumps to evaluator-friendly filenames like `train_epoch_<epoch>.json`
  - unify dump schema so training/evaluation no longer need compatibility branches

### V2/V3 Batch Diagnosis (COMPLETED)

- **Job**: 1850662 (completed)
- **Run analyzed**: `run_2026-03-04-12-59-01_308825` (nuclei, n3A, from Cytoplasm_imputation)
- **Output**: `/home/sz477/project/cell-imputing/Cytoplasm_imputation/run/run_2026-03-04-12-59-01_308825/evaluation/v2v3_diagnosis/`
- **Finding**: Raw nuclei data contains both 10X-v2 (8 donors, ~67K nuclei) and 10X-v3 (5 donors, ~40K nuclei). The banding pattern in raw heatmaps is confirmed to be from v2 vs v3 protocol differences.
- **Quantified v2/v3 count distribution gap (2026-03-11)**:
  - Nuclei: v2 mean 3,402 counts/cell vs v3 mean 17,042 (~5x); v2 detects 1,764 genes vs v3 3,461 (~2x); sparsity 93.2% vs 86.6%
  - Cells: v2 mean 13,529 counts/cell vs v3 mean 49,625 (~3.7x); v2 detects 4,551 genes vs v3 7,150 (~1.6x); sparsity 83.1% vs 73.4%
  - All values are raw integer counts (not normalized). Training code mixes v2+v3 with no batch correction.
  - Model likely learns protocol-specific count distributions rather than pure biological signal.
- **Files generated**:
  - `raw_nuclei_raw_heatmap_v2v3.png` — heatmap with v2/v3 color bar (natural order)
  - `raw_nuclei_raw_heatmap_v2v3_sorted.png` — sorted by protocol within each cell type
  - `raw_nuclei_raw_umap_v2v3.png` — UMAP colored by cell type, protocol, meta_batch_id
  - Same for cells (`raw_cell_*`)
  - `encoded_epoch{200,350,500}_umap_v2v3.png` — latent space UMAPs by sample/donor

---

## Code Changes (2026-03-11)

### New: `--protocol` argument (`train.py`, `data.py`)
- Added `--protocol {v2, v3, both}` to `train.py`
- Added `filter_by_protocol()` in `data.py` — used by both `create_dataloaders()` and `create_dataloaders_by_cell_type()`
- When `--protocol` is set, overrides the version digit in `--batch` (e.g. `--batch n3A --protocol v2` → filters for n2A)
- When `--protocol` is not set, original `--batch` filtering is preserved (backward compatible)

### New: `analysis/diagnose_v2v3_batch.py`
- Diagnostic script for v2/v3 batch effects
- Creates annotated heatmaps, raw data UMAPs, epoch encoded UMAPs
- SLURM script: `jobs/job_diagnose_v2v3.sh`

### Fix: `analysis/eval_utils.py` — protocol-aware `load_raw_data()`
- Added `protocol` and `batch` parameters to `load_raw_data()`
- When `protocol` is set, filters cells by `meta_batch_id` suffix (same logic as training `filter_by_protocol()`)
- Backward compatible — no filtering when `protocol=None`

### Fix: `analysis/eval_heatmaps.py` — reads protocol from run args
- `run_heatmap_analysis()` now reads `protocol` and `batch` from `arguments.json`
- Passes these to `load_raw_data()` so evaluation uses the same data subset as training

### Fix: `imputeVAE/staged_training.py` — model class & data path fixes
- Added `_create_model_from_args()` helper: creates VAE_NB/VAE_JSD/VAE_L1 based on `args.model`
- `train_loop_ABC_stage2_v2`: model_1 and model_2 now use `_create_model_from_args()` instead of hardcoded `VAE_JSD`
- `train_loop_ABC_stage3_v2`: same fix for model_1 and model_2
- Removed hardcoded `_g10k` data paths; now uses `args.nuclei_data_path` / `args.cell_data_path`
- Fixed adata reuse bug: loads separate adata for nuclei and cell data

---

## Previous Runs (from Cytoplasm_imputation)

| Run | Data | Batch | Epochs | Notes |
|-----|------|-------|--------|-------|
| `run_2026-03-04-12-59-01_308825` | nuclei v2+v3 | n3A | 600 | Nuclei stage 1 (old location) |
| `run_2026-03-04-17-42-43_3418456` | cells v2+v3 | c3A | 600 | Cell stage 1 (under imputeVAE/run/) |

---

## Known Issues & Next Steps

### Issues to Resolve

1. **V2/V3 count distribution gap in evaluation** — **FIXED (2026-03-11)**
   - Eval pipeline (`eval_heatmaps.py`) loaded ALL raw data (v2+v3) even when model trained on v3 only
   - Comparing v3-trained decoded output against v2+v3 mixed raw data produces misleading heatmaps
   - **Fix**: Added `protocol` and `batch` parameters to `eval_utils.py::load_raw_data()`. `eval_heatmaps.py::run_heatmap_analysis()` now reads `protocol` and `batch` from the run's `arguments.json` and passes them through, so evaluation data matches training data.

2. **`staged_training.py` hardcodes `VAE_JSD` for model loading** — **FIXED (2026-03-11)**
   - `train_loop_ABC_stage2_v2` and `train_loop_ABC_stage3_v2` created `VAE_JSD` models regardless of `--model vae_nb`
   - Loading `VAE_NB` state_dict into `VAE_JSD` would fail (VAE_NB has extra `log_theta` parameter)
   - **Fix**: Added `_create_model_from_args()` helper that selects VAE_NB/VAE_JSD/VAE_L1 based on `args.model`. Both stage2 and stage3 functions now use this helper for model_1 and model_2.

3. **`staged_training.py` hardcodes data paths with `_g10k` suffix** — **FIXED (2026-03-11)**
   - Lines 1028, 1036 hardcoded `args.data_path = "10X_nuclei_v2_v3_AIBS_g10k/"`
   - Reused same `adata` object for both modalities (cell and nuclei loaded from same adata)
   - **Fix**: Now uses `args.nuclei_data_path` / `args.cell_data_path` with fallback to non-`_g10k` defaults. Loads separate adata for nuclei and cell data via `load_raw_data()`.

4. **Stage 2 JSD is self-regularization, not cross-modality alignment**
   - Current stage 2 scripts use `VAE_NB`'s built-in JSD (latent vs prior)
   - This teaches the model to reconstruct the other modality and regularize its latent space
   - For explicit **cross-modality JSD alignment** (matching cell and nuclei latent distributions), need to write a new cross-modal fine-tuning loop for VAE_NB or adapt `train_loop_ABC_stage3_v2`

5. **V2/V3 count distribution gap — design consideration**
   - Nuclei: v2 mean 3,402 counts/cell vs v3 mean 17,042 (~5x gap)
   - Cells: v2 mean 13,529 counts/cell vs v3 mean 49,625 (~3.7x gap)
   - VAE_NB library-size normalization (`mu = library_size * softmax(rho)`) should in principle handle per-cell depth differences, but the systematic v2/v3 gap may still bias latent representations
   - Currently training on v3 only to avoid this; may revisit with explicit batch correction

### Next Steps

- [x] ~~Wait for stage 1 jobs to complete~~ — **DONE** (2026-03-13, jobs 928561 & 928562)
- [x] Fill in model paths in stage 2 scripts and submit
- [ ] Review v2v3 diagnosis plots to decide if v2 data should be included in training
- [x] Fix `staged_training.py` to support `VAE_NB` model loading (issues 2 & 3)
- [x] Fix evaluation pipeline to respect protocol filtering (issue 1)
- [ ] Consider whether to train with `--protocol both` (v2+v3 combined) as an experiment
- [ ] Restart Setup 3 joint run from epoch 371 using the saved epoch 370 checkpoint
- [ ] Diagnose why post-stage2 jobs `1894859` and `1894861` failed immediately
- [ ] Run evaluation on completed stage-2 models (`run_2026-03-13-00-55-11_1260842`, `run_2026-03-13-01-05-55_998747`)
- [ ] Set up evaluation pipeline (`analysis/evaluate.py`) for new runs
- [ ] Run evaluation on completed stage 1 models (cell: 928562, nuclei: 928561)
