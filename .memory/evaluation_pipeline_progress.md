# Unified Evaluation Pipeline - Implementation Progress

## Status: TESTING (all modules implemented, partial testing done)

## Implementation Order & Status

| # | File | Status | Notes |
|---|------|--------|-------|
| 1 | `analysis/eval_utils.py` | DONE | 280 lines. Shared utilities: load_common_filters, load_raw_data, load_decoded_stage1, compute_celltype_mean, stratified_subsample, find_run_folder |
| 2 | `analysis/eval_loss.py` | DONE + TESTED | 111 lines. Loss curve plots, per-celltype breakdowns. No raw data needed. Tested OK on run_2026-03-04-17-42-43_3418456 |
| 3 | `analysis/eval_decoder_scale.py` | DONE + TESTED | 213 lines. Decoder output scale stats over epochs. Tested OK on single epoch (599). ~2.7min per 4GB epoch dump |
| 4 | `analysis/eval_correlation.py` | DONE, imports OK | 380 lines. 3x3 correlation matrices, shift scores, multi-epoch shift trajectory. Requires raw data (~48GB). OOM on login node, needs SLURM |
| 5 | `analysis/eval_gene_correlation.py` | DONE, imports OK | 239 lines. Per-gene Pearson correlation, top/worst genes, histograms. Requires raw data |
| 6 | `analysis/eval_scib.py` | DONE, imports OK | 180 lines. iLISI/cLISI standalone for abc_env. Subprocess-isolated |
| 7 | `analysis/eval_heatmaps.py` | DONE, imports OK | 321 lines. Expression heatmaps, difference heatmaps, fractional error. Requires raw data |
| 8 | `analysis/evaluate.py` | DONE + TESTED | ~250 lines. Orchestrator. Tested with loss and scale successfully |
| 9 | Replace legacy wrappers | SKIPPED | generate_deep_report.py doesn't exist yet. generate_report.py works standalone |
| 10 | `jobs/job_submit_evaluate.sh` | DONE | SLURM: 64GB, 4 CPUs, pi_gerstein. Fixed path: `analysis/evaluate.py` (not `imputeVAE/analysis/`) |
| 11 | Test on run | IN PROGRESS | SLURM job 1420872 submitted for correlation,gene_corr,heatmaps on run_2026-03-04-17-42-43_3418456 |

## Testing Results

### Verified Working (login node)
1. `--analyses loss` -- 15.5s, 600 epochs, all plots generated
2. `--analyses scale --epochs 599` -- 162.6s for single epoch, CSVs + PNGs generated
3. All module imports verified OK

### Needs SLURM (OOM on login node)
- correlation, gene_corr, shift, heatmaps -- require loading 2x ~24GB raw H5ADs
- SLURM job 1420872 submitted: `sbatch jobs/job_submit_evaluate.sh run_2026-03-04-17-42-43_3418456 nuclei "correlation,gene_corr,heatmaps"`

### Not Yet Tested
- scib (requires abc_env via subprocess)
- Full `--analyses all` run

## Bugs Found & Fixed
- Path bug: scripts are at `analysis/` not `imputeVAE/analysis/` relative to project root. Fixed in job_submit_evaluate.sh
- Existing warning in generate_report.py:269 -- divide by zero when first loss value is 0 (pre-existing, not our bug)

## Key Finding: 10X-v2 vs 10X-v3 Count Distribution Mismatch (2026-03-11)

Training code mixes v2 and v3 data without any separation or batch correction. Raw count distributions differ dramatically:

### Nuclei (116,971 total: 76,525 v2 + 40,446 v3)
| Metric | 10X-v2 | 10X-v3 | Ratio |
|---|---|---|---|
| Mean total counts/cell | 3,402 | 17,042 | ~5x |
| Mean genes detected/cell | 1,764 | 3,461 | ~2x |
| Sparsity | 93.2% | 86.6% | |
| Non-zero median | 1.0 | 2.0 | |

### Cells (193,824 total: 122,641 v2 + 71,183 v3)
| Metric | 10X-v2 | 10X-v3 | Ratio |
|---|---|---|---|
| Mean total counts/cell | 13,529 | 49,625 | ~3.7x |
| Mean genes detected/cell | 4,551 | 7,150 | ~1.6x |
| Sparsity | 83.1% | 73.4% | |
| Non-zero median | 1.0 | 3.0 | |

- All values are raw integer counts (not normalized)
- v3 captures 3-5x more total counts and ~2x more genes per cell
- This is likely the source of the heatmap banding pattern
- Model may be learning protocol-specific count distributions rather than true biological signal

## Next Steps
1. Check SLURM job 1420872 output: `cat jobs/logs/eval_1420872.out`
2. If successful, run full `--analyses all` via SLURM
3. Test scib subprocess (may need to verify abc_env has correct scib version)

## Output Structure
```
run_folder/evaluation/
  manifest.json
  summary.txt
  loss_curves/
  decoder_scale/
  correlation/epoch_N/
  gene_correlation/epoch_N/
  shift_over_epochs/
  scib_metrics/epoch_N/
  heatmaps/epoch_N/
```

## Key Design Decisions
- shift_score = corr(nuclei_decoded, raw_cell) - corr(nuclei_decoded, raw_nuclei) (unnormalized)
- scib runs via `conda run -n abc_env python eval_scib.py ...` subprocess (30min timeout)
- All modules accept --run_folder and can run standalone
- Raw H5AD loaded once in orchestrator, passed to modules needing it
- Heatmaps default to last epoch only (expensive)
- 64GB SLURM memory for raw data (~24GB each) + epoch dumps (~4GB each)
- Each epoch dump is ~4GB JSON, ~2.7min to process for scale analysis
