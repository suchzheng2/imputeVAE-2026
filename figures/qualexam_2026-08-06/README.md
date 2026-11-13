# Qualifying Exam Figures 3, 4, 5

Generated: 2026-08-06

## Script
`generate_figures.py`

## Input files
- Eval JSONs: `run/{method}/eval/eval_*.json`
- Prediction arrays: `run/{method}/imputed_cell_from_nucleus.npy` (40166 x 10134, log1p, full data)
- Pre-computed deltas: `compartment_classification/gt_delta_per_ct.csv`
- Raw h5ad: loaded via `analysis/evaluate_baseline.py:load_raw_data()` pattern

## Output files
- `figure3_directed_reconstruction.{pdf,svg,png}` + `_data.csv` + `_caption.txt`
- `figure3_empirical_delta_matrix.csv`, `figure3_predicted_delta_matrix.csv`
- `figure4_cell_type_specificity.{pdf,svg,png}` + `_data.csv` + `_caption.txt`
- `figure5_variance_retention.{pdf,svg,png}` + `_data.csv` + `_caption.txt`
- `figure_data_audit.{txt,csv}`
- `missing_figure_data.txt`

## Data partition
All analyses use FULL DATA. No validation-only arrays available.
See `figure_data_audit.txt` for details.
