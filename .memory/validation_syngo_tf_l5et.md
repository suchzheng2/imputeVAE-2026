# SynGO, JASPAR TF, and L5 ET Diagnostic Validation (2026-08-15)

## SynGO Positive-Control Validation
- **Location:** `validation/syngo/syngo_brain_validation/`
- **Script:** `validation/syngo/scripts/syngo_brain_validation.py`
- **Gene set:** 1,275 non-ribosomal genes under SynGO CC synapse
- **Filtering:** human SynGO 1,678 -> mouse 1,655 -> in model 1,346 -> minus Rpl/Rps 71 = 1,275
- **Data:** split-aware validation only (`run/eval_v3_splitaware/cvae_imputed_valid.npy`)
- **17 cell types:** 12 neuronal + 5 non-neuronal (no L4, VLMC, SMC)

### SynGO Key Results
- Pearson r=0.945, R²=0.879, sign agreement=0.936
- Neuronal mean shift: 0.896 (emp), 0.959 (pred)
- Non-neuronal mean shift: 0.224 (emp), 0.151 (pred)
- **emp_diff = 0.673 [0.506, 0.828]** — strong neuronal > non-neuronal
- **pred_diff = 0.808 [0.704, 0.910]** — ImputeVAE recovers the pattern

### SynGO Sensitivity (excluding L5 ET)
- emp_diff INCREASES to 0.704 [0.562, 0.837] — pattern survives and strengthens
- R² improves from 0.879 to 0.944 (L5 ET was the main error source)
- Excluding low-sample CTs (Endo, Macrophage, OPC): emp_diff=0.632 [0.427, 0.832]
- Sensitivity script: `validation/syngo/syngo_brain_validation/syngo_sensitivity.py`
- Outputs: `syngo_sensitivity_summary.csv`, `syngo_sensitivity_exclude_l5et.csv`,
  `syngo_sensitivity_neuronal_vs_nonneuronal.png/pdf`,
  `syngo_distribution_sensitivity_summary.csv`, `syngo_distribution_sensitivity_panel.png/pdf`

### SynGO Biological Framing
- SynGO annotates encoded protein function/location, NOT RNA localization
- "SynGO provides brain-relevant biological consistency using protein-function/location annotations"
- Bootstrap: n=1000, seed=42, 95% percentile CI, resample CTs within groups
- Low-sample threshold: n_validation < 20 in either modality

## JASPAR TF Nuclear-Function Contrast (2026-08-15)
- **Location:** `validation/tf/tf_brain_validation/`
- **Script:** `validation/tf/scripts/jaspar_tf_validation.py`
- **Characterization:** `validation/tf/jaspar_characterization/`
- **Comparison:** `validation/tf/syngo_vs_tf/`
- **Source:** JASPAR CORE 2026 metadata (downloaded from mencius.uio.no)
- **Gene set:** 63 TF primary (non-ribosomal, in model), 61 singleton-only
- **Filtering:** JASPAR 4,572 rows -> mouse 371 -> 213 unique names -> 211 components -> 63 in model
- **Job:** 9057074 (scavenge, completed ~2.5 min)

### JASPAR Key Results
- Pearson r=0.904, R²=0.799
- **emp_diff = 0.257 [0.161, 0.353]** — weaker than SynGO (2.6x smaller)
- **pred_diff = 0.371 [0.308, 0.437]** — model slightly over-amplifies TF shift too
- Wasserstein improvement: **negative** (-0.050 median) — imputation doesn't improve TF distributions
- Proportion W-improved: 33.5% (vs 60.1% for SynGO)

### SynGO vs TF Comparison (strong specificity)
- SynGO emp_diff 0.673 vs TF emp_diff 0.257 (2.6x ratio)
- SynGO W improvement +0.150 vs TF W improvement -0.050
- TF singleton sensitivity nearly identical to primary (63 vs 61 genes)
- Pattern survives L5 ET exclusion in both gene sets
- Comparison outputs: `validation/tf/syngo_vs_tf/syngo_vs_jaspar_tf_summary.csv`, `.png/.pdf`

### TF Caveats
- TFs are nuclear-function contrast, NOT perfect negative control
- TF mRNAs can still be cytoplasmic
- JASPAR is a motif database; rows are motifs, not 1:1 gene records
- Complex motifs (Fos::Jun) split on "::" for primary; excluded in singleton sensitivity

## L5 ET Diagnostic (2026-08-15)
- **Location:** `validation/l5et_diagnostic/`
- **Script:** `validation/l5et_diagnostic/l5et_diagnostic.py`
- **Job:** 9207661 (scavenge, completed ~2 min)
- **21 output files** (7 CSVs, 6 PNGs, 6 PDFs, 1 README, 1 script)

### L5 ET Key Findings
- **Global positive bias:** slope=1.28, intercept=+0.47, 96.2% genes overpredicted (SynGO)
- **Not gene-specific:** intercept dominates; compare L5 IT slope=1.02, intercept=+0.06
- **Median other neuronal:** slope=1.00, intercept=-0.002 (near-perfect calibration)
- **Profile similarity:** L5 ET imputed most resembles L5 ET empirical (r=0.972), NOT L5 IT (r=0.896)
  - Model correctly identifies L5 ET identity, just over-amplifies shift magnitude
- **Sample-size asymmetry:** 32 cells vs 249 nuclei (7.8:1 ratio)
  - Not flagged by <20 threshold, but <50 cells is small for stable CT mean estimation
  - Only 32 of 161 total L5 ET cells in validation
- **Reference sensitivity:** valid_mean vs all_mean r=0.996 but MAE=0.075, individual genes differ by 0.35
- **Top overpredicted:** Kctd8, Tmem163, Vgf (SynGO); Bcl6, Bhlhe40, Egr1 (TF)

### L5 ET Conclusion
"L5 ET is a recurrent calibration stress-test population. ImputeVAE preserves much of the
gene ranking but over-amplifies the nucleus-to-cell shift magnitude in L5 ET, especially for
SynGO and TF genes. This likely reflects a combination of rare-cell-type calibration, small
validation whole-cell sample size, and unpaired cell/nucleus composition differences. The main
SynGO and TF conclusions are not driven by L5 ET because sensitivity analyses excluding L5 ET
remain consistent."
