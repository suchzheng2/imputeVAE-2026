# imputeVAE — Project Summary

*Last updated: 2026-07-23*

## Current Model

**Architecture:** Single-stage conditional VAE (`VAE_JSD_Cond`) with:
- **Input:** log1p-normalized scRNA-seq (10,134 genes, Config B gene list)
- **Conditioning:** Scalar modality embedding (1=cell, 0=nuclei) on both encoder and decoder
- **Loss:** MSE reconstruction + JSD divergence (normalized losses, no annealing needed)
- **Hidden:** 1024->512 encoder, 512->1024 decoder, 128-dim latent
- **Training data:** V3-only mouse motor cortex (Allen Brain Atlas), 71K cells + 40K nuclei
- **Best run:** `run/run_2026-07-18-19-40-51_570984` (job 7714771), 600 epochs, ~12h

**Core idea:** Encode nuclei with nuclei embedding -> decode with cell embedding -> produces imputed cell-like expression. The model learns the directed biological transformation, not just batch correction.

### Model artifacts

| Artifact | Path |
|----------|------|
| Model checkpoint | `run/run_2026-07-18-19-40-51_570984/models/model.pt` |
| Training args | `run/run_2026-07-18-19-40-51_570984/arguments.json` |
| Loss curves (plots) | `run/run_2026-07-18-19-40-51_570984/loss_curves/` |
| Training loss CSV | `run/run_2026-07-18-19-40-51_570984/train_epoch_loss/` |
| Validation loss CSV | `run/run_2026-07-18-19-40-51_570984/valid_epoch_loss/` |
| UMAPs (train+valid, per epoch) | `run/run_2026-07-18-19-40-51_570984/umap/` |
| Inline eval (shift, gene corr over time) | `run/run_2026-07-18-19-40-51_570984/inline_eval/` |
| Valid inline eval | `run/run_2026-07-18-19-40-51_570984/valid_inline_eval/` |

---

## Results (Full-Data cVAE)

### Head-to-head comparison

| Metric | cVAE (ours) | CT-mean | scVI | Raw Nuclei |
|--------|:-:|:-:|:-:|:-:|
| Gene correlation | **0.927** | 1.000* | 0.386 | 0.822 |
| R-squared | **0.798** | 1.000* | -0.669 | -- |
| Delta correlation | **0.950** | 1.000* | 0.524 | -- |
| Classification acc | **96.3%** | 100%* | 92.9% | 77.5% |
| Coexpression (Spearman) | **0.933** | 0.889 | 0.818 | 0.832 |
| Variance ratio | **0.158** | 0.000 | 0.034 | 0.697 |
| QQ gene corr | **0.944** | 0.808 | 0.930 | -- |
| Wasserstein | **0.300** | 0.399 | 0.947 | -- |
| DE overlap (top 100) | 0.489 | 0.639 | 0.451 | 0.467 |
| Per-cell corr (median) | 0.996 | 1.000 | 0.964 | -- |

*CT-mean is trivially perfect on CT-level metrics by construction (it IS the CT mean).

**Result locations:**

| Method | Eval JSON |
|--------|-----------|
| cVAE full (ours) | `run/cvae_full_eval/eval/eval_cvae_full.json` |
| cVAE 20% | `run/cvae_baseline/eval/eval_cvae.json` |
| CT-mean | `run/ctmean_baseline/eval/eval_ctmean.json` |
| scVI | `run/scvi_baseline/eval/eval_scvi.json` |
| CPM cVAE | `run/cpm_full_eval/eval/` (job 7885245, pending) |
| Cross-dataset (v2) | `run/cvae_full_eval/eval/eval_cvae_full_v2.json` |

**Comparison write-up:** `notes/experiments/baseline_comparison.md`

### Key wins over baselines

**vs scVI:** scVI treats cell/nuclei as batch effect and regresses it out rather than learning the directed transformation. Catastrophic on gene correlation (0.39 vs 0.93) and R-squared (-0.67 vs 0.80).

**vs CT-mean:** CT-mean can't produce any within-cell-type variation (variance ratio = 0). cVAE preserves 15.8% of real variance, has better coexpression (+0.04), better distributional fit (QQ 0.94 vs 0.81, Wasserstein 0.30 vs 0.40).

**Full vs 20% data:** More data improved variance (0.08->0.16, nearly doubled) and delta correlation (0.92->0.95) -- rare cell types benefit most.

### Per-cell-type delta correlation

| Cell Type | Delta r | Delta R-squared |
|-----------|---------|-----------------|
| L2/3 IT | 0.998 | 0.996 |
| L6 IT | 0.998 | 0.987 |
| L6 CT | 0.998 | 0.989 |
| L5 IT | 0.994 | 0.978 |
| Sst | 0.994 | 0.960 |
| L5/6 NP | 0.992 | 0.981 |
| Vip | 0.997 | 0.990 |
| Sncg | 0.990 | 0.976 |
| Pvalb | 0.989 | 0.910 |
| Lamp5 | 0.986 | 0.963 |
| L6b | 0.978 | 0.953 |
| Oligo | 0.964 | 0.828 |
| Macrophage | 0.962 | 0.915 |
| Endo | 0.954 | 0.893 |
| Astro | 0.952 | 0.865 |
| L5 ET | 0.950 | -0.560 |
| OPC | 0.885 | 0.629 |

Source: `run/cvae_full_eval/eval/eval_cvae_full.json` -> `delta_gt.per_ct_corr`

---

## Key Findings and Result Locations

### Finding 1: Model correctly adds cytoplasmic genes and removes nuclear lncRNAs

The imputation delta (imputed_cell - input_nuclei) shows biologically correct patterns:
- **Top increased:** Ribosomal proteins (Rpl/Rps), elongation factors (Eef1a1), calmodulin (Calm1), VAMP2, Tmsb4x, Actb -- all cytoplasmic/synaptic
- **Top decreased:** Malat1, Xist -- nuclear lncRNAs, correctly suppressed
- **Cell-type-specific:** Plp1 top only in Oligo (myelin), Mt1 top only in Astro, Camk2n1/Nrgn top in excitatory neurons
- 97% of genes (9,869/10,134) have positive median delta -- expected since nuclei miss cytoplasmic RNA

| Result | Path |
|--------|------|
| Delta analysis script | `analysis/delta_analysis.py` |
| Per-CT delta CSVs | `notes/experiments/delta_analysis/tables/<CT>_delta.csv` |
| Per-CT top genes | `notes/experiments/delta_analysis/tables/<CT>_top_genes.txt` |
| Top 100 gene lists | `notes/experiments/delta_analysis/top100_{increased,decreased}_genes.txt` |
| Global delta distribution plot | `notes/experiments/delta_analysis/global_delta_distribution.png` |
| Cross-CT delta heatmap | `notes/experiments/delta_analysis/cross_ct_delta_heatmap.png` |
| Consistency summary plot | `notes/experiments/delta_analysis/consistency_summary.png` |
| Delta histograms per CT | `notes/experiments/delta_analysis/histograms/` |
| Write-up | `notes/experiments/delta_analysis/README.md` |

Note: Delta analysis was run on the 20% dec-only model (job 4514645). The full-data cVAE eval confirms the same patterns via compartment metrics in `eval_cvae_full.json`.

### Finding 2: GO enrichment confirms cytoplasm/ribosome/synapse biology

GO analysis on the top increased genes shows:
- Ribosome (GO:0005840): p=3e-26
- Cytoplasm (GO:0005737): p=8e-08
- Synapse (GO:0045202): p=3e-14
- Decreased genes: no significant GO terms (mostly Gm* predicted + lncRNAs)

| Result | Path |
|--------|------|
| GO analysis script | `analysis/go_analysis.py` |
| GO results (training set) | `notes/experiments/delta_analysis/go_enrichment/increased_go_results.csv` |
| GO results (validation set) | `notes/experiments/delta_analysis/valid_set/go_enrichment/increased_go_results.csv` |
| Gene lists for GO input | `notes/experiments/delta_analysis/go_enrichment/top100_{increased,decreased}_genes.txt` |
| Background gene list | `notes/experiments/delta_analysis/go_enrichment/all_genes_background.txt` |

### Finding 3: Delta patterns generalize to held-out validation set

Training set delta vs validation set delta correlation: mean r=0.950 (overall), per-CT median r=0.989. Top-20 gene overlap between train/valid: 15.5/20 on average. High-n CTs (>50 cells): r>0.95; Low-n CTs (Astro=10, Endo=5): r~0.65.

| Result | Path |
|--------|------|
| Validation delta script | `analysis/delta_analysis_validation.py` |
| Per-CT delta CSVs (valid) | `notes/experiments/delta_analysis/valid_set/tables/<CT>_delta.csv` |
| Per-CT top genes (valid) | `notes/experiments/delta_analysis/valid_set/tables/<CT>_top_genes.txt` |
| Train vs valid comparison | `notes/experiments/delta_analysis/valid_set/train_vs_valid_comparison.csv` |
| Validation heatmap | `notes/experiments/delta_analysis/valid_set/cross_ct_delta_heatmap.png` |
| Validation consistency | `notes/experiments/delta_analysis/valid_set/consistency_summary.png` |
| Validation GO enrichment | `notes/experiments/delta_analysis/valid_set/go_enrichment/` |
| Enhanced plots | `notes/experiments/delta_analysis/valid_set/enhanced_plots/` |

### Finding 4: Compartment biology metrics validate directional correctness

From the full-data eval pipeline:
- **Ribosomal genes:** 60 found, 100% correct sign, mean delta +1.77, median +1.87
- **Cytoplasmic genes:** 18 found, 100% correct sign, mean delta +2.97, median +3.08
- **Neuron-specific genes:** 4.4x higher delta in neurons vs glia (ratio=4.42)
- **Nuclear lncRNAs (45-gene list):** 100% correct sign (re-eval job 7754430)

| Result | Path |
|--------|------|
| Compartment metrics | `run/cvae_full_eval/eval/eval_cvae_full.json` -> `compartment`, `ct_specific` |
| Nuclear lncRNA analysis script | `analysis/find_nuclear_lncrnas.py` |
| Data-driven lncRNA list (45 genes) | `notes/experiments/nuclear_lncrna_analysis/proposed_nuclear_lncrna_list.txt` |
| All genes delta for lncRNA analysis | `notes/experiments/nuclear_lncrna_analysis/all_genes_delta.csv` |
| Nuclear-enriched lncRNA details | `notes/experiments/nuclear_lncrna_analysis/nuclear_enriched_lncrnas.csv` |
| lncRNA metric write-up | `notes/experiments/baseline_comparison.md` (bottom section) |

### Finding 5: Imputed output preserves per-cell variation (but not fully)

Variance ratio = 0.158 (median within-CT per-gene variance of imputed / real cells). This means the model retains ~16% of real within-CT variation -- far better than CT-mean (0%) and scVI (3.4%), but still below raw nuclei (69.7%). More data helps: 20% model got 0.083, full model got 0.158 (nearly doubled).

| Result | Path |
|--------|------|
| Variance in eval JSON | `run/cvae_full_eval/eval/eval_cvae_full.json` -> `variance_ratio` |
| Quick variance check script | `analysis/quick_variance_check.py` (protocol-filtered) |
| Eval pipeline variance module | `analysis/eval/metrics/sample_level.py` |
| Within-CT Spearman | `run/cvae_full_eval/eval/eval_cvae_full.json` -> `within_ct_spearman` |

### Finding 6: Cell count imbalance affects rare cell types

At 20% subset: 13 CTs with <50 training samples (hit min_samples floor). Massive imbalance: L5 IT has 17,833 cells vs Endo has 113 cells (at 100%). Some CTs are modality-only: L6 IT Car3/SMC/VLMC (cell only), Doublet/Outside Mop (nuclei only).

| Result | Path |
|--------|------|
| Cell count diagnostic script | `analysis/cell_count_diagnostic.py` |
| Cell count CSV | `notes/experiments/cell_count_diagnostic/cell_counts.csv` |

### Finding 7: Imputed output cross-decodes to unseen v2 protocol data

Cross-dataset evaluation on v2 chemistry data (model trained only on v3) shows the model generalizes.

| Result | Path |
|--------|------|
| Cross-dataset eval job | 7758417 (completed) |
| Cross-dataset script | `analysis/cross_dataset_eval.py` |
| V2 eval JSON | `run/cvae_full_eval/eval/eval_cvae_full_v2.json` |

### Finding 8: Imputed data exported for downstream use

Full cross-decoded output (40K nuclei -> imputed cells) saved as numpy array.

| Result | Path |
|--------|------|
| Export script | `analysis/export_cvae_imputed.py` |
| Imputed expression matrix | `run/cvae_full_eval/imputed_cell_from_nucleus.npy` (40166 x 10134) |
| Observation metadata | `run/cvae_full_eval/nucleus_obs.csv` |

---

## Biological Validation

### Nuclear lncRNA metric (corrected 2026-07-20)
- Original 5-gene list had 2 genes (Meg3, Kcnq1ot1) with wrong expected sign -- ground truth shows them as cell-enriched in this tissue
- Meg3 is nuclear in glia (delta -0.80 to -0.86) but positive in neurons (+0.5 to +1.3); neurons dominate (~90%), so global mean is positive
- Expanded to 45 data-driven genes (ground truth delta < -0.2): cVAE gets 100% correct sign
- scVI gets 100% on old list only because it over-corrects everything downward (global mean_delta = -1.29)

---

## Key Design Decisions

1. **log1p >> CPM** -- confirmed across factorial experiment (16 runs). CPM models collapse or underperform.
2. **Normloss enables single-stage training** -- normalizing recon loss by epoch-0 value and JSD by ln(2) puts them on comparable [0,1] scale, eliminating need for multi-stage annealing.
3. **Scalar conditioning > learned embedding** -- fixed scalar (1=cell, 0=nuclei) works as well as or better than learned 8-dim embedding, with fewer parameters.
4. **Enc+dec conditioning > dec-only** -- conditioning both encoder and decoder gives better results than decoder-only.
5. **Cross-reconstruction deprioritized** -- compresses per-cell variance. The cVAE approach (single model, two embeddings) is cleaner.
6. **V3-only data** -- dropped v2 to avoid confounding chemistry batch effect.

Design decisions write-up: `notes/design_decisions.md`
Factorial experiment results: `notes/experiments/session_2026-06-12_results.md`

---

## Analysis Infrastructure

### Scripts

| Script | Purpose |
|--------|---------|
| `analysis/eval/runner.py` | Modular eval framework CLI (`python -m analysis.eval.runner`) |
| `analysis/evaluate_baseline_incremental.py` | Baseline eval pipeline (incremental saving, --skip) |
| `analysis/export_cvae_imputed.py` | Export cross-decoded .npy from model checkpoint |
| `analysis/delta_analysis.py` | Per-CT delta analysis (train set) |
| `analysis/delta_analysis_validation.py` | Delta analysis on held-out validation set |
| `analysis/go_analysis.py` | GO enrichment on delta gene lists |
| `analysis/find_nuclear_lncrnas.py` | Data-driven nuclear lncRNA identification |
| `analysis/quick_variance_check.py` | Within-CT variance ratio (protocol-filtered) |
| `analysis/cell_count_diagnostic.py` | Training sample counts per CT |
| `analysis/cross_dataset_eval.py` | Eval on unseen v2 protocol data |
| `analysis/baseline_ctmean.py` | CT-mean baseline |
| `analysis/baseline_scvi.py` | scVI baseline |
| `analysis/baseline_linear_regression.py` | Linear regression baseline |
| `analysis/plot_baseline_comparison.py` | Comparison bar charts |

### Job scripts

| Script | Purpose |
|--------|---------|
| `jobs/eval/run_eval_cpm_full.sh` | CPM cVAE full eval (job 7885245, pending) |

---

## Training History

See `notes/training_runs.md` for full inventory of all runs.
See `notes/experiments/` for detailed experiment write-ups.

---

## Open Items

- **CPM cVAE eval** (job 7885245) -- pending on pi_gerstein, will complete baseline comparison
- **Cross-dataset eval** (job 7758417, completed) -- v2 protocol generalization results in `eval_cvae_full_v2.json`
- **Variance** remains the hardest metric -- best is 0.158 (vs 0.697 raw), inherent tradeoff with CT-mean fidelity
- **Validation datasets** -- literature search for other tissues/species (ISMB poster priority)
- **RNA decay rate literature** -- nuclear vs cytoplasmic RNA lifetimes (not started)
- **Quantitative cross-modal metrics** -- PE says UMAPs not enough for paper (not started)
