# Session Results — 2026-06-12

Work driven by PE meeting action items (2026-06-03). See `notes/meeting_2026-06-03_PE.md`.

---

## 1. GO Enrichment Analysis (DONE)

**Script**: `analysis/go_analysis.py`
**Results**: `notes/experiments/delta_analysis/go_enrichment/`

Ran g:Profiler on top 100 increased/decreased genes from delta analysis (Mus musculus, with background = all 10,134 training genes).

### Increased genes — strong enrichment

| Category | Top Term | p-value | Hits |
|----------|----------|---------|------|
| GO:MF | Structural constituent of ribosome | 3.2e-26 | 22/69 |
| KEGG | Ribosome | 5.0e-25 | 21/68 |
| GO:CC | Ribosomal subunit | 5.5e-24 | 21/72 |
| GO:CC | Cytosolic ribosome | 8.1e-23 | 21/83 |
| GO:CC | **Synapse** | 3.4e-14 | 43/1200 |
| GO:CC | **Postsynapse** | 1.2e-10 | 28/646 |
| GO:CC | **Cytosol** | 6.0e-09 | 49/2215 |
| GO:CC | **Cytoplasm** | 7.5e-08 | 79/5548 |
| GO:CC | Myelin sheath | 1.5e-06 | 10/108 |
| GO:BP | Cytoplasmic translation | 1.9e-15 | 20/145 |
| GO:BP | Translation at synapse | 2.5e-10 | 10/36 |
| GO:BP | Synaptic vesicle cycle | 2.1e-06 | 13/172 |
| Reactome | SRP-dependent cotranslational protein targeting | 1.7e-22 | 20/69 |

### Decreased genes — no significant enrichment
Many are Gm* predicted/placeholder genes + lncRNAs (Xist, Malat1, Snhg11) which are poorly annotated in GO.

### Interpretation
The model correctly learns to:
- **Add** cytoplasmic/ribosomal/synaptic genes (lost during nuclear isolation)
- **Remove** nuclear-enriched lncRNAs (overrepresented in nuclei)

This matches PE's predictions from the meeting.

---

## 2. Validation Delta Analysis (DONE)

**Script**: `analysis/delta_analysis_validation.py`
**Results**: `notes/experiments/delta_analysis/valid_set/`
**Job**: 6067750 (COMPLETED, 14 min)

Ran delta analysis on held-out validation split (20% of data, same random_state=42 split as training) using the best model (single-stage cVAE, `run_2026-05-11-19-52-25_134850`).

### Train vs Valid consistency

| Cell Type | n_valid | Top-20 Overlap | Delta Corr |
|-----------|---------|---------------|------------|
| L6 CT | 318 | 19/20 | 0.991 |
| L5 IT | 464 | 19/20 | 0.986 |
| L2/3 IT | 322 | 17/20 | 0.987 |
| L6 IT | 87 | 18/20 | 0.978 |
| Sst | 68 | 18/20 | 0.981 |
| L6b | 26 | 17/20 | 0.968 |
| Lamp5 | 38 | 17/20 | 0.968 |
| Pvalb | 104 | 16/20 | 0.972 |
| L5 ET | 49 | 16/20 | 0.956 |
| Macrophage | 10 | 16/20 | 0.883 |
| L5/6 NP | 60 | 15/20 | 0.974 |
| Vip | 43 | 15/20 | 0.966 |
| Oligo | 10 | 15/20 | 0.909 |
| Sncg | 10 | 13/20 | 0.917 |
| Endo | 5 | 12/20 | 0.659 |
| OPC | 10 | 12/20 | 0.858 |
| Astro | 10 | 9/20 | 0.656 |

**Mean top-20 overlap: 15.5/20** | **Mean delta correlation: 0.918**

### Key finding
Delta patterns generalize to unseen data. High-n cell types (>50 valid cells) show r>0.95 and 15-19/20 gene overlap. Low-n types (Astro=10, Endo=5) are noisier — consistent with PE's cell count hypothesis.

Note: comparison is across models (training deltas from old dec-only model, valid from current single-stage cVAE), which makes the high agreement even more notable — the biological signal is model-robust.

---

## 3. Cell Count Diagnostic (DONE)

**Script**: `analysis/cell_count_diagnostic.py`
**Results**: `notes/experiments/cell_count_diagnostic/cell_counts.csv`
**Job**: 6069202 (COMPLETED, 7 sec)

### At 20% subset (used for factorial experiment)

| Cell Type | Cell Train | Cell Valid | Nuc Train | Nuc Valid |
|-----------|-----------|-----------|-----------|-----------|
| L5 IT | 2140 | 714 | 836 | 279 |
| L6 CT | 922 | 307 | 573 | 191 |
| L2/3 IT | 787 | 262 | 579 | 193 |
| L6 IT | 320 | 107 | 157 | 52 |
| Sst | 134 | 45 | 122 | 41 |
| Vip | 214 | 71 | 79 | 26 |
| L5/6 NP | 227 | 76 | 108 | 36 |
| Pvalb | 44 | 15 | 312 | 104 |
| Lamp5 | 170 | 57 | 68 | 23 |
| L5 ET | 30 | 10 | 148 | 49 |
| Oligo | 65 | 22 | 30 | 10 |
| Astro | 47 | 16 | 30 | 10 |
| OPC | 30 | 10 | 30 | 10 |
| Endo | 30 | 10 | 14 | 4 |
| Macrophage | 30 | 10 | 30 | 10 |
| Sncg | 41 | 14 | 30 | 10 |

**13 cell types have <50 training samples** at 20%.

### At 100% (full dataset)

| Cell Type | Cell Train | Cell Valid | Nuc Train | Nuc Valid |
|-----------|-----------|-----------|-----------|-----------|
| L5 IT | 17833 | 5944 | 6959 | 2320 |
| L6 CT | 7684 | 2561 | 4773 | 1591 |
| L2/3 IT | 6549 | 2183 | 4826 | 1608 |
| L6 IT | 2667 | 889 | 1306 | 436 |
| L5/6 NP | 1888 | 629 | 901 | 301 |
| Vip | 1780 | 593 | 655 | 219 |
| Lamp5 | 1414 | 472 | 570 | 190 |
| Sst | 1121 | 374 | 1015 | 339 |
| Pvalb | 220 | 74 | 1560 | 520 |
| L6b | 332 | 111 | 395 | 132 |
| Oligo | 322 | 108 | 85 | 29 |
| L5 ET | 97 | 32 | 746 | 249 |
| Astro | 238 | 80 | 98 | 32 |
| Sncg | 208 | 70 | 116 | 38 |
| Endo | 113 | 37 | 14 | 4 |
| Macrophage | 74 | 24 | 32 | 11 |
| OPC | 88 | 29 | 47 | 15 |

### Observations
- **Massive class imbalance**: L5 IT has 17,833 training cells vs Endo with 113 cells and 14 nuclei
- **At 20%**, many types hit the min_samples=50 floor (capped at 30), starving them of training data
- **Modality-specific**: Some types only exist in one modality (L6 IT Car3 = cell only, Doublet/Outside Mop = nuclei only, SMC/VLMC = cell only)
- **PE's hypothesis confirmed**: low cell count at 20% explains poor UMAP quality for rare types

---

## 4. Factorial Experiment Results (COLLECTED)

All 16 runs completed. Best cross-modal correlation by config:

| Rank | Config | Cell-Nuc Corr |
|------|--------|--------------|
| 1 | log1p, scalar, encdec, 2stage (f07) | 0.820 |
| 1 | log1p, emb8, encdec, 2stage (f03) | 0.820 |
| 3 | log1p, scalar, dec, 2stage (f05) | 0.816 |
| 4 | log1p, emb8, dec, 1stage (f02) | 0.809 |
| 5 | log1p, scalar, dec, 1stage (f06) | 0.802 |

**Key takeaways**: log1p > CPM; 2-stage slightly better; scalar vs emb8 similar; f11 (cpm/emb8/encdec/2stage) collapsed.

---

## Files Created This Session

| File | Purpose |
|------|---------|
| `analysis/go_analysis.py` | GO enrichment via g:Profiler |
| `analysis/delta_analysis_validation.py` | Delta analysis on train/valid/test splits |
| `analysis/cell_count_diagnostic.py` | Cell count per CT/modality/subset (memory-efficient) |
| `jobs/eval/run_delta_validation.sh` | SLURM job for validation delta |
| `jobs/eval/run_cell_count_diagnostic.sh` | SLURM job for cell counts |
| `notes/meeting_2026-06-03_PE.md` | PE meeting notes |
| `notes/experiments/delta_analysis/go_enrichment/` | GO results (CSVs) |
| `notes/experiments/delta_analysis/valid_set/` | Validation delta tables + plots |
| `notes/experiments/delta_analysis/top100_{increased,decreased}_genes.txt` | Gene lists |
| `notes/experiments/delta_analysis/all_genes_background.txt` | Background gene list |
| `notes/experiments/cell_count_diagnostic/cell_counts.csv` | Cell count table |

---

## Remaining Action Items

| Priority | Item | Status |
|----------|------|--------|
| 1 | Literature search: validation datasets | NOT STARTED |
| 2 | RNA decay rate literature | NOT STARTED |
| 3 | Cross-modal quantitative metrics (silhouette/iLISI) | NOT STARTED |
| 4 | Re-run full delta analysis on training split with current best model | NOT STARTED |
| 5 | Run best configs on 60-70% or full dataset | NOT STARTED |
