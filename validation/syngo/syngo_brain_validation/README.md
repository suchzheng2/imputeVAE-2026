# SynGO Brain-Specific Biological Consistency Analysis

## Question

Do non-ribosomal genes encoding synaptic proteins exhibit larger nucleus-to-cell
expression shifts in neuronal than in non-neuronal populations, and does ImputeVAE
reproduce the magnitude and cell-type dependence of those shifts?

SynGO annotates the function and localization of the encoded proteins, not RNA
localization. This analysis tests brain-specific biological consistency.

## Input Files

| Input | Path | Notes |
|-------|------|-------|
| Gene list | `data/training_gene_list_v3.csv` | 10,134 genes (Config B) |
| Split manifest | `data/split_manifest.tsv` | Train/valid/test split |
| Common cell types | `data/common_cell_types.csv` | 17 shared cell types |
| Common genes | `data/common_genes.csv` | Shared across modalities |
| Ortholog mapping | `data/external/mouse_human_orthologs.csv` | Ensembl BioMart |
| SynGO archive | `validation/syngo/syngo1.3_complete_data.zip` | v1.3 |
| Raw cells | `10X_cells_v2_v3_AIBS/data.h5ad` (c3A batch) | log1p normalized |
| Raw nuclei | `10X_nuclei_v2_v3_AIBS/data.h5ad` (n3A batch) | log1p normalized |
| ImputeVAE predictions | `run/eval_v3_splitaware/cvae_imputed_valid.npy` | Validation only |

## Validation Partition

- Only validation observations are used for all metrics (20% partition)
- Training observations are never loaded for evaluation
- Test observations are never loaded
- Split defined in `data/split_manifest.tsv`, random_state=42

## Matrix Shapes and Alignment

- Validation cells: shape verified at runtime
- Validation nuclei: shape verified at runtime
- ImputeVAE imputed: (8033, 10134), validation nuclei only
- Gene order: all three matrices verified identical (sorted by gene name)
- Cell type alignment: imputed vs observed nuclei verified at 100%
- Cross-decoded nuclei use the same validation nuclear observations

## SynGO Hierarchy Construction

1. SynGO v1.3 ontologies.xlsx: each CC term row contains propagated gene lists
2. GO:0045202 ("synapse") row contains all descendant-term genes
3. No recursive traversal needed; the comma-separated gene list is pre-propagated

## Human-to-Mouse Mapping

- Primary: one-to-one orthologs from `data/external/mouse_human_orthologs.csv`
  (Ensembl BioMart, mouse_sym unique, filtered to one-to-one human->mouse)
- Fallback: capitalization heuristic (GENE -> Gene) for genes not in ortholog table,
  labeled explicitly in syngo_gene_metadata.csv
- 1275 non-ribosomal genes in primary analysis

## Ribosomal Exclusion

Genes with mouse symbols starting with Rpl or Rps are excluded from the primary
set. They encode cytoplasmic ribosomal proteins expected to show large positive
shifts regardless of synaptic biology.

## Equations

For SynGO gene g and cell type c:
- empirical_delta[g,c] = mean(valid_cell[g,c]) - mean(valid_nuc[g,c])
- predicted_delta[g,c] = mean(imputed_valid_nuc[g,c]) - mean(valid_nuc[g,c])

Cell-type mean SynGO shift:
- empirical_mean_shift[c] = mean of empirical_delta[g,c] across SynGO genes g
- predicted_mean_shift[c] = mean of predicted_delta[g,c] across SynGO genes g

## Bootstrap Procedure

- 1000 replicates, seed=42
- Cell-type shifts: resample whole observations within modality and cell type;
  cross-decoded and observed nuclei are paired when resampling
- Neuronal vs non-neuronal: resample cell types within group (12 neuronal, 5 non-neuronal)
- 95% CI from 2.5th and 97.5th percentiles

## Neuronal / Non-Neuronal Classification

Neuronal (12): L2/3 IT, L5 ET, L5 IT, L5/6 NP, L6 CT, L6 IT, L6b, Lamp5, Pvalb, Sncg, Sst, Vip
Non-neuronal (5): Astro, Endo, Macrophage, OPC, Oligo

## Low-Sample Cell Types

Flagged (< 20 validation cells or nuclei): Endo, Macrophage, OPC

These are retained in the figure (diamond markers) but excluded from a sensitivity
cross-cell-type summary.

## Cross-Cell-Type Agreement

All 17 CTs: Pearson r = 0.8954, R2 = 0.7416
Excluding low-sample: Pearson r = 0.7972, R2 = 0.5042

## Statistical Limitations

1. Gene-cell-type observations are not statistically independent (genes are
   correlated, cell types share donors). Pooled correlations overstate effective
   sample size.
2. Cell-type-level analyses use 17 (or fewer) data points. Standard errors are wide.
3. Bootstrap CIs for cell-type shifts resample observations but do not account for
   gene-gene correlation within cell types.
4. Neuronal vs non-neuronal bootstrap resamples cell types (n=12 neuronal,
   n=5 non-neuronal), which yields wide intervals.
5. Rare cell types (Endo: 4 nuclei) have very few observations; their point
   estimates are unstable.
6. This is a distributional comparison (unpaired cells and nuclei), not a
   paired-cell analysis.

## Interpretation

SynGO annotates synaptic protein function and localization, not subcellular
RNA localization. This analysis establishes neurobiological consistency rather
than direct molecular validation of RNA localization.

## Output Files

- `syngo_filtering_counts.csv` - Gene set filtering flow
- `syngo_gene_metadata.csv` - Per-gene ortholog/subgroup metadata
- `syngo_gene_celltype_deltas.csv` - All gene x cell type deltas
- `syngo_overall_quality.csv` - Pooled imputation quality metrics
- `syngo_quality_by_celltype.csv` - Per-CT imputation quality
- `syngo_celltype_summary.csv` - CT mean shifts with bootstrap CIs
- `syngo_neuronal_vs_nonneuronal.csv` - Neuro/non-neuro comparison
- `syngo_subgroup_summary.csv` - Subgroup analyses
- `syngo_brain_validation.png/pdf` - Four-panel figure
