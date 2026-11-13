# Delta Analysis — What does the model change during imputation?

**Date**: 2026-05-20
**Model**: log1p, learned 8d embed, dec-only (job 4514645, `run_2026-05-08-00-23-35_3604390`)
**Data**: 40,166 nuclei (v3/n3A), 10,134 genes (Config B)
**Script**: `analysis/delta_analysis.py`

## Key Findings

### 1. Model systematically adds expression (positive delta dominates)
- 9,869 / 10,134 genes (97%) have positive median delta
- Only 265 genes have negative median delta
- Mean per-gene median delta = +0.50 (log1p scale, ~65% increase in raw counts)
- Biologically expected: nuclei miss cytoplasmic RNA, imputation should mostly increase

### 2. Top increased genes are cytoplasm-enriched proteins
| Gene | Description | # CTs where top-20 | Typical delta |
|------|-------------|---------------------|---------------|
| **Tmsb4x** | Thymosin beta-4, actin-sequestering (cytoplasmic) | 11/17 | +3.3 to +4.9 |
| **Camk2n1** | CaMKII inhibitor (cytoplasmic) | 3/17 (excitatory) | +3.1 to +4.1 |
| **Calm1/3** | Calmodulin (cytoplasmic) | excitatory neurons | +3.2 to +3.5 |
| **Actb** | Beta-actin (cytoplasmic) | most CTs | +2.8 to +3.1 |
| **Selenow** | Selenoprotein W (cytoplasmic) | inhibitory neurons | +3.1 to +3.4 |
| **Plp1** | Myelin proteolipid (Oligo-specific) | Oligo only | +3.8 |
| **Mt1** | Metallothionein (Astro-specific) | Astro only | +1.8 |
| **Rgs4** | RGS protein (neuronal cytoplasm) | excitatory neurons | +3.5 |
| **Nrgn** | Neurogranin (postsynaptic, cytoplasmic) | excitatory neurons | +3.2 |

Cell-type-specific patterns are biologically correct:
- Plp1 top in Oligo (myelin gene)
- Mt1 top in Astro (metal homeostasis)
- Camk2n1/Nrgn top in excitatory neurons (synaptic signaling)

### 3. Top decreased genes are nuclear-enriched lncRNAs
| Gene | Description | # CTs where top-20 | Typical delta |
|------|-------------|---------------------|---------------|
| **Xist** | X-inactivation lncRNA (nuclear) | 8/17 | -1.1 to -1.6 |
| **Malat1** | Nuclear paraspeckle lncRNA | 5/17 | -1.1 to -1.5 |
| **Meg3** | Maternally expressed lncRNA (nuclear) | Oligo | -0.9 |
| **Snhg11** | snoRNA host gene (nuclear) | Oligo, Macrophage | -1.1 to -1.5 |
| **Gm26905, Gm20642** | Predicted genes/pseudogenes | inhibitory neurons | -1.1 to -2.5 |

Model correctly suppresses nuclear-enriched transcripts that would be over-represented in snRNA-seq.

### 4. Changes are highly consistent within cell types
- 60-80% of all genes show consistent direction across >80% of cells within each CT
- Predominantly consistent increases (red >> blue in consistency plot)
- Low-n cell types (Endo n=23, OPC n=77) show fewer consistent genes — expected
- High-n CTs (L5/6 NP, Oligo, Macrophage) show 7,000-8,000 consistent genes out of 10,134
- `frac_positive = 1.000` for nearly all top increased genes — every single cell agrees

### 5. Cross-CT heatmap shows structured patterns
- Strong horizontal bands (e.g., Tmsb4x, ribosomal proteins) = genes shifted universally
- CT-specific blocks visible (e.g., Plp1 dark only in Oligo column)
- Very few blue (decreased) genes, clustered at top of heatmap (lncRNAs)
- Neuronal subtypes (L2/3 IT through L6b) share similar delta profiles
- Non-neuronal types (Astro, Endo, OPC, Oligo, Macrophage) have distinct patterns

### 6. Caveat: mean shift vs biological correction
The overwhelmingly positive skew could partly reflect regression toward cell-type means
(which are higher for cells than nuclei on average). Future work should separate:
- "Structured biological correction" (cytoplasm-enriched genes going up)
- "Mean shift" effect (all genes drifting toward higher cell expression)
- Could test by checking whether top delta genes are enriched for known cytoplasm-localized
  transcripts (GO:0005737) vs just being high-expression genes

## Output Files
- `tables/<CT>_delta.csv` — full gene-level delta stats per cell type
- `tables/<CT>_top_genes.txt` — top 20 positive/negative genes
- `histograms/<CT>_top_genes.png` — bar charts of top shifted genes
- `global_delta_distribution.png` — overall delta distribution
- `cross_ct_delta_heatmap.png` — heatmap of top genes across all CTs
- `consistency_summary.png` — genes with consistent direction per CT
