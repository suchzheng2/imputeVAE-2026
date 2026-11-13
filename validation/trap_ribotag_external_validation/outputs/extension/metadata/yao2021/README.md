# Yao 2021 MOp subclass markers (external marker list for the L5 mapping audit)

Source: Yao, Z. et al. *A transcriptomic and epigenomic cell atlas of the mouse primary motor cortex.* Nature 598, 103-110 (2021).
DOI 10.1038/s41586-021-03500-8, PMC8494649.
Supplementary Tables 1-9 (zip, 85 MB): https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-03500-8/MediaObjects/41586_2021_3500_MOESM4_ESM.zip
(linked from the article page https://www.nature.com/articles/s41586-021-03500-8 under "Supplementary information").
Downloaded 2026-09-25.

Used file: **Supplementary Table 6** (`marker.de.csv`, 4.27 M rows). Per the table guide: "For each of the 116 consensus
transcriptomic cell types, we performed differential expression (DE) analysis with respect to each of the other cell types.
The table reports the top 50 conserved DE genes in each direction for each comparison. Conserved DE genes are significant
in at least one dataset, while also having more than two-fold change in the same direction in all but one datasets."
Columns: cl1_cluster_label, cl2_cluster_label, cl1_cluster_id, cl2_cluster_id, direction ("up" = higher in cl1 than cl2,
"down" = higher in cl2; checked with Fezf2, Rorb, Tshz2, Bcl11b), genes.

Files here:
- `yao2021_table6_excitatory_pairs.csv.gz`: the 126,851 rows of Table 6 where both clusters are glutamatergic
  (30 clusters: L2/3 IT x3, L4/5 IT x2, L5 IT x4, L5 ET x4, L5/6 NP x3 + L5/6 NP CT, L6 CT x6, L6 IT x3, L6b x4; all 435 pairs present).
- `yao2021_l5_subclass_marker_fractions.tsv`: for each subclass X in {L5 IT, L5 ET, L5/6 NP} and each gene, the fraction of
  (X cluster, other cluster) pairs in which the gene is higher in the X cluster: `frac_up_vs_other_excitatory` (all 26-27
  non-X glutamatergic clusters) and `frac_up_vs_other_L5` (the 7-8 clusters of the other two L5 subclasses).
- Marker set rule used by `scripts/18_l5_mapping_audit_external_markers.py`: `frac_up_vs_other_L5 >= 0.8` and
  `frac_up_vs_other_excitatory >= 0.5`, giving 16 L5 IT, 29 L5 ET and 12 L5/6 NP genes (before intersecting with the
  10,134-gene model space). Nothing from this repo's own data enters the marker definition.
