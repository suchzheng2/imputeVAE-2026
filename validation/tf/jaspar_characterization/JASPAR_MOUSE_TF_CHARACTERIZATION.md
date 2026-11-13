# JASPAR CORE 2026 Mouse TF Characterization

## Source
JASPAR CORE 2026 metadata: ultimate_metadata_table_CORE.tsv
Downloaded from: https://mencius.uio.no/JASPAR/JASPAR_metadata/2026/

## Important notes
- JASPAR is a motif database; rows are motif models, not one-to-one gene records.
- Multiple motifs can correspond to the same TF; we collapse to unique TF symbols.
- Complex motifs (e.g., Fos::Jun) are split on "::" for the primary set.
- Hyphenated symbols (e.g., Nkx2-5) are NOT split.

## Filtering pipeline
- jaspar_core_total_rows: 4572
- human_core_rows_reference_only: 1645
- mouse_core_rows_primary: 371
- mouse_unique_motif_names: 213
- mouse_unique_base_ids: 217
- mouse_singleton_motif_rows: 354
- mouse_complex_motif_rows: 17
- unique_component_symbols_after_complex_split: 211
- unique_singleton_symbols: 203
- unique_symbols_contributed_by_complex_motifs: 8
- model_gene_count: 10134
- matched_exact_model_symbol: 59
- matched_case_insensitive_model_symbol: 4
- matched_mouse_style_fallback: 0
- in_model: 63
- ribosomal_Rpl_Rps: 0
- tf_primary_non_ribosomal: 63
- tf_singleton_non_ribosomal: 61

## Gene sets
- **TF_primary**: All unique component symbols after splitting "::" complexes
- **TF_singleton_only**: Motif names without "::" (sensitivity analysis)

## Mapping methods
- exact_model_match: 59
- case_insensitive_model_match: 4
- mouse_style_fallback: 0

## Files
- jaspar_core_2026_mouse_rows.csv: All mouse CORE motif rows
- jaspar_core_2026_mouse_complex_rows.csv: Complex/dimer motif rows only
- jaspar_core_2026_mouse_tf_components.csv: All component symbols with source motif
- jaspar_core_2026_mouse_tf_symbols.csv: Primary TF symbol list
- jaspar_core_2026_mouse_tf_singleton_symbols.csv: Singleton-only TF symbol list
- jaspar_core_2026_mouse_tf_primary_model_genes.csv: Mapping to model genes
- jaspar_core_2026_mouse_tf_filtering_counts.csv: Filtering pipeline counts
- jaspar_core_2026_mouse_tf_class_summary.csv: TF class distribution
- jaspar_core_2026_mouse_tf_family_summary.csv: TF family distribution
