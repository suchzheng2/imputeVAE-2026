# SynGO vs JASPAR TF Comparison

## Purpose
Test whether ImputeVAE recovers a strong neuronal whole-cell-minus-nucleus shift
specifically for SynGO synaptic-protein genes, or whether the same pattern appears
indiscriminately in JASPAR mouse transcription factors.

## Gene sets
- SynGO: Non-ribosomal genes under SynGO CC synapse (positive control for synaptic biology)
- JASPAR TF: Mouse transcription factors from JASPAR CORE 2026 (nuclear-function contrast)

## Key biological framing
- SynGO is a positive synaptic-protein validation set.
- JASPAR TFs are a nuclear-function contrast set, not a perfect negative control.
- TF mRNAs can still be cytoplasmic, but encoded TF proteins are nuclear/regulatory.
- SynGO and TFs annotate protein function/location, not RNA localization.

## Interpretation guide
A strong specificity result would be:
- SynGO: large neuronal > non-neuronal shift, recovered by ImputeVAE
- TF: smaller or absent neuronal > non-neuronal contrast

If TFs also show a strong neuronal shift, interpret cautiously: that may indicate
ImputeVAE learns a broad neuronal shift rather than a SynGO-specific pattern.

## Files
- syngo_vs_jaspar_tf_summary.csv: Side-by-side metrics
- syngo_vs_jaspar_tf_summary.png/pdf: 4-panel comparison figure
