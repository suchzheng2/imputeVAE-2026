# Library Size Analysis Plan (2026-04-15)

## Motivation
From meeting with Prashant: how much of the within-CT variance is driven by
library size differences vs. gene-specific regulation? Does the decoder
preserve library-size-related variation at all?

Fig 2 shows the decoder predicts nearly identical values for zero and nonzero
entries — it regresses to the CT mean. Library size is a known major driver of
within-CT variance in log1p data. Need to disentangle.

## Key Questions
1. Does the decoder preserve per-cell library size? (correlation raw vs decoded lib)
2. How much within-CT variance is library-size-explained vs residual?
3. After removing library size (proportion normalization), does the std collapse
   look different?
4. Does each latent dimension correlate with library size?

## Analyses
1. **Library size preservation**: scatter of raw vs decoded library size per cell
2. **Variance decomposition**: per-gene R² of expression ~ log(libsize), raw vs decoded
3. **Proportion-normalized comparison**: redo std_ratio on counts/libsize instead of log1p
4. **Per-gene libsize correlation**: does decoded expression correlate with raw library size?
5. **Latent-libsize correlation**: which latent dims encode library size?

## Implementation
- Script: `analysis/library_size_analysis.py`
- Library size from log1p: `np.expm1(X_log1p).sum(axis=1)` — exact, no reload needed
- R² shortcut: R² = Pearson(gene, log_lib)² — vectorized across genes per CT
- Output: `notes/experiments/SUC-6_variance_retention/library_size/`
- Job: `jobs/eval/run_library_size_analysis.sh` (no GPU, 150G, 2h)
