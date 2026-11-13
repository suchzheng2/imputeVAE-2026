# ISMB 2026 Abstract Review Summary

**Submission**: #1546
**Title**: ImputeVAE: Translating Single-Nucleus to Single-Cell Expression Profiles via Cross-Reconstructive Variational Autoencoders
**Reviews**: 6 total

## Consensus Strengths
- Problem framing well-motivated: nuc-to-cell as structured signal, not batch noise (R2, R3, R4, R5)
- Conditioned decoder praised as good design choice (R5)
- Technically sound framework (R2)
- Timely and potentially impactful (R6)

## Major Criticisms

### 1. Weak/missing baselines (R1, R4, R5, R6)
- No scVI or other VAE-based baselines
- No batch-correction method comparisons (e.g., Harmony, scanorama)
- Linear regression baseline questionable: only 17 data points (cell-type means), unclear if per-gene or transcriptome-wide
- Gain over linear regression is trivial (1.1% classification) and not highlighted

### 2. Circular evaluation / metric concerns (R1, R5)
- Latent alignment forces cell-type distributions together, then model evaluated on cell-type classification — setup to succeed
- Cell-type-level evaluation risks model just learning averaged profiles, not per-cell translations
- Bar plots misleading: y-axis starts at 70/30 instead of 0

### 3. Single-dataset, no generalization (R4, R5)
- Only Allen Brain mouse cortex evaluated
- Unclear if learned mapping transfers across tissues, species, or experimental settings
- Real-world scenario: what if one modality is limited or missing?

### 4. Incremental novelty (R2, R6)
- Integration of known components (VAE + alignment + cross-recon), not fundamentally new paradigm
- Architectural choices (JSD over KL, learned modality embedding) not justified or ablated

### 5. Technical/batch confounding (R4)
- How is technical variation between modalities handled vs biological signal?

### 6. Figure quality (R1)
- Resolution too low to read legends
- Bar plots exaggerate differences with truncated y-axes

## Reviewer-by-Reviewer

| Reviewer | Tone | Key Point |
|----------|------|-----------|
| R1 | Negative | Evaluation is flawed (circularity + weak baseline) |
| R2 | Lukewarm | Sound but incremental novelty |
| R3 | Neutral/positive | Descriptive summary, no criticism |
| R4 | Mixed | Interesting but needs baselines + generalization + ablation |
| R5 | Mixed | Good idea, insufficiently supported by evaluation |
| R6 | Weak accept (poster) | Interesting direction, needs rigor |

## Action Items for Paper Revision

### Already in progress
- [ ] **scVI baseline** (SUC-8) — directly addresses R1, R4, R6
- [ ] **Per-cell variance retention** (SUC-6) — addresses R5 concern about learning averages
- [ ] **Cross-species transfer** (SUC-11) — addresses R4 generalization concern

### New items from reviews
- [ ] **Fix circularity concern**: evaluate on held-out cell types not seen during alignment, or use per-cell metrics that don't depend on cell-type labels
- [ ] **Ablation study**: JSD vs KL divergence, cond vs shared decoder, with/without cross-recon — runs already exist, just need to present
- [ ] **Improve linear regression baseline**: use transcriptome-wide regression (full feature set -> per gene), not just cell-type means
- [ ] **Add batch-correction baselines**: run Harmony/scanorama treating modality as batch, show it fails
- [ ] **Fix figures**: y-axis from 0, higher resolution, clearer legends
- [ ] **Discuss technical vs biological variation** explicitly in methods
