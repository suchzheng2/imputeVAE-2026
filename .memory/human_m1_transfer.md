# Human M1 Zero-Shot Transfer (Task Group 2, 2026-08-16/17)

## Overview
- Zero-shot cross-species transfer: frozen mouse MOp cVAE applied to Allen Human M1 10x v3 snRNA-seq
- 76,533 nuclei, 2 donors, 127 clusters → 8 broad classes, 20 mouse-like subclasses
- Model: `run/run_2026-07-18-19-40-51_570984` (VAE_JSD_Cond, scalar, enc+dec, log1p)
- SLURM jobs: TG1 prep (9231013), TG2 inference (9231508, 4 min GPU), TG2 rerun (9243944)

## Location
- `validation/human_m1_transfer/` — all scripts, data, results, figures, reports
- Scripts: `scripts/01-09` (ortholog map, metadata, matrix prep, inference, summaries, comparison, figures, acceptance)
- Report: `reports/TASK_GROUP_2_ZERO_SHOT_INFERENCE_REPORT.md`
- Acceptance: `reports/task_group_2_acceptance_checks.txt` — 34/34 passed

## Gene Mapping
- MGI HOM_MouseHumanSequence.rpt → one-to-one orthologs
- 8,215 / 10,134 model genes mapped (81.1%), 1,919 unmapped (Gm*/Rik/locus)
- Known MGI gaps: Gapdh, Neat1 (grouped with pseudogenes, not rescued)

## Key Results
- **Predicted shifts**: neuronal > non-neuronal (excitatory/inhibitory 0.456 > astrocyte 0.148)
- **SynGO** (1,236 non-ribosomal synaptic genes): neuronal=0.775, non-neuronal=0.533, diff=0.243
- **TF** (60 genes): neuronal=0.261, non-neuronal=0.113, diff=0.148
- **Mouse-human similarity**: median Pearson r=0.715 across 17 cell classes, range [0.082, 0.803]
- **Donor consistency**: median r=0.986
- **QC**: 1.27% negative values (min=-0.082, LeakyReLU artifact, not clamped)

## Bug Fixes Applied (rerun job 9243944)
1. Modality direction text in report (id vs scalar clarified)
2. SynGO gene set: removed ribosomal contamination (1,301→1,236 genes)
3. All-mapped mouse similarity: loaded all 10,134 gene deltas (not just SynGO+TF)
4. Figure formula: gene count display fixed
5. README status updated

## Caveats
- No human whole-cell ground truth — all results hypothesis-generating
- Only 2 donors; L5 ET interpret cautiously (mouse calibration issue)
- 1,919 unmapped genes zero-filled and ignored

## Output Files
- `inference/human_m1_imputed_cell_log1p.npy` — 3.1 GB dense (76,533 × 10,134)
- `results/` — 16+ CSVs (pseudobulk, SynGO/TF, similarity, donor consistency)
- `figures/` — 4 figure sets (QC, delta summary, shift similarity, donor consistency)
