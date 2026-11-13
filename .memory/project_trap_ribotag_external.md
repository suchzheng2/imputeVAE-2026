---
name: project-trap-ribotag-external
description: External validation with compartment fractionation + TRAP/RiboTag GEO datasets (2026-09-18, DONE); results in validation/trap_ribotag_external_validation/
metadata:
  type: project
---

# TRAP/RiboTag + compartment external validation (DONE 2026-09-18)

- Location: `validation/trap_ribotag_external_validation/` (TASK.md plan with corrections in Section 0; README.md results; outputs/report/)
- Notes: `notes/experiments/trap_ribotag_external/README.md`
- SLURM: 10905766 pseudobulk, 10905946 limma-voom, 10905990 GSE174303, 10905994 final stats, 10906009 secondary
- H1 GSE210071 (cultured cortex, ALL triptolide-treated) cytosol-nucleus rho 0.38, AUROC 0.80; whole-nucleus rho 0.41, AUROC 0.82; GT delta 0.37/0.40 (model at biological ceiling)
- H1 GSE121199 (NAc MSN, no matched CT): D2 whole-nuclear rho 0.50; D1 fails (4/6 nuclear libraries < 5M reads; GT identical)
- H2 Pvalb/Sst RiboTag (GSE263082): r nucleus 0.43/0.45 -> imputed 0.73/0.73, real-cell ceiling 0.74/0.73; GSE77243 0.50 -> 0.78 both
- H3 RiboTag abundance vs pred delta rho 0.74-0.80, partial given nucleus expr 0.69-0.74
- IP/input, IP/UB enrichment contrasts: -0.27 to +0.24 (property mismatch, also for GT)
- Caveat: global H1 rank concordance similar for all 7 methods (0.37-0.41); discrimination only via AUROC and CT-resolved H2/H3
- Controls: Meg3 (+0.84 model) and Kcnq1ot1 are model-discordant (nuclear externally); Neat1 and Pvalb absent from model gene set; Malat1 absent from GSE210071 table
- Tooling that worked: R via `module purge; module load R-bundle-Bioconductor/3.18-foss-2022b-R-4.3.2` inside SLURM (edgeR/limma OK); downloads from login node

**Why:** Committee asked for validation with independent nuclear/cytoplasmic and ribosome-associated data.
**How to apply:** Cite as non-circular compartment + RiboTag evidence; do not claim sub-cytoplasmic or human generalization. Related: [[validation-syngo-tf-l5et]], [[human-m1-transfer]].

## Extension with newer RNA-seq datasets (2026-09-18, job 10910535)
- Report: `validation/trap_ribotag_external_validation/outputs/report/extension_newer_datasets.md`; figures `outputs/extension/figures/figE1-E5`
- Added: GSE133291 Furlanis (CamK2a, Scnn1a, PV, SST, VIP), GSE172484 (motor cortex L5b x2), GSE202172 (L5a), GSE141337, GSE142028, GSE141874, GSE139684, GSE99791 (astro), GSE211162 (oligo); GSE77243 Vip
- H2 holds in all 19 populations: neurons nucleus 0.26-0.50 -> imputed 0.54-0.78 (at real-cell ceiling); glia 0.23-0.39 -> 0.40-0.56; H3 rho 0.42-0.80 (2008 microarray 0.24-0.40: platform effect)
- Specificity: matched CT ranks first in 13/17 imputed vs 5/17 raw nuclei vs 11/17 real cells (2026-09-25: Scnn1a L4 and Rbp4 pan-L5 excluded; rule = skip populations with no single matching subclass, never pool)
- **Negative: pseudobulk H2 does NOT separate methods** (ImputeVAE vs scVI/scANVI/scGen/scButterfly margins 0.001-0.03; linear regression wins on glia and Vip). These tests validate the nucleus->cell shift, not ImputeVAE specifically. Method discrimination must come from per-cell structure, where [[project-variance-zero-nonzero-fullmodel]] shows the model is currently weak.

- 2026-09-25: marker cross-check redone with EXTERNAL markers (Yao 2021 MOp Supp. Table 6, `validation/trap_ribotag_external_validation/outputs/extension/metadata/yao2021/`, script 18) because real-cell-derived markers share the H2 real-cell ceiling (circular). ET-vs-IT sign matches literature for all 6 L5 populations; mapping unchanged. Present the Yao check, not the script-17 one.
