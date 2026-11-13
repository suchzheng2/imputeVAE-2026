---
name: Data flow audit findings
description: End-to-end variance tracking (raw→latent→decoded→cross) for pct5v2_cond model, revealing decoder inverts within/between-CT ratio
type: project
---

Data flow audit completed 2026-04-08 (job 3280539, `analysis/data_flow_audit.py`).
Model: pct5v2_cond. Results in `audit_results/pct5v2_cond/` and `notes/experiments/SUC-6_variance_retention/data_flow_audit_findings.md`.

Key findings:
- **Decoder inverts within/between-CT ratio**: raw cells 72% within-CT → decoded cells 32%. Between-CT variance is *amplified* (0.086→0.100).
- **Same-modality recon is 80% of the problem**: cell→cell retains only 20.4% of within-CT variance. Not a cross-modal issue.
- **Cross-decode adds ~35% additional loss** on top of same-mod (median ratio 0.65).
- **Nuclei recon is catastrophic**: within-CT var ratio 0.014 per gene, gene corr 0.209.
- **Latent within-CT is 4.5x smaller than between-CT** (0.21 vs 0.94 for cells). Encoder compresses within-CT relative to between-CT.
- **Inter-modality in latent only 2.1%** (vs 37% in earlier Config B diagnostic — pct5v2_cond's embedding works better).
- **Raw data highly asymmetric**: cell lib size 2.3x nuclei, 93% genes higher in cells.

**Why:** This clarifies variance collapse is primarily decoder-side (layer 2 of 3-layer problem), not encoder or cross-modal.
**How to apply:** Intervention should target decoder's use of within-CT z, not encoder or loss reweighting. Variance-preserving decoder penalty is the next logical experiment.
