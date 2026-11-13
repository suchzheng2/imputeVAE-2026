---
name: zero_nonzero_analysis
description: Zero/nonzero partition analysis (2026-04-09) — variance collapse is mostly zero-inflation artifact, decoder preserves scale of expressed genes
type: project
---

## Zero/Nonzero Partition Analysis (2026-04-09)

Motivated by Prashant's observation: reconstruction variance < input variance, is this scale compression? Pearson can't see scale differences.

**Method:** Partition all cell×gene entries into raw-zero vs raw-nonzero, compare decoder output in each group. Ran on pct5v2_cond, full training data (71K cells × 16K genes).

**Key finding:** Among nonzero entries, scale is preserved (std_ratio ~1.1) and Pearson is 0.86 (vs 0.80 for CT mean baseline). The "variance collapse" (within-CT var ratio 0.20) was driven by zero/nonzero mixing, not by compression of expressed values.

**Why:** ~60% of entries are zero. The continuous decoder (LeakyReLU) can't output exact zeros — it blends zeros and nonzeros to produce soft values near gene means. This Bernoulli structure accounts for ~73% of raw variance.

**Implication:** Reframes SUC-6 — the decoder works well on expressed genes. The "collapse" is an architectural limitation (continuous decoder vs sparse data), not a training failure. A two-headed decoder (Bernoulli + continuous) could recover missing variance.

**Why:** Prashant's question separated scale from structure, leading to this analysis.
**How to apply:** Use nonzero-only metrics alongside standard metrics. Consider two-headed decoder architecture for variance recovery. Update variance collapse narrative in any publication framing.

**Results:** `notes/experiments/zero_nonzero_analysis/`
**Script:** `analysis/zero_nonzero_analysis.py`, job 3296108
