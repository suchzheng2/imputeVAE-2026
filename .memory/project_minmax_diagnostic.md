---
name: Min/max columns added to within-CT nonzero stats
description: Added raw_min, raw_max, dec_min, dec_max to CSV for dynamic range clipping analysis; range_ratio = (dec_max-dec_min)/(raw_max-raw_min)
type: project
---

Added min/max columns to `analysis/within_ct_nonzero_stats.py` (2026-04-14).

**Why:** Std alone doesn't capture dynamic range clipping (mean-regression compressing tails). Min/max reveals decoder floor (~0.02 vs raw log1p(1)=0.69 for nonzero entries) and ceiling compression.

**How to apply:** Use `range_ratio = (dec_max - dec_min) / (raw_max - raw_min)` as a complement to std_ratio. Job 3596904 regenerates the CSV with these new columns.

Files changed: `analysis/within_ct_nonzero_stats.py`
