---
name: project-cbb-rip-talk
description: "CBB research-in-progress talk (Nov 2026, 20-25 min, PhD-student audience) prepared 2026-10-28 — deck, figures, full talk script, host intro; what the final deck shows and the honesty caveats to keep"
metadata:
  node_type: memory
  type: project
  originSessionId: 39d61ad4-99ed-5d57-a40f-b21fc437acfe
  modified: 2026-10-29T20:54:20.694Z
---

Location: `notes/presentation/CBB_RIP_2026-11/` — `talk_script.md` (20 sections, ~2,550 words at 110 wpm ≈ 23 min, plus an
Addendum with the final slide-9/11 scripts, gene-count answer and host intro), `figures/ripfig_*.png` (benchmark label-free,
generalisation, L5 IT variance, latent finding, TRAP H2, L5 IT gene examples), `make_*_figure*.py`, `build_deck*.py`,
`fill_rip_deck.py`, `CBB_RIP_talk_v{1,2,3_filled}.pptx`, `prequal_pages/`. The user's own final deck is
`cbb-rip-talk-10-28-2026.pptx` (uploaded 2026-10-28, kept outside the repo).

Final deck choices (user's decisions 2026-10-28): keep the **full-dataset benchmark table** (ImputeVAE 0.93 gene corr,
96 % classif, 16 % variance vs scVI/scANVI 3 %, linear 3x) with a spoken caveat that scVI/scANVI rise to ~0.86 vs 0.90
on the split-aware label-free benchmark; tier-1 slide uses the **per-cell-type correction-magnitude figure** (r 0.86 across
17 types, neuron/glia ratio real 3.3 vs predicted 4.6, L5 ET over-predicted with only 161 reference cells, scGen/scButterfly
near zero); two new results added: Spearman vs external RiboTag/TRAP (H2) and the L5 IT variance distribution;
variance slide ends with "library size is the latent axis, sex (Xist) is the next covariate".

**Why:** the user speaks English more slowly than a native speaker, so the script is paced at ~110 wpm with short sentences
and breath marks; they asked twice for simpler wording.
**How to apply:** for any later talk text, keep sentences short and numbers few; never say "held-out, same training data"
for the full-dataset table; quote split-aware numbers from [[project-benchmark-redo]] when the slide is the label-free figure.
See [[project-libsize-covariate]] for the variance/latent story on the limitation slides.
