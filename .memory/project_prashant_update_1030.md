---
name: project-prashant-update-1030
description: Status/storyline/gaps note for Prashant written 2026-10-30 to restart weekly meetings — proposed next steps (freeze covariate model, pick SCORCH application, cheap ablations, manuscript skeleton) and the open decisions
metadata:
  type: project
---

`notes/prashant_update_2026-10-30.md` — written 2026-10-30 at the end of the 10/29-30 session, after the CBB RIP talk
script was finalised ([[project-cbb-rip-talk]]). Summarises everything since the September meeting (TRAP L5 remap, variance
rank, split-aware benchmark, library-size covariate, label-free decode at ceiling, Bakken/Ding matched validation) and
ranks the gaps: (1) no downstream application, (2) thin unique capability, (3) ablations on final model (no conditioning,
no JSD, N-1 type, donor-held-out), (4) sex covariate, (5) human paired truth, (6) packaging.

Proposed order: freeze the covariate model as the reported model; choose the application (SCORCH HIV-vs-control DE on
imputed profiles is the only candidate with data on disk, see [[project-scorch-scoping]]); run cheap ablations in the
background; start the manuscript skeleton from the six-figure list. Recommended storyline: biology paper with the model
as the instrument (Option B), not a methods paper.

**Why:** these are decisions to make with Prashant, not unilaterally; nothing was launched after the note.
**How to apply:** before starting SCORCH or ablation jobs, check whether the meeting happened and what was decided;
if not, the note's §5 is the agenda. See [[project-libsize-covariate]] for the model to freeze.
