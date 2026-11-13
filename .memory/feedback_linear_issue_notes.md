---
name: Linear issue → notes folder
description: When creating a Linear issue, always create the corresponding notes/experiments/SUC-X_name/ folder with README.md
type: feedback
---

When creating a new Linear issue, ALWAYS also create:
1. `notes/experiments/SUC-{N}_{short_name}/README.md` — with goal, status, Linear link, approach
2. Update `notes/experiments/README.md` — add the new issue to the Active table
3. Update `MEMORY.md` — add the issue to the Linear Issues list

Do all three in the SAME action as creating the issue, not as a separate step.

**Why:** The project convention is that every Linear issue has a matching notes folder. Notes are version-controlled and persist across HPC sessions, while Linear is external. The notes folder is where configs, results, and figures accumulate during the experiment. Missing this creates orphaned issues with no local tracking.

**How to apply:** Any time you call `mcp__claude_ai_Linear__save_issue` to create a new issue (not update), immediately follow with creating the notes folder, updating the experiments README, and updating MEMORY.md.
