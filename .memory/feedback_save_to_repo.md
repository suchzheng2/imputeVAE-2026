---
name: HPC session persistence rules
description: All persistent files (memory, settings, notes) must be saved in repo and synced to system dirs on session start
type: feedback
---

This project runs on HPC where `~/` is wiped between sessions. Two sync rules:

### 1. Memory files
- **Source of truth:** repo `.memory/` dir and `MEMORY.md`
- **On session start:** copy `.memory/*` to `~/.claude/projects/.../memory/`
- **On every update:** write to BOTH system memory AND repo `.memory/`

### 2. Claude settings
- **Source of truth:** repo `.claude/settings.local.json`
- **On session start:** copy to system `~/.claude/` if needed
- **On update:** always write back to repo `.claude/`

### 3. Project notes
- Save all notes, decisions, run info to `notes/` directory in repo

**Why:** The HPC system wipes `~/` between sessions. Only files in the repo persist.

**How to apply:** On EVERY session start, immediately run BOTH syncs before doing anything else:
1. `cp .memory/* ~/.claude/projects/.../memory/`
2. `cp .claude/settings.local.json ~/.claude/projects/.../settings.local.json`
Never write only to system dirs — always write to repo first, then sync.
