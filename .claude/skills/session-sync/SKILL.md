---
name: session-sync
description: Sync memory files and settings from repo to system dirs. Run this at the start of every session since ~/  is wiped between HPC sessions.
---

# Session Sync

This project runs on HPC where `~/` (including `~/.claude/`) is wiped between sessions. The repo is the source of truth.

Run these two syncs immediately:

## 1. Memory files

Copy all files from repo `.memory/` to the system memory directory:

```
cp .memory/* ~/.claude/projects/-gpfs-gibbs-pi-gerstein-sz477-cell-v-nucleus-imputeVAE/memory/
```

Verify the copy succeeded by listing both directories and confirming file counts match.

## 2. Settings

Copy the repo settings to the active project settings location:

```
cp .claude/settings.local.json ~/.claude/projects/-gpfs-gibbs-pi-gerstein-sz477-cell-v-nucleus-imputeVAE/settings.local.json
```

Verify with `diff` that the two files are identical.

## 3. Confirm

After both syncs, print a short summary:
- Number of memory files synced
- Whether settings are in sync
- Any errors encountered
