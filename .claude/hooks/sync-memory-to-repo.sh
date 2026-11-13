#!/bin/bash
# Sync memory files from ~/.claude/projects/.../memory/ to repo .memory/
# Triggered by PostToolUse on Write|Edit

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

# Only process files in the system memory directory
if [[ "$FILE_PATH" == *"/.claude/projects/"*"/memory/"* ]]; then
  FILENAME=$(basename "$FILE_PATH")
  BACKUP_DIR="/gpfs/gibbs/pi/gerstein/sz477/cell_v_nucleus/imputeVAE/.memory"

  mkdir -p "$BACKUP_DIR"
  cp "$FILE_PATH" "$BACKUP_DIR/$FILENAME"
fi

exit 0
