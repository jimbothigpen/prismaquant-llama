#!/bin/bash
# Migrate legacy prismaquant-llama paths to the consolidated layout under
# ~/.prismaquant-llama/.
#
# Usage:
#   tools/migrate-paths.sh [--dry-run]
#
# What moves:
#   ~/.cache/prismaquant-wizard/binary-types/*.json
#       -> ~/.prismaquant-llama/cache/binary-types/
#
#   ~/.cache/prismaquant-llama/binary-types/*.json
#       -> ~/.prismaquant-llama/cache/binary-types/
#
#   ~/.config/prismaquant-llama/system-default-format-perf.json
#       -> ~/.prismaquant-llama/config/system-default-format-perf.json
#
# What does NOT move:
#   ~/prismaquant-builds/   (pipeline outputs — usually large, leave for the
#                            user to relocate manually if desired; the new
#                            DEFAULT_OUTPUT_ROOT is ~/.prismaquant-llama/builds
#                            but pipeline runs always pass --output explicitly
#                            on these systems)
#
# Idempotent: skips files that already exist at the destination. With
# --dry-run, only prints what would happen.

set -euo pipefail

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

NEW_ROOT="${PRISMAQUANT_LLAMA_ROOT:-$HOME/.prismaquant-llama}"
NEW_CACHE="$NEW_ROOT/cache/binary-types"
NEW_CONFIG="$NEW_ROOT/config"

# Source locations (legacy)
OLD_WIZARD_CACHE="$HOME/.cache/prismaquant-wizard/binary-types"
OLD_LLAMA_CACHE="$HOME/.cache/prismaquant-llama/binary-types"
OLD_CONFIG="$HOME/.config/prismaquant-llama"

[[ "$DRY_RUN" -eq 1 ]] && echo "[DRY-RUN] no files will be moved"
echo ""
echo "Target root: $NEW_ROOT"

run() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "  would: $*"
    else
        echo "  + $*"
        "$@"
    fi
}

migrate_dir() {
    local src="$1" dst="$2"
    [[ ! -d "$src" ]] && return 0
    local count
    count=$(find "$src" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l)
    [[ "$count" -eq 0 ]] && return 0
    echo ""
    echo "$src ($count files)"
    echo "  -> $dst"
    run mkdir -p "$dst"
    while IFS= read -r f; do
        local name dst_file
        name=$(basename "$f")
        dst_file="$dst/$name"
        if [[ -f "$dst_file" ]]; then
            echo "  skip $name (already at destination)"
        else
            run mv "$f" "$dst_file"
        fi
    done < <(find "$src" -maxdepth 1 -type f -name '*.json')
}

migrate_dir "$OLD_WIZARD_CACHE"  "$NEW_CACHE"
migrate_dir "$OLD_LLAMA_CACHE"   "$NEW_CACHE"

# Migrate single config file
if [[ -f "$OLD_CONFIG/system-default-format-perf.json" ]]; then
    echo ""
    echo "$OLD_CONFIG/system-default-format-perf.json"
    echo "  -> $NEW_CONFIG/system-default-format-perf.json"
    if [[ -f "$NEW_CONFIG/system-default-format-perf.json" ]]; then
        echo "  skip (already at destination)"
    else
        run mkdir -p "$NEW_CONFIG"
        run mv "$OLD_CONFIG/system-default-format-perf.json" "$NEW_CONFIG/system-default-format-perf.json"
    fi
fi

echo ""
if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "DRY-RUN complete. Re-run without --dry-run to apply."
else
    echo "Migration complete. Verify with:"
    echo "  ls -la $NEW_CACHE/"
    echo "  ls -la $NEW_CONFIG/"
fi
