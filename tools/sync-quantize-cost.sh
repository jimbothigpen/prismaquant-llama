#!/bin/bash
# Sync vendored quantize-cost source from a frankenturbo2 checkout.
#
# Canonical source lives in jimbothigpen/frankenturbo2 at tools/quantize-cost/.
# This package vendors a copy at src/pipeline/cpp/quantize-cost/ so users without
# frankenturbo2 on disk can still build llama-quantize-cost into a vanilla
# llama.cpp tree. Drift between the two copies is the cost of vendoring; this
# script + the recorded source SHA in VENDORED-FROM.txt keep that drift visible.
#
# Usage:
#   tools/sync-quantize-cost.sh                    # uses $FRANKENTURBO2_ROOT or /path/to/llama-fork
#   tools/sync-quantize-cost.sh /path/to/frankenturbo2
#
# After running, commit BOTH the updated source files AND VENDORED-FROM.txt.

set -euo pipefail

ft_root="${1:-${FRANKENTURBO2_ROOT:-/path/to/llama-fork}}"
if [[ ! -d "$ft_root" ]]; then
    echo "ERROR: frankenturbo2 root not found at $ft_root" >&2
    echo "       Pass the path as arg 1, or set FRANKENTURBO2_ROOT." >&2
    exit 1
fi
ft_root="$(cd "$ft_root" && pwd)"
if [[ ! -d "$ft_root/.git" ]]; then
    echo "ERROR: $ft_root is not a git repo" >&2
    exit 1
fi

src_dir="$ft_root/tools/quantize-cost"
if [[ ! -f "$src_dir/quantize-cost.cpp" ]]; then
    echo "ERROR: $src_dir/quantize-cost.cpp missing" >&2
    exit 1
fi

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
dst_dir="$repo_root/src/pipeline/cpp/quantize-cost"
mkdir -p "$dst_dir"

rsync -av "$src_dir/quantize-cost.cpp" "$src_dir/CMakeLists.txt" "$dst_dir/" >/dev/null

sha="$(git -C "$ft_root" rev-parse HEAD)"
sha_short="$(git -C "$ft_root" rev-parse --short HEAD)"
dirty=""
if ! git -C "$ft_root" diff --quiet -- tools/quantize-cost/ 2>/dev/null; then
    dirty=" (DIRTY: tools/quantize-cost/ has uncommitted changes — re-sync after commit)"
fi
ts="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

cat > "$dst_dir/VENDORED-FROM.txt" <<EOF
# This directory mirrors tools/quantize-cost/ from jimbothigpen/frankenturbo2.
# Update: run tools/sync-quantize-cost.sh from the prismaquant-llama root,
# then commit the modified C++ source + CMakeLists.txt + this file together.

source_repo:    https://github.com/jimbothigpen/frankenturbo2
source_path:    tools/quantize-cost/
source_sha:     $sha$dirty
source_short:   $sha_short
synced_at:      $ts
synced_files:   quantize-cost.cpp, CMakeLists.txt
EOF

echo
echo "  ✓ synced $dst_dir/{quantize-cost.cpp,CMakeLists.txt}"
echo "  ✓ recorded source_sha = $sha_short in VENDORED-FROM.txt"
if [[ -n "$dirty" ]]; then
    echo "  ⚠ frankenturbo2 working tree was dirty — commit changes upstream and re-sync"
fi
