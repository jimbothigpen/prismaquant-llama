#!/usr/bin/env bash
# Wait for Stage H (post-comparison-stage-h.sh → qwopus-pipeline-stage-h.sh)
# to drain, then re-launch the InnerQ comparison sweep. The first attempt
# aborted after 2 of 45 combos because (a) llama-bench didn't recognize
# turbo3_tcq as a -ctk value (printed help and exited rc=1), and (b) the
# pipefail-grep pattern in qwopus-innerq-comparison.sh didn't tolerate the
# empty bench CSV. Both bugs fixed in this session; the script's per-combo
# caching means already-completed BF16/f16 + BF16/turbo3 will cache-skip.

set -euo pipefail

REPO=/path/to/llama-fork
QPDIR=/home/builduser/kernel-work/prismaquant-experiment/qwopus-pipeline

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] [POST-STAGE-H-COMPARISON-RERUN] starting; waiting for Stage H to drain..."

while true; do
    set +o pipefail
    sh_alive=$(pgrep -f 'scripts/(qwopus-pipeline-stage-h|post-comparison-stage-h)\.sh' 2>/dev/null | wc -l)
    runners=$(pgrep -f 'llama-perplexity|llama-bench' 2>/dev/null | wc -l)
    set -o pipefail
    if [[ "$sh_alive" -eq 0 ]] && [[ "$runners" -eq 0 ]]; then
        echo "[$(ts)] [POST-STAGE-H-COMPARISON-RERUN] Stage H drained, ai00 GPU free"
        break
    fi
    sleep 120
done

# 30s GPU buffer
sleep 30

# Re-launch comparison sweep. Per-combo caching skips BF16/f16 + BF16/turbo3
# (already in summary CSV from the first attempt). It will re-attempt
# BF16/turbo3_tcq with the fixed llama-bench parser; pipefail-safe grep
# extraction means a failure no longer aborts the loop.
#
# Append to the existing innerq-comparison-summary.csv to preserve the
# 2 valid rows.
echo "[$(ts)] [POST-STAGE-H-COMPARISON-RERUN] launching qwopus-innerq-comparison.sh"
"$REPO/scripts/qwopus-innerq-comparison.sh" 2>&1 \
    | tee "$QPDIR/logs/innerq-comparison-rerun.log"

echo "[$(ts)] [POST-STAGE-H-COMPARISON-RERUN] comparison sweep re-run complete"
