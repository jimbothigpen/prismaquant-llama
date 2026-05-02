#!/usr/bin/env bash
# Wait for wait-then-comparison.sh to drain, then launch Stage H to collect
# the missing 27-GGUF prismaquant PPL data. Stage H originally crashed at
# startup (pipefail bug fixed in commit 86d936c8b7); this is the queued
# re-launch. Stage H needs exclusive ai00 GPU, so it has to follow the
# comparison sweep rather than parallel.

set -euo pipefail

REPO=/path/to/llama-fork
QPDIR=/home/builduser/kernel-work/prismaquant-experiment/qwopus-pipeline

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] [POST-COMPARISON-STAGE-H] starting; waiting for wait-then-comparison.sh to drain..."

# Poll every 2 minutes. Wrapper drains when it finishes its STEP 3
# (qwopus-innerq-comparison.sh) — at which point pgrep finds neither the
# wrapper, the comparison script, nor a running llama-perplexity/bench.
while true; do
    set +o pipefail
    wrapper_alive=$(pgrep -f 'scripts/wait-then-comparison\.sh' 2>/dev/null | wc -l)
    inner_alive=$(pgrep -f 'scripts/(rotation-knob-eval-quantized|qwopus-innerq-comparison)\.sh' 2>/dev/null | wc -l)
    runners=$(pgrep -f 'llama-perplexity|llama-bench' 2>/dev/null | wc -l)
    set -o pipefail
    if [[ "$wrapper_alive" -eq 0 ]] && [[ "$inner_alive" -eq 0 ]] && [[ "$runners" -eq 0 ]]; then
        echo "[$(ts)] [POST-COMPARISON-STAGE-H] wrapper + sweeps drained, ai00 GPU free"
        break
    fi
    sleep 120
done

# 30s GPU buffer
sleep 30

echo "[$(ts)] [POST-COMPARISON-STAGE-H] launching qwopus-pipeline-stage-h.sh"
"$REPO/scripts/qwopus-pipeline-stage-h.sh" 2>&1 \
    | tee "$QPDIR/logs/stage-h.log"

echo "[$(ts)] [POST-COMPARISON-STAGE-H] Stage H complete"
