#!/usr/bin/env bash
# Wait for the active prismaquant evaluation chain (rotation-knob-eval +
# qwopus-pipeline-stage-h) to drain, then:
#   1) Run a quick Phase 6 InnerQ smoke test (Llama-3.2-3B Q6_K turbo3_innerq
#      chunks=4 with TURBO_INNERQ_TOKENS=100). Captures the [InnerQ]
#      calibration log lines + final PPL. Bails out (no comparison sweep)
#      if the result is NaN/non-finite or wildly off the identity baseline.
#   2) Launch the InnerQ comparison sweep against the 9 standard
#      Qwopus3.5-9B-v3.5 mainline quants.
#
# Order matters: we don't want to burn ~10h on a 45-combo comparison sweep
# (which includes turbo3_innerq and turbo4_innerq columns) if Phase 6's
# calibration is producing garbage.
#
# Run detached (setsid nohup) so it survives session/Claude exits.

set -euo pipefail

REPO=/path/to/llama-fork
QPDIR=/home/builduser/kernel-work/prismaquant-experiment/qwopus-pipeline
SMOKE_LOG="$QPDIR/logs/phase6-innerq-smoke.log"
IDENTITY_BASELINE_PPL=15.7178   # Phase 6.5 alias / identity-scales reference

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] [WAIT-THEN-COMPARISON] starting; waiting for chain to drain..."

# Poll every 2 minutes until both rotation-eval and stage-h scripts are gone.
# pgrep returns 1 when no matches — under set -e + pipefail that aborts the
# script. Disable pipefail / use OR-true to keep polling robust.
while true; do
    set +o pipefail
    chain_running=$(pgrep -f 'scripts/(rotation-knob-eval|qwopus-pipeline-stage-h)\.sh' 2>/dev/null | wc -l)
    runners=$(pgrep -f 'llama-perplexity|llama-bench' 2>/dev/null | wc -l)
    set -o pipefail
    if [[ "$chain_running" -eq 0 ]] && [[ "$runners" -eq 0 ]]; then
        echo "[$(ts)] [WAIT-THEN-COMPARISON] chain drained, ai00 GPU free (chain=$chain_running runners=$runners)"
        break
    fi
    sleep 120
done

# 30s GPU warmup-cooldown buffer
sleep 30

# ---- Step 1: Phase 6 InnerQ calibration smoke test ----
echo "[$(ts)] [WAIT-THEN-COMPARISON] STEP 1: Phase 6 InnerQ smoke test (chunks=4, TURBO_INNERQ_TOKENS=100)"
TURBO_INNERQ_TOKENS=100 "$REPO/build/bin/llama-perplexity" \
    -m /mnt/cephfs/0/Container/models/unsloth/Llama-3.2-3B-Instruct-GGUF/frankenturbo/Llama-3.2-3B-Instruct-Q6_K-PLLow.gguf \
    -f /opt/llama/models/unsloth/Llama-3.2-3B-Instruct-GGUF/wikitext103-calibration.txt \
    --no-mmap -ngl 99 -fa on -ctk turbo3_innerq -ctv turbo3_innerq --chunks 4 -c 512 \
    > "$SMOKE_LOG" 2>&1 \
    || echo "[$(ts)] [WAIT-THEN-COMPARISON] WARN: smoke perplexity exit non-zero (continuing diagnosis)"

# Capture the InnerQ event log + the final PPL.
echo ""
echo "[$(ts)] [WAIT-THEN-COMPARISON] === Phase 6 smoke output ==="
grep -E '\[InnerQ\]|Final estimate' "$SMOKE_LOG" || true
echo ""

# Validate: PPL must be finite and within ±5.0 of the identity baseline. The
# identity baseline is 15.7178; calibrated InnerQ should land somewhere near
# it (slightly better if InnerQ helps, similar if max_ratio < 1.2 triggers
# the auto-disable). NaN, infinite, or wildly different values mean Phase 6
# is broken — bail out before wasting ~10h on the comparison sweep.
smoke_ppl=$(grep "Final estimate" "$SMOKE_LOG" | head -1 | grep -oE "PPL = [0-9.]+" | awk '{print $3}')
if [[ -z "$smoke_ppl" ]]; then
    echo "[$(ts)] [WAIT-THEN-COMPARISON] FATAL: smoke test produced no PPL (NaN or crash). Skipping comparison sweep."
    echo "[$(ts)] [WAIT-THEN-COMPARISON] Smoke log: $SMOKE_LOG"
    exit 1
fi

# Bash-arithmetic float comparison via bc.
delta=$(echo "$smoke_ppl - $IDENTITY_BASELINE_PPL" | bc -l)
abs_delta=$(echo "if ($delta < 0) -($delta) else $delta" | bc -l)
out_of_range=$(echo "$abs_delta > 5.0" | bc -l)

if [[ "$out_of_range" -eq 1 ]]; then
    echo "[$(ts)] [WAIT-THEN-COMPARISON] FATAL: smoke PPL=$smoke_ppl is > 5.0 from identity baseline $IDENTITY_BASELINE_PPL. Skipping comparison sweep."
    echo "[$(ts)] [WAIT-THEN-COMPARISON] Smoke log: $SMOKE_LOG"
    exit 1
fi

echo "[$(ts)] [WAIT-THEN-COMPARISON] Smoke PPL=$smoke_ppl (Δ=$delta vs identity $IDENTITY_BASELINE_PPL) — within tolerance, proceeding"

# ---- Step 2: rotation-knob eval (quantized KV) ----
# The original rotation-knob-eval.sh hard-coded f16 KV, which made the
# LLAMA_ATTN_ROT_*_OVERRIDE env vars no-ops. This focused 8-run follow-up
# (1 model × 2 KV types × 4 configs ≈ 25 min) actually exercises the
# rotation override path with quantized KV.
echo ""
echo "[$(ts)] [WAIT-THEN-COMPARISON] STEP 2: rotation-knob-eval-quantized (~25 min)"
"$REPO/scripts/rotation-knob-eval-quantized.sh" 2>&1 \
    | tee "$QPDIR/logs/rotation-eval-quantized.log" \
    || echo "[$(ts)] [WAIT-THEN-COMPARISON] WARN: rotation-quantized exit non-zero (continuing to comparison sweep)"

# ---- Step 3: InnerQ comparison sweep ----
echo ""
echo "[$(ts)] [WAIT-THEN-COMPARISON] STEP 3: launching qwopus-innerq-comparison.sh"
"$REPO/scripts/qwopus-innerq-comparison.sh" 2>&1 \
    | tee "$QPDIR/logs/innerq-comparison.log"

echo "[$(ts)] [WAIT-THEN-COMPARISON] comparison sweep complete"
