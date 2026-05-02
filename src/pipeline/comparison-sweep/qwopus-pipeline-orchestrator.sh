#!/usr/bin/env bash
# Top-level Qwopus prismaquant pipeline orchestrator.
#
# Sequence (all stages idempotent):
#   D — quantize-cost (assumed already running; we just poll for completion)
#   E — bridge probe.pkl → bridge.json
#   F — allocator × 27 recipes
#   G — quantize × 27 GGUFs, SPLIT between ai00 (1of2) and ai01 (2of2) in parallel
#   ...wait for both halves to finish...
#   ROT — rotation-knob 4×3 eval matrix on ai00 (gates Stage H)
#   H — PPL × 27 on ai00 (chunks=20, sequential, ~5h)
#
# Run from ai00. ai01 must be reachable on port 2229 with the same git revision
# of /path/to/llama-fork checked out.

set -euo pipefail

REPO=/path/to/llama-fork
QPDIR=/home/builduser/kernel-work/prismaquant-experiment/qwopus-pipeline
COSTS=$QPDIR/qwopus-costs.csv
BRIDGE=$QPDIR/qwopus-bridge.json
GGUF_DIR=/mnt/cephfs/0/Container/models/Jackrong/Qwopus3.5-9B-v3.5-GGUF/prismaquant
AI01_HOST=ai01
AI01_PORT=2229

mkdir -p "$QPDIR/logs"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*"; }

# --- D: wait for completion ---
say "===== Stage D: wait for quantize-cost ====="
while pgrep -f "llama-quantize-cost.*Qwopus" > /dev/null 2>&1; do
    n=$(wc -l < "$COSTS" 2>/dev/null || echo 0)
    say "Stage D running — costs.csv has $n rows"
    sleep 60
done
n=$(wc -l < "$COSTS" 2>/dev/null || echo 0)
if [[ ! -f "$COSTS" ]] || [[ "$n" -lt 50 ]]; then
    say "ERROR: Stage D did not produce a usable costs.csv (rows=$n)"; exit 1
fi
say "Stage D done — costs.csv has $n rows"

# --- E: bridge ---
if [[ -f "$BRIDGE" ]]; then
    say "Stage E: bridge already exists (skip)"
else
    say "===== Stage E: bridge ====="
    python3 "$REPO/tools/prismaquant/scripts/bridge_probe_to_gguf.py" \
        --probe /mnt/cephfs/0/Container/models/Jackrong/Qwopus3.5-9B-v3.5-GGUF/probe-work/qwopus-probe-16x512.pkl \
        --output "$BRIDGE" \
        --aggregate sum \
        --unmapped-out "$QPDIR/qwopus-bridge-unmapped.json" \
        --verify-gguf /mnt/cephfs/0/Container/models/Jackrong/Qwopus3.5-9B-v3.5-GGUF/Qwopus3.5-9B-v3.5-BF16.gguf \
        > "$QPDIR/logs/bridge.log" 2>&1
    say "Stage E done"
fi

# --- F: allocator ---
say "===== Stage F: allocator (27 recipes) ====="
"$REPO/scripts/qwopus-pipeline-stage-f.sh" 2>&1 | tee -a "$QPDIR/logs/stage-f.log"

# --- G split: ai00 (1of2) + ai01 (2of2) in parallel ---
say "===== Stage G split: ai00 1of2 + ai01 2of2 ====="
say "  launching ai01 2of2 via ssh..."
ssh -p $AI01_PORT $AI01_HOST "mkdir -p $QPDIR/logs/stage-g-ai01; nohup $REPO/scripts/qwopus-pipeline-stage-g.sh 2of2 > $QPDIR/logs/stage-g-ai01.log 2>&1 &" || {
    say "ERROR: failed to launch Stage G on ai01"; exit 1
}
ssh -p $AI01_PORT $AI01_HOST "pgrep -f 'qwopus-pipeline-stage-g.sh 2of2'" > /dev/null && say "  ai01 process started"
say "  running ai00 1of2 in foreground..."
"$REPO/scripts/qwopus-pipeline-stage-g.sh" 1of2 2>&1 | tee -a "$QPDIR/logs/stage-g-ai00.log"
say "  ai00 1of2 done — waiting for ai01 2of2..."
while ssh -p $AI01_PORT $AI01_HOST "pgrep -f 'qwopus-pipeline-stage-g.sh 2of2' > /dev/null 2>&1"; do
    say "  ai01 still running"
    sleep 60
done
say "  ai01 2of2 done"

# Verify all 27 GGUFs present
n_gguf=$(ls "$GGUF_DIR"/Qwopus3.5-9B-v3.5-*.gguf 2>/dev/null | wc -l)
say "  GGUFs produced: $n_gguf / 27"
if [[ "$n_gguf" -lt 25 ]]; then
    say "WARN: fewer GGUFs than expected — proceeding anyway"
fi

# --- ROT: rotation-knob 4×3 eval matrix on ai00 ---
say "===== Rotation-knob eval matrix (4×3, ai00 only) ====="
"$REPO/scripts/rotation-knob-eval.sh" 2>&1 | tee -a "$QPDIR/logs/rotation-eval.log"

# --- H: PPL × 27 on ai00 ---
say "===== Stage H: PPL × 27 on ai00 ====="
"$REPO/scripts/qwopus-pipeline-stage-h.sh" 2>&1 | tee -a "$QPDIR/logs/stage-h.log"

say "===== Pipeline complete ====="
say "  costs:   $COSTS"
say "  bridge:  $BRIDGE"
say "  recipes: $QPDIR/recipes/"
say "  GGUFs:   $GGUF_DIR/"
say "  rotation eval: /home/builduser/kernel-work/rotation-knob-eval/results.csv"
say "  PPL:     $QPDIR/stage-h-summary.csv"
