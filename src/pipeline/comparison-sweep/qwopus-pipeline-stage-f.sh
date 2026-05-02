#!/usr/bin/env bash
# Qwopus prismaquant Stage F: 27 recipes (9 priorities × 3 budgets, ±0.25 band).
#
# Inputs (pre-Stage-F):
#   - costs.csv from Stage D
#   - bridge.json from Stage E
#   - BF16 GGUF + format-tps + pinned-tensors (in repo)
#
# Output: 27 JSON recipes under qwopus-pipeline/recipes/.

set -euo pipefail

REPO=/path/to/llama-fork
QPDIR=/home/builduser/kernel-work/prismaquant-experiment/qwopus-pipeline
QWOPUS_BF16=/mnt/cephfs/0/Container/models/Jackrong/Qwopus3.5-9B-v3.5-GGUF/Qwopus3.5-9B-v3.5-BF16.gguf

COSTS=$QPDIR/qwopus-costs.csv
BRIDGE=$QPDIR/qwopus-bridge.json
TPS=$REPO/tools/prismaquant/format-tps-gfx1150.json
PINNED=$REPO/tools/prismaquant/pinned-tensors-qwen36.json  # output=Q6_K, token_embd=Q8_0 — universal
ALLOC=$REPO/tools/prismaquant/scripts/allocator.py
FORMATS_GGML=Q4_K,Q5_K,Q6_K,Q8_0,IQ4_K,IQ4_KS,IQ4_KSS,IQ4_KT,IQ4_XS,IQ3_K,IQ3_KS,IQ2_K

mkdir -p "$QPDIR/recipes" "$QPDIR/logs"

# 9 priorities × 3 budgets = 27 recipes. Priority spec: PPL-TG-PP digits.
PRIORITIES=(333 900 090 009 522 252 225 531 153)
BUDGETS=(4.0 5.25 6.5)

ts() { date '+%H:%M:%S'; }

for prio in "${PRIORITIES[@]}"; do
    for budget in "${BUDGETS[@]}"; do
        recipe="$QPDIR/recipes/qwopus-${budget}-${prio}.json"
        log="$QPDIR/logs/allocator-${budget}-${prio}.log"
        if [[ -f "$recipe" ]]; then
            echo "[$(ts)] CACHE budget=$budget prio=$prio"
            continue
        fi
        echo "[$(ts)] ALLOC budget=$budget prio=$prio"
        python3 "$ALLOC" \
            --bridge "$BRIDGE" \
            --costs "$COSTS" \
            --budget-gb "$budget" \
            --budget-band-gb 0.25 \
            --priority "$prio" \
            --tps "$TPS" \
            --gguf "$QWOPUS_BF16" \
            --propagate-from-exemplars \
            --exemplar-layers 0,3 \
            --pinned "$PINNED" \
            --allow-types "$FORMATS_GGML" \
            --recipe-out "$recipe" \
            > "$log" 2>&1 \
            || { echo "[$(ts)] FAIL budget=$budget prio=$prio (log: $log)" >&2; exit 1; }
    done
done

echo ""
echo "[$(ts)] ===== Stage F complete: $(ls "$QPDIR/recipes/qwopus-"*.json 2>/dev/null | wc -l) recipes ====="
