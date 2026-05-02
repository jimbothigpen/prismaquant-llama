#!/usr/bin/env bash
# Sequential chunks=100 PPL eval for the 3 uniform baselines that match
# the prismaquant 14G / 19G / 21G recipes by size. Apples-to-apples comparator.
# Runs on ai00 (gfx1150) — no HSA_OVERRIDE.
set -uo pipefail

ROOT=/home/builduser/kernel-work/prismaquant-experiment
PERP=/path/to/llama-fork/build/bin/llama-perplexity
WIKI=/mnt/cephfs/0/Container/models/wikitext103-calibration.txt
LOGD=$ROOT/work/logs
GGUFDIR=/mnt/cephfs/0/Container/models/unsloth/Qwen3.6-35B-A3B-GGUF/frankenturbo

unset HSA_OVERRIDE_GFX_VERSION

# Match prismaquant tag to baseline format
declare -A BASELINES=(
    ["19G"]="IQ4_K"
    ["21G"]="TQ4_1S"
)

for tag in 14G 19G 21G; do
    fmt="${BASELINES[$tag]}"
    gguf="$GGUFDIR/Qwen3.6-35B-A3B-${fmt}-PLLow.gguf"
    log="$LOGD/ppl-baseline-${fmt}-c100.log"
    if [[ ! -f "$gguf" ]]; then
        echo "[$(date +%H:%M:%S)] SKIP $fmt: $gguf not found"
        continue
    fi
    echo "[$(date +%H:%M:%S)] starting $fmt (matches prismaquant $tag) chunks=100 -> $log"
    "$PERP" \
        -m "$gguf" \
        -f "$WIKI" \
        -c 4096 -b 2048 -ctk f16 -ctv f16 -fa on -ngl 99 \
        --chunks 100 --no-mmap \
        > "$log" 2>&1
    rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "[$(date +%H:%M:%S)] WARN: $fmt exit=$rc"
    fi
    final=$(grep "Final estimate" "$log" | tail -1)
    echo "[$(date +%H:%M:%S)] $fmt: $final"
done

echo "[$(date +%H:%M:%S)] all baselines done"
