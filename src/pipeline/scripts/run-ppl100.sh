#!/usr/bin/env bash
# Sequential chunks=100 PPL eval for the 3 prismaquant GGUFs on ai00.
# No HSA_OVERRIDE — ai00 is gfx1150 native.
set -uo pipefail

ROOT=/home/builduser/kernel-work/prismaquant-experiment
PERP=/path/to/llama-fork/build/bin/llama-perplexity
WIKI=/mnt/cephfs/0/Container/models/wikitext103-calibration.txt
LOGD=$ROOT/work/logs

unset HSA_OVERRIDE_GFX_VERSION

for tag in 14G 19G 21G; do
    gguf="/mnt/cephfs/0/Container/models/unsloth/Qwen3.6-35B-A3B-GGUF/frankenturbo/Qwen3.6-35B-A3B-prismaquant-$tag.gguf"
    log="$LOGD/ppl-$tag-c100.log"
    echo "[$(date +%H:%M:%S)] starting $tag chunks=100 -> $log"
    "$PERP" \
        -m "$gguf" \
        -f "$WIKI" \
        -c 4096 -b 2048 -ctk f16 -ctv f16 -fa on -ngl 99 \
        --chunks 100 --no-mmap \
        > "$log" 2>&1
    rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "[$(date +%H:%M:%S)] WARN: $tag exit=$rc"
    fi
    final=$(grep "Final estimate" "$log" | tail -1)
    echo "[$(date +%H:%M:%S)] $tag: $final"
done

echo "[$(date +%H:%M:%S)] all done"
