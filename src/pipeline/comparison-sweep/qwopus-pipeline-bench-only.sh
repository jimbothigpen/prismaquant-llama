#!/usr/bin/env bash
# Qwopus prismaquant Stage H — BENCH ONLY (PP512 + TG128 per GGUF).
#
# Sister script to qwopus-pipeline-stage-h.sh, but skips the slow PPL step
# so user gets PP/TG data quickly. Run after Stage G; PPL can run later
# via the full stage-h.sh (which caches bench results from this run).
#
# ai00-only because Qwopus is qwen3_5 hybrid which crashes on ai01 HIP.
# Bench needs exclusive GPU.

set -euo pipefail

REPO=/path/to/llama-fork
QPDIR=/home/builduser/kernel-work/prismaquant-experiment/qwopus-pipeline
QWOPUS=/mnt/cephfs/0/Container/models/Jackrong/Qwopus3.5-9B-v3.5-GGUF
BENCH=$REPO/build/bin/llama-bench

INDIR=$QWOPUS/prismaquant
mkdir -p "$QPDIR/logs/stage-h-bench"

SUMMARY=$QPDIR/bench-summary.csv
echo "recipe,size_bytes,size_gb,pp512_tps,tg128_tps,wallclock_sec" > "$SUMMARY"

ts() { date '+%H:%M:%S'; }

mapfile -t GGUFS < <(ls "$INDIR/Qwopus3.5-9B-v3.5-"*.gguf 2>/dev/null | sort)
echo "[$(ts)] Bench-only — ${#GGUFS[@]} GGUFs to bench"

for gguf in "${GGUFS[@]}"; do
    base=$(basename "$gguf" .gguf)
    blog="$QPDIR/logs/stage-h-bench/$base.csv"
    sz=$(stat -c%s "$gguf" 2>/dev/null)
    sgb=$(echo "scale=4; $sz/1073741824" | bc)

    if [[ -f "$blog" ]] && grep -q '"512","0","0"' "$blog" 2>/dev/null; then
        echo "[$(ts)] CACHE $base"
        pp512=$(grep -E '"512","0","0"' "$blog" 2>/dev/null | head -1 | awk -F, '{gsub(/"/,""); print $(NF-1)}')
        tg128=$(grep -E '"0","128","0"' "$blog" 2>/dev/null | head -1 | awk -F, '{gsub(/"/,""); print $(NF-1)}')
        echo "$base,$sz,$sgb,${pp512:-},${tg128:-},cached" >> "$SUMMARY"
        continue
    fi

    echo "[$(ts)] BENCH $base"
    t0=$(date +%s)
    "$BENCH" \
        -m "$gguf" \
        -p 512 -n 128 \
        -t 12 -ngl 99 -fa 1 \
        --output csv \
        > "$blog" 2>&1 \
        || echo "[$(ts)] WARN $base bench non-zero exit (continuing)"
    t1=$(date +%s)

    pp512=$(grep -E '"512","0","0"' "$blog" 2>/dev/null | head -1 | awk -F, '{gsub(/"/,""); print $(NF-1)}')
    tg128=$(grep -E '"0","128","0"' "$blog" 2>/dev/null | head -1 | awk -F, '{gsub(/"/,""); print $(NF-1)}')
    echo "$base,$sz,$sgb,${pp512:-},${tg128:-},$((t1-t0))" >> "$SUMMARY"
    echo "[$(ts)]   pp=${pp512:-?} tg=${tg128:-?}  $((t1-t0))s"
done

echo ""
echo "[$(ts)] ===== Bench-only complete ====="
column -t -s, "$SUMMARY"
