#!/usr/bin/env bash
# Qwopus prismaquant Stage H: PPL eval each of 27 GGUFs at chunks=20.
#
# ai00-only because Qwopus is qwen3_5 hybrid which crashes on ai01 HIP (#42).
# Sequential (no parallelism — single GPU).
#
# Output: per-recipe PPL log + summary CSV.

set -euo pipefail

REPO=/path/to/llama-fork
QPDIR=/home/builduser/kernel-work/prismaquant-experiment/qwopus-pipeline
QWOPUS=/mnt/cephfs/0/Container/models/Jackrong/Qwopus3.5-9B-v3.5-GGUF
WIKI=/mnt/cephfs/0/Container/models/wikitext103-calibration.txt
PERP=$REPO/build/bin/llama-perplexity
BENCH=$REPO/build/bin/llama-bench

INDIR=$QWOPUS/prismaquant
mkdir -p "$QPDIR/logs/stage-h" "$QPDIR/logs/stage-h-bench"

SUMMARY=$QPDIR/stage-h-summary.csv
echo "recipe,size_bytes,size_gb,ppl,stderr,ppl_wallclock_sec,pp512_tps,tg128_tps,bench_wallclock_sec" > "$SUMMARY"

ts() { date '+%H:%M:%S'; }

mapfile -t GGUFS < <(ls "$INDIR/Qwopus3.5-9B-v3.5-"*.gguf 2>/dev/null | sort)
echo "[$(ts)] Stage H — ${#GGUFS[@]} GGUFs to eval (bench + PPL)"

for gguf in "${GGUFS[@]}"; do
    base=$(basename "$gguf" .gguf)
    log="$QPDIR/logs/stage-h/$base.log"
    blog="$QPDIR/logs/stage-h-bench/$base.csv"
    sz=$(stat -c%s "$gguf" 2>/dev/null)
    sgb=$(echo "scale=4; $sz/1073741824" | bc)

    # ---- llama-bench: PP512 / TG128 (exclusive GPU) ----
    if [[ -f "$blog" ]] && grep -q '"512","0","0"' "$blog" 2>/dev/null; then
        echo "[$(ts)] CACHE bench $base"
        bwall="cached"
    else
        echo "[$(ts)] BENCH $base"
        bt0=$(date +%s)
        "$BENCH" \
            -m "$gguf" \
            -p 512 -n 128 \
            -t 12 -ngl 99 -fa 1 \
            --output csv \
            > "$blog" 2>&1 \
            || echo "[$(ts)] WARN $base bench exit non-zero (continuing)"
        bt1=$(date +%s)
        bwall=$((bt1-bt0))
        echo "[$(ts)]   bench wallclock ${bwall}s"
    fi
    # Pipefail-safe extraction (grep fails if no match, so OR-true at the end).
    pp512=$( (grep -E '"512","0","0"' "$blog" 2>/dev/null || true) | head -1 | awk -F, '{gsub(/"/,""); print $(NF-1)}')
    tg128=$( (grep -E '"0","128","0"' "$blog" 2>/dev/null || true) | head -1 | awk -F, '{gsub(/"/,""); print $(NF-1)}')

    # ---- llama-perplexity: chunks=20 ----
    if [[ -f "$log" ]] && grep -q "Final estimate" "$log"; then
        ppl=$(grep "Final estimate" "$log" | head -1 | grep -oE "PPL = [0-9.]+" | awk '{print $3}')
        err=$(grep "Final estimate" "$log" | head -1 | grep -oE "\+/- [0-9.]+" | awk '{print $2}')
        echo "$base,$sz,$sgb,$ppl,$err,cached,${pp512:-},${tg128:-},${bwall:-cached}" >> "$SUMMARY"
        echo "[$(ts)] CACHE PPL $base PPL=$ppl pp=${pp512:-?} tg=${tg128:-?}"
        continue
    fi
    echo "[$(ts)] PPL $base"
    t0=$(date +%s)
    "$PERP" \
        -m "$gguf" -f "$WIKI" \
        -c 4096 -b 2048 -ctk f16 -ctv f16 -fa on -ngl 99 \
        --chunks 20 --no-mmap \
        > "$log" 2>&1 \
        || echo "[$(ts)] WARN $base PPL exit non-zero (continuing)"
    t1=$(date +%s)
    # Pipefail-safe (grep fails if PPL crashed → no "Final estimate" line).
    ppl=$( (grep "Final estimate" "$log" 2>/dev/null || true) | head -1 | grep -oE "PPL = [0-9.]+" | awk '{print $3}' || true)
    err=$( (grep "Final estimate" "$log" 2>/dev/null || true) | head -1 | grep -oE "\+/- [0-9.]+" | awk '{print $2}' || true)
    echo "$base,$sz,$sgb,${ppl:-},${err:-},$((t1-t0)),${pp512:-},${tg128:-},${bwall:-}" >> "$SUMMARY"
    echo "[$(ts)]   PPL=${ppl:-?} ± ${err:-?}  $((t1-t0))s  pp=${pp512:-?} tg=${tg128:-?}"
done

echo ""
echo "[$(ts)] ===== Stage H complete ====="
column -t -s, "$SUMMARY"
