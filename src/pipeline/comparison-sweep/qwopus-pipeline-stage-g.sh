#!/usr/bin/env bash
# Qwopus prismaquant Stage G: apply 27 recipes via llama-quantize.
#
# Designed to be invoked on either machine. Pass:
#   --slice 1of2  -> first half (ai00)
#   --slice 2of2  -> second half (ai01)
#   --slice 1of4..4of4  -> quarters (used to spread work across 2 machines after
#                         a partial 1of2 — e.g. ai00 runs 3of4, ai01 runs 4of4
#                         to cover the missing second half in parallel).
#   --slice all   -> all 27 (default)
#
# Quantize is pure CPU + RAM (no GPU). Both ai00 and ai01 work fine.
# Output: 27 GGUFs at qwopus-pipeline/quantized/.

set -euo pipefail

REPO=/path/to/llama-fork
QPDIR=/home/builduser/kernel-work/prismaquant-experiment/qwopus-pipeline
QWOPUS=/mnt/cephfs/0/Container/models/Jackrong/Qwopus3.5-9B-v3.5-GGUF
QWOPUS_BF16=$QWOPUS/Qwopus3.5-9B-v3.5-BF16.gguf
IMATRIX=$QWOPUS/Qwopus3.5-9B-v3.5-bartowski-imatrix.gguf
QUANTIZE=$REPO/build/bin/llama-quantize

# Output GGUFs go to cephfs so both machines can access for Stage H
OUTDIR=$QWOPUS/prismaquant
mkdir -p "$OUTDIR" "$QPDIR/logs/stage-g"

SLICE="${1:-all}"

ts() { date '+%H:%M:%S'; }

# Enumerate recipes in stable order
mapfile -t RECIPES < <(ls "$QPDIR/recipes/qwopus-"*.json 2>/dev/null | sort)
if [[ ${#RECIPES[@]} -eq 0 ]]; then
    echo "[$(ts)] ERROR: no recipes in $QPDIR/recipes/" >&2
    exit 1
fi
total=${#RECIPES[@]}
half=$((total / 2))
# Quarter boundaries (integer division — for total=27: q1=6, q2=13, q3=20, end=27,
# so quarters are 6/7/7/7 recipes). 1of2 == 1of4+2of4; 2of2 == 3of4+4of4.
q1_end=$((total / 4))
q2_end=$((total / 2))
q3_end=$((3 * total / 4))

case "$SLICE" in
    1of2) RANGE=( "${RECIPES[@]:0:$half}" ) ;;
    2of2) RANGE=( "${RECIPES[@]:$half}" ) ;;
    1of4) RANGE=( "${RECIPES[@]:0:$q1_end}" ) ;;
    2of4) RANGE=( "${RECIPES[@]:$q1_end:$((q2_end - q1_end))}" ) ;;
    3of4) RANGE=( "${RECIPES[@]:$q2_end:$((q3_end - q2_end))}" ) ;;
    4of4) RANGE=( "${RECIPES[@]:$q3_end}" ) ;;
    all)  RANGE=( "${RECIPES[@]}" ) ;;
    *) echo "Unknown slice: $SLICE (valid: 1of2 / 2of2 / 1of4 / 2of4 / 3of4 / 4of4 / all)"; exit 2 ;;
esac

echo "[$(ts)] Stage G slice=$SLICE — $((${#RANGE[@]})) recipes (of $total total)"

for recipe in "${RANGE[@]}"; do
    base=$(basename "$recipe" .json)             # qwopus-4.0-333
    out_gguf="$OUTDIR/Qwopus3.5-9B-v3.5-${base#qwopus-}.gguf"  # Qwopus3.5-9B-v3.5-4.0-333.gguf
    log="$QPDIR/logs/stage-g/$base.log"
    if [[ -f "$out_gguf" ]]; then
        echo "[$(ts)] CACHE $base"
        continue
    fi
    # Allocator emits both .json and .txt sibling. The .txt is the per-tensor
    # file consumed by --tensor-type-file.
    recipe_txt="${recipe%.json}.txt"
    if [[ ! -f "$recipe_txt" ]]; then
        echo "[$(ts)] WARN $base: no .txt sibling at $recipe_txt — skipping" >&2
        continue
    fi
    echo "[$(ts)] QUANT $base -> $(basename "$out_gguf")"
    "$QUANTIZE" \
        --imatrix "$IMATRIX" \
        --tensor-type-file "$recipe_txt" \
        "$QWOPUS_BF16" \
        "$out_gguf" \
        IQ4_KS \
        > "$log" 2>&1 \
        || { echo "[$(ts)] FAIL $base (log: $log)" >&2; rm -f "$out_gguf"; continue; }
    sz=$(stat -c%s "$out_gguf" 2>/dev/null)
    echo "[$(ts)]   -> $(echo "scale=2; $sz/1073741824" | bc) GB"
done

echo ""
echo "[$(ts)] ===== Stage G slice=$SLICE complete ====="
ls -la "$OUTDIR" 2>/dev/null | head -30
