#!/usr/bin/env bash
# Master pipeline: from probe.pkl + sweep done -> 3 prismaquant GGUFs + PPL.
#
# Stages (each is idempotent — checks for output file before running):
#   A. Wait for probe.pkl (ai01 CPU bartowski probe)
#   B. Wait for sweep summary marker
#   C. Generate Qwen3.6 imatrix on ai00 GPU using bartowski calibration
#   D. Run quantize-cost on BF16 GGUF + imatrix -> costs.csv
#   E. Bridge probe.pkl -> bridge.json (HF names -> GGUF names + Fisher)
#   F. Run allocator at budgets 21.15 / 18.83 / 14.21 GB -> 3 recipes
#   G. Apply each recipe via llama-quantize -> 3 prismaquant GGUFs
#   H. PPL eval each at chunks=100 vs same-size PLLow baselines
#   I. Write a summary report
#
# Launch detached:
#   setsid nohup ./run-pipeline.sh < /dev/null \
#     >>/home/builduser/kernel-work/prismaquant-experiment/work/logs/pipeline.log 2>&1 &

set -euo pipefail

ROOT=/home/builduser/kernel-work/prismaquant-experiment
SWEEP_MARKERS=/home/builduser/kernel-work/full-sweep-2026-04-30/markers
LOG=$ROOT/pipeline.log
SUMMARY=$ROOT/pipeline-summary.md
STATUS=$ROOT/pipeline.status

QWEN_HF=/mnt/cephfs/0/Container/models/Qwen/Qwen3.6-35B-A3B
QWEN_BF16_SPLIT=/mnt/data/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-BF16-00001-of-00002.gguf
QWEN_BF16=/mnt/data/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-BF16-merged.gguf
GGUF_SPLIT_BIN=/path/to/llama-fork/build/bin/llama-gguf-split
QWEN_OUTDIR=/mnt/cephfs/0/Container/models/unsloth/Qwen3.6-35B-A3B-GGUF/frankenturbo
PROBE_PKL=$ROOT/qwen36-probe-bartowski.pkl
BARTOWSKI=/mnt/cephfs/0/Container/models/bartowski-calibration-v3.txt
WIKITEXT=/mnt/cephfs/0/Container/models/wikitext103-calibration.txt
IMATRIX=$QWEN_OUTDIR/Qwen3.6-35B-A3B-bartowski-imatrix.gguf
COSTS=$ROOT/qwen36-costs.csv
BRIDGE=$ROOT/qwen36-bridge.json
QUANTIZE_BIN=/path/to/llama-fork/build/bin/llama-quantize
# All GGUF outputs go to cephfs (QWEN_OUTDIR above) so both ai00 and ai01
# can read them; /home/builduser/* is local-only on each machine.
QUANTIZE_COST_BIN=/path/to/llama-fork/build/bin/llama-quantize-cost
IMATRIX_BIN=/path/to/llama-fork/build/bin/llama-imatrix
PERPLEXITY_BIN=/path/to/llama-fork/build/bin/llama-perplexity

# Format catalog (locked in project memory)
FORMATS=Q4_K_M,Q5_K_M,Q6_K,Q8_0,IQ4_K,IQ4_KS,IQ4_KSS,IQ4_KT,IQ4_XS,IQ3_K,IQ3_KS,IQ2_K
# These are the actual ggml types for our locked candidates (Q4_K_M -> Q4_K kernel,
# but the FTYPE set will use Q4_K so we list lowercase ggml type names below).
FORMATS_GGML=Q4_K,Q5_K,Q6_K,Q8_0,IQ4_K,IQ4_KS,IQ4_KSS,IQ4_KT,IQ4_XS,IQ3_K,IQ3_KS,IQ2_K

# Three budgets (locked decisions)
declare -A BUDGETS=(
    ["21G"]="21.15"
    ["19G"]="18.83"
    ["14G"]="14.21"
)

PINNED=$ROOT/pinned-tensors.json

ts() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*" | tee -a "$LOG"; }
set_status() { echo "[$(ts)] $*" > "$STATUS"; say "$*"; }

# Hostname-conditional HSA override.
# ai00 is gfx1150 native — must NOT set the override (causes silent perplexity death).
# ai01 is gfx1102/gfx1103 → emulate as gfx1100 via this override.
HOST=$(hostname)
if [[ "$HOST" == "ai00" ]]; then
    HSA_OVERRIDE=""
else
    HSA_OVERRIDE="HSA_OVERRIDE_GFX_VERSION=11.0.2"
fi
say "host=$HOST  HSA_OVERRIDE='${HSA_OVERRIDE:-(unset)}'"

mkdir -p "$ROOT/recipes" "$ROOT/work/logs"

# Pinned tensors (from locked methodology)
cat > "$PINNED" <<'PINEOF'
{
  "output.weight": "Q6_K",
  "token_embd.weight": "Q8_0"
}
PINEOF

say "=================================================="
say "prismaquant pipeline starting (pid=$$)"
say "  probe   = $PROBE_PKL"
say "  bf16    = $QWEN_BF16"
say "  outdir  = $QWEN_OUTDIR"
say "  budgets = ${!BUDGETS[*]}"
say "=================================================="

# ============================================================================
# Stage A — wait for probe.pkl
# ============================================================================
set_status "A. waiting for probe.pkl ($PROBE_PKL)"
while [[ ! -f "$PROBE_PKL" ]]; do
    sleep 60
done
say "A. probe.pkl exists ($(stat -c%s $PROBE_PKL) bytes)"

# ============================================================================
# Stage B — wait for sweep summary
# ============================================================================
set_status "B. waiting for sweep summary marker"
while [[ ! -f "$SWEEP_MARKERS/50-summary.done" ]]; do
    sleep 60
done
say "B. sweep complete"

# Wait an extra few seconds for ai00 to flush GPU state
sleep 10

# ============================================================================
# Stage B0 — merge BF16 splits to single file (quantize-cost can't read splits)
# ============================================================================
if [[ -f "$QWEN_BF16" ]]; then
    say "B0. merged BF16 already exists: $QWEN_BF16 (skip)"
else
    set_status "B0. merging BF16 splits to single GGUF"
    "$GGUF_SPLIT_BIN" --merge "$QWEN_BF16_SPLIT" "$QWEN_BF16" \
        2>&1 | tee -a "$ROOT/work/logs/merge.log"
    if [[ ! -f "$QWEN_BF16" ]]; then
        set_status "FAIL: B0 merged GGUF not produced"
        exit 1
    fi
    say "B0. merged: $(stat -c%s $QWEN_BF16) bytes"
fi

# ============================================================================
# Stage C — generate Qwen3.6 imatrix on ai00 GPU
# ============================================================================
if [[ -f "$IMATRIX" ]]; then
    say "C. imatrix already exists: $IMATRIX (skip)"
else
    set_status "C. generating imatrix with bartowski calibration"
    env $HSA_OVERRIDE \
        "$IMATRIX_BIN" \
        -m "$QWEN_BF16" \
        -f "$BARTOWSKI" \
        -o "$IMATRIX" \
        -c 4096 -ngl 99 --no-mmap \
        --chunks 200 \
        2>&1 | tee -a "$ROOT/work/logs/imatrix.log"
    if [[ ! -f "$IMATRIX" ]]; then
        set_status "FAIL: C imatrix not produced"
        exit 1
    fi
    say "C. imatrix produced ($(stat -c%s $IMATRIX) bytes)"
fi

# ============================================================================
# Stage D — quantize-cost (BF16 -> per-tensor MSE per format)
# ============================================================================
if [[ -f "$COSTS" ]]; then
    say "D. costs already exist: $COSTS (skip)"
else
    set_status "D. measuring per-(tensor,format) MSE (representative-subset: token_embd+output+blk.{0,3}.*)"
    "$QUANTIZE_COST_BIN" \
        --model "$QWEN_BF16" \
        --types "$FORMATS_GGML" \
        --imatrix "$IMATRIX" \
        --include-regex '^(token_embd|output|blk\.(0|3))\.' \
        --output "$COSTS" \
        2>&1 | tee -a "$ROOT/work/logs/quantize-cost.log"
    rc=${PIPESTATUS[0]}
    if [[ $rc -ne 0 ]] || [[ ! -f "$COSTS" ]]; then
        set_status "FAIL: D quantize-cost exit=$rc, costs file missing or empty"
        exit 1
    fi
    say "D. costs.csv: $(wc -l < $COSTS) rows"
fi

# ============================================================================
# Stage E — bridge probe.pkl -> bridge.json (HF -> GGUF tensor names)
# ============================================================================
if [[ -f "$BRIDGE" ]]; then
    say "E. bridge already exists: $BRIDGE (skip)"
else
    set_status "E. bridging probe.pkl tensor names to GGUF"
    /usr/bin/python3 \
        "$ROOT/scripts/bridge_probe_to_gguf.py" \
        --probe "$PROBE_PKL" \
        --output "$BRIDGE" \
        --aggregate sum \
        --unmapped-out "$ROOT/qwen36-bridge-unmapped.json" \
        2>&1 | tee -a "$ROOT/work/logs/bridge.log"
    if [[ ! -f "$BRIDGE" ]]; then
        set_status "FAIL: E bridge not produced"
        exit 1
    fi
fi

# ============================================================================
# Stage F — allocator (3 budgets) + pareto sweep
# ============================================================================
set_status "F. allocator solving for 3 budgets"
PARETO_BUDGETS="11.0,12.5,14.21,16.0,17.0,18.83,19.5,21.15,23.0,25.0"
for tag in "${!BUDGETS[@]}"; do
    budget="${BUDGETS[$tag]}"
    recipe_json="$ROOT/recipes/recipe-$tag.json"
    if [[ -f "$recipe_json" ]]; then
        say "F. recipe-$tag already exists (skip)"
        continue
    fi
    say "F. budget $tag = $budget GB"
    /usr/bin/python3 \
        "$ROOT/scripts/allocator.py" \
        --bridge "$BRIDGE" \
        --costs "$COSTS" \
        --budget-gb "$budget" \
        --pinned "$PINNED" \
        --recipe-out "$recipe_json" \
        --pareto-csv "$ROOT/recipes/pareto-$tag.csv" \
        --pareto-budgets-gb "$PARETO_BUDGETS" \
        --allow-types "$FORMATS_GGML" \
        --gguf "$QWEN_BF16" \
        --propagate-from-exemplars \
        --exemplar-layers "0,3" \
        2>&1 | tee -a "$ROOT/work/logs/allocator-$tag.log"
    rc=${PIPESTATUS[0]}
    if [[ $rc -ne 0 ]] || [[ ! -f "$recipe_json" ]]; then
        set_status "FAIL: F $tag allocator exit=$rc, recipe missing"
        exit 1
    fi
done

# ============================================================================
# Stage G — apply each recipe via llama-quantize
# ============================================================================
for tag in "${!BUDGETS[@]}"; do
    out_gguf="$QWEN_OUTDIR/Qwen3.6-35B-A3B-prismaquant-$tag.gguf"
    if [[ -f "$out_gguf" ]]; then
        say "G. $tag GGUF already exists (skip)"
        continue
    fi
    set_status "G. quantizing for budget $tag"
    env $HSA_OVERRIDE \
        "$QUANTIZE_BIN" \
        --imatrix "$IMATRIX" \
        --tensor-type-file "$ROOT/recipes/recipe-$tag.txt" \
        "$QWEN_BF16" \
        "$out_gguf" \
        IQ4_KS \
        2>&1 | tee -a "$ROOT/work/logs/quantize-$tag.log"
    rc=${PIPESTATUS[0]}
    if [[ $rc -ne 0 ]]; then
        set_status "FAIL: G $tag llama-quantize exit=$rc"
        rm -f "$out_gguf"
        exit 1
    fi
    if [[ ! -f "$out_gguf" ]]; then
        set_status "FAIL: G $tag GGUF not produced (no error, but missing output)"
        exit 1
    fi
    say "G. $tag GGUF: $(stat -c%s $out_gguf) bytes ($(echo "scale=2; $(stat -c%s $out_gguf) / 1073741824" | bc) GB)"
done

# ============================================================================
# Stage H — PPL eval each at chunks=100
# ============================================================================
for tag in "${!BUDGETS[@]}"; do
    out_gguf="$QWEN_OUTDIR/Qwen3.6-35B-A3B-prismaquant-$tag.gguf"
    ppl_log="$ROOT/work/logs/ppl-$tag.log"
    if [[ ! -f "$out_gguf" ]]; then continue; fi
    if [[ -f "$ppl_log" ]] && grep -q "Final estimate" "$ppl_log"; then
        say "H. PPL $tag already done (skip)"
        continue
    fi
    set_status "H. PPL evaluation for $tag (chunks=100)"
    env $HSA_OVERRIDE \
        "$PERPLEXITY_BIN" \
        -m "$out_gguf" \
        -f "$WIKITEXT" \
        -c 4096 -b 2048 -ctk f16 -ctv f16 -fa on -ngl 99 --chunks 100 --no-mmap \
        > "$ppl_log" 2>&1 || say "H. WARN: $tag perplexity exited non-zero (continuing)"
    say "H. $tag PPL: $(grep 'Final estimate' "$ppl_log" | head -1 || echo "(no Final estimate)")"
done

# ============================================================================
# Stage I — summary report
# ============================================================================
set_status "I. writing summary report"
{
    echo "# prismaquant pipeline — Qwen3.6-35B-A3B"
    echo ""
    echo "Started: see pipeline.log"
    echo "Finished: $(ts)"
    echo ""
    echo "## Recipes"
    echo ""
    echo "| Tag | Budget | Actual | Loss surrogate | λ |"
    echo "|---|---|---|---|---|"
    for tag in "${!BUDGETS[@]}"; do
        recipe="$ROOT/recipes/recipe-$tag.json"
        [[ -f "$recipe" ]] || continue
        b=$(jq -r .budget_gb "$recipe")
        a=$(jq -r .actual_size_gb "$recipe")
        l=$(jq -r .loss_surrogate "$recipe")
        lm=$(jq -r .lambda "$recipe")
        echo "| $tag | $b GB | $a GB | $l | $lm |"
    done
    echo ""
    echo "## PPL results"
    echo ""
    echo "| GGUF | Size | PPL (chunks=100) |"
    echo "|---|---|---|"
    for tag in "${!BUDGETS[@]}"; do
        out_gguf="$QWEN_OUTDIR/Qwen3.6-35B-A3B-prismaquant-$tag.gguf"
        ppl_log="$ROOT/work/logs/ppl-$tag.log"
        [[ -f "$ppl_log" ]] || continue
        size_bytes=$(stat -c%s "$out_gguf" 2>/dev/null || echo 0)
        size_gb=$(echo "scale=2; $size_bytes / 1073741824" | bc)
        ppl=$(grep 'Final estimate' "$ppl_log" | head -1 | grep -oE 'PPL = [0-9.]+ \+/- [0-9.]+' || echo "—")
        echo "| prismaquant-$tag | $size_gb GB | $ppl |"
    done
    echo ""
    echo "Compare against existing PLLow baselines in WEIGHT-QUANTS.md."
} > "$SUMMARY"

set_status "DONE — see $SUMMARY"
say "=================================================="
say "Pipeline complete."
say "  recipes: $ROOT/recipes/"
say "  GGUFs:   $QWEN_OUTDIR/"
say "  summary: $SUMMARY"
say "=================================================="
