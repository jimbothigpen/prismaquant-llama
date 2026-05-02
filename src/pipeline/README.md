# tools/prismaquant — Bayesian per-tensor mixed quantization for GGUF

Pipeline that turns a model into 1+ GGUF files where every Linear gets a different ggml format chosen by `argmin(½·H_trace·MSE + λ·size)` — closed-form Δloss surrogate from the [prismaquant project](https://github.com/RobTand/prismaquant), adapted to llama.cpp's quant catalog.

See [`../../PRISMAQUANT.md`](../../PRISMAQUANT.md) for methodology, recipe gallery, and strategy guidance. This directory contains the runnable pipeline.

## Layout

```
tools/prismaquant/
├── run-pipeline.sh           — master orchestrator (stages A–I)
├── format-tps-gfx1150.json   — per-format PP/TG TPS table from ai00 sweep
│                               (drives the multi-objective allocator)
├── pinned-tensors-qwen36.json— example pin-list (token_embd, output)
├── scripts/
│   ├── allocator.py          — multi-choice knapsack solver, multi-objective
│   ├── bridge_probe_to_gguf.py — HF→GGUF tensor-name mapping (qwen35moe-aware)
│   ├── run-ppl100.sh         — chained PPL eval at chunks=100
│   └── run-baselines-c100.sh — chained PPL on uniform baselines for comparison
└── recipes/                  — example outputs from a Qwen3.6-35B-A3B run
    ├── qwen36-14G-pure-ppl.json    — pure-PPL allocator (priority 900)
    ├── qwen36-19G-pure-ppl.json
    └── qwen36-21G-pure-ppl.json
```

## Dependencies

- `prismaquant` upstream package (probe runner): `pip install --user --break-system-packages -e <prismaquant repo>`
- `transformers`, `torch`, `safetensors`, `numpy`, `accelerate`, `datasets`
- llama.cpp built with `tools/quantize-cost` (`-DGGML_BUILD_TOOLS=ON`)

## Quick start (for a new model)

```bash
# 1. Generate Hessian probe (from upstream prismaquant)
python3 -m prismaquant.incremental_probe \
    --model /path/to/HF_model \
    --dataset /path/to/calibration.txt \
    --nsamples 16 --seqlen 512 \
    --device cpu --dtype bf16 \
    --output probe.pkl \
    --activation-cache-dir act-cache --work-dir work

# 2. Generate imatrix
llama-imatrix -m model-BF16.gguf -f calibration.txt -o imatrix.gguf -c 4096 -ngl 99 --no-mmap --chunks 200

# 3. Measure per-(tensor, format) MSE
llama-quantize-cost --model model-BF16.gguf --types Q4_K,Q5_K,Q6_K,IQ4_K,... \
                    --imatrix imatrix.gguf --output costs.csv \
                    --include-regex '^(token_embd|output|blk\.(0|3))\.'

# 4. Bridge HF tensor names → GGUF tensor names
python3 scripts/bridge_probe_to_gguf.py --probe probe.pkl --output bridge.json

# 5. Allocate at a target size + priority
python3 scripts/allocator.py \
    --bridge bridge.json --costs costs.csv \
    --budget-gb 5.5 --budget-band-gb 0.25 \
    --priority 522 --tps format-tps-gfx1150.json \
    --gguf model-BF16.gguf --propagate-from-exemplars \
    --pinned pinned-tensors-qwen36.json \
    --recipe-out recipe.json

# 6. Apply recipe
llama-quantize --imatrix imatrix.gguf --tensor-type-file recipe.txt \
               model-BF16.gguf out.gguf IQ4_KS
```

## The `--priority NNN` knob

Three digits, one per dimension `(PPL, TG, PP)` summing to 9 by convention. The allocator combines normalized per-tensor costs with these weights:

| Priority | Meaning |
|---|---|
| `900` | pure PPL (original allocator, ignore speed) |
| `333` | balanced PPL/TG/PP |
| `522` | PPL primary, TG=PP secondary |
| `252` | TG primary, PPL=PP secondary |
| `225` | PP primary, PPL=TG secondary |
| `531` | PPL > TG > PP strict |
| `153` | TG > PP > PPL strict |
| `090` | pure TG (fastest TG at this size) |
| `009` | pure PP (fastest PP at this size) |

`--budget-band-gb 0.25` accepts any recipe within ±0.25 GB of the target — the discrete format space sometimes has gaps that make exact-match infeasible.

## Expected output

A successful run produces:

- **`recipe.json`** — per-tensor format assignment, looks like:
  ```json
  {
    "blk.0.attn_q.weight": "Q6_K",
    "blk.0.attn_k.weight": "Q5_K",
    "blk.0.ffn_gate.weight": "IQ4_KS",
    ...
  }
  ```
- **Final GGUF**: e.g., `Qwopus3.5-9B-v3.5-PQ5.25-522.gguf` (per the
  naming scheme documented in `../../PRISMAQUANT.md`).
- **PPL parity**: prismaquant 5.25 GB recipe should land within ~0.05
  PPL of a uniform Q5_K_M (5.5 GB) at lower bytes — typical Pareto win.

Concrete Pareto-winner numbers from Qwopus3.5-9B-v3.5 Stage H eval:

| Recipe | Total GB | PPL |
|---|---:|---:|
| `PQ6.5-522` (best 6.5GB) | 6.74 | 7.4269 |
| `PQ5.25-900` (sweet spot) | 5.36 | 7.4701 |
| `PQ4.0-252` (extreme) | 3.83 | 8.9116 |

## Troubleshooting

- **`llama-quantize-cost` missing**: build llama.cpp with
  `-DGGML_BUILD_TOOLS=ON` (some CMake presets disable tools to speed up
  iteration; re-enable for prismaquant).
- **Bridge produces empty mapping**: the `bridge_probe_to_gguf.py`
  script knows about `qwen35moe`, `qwopus`, dense Llama, and Mixtral.
  For other architectures, update `MODEL_NAME_PATTERNS` in the script.
- **Allocator infeasible at exact budget**: increase
  `--budget-band-gb` (default 0.25 GB) or pick a different
  `--priority`. The discrete format space sometimes has gaps.
- **Empty `costs.csv`**: ensure `--include-regex` matches at least one
  block (default exemplar pattern is `^(token_embd|output|blk\.(0|3))\.`).
- **Recipe gives worse PPL than uniform Q at same size**: usually means
  the imatrix is poorly calibrated. Re-run `llama-imatrix` with a
  larger `--chunks` value (200+) on a calibration set that matches
  your target use case.

## Notes

- The example recipes in `recipes/` are from a `--priority 900` (pure-PPL) run on Qwen3.6-35B-A3B with the canonical 12-format whitelist. For multi-priority recipes, run the allocator with `--priority` and `--tps`.
- Cost propagation (`--propagate-from-exemplars --exemplar-layers 0,3`) measures only a few representative layers and copies cost data to peer-type peers. Saves ~12× on Stage D wallclock for hybrid SSM models. For pure-dense models, `0,1` is sufficient.
- Per-format TPS data in `format-tps-gfx1150.json` is from gfx1150 (ai00). Format-relative ratios should transfer to other ROCm/HIP hardware; absolute values won't. Re-bench on your hardware for accurate priority weights.

## Naming scheme for output GGUFs

Prismaquant GGUFs use `<base>-PQ<budget>-<XYZ>.gguf` where `<XYZ>` is
the priority code. See [`../../PRISMAQUANT.md`](../../PRISMAQUANT.md)
for the full key.
