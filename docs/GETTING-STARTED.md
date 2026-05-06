# Getting Started with prismaquant-llama

A hands-on walkthrough for first-time users. By the end, you'll have:

- A working install
- A first prismaquant GGUF for a real model
- A calibrated system perf table for your binary + hardware
- Working knowledge of budgets, priorities, and the output layout

**Time budget**: ~10 min reading, ~30 min for the first hands-on run,
~1–2 hr if you also do the optional system calibration.

---

## What prismaquant-llama actually does

> **Credit**: the per-tensor mixed-precision allocation methodology (the
> Bayesian closed-form Δloss surrogate `0.5 · H_trace · MSE`, the probe →
> cost → allocate pipeline) was designed by
> [**RobTand**](https://github.com/RobTand/prismaquant) for the original
> [`prismaquant`](https://github.com/RobTand/prismaquant) project (vLLM /
> compressed-tensors). prismaquant-llama brings that methodology to
> llama.cpp / GGUF.

Most quantizers pick a single format like `Q4_K_M` and apply it to every
tensor. prismaquant-llama instead picks a **different format for each
tensor** based on:

1. How much each tensor "matters" (Fisher information from a probe pass)
2. How much PPL-error each format introduces for that tensor (cost measurement)
3. Performance characteristics on **your** binary + hardware (calibration)
4. A user-set **budget** (target file size as % of BF16) and **priority**
   (PPL quality / TG speed / PP speed weighting)

Result: a mixed-format GGUF that's typically 5–15% smaller than a uniform
K-quant at equal PPL, OR equal size at noticeably better PPL.

---

## Prerequisites

1. **A llama.cpp build** with `llama-quantize`, `llama-imatrix`,
   `llama-perplexity`, `llama-bench`. Any modern fork works (mainline,
   ik_llama, frankenturbo2). Either put them on `$PATH`, or set
   `path = "/your/llama/bin"` in the config (see below).

2. **`llama-quantize-cost`** — Stage E needs a custom binary not in
   mainline. The C++ source ships inside this package. Find and build:

   ```bash
   pip install --user prismaquant-llama
   SRC=$(python -c "import prismaquant_llama; print(prismaquant_llama.quantize_cost_source_path())")
   cp -r "$SRC" /path/to/your/llama.cpp/tools/
   echo 'add_subdirectory(quantize-cost)' >> /path/to/your/llama.cpp/tools/CMakeLists.txt
   cmake --build /path/to/your/llama.cpp/build --target llama-quantize-cost
   ```

   Resulting binary lives next to `llama-quantize`.

3. **The `prismaquant` Python package** — Stage C (Hessian probe) shells
   out to `python3 -m prismaquant.incremental_probe`. Install our fork
   (carries Gemma-4 + NemotronH patches):

   ```bash
   pip install git+https://github.com/jimbothigpen/prismaquant.git
   ```

4. **Python 3.10+**

5. **Disk space**: roughly **2–2.5× the BF16 size** of your target model.
   For a 9B model that's ~40–45 GB peak.

6. **GPU** recommended (CPU-only works but is much slower).

### Picking the right HuggingFace model

`run` requires a **safetensors** repo — the same string you'd pass to
`from_pretrained()`. Rules:

- ✅ `unsloth/gemma-3-4b-it` (safetensors)
- ❌ `unsloth/gemma-3-4b-it-GGUF` (already-quantized — Stage B has nothing
  to convert)
- ❌ `bartowski/gemma-3-4b-it-GGUF` (same)

Look for `model.safetensors` / `model-00001-of-N.safetensors` files in the
HF "Files and versions" tab. Gated models (Llama, Gemma, etc.):
`hf auth login` once and accept the license in your browser first.

---

## Install

```bash
pip install --user prismaquant-llama
prismaquant-llama --help
```

The CLI is two subcommands: `run` and `calibrate`. That's the whole tool.

The first time you invoke it, a starter config gets dropped at
`~/.prismaquant-llama/config.toml` and you'll see:

```
[prismaquant-llama] wrote starter config → /home/you/.prismaquant-llama/config.toml
[prismaquant-llama] edit it to set defaults; CLI flags override per run.
```

That file is heavily commented; open it and tune to taste.

---

## Step 1: First run

```bash
prismaquant-llama run unsloth/gemma-3-4b-it
```

That's the whole minimal command. Before any heavy work starts, you'll
see a pre-flight estimate and an accept/abort prompt:

```
┌─ prismaquant-llama run ─ gemma-3-4b-it ────
│ Budget:    25% of BF16    Priority: 111
│ Formats:   12    imatrix_chunks: 50    ppl_chunks: 50
│
│ Estimated disk:
│   source / HF download:   ~9.3 GB
│   BF16 GGUF:              ~9.8 GB
│   final PQ GGUF:          ~2.4 GB
│   peak during pipeline:   ~21.5 GB
│
│ Estimated wall time:    23m–58m  (rough)
│
│ Free disk:              ~48.2 GB
└─────────────────────────────────────────────
Proceed? [y/N]
```

Type `y` to proceed, anything else (or just hit return) to abort.
For scripted use, pass `--yes` to skip the prompt.

Defaults applied (visible in the estimate block):

| Default | Value |
|---|---|
| `path` | `$PATH` lookup |
| `quants` | 12 mainline formats: `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, `IQ3_XXS`, `IQ3_XS`, `IQ3_S`, `IQ3_M`, `IQ4_XS`, `IQ4_NL`, `BF16` |
| `budget` | 25% of BF16 |
| `priority` | 111 (equal PPL/TG/PP) |
| `ppl_corpus` | bundled wikitext-2-raw |
| `imatrix_corpus` | bundled bartowski-v5-semantic |
| `ppl_chunks` / `imatrix_chunks` | 50 each |
| `convert_script` / `libs` | auto-discover / no override |

This runs all 9 stages:

| Stage | What | Wall time |
|---|---|---|
| A | Download HF model | 1–5 min |
| B | Convert to BF16 GGUF | 1–3 min |
| C | Probe + Fisher backward | 3–8 min |
| D | imatrix generation | 2–4 min |
| E | per-(tensor, format) MSE costs | 3–6 min |
| F | bridge HF→GGUF names | <1 sec |
| G | allocate formats per tensor | <1 sec |
| H | apply allocation | 1–3 min |
| I | PPL eval | 1–2 min |

Final GGUF lands at `~/.prismaquant-llama/ggufs/gemma-3-4b-it-PQ25-111.gguf`.

**Re-running** the same model with different budget/priority is fast —
Stages A–E cache and skip on re-run. Only F–I re-execute (~5–10 min).

```bash
# Try a quality-priority recipe at 30% budget
prismaquant-llama run unsloth/gemma-3-4b-it --budget 30 --priority 522
```

---

## Step 2: Editing the config

`~/.prismaquant-llama/config.toml`:

```toml
[prismaquant-llama]
base           = "~/.prismaquant-llama/"
path           = ""
quants         = ["Q3_K","Q4_K","Q5_K","Q6_K","Q8_0",
                  "IQ3_XXS","IQ3_XS","IQ3_S","IQ3_M",
                  "IQ4_XS","IQ4_NL","BF16"]
budget         = 25
priority       = "111"
ppl_corpus     = ""
imatrix_corpus = ""
ppl_chunks     = 50
imatrix_chunks = 50
convert_script = ""
libs           = ""
```

**Common tweaks:**

- `path = "/home/me/llama.cpp/build/bin"` — point at a specific build
- `budget = 30` — slightly larger files at slightly better quality
- `priority = "522"` — quality-first (PPL-heavy)
- `priority = "252"` — TG-first (good for chat / streaming)
- `priority = "225"` — PP-first (good for long-context)
- `imatrix_chunks = 200` — production-grade imatrix (slower)
- `ppl_chunks = 100` — tighter PPL stderr in eval

**Adding fork-specific quants** (ik_llama, frankenturbo2, etc.) — just
append to the `quants` list. The shipped default is mainline-only because
that's the lowest-common-denominator. To see what your binary supports:

```bash
llama-quantize --help
```

Pick the quants you want and add them. Avoid 2-bit quants (IQ2_S, etc.)
unless you specifically need extreme compression — they're a documented
PPL cliff on most architectures.

---

## Step 3: Understand the output

After a run completes:

```
~/.prismaquant-llama/
├── _shared/                                        cached, survives re-runs
│   ├── hf-cache/gemma-3-4b-it/                     Stage A
│   ├── bf16/gemma-3-4b-it-BF16.gguf                Stage B (~7 GB)
│   ├── probe/gemma-3-4b-it-probe.pkl               Stage C
│   └── imatrix-cache/<sha>__<sha>__c50.imatrix.gguf  Stage D
├── ggufs/
│   └── gemma-3-4b-it-PQ25-111.gguf                 ← THE FINAL GGUF
└── work/gemma-3-4b-it-<timestamp>/                 per-run scratch
    ├── costs/costs.csv                              Stage E
    ├── bridge.json                                  Stage F
    ├── recipes/recipe-PQ25-111.json                 Stage G
    └── logs/                                        all subprocess output
```

**Filename convention**: `<model>-PQ<budget>-<priority>.gguf` — so
`PQ25-111` is "25% target, equal-priority".

---

## Step 4: Calibrate the perf table (optional but recommended)

prismaquant-llama ships with a hardware-agnostic system perf table
(Qwen3-8B at chunks=200, ratio-only). It works out of the box but isn't
optimal for *your* binary + hardware.

To calibrate against your setup, pick a smallish reference model (a 1B–9B
dense model is ideal) and run:

```bash
prismaquant-llama calibrate system Qwen/Qwen3-8B \
    --ppl-chunks 100
```

This:

1. Generates an imatrix on the reference GGUF (using the configured
   `imatrix_corpus`) once, and caches it in `_shared/imatrix-cache/`
   for re-use by future `run` invocations.
2. For each format in your `quants` list: quantizes the reference GGUF
   to that format **using the imatrix**, runs `llama-perplexity`,
   runs `llama-bench`.
3. Persists results to `~/.prismaquant-llama/calibration/system.json`
   *after every format*. The runner picks this up automatically on
   subsequent `run` invocations.

Wall time: ~5–15 min per format × N formats. For the default 12-quant
list that's roughly an hour and a half; for a fork-extended list of
~22 quants, ~3–4 hours.

**Resume-safe**: if a calibration run is killed or crashes mid-sweep,
just re-invoke the same command. Already-completed formats are skipped
and the run picks up where it left off.

**Live logs**: subprocess output streams to per-format log files
under `<base>/work/<run>/logs/`. To watch:

```bash
# Meta progress (which format / which step)
tail -f ~/.prismaquant-llama/work/Qwen3-8B-*/logs/calibrate.log

# Live subprocess output for a specific format
tail -f ~/.prismaquant-llama/work/Qwen3-8B-*/logs/calibrate-Q4_K.log
```

**Tip**: a quick calibration with `--ppl-chunks 50` is faster and adequate
for ranking adjacent K-quants. Bump to 200 if you care about precise
PPL-Δ values.

### Per-model calibration (rare)

If a specific model behaves unusually under quantization, run a model-
specific calibration that overrides the system default for that model:

```bash
prismaquant-llama calibrate model unsloth/gemma-3-4b-it
# → ~/.prismaquant-llama/calibration/models/gemma-3-4b-it.json
```

The pipeline picks model > system > shipped, so a per-model file always
wins.

---

## Step 5: Common workflows

### Sweep budgets at one priority

```bash
for B in 20 25 30 35; do
    prismaquant-llama run unsloth/gemma-3-4b-it --budget $B --priority 522
done
```

Stages A–E cache, only F–I run per iteration. ~25–40 min for 4 GGUFs.

### Compare priorities at one budget

```bash
for P in 111 522 252; do
    prismaquant-llama run unsloth/gemma-3-4b-it --budget 25 --priority $P
done
```

3 GGUFs at the same target budget, different format choices per tensor.

### Disk hygiene

`--purge yes` (default) cleans up after each run. To re-use HF caches +
BF16 + imatrix between runs, pass `--purge no` until you're satisfied:

```bash
prismaquant-llama run unsloth/gemma-3-4b-it --purge no
prismaquant-llama run unsloth/gemma-3-4b-it --budget 30 --priority 522 --purge no
prismaquant-llama run unsloth/gemma-3-4b-it --budget 35 --priority 252  # default purge yes — cleans up
```

### Multiple binaries (mainline + fork)

Maintain separate config files and use `--config`:

```bash
# Mainline build
prismaquant-llama run unsloth/gemma-3-4b-it \
    --config ~/configs/mainline.toml

# ik_llama fork build
prismaquant-llama run unsloth/gemma-3-4b-it \
    --config ~/configs/ikllama.toml
```

Each config has its own `path`, `base`, and `quants` — completely
independent caches and outputs.

### Library path issues

If your llama.cpp binaries depend on `libllama.so` next to them but your
loader doesn't see it:

```bash
prismaquant-llama run unsloth/gemma-3-4b-it --libs /opt/llama/lib
```

`--libs` is prepended to `LD_LIBRARY_PATH` for every subprocess call.

---

## Reference

### Useful flags (run subcommand)

```bash
prismaquant-llama run INPUT \
    [--config PATH]              # alternative config.toml
    [--libs DIR]                 # extra LD_LIBRARY_PATH dir
    [--base DIR]                 # working directory (default: from config)
    [--path DIR]                 # llama.cpp binary directory (default: $PATH)
    [--quants Q1,Q2,...]         # whitelist (default: from config)
    [--budget INT]               # % of BF16 (default: from config)
    [--priority XYZ]             # PPL/TG/PP ratio (default: from config)
    [--ppl-corpus PATH|URL]      # default: bundled wikitext
    [--imatrix-corpus PATH|URL]  # default: bundled bartowski-v5
    [--imatrix PATH|URL]         # use existing imatrix file (skip Stage D)
    [--ppl-chunks N]             # default: from config
    [--imatrix-chunks N]         # default: from config
    [--convert-script PATH]      # convert_hf_to_gguf.py location (default: auto-discover)
    [--purge {yes,no}]           # default: yes
    [--yes | -y]                 # skip the pre-flight confirmation prompt
```

### Useful flags (calibrate subcommand)

```bash
prismaquant-llama calibrate {system|model} INPUT \
    [--config PATH]              # alternative config.toml
    [--libs DIR]                 # extra LD_LIBRARY_PATH dir
    [--base DIR]                 # working directory (default: from config)
    [--path DIR]                 # llama.cpp binary directory (default: $PATH)
    [--quants Q1,Q2,...]         # whitelist (default: from config)
    [--ppl-corpus PATH|URL]      # default: bundled wikitext
    [--imatrix-corpus PATH|URL]  # default: bundled bartowski-v5
    [--imatrix PATH|URL]         # use existing imatrix file (skip generation)
    [--ppl-chunks N]             # default: from config
    [--imatrix-chunks N]         # default: from config
    [--convert-script PATH]      # convert_hf_to_gguf.py location (default: auto-discover)
    [--purge {yes,no}]           # default: yes
    [--yes | -y]                 # skip the pre-flight confirmation prompt
```

### Pre-flight prompt

Both subcommands show an estimate of disk usage and wall-time range
before doing any heavy work, and prompt `Proceed? [y/N]`. Default is
N — you have to actively confirm. Use `--yes` (or `-y`) to skip the
prompt; this is required for any non-interactive / scripted use,
otherwise the command exits with an error on non-TTY stdin.

The estimate uses HF API to query weight sizes for HF-id inputs;
on-disk inputs scan locally. If HF is unreachable, the source-size
field shows `(unknown — HF API unreachable?)` but the estimate
continues with what's known.

### Troubleshooting

**"llama-quantize not found"** — set `path` in config.toml to your
llama.cpp binary directory, or pass `--path /your/bin`.

**"llama-quantize-cost not found"** — build from the bundled C++ source
(see Prerequisites section).

**"prismaquant package not installed"** (Stage C) — `pip install
git+https://github.com/jimbothigpen/prismaquant.git`.

**"convert_hf_to_gguf.py not found"** — the script lives at the root of
your llama.cpp source tree and is **not** installed by
`cmake --install` / `make install`. Three fixes:
- set `convert_script = "/path/to/convert_hf_to_gguf.py"` in
  `~/.prismaquant-llama/config.toml`
- pass `--convert-script /path/to/convert_hf_to_gguf.py` per run
- add the script to `$PATH` (or symlink it next to `llama-quantize`)

**Wrong-shape HF model error during Stage B** — usually means INPUT
points at a `-GGUF` repo. Use the safetensors base repo instead.

**Stage I crashes inside the GPU backend** — the GGUF was produced
cleanly by Stage H; you can run `llama-perplexity` separately later.
Add `--ppl-chunks 0` if you want to skip the eval entirely (or just
ignore the Stage I error — your final GGUF is in `ggufs/`).

---

## Next steps

- Read [`docs/methodology.md`](methodology.md) for the math behind the
  allocator's surrogate scoring.
- Read [`README.md`](../README.md) for the high-level reference.

If you build something interesting or hit an edge case, please file an
issue at <https://github.com/jimbothigpen/prismaquant-llama/issues>.
