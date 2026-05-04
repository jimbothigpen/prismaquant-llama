# Getting Started with prismaquant-llama

A hands-on walkthrough for first-time users. By the end, you'll have:
- A working install with auto-discovered llama.cpp binaries
- A first prismaquant GGUF for a real model
- An empirical performance file calibrated to your binary + hardware
- Working knowledge of budgets, priorities, and the output layout

**Time budget for this doc**: ~10 min reading, ~30 min for the first hands-on run, ~1-2 hr for full hardware calibration if you want it.

---

## What prismaquant-llama actually does

> **Credit**: the per-tensor mixed-precision allocation methodology (the Bayesian
> closed-form Δloss surrogate `0.5 · H_trace · MSE`, the probe → cost → allocate
> pipeline shape) was designed by [**RobTand**](https://github.com/RobTand/prismaquant)
> for [the original `prismaquant`](https://github.com/RobTand/prismaquant) project
> (vLLM / compressed-tensors). prismaquant-llama is an adaptation that brings
> that methodology to llama.cpp / GGUF — the math + design are RobTand's; this
> repo is the llama.cpp-side plumbing.

Most quantizers (mainline llama.cpp, etc.) pick a single format like `Q4_K_M` and apply it to every tensor. prismaquant-llama instead picks a **different format for each individual tensor** based on:

1. How much each tensor "matters" (Fisher information from a probe pass)
2. How much PPL-error each format introduces for that tensor (cost measurement)
3. Performance characteristics on **your** binary + hardware (calibration)
4. A user-set **budget** (target file size) and **priority** (PPL-quality vs prompt-processing speed vs token-generation speed)

Result: a mixed-format GGUF that's typically 5-15% smaller than a uniform K-quant at equal PPL, OR equal size at noticeably better PPL.

---

## Prerequisites

You need:

1. **A llama.cpp binary set** with `llama-quantize`, `llama-imatrix`, `llama-perplexity`, `llama-bench`. Any modern fork works (mainline, ik_llama, frankenturbo2, etc.). Build them once and put them on disk. **For zero-config runs, make sure these binaries are on `$PATH`** (the tool will find them automatically); otherwise pass `--binary` explicitly.

2. **Python 3.10+** and `pip` (or your package manager)

3. *(Optional)* **A custom calibration corpus** — a plain text file with diverse prose. **prismaquant-llama ships with a default corpus** (a copy of [bartowski-calibration-v3.txt](https://gist.github.com/bartowski1182/eb213dccb3571f863da82e99418f81e8) by **bartowski**, ~280 KB, public-domain text from Wikipedia / public code / etc.) so first runs work out of the box. Bring your own corpus only if you want one closer to your target deployment domain.

4. **Disk space**: ~3× the BF16 size of your target model (for HF cache + BF16 GGUF + intermediate work). For a 9B model that's ~50 GB.

5. **GPU** (recommended): supported by your llama.cpp build. CPU-only works but is much slower.

### Picking the right HuggingFace model path

`--hf-model` takes a HuggingFace **safetensors** model repo ID — the same string you'd plug into `from_pretrained()`. A few rules:

- **Must be safetensors** (i.e., the repo contains `model.safetensors` / `model-00001-of-N.safetensors` files). prismaquant-llama runs Stage B convert-hf-to-gguf itself; it does not consume third-party GGUFs.
- **Do NOT pass a GGUF-only repo** (e.g., `unsloth/gemma-3-4b-it-GGUF`, `bartowski/Llama-3-8B-Instruct-GGUF`, anything with `-GGUF` in the name). These have nothing for the convert step to chew on. Pick the *base* repo instead.
- **Most HF model cards link to the right repo.** On the HF page, look for "Files and versions" — if you see `.safetensors` files, it's the right one. If you see only `.gguf`, scroll to "Use this model" or check the model card for a link to the un-quantized base.
- **Quick examples**:
  - ✅ `unsloth/gemma-3-4b-it` (safetensors)
  - ✅ `meta-llama/Llama-3.2-1B-Instruct` (safetensors, gated — accept license first)
  - ❌ `unsloth/gemma-3-4b-it-GGUF` (already-quantized GGUF — convert step has nothing to do)
  - ❌ `bartowski/gemma-3-4b-it-GGUF` (same — quantized output, not source)
- **Gated models** (Llama, Gemma, etc.): `huggingface-cli login` once, accept the license on the model card in your browser, then prismaquant-llama can download it.

---

## Install

From source (recommended for now):

```bash
git clone https://github.com/jimbothigpen/prismaquant-llama
cd prismaquant-llama
pip install -e .
```

Verify:

```bash
prismaquant-llama --help
prismaquant-llama paths find-binaries
```

The second command auto-discovers llama.cpp tools on common paths. If yours aren't found, you'll pass them via `--binary`.

---

## Step 1: First run (the 30-minute walkthrough)

Pick a small model for your first run. Recommended: `unsloth/gemma-3-4b-it` (downloads ~10 GB, full pipeline takes ~15 min on a modern GPU).

**The minimal first-run command is one flag:**

```bash
prismaquant-llama pipeline run --hf-model unsloth/gemma-3-4b-it
```

That's it. This works out of the box on most setups, with no config files and no other flags. Defaults that apply:

| Default | Value |
|---|---|
| `--binary` | auto-discovered via `$PATH` (and a few common build-dir locations) |
| `--calibration` | bundled corpus shipped with the package (bartowski-v3, ~280 KB) |
| `--output` | `~/.prismaquant-llama/builds/<model-name>-prismaquant/` (auto-created) |
| `--budget-gb` | 25% of the BF16 GGUF size |
| `--priority` | `333` (equal weight: PPL/TG/PP) |
| `--formats` | mainline llama.cpp set (Q2_K through Q8_0 + IQ2_S/IQ3_XXS/IQ3_S/IQ4_XS/IQ4_NL) |

**Every default is overrideable on the CLI** — pass `--binary`, `--calibration`, `--output`, `--budget-gb`, `--priority`, `--formats` at any time and the flag wins over the default. For example, to set everything explicitly:

```bash
prismaquant-llama pipeline run \
    --hf-model unsloth/gemma-3-4b-it \
    --binary /path/to/your/llama-quantize \
    --calibration /path/to/your-domain-corpus.txt \
    --output ~/quants/gemma-3-4b-it-prismaquant
```

For persistent overrides (so you don't pass them every run), see [Setting persistent preferences via `config.toml`](#setting-persistent-preferences-via-configtoml) below.

This will:

| Stage | What it does | Wall time |
|---|---|---|
| A | Download HF model | 1-5 min |
| B | Convert to BF16 GGUF (one-time per model) | 1-3 min |
| C | Probe forward + Fisher backward (on the model's BF16 GGUF) | 3-8 min |
| D | Generate imatrix from calibration corpus | 2-4 min |
| E | Measure per-(tensor, format) MSE costs | 3-6 min |
| F | Bridge probe → costs | <1 sec |
| G | Allocate formats per tensor | <1 sec |
| H | Apply allocation: `llama-quantize` writes the final GGUF | 1-3 min |
| I | Eval (PPL) on the final GGUF | 1-2 min |

The default behavior is **auto-budget at 25% of BF16 size + equal priority (333)**. So for a ~7 GB BF16, you'll get a ~1.8 GB GGUF.

**Re-running with different parameters** (e.g. different budget) is fast — Stages A-D cache to `_shared/` and skip on re-run.

---

## Step 2: Understanding the output

After the run completes, your output dir looks like this:

```
~/quants/gemma-3-4b-it-prismaquant/
├── _shared/                                   ← cached, survives re-runs
│   ├── hf-cache/gemma-3-4b-it/                ← Stage A
│   ├── bf16/gemma-3-4b-it-BF16.gguf           ← Stage B (~7 GB)
│   ├── probe/gemma-3-4b-it-probe.pkl          ← Stage C
│   ├── calibration/                           ← cached corpus chunks
│   └── imatrix-cache/<model>-BF16.imatrix.gguf ← Stage D
├── ggufs/
│   └── gemma-3-4b-it-PQ1.81-333.gguf          ← THE FINAL GGUF
└── work/
    └── gemma-3-4b-it-<timestamp>/             ← per-run scratch
        ├── costs/costs.csv                    ← Stage E
        ├── bridge.json                        ← Stage F
        ├── recipes/recipe-PQ1.81-333.json     ← Stage G (per-tensor format choices)
        └── logs/                              ← all subprocess outputs
```

**Filename convention**: `<model>-PQ<budget-gb>-<priority>.gguf`. So `PQ1.81-333` means "1.81 GB target, equal-priority". Higher budgets give larger files (better quality); priority codes are explained next.

---

## Step 3: Customizing the run

### Budget (target file size)

```bash
# Larger output (better quality), 35% of BF16 instead of 25%:
prismaquant-llama pipeline run \
    --budget-auto-ratio 0.35 \
    --hf-model unsloth/gemma-3-4b-it \
    --binary /path/to/llama-quantize \
    --calibration /path/to/calibration-corpus.txt

# Or specify exactly:
prismaquant-llama pipeline run \
    --budget-gb 2.5 \
    ...
```

### Priority — PPL/TG/PP weighting

The 3-digit `XYZ` priority code controls the allocator's preference. **The three digits should sum to 9** (the allocator normalizes them to ratios, so `333` and `111` mean the same thing — but using `9` as the total is the convention so all examples are directly comparable).

- **X** = PPL-quality weight (0-9)
- **Y** = TG (token generation) speed weight (0-9)
- **Z** = PP (prompt processing) speed weight (0-9)

Common combinations (all sum to 9):

| Priority | Use case |
|---|---|
| `333` | Default — equal weight, balanced |
| `522` | Quality-first — best PPL at the budget (production-grade output) |
| `252` | TG-first — fast generation (chat / streaming use) |
| `225` | PP-first — fast prompt eval (long-context use) |
| `900` | Pure-quality — completely ignore speed (the original allocator's behavior before multi-objective was added) |
| `441` | Quality + TG, minimal PP weight |
| `414` | Quality + PP, minimal TG weight |

```bash
# Quality-first run (explicit budget + priority):
prismaquant-llama pipeline run \
    --hf-model unsloth/gemma-3-4b-it \
    --binary /path/to/llama-quantize \
    --calibration /path/to/calibration-corpus.txt \
    --budget-gb 2.0 \
    --priority 522 \
    --output ~/quants/gemma-3-4b-it-prismaquant
```

### Format whitelist

The allocator default is **mainline llama.cpp formats** spanning 2-bit through 8-bit:

```
Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0,
IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL
```

> 📝 **About `Q4_K_S` / `Q4_K_M` / `Q3_K_L` etc.** — these are *whole-model presets* of the regular `llama-quantize` CLI (e.g., `Q4_K_M` = "mostly Q4_K plus Q6_K for output"). They aren't separate ggml types. Per-tensor formats are the base types only: `Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K`. prismaquant *is itself* a per-tensor mixer — it does what `_S/_M/_L` presets do, but with allocator-driven tensor-level decisions instead of fixed pinning rules. So you'd never include `Q4_K_M` in `--formats` (it's redundant with prismaquant's job). `discover --all` shows the variants annotated as CLI-only.

### Setting persistent preferences via `config.toml`

For everything *other* than the formats list — paths, default budget ratio, default priority, default binary set, calibration corpus, mmap behavior — there's a unified TOML config at:

```
~/.prismaquant-llama/config/config.toml
```

The repo ships a fully-commented example you can use as a template:

```bash
mkdir -p ~/.prismaquant-llama/config
cp examples/config.toml ~/.prismaquant-llama/config/config.toml
# edit to taste
```

Sections:
- `[paths]` — output_root, hf_cache, scratch, calibration dir
- `[binaries]` — register one or more binary sets (mainline / fork / etc.) and pick a default
- `[defaults]` — budget_auto_ratio, priority, chunks_imatrix, chunks_eval, ctx, no_mmap, calibration_corpus
- `[huggingface]` — default_revision, download_resume
- `[wizard]` — setup_complete, auto_suggest_perf, disk_warn_pct

When set, the values fill in the argparse defaults, so:
- `--binary` resolves from `[binaries.<default_set>]` (otherwise auto-discovers via `$PATH`)
- `--calibration` resolves from `[defaults] calibration_corpus` (otherwise the bundled bartowski-v3 corpus)
- `--output` resolves from `[paths] output_root` (otherwise `~/.prismaquant-llama/builds/`)
- `--budget-auto-ratio`, `--priority`, `--chunks-imatrix`, `--chunks-eval`, `--ctx`, `--no-mmap` all pick up your preferences

CLI args always win over `config.toml`. Missing keys fall through to built-in defaults — the tool always has a working fallback.

### Setting your own default formats list (per-user config)

Tired of typing `--formats X,Y,Z` on every command? Drop a file at:

```
~/.prismaquant-llama/config/default-formats.txt
```

with one format per line. Lines starting with `#` are comments. The repo ships a ready-to-use example covering wide mainline + IK-K extensions:

```bash
mkdir -p ~/.prismaquant-llama/config
cp examples/default-formats.txt ~/.prismaquant-llama/config/default-formats.txt
# (then edit to add/remove formats as needed)
```

The shipped example contains mainline llama.cpp formats only (works on any binary). If your binary is a fork (ik_llama, frankenturbo2, etc.), append the fork's extension formats — run `prismaquant-llama discover <binary>` to see what's supported.

Precedence (highest → lowest):

1. **CLI `--formats X,Y,Z`** — overrides everything for one run
2. **`~/.prismaquant-llama/config/default-formats.txt`** — your personal default
3. **Built-in default** (`Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL`) — mainline-compatible

When the user file is loaded you'll see a one-line notice in the run output:
```
[pipeline] using formats from /home/.../default-formats.txt: Q4_K,Q5_K,...
```

This keeps prismaquant-llama compatible with stock `ggml-org/llama.cpp` builds out of the box. **No fork extensions are included by default** — you opt in.

To see exactly what your binary supports + get copy-paste-ready preset strings:

```bash
prismaquant-llama discover /path/to/llama-quantize
```

The discover command emits a "Suggested --formats presets" section that lists, for your specific binary:

- **conservative** (5 mainline staples) — safe for any binary
- **wide mainline** (allocator's default) — adds 2/3-bit options
- **wide + IK-K extensions** — only shown if your binary supports them (ik_llama, frankenturbo2, etc.)

Pick the preset you want and paste the entire `--formats` string into your `pipeline run` command.

⚠️ **`IQ2_K` is excluded from all presets** — it's a documented PPL cliff (+30 PPL vs F16 across every model we've measured). Mainline 2-bit i-quants (`IQ2_S`, `IQ2_XS`, `IQ2_XXS`) are also lossy but allocator-tolerable. `IQ1_S` and `IQ1_M` are very lossy and excluded too. If you specifically need extreme compression, you can add any of these via explicit `--formats`.

### Re-running with caching

Re-running with the SAME `--hf-model` and `--output` reuses Stages A-D. Only E, F, G, H, I run. Typical re-run wall time: **~5-10 min**.

```bash
# Run #1 — auto budget
prismaquant-llama pipeline run --hf-model unsloth/gemma-3-4b-it ... -o ~/quants/gemma-3

# Run #2 — same model, larger budget — caches A-D, runs in 5 min
prismaquant-llama pipeline run --budget-gb 2.5 --hf-model unsloth/gemma-3-4b-it ... -o ~/quants/gemma-3

# Run #3 — same again, PPL-priority — caches A-D, runs in 5 min
prismaquant-llama pipeline run --priority 522 --hf-model unsloth/gemma-3-4b-it ... -o ~/quants/gemma-3
```

Each run produces a differently-named GGUF in `~/quants/gemma-3/ggufs/`, so you can compare.

---

## Step 4: Calibration — make the allocator hardware-aware

The allocator uses a **performance file** that says, for each format on your binary+hardware: how big is it, what's its PPL vs F16, and what's its prompt-processing / token-generation speed. This file lives at:

```
~/.prismaquant-llama/cache/binary-types/<binary-sha256>__chunks<N>-perf.json
```

prismaquant-llama ships with a **default** perf file (Qwen3-8B at 200 chunks, hardware-normalized ratios). It's adequate for first runs but **not optimized for your binary or hardware**.

To get a calibrated perf file for *your* setup, use `prismaquant-llama calibrate deep`:

### Calibration tiers

| Flag | Chunks | Wall time (10-12 formats, 9B model) | Reliability | Use case |
|---|---|---|---|---|
| `--quick` | 10 | ~25 min | ±0.2 PPL | Smoke test / dev iteration |
| (default) | 25 | ~50 min | ±0.13 PPL | Balanced |
| `--deep` | 50 | **~1.5-2 hr** | ±0.09 PPL | **Production per-binary perf files** |
| `--thorough` | 100 | ~3-4 hr | ±0.06 PPL | Cross-binary baseline |
| `--reference` | 200 | ~6-12 hr | ±0.04 PPL | One-time global default |

**Recommended first calibration: `--deep` on a small dense model.** A 4B-9B dense model is ideal — captures the format hierarchy without taking all weekend.

```bash
prismaquant-llama calibrate deep \
    --binary /path/to/your/llama-quantize \
    --ref-model /path/to/some-9B-BF16.gguf \
    --calibration-corpus /path/to/bartowski-calibration-v3.txt \
    --formats BF16,F16,Q4_K,Q5_K,Q6_K,Q8_0,IQ4_XS,IQ4_K,IQ4_KS,IQ4_KSS,IQ3_K,IQ3_KS \
    --deep \
    --set-as-system-default
```

The `--set-as-system-default` flag installs the result as your fallback for all future pipeline runs (any binary, any model).

### Bench-only mode (`--skip-ppl`)

If your hardware can run bench but not perplexity reliably, use `--skip-ppl` to record only pp/tg numbers:

```bash
prismaquant-llama calibrate deep \
    --skip-ppl \
    --deep \
    --binary /path/to/llama-quantize \
    ... etc
```

The output perf file has `null` ppl/Δ fields. Pair with a separate PPL-only run on a working host and merge the two files (PPL from one machine + pp/tg from another) to get a complete cross-machine perf file.

### Where calibration data lives

| File | Purpose |
|---|---|
| `~/.prismaquant-llama/cache/binary-types/<sha>__chunks<N>-calibrated.json` | Full calibration cache (resume-safe — re-runs pick up where they left off) |
| `~/.prismaquant-llama/cache/binary-types/<sha>__chunks<N>-perf.json` | Allocator-consumed format-perf subset |
| `~/.config/prismaquant-llama/system-default-format-perf.json` | System-wide default (set via `--set-as-system-default`) |

The allocator walks a 4-tier priority chain to find a perf file:

1. **Per-binary cache** — `~/.prismaquant-llama/cache/binary-types/<binary-sha>__chunks*-perf.json`
2. **`--format-perf <file>`** — explicit override per pipeline run
3. **System default** — `~/.config/prismaquant-llama/system-default-format-perf.json`
4. **Package examples** — shipped `examples/format-perf-default.json` (Qwen3-8B chunks=200)

Tier 1 always wins over Tier 4. Within Tier 1, the highest-`chunks` variant wins.

---

## Step 5: Common workflows

### Generate a sweep of GGUFs at different budgets

```bash
MODEL=unsloth/gemma-3-4b-it
BIN=/path/to/llama-quantize
CAL=/path/to/calibration-corpus.txt
OUT=~/quants/gemma-3-4b-it

for BUDGET in 1.5 2.0 2.5 3.0; do
    prismaquant-llama pipeline run \
        --hf-model $MODEL --binary $BIN --calibration $CAL --output $OUT \
        --budget-gb $BUDGET --priority 522
done
```

After: 4 GGUFs in `$OUT/ggufs/`, each named with its budget label. Stages A-D run once total; only E-I per budget. Total wall time: ~25-40 min for 4 GGUFs.

### Compare two priorities at the same budget

```bash
for PRIO in 333 522 252; do
    prismaquant-llama pipeline run \
        --hf-model $MODEL --binary $BIN --calibration $CAL --output $OUT \
        --budget-gb 2.0 --priority $PRIO
done
```

After: 3 GGUFs at the same target budget but with different format choices. Compare with `llama-perplexity` and `llama-bench` to see the trade-off.

### Skip Stage I when iterating

If you're tweaking budget/priority and just want the GGUF (no PPL eval), pass `--skip-eval`. Saves ~1-2 min per iteration.

```bash
prismaquant-llama pipeline run --skip-eval ...
```

You can always run `llama-perplexity` manually on the produced GGUF later.

### Disk hygiene

After you're satisfied with a run, you can reclaim space:

```bash
prismaquant-llama pipeline run \
    --clean-shared \      # delete _shared/hf-cache + _shared/bf16 (largest items)
    --clean-imatrix \     # also delete _shared/imatrix-cache
    --clean-probe \       # also delete _shared/probe
    ...
```

Or `--clean-all` for everything. Default is to retain `_shared/` so future runs cache-hit.

---

## Reference

### Useful subcommands

```bash
# What does my current setup look like?
prismaquant-llama paths layout --root ~/quants/some-output-dir

# Where are my llama.cpp binaries?
prismaquant-llama paths find-binaries

# What formats does my llama-quantize binary support?
# (also emits "Suggested --formats presets" footer with copy-paste-ready strings:
#  conservative / wide mainline / wide + IK-K extensions if your binary supports them)
prismaquant-llama discover /path/to/llama-quantize

# Run a calibration (see step 4 above)
prismaquant-llama calibrate deep --help
```

### Per-user config files (cheat sheet)

All in `~/.prismaquant-llama/config/`:

| File | Purpose | Format |
|---|---|---|
| `config.toml` | All persistent prefs: paths, default binary set, budget ratio, priority, calibration corpus, mmap behavior, etc. | TOML — see `examples/config.toml` |
| `default-formats.txt` | Personal default `--formats` list (line-oriented, comments OK) | One format per line |
| `system-default-format-perf.json` | System-wide perf file (set by `calibrate deep --set-as-system-default`) | JSON, tool-generated |

CLI args always win over these files. Missing keys/files fall through to built-in defaults.

### Environment variables

| Variable | Effect |
|---|---|
| `PRISMAQUANT_DEFAULT_FORMAT_PERF=/path/to/perf.json` | Override the auto-discovered perf file |
| `PRISMAQUANT_PROBE_RETAIN_CROSS_CHUNK=0` | Disable LayerCache retention across probe chunks (memory-tight hosts) |
| `PRISMAQUANT_LLAMA_ROOT=/path/to/dir` | Override the root for `~/.prismaquant-llama/` (cache + config + builds) |

### Useful flags summary

```bash
prismaquant-llama pipeline run \
    --hf-model HF_ID                         # required (safetensors repo, NOT a -GGUF repo)
    --binary PATH                            # optional; auto-discovers via $PATH (use [binaries.<default_set>] in config.toml to pin a specific build)
    --calibration PATH                       # optional; falls through CLI > [defaults] calibration_corpus > bundled bartowski-v3 corpus
    --output, -o DIR                         # optional; defaults to ~/.prismaquant-llama/builds/, override via [paths] output_root in config.toml. Per-model subdir auto-derived from --hf-model.
    --budget-gb FLOAT                        # exact target size
    --budget-auto-ratio 0.25                 # alternative: fraction of BF16
    --budget-band-gb 0.25                    # allocator wiggle room
    --priority 333                           # XYZ format
    --formats LIST                           # whitelist
    --chunks-imatrix N                       # imatrix corpus chunks (default 100)
    --chunks-eval N                          # final PPL eval chunks (default 50)
    --skip-eval                              # skip Stage I
    --format-perf PATH                       # override perf file for this run
    --clean-shared                           # clean intermediates after success
    --mmap                                   # enable mmap (default disabled)
    --dry-run                                # print plan, don't execute
```

---

## Troubleshooting

**"llama-quantize not found"** — pass `--binary /full/path/to/llama-quantize`. Or: `prismaquant-llama paths find-binaries` to see what was auto-discovered.

**"calibration corpus not found"** — should be rare since the package ships a default corpus. If you do see this, you've explicitly passed `--calibration <path>` (or set `[defaults] calibration_corpus` in `config.toml`) to a path that doesn't exist. Either fix the path or omit the flag entirely to use the bundled corpus.

**Wrong-shape HF model error during Stage B (convert step)** — usually means `--hf-model` points to a GGUF-only repo (anything with `-GGUF` in the name like `unsloth/gemma-3-4b-it-GGUF`). prismaquant-llama needs the *safetensors* base repo, not pre-quantized GGUFs. Find the un-quantized base repo (usually linked from the GGUF model card) and pass that instead. See [Picking the right HuggingFace model path](#picking-the-right-huggingface-model-path) above.

**Stage I crashes inside the GPU backend** — if your hardware/driver has trouble running perplexity on the produced GGUF (architecture-specific kernel issues etc.), pass `--skip-eval` to skip Stage I. The GGUF itself still gets produced cleanly by Stage H — you can run `llama-perplexity` separately later, or on a different host.

**OOM during Stage E (cost measurement) on a large model** — currently Stage E loads one tensor at a time, but very wide tensors (>2 GB) can spike. For now, the workaround is to run on a host with more RAM.

**"every run produces a slightly different bpw label"** — that's normal. The allocator targets your `--budget-gb` and the actual achieved bpw varies by model. The filename suffix reflects the target, not the achieved size.

**"how do I know if my calibration was good?"** — compare the Δ-PPL ranges your binary gives vs the package default. If your binary's IQ4_KS Δ-PPL is within 10% of the default, your calibration is fine. If it differs by more, your hardware behaves substantively differently — your calibration is more important.

---

## Next steps

- Explore [`docs/methodology.md`](methodology.md) for the math behind allocator surrogate scoring
- Read [`README.md`](../README.md) for the full reference
- Run `prismaquant-llama wizard` for an interactive TUI (still scaffold-stage but works for simple flows)

If you build something interesting or hit an edge case, please file an issue at https://github.com/jimbothigpen/prismaquant-llama/issues.
