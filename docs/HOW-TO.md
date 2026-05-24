# prismaquant-llama HOW-TO

Comprehensive operator and developer reference for prismaquant-llama. Covers
installation, configuration, the calibration workflow, the full nine-stage
pipeline, all four CLI subcommands, and worked examples for common patterns.

**Scope vs GETTING-STARTED.md.** [`docs/GETTING-STARTED.md`](GETTING-STARTED.md)
is a hands-on tutorial covering a first run from scratch. This document is the
exhaustive reference — every flag, every config key, every stage. Read
GETTING-STARTED first if you're new to the tool; come back here when you need
specifics.

---

## Contents

1. [Prerequisites and Install](#1-prerequisites-and-install)
2. [Configuration](#2-configuration)
3. [Calibrate](#3-calibrate)
4. [Run — the nine-stage pipeline](#4-run--the-nine-stage-pipeline)
5. [explore — sweep without producing a GGUF](#5-explore--sweep-without-producing-a-gguf)
6. [show-frontier — display Stage-K results](#6-show-frontier--display-stage-k-results)
7. [CLI reference](#7-cli-reference)
8. [TOML config reference](#8-toml-config-reference)
9. [Worked examples](#9-worked-examples)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Prerequisites and Install

### 1.1 Python and system requirements

- Python 3.10 or newer.
- A llama.cpp build with `llama-quantize`, `llama-imatrix`, `llama-perplexity`,
  and `llama-bench`. Any modern fork works (mainline ggml-org, ik_llama,
  jimbothigpen/llama.cpp). Either add the binaries to `$PATH` or set
  `path = "/your/llama/build/bin"` in the config.
- Disk space: roughly **2–2.5× the BF16 size** of the target model at peak
  (source safetensors + BF16 GGUF + final GGUF simultaneously on disk).
- GPU recommended; CPU-only works but is much slower for probe and imatrix.

### 1.2 The prismaquant-llama package

**Development (editable) install — builduser pattern:**

```bash
# From the repo root (cephfs canonical path):
pip install --break-system-packages -e \
    /mnt/cephfs/0/Container/systems/ai00/users/builduser/projects/prismaquant-llama/src/jimbothigpen/prismaquant-llama

# Or from a local clone:
git clone https://github.com/jimbothigpen/prismaquant-llama
cd prismaquant-llama
pip install --break-system-packages -e .
```

An editable install picks up Python-only source changes without reinstalling.
Re-run `pip install -e .` only when `pyproject.toml` deps change or when the
`direct_url.json` in your site-packages points at a stale path.

**Development extras** — the `[dev]` extra installs pytest and friends:

```bash
pip install -e ".[dev]"
```

**Wheel install — llamauser / production pattern:**

The llamauser venv on ai00 and ai01 lives at
`/home/llamauser/.venvs/prismaquant` (plural `.venvs`, no hyphen). Build the
wheel as builduser, then install it for llamauser:

```bash
# Build a pure-Python wheel (stage to cephfs, not /tmp):
pip wheel --no-deps -w /path/to/stage/ \
    /mnt/cephfs/0/Container/systems/ai00/users/builduser/projects/prismaquant-llama/src/jimbothigpen/prismaquant-llama

# Install into llamauser's venv on ai00:
sudo -u llamauser /home/llamauser/.venvs/prismaquant/bin/pip install \
    --force-reinstall --no-deps /path/to/stage/prismaquant_llama-*-py3-none-any.whl

# Repeat on ai01 (SSH on port 2229).
```

**Cross-user egg-info pitfall.** `pip install -e <shared-path>` as user A
writes `<pkg>.egg-info/` owned by A inside the shared source tree. User B's
`pip install <path>` then fails with a timestamp-update error even with read
access. Go through a wheel for the second user; `--no-build-isolation` does not
help.

### 1.3 Runtime deps for Stage B convert

`prismaquant-llama run` shells out to `convert_hf_to_gguf.py` (from your
llama.cpp source tree) at Stage B. That script imports `gguf`, `sentencepiece`,
and `google.protobuf`, none of which are declared deps of prismaquant-llama
itself. Install the fork-vendored versions:

```bash
pip install git+https://github.com/jimbothigpen/llama.cpp gguf-py
pip install sentencepiece protobuf
```

Use the fork's `gguf-py`, not the mainline PyPI `gguf` package — the fork's
version includes `MODEL_ARCH.EAGLE3` and other additions absent from upstream.

The test suite also needs these three packages
(`tests/test_preflight_deps.py` reports 2/3 failures when they're absent — see
[§10 Troubleshooting](#10-troubleshooting)).

### 1.4 The prismaquant Python package (Stage C)

Stage C (Hessian probe) invokes `python3 -m prismaquant.incremental_probe` as a
subprocess. This is **not a Python import** and is not declared in
`pyproject.toml`, so `grep "import prismaquant"` returns zero hits; the
dependency is invisible to static analysis. Install the jimbothigpen fork (which
carries Gemma-3, Gemma-4, and NemotronH patches absent from upstream):

```bash
pip install git+https://github.com/jimbothigpen/prismaquant.git
```

`calibrate` does **not** invoke Stage C and does not require this package. Only
`run` and `explore` touch Stage C.

### 1.5 Building llama-quantize-cost

Stage E measures per-(tensor, format) MSE using `llama-quantize-cost`, a custom
binary not in mainline llama.cpp. Clone it into your llama.cpp tree's `tools/`
directory and build:

```bash
git clone https://github.com/jimbothigpen/llama-quantize-cost \
    /path/to/your/llama.cpp/tools/quantize-cost
echo 'add_subdirectory(quantize-cost)' \
    >> /path/to/your/llama.cpp/tools/CMakeLists.txt
cmake --build /path/to/your/llama.cpp/build --target llama-quantize-cost
```

The resulting binary lands next to `llama-quantize` in your build's `bin/`
directory. It works against mainline, ik_llama, and jimbothigpen/llama.cpp
without modification. See the tool's README for a verify workflow.

**Symlink note.** Older setups may have a `tools/quantize-cost` symlink that
pointed at a local path. The canonical location on this ecosystem is:

```
/mnt/cephfs/0/Container/systems/ai00/users/builduser/projects/llama-quantize-cost/
```

If the symlink is broken, re-create it or build from the repository directly.

### 1.6 Relationship to upstream prismaquant

[RobTand/prismaquant](https://github.com/RobTand/prismaquant) is the canonical
Bayesian allocator targeting vLLM / compressed-tensors. prismaquant-llama is a
separate GGUF/llama.cpp-targeting adapter: it uses upstream's closed-form Δloss
surrogate (`½ · H_trace · MSE`) and its Lagrangian bisection but replaces
everything else with a GGUF-native toolchain. The jimbothigpen fork of
prismaquant (installed at §1.4) carries additional model architecture patches
absent from upstream; pin to a known commit when freezing a measurement baseline.

---

## 2. Configuration

### 2.1 The config file

The config file lives at `~/.prismaquant-llama/config.toml`. It is
**auto-installed** on first invocation — even `prismaquant-llama --help`
triggers the install and prints:

```
[prismaquant-llama] wrote starter config → /home/you/.prismaquant-llama/config.toml
[prismaquant-llama] edit it to set defaults; CLI flags override per run.
```

The shipped default is heavily commented and documents every key. Edit by hand.
There is no `setup` subcommand.

**Alternate configs.** Use `--config /path/to/other.toml` to point at a
different file for one invocation. This is the recommended pattern for managing
multiple llama.cpp builds (one config per fork, each with its own `path`,
`base`, and `quants`):

```bash
prismaquant-llama run unsloth/gemma-3-4b-it --config ~/configs/ik-llama.toml
```

### 2.2 Annotated complete config

```toml
# ~/.prismaquant-llama/config.toml

[prismaquant-llama]

# Working directory. All downloaded and generated artifacts land here.
# Override per-run: --base /some/path
base = "~/.prismaquant-llama/"

# llama.cpp binary directory. Empty = discover via $PATH.
# Override per-run: --path ~/llama.cpp/build/bin
path = ""

# Format whitelist — the allocator picks from this list only.
# Mainline defaults below. Fork users should append fork-specific formats
# (e.g. "IQ4_K", "IQ4_KS", "IQ3_K" for ik_llama).
# Override per-run: --quants Q4_K,Q5_K,Q6_K,Q8_0
quants = [
    "Q3_K",
    "Q4_K",
    "Q5_K",
    "Q6_K",
    "Q8_0",
    "IQ3_XXS",
    "IQ3_XS",
    "IQ3_S",
    "IQ3_M",
    "IQ4_XS",
    "IQ4_NL",
    "BF16",
]

# Reference conversion format. "bf16" (default) or "f16".
# Use "f16" if your GPU's GEMM library lacks BF16 kernels (e.g. gfx1102/1103
# under ROCm / Tensile). Field names in calibration JSON (ppl_delta_vs_f16,
# pp_ratio_vs_bf16, etc.) retain their legacy names regardless.
reference_format = "bf16"

# MTP / NEXTN tensor format. Only active when the model has MTP layers
# (num_nextn_predict_layers > 0). BF16 is safe until quantized MTP weights
# are validated for your inference stack.
mtp_format = "BF16"

# Fisher row-weighted output MSE. When true, Stage C emits h-detail blobs and
# a new sub-stage between F and E emits per-tensor Fisher sidecars for
# llama-quantize-cost. Allocator only consumes the resulting column when
# PRISMAQUANT_FISHER_OUTPUT_MSE_ALLOCATOR=1 is also set.
fisher_output_mse = false

# Target final GGUF size. Three forms accepted:
#   budget = 25         → 25% of BF16 GGUF (default)
#   budget = "4.5bpw"  → average bpw over unpinned allocator domain
#   budget = "16GB"     → absolute gigabytes
# Override per-run: --budget 30  or  --budget 4.5bpw  or  --budget 16GB
budget = 25

# Three-digit PPL/TG/PP priority. Digits are PPL weight (X), TG weight (Y),
# PP weight (Z). Only ratios matter: "111" = "333" = "999".
# Common combos:
#   "111" — balanced (default)
#   "522" — quality-first
#   "252" — token-generation-first (chat / streaming)
#   "225" — prompt-processing-first (long-context)
#   "900" — pure PPL
# Override per-run: --priority 522
priority = "111"

# PPL corpus. Empty = bundled wikitext-2-raw (~1.3 MB). Path or URL accepted.
# Override per-run: --ppl-corpus /path/or/url
ppl_corpus = ""

# imatrix corpus. Empty = bundled bartowski-imatrix-v5-semantic (~1.5 MB).
# Override per-run: --imatrix-corpus /path/or/url
imatrix_corpus = ""

# Chunks for llama-perplexity. Higher = tighter stderr, more wall time.
# ~±0.09 PPL stderr at 50. Override per-run: --ppl-chunks 100
ppl_chunks = 50

# Chunks for llama-imatrix. Higher = better activation coverage.
# Override per-run: --imatrix-chunks 200
imatrix_chunks = 50

# Path to convert_hf_to_gguf.py. Empty = auto-discover from the llama.cpp
# source tree (walks two levels up from `path`). Set explicitly when auto-
# discovery fails. Override per-run: --convert-script /path/to/script.py
convert_script = ""

# Extra directory prepended to LD_LIBRARY_PATH for all subprocess calls.
# Empty = no override. Override per-run: --libs ~/llama.cpp/build/lib
libs = ""

# ── Optional / advanced keys not shown in the shipped default ──────────────
# These keys are read by load_config with sensible defaults; add them when
# needed. They are not auto-written by the first-run installer.

# Stage K: KL/PPL-validated frontier picker. Sweeps kl_priorities at your
# budget, quantizes each candidate, runs short PPL, and picks the lowest-PPL
# recipe before Stage H. Off by default to keep existing pipelines unchanged.
# kl_validate = false
# kl_priorities = ["100", "300", "500", "700", "900"]
# kl_ppl_chunks = 20   # short chunks — just for ranking, not final eval

# Memory-loading mode for llama-imatrix (Stage D) and llama-perplexity
# (Stages I, K-ref, K-cand, calibration). Default false = OS-level mmap;
# RAM scales with VRAM+KV only, not model file size. Set to true to restore
# the old --no-mmap behavior (eager full-model load into RAM). OOMs on any
# model that does not fit in available RAM.
# imatrix_eager_load = false   # Stage D
# ppl_eager_load = false       # Stages I, K-ref, K-cand, calibration


[precondition]

# Stage F+ pre-conditioning method. "off" skips F+ (default); "awq" runs
# AWQ channel-rescale with γ-fold into the predecessor RMSNorm.
mode = "off"

# bpw floor below which F+ skips the tensor (the "disable below 4 bits" rule).
bpw_floor = 4.0
```

---

## 3. Calibrate

Calibration measures per-format PPL and throughput on a reference model and
writes a JSON perf file. The allocator reads this file to score the TG and PP
speed components of the per-tensor cost function. Without it the allocator falls
back to the shipped hardware-agnostic default (`system.json.default`), which
covers relative ratios only and may not match your binary or GPU.

Two modes, one subcommand:

```bash
prismaquant-llama calibrate system INPUT [flags]   # → calibration/system.json
prismaquant-llama calibrate model  INPUT [flags]   # → calibration/models/<name>.json
```

**Lookup priority at `run` time:** model-specific > system > shipped default.

### 3.1 When to calibrate

- **System calibration** once per (binary build, hardware) pair. Use a
  smallish dense reference model (1B–9B). This covers all future `run`
  invocations unless a model-specific file overrides it.

- **Model calibration** when a target model has unusual sensitivity or
  architecture. The calibration data is used only for that model name.

- **One-shot calibrate + run** with `run --calibrate`: runs `calibrate model`
  first, then the full pipeline. Stages A, B, and D are shared so there is no
  duplicated work. Auto-skips when a complete model.json already exists for the
  configured `quants` list. Useful for production builds where you want
  model-specific data without a separate calibration command.

### 3.2 Input forms

`calibrate` accepts all four input forms (unlike `run` which requires
safetensors — `calibrate` does not run Stage C so it does not need them):

| Form | Example |
|---|---|
| HuggingFace id | `Qwen/Qwen3-8B` |
| On-disk safetensors dir | `/data/models/qwen3-8b/` |
| On-disk f16/bf16 GGUF | `/data/ggufs/qwen3-8b-BF16.gguf` |
| URL(s) to GGUF | `https://host/model.gguf` (comma-separate split files) |

### 3.3 Calibration pipeline

1. Resolve input → BF16 GGUF on disk (download or convert as needed, reusing
   shared Stage A/B cache if present).
2. Generate imatrix once using `cfg.imatrix_corpus` — same corpus as `run`
   Stage D so calibration measurements aren't biased against i-quants and
   IK-family formats that depend on imatrix weighting.
3. For each format in `cfg.quants` (plus the reference format):
   - Quantize the reference GGUF to that format using the imatrix.
   - Run `llama-perplexity` → ppl, ppl_delta_vs_f16.
   - Run `llama-bench` → pp (t/s), tg (t/s), ratios vs the reference format.
   - Write results to the output JSON after every format (resume-safe).
   - Delete the format-specific GGUF to keep peak disk usage bounded.

### 3.4 Output layout

```
calibration/
├── system.json                  # from `calibrate system`
└── models/<sanitized-name>.json # from `calibrate model`
```

The perf JSON schema (version 4):

```json
{
  "_schema_version": 4,
  "_reference_model": "Qwen3-8B",
  "_reference_format": "BF16",
  "_calibrated_at": "2026-05-23T10:00:00+00:00",
  "_calibration_chunks": 50,
  "BF16": {
    "bpw": 16.0,
    "size_bytes": 16234567890,
    "ppl": 5.2345,
    "pp": 1234.5,
    "tg": 45.6,
    "pp_ratio_vs_bf16": 1.0,
    "tg_ratio_vs_bf16": 1.0
  },
  "Q4_K": {
    "bpw": 4.58,
    "size_bytes": 4675432100,
    "ppl": 5.6781,
    "ppl_delta_vs_f16": 0.4436,
    "pp": 1567.8,
    "tg": 52.3,
    "pp_ratio_vs_bf16": 1.27,
    "tg_ratio_vs_bf16": 1.15
  }
}
```

Keys starting with `_` are metadata. Per-format entries include the measured
`bpw`, `ppl`, `pp`, `tg`, and their ratios/deltas vs the reference format.
`ppl_delta_vs_f16` is absent for the reference entry itself.

### 3.5 Resume behavior

Calibration is resume-safe: the JSON is written after every format. If the run
is killed mid-sweep, re-invoke the same command — already-complete formats are
detected and skipped. A format entry is "complete" when all of `bpw`, `ppl`,
`pp`, `tg` are present and `error` is absent.

### 3.6 Live logs

```bash
# Meta progress (which format / which step):
tail -f ~/.prismaquant-llama/work/<run>/logs/calibrate.log

# Live subprocess output for one format:
tail -f ~/.prismaquant-llama/work/<run>/logs/calibrate-Q4_K.log
```

---

## 4. Run — the nine-stage pipeline

### 4.1 Overview

`prismaquant-llama run` executes nine stages (A through I), with two optional
extras (Stage F+ and Stage K). Each stage is idempotent and caches by file
existence; re-running with a different `--budget` or `--priority` skips A–E and
re-runs only F–I (~5–10 min on most models).

```bash
prismaquant-llama run INPUT [flags]
```

INPUT must be a HuggingFace id or an on-disk safetensors directory. Stage C
(Hessian probe) requires safetensors; on-disk GGUF inputs are not accepted.

### 4.2 Stage table

| Stage | Purpose | Tool / Script |
|---|---|---|
| A | Download HF safetensors | `huggingface_hub.snapshot_download` |
| B | Convert safetensors → BF16 GGUF | `convert_hf_to_gguf.py` |
| C | Hessian probe (Fisher trace) | `prismaquant.incremental_probe` |
| D | imatrix generation | `llama-imatrix` |
| E | Per-(tensor, format) MSE costs | `llama-quantize-cost` |
| F | Bridge HF → GGUF tensor names | bundled `bridge_probe_to_gguf.py` |
| G | Multi-choice knapsack allocation | bundled `allocator.py` |
| F+ | (optional) Pre-condition BF16 weights | bundled `precondition.py` |
| H | Apply recipe | `llama-quantize --tensor-type-file` |
| I | Final PPL eval | `llama-perplexity` |
| K | (optional) KL/PPL-validated frontier | `llama-quantize` + `llama-perplexity` |

Stage K runs between G and H when `kl_validate = true`. Stage F+ runs between G
(or K) and H when `[precondition].mode != "off"`. Stage D may re-run as D′
after F+ if F+ folded weights (so the imatrix reflects the updated magnitudes).

### 4.3 Stage A — HF download

Caches under `_shared/hf-cache/<model>/` with a `.download.complete` marker.
Skip condition: the marker exists. Only runs when INPUT is an HF id; on-disk
safetensors directories skip straight to Stage B.

Stage A uses `huggingface_hub.snapshot_download`. Gated models (Llama, Gemma,
etc.) require `hf auth login` and browser-side license acceptance.

### 4.4 Stage B — BF16 conversion

Converts the safetensors directory to a BF16 GGUF (or F16 if `reference_format
= "f16"`) using `convert_hf_to_gguf.py` from the llama.cpp source tree.

Cache key: output file existence at `_shared/bf16/<model>-BF16.gguf`. The
filename always reflects the reference format (e.g. `...-F16.gguf` when
`reference_format = "f16"`).

`convert_hf_to_gguf.py` is **not installed by `cmake --install`**; it lives at
the root of the llama.cpp source tree. Auto-discovery walks two levels up from
the `llama-quantize` binary path, then checks next to the binary, then `$PATH`.
Set `convert_script` in config or pass `--convert-script` if auto-discovery
fails.

### 4.5 Stage C — Hessian probe

Runs `python3 -m prismaquant.incremental_probe` on the safetensors directory.
One calibration forward pass produces the per-Linear Fisher trace (`H_trace`)
— the empirical measure of how much the loss moves when each tensor's weights
change.

Cache key: `_shared/probe/<model>-probe.pkl`. Also writes activation tensors to
`_shared/probe/act-cache/`. When `fisher_output_mse = true`, also writes
per-Linear h-detail blobs to `_shared/probe/h-detail/` (roughly doubles probe
output disk; re-probe is triggered the first time you toggle the flag).

**Act-cache trap.** The act-cache filenames are per-Linear-name only
(e.g. `model__layers__0__mlp__gate_proj.pt`) — two models sharing the same
layer name structure (e.g. Qwen3.5-4B and Qwen3.5-9B) would overwrite each
other's act-cache files in a shared probe dir. Avoid probing two same-arch
models to the same cache dir simultaneously; delete stale `.pt` files if you
see `scale length X ≠ in_features Y ... likely a stale act-cache` errors.

### 4.6 Stage D — imatrix generation

Generates the importance matrix using `llama-imatrix` on the BF16 GGUF.

Cache key: `_shared/imatrix-cache/<model-sha>__<corpus-sha>__c<chunks>.imatrix.gguf`.
Different (`--imatrix-chunks`, corpus, or BF16 content) → different cache entry,
so changing chunk count invalidates cleanly without manual removal.

**RAM behavior (since 2026-05-22).** `llama-imatrix` uses OS-level mmap by
default (streaming; RAM bounded by VRAM + KV cache). Set `imatrix_eager_load =
true` to restore the pre-2026-05-22 `--no-mmap` behavior (full model eagerly
loaded into RAM before computation). Use for small models that fit in RAM when
predictable timing matters; OOMs on any model whose BF16 size exceeds available
RAM.

Override the generated imatrix with `--imatrix /path/or/url` to skip Stage D
entirely and use an existing file.

### 4.7 Stage E — per-(tensor, format) MSE costs

Runs `llama-quantize-cost` on a representative subset of layers (exemplar
layers auto-detected, default 0 and 3 for hybrid architectures), then
propagates costs to peer layers by architectural type. Outputs a CSV.

Cache key: `_shared/costs-cache/<bf16-sha>__<imatrix-sha>__<formats-hash>.csv`.
An atomic write (`.tmp` renamed on success) prevents partial CSVs from being
treated as complete on a re-run after a kill.

**Format whitelist**: costs are measured only for formats in `cfg.quants`.
Expanding the whitelist (e.g. adding fork-specific formats) invalidates the
cache because the `formats-hash` changes.

**Fisher sidecar mode.** When `fisher_output_mse = true`, Stage E-pre emits
per-tensor Fisher sidecar `.bin` files before the main cost measurement.
`llama-quantize-cost` is then invoked with `--fisher-sidecar` and writes a
`fisher_output_mse` column alongside `mse`. The allocator uses the
`fisher_output_mse` column only when `PRISMAQUANT_FISHER_OUTPUT_MSE_ALLOCATOR=1`
is set in the environment.

### 4.8 Stage F — HF→GGUF tensor name bridge

Runs `bridge_probe_to_gguf.py`, which maps HuggingFace Linear tensor names (as
used in the probe) to GGUF tensor names (as used in `llama-quantize`). Outputs
`work/<run>/bridge.json` (Fisher trace keyed by GGUF tensor name) and
optionally `work/<run>/mtp-tensors.json` (list of GGUF tensor names belonging to
MTP/NEXTN layers).

Cache key: `bridge.json` file existence in the run directory.

### 4.9 Stage G — multi-choice knapsack allocation

Runs the bundled `allocator.py`, which solves:

```
fmt[t] = argmin_f ( w_PPL · ĉ_ppl + w_TG · ĉ_tg + w_PP · ĉ_pp + λ · size )
```

per-tensor independently for each Lagrange multiplier λ, then bisects λ to find
the value whose total size lands at or below the budget (within a 0.25 GB
tolerance band). ~50 bisection iterations; runtime is sub-second.

The priority string XYZ maps to weights as: **X = PPL weight, Y = TG weight,
Z = PP weight**. (Note: some earlier documentation inverted Y/Z; the code in
`parse_priority` and `config.toml.default` are authoritative.)

Hard pins applied before allocation:
- `output.weight` → `Q6_K`
- `token_embd.weight` → `Q8_0`

Attention weight floor: any tensor matching
`^blk\..*\.attn_(q|k|v|qkv|gate|output)\.weight$` is forced to ≥ 4.0 bpw.

MTP tensors (when `mtp_tensors.json` is present) are pinned to `cfg.mtp_format`
and excluded from the DP budget.

Outputs:
- `work/<run>/recipes/recipe-PQ<bpw>-<priority>.json` — the allocation.
- `work/<run>/recipes/recipe-PQ<bpw>-<priority>.txt` — `tensor=format` lines
  fed directly to `llama-quantize --tensor-type-file`.

`<bpw>` is the target bits-per-weight (2 decimals, trailing zeros stripped):
e.g. `PQ4.5` (from `--budget 4.5bpw`), `PQ4.85` (from `--budget 25`,
derived from model domain parameters after Stage E).

**recipe.json schema:**

```json
{
  "budget_gb": 2.45,
  "budget_input": "4.5bpw",
  "target_bpw": 4.5,
  "actual_size_gb": 2.43,
  "loss_surrogate": 0.003412,
  "lambda": 1.23e-5,
  "priority": "111",
  "weights": [0.333, 0.333, 0.333],
  "format_counts": {"Q4_K": 142, "Q5_K": 31, "Q8_0": 8, "Q6_K": 4},
  "recipe": {
    "blk.0.attn_q.weight": "Q5_K",
    "blk.0.attn_k.weight": "Q5_K",
    "blk.0.ffn_gate.weight": "Q4_K",
    "output.weight": "Q6_K",
    "token_embd.weight": "Q8_0"
  }
}
```

The per-tensor assignment map is under the **`recipe`** key, not at the top
level. Consumers reading the JSON directly must use
`data.get("recipe") or data.get("assignments") or data` for compatibility.

**Saturated-λ behavior.** If the budget is below the floor-respecting minimum
(e.g. budget 22% when attention floors push the minimum to ~26%), λ saturates
the positive-λ loop and the score collapses to `λ·size` — the allocator picks
the smallest-bpw format that satisfies all floor constraints. The `explore`
subcommand surfaces this as all cells collapsing to the same recipe.

### 4.10 Stage K — KL/PPL-validated frontier picker (optional)

Enabled by `kl_validate = true` in config. Runs between Stage G and Stage H.

For each priority in `cfg.kl_priorities` (default `["100","300","500","700","900"]`
plus the user's own priority), Stage K:
1. Runs the allocator at that priority to produce a candidate recipe.
2. De-duplicates by recipe SHA (inner `recipe` dict, not the full file — see
   §4.9 note on saturated-λ collisions).
3. For unique recipes: quantizes with `llama-quantize`, runs a short PPL pass
   (`cfg.kl_ppl_chunks`, default 20 chunks).
4. Tags each candidate with `is_pareto: bool` and selects the lowest-PPL recipe
   as the winner.
5. Optionally runs one BF16 reference PPL pass (`_stage_k_reference_ppl`) for
   the `ppl_diff` overlay in `show-frontier`.

Outputs: `work/<run>/stage-k/summary-PQ<bpw>.json` (schema version 3),
where `<bpw>` is the canonical target bpw (e.g. `PQ4.5`, `PQ4.85`).

The `show-frontier` subcommand re-renders these summaries without re-running
anything.

### 4.11 Stage F+ — pre-conditioning (optional)

Enabled by `[precondition].mode = "awq"` in config or `--precondition awq` on
the CLI. Runs between Stage G/K and Stage H.

Stage F+ applies activation-aware AWQ channel rescale (γ-fold into the
predecessor RMSNorm) to BF16 weights of tensors whose recipe-chosen format is
≥ `bpw_floor` (default 4.0). Tensors assigned to IQ3 / Q3 / etc. pass through
unchanged — the "disable below 4 bits" rule.

When F+ runs and folds at least one tensor, Stage D re-runs on the
pre-conditioned BF16 GGUF (D′) so the imatrix used by Stage H reflects the
rescaled weight magnitudes.

### 4.12 Stage H — apply recipe

Runs `llama-quantize --imatrix <imatrix> --tensor-type-file <recipe.txt>
<bf16.gguf> <out.gguf> Q4_K`. The `Q4_K` positional fallback is overridden per
tensor by the `--tensor-type-file` assignment, so the output is the full
mixed-format recipe.

Cache key: a sidecar `<gguf>.cachekey.json` recording SHA-256 of the BF16 GGUF,
imatrix, and recipe.txt. Cache hits skip quantize entirely. Mismatches (e.g.
after a pre-conditioning re-run or a llama.cpp bugfix) trigger a rebuild.

Output: `ggufs/<model>-PQ<bpw>-<priority>.gguf`.

### 4.13 Stage I — final PPL eval

Runs `llama-perplexity` on the final GGUF and prints the PPL to stdout and the
run-complete banner. Not cached — it always re-runs if the stage is reached (the
GGUF is already cached by Stage H so repeated `run` invocations pay Stage I
time even if H is a cache hit).

**RAM behavior.** Same as Stage D: mmap by default since 2026-05-22; set
`ppl_eager_load = true` to restore `--no-mmap`.

Stage I failure (e.g. GPU backend crash after a successful Stage H) does not
delete the final GGUF. The GGUF is usable; re-run `llama-perplexity` manually
if you need the number.

### 4.14 Caching summary

| Stage | Cache key | Artifact |
|---|---|---|
| A | `.download.complete` marker | `_shared/hf-cache/<model>/` |
| B | output GGUF file existence | `_shared/bf16/<model>-BF16.gguf` |
| C | `<model>-probe.pkl` file existence | `_shared/probe/` |
| D | `<model-sha>__<corpus-sha>__c<N>.imatrix.gguf` | `_shared/imatrix-cache/` |
| E | `<bf16-sha>__<imatrix-sha>__<fmt-hash>.csv` | `_shared/costs-cache/` |
| F | `bridge.json` file existence | `work/<run>/` |
| G | `recipe-PQ<bpw>-<P>.json` file existence | `work/<run>/recipes/` |
| H | `<gguf>.cachekey.json` SHA match | `ggufs/` |

Re-running with a different budget/priority (F and G invalidate): ~5–10 min.
Re-running after a llama.cpp upgrade: use `--force` to wipe all artifacts
associated with the model and start from scratch.

### 4.15 Output layout

```
{base}/                               (default: ~/.prismaquant-llama/)
├── _shared/
│   ├── hf-cache/<model>/             Stage A: HF safetensors
│   ├── bf16/<model>-BF16.gguf        Stage B: reference GGUF
│   ├── imatrix-cache/                Stage D: keyed imatrix files
│   ├── probe/<model>-probe.pkl       Stage C: Fisher probe
│   └── costs-cache/                  Stage E: keyed costs.csv files
├── ppl-corpus/                       downloaded PPL corpora (URL inputs)
├── imatrix-corpus/                   downloaded imatrix corpora (URL inputs)
├── calibration/
│   ├── system.json                   from `calibrate system`
│   └── models/<model>.json           from `calibrate model`
├── ggufs/
│   └── <model>-PQ<bpw>-<priority>.gguf      final output
└── work/<model>-<timestamp>/         per-run scratch
    ├── costs/costs.csv               Stage E
    ├── bridge.json                   Stage F
    ├── mtp-tensors.json              Stage F (MTP models only)
    ├── recipes/
    │   ├── recipe-PQ<bpw>-<P>.json   Stage G
    │   └── recipe-PQ<bpw>-<P>.txt    Stage G (fed to llama-quantize)
    ├── stage-k/                      Stage K (when kl_validate=true)
    │   └── summary-PQ<bpw>.json
    └── logs/                         all subprocess output
        ├── stage-A.log
        ├── stage-B.log
        ...
        └── stage-I.log
```

Filename convention: `<model>-PQ<bpw>-<priority>.gguf`, where `<bpw>` is the
target bits-per-weight over the unpinned allocator domain (2 decimal places,
trailing zeros stripped). For `--budget 4.5bpw` the label is `PQ4.5` exactly;
for `--budget 25` or `--budget 16GB` the bpw is derived after Stage E.
Example: `gemma-3-4b-it-PQ4.85-111.gguf` (from `--budget 25`).

### 4.16 Purge

`--purge yes` (default): deletes what this run downloaded or generated at the
end, except final GGUFs and per-run logs. Never touches user-supplied on-disk
inputs (safetensors directories, GGUF files, corpus files provided by path).

Specifically removes when `--purge yes` and INPUT was an HF id or URL:
- `_shared/hf-cache/<model>/`
- `_shared/bf16/<model>-BF16.gguf`
- `_shared/imatrix-cache/<model>*`
- `_shared/probe/<model>*`
- `_shared/costs-cache/<bf16-sha>*` for this run's costs CSV

`--purge no`: keep everything for re-use by subsequent runs. Use when sweeping
multiple budgets/priorities on the same model to avoid re-downloading.

---

## 5. explore — sweep without producing a GGUF

`explore` runs Stages A–F (cached if present, sharing all intermediates with
any prior `run` on the same model), then sweeps the cartesian product of one or
more budgets × priorities through the allocator. No GGUF is produced and no PPL
eval runs.

```bash
prismaquant-llama explore INPUT \
    --budgets 22,25,28,32 \
    --priorities 111,522,252,225,323
```

Output: a Markdown table on stdout (plus optional CSV/Markdown files) with
one row per (budget, priority) cell, showing actual size, predicted ΔPPL,
predicted TG, predicted PP, and the top format counts from the resulting recipe.

Predicted metrics use the calibration perf file (model-specific > system >
shipped) via a size-weighted aggregation over per-format `ppl_delta_vs_f16` /
`tg` / `pp`. This is an approximation — it doesn't weight by per-tensor
Fisher sensitivity — but it surfaces backend-specific quality regressions
directly: e.g. on Vulkan, a format with a known quality regression shows up as a
large predicted ΔPPL on cells where the allocator picks it heavily.

Use `explore` to pick a `(budget, priority)` pair before committing to a full
`run`. The `--from-explore` flag on `show-frontier` can then compare the
simulator predictions against measured Stage-K values.

---

## 6. show-frontier — display Stage-K results

Stage K writes one `summary-PQ<bpw>.json` per run (when `kl_validate =
true`). `show-frontier` re-renders these summaries without re-running anything:

```bash
# Latest run, all budgets:
prismaquant-llama show-frontier unsloth/gemma-3-4b-it

# Filter to one budget:
prismaquant-llama show-frontier Qwen3.5-4B --budget 25

# All historical runs:
prismaquant-llama show-frontier Qwen3.5-4B --all-runs

# Machine-readable outputs (combinable):
prismaquant-llama show-frontier Qwen3.5-4B \
    --output-csv frontier.csv \
    --output-json frontier.json \
    --output-md frontier.md

# Compare explore predictions vs measured Stage-K:
prismaquant-llama show-frontier Qwen3.5-4B \
    --from-explore sweep.csv
```

INPUT accepts any form that resolves to a model name: an HF id, safetensors
dir, GGUF path, or the bare sanitized model name (e.g. `Qwen3.5-4B`). The
input does not need to exist on disk; only the historical `work/<run>/` directory
needs to be present.

Stdout always renders a text table grouped by run, with `*` marking Pareto
candidates and `★` marking the winner (lowest PPL).

`--from-explore PATH` attaches simulator-predicted size and ΔPPL from a prior
`explore --output-csv` alongside the measured values. When the summary carries
`reference_ppl_f16` (schema version 3), a `ppl_diff = measured − ref − pred_ΔPPL`
column also appears, quantifying how well the simulator predicted quality.

---

## 7. CLI reference

### 7.1 Top-level

```
prismaquant-llama [--version | -V]
prismaquant-llama [--help | -h | help]
prismaquant-llama run       INPUT [flags]
prismaquant-llama calibrate {system|model} INPUT [flags]
prismaquant-llama explore   INPUT [flags]
prismaquant-llama show-frontier INPUT [flags]
```

The starter config auto-install triggers on any invocation not using `--config`.

### 7.2 `run` flags

| Flag | Default | Description |
|---|---|---|
| `INPUT` | *(required)* | HF id or on-disk safetensors directory |
| `--config PATH` | `~/.prismaquant-llama/config.toml` | Alternate config file |
| `--libs DIR` | (none) | Prepend to `LD_LIBRARY_PATH` for all subprocesses |
| `--base DIR` | from config | Working directory |
| `--path DIR` | from config | llama.cpp binary directory |
| `--quants Q,...` | from config | Comma-separated format whitelist |
| `--budget SPEC` | from config | Target size: `25` or `25%` (% of BF16), `4.5bpw` (bpw), `16GB` (absolute) |
| `--priority XYZ` | from config | 3-digit PPL/TG/PP ratio |
| `--ppl-corpus PATH\|URL` | from config | PPL corpus (empty = bundled wikitext) |
| `--imatrix-corpus PATH\|URL` | from config | imatrix corpus (empty = bundled bartowski-v5) |
| `--imatrix PATH\|URL` | (generate) | Existing imatrix file; skips Stage D |
| `--ppl-chunks N` | from config | Chunks for `llama-perplexity` |
| `--imatrix-chunks N` | from config | Chunks for `llama-imatrix` |
| `--convert-script PATH` | auto-discover | Path to `convert_hf_to_gguf.py` |
| `--purge {yes,no}` | `yes` | Clean up artifacts after run |
| `--yes` / `-y` | false | Skip pre-flight confirmation (required for scripts) |
| `--calibrate` | false | Run `calibrate model` first (skipped if already complete) |
| `--calibrate-chunks N` | from `--ppl-chunks` | Chunks override for calibration step only |
| `--force` | false | Delete all prior artifacts for this model and recompute from scratch |
| `--precondition {off,awq}` | from config | Stage F+ method stack |
| `--precondition-bpw-floor FLOAT` | from config | bpw floor for Stage F+ |

**Example:**

```bash
prismaquant-llama run unsloth/gemma-3-4b-it \
    --budget 28 --priority 522 \
    --imatrix-chunks 200 --ppl-chunks 100 \
    --purge no --yes
```

### 7.3 `calibrate` flags

```
prismaquant-llama calibrate {system|model} INPUT [flags]
```

| Flag | Default | Description |
|---|---|---|
| `{system,model}` | *(required)* | Write system.json or models/<name>.json |
| `INPUT` | *(required)* | HF id, safetensors dir, on-disk GGUF, or GGUF URL(s) |
| `--config PATH` | `~/.prismaquant-llama/config.toml` | Alternate config |
| `--libs DIR` | (none) | Prepend to `LD_LIBRARY_PATH` |
| `--base DIR` | from config | Working directory |
| `--path DIR` | from config | llama.cpp binary directory |
| `--quants Q,...` | from config | Format whitelist |
| `--ppl-corpus PATH\|URL` | from config | PPL corpus |
| `--ppl-chunks N` | from config | Chunks for `llama-perplexity` |
| `--imatrix-corpus PATH\|URL` | from config | imatrix corpus |
| `--imatrix-chunks N` | from config | Chunks for `llama-imatrix` |
| `--imatrix PATH\|URL` | (generate) | Existing imatrix file |
| `--convert-script PATH` | auto-discover | Path to `convert_hf_to_gguf.py` |
| `--purge {yes,no}` | `yes` | Clean up after calibration |
| `--yes` / `-y` | false | Skip pre-flight confirmation |
| `--force` | false | Delete prior calibration artifacts and recompute |

### 7.4 `explore` flags

```
prismaquant-llama explore INPUT [flags]
```

| Flag | Default | Description |
|---|---|---|
| `INPUT` | *(required)* | HF id or on-disk safetensors directory |
| `--budgets SPEC,...` | `22,25,28,32` | Comma-separated budgets (all must use the same unit form) |
| `--priorities P,...` | `111,522,252,225,323` | Comma-separated priority specs |
| `--config PATH` | `~/.prismaquant-llama/config.toml` | Alternate config |
| `--libs DIR` | (none) | Prepend to `LD_LIBRARY_PATH` |
| `--base DIR` | from config | Working directory |
| `--path DIR` | from config | llama.cpp binary directory |
| `--quants Q,...` | from config | Format whitelist |
| `--ppl-corpus PATH\|URL` | from config | PPL corpus |
| `--imatrix-corpus PATH\|URL` | from config | imatrix corpus |
| `--imatrix PATH\|URL` | (generate) | Existing imatrix file |
| `--ppl-chunks N` | from config | Chunks (used by Stage C/D; not Stage I — explore skips I) |
| `--imatrix-chunks N` | from config | Chunks for `llama-imatrix` |
| `--convert-script PATH` | auto-discover | Path to `convert_hf_to_gguf.py` |
| `--output-csv PATH` | (none) | Also write sweep results as CSV |
| `--output-md PATH` | (none) | Also write sweep results as Markdown |
| `--yes` / `-y` | false | Skip pre-flight confirmation |

### 7.5 `show-frontier` flags

```
prismaquant-llama show-frontier INPUT [flags]
```

| Flag | Default | Description |
|---|---|---|
| `INPUT` | *(required)* | Model name, HF id, safetensors dir, or GGUF path |
| `--config PATH` | `~/.prismaquant-llama/config.toml` | Alternate config |
| `--base DIR` | from config | Base directory to search for `work/<run>/` |
| `--budget SPEC` | (all) | Restrict to one PQ budget (e.g. `25`, `4.5bpw`, `16GB`) |
| `--run LABEL` | (latest) | Exact run label (e.g. `Qwen3.5-4B-20260515-103000`) |
| `--all-runs` | false | Print every run, not just the latest |
| `--output-csv PATH` | (none) | Write one-row-per-candidate CSV |
| `--output-json PATH` | (none) | Write aggregated frontiers as JSON |
| `--output-md PATH` | (none) | Write Markdown document |
| `--from-explore PATH` | (none) | Attach explore CSV predictions (join on budget_label + priority) |

---

## 8. TOML config reference

### 8.1 `[prismaquant-llama]` section

All keys have CLI flag overrides; see §7 for flag names.

| Key | Type | Default | Description |
|---|---|---|---|
| `base` | string (path) | `"~/.prismaquant-llama/"` | Working directory root |
| `path` | string (path) | `""` | llama.cpp binary dir; empty = `$PATH` |
| `quants` | list of strings | 12 mainline formats | Format whitelist for allocator and cost measurement |
| `reference_format` | string | `"bf16"` | `"bf16"` or `"f16"`; controls Stage B outtype and calibration reference |
| `mtp_format` | string | `"BF16"` | Format for MTP/NEXTN tensors; only active for MTP models |
| `fisher_output_mse` | bool | `false` | Enable Fisher row-weighted output MSE path in Stage E |
| `budget` | int or string | `25` | Target size: int/float → % of BF16; `"4.5bpw"` → bpw; `"16GB"` → absolute GB |
| `priority` | string | `"111"` | 3-digit PPL/TG/PP weights (X=PPL, Y=TG, Z=PP) |
| `ppl_corpus` | string | `""` | PPL corpus; empty = bundled wikitext-2-raw |
| `imatrix_corpus` | string | `""` | imatrix corpus; empty = bundled bartowski-v5 |
| `ppl_chunks` | int | `50` | Chunks for `llama-perplexity` (≥1) |
| `imatrix_chunks` | int | `50` | Chunks for `llama-imatrix` (≥1) |
| `convert_script` | string (path) | `""` | Path to `convert_hf_to_gguf.py`; empty = auto-discover |
| `libs` | string (path) | `""` | Extra dir prepended to `LD_LIBRARY_PATH`; empty = no override |
| `kl_validate` | bool | `false` | Enable Stage K (KL/PPL-validated frontier picker) |
| `kl_priorities` | list of strings | `["100","300","500","700","900"]` | Priorities swept by Stage K |
| `kl_ppl_chunks` | int | `20` | Chunks for Stage K PPL passes (short; ranking only) |
| `imatrix_eager_load` | bool | `false` | Pass `--no-mmap` to `llama-imatrix` (Stage D); eager full-model RAM load |
| `ppl_eager_load` | bool | `false` | Pass `--no-mmap` to `llama-perplexity` (Stages I, K-ref, K-cand, calibration) |

Keys `kl_validate`, `kl_priorities`, `kl_ppl_chunks`, `imatrix_eager_load`, and
`ppl_eager_load` are **not written** by the first-run installer; add them
manually when needed.

### 8.2 `[precondition]` section

| Key | Type | Default | Description |
|---|---|---|---|
| `mode` | string | `"off"` | Stage F+ method: `"off"` (skip) or `"awq"` (AWQ channel rescale) |
| `bpw_floor` | float | `4.0` | Skip F+ for tensors below this bpw; the "disable below 4 bits" cutoff |

---

## 9. Worked examples

### 9.1 Tight budget — maximum compression

**Goal:** minimize file size; accept quality loss.

```bash
prismaquant-llama run unsloth/gemma-3-4b-it \
    --budget 20 \
    --priority 111 \
    --yes
```

At 20% of BF16 (below the IQ3_XXS range for most models), the allocator
operates near the budget-feasibility floor. High-Fisher tensors (early attention
layers, `token_embd`, `output`) receive 4–5 bit formats while the bulk of
mid-depth FFN tensors fall to IQ3_M / IQ3_XS. The recipe concentrates
precision where the loss surrogate says it matters most.

Expected recipe shape: dominated by IQ3-family formats; `output.weight` pinned
to Q6_K and `token_embd.weight` pinned to Q8_0 regardless. If the budget is
below the attention-floor minimum (attention tensors must be ≥ 4.0 bpw), the
allocator saturates and collapses to the lowest-bpw-above-floor recipe for
attention tensors across all priorities — `explore` will show this as all cells
producing the same recipe.

Trade-off: smallest possible file size; PPL degradation is qualitatively
larger than at moderate budgets because the allocator has less room to protect
high-Fisher tensors.

### 9.2 Loose budget — quality-first production build

**Goal:** quality close to BF16 at a still-practical size.

```bash
prismaquant-llama run unsloth/gemma-3-4b-it \
    --budget 80 \
    --priority 900 \
    --imatrix-chunks 200 \
    --ppl-chunks 100 \
    --yes
```

At 80% of BF16 with pure-PPL priority (`900`), the allocator can assign Q8_0
or even BF16 to the most sensitive tensors, with Q5_K / Q6_K for the remainder.
The recipe approaches the behavior of a hand-tuned high-quality uniform quant.

At loose budgets, prismaquant's advantage over a well-designed uniform quant
narrows — the mixed-precision signal is smaller when no tensor needs to be
starved. Use `--imatrix-chunks 200` and `--ppl-chunks 100` to get tight stderr
on the final eval number when quality precision matters.

Trade-off: near-BF16 quality at a fraction of the storage; the pipeline cost
(probe + costs) is the same regardless of budget.

### 9.3 PPL-weighted — prioritize decode quality

**Goal:** minimize perplexity; throughput is secondary.

```bash
prismaquant-llama run unsloth/gemma-3-4b-it \
    --budget 25 \
    --priority 522 \
    --yes
```

Priority `522` sets PPL:TG:PP = 5:2:2. The allocator's cost function favors
formats with low Δloss even when their TG or PP throughput is lower. On
calibrated hardware, this typically shifts several tensors up one format tier
relative to `111` — e.g. Q5_K where `111` would have picked Q4_K — accepting
slightly slower token generation in exchange for PPL closer to BF16.

Qualitative direction: fewer IQ3 / Q3 tensors; more Q5_K / Q6_K in the recipe
at the same budget. The size comes in near the `--budget 25` target since the
allocator still respects the budget constraint; what changes is which tensors get
the headroom.

### 9.4 TG-weighted — token-generation throughput

**Goal:** maximize token generation speed without wrecking quality.

```bash
prismaquant-llama run unsloth/gemma-3-4b-it \
    --budget 25 \
    --priority 252 \
    --yes
```

Priority `252` sets PPL:TG:PP = 2:5:2. The TG term penalizes formats with high
dequantization latency on your specific binary and hardware (from the
calibration's `tg` column). Formats that are fast to dequantize win even when
their MSE is slightly higher.

Practical effect depends heavily on hardware: on CUDA backends the throughput
differences between K-quant tiers are small; on some ROCm or CPU backends the
spread is larger and this priority produces a noticeably different recipe.
Without system calibration (`calibrate system`) the TG scores fall back to the
shipped hardware-agnostic ratios, which may not reflect your hardware.

### 9.5 PP-weighted — prompt-processing throughput

**Goal:** maximize prefill speed for long-context workloads.

```bash
prismaquant-llama run unsloth/gemma-3-4b-it \
    --budget 25 \
    --priority 225 \
    --yes
```

Priority `225` sets PPL:TG:PP = 2:2:5. The PP term penalizes formats whose
matrix-multiply throughput (the `pp` column from calibration) is relatively
low. Formats that map to efficient GEMM shapes win.

Qualitative effect: similar to TG-weighted but with attention on the
prompt-processing pass rather than the decode pass. For most hardware the two
priorities produce similar recipes; they diverge more on hardware where
prefill and decode bottlenecks differ (e.g. CPU backends, very large batch
sizes).

### 9.6 Mixed weighting + explore before committing

**Goal:** balance quality and throughput; explore the space before running.

```bash
# Sweep (budgets × priorities) without producing any GGUF:
prismaquant-llama explore unsloth/gemma-3-4b-it \
    --budgets 20,25,28,32 \
    --priorities 111,522,252,225,500,323 \
    --output-csv sweep.csv \
    --output-md sweep.md \
    --yes

# Inspect the table; pick e.g. budget=28, priority=323:
prismaquant-llama run unsloth/gemma-3-4b-it \
    --budget 28 \
    --priority 323 \
    --yes
```

Priority `323` sets PPL:TG:PP = 3:2:3 — quality and prefill are equally
prioritized, with TG de-emphasized. The `explore` matrix shows predicted ΔPPL
and throughput for every cell so you can compare before committing hours of
compute to a `run`.

The `explore` output's `top_formats` column shows which formats dominate the
recipe in each cell. If a cell shows an unexpected format (e.g. IQ4_KSS-heavy
when your hardware has a known regression for that format), the explore matrix
surfaces it via the predicted ΔPPL before you quantize.

After running with `kl_validate = true`, compare simulator predictions against
measurements:

```bash
prismaquant-llama show-frontier unsloth/gemma-3-4b-it \
    --from-explore sweep.csv
```

### 9.7 Budget by bits-per-weight

**Goal:** target a specific average bpw rather than a % of BF16.

```bash
prismaquant-llama run unsloth/gemma-3-4b-it \
    --budget 4.5bpw \
    --priority 111 \
    --yes
```

The bpw budget is resolved after Stage E (costs.csv) using the allocator
domain: `pinned_bytes + bpw × unpinned_params / 8`. Hard-pinned tensors
(`output.weight` at Q6_K, `token_embd.weight` at Q8_0) count at their
fixed sizes; only the remaining free tensors are summed. The resulting
`budget_gb` is then used exactly like a GB-form budget.

Output filename: `gemma-3-4b-it-PQ4.5-111.gguf`.

`explore` accepts bpw budgets too, but all budgets in one sweep must use
the same unit form (mixed units raise a clear error):

```bash
prismaquant-llama explore unsloth/gemma-3-4b-it \
    --budgets 3bpw,3.5bpw,4bpw,4.5bpw \
    --priorities 111,522 \
    --yes
```

### 9.8 Budget by absolute GB

**Goal:** produce a GGUF that fits in exactly N GB (e.g. to fill a VRAM
budget to the nearest round number).

```bash
prismaquant-llama run unsloth/gemma-3-4b-it \
    --budget 4GB \
    --priority 111 \
    --yes
```

The GB value must be ≤ the BF16 GGUF size or the run aborts immediately
after Stage B with a clear error message. The pipeline uses the exact GB
value as `budget_gb`; bisection finds the λ whose recipe total lands at
or below that target.

Output filename: e.g. `gemma-3-4b-it-PQ4.48-111.gguf` (exact bpw derived
from the 4 GB budget and model domain parameters after Stage E).

---

## 10. Troubleshooting

### Stage B `convert_hf_to_gguf.py` deps missing

**Symptom:** `run` fails early with `ModuleNotFoundError: No module named 'gguf'`
(or `sentencepiece`, or `google.protobuf`). The test suite shows 2/3 failures in
`test_preflight_deps.py` when these packages are absent:

- `test_preflight_passes_when_all_deps_present` — fails if `gguf` / `sentencepiece` /
  `protobuf` are missing.
- `test_preflight_fails_when_gguf_missing` — fails if `gguf` is missing.

**Fix:**

```bash
pip install git+https://github.com/jimbothigpen/llama.cpp gguf-py
pip install sentencepiece protobuf
```

Use the fork's `gguf-py`, not PyPI's `gguf`.

### Stage D / Stage I OOM with eager-load config

**Symptom:** `llama-imatrix` or `llama-perplexity` crashes with an out-of-memory
error, typically on models > ~15B at BF16.

**Cause:** Before 2026-05-22, all five PPL/imatrix subprocess calls hardcoded
`--no-mmap`, forcing the entire model into RAM. As of 2026-05-22, `--no-mmap`
is opt-in via `imatrix_eager_load` and `ppl_eager_load` (both default `false`).
If you have either flag set to `true` in your config and the model does not fit
in RAM, you will OOM.

**Fix:** Remove or set to `false` in your config:

```toml
imatrix_eager_load = false
ppl_eager_load = false
```

### `llama-quantize` / `llama-quantize-cost` not found

**Symptom:** `FileNotFoundError: llama-quantize not found on $PATH`.

**Fix:** Either:
- Set `path = "/your/llama.cpp/build/bin"` in `~/.prismaquant-llama/config.toml`.
- Or pass `--path /your/llama.cpp/build/bin` per run.

For `llama-quantize-cost` specifically, build it from source (§1.5). It is not
in mainline llama.cpp.

### `convert_hf_to_gguf.py` not found

**Symptom:** Stage B fails with a `FileNotFoundError` about `convert_hf_to_gguf.py`.

**Cause:** The script is not installed by `cmake --install` / `make install` —
it lives at the root of the llama.cpp source tree.

**Fix (any one of):**
- Set `convert_script = "/path/to/your/llama.cpp/convert_hf_to_gguf.py"` in
  config.
- Pass `--convert-script /path/to/convert_hf_to_gguf.py` per run.
- Symlink the script next to `llama-quantize` so auto-discovery finds it.

### prismaquant package not installed (Stage C)

**Symptom:** Stage C fails with `prismaquant package not installed` or
`ModuleNotFoundError: No module named 'prismaquant'`.

**Fix:**

```bash
pip install git+https://github.com/jimbothigpen/prismaquant.git
```

Note: `calibrate` does not run Stage C and does not require this package.
Only `run` and `explore` invoke Stage C.

### Stage I PPL failure after successful Stage H

**Symptom:** The final GGUF exists and is usable, but Stage I reports no `Final
estimate` line in its log.

**Cause:** Common with large models on GPU backends where perplexity evaluation
can crash after successful quantization.

**Fix:** The GGUF is valid. Run `llama-perplexity` manually against it:

```bash
llama-perplexity \
    -m ~/.prismaquant-llama/ggufs/<model>-PQ<B>-<P>.gguf \
    -f <corpus> -c 4096 -b 2048 -ctk f16 -ctv f16 -fa on -ngl 99 \
    --chunks 50
```

Or re-run `prismaquant-llama run` with the same flags — Stage H is a cache hit
(cachekey.json match) so only Stage I re-runs.

### Stale act-cache causes Stage C to fail

**Symptom:** `scale length X ≠ in_features Y ... likely a stale act-cache`.

**Cause:** Two models with the same Linear layer names (e.g. Qwen3.5-4B and
Qwen3.5-9B) share act-cache filenames in the same probe directory. A larger
model's probe overwrites smaller model files with wrong-shape tensors.

**Fix:** Delete the stale probe artifacts:

```bash
rm -f ~/.prismaquant-llama/_shared/probe/<affected-model>-probe.pkl
rm -f ~/.prismaquant-llama/_shared/probe/act-cache/*.pt
```

Then re-probe. Avoid probing two models with the same architecture simultaneously
when they share the same `base` directory.

### `explore` rejects `--budgets` with "mixed units"

**Symptom:** `ValueError: mixed budget units: ...` when passing a
comma-separated `--budgets` list to `explore`.

**Cause:** `explore` requires all budgets in a single sweep to use the same
unit form (all %, all bpw, or all GB). Mixing forms (e.g. `--budgets 25,4.5bpw`)
is rejected because the Pareto surface across mixed units is not meaningful.

**Fix:** Use a single unit form for all budgets in one sweep. Run separate
sweeps (different `--output-csv` files) if you need to compare across unit forms.

### TOML parse failure — missing `[prismaquant-llama]` section

**Symptom:** `ValueError: config at ... is missing the [prismaquant-llama] section`.

**Fix:** Ensure your `config.toml` has `[prismaquant-llama]` as its first section
header. The most common cause is manually editing the file and accidentally
deleting the section header. The shipped default at
`src/prismaquant_llama/data/config.toml.default` is the reference.

---

## Cross-references

- [`README.md`](../README.md) — high-level overview, quick start, output layout
- [`docs/GETTING-STARTED.md`](GETTING-STARTED.md) — hands-on first-run tutorial
- [`docs/methodology.md`](methodology.md) — the math behind the Δloss surrogate and allocator
- [`CHANGELOG.md`](../CHANGELOG.md) — change history, including the 2026-05-22 streaming/no-mmap change
