# prismaquant-llama

> **Bring [prismaquant](https://github.com/RobTand/prismaquant)'s
> Bayesian per-tensor mixed-precision allocation to llama.cpp / GGUF.**

A CLI that adapts the prismaquant allocator (originally built for vLLM /
compressed-tensors) to **any build of llama.cpp** — mainline
[`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) or any fork.
The only extra build-side requirement is the
[`llama-quantize-cost`](https://github.com/jimbothigpen/llama-quantize-cost)
tool — clone it into your llama.cpp tree's `tools/`, register the
subdirectory, and rebuild.

> **Disclosure — vibe-coded.** I'm an enthusiast, not a programmer. Every
> line of code, doc, and commit message in this repo was written with
> [Claude Code](https://claude.com/claude-code) doing the actual
> implementation; I drive the design, review changes, and decide what
> ships. The mathematical core (prismaquant's closed-form Δloss surrogate)
> is [RobTand](https://github.com/RobTand/prismaquant)'s work.

> **Status: alpha.** End-to-end pipeline executes through all nine stages
> (A→I) without manual intervention. Four CLI subcommands (`run`,
> `calibrate`, `explore`, `show-frontier`); persistent defaults via a
> hand-edited TOML config.

## What this does

For each Linear in your model, prismaquant picks a *different* ggml format
under a total-size budget, minimizing measured Δloss `= ½ · H_trace · MSE`.
Output is a standard GGUF — no patches, no custom runtime. Validated wins
on Qwen3.6-35B-A3B (qwen35moe hybrid): prismaquant 14 GB recipe = 6.13 PPL
vs. best uniform IQ3_KS at 14.2 GB = 6.61 PPL — **−0.48 PPL at the same
memory footprint** (see [`docs/methodology.md`](docs/methodology.md)).

## Install

```bash
pip install --user prismaquant-llama          # once published to PyPI
# or, for development:
pip install -e .
```

The package is a pure-Python orchestrator — bundled corpora, pipeline
scripts, and the shipped default config.toml all live inside the
installed package and require no source-tree access at runtime. The C++
build-time dependency (`llama-quantize-cost`) is a separate small repo;
see below.

### Required external tools

prismaquant-llama shells out to:

| Tool | Source | Notes |
|---|---|---|
| `llama-quantize` | your llama.cpp build | standard mainline |
| `llama-imatrix` | your llama.cpp build | standard mainline |
| `llama-perplexity` | your llama.cpp build | standard mainline |
| `llama-bench` | your llama.cpp build | standard mainline |
| `llama-quantize-cost` | **[`jimbothigpen/llama-quantize-cost`](https://github.com/jimbothigpen/llama-quantize-cost)** | drop-in tool for any llama.cpp fork; not yet in mainline |
| `prismaquant` Python package | **[`jimbothigpen/prismaquant`](https://github.com/jimbothigpen/prismaquant)** | provides `prismaquant.incremental_probe` for Stage C. Install: `pip install git+https://github.com/jimbothigpen/prismaquant.git` |

### Building `llama-quantize-cost`

Clone the repo into your llama.cpp tree's `tools/` directory, wire it
into the build, and compile:

```bash
git clone https://github.com/jimbothigpen/llama-quantize-cost \
    /path/to/your/llama.cpp/tools/quantize-cost
echo 'add_subdirectory(quantize-cost)' >> /path/to/your/llama.cpp/tools/CMakeLists.txt
cmake --build /path/to/your/llama.cpp/build --target llama-quantize-cost
```

The resulting binary lands at `build/bin/llama-quantize-cost` — same
directory as `llama-quantize`. Works against mainline, ik_llama,
frankenturbo2, etc. The tool's full README documents the build / verify
workflow.

## Usage

Four subcommands.

```bash
prismaquant-llama run INPUT [flags]
prismaquant-llama calibrate {system|model} INPUT [flags]
prismaquant-llama explore INPUT --budgets ... --priorities ... [flags]
prismaquant-llama show-frontier INPUT [flags]
```

### `run` — full pipeline

```bash
prismaquant-llama run unsloth/gemma-3-4b-it
```

That's it. First invocation auto-installs a starter
`~/.prismaquant-llama/config.toml` and uses bundled defaults for everything
else. INPUT must be a HuggingFace id or an on-disk safetensors directory —
the Bayesian probe (Stage C) requires safetensors.

To override defaults for one run:

```bash
prismaquant-llama run unsloth/gemma-3-4b-it \
    --budget 30 \
    --priority 522 \
    --quants Q4_K,Q5_K,Q6_K,Q8_0,IQ4_XS \
    --imatrix-chunks 200 \
    --ppl-chunks 100
```

For persistent overrides, edit `~/.prismaquant-llama/config.toml`. The
shipped default is heavily commented and explains every key.

#### One-shot model calibration + run

For production builds where you want the allocator to use model-specific
perf data, pass `--calibrate` to `run`:

```bash
prismaquant-llama run unsloth/gemma-3-4b-it --calibrate
```

This runs `calibrate model` against the input first (writing
`calibration/models/<name>.json`), then proceeds straight into the
pipeline. Stages A (download), B (convert), and D (imatrix) are
shared between the two phases, so there's no duplicated work — the
extra cost is just the per-format quantize+PPL+bench loop.

`--calibrate` is automatically a no-op when a complete model
calibration already exists for the configured `quants` list, so re-runs
of the same model with different budget/priority cache-hit cleanly.

`--calibrate-chunks N` overrides `ppl_chunks` for the calibration
step only — useful if you want high-fidelity measurements during
calibration but a fast Stage I eval at the end (e.g. `--calibrate-chunks
200 --ppl-chunks 50`).

### `calibrate` — measure per-format perf

`calibrate system` builds the system-wide perf table the allocator
consumes by default. Run once with a representative reference model
(typically a small dense like Llama-3.2-1B-BF16 or Qwen3-8B):

```bash
prismaquant-llama calibrate system Qwen/Qwen3-8B
# → ~/.prismaquant-llama/calibration/system.json
```

`calibrate model` builds a model-specific perf table that overrides the
system default for that one model. Useful when a target model has unusual
sensitivity:

```bash
prismaquant-llama calibrate model unsloth/gemma-3-4b-it
# → ~/.prismaquant-llama/calibration/models/gemma-3-4b-it.json
```

Calibration accepts all four input forms (HF id, on-disk safetensors dir,
on-disk f16/bf16 GGUF, URL to GGUF — comma-separate split files), since
calibrate doesn't run the probe stage.

Internally, calibration generates an imatrix (using the configured
`imatrix_corpus`) on the reference GGUF once, then passes it to every
per-format `llama-quantize` invocation in the loop. This matches what
`run` does at Stage H, so calibration measurements aren't biased
against the i-quants and IK-family formats that depend on imatrix
weighting. The generated imatrix lands in `_shared/imatrix-cache/` and
is reused by subsequent `run` invocations on the same model.

Per-format perf data is persisted to `system.json` (or
`models/<name>.json`) after every format. If a calibration run is
killed mid-sweep, just re-invoke the same command — already-measured
formats are skipped and the run picks up where it left off. Live
subprocess output streams to `<base>/work/<run>/logs/calibrate-<fmt>.log`
(per format) plus a meta `calibrate.log` showing the high-level
progress. Useful for `tail -f` while a multi-hour calibration runs.

### `explore` — sweep (budget × priority) without producing a GGUF

`explore` runs Stages A–F as usual (cached on subsequent calls), then
sweeps the cartesian product of one or more budgets and priorities
through the allocator. For each cell it reports predicted size,
ΔPPL, TG, and PP using calibration data — no GGUF is produced and
no PPL eval runs. Useful for deciding which (budget, priority) to
commit to a full `run`.

```bash
prismaquant-llama explore unsloth/gemma-3-4b-it \
    --budgets 22,25,28,32 \
    --priorities 111,522,252,225,323
```

Output is a Markdown table (printed to stdout) plus optional CSV /
Markdown files via `--output-csv` / `--output-md`. Predicted ΔPPL is a
size-weighted aggregate of per-format `ppl_delta_vs_f16` from the
calibration JSON (model-specific if it exists, else system-default).
This is an approximation — it doesn't account for sensitivity-weighted
mixing — but it surfaces backend-specific quality issues directly:
e.g. on Vulkan the IQ4_KSS quality regression shows up as a large
predicted ΔPPL on rows where the allocator picks IQ4_KSS-heavy
recipes, which would be invisible from the cost CSV alone.

### `show-frontier` — re-display Stage K results

If you enable Stage K (set `kl_validate = true` in `config.toml`), each
`run` writes a `summary-PQ<budget>{,-fisher}.json` to its per-run
`work/<run>/stage-k/` directory. The summary records every priority the
validator quantized + measured, with `is_pareto: bool` annotations and
the chosen winner.

`show-frontier` reads those summaries back without re-running anything:

```bash
prismaquant-llama show-frontier unsloth/gemma-3-4b-it
prismaquant-llama show-frontier Qwen3.5-4B --budget 25 --all-runs
prismaquant-llama show-frontier Qwen3.5-4B \
    --output-csv frontier.csv \
    --output-json frontier.json \
    --output-md frontier.md
```

INPUT accepts any of the forms `run` does — HF id, safetensors dir,
GGUF, or just the bare sanitized model name (e.g. `Qwen3.5-4B`). The
input does **not** need to still exist on disk; only the historical
`work/<run>/` directory has to.

Filters:

- `--budget N` — restrict to one PQ budget (matches summaries for that
  budget across the run, including `-fisher` variants).
- `--run LABEL` — exact run label (e.g. `Qwen3.5-4B-20260515-103000`).
  Default: latest run for the model.
- `--all-runs` — render every run for the model, not just the latest.

Stdout is always a human-readable table grouped by run, with `*` marking
Pareto candidates and `★` marking the winner. The optional outputs all
share the same in-memory shape and may be combined freely:

- `--output-csv PATH` — one row per candidate; columns include `run,
  summary_file, budget_gb, fisher, user_priority, winner_priority,
  priority, size_gb, ppl, is_pareto, is_winner`.
- `--output-json PATH` — aggregated `{schema_version: 2, frontiers:
  [...]}` document; each frontier carries `recipe` and `candidate_gguf`
  paths. Bumped to 2 when the `ppl_diff` overlay column shipped (S11).
- `--output-md PATH` — Markdown document with one section per summary.
- `--from-explore PATH` — attach simulator-predicted size + ΔPPL from
  a prior `explore` CSV alongside the measured Stage-K columns
  (`pred_size_gb`, `pred_dppl`, `size_diff_gb`). When the Stage-K
  summary also carries `reference_ppl_f16` (schema_version ≥ 3), an
  additional `ppl_diff = measured_ppl − reference_ppl_f16 − pred_dppl`
  column surfaces in text, Markdown, and CSV outputs.

Stage K's `summary-PQ*.json` files carry `schema_version: 3` at the
top level. Schema history: v1 (≤ pre-2026-05-17) had no version key,
v2 added the explicit field, v3 (S11) adds an optional
`reference_ppl_f16` for use as the baseline in the `ppl_diff` overlay
column. All older summaries parse unchanged.

### Universal flags

These work on both `run` and `calibrate`:

- `--config /some/other.toml` uses an alternative config file. Useful for
  maintaining separate configs per fork (one mainline, one ik_llama, etc.) —
  point each config's `path` and `base` at the appropriate directories.
- `--libs /opt/llama/lib` prepends a directory to `LD_LIBRARY_PATH` for
  every subprocess call. Use when your llama.cpp libs aren't on the
  system loader path.
- `--convert-script /path/to/convert_hf_to_gguf.py` overrides the
  convert-script location. Auto-discovery looks two levels up from
  `path` and next to the binary; set this if your install puts the
  script elsewhere.
- `--yes` / `-y` skips the pre-flight disk + time confirmation prompt.
  Required for non-interactive / scripted use; without it on a
  non-TTY shell the command exits with an error.

### Pre-flight estimate

Before any heavy work begins, both subcommands print a summary block
showing the model, formats list, estimated disk usage at peak, free
disk available, and a rough wall-time range. Then they prompt
`Proceed? [y/N]`. Default is **N** — you have to actively confirm.
The estimate is also useful as a sanity check that you're about to run
what you intended (correct model, correct budget, correct quants list).

## Configuration

`~/.prismaquant-llama/config.toml` (auto-installed on first run). Single
section, flat keys, fully commented:

```toml
[prismaquant-llama]
base           = "~/.prismaquant-llama/"
path           = ""                                # empty = $PATH
quants         = ["Q3_K","Q4_K","Q5_K","Q6_K","Q8_0",
                  "IQ3_XXS","IQ3_XS","IQ3_S","IQ3_M",
                  "IQ4_XS","IQ4_NL","BF16"]
budget         = 25                                # % of BF16
priority       = "111"                             # PPL/TG/PP, each 0–9
ppl_corpus     = ""                                # empty = bundled wikitext
imatrix_corpus = ""                                # empty = bundled bartowski-v5
ppl_chunks     = 50
imatrix_chunks = 50
convert_script = ""                                # empty = auto-discover; set if cmake-installed
libs           = ""                                # empty = no LD_LIBRARY_PATH override
```

**`convert_script`**: `convert_hf_to_gguf.py` is NOT installed by
`cmake --install` / `make install` — it stays at the root of your
llama.cpp source tree. We auto-find it at `<fork>/convert_hf_to_gguf.py`
(walking up two levels from `path`), or next to the binary, or via
`$PATH`. If your install puts it elsewhere, set this key explicitly
or pass `--convert-script /path/to/convert_hf_to_gguf.py`.

**Mainline-only quants** ship as the default. Users running a fork
(ik_llama, frankenturbo2, etc.) can hand-add fork-specific formats —
run `llama-quantize --help` against your binary to see the full list.

**Bundled corpora** (used when `ppl_corpus` / `imatrix_corpus` are empty):

| Default | Source | Size |
|---|---|---|
| PPL | wikitext-2-raw test split | ~1.3 MB |
| imatrix | [bartowski-imatrix-v5-semantic](https://huggingface.co/datasets/lemon07r/bartowski-imatrix-v5-semantic) | ~1.5 MB |

Set `ppl_corpus` or `imatrix_corpus` to a path or URL to use your own.
URLs are downloaded to `{base}/ppl-corpus/` or `{base}/imatrix-corpus/`
and reused on subsequent runs (subject to `--purge` cleanup).

## Output layout

All paths under `{base}` (default `~/.prismaquant-llama/`):

```
base/
├── _shared/              cached intermediates (reused across runs)
│   ├── hf-cache/<model>/        downloaded HF safetensors
│   ├── bf16/<model>-BF16.gguf   BF16 conversions
│   ├── gguf-cache/              GGUFs downloaded from URL (calibrate)
│   ├── imatrix-cache/           imatrix files
│   └── probe/                   prismaquant probe artifacts
├── ppl-corpus/           downloaded PPL corpora
├── imatrix-corpus/       downloaded imatrix corpora
├── calibration/
│   ├── system.json              from `calibrate system`
│   └── models/<model>.json      from `calibrate model`
├── ggufs/                final prismaquant GGUFs
└── work/<run>/           per-run scratch (recipes, costs, logs)
```

Filename convention for final outputs:
`<model>-PQ<budget>-<priority>.gguf` — e.g., `gemma-3-4b-it-PQ25-111.gguf`.

## `--purge` cleanup

`--purge yes` (default): delete artifacts this invocation downloaded or
generated, except final GGUFs and per-run logs. Never touches user-supplied
on-disk inputs (safetensors directories, GGUF files, corpus files you
passed by path).

`--purge no`: keep everything for re-use by subsequent runs.

Concretely, `--purge yes` removes:

- HF safetensors download under `_shared/hf-cache/<model>/` (only when
  INPUT was an HF id)
- Downloaded GGUFs under `_shared/gguf-cache/<model>/` (only when INPUT
  was a URL)
- BF16 GGUF under `_shared/bf16/<model>-BF16.gguf` (only when generated
  this run from non-on-disk input)
- imatrix files for this model under `_shared/imatrix-cache/`
- probe artifacts for this model under `_shared/probe/`
- Downloaded corpora under `ppl-corpus/` and `imatrix-corpus/` (only when
  the corpus was a URL)

## Pipeline stages

The `run` subcommand executes 9 stages:

| Stage | What | Tool |
|---|---|---|
| A | Download HF safetensors (if HF id input) | `huggingface_hub` |
| B | Convert safetensors → BF16 GGUF | `convert_hf_to_gguf.py` (from llama.cpp) |
| C | Hessian probe (Bayesian sensitivity) | `prismaquant.incremental_probe` |
| D | imatrix generation | `llama-imatrix` |
| E | per-(tensor, format) MSE costs | `llama-quantize-cost` |
| F | bridge HF → GGUF tensor names | bundled script |
| G | multi-choice knapsack allocation | bundled `allocator.py` |
| H | apply allocation | `llama-quantize` |
| I | final PPL eval | `llama-perplexity` |

Each stage is idempotent and caches by file existence. Re-running a model
with different budget/priority skips A–E and re-runs only F–I (~5–10 min
on most models).

## Repo layout

```
prismaquant-llama/
├── src/prismaquant_llama/
│   ├── cli.py                  unified dispatcher
│   ├── pipeline_runner.py      run subcommand + Stages A–I
│   ├── calibration.py          calibrate subcommand
│   ├── explore.py              explore subcommand (budget × priority sweep)
│   ├── config.py               config.toml loader + first-run installer
│   ├── input_resolver.py       4-way input classification
│   ├── paths.py                directory layout
│   ├── data/                   bundled corpora + system.json.default + config.toml.default
│   └── scripts/                bundled allocator + bridge
├── docs/
├── pyproject.toml
└── README.md
```

## License

MIT (see LICENSE).

## Relationship to upstream prismaquant

[`RobTand/prismaquant`](https://github.com/RobTand/prismaquant) is the
canonical Bayesian mixed-precision allocator project; it targets vLLM and
compressed-tensors. **prismaquant-llama** is a separate
GGUF/llama.cpp-targeting adapter using the upstream allocator's
mathematical core (the closed-form Δloss surrogate). The
runner / bridge / quantize toolchain here is GGUF-specific.
