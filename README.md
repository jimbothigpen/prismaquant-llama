# prismaquant-llama

> **Bring [prismaquant](https://github.com/RobTand/prismaquant)'s
> Bayesian per-tensor mixed-precision allocation to llama.cpp / GGUF.**

A CLI that adapts the prismaquant allocator (originally built for vLLM /
compressed-tensors) to **any build of llama.cpp** — mainline
[`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) or any fork.
The only build-side requirement is the `llama-quantize-cost` tool, whose
source ships inside the package; drop it into your llama.cpp tree, rebuild,
and you're done.

> **Disclosure — vibe-coded.** I'm an enthusiast, not a programmer. Every
> line of code, doc, and commit message in this repo was written with
> [Claude Code](https://claude.com/claude-code) doing the actual
> implementation; I drive the design, review changes, and decide what
> ships. The mathematical core (prismaquant's closed-form Δloss surrogate)
> is [RobTand](https://github.com/RobTand/prismaquant)'s work.

> **Status: alpha.** End-to-end pipeline executes through all nine stages
> (A→I) without manual intervention. Two CLI subcommands (`run` +
> `calibrate`); persistent defaults via a hand-edited TOML config.

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

The package is fully self-contained after install — bundled corpora,
pipeline scripts, the C++ source for `llama-quantize-cost`, and the
shipped default config.toml all live inside the installed package and
require no source-tree access at runtime.

### Required external tools

prismaquant-llama is a pure-Python orchestrator; it shells out to:

| Tool | Source | Notes |
|---|---|---|
| `llama-quantize` | your llama.cpp build | standard mainline |
| `llama-imatrix` | your llama.cpp build | standard mainline |
| `llama-perplexity` | your llama.cpp build | standard mainline |
| `llama-bench` | your llama.cpp build | standard mainline |
| `llama-quantize-cost` | **bundled — see below** | not yet upstream |
| `prismaquant` Python package | **[`jimbothigpen/prismaquant`](https://github.com/jimbothigpen/prismaquant)** | provides `prismaquant.incremental_probe` for Stage C. Install: `pip install git+https://github.com/jimbothigpen/prismaquant.git` |

### Building `llama-quantize-cost`

The C++ source ships inside the installed package. To find it:

```bash
python -c "import prismaquant_llama; print(prismaquant_llama.quantize_cost_source_path())"
```

Copy that directory into your llama.cpp tree's `tools/`, register it, and
rebuild:

```bash
SRC=$(python -c "import prismaquant_llama; print(prismaquant_llama.quantize_cost_source_path())")
cp -r "$SRC" /path/to/your/llama.cpp/tools/
echo 'add_subdirectory(quantize-cost)' >> /path/to/your/llama.cpp/tools/CMakeLists.txt
cmake --build /path/to/your/llama.cpp/build --target llama-quantize-cost
```

The resulting binary lands at `build/bin/llama-quantize-cost` — same
directory as `llama-quantize`. Works against mainline, ik_llama,
frankenturbo2, etc.

## Usage

Two subcommands. That's the whole CLI.

```bash
prismaquant-llama run INPUT [flags]
prismaquant-llama calibrate {system|model} INPUT [flags]
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

### `--config` and `--libs`

Both flags are universal across commands:

- `--config /some/other.toml` uses an alternative config file. Useful for
  maintaining separate configs per fork (one mainline, one ik_llama, etc.) —
  point each config's `path` and `base` at the appropriate directories.
- `--libs /opt/llama/lib` prepends a directory to `LD_LIBRARY_PATH` for
  every subprocess call. Use when your llama.cpp libs aren't on the
  system loader path.

## Configuration

`~/.prismaquant-llama/config.toml` (auto-installed on first run). Single
section, flat keys, fully commented:

```toml
[prismaquant-llama]
base           = "~/.prismaquant-llama/"
path           = ""                                # empty = $PATH
quants         = ["Q3_K","Q4_K","Q5_K","Q6_K","Q8_0",
                  "IQ3_XXS","IQ3_S","IQ4_XS","IQ4_NL"]
budget         = 25                                # % of BF16
priority       = "111"                             # PPL/TG/PP, each 0–9
ppl_corpus     = ""                                # empty = bundled wikitext
imatrix_corpus = ""                                # empty = bundled bartowski-v5
ppl_chunks     = 50
imatrix_chunks = 50
convert_script = ""                                # empty = auto-discover; set if cmake-installed
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
│   ├── config.py               config.toml loader + first-run installer
│   ├── input_resolver.py       4-way input classification
│   ├── paths.py                directory layout
│   ├── data/                   bundled corpora + system.json.default + config.toml.default
│   ├── scripts/                bundled allocator + bridge
│   └── cpp/quantize-cost/      C++ source for llama-quantize-cost
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
