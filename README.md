# prismaquant-llama

> **Bring [prismaquant](https://github.com/RobTand/prismaquant)'s
> Bayesian per-tensor mixed-precision allocation to llama.cpp / GGUF.**

A CLI + interactive TUI that adapts the prismaquant allocator
(originally targeted at vLLM / compressed-tensors) to work with **any
build of llama.cpp** — mainline
[`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) or any
fork. The only build requirement on the llama.cpp side is the
`llama-quantize-cost` tool, whose source is vendored in this repo
([`src/pipeline/cpp/quantize-cost/`](src/pipeline/cpp/quantize-cost/));
drop those two files into your llama.cpp's `tools/quantize-cost/`,
re-build with `-DGGML_BUILD_TOOLS=ON`, and you're done.

> **Disclosure — this is vibe-coded.** I'm an enthusiast, not a
> programmer. Every line of code, doc, and commit message in this repo
> was written with [Claude Code](https://claude.com/claude-code) doing
> the actual implementation; I drive the design decisions, review
> changes, and decide what ships. This is disclosed up front because the
> volume of activity here would otherwise be misleading — assume
> AI-assisted unless explicitly stated otherwise. Issues and PRs are
> still welcome; just calibrate expectations accordingly. The
> mathematical core (prismaquant's closed-form Δloss surrogate) is
> [RobTand](https://github.com/RobTand/prismaquant)'s work, not mine.

> **Status: alpha / working preview.** End-to-end pipeline executes
> through all nine stages (A→I) without manual intervention; format
> auto-discovery + calibration + auto-budget defaults make typical use
> a single command. The interactive TUI wrapper around the pipeline is
> still scaffold (4 screens, no back-navigation yet) but the
> `pipeline run` subcommand is the real thing. See
> [What works today](#what-works-today) and [TODO list](#todo).

> 🆕 **New here? Start with [`docs/GETTING-STARTED.md`](docs/GETTING-STARTED.md)** —
> a hands-on walkthrough that gets you from zero to a working
> prismaquant GGUF in ~30 minutes, then walks through calibration,
> budget/priority customization, and common workflows.

## What this does

For each Linear in your model, prismaquant picks a *different* ggml
format under a total-size budget, minimizing measured Δloss
`= ½ · H_trace · MSE`. Output is a standard GGUF — no patches, no custom
runtime. Validated wins on Qwen3.6-35B-A3B (qwen35moe hybrid):
prismaquant 14 GB recipe = 6.13 PPL vs best uniform IQ3_KS at 14.2 GB =
6.61 PPL — **−0.48 PPL at the same memory footprint** (see
[`docs/methodology.md`](docs/methodology.md) for the full analysis).

This project provides:
- An interactive TUI that lowers the barrier for new users
- A unified CLI (`prismaquant-llama`) with subcommands for each pipeline stage
- Auto-discovery of formats from any llama-quantize binary
- Empirical calibration (size / PPL / throughput) on the user's binary + hardware
- Path management for the multi-stage build artifacts (HF safetensors,
  imatrix, probe, costs, recipes, GGUFs)

## Install

```bash
pip install --user prismaquant-llama          # once published to PyPI
# or, for development:
pip install -e .
```

### Required external dependencies

prismaquant-llama is a pure-Python orchestrator; it shells out to
several external tools that must be present on `$PATH` (or pointed at
via `--binary`). One of these is custom and must be built against your
llama.cpp fork.

| Tool | Source | Notes |
|---|---|---|
| `llama-quantize` | your llama.cpp fork | standard mainline tool |
| `llama-imatrix` | your llama.cpp fork | standard mainline tool |
| `llama-perplexity` | your llama.cpp fork | standard mainline tool |
| `llama-bench` | your llama.cpp fork | standard mainline tool |
| `llama-quantize-cost` | **see [`src/pipeline/cpp/quantize-cost/`](src/pipeline/cpp/quantize-cost/)** | not yet upstream — drop the source into your llama.cpp tree and rebuild with `-DGGML_BUILD_TOOLS=ON`. The README in that directory has full instructions. |
| `prismaquant` Python package | **[`jimbothigpen/prismaquant`](https://github.com/jimbothigpen/prismaquant)** | provides `prismaquant.incremental_probe` for the Hessian probe stage. **Use this fork** rather than upstream `RobTand/prismaquant` — the fork carries patches needed for Gemma-4 (iSWA, kv-sharing) and NemotronH (Mamba-2 hybrid) architectures. See [`FORK-NOTES.md`](https://github.com/jimbothigpen/prismaquant/blob/main/FORK-NOTES.md) for the full patch list. Install: `pip install git+https://github.com/jimbothigpen/prismaquant.git` |

If your fork already ships `llama-quantize-cost` (e.g.,
[`jimbothigpen/frankenturbo2`](https://github.com/jimbothigpen/frankenturbo2)
at `tools/quantize-cost/`), you don't need to copy anything — just
ensure that fork's `build/bin/llama-quantize-cost` is on `$PATH` (or
matches the `--binary` directory).

## Quick run

> Want a guided walkthrough instead of these one-liners?
> See [`docs/GETTING-STARTED.md`](docs/GETTING-STARTED.md) — covers
> the same content with explanations, sensible default values, and a
> stage-by-stage timing table.

```bash
# Recommended: full pipeline, auto-budget (25% × BF16), equal-priority
prismaquant-llama pipeline run \
    --hf-model google/gemma-4-E4B-it \
    --binary /path/to/your-fork/build/bin/llama-quantize \
    --calibration /path/to/wikitext-or-bartowski.txt \
    --output ~/prismaquant-builds

# Same model, explicit budget + PPL-heavy priority
prismaquant-llama pipeline run \
    --hf-model google/gemma-4-E4B-it \
    --binary /path/to/llama-quantize \
    --calibration /path/to/calibration.txt \
    --output ~/prismaquant-builds \
    --budget-gb 4.5 --priority 522

# Re-run with different budgets — Stages A-D cache hit, runs in ~5-10 min
prismaquant-llama pipeline run --hf-model google/gemma-4-E4B-it \
    --calibration ... --output ~/prismaquant-builds --budget-gb 3.0
prismaquant-llama pipeline run --hf-model google/gemma-4-E4B-it \
    --calibration ... --output ~/prismaquant-builds --budget-gb 6.0

# Format auto-discovery against a binary (no other files required)
prismaquant-llama discover /path/to/llama-quantize

# Empirical calibration — quick mode (~25 min for 53 formats)
prismaquant-llama calibrate quick --binary /path/to/llama-quantize \
    --ref-model /path/to/Llama-3.2-1B-BF16.gguf --formats all

# Inspect the output dir layout that would be created for a given root + model
prismaquant-llama paths layout --output ~/prismaquant-builds --model-name myModel

# Bare invocation drops you into the (still-scaffold) wizard
prismaquant-llama
```

## CLI subcommands

| Command | Purpose |
|---|---|
| **`prismaquant-llama pipeline run`** | **End-to-end build (recommended).** Runs A→I with auto-budget + cache-aware re-runs. |
| `prismaquant-llama discover` | auto-discover supported formats from a binary's `--help` |
| `prismaquant-llama calibrate {quick,deep,ingest}` | empirical calibration; quick = size only (~25 min), deep = + PPL/bench (overnight), ingest = absorb prismaquant Stage D cost.csv |
| `prismaquant-llama paths {layout,find-binaries}` | inspect output dir layout / discover llama.cpp tool binaries |
| `prismaquant-llama wizard` | (alpha) interactive TUI — currently the same 4-screen scaffold; production-quality TUI still TODO |
| `prismaquant-llama` (no args) | drops you into the wizard (default) |

Run `prismaquant-llama <subcommand> --help` for per-subcommand options.

## Screen flow (when running the TUI)

| Screen | Asks | Maps to |
|---|---|---|
| 1 | HF model ID + revision | `huggingface-cli download` |
| 2 | Calibration corpus (preset or path) | `--dataset` to probe + `-f` to imatrix |
| 3 | Format whitelist (auto-discovered from binary) | `--types` to `llama-quantize-cost` |
| 4 | Priority XYZ + budget GB | `--priority` + `--budget-gb` to allocator |

## Format discovery (Screen 3 / `discover` subcommand)

The format whitelist is built dynamically by parsing `<binary> --help`,
intersected with a metadata overlay. Four discovery layers, merged in
order — later layers override earlier:

1. **Heuristic** (always) — extract bpw + family + source from
   `<binary> --help` output text. Works on any binary, including
   forks the wizard has never seen.
2. **`format_metadata_base.json`** (ships in this package) — covers
   mainline llama.cpp + ikllama with curated bpw, recommend flags, notes.
3. **`format_metadata_<forkname>.json`** (fork-supplied, **OPTIONAL**)
   — convention path: `<binary>/../../tools/prismaquant/format_metadata_*.json`.
   Forks ship their own file describing fork-specific formats. **No
   buy-in required**: forks without an extension file still work; their
   custom formats fall through to heuristic classification.
4. **`~/.config/prismaquant-llama/format_metadata_*.json`** (user
   override) — final layer; flip `recommend` flags for personal taste.

Cached per binary SHA-256 in `~/.cache/prismaquant-llama/binary-types/`.

## Empirical calibration (`calibrate` subcommand)

Three modes for filling the metadata with measured data instead of
relying on the heuristic alone:

| Mode | What it measures | Time (53 formats) | When to use |
|---|---|---|---|
| `quick` | output size → bpw | ~25 min on 1B ref model | one-time setup, accurate bpw on this binary |
| `deep`  | + PPL Δ vs f16, + PP/TG tps | 25 min – 10 hr (depends on `--chunks`) | produces "your binary, your hardware" curve |
| `ingest`| reads prismaquant Stage D cost.csv | seconds | automatic, every pipeline run feeds the cache |

Calibrated metadata persists at
`~/.cache/prismaquant-llama/binary-types/<binary-sha256>__chunks<N>-calibrated.json`.
Each chunks tier maintains an independent cache file — re-running at a
higher tier doesn't cache-hit the lower-tier measurements.

### Chunks presets — PPL-Δ confidence vs. wall time

`calibrate deep` accepts `--chunks N` directly OR one of the named presets:

| Flag | N | Reliable Δ | Use case |
|---|---:|---:|---|
| `--quick` | 10 | ±0.20 PPL | smoke test / dev iteration |
| (default) | **25** | ±0.13 PPL | balanced — ranks adjacent K-quants for most ≥4B models |
| `--deep` | 50 | ±0.09 PPL | production per-binary perf files |
| `--thorough` | 100 | ±0.06 PPL | cross-binary baselines |
| `--reference` | 200 | ±0.04 PPL | one-time global-default ship |

Why this matters: the shipped `examples/format-perf-default.json` is
calibrated at `--reference` (200 chunks). When the allocator consults
the perf file, it walks a 4-tier priority chain:

```
Tier 1: per-binary cache       ~/.cache/.../<sha>__chunks<N>-perf.json
Tier 2: --format-perf override per-run flag
Tier 3: system default         ~/.config/prismaquant-llama/system-default-format-perf.json
Tier 4: package examples       examples/format-perf-default.json   ← shipped at --reference
```

**Tier 1 always wins over Tier 4** — even if Tier 1 is a `--quick` (10
chunks) calibration and Tier 4 is `--reference` (200 chunks). This is
intentional: per-binary captures hardware-specific throughput (pp/tg)
that the hardware-normalized shipped default deliberately strips. Within
Tier 1, the highest-chunks variant wins automatically.

⚠️ **Heads-up on calibration quality**: a hasty `--quick` calibration
on your binary will outrank the high-fidelity shipped default for that
binary. PPL-Δ noise at chunks=10 can flip the ranking of adjacent
K-quants, which propagates into allocator scoring. If you want best
allocation quality:

- Run at least `--deep` (50 chunks) for any per-binary cache you intend
  to keep (or `--thorough` if you can spare the time).
- Reserve `--quick` for dev iteration where rankings aren't critical.
- If a fast calibration is the only one available, the allocator still
  produces a valid recipe — the worst-case is that it picks a slightly
  suboptimal format for one or two layers.

To explicitly use the shipped default instead of your local calibration:
`prismaquant-llama pipeline run --format-perf $(python -c 'import prismaquant_llama, pathlib; print(pathlib.Path(prismaquant_llama.__file__).parents[2] / "examples/format-perf-default.json")') ...`

## Output directory layout

```
<output>/                            ← --output / -o, default ~/prismaquant-builds/
├── _shared/                         ← reusable across runs
│   ├── calibration/                 ← wikitext / c4 / the-pile
│   ├── hf-cache/                    ← HF safetensors
│   └── imatrix-cache/               ← keyed by (model_sha, corpus_sha, chunks)
├── ggufs/                           ← final outputs (override with --output-ggufs)
└── work/<model>-<timestamp>/        ← per-run scratch (cleanable with --keep-work=false)
    ├── bf16/, probe/, costs/, recipes/, logs/
```

## How any llama.cpp build plugs in

prismaquant-llama works against **any llama.cpp source tree** — mainline
`ggml-org/llama.cpp` or any fork. The contract is exactly two pieces:

1. **Required**: build `llama-quantize-cost`. The source lives at
   [`src/pipeline/cpp/quantize-cost/`](src/pipeline/cpp/quantize-cost/)
   in this repo (a single `.cpp` + `CMakeLists.txt`). Copy that
   directory into your llama.cpp's `tools/quantize-cost/`, register it
   in your top-level `tools/CMakeLists.txt` (`add_subdirectory(quantize-cost)`),
   and rebuild with `-DGGML_BUILD_TOOLS=ON`. The standard
   `llama-quantize`, `llama-perplexity`, `llama-bench`, and
   `llama-imatrix` come with any llama.cpp build by default — no
   patches needed there.

2. *Optional*: ship `tools/prismaquant/format_metadata_<forkname>.json`
   describing your fork's weight quants. Without this, fork-specific
   formats are still discovered automatically (via `<binary> --help`
   parsing) and heuristic-classified; maintainer-supplied metadata is
   purely additive polish (curated `recommend` flags + helpful `note`
   strings).

That's it — no other patches, hooks, or fork modifications. If your
build's binaries respond to standard `--help` and produce GGUFs, the
allocator works.

## Repo layout

```
prismaquant-llama/
├── src/
│   ├── prismaquant_llama/         ← Python package (TUI + dispatcher + helpers)
│   │   ├── cli.py                  unified CLI dispatcher
│   │   ├── wizard.py               interactive TUI (4 screens)
│   │   ├── format_discovery.py     parse <binary> --help + metadata overlay
│   │   ├── format_metadata_base.json  base metadata (mainline + ikllama)
│   │   ├── calibration.py          empirical bpw / PPL / bench calibration
│   │   └── paths.py                binary discovery + output dir tree
│   └── pipeline/                   ← shell + python pipeline scripts
│       ├── run-pipeline.sh          master orchestrator
│       ├── scripts/allocator.py     multi-choice knapsack solver
│       ├── scripts/bridge_probe_to_gguf.py  HF→GGUF tensor name bridge
│       ├── comparison-sweep/        staged comparison sweep tooling
│       └── PARAMETERIZATION-TODO.md (scripts have hard-coded paths; refactor before publish)
├── examples/
│   ├── recipes/                    sample allocator outputs
│   ├── pinned-tensors-qwen36.json
│   └── format-tps-gfx1150.json
├── docs/
│   └── methodology.md               prismaquant methodology + recipe gallery
├── pyproject.toml
└── README.md (this file)
```

## What works today

The `pipeline run` subcommand executes the full A→I sequence (download,
convert, probe, imatrix, costs, bridge, allocate, quantize, eval) in
one invocation. Each stage is idempotent and cache-aware — re-runs
skip completed stages.

```bash
# Minimal invocation — auto-budget at 25% of BF16, equal-priority
prismaquant-llama pipeline run \
    --hf-model google/gemma-4-E4B-it \
    --binary /path/to/your-fork/build/bin/llama-quantize \
    --calibration /path/to/wikitext-or-bartowski.txt \
    --output /path/to/output-root
```

| Capability | What it gives you |
|---|---|
| **Auto-budget** | Default `--budget-gb` derives 25% of the BF16 GGUF size after Stage B (slightly tighter than mainline IQ4_XS at ~28%). Override with explicit `--budget-gb`. |
| **Equal-weight priority default** | `--priority 333` — neutral PPL/TG/PP weighting. Override with e.g. `522` for PPL-heavy or `252` for prefill-heavy. |
| **Format auto-discovery** | Parses `<binary> --help` to learn what formats your fork supports. Heuristic-classifies unknown formats (bpw/family/source) so any fork works without metadata files. Curated `format_metadata_base.json` (ships) covers mainline; forks can drop in `format_metadata_<forkname>.json` for polish. |
| **Convention-path metadata loading** | Forks place metadata at `<binary>/../../tools/prismaquant/format_metadata_*.json` and we pick it up automatically. Works for `frankenturbo2` reference fork. |
| **Shared BF16 + imatrix caches** | First run pays the convert + imatrix cost; subsequent runs (different budgets/priorities, same model) skip straight to allocate + quantize. ~10-30 min saved per re-run. |
| **Per-host HSA override** | Auto-applies `HSA_OVERRIDE_GFX_VERSION=11.0.2` on hostnames that need it (currently `ai01`-pattern; extend per fleet). Avoids manual env wrapping. |
| **Per-stage logs** | `<work>/logs/stage-{A..I}.log` for each run. Pipeline failure points to the right log file in the error message. |
| **Standalone subcommands** | `discover`, `calibrate quick/deep/ingest`, `paths layout/find-binaries`, `wizard` for users who only want one piece of the workflow. |

### Validated end-to-end on

- `google/gemma-4-E4B-it` (~4B effective, iSWA hybrid — required upstream prismaquant patches; see [`docs/gemma4-profile.md`](docs/gemma4-profile.md))
- `unsloth/gpt-oss-20b-BF16` (~20B MoE — clean DefaultProfile path)
- `Jackrong/Qwopus3.5-9B-v3.5` (~9B dense; original validation set, 28-row sweep across 3 budgets × 9 priorities)

### TODO

What's still missing for production-readiness:

- [ ] **TUI wrapper** for `pipeline run` (subcommand exists; the 4-screen TUI front-end is still stub)
- [ ] **Calibration preset paths** — auto-download wikitext-103, c4, the-pile by name (currently user passes a local file path)
- [ ] **Architecture autodetect** in the TUI — peek at HF `config.json` to surface arch-specific notes (e.g. "this is a hybrid Mamba-2; expect probe to take longer")
- [ ] **Recipe preview before execute** — show estimated final size + per-tensor format breakdown before Stage H runs
- [ ] **Per-stage retry + better diagnostics** on transient failures
- [ ] **prismaquant upstream PRs** — our [`jimbothigpen/prismaquant`](https://github.com/jimbothigpen/prismaquant) fork carries 8 generic + 1 architecture-specific patch ([`FORK-NOTES.md`](https://github.com/jimbothigpen/prismaquant/blob/main/FORK-NOTES.md)). Generic patches are clean upstream candidates; PRs deferred until prismaquant-llama gets a few weeks of in-use validation
- [ ] **Parameterize `src/pipeline/*`** legacy shell scripts (largely superseded by `pipeline_runner.py`; keep for reference until removed)

## License

MIT (see LICENSE).

## Contributing

Issues + PRs welcome. The wizard component is intentionally small (4 screens, 4 modules);
resist scope creep there. Power users who need anything beyond the 4 dimensions the
wizard exposes should fall through to `src/pipeline/run-pipeline.sh` directly.

## Relationship to upstream prismaquant

[`RobTand/prismaquant`](https://github.com/RobTand/prismaquant) is the
canonical Bayesian mixed-precision allocator project; it targets vLLM
and compressed-tensors. **prismaquant-llama** is a separate
GGUF/llama.cpp-targeting adapter; we use the upstream allocator's
mathematical core (the closed-form Δloss surrogate) but the
runner/bridge/quantize toolchain here is targeted at the GGUF format
and llama.cpp's quant catalog.
