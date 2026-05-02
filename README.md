# prismaquant-llama

> **Bring [prismaquant](https://github.com/RobTand/prismaquant)'s
> Bayesian per-tensor mixed-precision allocation to llama.cpp / GGUF.**

A CLI + interactive TUI that adapts the prismaquant allocator
(originally targeted at vLLM / compressed-tensors) to work with **any
prismaquant-enabled fork** of [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp).

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

> **Status: alpha / scaffold.** CLI dispatcher and TUI flow are wired;
> end-to-end pipeline execution is still a stub. See [TODO list](#todo) below.

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

## Quick run

```bash
# Bare invocation drops you into the wizard (auto-discovers llama-quantize from $PATH)
prismaquant-llama

# Explicit subcommand
prismaquant-llama wizard --binary /path/to/your-fork/build/bin/llama-quantize

# Format auto-discovery against a specific binary
prismaquant-llama discover /path/to/llama-quantize

# Empirical calibration — quick mode (~25 min for 53 formats)
prismaquant-llama calibrate quick --binary /path/to/llama-quantize \
    --ref-model /path/to/Llama-3.2-1B-BF16.gguf --formats all

# Show what dirs would be created for a given output root
prismaquant-llama paths layout --output ~/prismaquant-builds --model-name myModel
```

## CLI subcommands

| Command | Purpose |
|---|---|
| `prismaquant-llama` (no args) | drops you into the wizard (default) |
| `prismaquant-llama wizard` | interactive TUI — explicit subcommand |
| `prismaquant-llama discover` | auto-discover supported formats from a binary's `--help` |
| `prismaquant-llama calibrate {quick,deep,ingest}` | empirical calibration; quick = size only (~25 min), deep = + PPL/bench (overnight), ingest = absorb prismaquant Stage D cost.csv |
| `prismaquant-llama paths {layout,find-binaries}` | inspect output dir layout / discover llama.cpp tool binaries |
| `prismaquant-llama pipeline` | (TODO) direct pipeline exec without the TUI |

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
| `deep`  | + PPL Δ vs f16, + PP/TG tps | 6-12 hours | weekend job; produces "your binary, your hardware" curve |
| `ingest`| reads prismaquant Stage D cost.csv | seconds | automatic, every pipeline run feeds the cache |

Calibrated metadata persists at
`~/.cache/prismaquant-llama/binary-types/<binary-sha256>-calibrated.json`.

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

## How forks plug in

A prismaquant-enabled fork only needs to do two things:

1. Ship `tools/prismaquant/run-pipeline.sh` (or use the bundled one in
   this package's `src/pipeline/`).
2. *Optionally* ship `tools/prismaquant/format_metadata_<forkname>.json`
   describing fork-specific weight quants.

Step 2 is **strictly optional** — fork-specific formats without
metadata are heuristic-classified and surfaced under `--all-formats`.
Maintainer-supplied metadata is purely additive polish (curated
`recommend` flags + helpful `note` strings).

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

## TODO

The wizard is a scaffold. Outstanding work to make it production-ready:

- [ ] Wire `prismaquant-llama pipeline` to actually exec `src/pipeline/run-pipeline.sh`
- [ ] HF download integration via `huggingface_hub.snapshot_download`
- [ ] Calibration preset paths (auto-download wikitext-103 etc.)
- [ ] Imatrix cache hit/miss detection (keyed by `(model_sha, corpus_sha, chunks)`)
- [ ] Per-screen "back" navigation in the TUI
- [ ] Architecture autodetect (peek at HF `config.json` to guess bridge model arch)
- [ ] Recipe preview before execute (estimated time + disk + PPL prediction)
- [ ] Error handling per stage (retry + diagnostic output)
- [ ] End-to-end test against a small reference model
- [ ] Parameterize `src/pipeline/*` scripts (replace hard-coded paths with env vars
      — see `src/pipeline/PARAMETERIZATION-TODO.md`)

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
