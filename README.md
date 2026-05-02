# prismaquant-wizard

Interactive TUI for customizing [prismaquant](https://github.com/vasilisn/prismaquant)
GGUF builds, designed to work with **any prismaquant-enabled fork** of
[`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp).

> **Status: alpha / scaffold.** Screen flow is wired; pipeline execution
> is still a stub. See [TODO list](#todo) below.

## What this is

A Python TUI that wraps the prismaquant pipeline (HF download → Hessian
probe → imatrix → cost measurement → allocator → quantize → eval) so
you can roll a custom prismaquant GGUF without learning the full CLI
surface. Lowers the barrier for new users; for power users, prints the
equivalent shell command at each step so you can copy-paste later.

## Why a wizard

The prismaquant pipeline has 9 stages, each with 3-15 flags. New users
mostly need to customize 4 things: **the model**, **the calibration
corpus**, **the format whitelist**, and **the priority + budget**. The
wizard exposes exactly those four as screens and pipes everything else
through to the existing CLI scripts.

## Install

```bash
pip install --user prismaquant-wizard            # once published to PyPI
# or, for development:
pip install -e .
```

## Quick run

```bash
# auto-discover llama-quantize from $PATH
prismaquant-wizard

# explicit binary (e.g. a fork's build/bin)
prismaquant-wizard --binary /path/to/your-fork/build/bin/llama-quantize

# walk the screens but don't execute the pipeline (stub mode)
prismaquant-wizard --dry-run
```

## Screen flow

| Screen | Asks | Maps to |
|---|---|---|
| 1 | HF model ID + revision | `huggingface-cli download` |
| 2 | Calibration corpus (preset or path) | `--dataset` to probe + `-f` to imatrix |
| 3 | Format whitelist (auto-discovered from binary) | `--types` to `llama-quantize-cost` |
| 4 | Priority XYZ + budget GB | `--priority` + `--budget-gb` to allocator |

## Format discovery (Screen 3)

The whitelist is built dynamically by parsing `<binary> --help`,
intersected with a curated metadata overlay. Three discovery layers:

1. **Heuristic** (always) — extract bpw + family + source from the
   `--help` output's format names and descriptions.
2. **`format_metadata_base.json`** (ships in this package) — overrides
   heuristic for known mainline + ikllama formats with curated bpw,
   recommend flag, notes.
3. **`format_metadata_<forkname>.json`** (fork-supplied, OPTIONAL) —
   convention path: `<binary>/../../tools/prismaquant/format_metadata_*.json`.
   Forks ship their own file describing fork-specific formats. **No
   buy-in required**: forks without an extension file still work; their
   custom formats fall through to heuristic classification.
4. **`~/.config/prismaquant-wizard/format_metadata_*.json`** (user
   override) — final layer; flip `recommend` flags for personal taste.

Cached per binary SHA-256 in `~/.cache/prismaquant-wizard/binary-types/`.

## Empirical calibration

Three modes for filling metadata with measured data instead of relying
on the heuristic:

```bash
# Quick: derive accurate bpw for every format the binary supports (~25 min)
prismaquant-wizard-calibrate quick \
    --binary /path/to/llama-quantize \
    --ref-model /path/to/Llama-3.2-1B-BF16.gguf \
    --formats all

# Deep: full PPL Δ vs f16 + PP/TG bench sweep (~6-12 hours, overnight)
prismaquant-wizard-calibrate deep \
    --binary /path/to/llama-quantize \
    --ref-model /path/to/Llama-3.2-1B-BF16.gguf \
    --calibration-corpus /path/to/wikitext.txt \
    --formats Q4_K_M,Q5_K_M,Q6_K,IQ4_K,IQ4_KS

# Ingest: absorb cost.csv from a prior prismaquant pipeline run (free)
prismaquant-wizard-calibrate ingest \
    --cost-csv /path/to/cost.csv \
    --binary /path/to/llama-quantize
```

Calibrated metadata persists at
`~/.cache/prismaquant-wizard/binary-types/<binary-sha256>-calibrated.json`
and is loaded by the wizard as Layer 1.5 (overrides heuristic, falls
under static maintainer files when both are present).

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

1. Ship `tools/prismaquant/run-pipeline.sh` (the convention pipeline
   script — already standard in prismaquant-enabled forks).
2. *Optionally* ship `tools/prismaquant/format_metadata_<forkname>.json`
   describing fork-specific weight quants.

Step 2 is **strictly optional** — fork-specific formats without
metadata are heuristic-classified and surfaced under `--all-formats`.
Maintainer-supplied metadata is purely additive polish (curated
`recommend` flags + helpful `note` strings).

## TODO

The wizard is a scaffold. Outstanding work to make it production-ready:

- [ ] Wire `run_pipeline()` to actually exec `<fork>/tools/prismaquant/run-pipeline.sh`
- [ ] HF download integration via `huggingface_hub.snapshot_download`
- [ ] Calibration preset paths (auto-download wikitext-103 etc.)
- [ ] Imatrix cache hit/miss detection (keyed by `(model_sha, corpus_sha, chunks)`)
- [ ] Per-screen "back" navigation
- [ ] Architecture autodetect (peek at HF `config.json` to guess bridge model arch)
- [ ] Recipe preview before execute (estimated time + disk + PPL prediction)
- [ ] Error handling per stage (retry + diagnostic output)
- [ ] End-to-end test against a small reference model

## License

MIT (see LICENSE).

## Contributing

Issues + PRs welcome. The wizard is intentionally small (4 screens, 4 modules);
resist scope creep. Power users who need anything beyond the 4 dimensions the
wizard exposes should fall through to `<fork>/tools/prismaquant/run-pipeline.sh`
directly.
