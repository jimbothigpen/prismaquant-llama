# Changelog

All notable changes to `prismaquant-llama` are recorded here.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project has not cut a tagged release yet (`pyproject.toml` reports
`0.2.0a1`); the entries below are grouped by the date the work landed on
`main`.

## [Unreleased]

### 2026-05-17 — Stage K cross-priority recipe dedup

- Stage K now short-circuits identical recipes across priorities by
  SHA-256 of the recipe JSON. When the allocator resolves a later
  priority to byte-identical tensor assignments (common on wide
  priority sweeps where the budget pins the same recipe across several
  priority strings), the second occurrence reuses the first's
  quantize + PPL artifacts instead of re-running them.
- Summary candidates now carry two new optional fields: `recipe_sha`
  (hex SHA-256 of the recipe JSON) and `duplicate_of` (the priority
  string of the original entry whose result was reused). Pareto
  semantics are unchanged: identical `(size_gb, ppl)` points stay on
  the frontier together.
- `summary-PQ<budget>{,-fisher}.json` `schema_version` bumped to `2`
  to reflect the new optional candidate fields. The `show-frontier`
  parser remains `.get`-based, so pre-S7 summaries continue to load
  identically; pre-S7 summaries report `summary_schema_version: 1` in
  the show-frontier JSON output.
- `show-frontier --output-csv` gains `duplicate_of` + `recipe_sha`
  columns; `--output-json` carries the same fields per candidate plus
  `summary_schema_version` per frontier.

### 2026-05-17 — Stage K summary schema versioning + `show-frontier` docs

- Stage K (`stage_k_validate`) now writes `"schema_version": 1` as the
  first key of every `summary-PQ<budget>{,-fisher}.json` it produces.
  Matches the field `show-frontier --output-json` has been emitting and
  gives downstream consumers a forward-compat marker.
- `show-frontier`'s JSON parser is `.get`-based end-to-end, so pre-S6
  summaries (no `schema_version`) continue to load unchanged — no
  migration step.
- `README.md`: documented `show-frontier`, including the
  `kl_validate = true` prereq, input semantics, filter flags
  (`--budget`, `--run`, `--all-runs`), `--output-{csv,json,md}`, and the
  schema versioning + backward-compat note. Status banner + usage intro
  bumped from "three subcommands" to "four". (commit `232b290`)

### 2026-05-17 — `show-frontier` machine-readable output

- `show-frontier` gained `--output-csv PATH`, `--output-json PATH`, and
  `--output-md PATH` so downstream tooling can ingest Pareto frontier
  results without scraping the text table. All three flags can be
  combined; stdout text rendering is preserved. JSON output carries its
  own top-level `schema_version: 1`. (commit `3e08980`)

### 2026-05-16 — Stage K Pareto frontier + `show-frontier` subcommand

- `stage_k_validate` now tags every candidate with `is_pareto: bool`
  (non-dominated in `(size_gb, ppl)` with strict-on-one tie semantics)
  and writes the flag into `summary-PQ{B}{,-fisher}.json`. A
  `K. pareto frontier (N/M): p1, p2, …` log line follows the winner.
- New CLI: `prismaquant-llama show-frontier INPUT [--budget B]
  [--run LABEL] [--all-runs]` renders the size/PPL curve sorted by
  size, with `*` for frontier points and `★` for the winner. Resolution
  uses `input_resolver.sanitize_model_name` directly so historical work
  dirs are accessible even if the original input no longer exists.
- `cli.py` dispatcher + docstring updated; project now exposes four
  subcommands (`calibrate`, `run`, `explore`, `show-frontier`).
- PrismaClip-RBC parity (upstream commits `54b65c7`, `7b1dd5c`):
  reclassified Not Applicable — the feature is gated by the
  `NVFP4_CLIPPED` serving format and has no analog in the K-quant GGUF
  pipeline. (commit `2802b4c`)

### 2026-05-13 — KL/PPL-validated frontier picker

- Stage K now runs a real-quantize sweep over candidate recipes and
  scores them by KL divergence + perplexity against the reference
  format, replacing the prior cost-estimate-only ranking. Gated by
  `kl_validate = true` in `config.toml`. (commit `2d5e669`)

### 2026-05-13 — Sidecar tied-LM-head fallback

- The Fisher sidecar pipeline now falls back to `token_embd.weight` when
  a GGUF lacks an explicit `output.weight` tensor (tied-embedding
  models). Eliminates the spurious "missing lm_head" miss that affected
  Qwen and other tied-embed architectures. (commit `2ed4374`)

### 2026-05-13 — Allocator consumes `fisher_output_mse`

- The allocator now reads the per-tensor `fisher_output_mse` field
  emitted by the cost sidecar and validates it against actual GGUF
  tensor dimensions before allocating budget. Catches stale or
  wrong-dimension cost rows (the failure mode that produced the S2
  "fisher sidecar wrong-dim" investigation) at allocator time rather
  than silently corrupting the recipe. (commit `e55dca2`)

### 2026-05-13 — Optional `fisher_output_mse` via `llama-quantize-cost` sidecar

- New optional pipeline stage that produces per-tensor
  `fisher_output_mse` measurements via the `llama-quantize-cost`
  sidecar binary, threaded through Stage E cost collection. Disabled by
  default; enable by setting the cost sidecar in `config.toml`.
  (commit `f401d95`)

## Notes

- Upstream `prismaquant` merge range `9b4dc69..f49d5af` is fully
  reflected in the entries above. No new upstream commits past
  `f49d5af` as of 2026-05-17.
- Per-commit messages on `main` carry the granular reasoning behind
  each change.
