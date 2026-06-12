# Changelog

All notable changes to `prismaquant-llama` are recorded here.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project has not cut a tagged release yet (`pyproject.toml` reports
`0.2.0a1`); the entries below are grouped by the date the work landed on
`main`.

## [Unreleased]

### 2026-05-28 — MoE-aware Stage D: `--override-kv` expert coverage, `moe_all_experts_imatrix` knob

Ensures every routed expert in a Mixture-of-Experts model receives imatrix
calibration data during Stage D `llama-imatrix`.

**Problem.** MoE models route each token to a sparse subset of experts (top-k).
With typical calibration corpora and the default `expert_used_count`, many
experts are never activated and therefore receive no imatrix gradient signal.
Those experts quantize to coarse default scales, hurting quantization quality
on any token that does route through them at inference time.

**Fix.** Auto-detect MoE at Stage D entry by reading `general.architecture`,
`<arch>.expert_count`, and `<arch>.expert_used_count` from the BF16 GGUF.
If `expert_count > 1` and `expert_count != expert_used_count`, pass:

```
--override-kv <arch>.expert_used_count=int:<expert_count>
```

to `llama-imatrix`, forcing all routed experts active on every calibration
token. The GGUF file on disk is **never modified** — the override is an
in-memory metadata patch at model-load time via llama.cpp's `--override-kv`
mechanism.

**New `Config` field** `moe_all_experts_imatrix: bool = True`. Default ON so
MoE models get full expert coverage without any user action.

**New TOML key** `moe_all_experts_imatrix = true` in `[prismaquant-llama]`.
Set to `false` to opt out globally (e.g. GPU-poor hosts where the Stage D
wall-time multiplier is unacceptable).

**New CLI flag** `--moe-all-experts-imatrix` / `--no-moe-all-experts-imatrix`
(`argparse.BooleanOptionalAction`) on all three user-facing subcommands (`run`,
`calibrate`, `explore`). Default `None` (uses config value); explicit flag
overrides `cfg.moe_all_experts_imatrix` for that invocation. Wired through
`cfg_from_args`.

**Arch coverage** — detection is prefix-agnostic: `general.architecture` is
read from GGUF metadata and used to construct the key names dynamically. Known
MoE arch prefixes (verified against llama.cpp `src/llama-arch.cpp`):

| Model family | GGUF `general.architecture` |
|---|---|
| Mixtral / LLaMA-MoE | `llama` |
| LLaMA 4 | `llama4` |
| Qwen2-MoE | `qwen2moe` |
| Qwen3-MoE | `qwen3moe` |
| Qwen3.5-MoE | `qwen35moe` |
| Qwen3-VL-MoE | `qwen3vlmoe` |
| Phi-MoE | `phimoe` |
| DeepSeek-V1 | `deepseek` |
| DeepSeek-V2 / V3 | `deepseek2` |
| OLMoE | `olmoe` |
| Granite-MoE | `granitemoe` |
| Jamba | `jamba` |

**DeepSeek shared-expert note.** DeepSeek-V2/V3 have a separate
`expert_shared_count` (shared experts always active, not subject to top-k
routing). The `--override-kv` on `expert_used_count` only affects the
*routed* pool — it expands the routed top-k to cover all routed experts while
leaving shared experts untouched (they were never sparse to begin with).

**Wall-time impact.** Stage D wall-time multiplies by
`expert_count / expert_used_count` — for example:

- Mixtral-8x7B (8 experts, top-2): ~4×
- Phi-3.5-MoE (16 experts, top-2): ~8×
- Qwen3-30B-A3B (128 experts, top-8): ~16×
- Qwen3-235B-A22B (128 experts, top-8): ~16×
- DeepSeek-V3 (256 routed experts, top-8): ~32×

Stage D is a single pass over the calibration corpus; absolute time is bounded
by `imatrix_chunks × imatrix_ctx` token count regardless of the multiplier.

**Cache-key extension** — `_shared/imatrix-cache/` filenames now include an
optional `__moe{N}` segment when the MoE override is active:

```
# Non-MoE or opted-out (unchanged from prior key shape)
<model-sha12>__<corpus-sha12>__c<chunks>__x<ctx>.imatrix.gguf

# MoE with override active (new)
<model-sha12>__<corpus-sha12>__c<chunks>__x<ctx>__moe<expert_count>.imatrix.gguf
```

Existing MoE imatrix files cached **without** the override (any prior
prismaquant-llama version) will be regenerated on the next MoE run. Existing
non-MoE / dense-model caches are **unaffected** (key shape unchanged). Disk
cost per imatrix is small (~MB-scale). There is **no backwards-compat fallback**
— clean break for MoE models.

**Community-recipe confirmation.** The `expert_used_count = expert_count`
override pattern is canonical in the llama.cpp quantization community:
- mradermacher (Qwen3-MoE imatrix recipe, 2026-05):
  `--override-kv qwen3moe.expert_used_count=int:24` (uses intermediate value
  rather than full count; see note below)
- Broader community consensus: override is required for MoE imatrix quality;
  exact value is model-dependent.

Note: some community recipes use fewer-than-max experts (e.g. 24/128 for
Qwen3-235B). prismaquant-llama defaults to the full `expert_count` for
maximum coverage; use `--no-moe-all-experts-imatrix` and a manual `--imatrix`
file if you prefer a partial-expert calibration.

### 2026-05-28 — `--imatrix-ctx` CLI flag, `imatrix_ctx` TOML key, and cache-key extension

Exposes the `llama-imatrix` context size (`-c` flag) as a first-class knob on
all three user-facing subcommands.

**New `Config` field** `imatrix_ctx: int = 512`. Default is **512**, which is:

- The `llama-imatrix` built-in default (`params.n_ctx = 512` in
  `tools/imatrix/imatrix.cpp`).
- The explicit personal choice of ikawrakow (the tool's primary developer):
  "I use 512 tokens most of the time."
- The recipe used by ubergarm (DeepSeek-V3, Qwen3-235B public imatrix files)
  and confirmed as "usually performs better than 4096" by mradermacher.
- The community standard `(-c 512 --chunks 2000)` cited across llama.cpp
  discussions #5263 and #5006.

Prior hardcoded default was 4096 — **changed to 512** on the above evidence.
The knob is now exposed so users can override per-run.

**New TOML key** `imatrix_ctx = 512` in the `[prismaquant-llama]` section
(near `imatrix_chunks`). Parsed by `load_config`; raises `ValueError` if value
< 1; defaults to 512 when absent.

**New CLI flag** `--imatrix-ctx N` on `run`, `calibrate`, and `explore`
subcommands. Default `None` (uses config value); explicit value overrides
`cfg.imatrix_ctx` for that invocation. Wired through `cfg_from_args`.

**`stage_d_imatrix` signature simplified**: the now-redundant `ctx: int = 4096`
parameter has been removed; the function reads `cfg.imatrix_ctx` directly,
symmetric to how `cfg.imatrix_chunks` is used.

**Cache-key extension** — `_shared/imatrix-cache/` filenames now include
`__x{ctx}` between the chunks segment and the `.imatrix.gguf` suffix:

```
# Old key (no ctx segment — produced by any prior prismaquant-llama version)
<model-sha12>__<corpus-sha12>__c<chunks>.imatrix.gguf

# New key (ctx segment added)
<model-sha12>__<corpus-sha12>__c<chunks>__x<ctx>.imatrix.gguf
```

Existing cached imatrix files under the old key will **not** be picked up and
will be regenerated on the next run. Old files remain on disk; remove them
manually if desired (`_shared/imatrix-cache/*.imatrix.gguf` with no `__x`
segment). Disk cost per imatrix is small (~MB-scale). There is **no
backwards-compat fallback** — clean break.

### 2026-05-27 — Global `--no-mmap` CLI flag and `no_mmap` TOML key

A new per-invocation override forces the mmap-disable flag on every llama-binary
subprocess that supports it.

**New CLI flag** `--no-mmap` on all four subcommands (`calibrate`, `run`,
`explore`, `show-frontier`). Accepted everywhere for surface consistency; a
no-op on `show-frontier` which invokes no llama-binaries.

**New TOML key** `no_mmap = false` in the `[prismaquant-llama]` section. Parsed
by `load_config`; defaults to `false` when absent (no change to existing
configs). The CLI flag is a one-shot override — passing `--no-mmap` sets
`cfg.no_mmap = True` for that invocation regardless of the TOML value.

**Existing per-stage toggles preserved.** `imatrix_eager_load` and
`ppl_eager_load` remain fully functional. The new global knob ORs over them at
each call site:

- Stage D `llama-imatrix` gate: `cfg.no_mmap or cfg.imatrix_eager_load`
- Stages I, K-ref, K-cand, calibration `llama-perplexity` gate:
  `cfg.no_mmap or cfg.ppl_eager_load`

**Newly gated call site** (previously ungated; only `cfg.no_mmap` applies):

- Calibration `llama-bench` → `--mmap 0` (llama-bench uses a value flag, not
  `--no-mmap`; user-confirmed)

**Binary support audit** (verified via `--help` against deployed builds):

| Binary | Flag form | Gated |
|---|---|---|
| `llama-imatrix` | `--no-mmap` | Yes |
| `llama-perplexity` | `--no-mmap` | Yes |
| `llama-bench` | `--mmap 0` | Yes (calibration only) |
| `llama-quantize` | not supported | No — binary lacks mmap flag |
| `llama-quantize-cost` | not supported | No — binary lacks mmap flag |

`llama-quantize` and `llama-quantize-cost` lack any mmap-disable option in the
deployed binaries. Passing an unsupported flag would fail the subprocess, so
those call sites are left ungated. `llama-quantize-cost` support (if needed)
requires a separate change in `jimbothigpen/llama-quantize-cost`.

**Default behaviour unchanged**: no mmap-disable flag is passed to any
subprocess unless `--no-mmap` is given on the CLI or `no_mmap = true` is set
in config.

### 2026-05-27 — Post-Stage-B GGUF metadata patch for `--no-mtp` models

After Stage B completes for `--no-mtp` models (Qwen3.5/3.6), the BF16 GGUF
is now patched in place to correct stale KV metadata left by the upstream
`convert_hf_to_gguf.py --no-mtp`: that flag strips the MTP-head tensors but
does **not** update `<arch>.block_count` or zero `<arch>.nextn_predict_layers`,
causing Stage D (`llama-imatrix`) to fail with
`missing tensor 'blk.32.attn_norm.weight'` when it walks block_count.

The patch decrements `block_count` by `nextn_predict_layers` and zeros
`nextn_predict_layers` via an in-place mmap write (fixed-size uint32 — no byte
offset shifts). It is defensive: becomes a no-op once upstream ships its own
fix (field will already be 0), and is skipped entirely for any GGUF that lacks
the `nextn_predict_layers` field.

### 2026-05-27 — Qwen3.5/3.6 Stage B: trunk-only BF16 via `--no-mtp`

Stage B (`convert_to_bf16`) now passes `--no-mtp` to `convert_hf_to_gguf.py`
when the source model is Qwen3.5 or Qwen3.6 (detected by HF `architectures[0]`
starting with `Qwen3_5` or `Qwen3_6`).

**Problem:** Without `--no-mtp`, the convert script bundles the MTP head as
block 32 in the BF16 GGUF. That block's tensor structure differs from the 32
main attention blocks (it lacks `attn_norm.weight`), so `llama-imatrix` rejects
the file at Stage D with `missing tensor 'blk.32.attn_norm.weight'` and the
calibration pipeline fails.

**Fix:** The trunk-only BF16 produced by `--no-mtp` loads correctly in all
downstream stages (D, C, E, F, I). The separate MTP-quantize passthrough at
Stage H (`--mtp-tensors / --mtp-format`) is unaffected.

**Requirement:** `convert_hf_to_gguf.py` must have `--no-mtp` flag support for
Qwen3.5/3.6 inputs. Older convert script builds without this flag will exit with
an argparse error (exit code 2) on these architectures.

### 2026-05-24 — bpw-canonical GGUF output filenames (v2 filename scheme)

The GGUF output filename now always encodes the **target bits-per-weight** over
the unpinned allocator domain, regardless of `--budget` input form.

**v1 (replaced):** `PQ25` (25%), `PQ4p5bpw` (4.5 bpw, `.`→`p`), `PQ16gb` (16 GB).  
**v2 (current):** `PQ4.5` (bpw form; exact), `PQ4.85` (% form — bpw derived after
Stage E), `PQ5.2` (GB form — bpw derived after Stage E). Formatted as 2 decimal
places with trailing zeros stripped: `4.0` → `PQ4`, `4.50` → `PQ4.5`.

Changes:

- `budget.format_bpw_label(bpw)` — new public helper implementing the canonical
  formatter: `f"{bpw:.2f}".rstrip("0").rstrip(".")`.
- `BudgetSpec.filename_label` — `bpw` form now returns `PQ{format_bpw_label(value)}`.
  `pct` and `gb` forms retain v1-style labels (`PQ25`, `PQ16gb`) for
  `show-frontier --budget` glob-filtering only; they no longer appear in
  pipeline output filenames.
- `paths.gguf_output_path(model, bpw, priority)` — signature changed from
  `budget_label: str` to `bpw: float`; always builds `PQ{format_bpw_label(bpw)}`.
- `pipeline_runner._compute_target_bpw_from_gb()` — new helper; inverts
  `_bpw_budget_gb` to derive target bpw from `budget_gb` after costs.csv is known.
  Used for `pct` and `gb` input forms (Stage E deferred).
- `stage_g_allocate`, `stage_k_validate`, `stage_h_quantize` — accept
  `target_bpw: float`; use `format_bpw_label` to build the `PQ<bpw>` recipe/output label.
- `allocator.py` gains `--target-bpw` argument; writes `target_bpw` into `recipe.json`
  alongside the existing `budget_gb` and `budget_input` fields.
- `explore.py` — computes `target_bpw` per cell; adds `budget_label` column
  (`PQ<bpw>`) to the output CSV for the `show-frontier --from-explore` overlay join.
- `show_frontier._SUMMARY_LABEL_RE` — updated from `PQ[a-zA-Z0-9]+` to
  `PQ\d+(?:\.\d+)?` (matches `PQ4.5`, `PQ3.47`, `PQ4`; rejects v1-style labels).
- `show_frontier._load_explore_overlay()` — prefers `budget_label` column when
  present; falls back to deriving the label from `budget` or `budget_pct`.

This is a **clean break from v1**: v1 filenames (`PQ4p5bpw`, `PQ25`, `PQ16gb`)
are no longer produced. Existing v1 `summary-PQ*.json` files will not match the
updated `_SUMMARY_LABEL_RE`; rename or re-run to regenerate with v2 labels.

### 2026-05-23 — Multi-unit `--budget` flag

`--budget` (CLI) and `budget` (TOML) now accept three input forms, all
converging on the existing internal `budget_gb` representation:

- **`25` or `25%`** — percent of BF16 GGUF size (back-compat; bare integer
  is still interpreted as %, not bpw or GB).
- **`4.5bpw`, `3bpw`** — average bits-per-weight over the unpinned allocator
  domain. Resolved after Stage E (costs.csv) using hard-pinned tensor sizes +
  unpinned parameter count.
- **`16GB`, `22GB`** — absolute gigabytes. Validated against BF16 GGUF size
  immediately after Stage B; raises an error if the value exceeds the model
  size.

Output filename encodes the form: `PQ25` (%), `PQ4p5bpw` (bpw, `.`→`p`),
`PQ16gb` (GB, lowercase). `recipe.json` gains a `budget_input` field recording
the raw user string alongside the existing `budget_gb` canonical value.

`explore --budgets` accepts the same forms but rejects mixed units in a
single sweep (e.g. `25,4.5bpw` raises a clear error).

`show-frontier --budget` now accepts any of the three forms and routes the
value through `parse_budget()` to derive the glob pattern for summary files.

### 2026-05-22 — Streaming-claim regression fix: `--no-mmap` now opt-in

- **Fix:** Stages D (imatrix), I (final PPL), K-ref/K-cand (sweep PPL), and
  calibration PPL no longer pass `--no-mmap` to `llama-imatrix` /
  `llama-perplexity` by default. Previously all five sites hardcoded
  `--no-mmap`, forcing the entire model into RAM before any computation
  (O(model_size); ~1.3 TB for 671B BF16, ~336 GB for 671B @ 4 bpw). This
  contradicted the documented streaming / bounded-RAM property and caused
  OOM on any model that did not fit in available RAM.

- **New config keys** in `[prismaquant-llama]` (both default `false`):
  - `imatrix_eager_load = true` — restores `--no-mmap` for Stage D
    (`llama-imatrix`). Use for small models that fit in RAM when
    predictable imatrix timing matters.
  - `ppl_eager_load = true` — restores `--no-mmap` for Stages I, K-ref,
    K-cand, and calibration PPL (`llama-perplexity`). Same trade-off.

- **Default behavior change:** with both flags `false` (new default),
  `llama-imatrix` and `llama-perplexity` use OS-level mmap (`use_mmap=true`),
  streaming model weights from disk on demand. Peak RAM is bounded by GPU
  VRAM + KV cache + a small CPU buffer; total RAM no longer scales with
  model file size. Runs become disk-I/O-bound (cephfs read bandwidth) rather
  than RAM-bound.

- **No other behavior changes.** Stages B, E, H, and glue stages were
  already streaming-safe and are unaffected. (commits `2f0fb5c`, `b50fcf3`,
  `a413950`)

### 2026-05-22 — MTP `forced_passthrough` budget exclusion

- Allocator now excludes MTP tensors from the DP budget when `--mtp-format BF16`
  is configured (`scripts/allocator.py`, ~28 ins / 4 del). Port of upstream
  `prismaquant` commit `6261632` (GGUF-native path). Prevents MTP layers from
  consuming size quota budgeted for the main model body. (commit `a287266`)

### 2026-05-22 — Fisher upstream-symbol regression smoke

- New `tests/test_fisher_contract.py`: validates that the Fisher-related symbols
  exported by the upstream `prismaquant` package remain importable under their
  expected paths. Auto-skips when `prismaquant` is not installed. (commit `e69170c`)

### 2026-05-17 — NVFP4 added to `_BPW_FALLBACK`

- `precondition._BPW_FALLBACK` gains `"NVFP4": 4.5` (NVIDIA NVFP4 / mainline
  `GGML_TYPE_NVFP4` — 64-element block, 32B packed E2M1 + 4B UE4M3 sub-block
  scales = 36 B / 64 = 4.5 bpw exactly). Allocator can now resolve a bpw for
  NVFP4-assigned tensors without a measured costs.csv row (e.g. tensors
  assigned by shape-propagation from exemplar layers); previously the lookup
  fell off the end and the tensor was silently skipped as `skip:unknown`.
- No other code changes; the upstream llama.cpp quantize-cost binary already
  accepts `--types NVFP4` (resolves via `ggml_type_name()` enumeration; no
  hardcoded whitelist there) so costs.csv rows for NVFP4 populate normally
  when requested.

### 2026-05-17 — `show-frontier` overlay v2: `ppl_diff` column (S11)

- Stage K `summary-PQ*.json` schema bumped to `schema_version: 3`. New
  optional field `reference_ppl_f16` records a one-time BF16 PPL pass
  on `bf16_path` using byte-identical llama-perplexity flags as the
  per-candidate runs (same `-c/-b/-ctk/-ctv/-fa/-ngl/--chunks`). The
  reference is cached in `work/stage-k/ref-ppl-f16.json` keyed by
  (bf16_path, ppl_corpus, chunks) so subsequent runs skip the
  measurement when the inputs match.
- `show-frontier --from-explore` now surfaces a `ppl_diff` column
  (`measured_ppl − reference_ppl_f16 − pred_dppl`) in text, Markdown,
  and CSV outputs whenever the loaded summary carries
  `reference_ppl_f16`. v2 summaries (no reference) keep their original
  output unchanged.
- Aggregated `--output-json` document bumped to `schema_version: 2` to
  reflect the new candidate-level `ppl_diff` key.
- Three new tests under `tests/test_show_frontier_overlay.py` cover
  v3 round-trip with ppl_diff, v3-without-explore behavior, and v2
  backward-compat.

### 2026-05-17 — Stage K dedup hash narrowed to inner assignments

- `stage_k_validate` now hashes the inner `recipe` (per-tensor
  assignment dict) in canonical-JSON form, not the whole recipe file.
  The allocator records `priority`, `weights`, `lambda`, and
  `loss_surrogate` in the recipe JSON too; those vary by priority even
  under saturated-λ, so the prior full-file SHA never collided in
  practice — the dedup loop shipped in S7 was dead code on real data.
- Verified against a 9-priority × 5-budget sweep on cached Qwopus3.5-9B
  bridge/costs: at saturated budgets (1.6/1.8/2.5 GB) all nine
  priorities collapse to one assignment-SHA; at near-floor budgets
  (2.0/2.2 GB) three pairs collide. Of 45 recipes, 30 would now be
  short-circuited; previously zero.
- No `schema_version` bump: the `recipe_sha` field's *semantics* change
  (full-file → inner-recipe canonical hash) but no consumer compared
  pre/post-S7 SHAs across summaries — show-frontier only displays them.

### 2026-05-17 — `show-frontier --from-explore` overlay

- `show-frontier` gained `--from-explore PATH` which reads an `explore`
  CSV (`explore --output-csv PATH`) and attaches simulator-predicted
  size + ΔPPL alongside each measured Stage-K candidate. Join key is
  `(budget_pct extracted from summary-PQ{N} filename, priority)`.
- New candidate fields when overlay is active: `pred_size_gb`,
  `pred_dppl`, `size_diff_gb` (measured − predicted).
- Renderers conditionally extend their column sets only when overlay
  is in use — text adds `pred_GB / pred_ΔPPL / sizeΔ`; Markdown adds
  the same; CSV adds `pred_size_gb / pred_dppl / size_diff_gb`. Calls
  without `--from-explore` produce byte-identical output to before.
- JSON adds `has_explore_overlay` (per record) and `budget_pct` so
  downstream consumers can detect overlay-enabled documents without
  parsing the filename.

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

- Upstream `prismaquant` synced to `73fff34` on 2026-06-12 (45-commit RobTand catch-up from
  `6261632`). All probe CLI contracts verified; Fisher sidecar byte-identical; zero P0 impact.
- Upstream `prismaquant` synced to `6261632` on 2026-05-22 (prior sync
  stopped at `f49d5af`). The 2026-05-22 MTP budget-exclusion entry above
  ports the `6261632` behavior to the GGUF-native allocator path.
- Per-commit messages on `main` carry the granular reasoning behind
  each change.
