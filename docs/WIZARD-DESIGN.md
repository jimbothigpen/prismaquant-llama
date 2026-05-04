# prismaquant-llama Wizard — Design Specification

Design doc for the expanded interactive TUI wizard. Supersedes the existing
4-screen scaffold at `src/prismaquant_llama/wizard.py`. This is a design,
not implementation — to be implemented in chunks after design review.

---

## Goal

Guide a brand-new user from "fresh install" to "producing prismaquant ggufs
that match their hardware + binary + use case" via a series of focused screens
with plain-language explanations, sensible defaults, validation, and disk/time
warnings.

Two-tier flow:

- **Phase 1 — First-run setup** (one-time): establishes user defaults that
  persist in `~/.prismaquant-llama/config/`. Skipped on subsequent runs unless
  user invokes `prismaquant-llama wizard --setup` to redo it.
- **Phase 2 — Perf-file bootstrap** (optional, one-time): offers to generate a
  proper system-default perf file calibrated against the user's binary +
  hardware. This is the slow part (~50 min for `--deep`, ~6-12 hr for
  `--reference`); user must opt in.
- **Phase 3 — Per-run wizard** (every invocation): pick a model, confirm
  defaults (or override for this run), generate optional model+binary perf
  file, review the final pipeline command, execute or copy-paste.

The same screens are reachable via direct subcommands for power users
(`prismaquant-llama paths set-defaults`, `prismaquant-llama calibrate deep`,
etc.) — the wizard is a guided orchestration layer, not the only entry point.

---

## Phase 1 — First-Run Setup

Triggered when:
- User runs `prismaquant-llama` with no arguments AND
- `~/.prismaquant-llama/config/wizard-setup-complete` doesn't exist

Each screen ends by writing the user's choice to a config file under
`~/.prismaquant-llama/config/`. Final screen creates `wizard-setup-complete`
to mark setup done.

### 1.0 Welcome

Plain-text intro:
- What prismaquant-llama does (one paragraph: per-tensor mixed-format quantization)
- What this wizard will configure (~5-7 minutes of yes/no questions)
- Mention `--setup` flag for reconfiguring later
- Offer to skip ("I know what I'm doing, take me to the per-run wizard")

### 1.1 Default paths

Show:
```
Where should prismaquant-llama keep its working files?

  Output ggufs:        ~/.prismaquant-llama/builds/
  HF safetensors:      ~/.prismaquant-llama/cache/hf/
  Per-binary cache:    ~/.prismaquant-llama/cache/binary-types/
  Scratch (temp):      ~/.prismaquant-llama/scratch/
  Config files:        ~/.prismaquant-llama/config/

The defaults are good for most users. Pick custom only if you have a
big external drive or shared filesystem (e.g., NFS, cephfs) where you
want the working data to live.
```

Options:
- [Default] use `~/.prismaquant-llama/`
- [Custom] prompt for a single root directory (subdirs auto-derive)
- [Per-purpose custom] one prompt per role (advanced)

Validation: target dir is writable, has enough free space (warn if <50 GB).

Saved to: `~/.prismaquant-llama/config/paths.json`

⚠️ **Disk space warning**: explain the rule of thumb — "expect ~3× the BF16
model size during a run. For a 9B model that's ~50 GB; for 70B it's ~400 GB."

### 1.2 Default binary path

Show:
```
prismaquant-llama needs to call llama.cpp's quantize / imatrix / perplexity
/ bench tools. Where are your built binaries?
```

Steps:
1. Auto-discover via `paths.find_binary()` — checks common dirs:
   `/path/to/llama.cpp/build/bin`, `/usr/local/bin`, `~/llama.cpp/build/bin`, etc.
2. Display what was found:
   ```
   ✓ /path/to/llama.cpp/build/bin/llama-quantize    (mainline, b1258)
   ✓ /path/to/llama.cpp/build/bin/llama-perplexity
   ✓ /path/to/llama.cpp/build/bin/llama-bench
   ✓ /path/to/llama.cpp/build/bin/llama-imatrix
   ✓ /path/to/llama.cpp/build/bin/llama-quantize-cost
   ```
3. If all found: confirm and proceed.
4. If some missing: prompt for path to the missing one(s).
5. If `llama-quantize-cost` missing: special handling — show vendoring
   instructions (link to README's "How any llama.cpp build plugs in"
   section) and offer to defer setup until they've built it.

Validation: each binary must be executable + return non-error from
`--help` (catches the libllama.so missing case — auto-injects
`LD_LIBRARY_PATH` from sibling lib/ if found).

Saved to: `~/.prismaquant-llama/config/binaries.json`

Multi-binary support: user can register multiple sets (e.g., "mainline",
"frankenturbo2", "ik_llama") and switch between them via per-run wizard.

### 1.3 Default allowed quant types

This screen *runs `discover` against the user's binary first*, then offers
the suggested presets footer as the choice list:

```
Your binary supports 9 recommended quant formats:

  Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, IQ4_XS, ...

Which list should the allocator default to?

  ▷ conservative (5 mainline staples — safe for any binary)
      Q4_K, Q5_K, Q6_K, Q8_0, IQ4_XS

    wide mainline (broad bpw coverage, current built-in default)
      Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL

    wide + IK-K extensions (this binary supports 5 fork formats)
      ...above... + IQ4_K, IQ4_KS, IQ4_KSS, IQ3_K, IQ3_KS

    custom (pick formats one by one)
```

Selecting "custom" walks them through a multi-select of every format the
binary supports, with annotations from the metadata (cliff warnings,
"slow", etc.).

Saved to: `~/.prismaquant-llama/config/default-formats.txt` (the file
already wired into pipeline_runner via the loader at
`paths.load_user_default_formats()`).

ℹ️ **Explanation snippet**:
```
prismaquant picks a different format for each tensor. More formats in
the list = more allocator flexibility = (usually) slightly better PPL.
Trade-off: more formats = slightly longer cost-measurement step.
```

### 1.4 Default size-ratio preference

```
What target file size do you want as the default budget?

This is the fraction of the BF16 model size the final GGUF should
target. The allocator picks per-tensor formats to hit this number.

  ▷ 25% — slightly tighter than IQ4_XS (~27%); good quality/size balance
    20% — ~Q3_K_M territory, more aggressive
    30% — ~IQ4_KS territory, better quality
    40% — ~Q5_K_M territory, conservative
    Other (specify percentage)
```

Saved to: `~/.prismaquant-llama/config/defaults.json`
(`{"budget_auto_ratio": 0.25, ...}`)

ℹ️ **Disk space warning**:
```
A 9B BF16 GGUF is ~17 GB. At 25% you'll get a ~4.3 GB output GGUF.
At 40%, ~6.8 GB. Plan accordingly.
```

### 1.5 Default priorities (PPL / TG / PP)

```
prismaquant balances three goals when picking formats per tensor:

  PPL  — quality (lower perplexity)
  TG   — token generation speed (tokens/sec when generating)
  PP   — prompt processing speed (tokens/sec when ingesting prompt)

You assign weights as 3 digits summing to 9. The default is balanced
(equal weight to all three):

  ▷ 333 — Balanced (recommended for most users)
    522 — Quality-first (best PPL at the budget)
    252 — TG-first (fast chat / streaming)
    225 — PP-first (long-context use)
    900 — Pure-quality (ignore speed entirely)
    Custom (enter your own 3-digit XYZ)
```

Validation: 3 digits, sums to ≤9 (warn if >9, since allocator normalizes
but convention is sum=9 for comparability).

Saved to: `~/.prismaquant-llama/config/defaults.json`
(`{"priority": "333", ...}`)

### 1.6 HuggingFace token setup (optional)

```
Many models on HuggingFace are gated (e.g., Llama, Gemma) and require
authentication to download. If you'll use prismaquant on those models,
we'll save your HF token now.

[skip if you only use ungated models like Qwen]
```

Options:
- [Skip] don't save a token (warn: gated downloads will fail)
- [Paste token] save to `~/.cache/huggingface/token` (standard HF location)
- [Use existing] detect existing `huggingface-cli login` session

Validation: if pasted, ping `https://huggingface.co/api/whoami-v2` to verify.

⚠️ **Security note**: token is saved to user's `~/.cache/huggingface/token`
with `0600` perms. Mention that prismaquant-llama itself doesn't read it —
it's used by `huggingface_hub` library transparently.

### 1.7 Default calibration corpus

```
prismaquant uses a calibration corpus for two stages:
  - imatrix generation (which tensors matter for which weights)
  - perplexity evaluation (the final quality measurement)

You need a plain-text file with diverse English prose (~10 MB is plenty).

Recommended starter: bartowski-calibration-v3.txt (~5 MB, well-tested).

  ▷ Download bartowski-calibration-v3.txt now (~5 MB, ~10 sec)
    I have my own corpus, let me specify the path
    Skip — I'll specify per-run
```

If "download": fetch from a stable URL (gist or mirror), save to
`~/.prismaquant-llama/cache/calibration/bartowski-calibration-v3.txt`.

Validation: file is plain-text, ≥1 MB, ≥1000 lines.

Saved to: `~/.prismaquant-llama/config/defaults.json`
(`{"calibration_corpus": "/abs/path/to/file.txt", ...}`)

### 1.8 First-run setup complete

Show summary of all defaults set, confirm with user.

Write `~/.prismaquant-llama/config/wizard-setup-complete` (just a marker file).

Offer to proceed to **Phase 2** (perf file bootstrap) or jump straight to
Phase 3 (per-run wizard).

---

## Phase 2 — Perf-File Bootstrap (Optional, One-Time)

```
The allocator works best when it has accurate measurements of how each
format performs on YOUR binary + hardware. The package ships a default
perf file calibrated on a Qwen3-8B at chunks=200, but it was generated
on different hardware. You can:

  ▷ Use the shipped default (good enough for most cases)
    Calibrate your own at --deep tier (50 chunks, ~50 min on a 9B model)
    Calibrate at --thorough tier (100 chunks, ~2 hr)
    Calibrate at --reference tier (200 chunks, ~6-12 hr — for hard-core users)
    Skip for now (you can run `prismaquant-llama calibrate deep ...` later)
```

If user picks a calibration tier:

1. **Reference model selection**:
   ```
   To calibrate, prismaquant needs a BF16 GGUF of a reference model.
   The package recommends a 8-9B dense model (large enough to capture
   format quality differences, small enough to finish in reasonable time).

     ▷ Use Qwopus3.5-9B-v3.5 (download ~20 GB, then convert to BF16)
       Use a model I already have (path)
       Skip
   ```

2. **Disk + time confirmation**:
   ```
   This will:
     - Download ~20 GB of safetensors
     - Convert to BF16 GGUF (~17 GB)
     - Run quantize + perplexity + bench for each of N formats
     - Total time: ~50 min on a modern GPU at --deep tier
     - Total disk peak: ~50 GB (during quantize sweeps)
     - Final perf file: 5 KB

   Continue? [Y/n]
   ```

3. **Run** the calibration in foreground with a progress bar (per-format
   pp/tg/ppl/bpw as it lands). Output appended live.

4. **Auto-install** the resulting perf file as system default (calls
   `set_system_default_perf` from calibration.py).

5. **Show summary**: paste the table of (format, bpw, ppl, ppl_delta, pp,
   tg) so user has a baseline reference.

⚠️ **Time/disk warnings** prominently before each tier; allow user to
back out.

---

## Phase 3 — Per-Run Wizard

Triggered every time `prismaquant-llama` is run with no subcommand (after
first-run setup is complete). All Phase 1 defaults are pre-selected; user
can confirm with [Enter] or override per screen.

### 3.1 Pick model

```
Which HuggingFace model do you want to quantize?

  ▷ Recently quantized:
      [1] unsloth/gemma-3-4b-it (last run: 2026-05-03)
      [2] Qwen/Qwen3.5-122B-A10B (last run: 2026-05-04)
    [3] Browse popular models (top trending)
    [4] Enter HF model ID directly
    [5] I have a local BF16 GGUF (skip download)
```

Options 3 + 4 hit HF API to validate model exists before proceeding.
Option 5 skips Stages A+B (download + convert).

For browse: cache last 24 hours of HF trending top-50 to prevent
rate-limiting.

After pick: show estimated disk usage:
```
unsloth/gemma-3-4b-it
  HF safetensors:  ~10 GB
  BF16 GGUF:       ~7.7 GB
  Total during run: ~18 GB peak
  Final output:    ~1.94 GB (at 25% budget)
  Free disk: 730 GB ✓
```

### 3.2 Confirm calibration corpus

Default to first-run-setup choice. Override allowed:
```
Calibration corpus: /home/user/.../bartowski-calibration-v3.txt ✓
[ENTER to confirm, or specify a different path]
```

### 3.3 Confirm budget + priority

Show defaults from setup, allow override:
```
Budget:    25% of BF16 (~1.94 GB for this model)  [override?]
Priority:  333 (balanced)                          [override?]
```

### 3.4 Confirm format whitelist

Default to first-run-setup choice. Allow toggle to alternate presets:
```
Formats: wide mainline + IK-K extensions (16 formats)  [override?]
```

If user wants override: show the same multi-select from screen 1.3.

### 3.5 Optional: model+binary perf file

```
For best allocator decisions on THIS specific model, you can run a
quick (--quick, ~25 min) calibration that generates a perf file
specific to (this binary + this model). It overrides the system
default for this model's runs.

  ▷ Use the system default perf file (faster, good enough)
    Quick calibration on this model (~25 min, +0.04 PPL precision)
```

⚠️ **Time warning**: explicitly state the additional wall time.

### 3.6 Review

Show the equivalent shell command:
```
This is what will execute:

  prismaquant-llama pipeline run \
      --hf-model unsloth/gemma-3-4b-it \
      --binary /path/to/llama.cpp/build/bin/llama-quantize \
      --calibration ~/.prismaquant-llama/cache/calibration/bartowski-calibration-v3.txt \
      --output ~/.prismaquant-llama/builds/gemma-3-4b-it-prismaquant \
      --budget-auto-ratio 0.25 \
      --priority 333

Estimated total wall time: 15-25 min on your GPU.

  ▷ Run it now
    Copy command to clipboard, exit
    Save command to a script, exit
    Cancel
```

If "run": kicks off `pipeline_runner.run_full_pipeline(cfg)` with live
progress output. On success, offers to load the result in `llama-cli`
for a smoke test.

If "copy" or "save": exits with the command for the user to run when
ready (e.g., overnight or queued).

### 3.7 Post-run

```
✓ Done. Your prismaquant GGUF is at:
    ~/.prismaquant-llama/builds/gemma-3-4b-it-prismaquant/ggufs/gemma-3-4b-it-PQ1.81-333.gguf

  Size: 1.86 GB
  PPL: 11.6745 (BF16 baseline: 11.2289, Δ +0.45)

What next?

  ▷ Generate a sweep at multiple budgets
    Try this GGUF in llama-cli (smoke test)
    Pick another model
    Exit
```

---

## Cross-cutting concerns

### Resume / mid-flow exit

Every screen writes a state file (`~/.prismaquant-llama/scratch/wizard-state.json`)
on each transition. If the wizard is interrupted (Ctrl-C, kernel oops, SSH
disconnect), the next invocation reads the state and offers:
```
Looks like a previous wizard run was interrupted at "format whitelist".
Resume? [Y/n/start over]
```

### Help on every screen

Each screen accepts `?` as input to show extended help. The InquirerPy
library supports custom keybinds for this.

### Disk-space pre-check

Before any step that writes substantial data, run `df` on the target
directory and warn if `available - estimated < 10%`. Hard-stop if
`available < estimated`.

### Time estimates calibration

The wall-time estimates printed at each step come from a small lookup
table in code, calibrated against the user's GPU class via
`amd-smi`/`nvidia-smi` lookup (if available). Fall back to "modern GPU"
estimates otherwise. Show "± 50% — your hardware may differ" caveat.

### Power-user opt-out

`prismaquant-llama wizard --skip-setup` jumps straight to per-run flow
even on first run, using built-in defaults. For users who hate wizards
or want to script.

`prismaquant-llama --no-tui` always uses CLI args, never enters wizard.

### Internationalization

Out of scope for v1. All strings in English. Hooks for translation can
be added later if there's demand.

---

## Implementation sequencing

Suggested order to land this in chunks:

| Chunk | Scope | Effort | Dependency |
|---|---|---|---|
| 1 | First-run-setup detection + welcome screen | ~1 hr | none |
| 2 | Screens 1.1 (paths) + 1.2 (binary) — saves to config | ~3 hr | InquirerPy file/path prompts, validation logic |
| 3 | Screen 1.3 (formats) — wires `discover` output into the multi-select | ~2 hr | already have `discover` + presets logic |
| 4 | Screens 1.4 (budget) + 1.5 (priority) | ~2 hr | simple inputs + validation |
| 5 | Screen 1.6 (HF token) + 1.7 (corpus download) | ~3 hr | HF API ping + download logic |
| 6 | Phase 2 (perf-file bootstrap) | ~4 hr | wraps existing `calibrate deep` + progress display |
| 7 | Phase 3 per-run wizard rewrite using saved defaults | ~5 hr | refactor existing 4 screens to consume saved defaults |
| 8 | Resume/state-file logic | ~2 hr | saves on each transition |
| 9 | Help text + disk warnings + time estimates | ~2 hr | passes over all screens |
| 10 | Polish, testing, docs | ~3 hr | end-to-end smoke test |

**Total: ~27 hours of focused work.** Can be split across multiple sessions.

---

## Open design questions

These are deliberately left for design review before implementation:

1. **InquirerPy or alternative?** InquirerPy is already used by current
   scaffold but its multi-select UX can be clunky for long lists. Worth
   evaluating `prompt_toolkit` (more flexible, more code) or `rich + click`
   (simpler, less interactive).

2. **Should Phase 2 be auto-suggested at end of Phase 1, or only via
   explicit invocation?** Auto-suggest is friendlier; explicit-only avoids
   surprising new users with an hour-long step.

3. **How to handle multi-binary users?** A user with both mainline and
   frankenturbo2 builds may want different default formats per binary.
   Solution: Phase 1.2 lets them register multiple binaries, each with
   its own format preset. Per-run wizard asks which binary first.

4. **Should the wizard manage HF cache cleanup?** A user with many
   prismaquant runs accumulates 10s of GBs of HF safetensors. A "garbage
   collect" sub-wizard could help. Probably out of scope for v1.

5. **What about `--non-interactive` for CI users?** A "wizard generates a
   config file" mode where every prompt has a flag equivalent
   (`--default-formats-preset wide-mainline`, etc.) and emits a JSON config
   for unattended use. Scriptability matters.

6. **Should we ship a `wizard-presets.json` for "common configurations"?**
   E.g., "Apple Silicon (Metal)" preset that auto-picks --skip-eval flags,
   "AMD iGPU" preset that pre-selects safe formats. Shipping curated
   presets might help newcomers. Probably v1.5.

---

## Implementation note for whoever picks this up

The existing wizard scaffold at `src/prismaquant_llama/wizard.py` already
has:
- WizardState dataclass
- 4 per-run screens (model, calibration, formats, priority+budget)
- `run_pipeline(state, dry_run)` that translates state → CLI command
- InquirerPy soft-dependency

You can largely **reuse** the per-run flow as Phase 3, refactor it to
consume saved defaults from Phase 1, and add Phase 1 + Phase 2 ahead of
it.

Don't try to land everything in one PR. Each chunk in the sequencing
table above is a reasonable single-PR unit with its own user-facing
benefit.
