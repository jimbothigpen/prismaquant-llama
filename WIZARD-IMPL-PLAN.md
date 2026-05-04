# prismaquant-llama Wizard — New Session Kickoff

Self-contained brief for a future session implementing the expanded TUI wizard.
Read this end-to-end first; it's the execution guide. The design spec lives at
`docs/WIZARD-DESIGN.md` and contains the full screen-by-screen content.

---

## Goal in one sentence

Expand the existing 4-screen `wizard.py` scaffold into a guided onboarding
flow that walks users through first-run setup (paths, binaries, format
defaults, budget, priority, HF token, calibration corpus), optional perf-file
bootstrap, and per-run model selection — with explanations, disk/time
warnings, and resume support.

---

## State of play

### What already exists in the tree

- `src/prismaquant_llama/wizard.py` — 23 KB scaffold with 4 per-run screens
  (model, calibration corpus, formats, priority+budget), uses InquirerPy.
  This is your starting point for **Phase 3** (per-run wizard). Refactor,
  don't rewrite — most of the per-run flow is reusable.
- `src/prismaquant_llama/paths.py` — already has:
  - `DEFAULT_CONFIG_DIR = ~/.prismaquant-llama/config/`
  - `DEFAULT_OUTPUT_ROOT`, `DEFAULT_CACHE_ROOT`, `DEFAULT_SCRATCH_ROOT`
  - `find_binary()`, `discover_companion_binaries()`
  - `load_user_default_formats()` — reads `default-formats.txt`
- `src/prismaquant_llama/format_discovery.py` — `discover_formats()` returns
  per-binary format dict. The CLI subcommand emits a "Suggested presets"
  footer with conservative / wide-mainline / wide+IK-K options. Reuse this
  logic for Phase 1.3 (default format whitelist screen).
- `examples/default-formats.txt` — shipped template. Phase 1.3 should
  generate one of these from the user's selection.
- `examples/format-perf-default.json` — the package-shipped Qwen3-8B
  reference perf file. Phase 2 generates a user-specific replacement.
- `src/prismaquant_llama/calibration.py` has `calibrate_deep()` with
  `--skip-ppl` flag and `set_system_default_perf()`. Phase 2 wraps these.

### What's NOT yet done

- First-run-setup detection (`wizard-setup-complete` marker file)
- Phase 1 screens 1.0-1.8 (welcome, paths, binaries, formats, budget,
  priority, HF token, corpus, summary)
- Phase 2 (perf-file bootstrap with progress display)
- Refactor of existing 4 screens to consume saved Phase 1 defaults
- Resume / state-file logic
- Help text + disk pre-checks + time estimates
- Multi-binary support (multiple registered binary sets, switch per run)

The design spec at `docs/WIZARD-DESIGN.md` has full screen-by-screen content
(prompts, options, validation rules, warning text). Don't rewrite the design
in this session — implement against it.

---

## Suggested implementation order

Work in PR-sized chunks. Each chunk should be independently testable +
reviewable. Don't try to land everything at once.

| # | Chunk | Effort | Files touched | What's testable at the end |
|---|---|---|---|---|
| 1 | First-run-setup detection + welcome screen | 1 hr | wizard.py | `prismaquant-llama` shows welcome on fresh install, "skip" works |
| 2 | Screen 1.1 (paths) + 1.2 (binary) — saves to config | 3 hr | wizard.py, paths.py | User can pick paths + binary; configs persist |
| 3 | Screen 1.3 (formats) — wires `discover` presets into multi-select | 2 hr | wizard.py | Discover-aware default-formats.txt generated |
| 4 | Screens 1.4 (budget) + 1.5 (priority) | 2 hr | wizard.py | XYZ priority validated, budget ratio saved |
| 5 | Screen 1.6 (HF token) + 1.7 (corpus download) | 3 hr | wizard.py + new download helper | Token validated against HF API, corpus cached locally |
| 6 | Phase 2 (perf-file bootstrap with progress display) | 4 hr | wizard.py, calibration.py wrapper | Optional calibration flow runs end-to-end |
| 7 | Phase 3 per-run wizard rewrite using saved defaults | 5 hr | wizard.py | Existing 4 screens consume Phase 1 defaults |
| 8 | Resume/state-file logic | 2 hr | wizard.py | Ctrl-C-then-restart picks up where it left off |
| 9 | Help text + disk pre-checks + time estimates | 2 hr | wizard.py + helpers | `?` on any screen shows extended help; disk-warn before big writes |
| 10 | Polish, testing, docs | 3 hr | wizard.py + README | End-to-end smoke test, README pointer added |

**Total: ~27 hr.** Realistically 3-4 dedicated sessions.

---

## Critical references (read these before writing code)

1. **`docs/WIZARD-DESIGN.md`** — the design spec. Has every screen's exact
   prompt text, options, validation, and warnings. Implement against this,
   don't redesign on the fly.

2. **`docs/GETTING-STARTED.md`** — the user-facing how-to. Match its
   conventions (default paths, format names, budget percentages) so the
   wizard's choices feel consistent with the doc.

3. **Existing `wizard.py`** — read it end-to-end before touching it. The
   `WizardState` dataclass + the 4 existing screens are reusable as Phase 3.
   Don't accidentally break the working scaffold.

4. **Auto-memory entries** — relevant context the future session should
   pick up automatically:
   - `feedback_no_mmap.md` — pass `--no-mmap` for tests on this system
   - `reference_ai01_ssh_port.md`, `reference_model_paths.md`
   - `project_gfx1103_native_infeasible.md` — context for why ai01 has
     specific binary constraints

---

## Constraints

- **InquirerPy is the existing TUI library.** Stick with it unless you find
  a hard blocker. Don't introduce a new dependency without weighing
  alternatives — see "Open design questions" #1 in `WIZARD-DESIGN.md`.
- **Don't break existing CLI subcommands.** `pipeline run`, `discover`,
  `calibrate deep`, `paths` etc. must continue to work unchanged. The
  wizard is a layer on top, not a replacement.
- **Saved configs must be human-editable.** Use plain JSON (not pickle)
  for config files. Document the schema in `WIZARD-DESIGN.md` if needed.
  Users should be able to edit `~/.prismaquant-llama/config/defaults.json`
  by hand.
- **Don't run heavy GPU work without confirming with the user.** Phase 2
  can run a 50-min calibration; that's an explicit opt-in. No background
  "warm up the cache" work.
- **No-internet path must work.** First-run setup should be completable
  without network access (HF token + corpus download are the only
  network-requiring screens; they need clean "skip" paths).

---

## Useful sanity checks during implementation

```bash
# 1. Wizard launches on bare invocation
prismaquant-llama
# → should hit Phase 1 welcome on first run, Phase 3 per-run on subsequent

# 2. Config files persist in the right place
ls ~/.prismaquant-llama/config/
# → expect: paths.json, binaries.json, defaults.json, default-formats.txt

# 3. Saved defaults are honored by `pipeline run` (already-tested loader)
prismaquant-llama pipeline run --hf-model unsloth/gemma-3-4b-it ...
# → should print "[pipeline] using formats from .../default-formats.txt: ..."

# 4. --skip-setup bypasses Phase 1 cleanly
prismaquant-llama wizard --skip-setup
# → goes straight to Phase 3 even on first run

# 5. State file resume works
# Run wizard, Ctrl-C halfway, run again
# → should offer "Resume from <last screen>?"
```

---

## Open design questions to resolve before implementing

These are deliberately punted from the design spec — answer them in chunk 1
or 2 so the rest of the implementation stays consistent:

1. **InquirerPy vs prompt_toolkit vs rich+click** — InquirerPy is current
   choice. If it has hard blockers (e.g., bad multi-select UX for long
   format lists), evaluate alternatives. Document the decision in code
   comments.

2. **Auto-suggest Phase 2 at end of Phase 1, or explicit-only?** Friendly
   vs surprising. Pick one and commit to it.

3. **Multi-binary users** — register multiple binary sets in Phase 1.2
   (`mainline`, `frankenturbo2`, etc.), or just one? Multi adds complexity
   but matches real-world use on this exact dev system.

4. **`--non-interactive` mode** — should every interactive prompt have a
   flag equivalent so wizards can run in CI / scripted setups? If yes,
   that's a small added burden on every screen.

---

## Suggested first action in new session

1. **Read `docs/WIZARD-DESIGN.md` end-to-end** — full design context.
2. **Read existing `src/prismaquant_llama/wizard.py` end-to-end** — see
   what's already wired, what `WizardState` holds, how InquirerPy is used.
3. **Run `prismaquant-llama` once** with no args — see the current
   scaffold's behavior so you don't regress it.
4. **Pick chunk 1 from the sequencing table** (first-run-setup detection +
   welcome screen). Implement it as a minimal PR — `wizard-setup-complete`
   marker file detection + a welcome screen that branches to existing flow.
5. **Open a PR with chunk 1**, get feedback before proceeding to chunk 2.

The 4 chunks of Phase 1 (paths/binary/formats/budget+priority) can probably
land in one larger PR if the chunks 2-4 work feels mechanical. Phase 2
(calibration) should be its own PR. Phase 3 refactor is the trickiest —
do it after Phase 1 is settled.

---

## Don't get distracted by

- **Internationalization** — out of scope for v1, English only.
- **GUI toolkit** (Qt / GTK / web) — the design is explicitly TUI-only.
- **Migration of existing user configs** — fresh-install assumption is fine
  for v1; users who've already used prismaquant-llama can re-run the wizard
  to populate config files.
- **HF cache GC sub-wizard** — design spec calls this out as v1.5+.
- **Wizard-shipped presets for common hardware** — also v1.5+.

Stay focused on the core flow. Polish comes later.
