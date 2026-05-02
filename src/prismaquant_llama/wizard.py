#!/usr/bin/env python3
"""
prismaquant-wizard — interactive TUI for customizing prismaquant GGUF builds.

STATUS: stub (task #31). Screen functions are scaffolded with TODO stubs.
The flow + CLI-equivalent printing is wired up so you can see the shape.
Run with --dry-run to walk through the screens without executing the
pipeline.

Pipeline stages this wizard wraps (see ../run-pipeline.sh for full detail):

    A. download HF safetensors                      → screen 1
    B. select calibration corpus                    → screen 2
    C. generate Hessian probe                       → automatic
    D. generate imatrix                             → automatic
    E. measure per-(tensor, format) MSE             → screen 3 (whitelist)
    F. bridge HF→GGUF tensor names                  → automatic
    G. allocate (knapsack)                          → screen 4 (priority + budget)
    H. apply recipe (llama-quantize)                → automatic
    I. evaluate (PPL + bench)                       → optional

Design principles:
- Wizard is onboarding/iteration UX, NOT the primary interface.
- At each step, print the equivalent shell command. Users should be able
  to copy-paste the printed sequence to reproduce the run without the wizard.
- Cache HF safetensors + imatrix between runs keyed by (model_hash,
  calibration_hash). Rerunning with the same inputs should be ~instant.
- Power-user escape hatch: --resume <state-file> jumps mid-flow.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# Soft-dep imports: print helpful message if missing.
try:
    from InquirerPy import inquirer
    from InquirerPy.base.control import Choice
except ImportError:
    inquirer = None  # type: ignore

# Sibling modules — handle both package import (from wheel) and direct script run
try:
    from .format_discovery import discover_formats, FormatInfo
    from .paths import (find_binary, discover_companion_binaries, WorkPaths,
                        add_path_args, resolve_paths_from_args, DEFAULT_OUTPUT_ROOT)
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from format_discovery import discover_formats, FormatInfo  # type: ignore  # noqa: E402
    from paths import (find_binary, discover_companion_binaries, WorkPaths,    # type: ignore  # noqa: E402
                       add_path_args, resolve_paths_from_args, DEFAULT_OUTPUT_ROOT)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CACHE_ROOT = Path(os.environ.get("PRISMAQUANT_CACHE",
                                  Path.home() / ".cache" / "prismaquant"))

CALIBRATION_PRESETS = {
    "wikitext-103": "wikitext-103/wiki.test.raw",
    "c4-en":        "c4/en.subset.txt",
    "the-pile":     "the-pile/sample.txt",
    "(custom)":     None,  # prompt for path
}

# Hard-coded fallback if discovery fails. Mirrors prior wizard default.
HARDCODED_FALLBACK_FORMATS = [
    "Q4_K", "Q5_K", "Q6_K", "Q8_0",
    "IQ4_XS", "IQ4_NL",
    "IQ4_K", "IQ4_KS", "IQ4_KSS",
    "IQ3_K", "IQ3_KS",
    "IQ2_K",
]

PRIORITY_PRESETS = {
    "900 — pure PPL (best quality)":            "900",
    "522 — PPL primary, balanced PP/TG":        "522",
    "333 — fully balanced":                     "333",
    "252 — PP-favoring":                        "252",
    "225 — TG-favoring":                        "225",
    "(custom)":                                 None,
}


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WizardState:
    """User selections across all 4 screens. Serializable for --resume."""
    hf_model: Optional[str] = None
    hf_revision: str = "main"
    calibration: Optional[str] = None  # path or preset key
    binary_path: Optional[str] = None  # set by main() after find_binary() resolves
    formats: list[str] = field(default_factory=lambda: list(HARDCODED_FALLBACK_FORMATS))
    show_all_formats: bool = False  # if True, screen 3 lists every format the binary supports
    priority: str = "522"
    budget_gb: float = 5.25
    budget_band_gb: float = 0.25
    output_name: Optional[str] = None  # auto-derived from model + budget + priority

    def cache_key(self) -> str:
        """Hash of inputs that determine downstream artifacts."""
        h = hashlib.sha256()
        h.update(f"{self.hf_model}@{self.hf_revision}".encode())
        h.update(str(self.calibration).encode())
        return h.hexdigest()[:12]

    def auto_output_name(self) -> str:
        # mirrors PRISMAQUANT.md naming: <base>-PQ<budget>-<XYZ>.gguf
        base = (self.hf_model or "model").split("/")[-1]
        return f"{base}-PQ{self.budget_gb}-{self.priority}.gguf"


# ─────────────────────────────────────────────────────────────────────────────
# Screen 1: model selection
# ─────────────────────────────────────────────────────────────────────────────

def screen_1_model(state: WizardState) -> None:
    """
    SCREEN 1 — HF MODEL SELECTION

    ┌──────────────────────────────────────────────────────────────┐
    │ prismaquant-wizard — Step 1/4: Model                         │
    ├──────────────────────────────────────────────────────────────┤
    │ HuggingFace model ID:  Jackrong/Qwopus3.5-9B-v3.5            │
    │ Revision (branch/tag): main                                  │
    │                                                              │
    │ Cached: yes (~/.cache/prismaquant/<key>/safetensors/)        │
    │                                                              │
    │  [ Continue ]   [ Browse cache ]   [ Quit ]                  │
    └──────────────────────────────────────────────────────────────┘

    Equivalent CLI:
        huggingface-cli download <model> --revision <rev> \\
            --local-dir <cache>/<key>/safetensors
    """
    if inquirer:
        state.hf_model = inquirer.text(
            message="HuggingFace model ID:",
            default=state.hf_model or "Jackrong/Qwopus3.5-9B-v3.5",
        ).execute()
        state.hf_revision = inquirer.text(
            message="Revision (branch/tag):",
            default=state.hf_revision,
        ).execute()
    else:
        state.hf_model = input("HF model ID: ").strip() or "Jackrong/Qwopus3.5-9B-v3.5"

    print(f"  → Cache key: {state.cache_key()}")
    print(f"  → CLI: huggingface-cli download {state.hf_model} "
          f"--revision {state.hf_revision} "
          f"--local-dir {CACHE_ROOT}/{state.cache_key()}/safetensors")


# ─────────────────────────────────────────────────────────────────────────────
# Screen 2: calibration corpus
# ─────────────────────────────────────────────────────────────────────────────

def screen_2_calibration(state: WizardState) -> None:
    """
    SCREEN 2 — CALIBRATION CORPUS

    ┌──────────────────────────────────────────────────────────────┐
    │ prismaquant-wizard — Step 2/4: Calibration                   │
    ├──────────────────────────────────────────────────────────────┤
    │ Pick a preset or supply a custom path:                       │
    │   ( ) wikitext-103     (recommended for general English)     │
    │   ( ) c4-en            (web-style English)                   │
    │   ( ) the-pile         (broad domain mix)                    │
    │   (•) (custom path)                                          │
    │                                                              │
    │ Custom path: /mnt/cephfs/0/Container/models/wikitext103-...  │
    │                                                              │
    │  [ Continue ]   [ Back ]                                     │
    └──────────────────────────────────────────────────────────────┘

    Equivalent CLI: passed as --dataset to incremental_probe + as -f
    to llama-imatrix.
    """
    if inquirer:
        choice = inquirer.select(
            message="Calibration corpus:",
            choices=list(CALIBRATION_PRESETS.keys()),
            default="(custom)",
        ).execute()
        if CALIBRATION_PRESETS[choice] is None:
            state.calibration = inquirer.filepath(
                message="Path to calibration text:",
                default=state.calibration or "",
                validate=lambda p: Path(p).exists(),
            ).execute()
        else:
            state.calibration = CALIBRATION_PRESETS[choice]
    else:
        state.calibration = input("Calibration path: ").strip()

    print(f"  → Calibration: {state.calibration}")


# ─────────────────────────────────────────────────────────────────────────────
# Screen 3: format whitelist
# ─────────────────────────────────────────────────────────────────────────────

def screen_3_formats(state: WizardState) -> None:
    """
    SCREEN 3 — WEIGHT QUANT FORMAT WHITELIST (auto-discovered)

    ┌──────────────────────────────────────────────────────────────┐
    │ prismaquant-wizard — Step 3/4: Format whitelist              │
    ├──────────────────────────────────────────────────────────────┤
    │ Discovered 53 formats from the binary; showing 15 recommended│
    │ ([a] to toggle "show all"; [r] to revert to recommended).    │
    │                                                              │
    │ Pick which formats the allocator may consider per-tensor:    │
    │   [x] Q4_K       (4.50 bpw, k,  mainline)                    │
    │   [x] Q5_K       (5.50 bpw, k,  mainline)                    │
    │   [x] Q6_K       (6.56 bpw, k,  mainline)                    │
    │   [x] Q8_0       (8.50 bpw, k,  mainline)                    │
    │   [x] IQ4_XS     (4.25 bpw, i,  mainline)                    │
    │   [x] IQ4_K      (4.50 bpw, ik, ikllama)  ← imatrix-aware    │
    │   [x] IQ4_KS     (4.25 bpw, ik, ikllama)                     │
    │   [x] IQ4_KSS    (4.00 bpw, ik, ikllama)                     │
    │   [x] IQ3_K      (3.44 bpw, ik, ikllama)                     │
    │   [x] IQ3_KS     (3.19 bpw, ik, ikllama)                     │
    │   [x] IQ2_K      (2.38 bpw, ik, ikllama) ← extreme           │
    │   ...                                                        │
    │                                                              │
    │  [ Continue ]   [ Back ]   [ a: show all ]                   │
    └──────────────────────────────────────────────────────────────┘

    Auto-discovery: parses `<binary> --help` to enumerate supported
    types, intersects with metadata overlay (format_metadata.json), and
    caches per binary hash at ~/.cache/prismaquant/binary-types/.

    Equivalent CLI: --types Q4_K,Q5_K,Q6_K,... to llama-quantize-cost.
    """
    binary = Path(state.binary_path)
    if not binary.exists():
        print(f"  ⚠ binary not found at {binary}; using hard-coded fallback list")
        choices_info: dict[str, FormatInfo] = {
            n: FormatInfo(name=n, recommend=True, in_metadata=False)
            for n in HARDCODED_FALLBACK_FORMATS
        }
    else:
        choices_info = discover_formats(binary, all_formats=state.show_all_formats)
        if not choices_info:
            print(f"  ⚠ discovery returned 0 formats — falling back")
            choices_info = discover_formats(binary, all_formats=True)

    print(f"  → Discovered {len(choices_info)} formats from {binary.name} "
          f"({'all' if state.show_all_formats else 'recommended only'})")

    if inquirer:
        # Format each choice with bpw + family + source for readability
        labels = []
        for name, info in choices_info.items():
            bpw = f"{info.bpw:.2f}" if info.bpw is not None else "  ?"
            tag = f"({bpw} bpw, {info.family}, {info.source})"
            note = f"  ← {info.note}" if info.note and info.recommend else ""
            labels.append((name, f"{name:<12} {tag}{note}"))
        state.formats = inquirer.checkbox(
            message="Allocator format whitelist:",
            choices=[
                Choice(name, name=label, enabled=(name in state.formats or
                                                  (not state.formats and choices_info[name].recommend)))
                for name, label in labels
            ],
            instruction="(space to toggle, enter to confirm)",
        ).execute()
    print(f"  → Formats: {','.join(state.formats)}")


# ─────────────────────────────────────────────────────────────────────────────
# Screen 4: priority + budget
# ─────────────────────────────────────────────────────────────────────────────

def screen_4_priority_budget(state: WizardState) -> None:
    """
    SCREEN 4 — PRIORITY + BUDGET

    ┌──────────────────────────────────────────────────────────────┐
    │ prismaquant-wizard — Step 4/4: Priority + budget             │
    ├──────────────────────────────────────────────────────────────┤
    │ Priority XYZ — X=PPL weight, Y=PP weight, Z=TG weight (0-9)  │
    │   ( ) 900 — pure PPL (best quality)                          │
    │   (•) 522 — PPL primary, balanced PP/TG  ← Recommended       │
    │   ( ) 333 — fully balanced                                   │
    │   ( ) 252 — PP-favoring                                      │
    │   ( ) 225 — TG-favoring                                      │
    │   ( ) (custom)                                               │
    │                                                              │
    │ Target budget: [——————•——————] 5.25 GB                       │
    │   range: 2.0 — 16.0   |  band: ±0.25 GB                      │
    │                                                              │
    │ Output filename: Qwopus3.5-9B-v3.5-PQ5.25-522.gguf            │
    │                                                              │
    │  [ Run pipeline ]   [ Back ]   [ Save as preset ]            │
    └──────────────────────────────────────────────────────────────┘

    Equivalent CLI: --priority 522 --budget-gb 5.25 --budget-band-gb 0.25
    to scripts/allocator.py.
    """
    if inquirer:
        choice = inquirer.select(
            message="Priority preset:",
            choices=list(PRIORITY_PRESETS.keys()),
            default="522 — PPL primary, balanced PP/TG",
        ).execute()
        if PRIORITY_PRESETS[choice] is None:
            state.priority = inquirer.text(
                message="Custom priority (3 digits, e.g. 711):",
                default=state.priority,
                validate=lambda s: s.isdigit() and len(s) == 3,
            ).execute()
        else:
            state.priority = PRIORITY_PRESETS[choice]
        state.budget_gb = float(inquirer.text(
            message="Target budget (GB):",
            default=str(state.budget_gb),
            validate=lambda s: 0.5 < float(s) < 100,
        ).execute())
    state.output_name = state.auto_output_name()
    print(f"  → Output: {state.output_name}")
    print(f"  → CLI: scripts/allocator.py --priority {state.priority} "
          f"--budget-gb {state.budget_gb} --budget-band-gb {state.budget_band_gb}")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runner
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(state: WizardState, dry_run: bool = False,
                 output_root: Optional[Path] = None,
                 keep_work: bool = False) -> None:
    """
    Run the prismaquant build pipeline derived from wizard state.

    A→I stages handled by pipeline_runner.run_full_pipeline(). Wizard's
    job ends here; pipeline_runner does subprocess + logging.
    """
    print()
    print("=" * 66)
    print(" Pipeline plan (dry-run)" if dry_run else " Running pipeline")
    print("=" * 66)
    print(f"  HF model:    {state.hf_model}@{state.hf_revision}")
    print(f"  Calibration: {state.calibration}")
    print(f"  Formats:     {','.join(state.formats)}")
    print(f"  Priority:    {state.priority}  Budget: {state.budget_gb} GB")
    print(f"  Output:      {state.output_name}")
    print()
    if dry_run:
        print("  → pipeline_runner would be invoked with the above settings.")
        print("  → Re-run without --dry-run to execute.")
        return

    # Lazy import to avoid circulars + keep wizard.py importable when pipeline
    # deps (huggingface_hub, etc.) aren't installed.
    try:
        from .pipeline_runner import PipelineConfig, run_full_pipeline
    except ImportError:
        from pipeline_runner import PipelineConfig, run_full_pipeline  # type: ignore

    cfg = PipelineConfig(
        hf_model=state.hf_model,
        hf_revision=state.hf_revision,
        binary=Path(state.binary_path) if state.binary_path else None,
        calibration=Path(state.calibration) if state.calibration else None,
        output_root=output_root or Path.home() / "prismaquant-builds",
        budget_gb=state.budget_gb,
        priority=state.priority,
        formats=state.formats,
    )
    run_full_pipeline(cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="prismaquant interactive wizard (stub) — interactive TUI for "
                    "customizing prismaquant GGUF builds. Wraps tools/prismaquant/"
                    "run-pipeline.sh in any prismaquant-enabled fork.")
    p.add_argument("--dry-run", action="store_true",
                   help="walk screens, print CLI plan, don't execute")
    p.add_argument("--resume", type=Path,
                   help="resume from saved state file")
    p.add_argument("--save", type=Path,
                   help="save state to file at end (for later --resume)")
    p.add_argument("--all-formats", action="store_true",
                   help="show every format the binary supports (not just recommended)")
    add_path_args(p)   # registers --binary, --output, --keep-work, --hf-cache, etc.
    args = p.parse_args(argv)

    if inquirer is None:
        print("ERROR: InquirerPy not installed. Run:", file=sys.stderr)
        print("       pip install --user --break-system-packages "
              "InquirerPy huggingface_hub PyYAML", file=sys.stderr)
        return 1

    # Resolve binary via 3-tier search (--binary → $PATH → build dirs → error)
    try:
        binary = find_binary("quantize", user_path=args.binary)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"  binary: {binary}")

    state = WizardState()
    if args.resume and args.resume.exists():
        state = WizardState(**json.loads(args.resume.read_text()))
        print(f"Resumed from {args.resume}")
    state.binary_path = str(binary)
    state.show_all_formats = args.all_formats

    print("\n=== prismaquant-wizard (stub) ===\n")
    screen_1_model(state)
    print()
    screen_2_calibration(state)
    print()
    screen_3_formats(state)
    print()
    screen_4_priority_budget(state)

    run_pipeline(state, dry_run=args.dry_run)

    if args.save:
        args.save.write_text(json.dumps(asdict(state), indent=2))
        print(f"\nState saved to {args.save}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
