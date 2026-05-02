"""
prismaquant-llama — unified CLI dispatcher.

Single entry point that delegates to one of the component modules:

    prismaquant-llama wizard       interactive TUI (default if no subcommand)
    prismaquant-llama discover     auto-discover formats from a binary
    prismaquant-llama calibrate    empirical calibration (quick/deep/ingest)
    prismaquant-llama paths        inspect path layout / discover binaries
    prismaquant-llama pipeline     direct pipeline exec without the TUI

Each subcommand's flags are forwarded verbatim to that module's `main()`.
This keeps the existing per-module CLIs working while presenting a
single discoverable entry point (mirrors `git`, `docker`, `gh`).
"""

from __future__ import annotations
import argparse
import sys
from typing import Optional


SUBCOMMANDS = {
    "wizard":    ("interactive TUI for customizing prismaquant GGUF builds (default)",
                  "prismaquant_llama.wizard"),
    "discover":  ("auto-discover supported formats from a llama-quantize binary",
                  "prismaquant_llama.format_discovery"),
    "calibrate": ("empirical calibration: quick / deep / ingest",
                  "prismaquant_llama.calibration"),
    "paths":     ("inspect path layout and discover llama.cpp tool binaries",
                  "prismaquant_llama.paths"),
    "pipeline":  ("direct pipeline exec (run-pipeline.sh) without the TUI",
                  "prismaquant_llama.pipeline_runner"),
}


def _print_root_help() -> None:
    print(__doc__.strip())
    print()
    print("Subcommands:")
    for name, (desc, _) in SUBCOMMANDS.items():
        print(f"  {name:<11} {desc}")
    print()
    print("Run `prismaquant-llama <subcommand> --help` for per-subcommand options.")


def main(argv: Optional[list[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]

    # Bare invocation → run the wizard
    if not argv:
        from prismaquant_llama import wizard
        return wizard.main([])

    first = argv[0]

    # Top-level help / version
    if first in ("-h", "--help", "help"):
        _print_root_help()
        return 0
    if first in ("-V", "--version"):
        from prismaquant_llama import __version__
        print(f"prismaquant-llama {__version__}")
        return 0

    if first not in SUBCOMMANDS:
        # Unknown subcommand — print help to stderr, return 2 (argparse-compatible)
        print(f"prismaquant-llama: unknown subcommand: {first}", file=sys.stderr)
        print(f"  Try `prismaquant-llama --help` for the list of subcommands.",
              file=sys.stderr)
        return 2

    # Dispatch — import the module lazily so import errors in unused subcommands
    # don't break the whole CLI.
    _, module_path = SUBCOMMANDS[first]
    try:
        import importlib
        module = importlib.import_module(module_path)
    except ImportError as e:
        print(f"prismaquant-llama: failed to import {module_path}: {e}", file=sys.stderr)
        if first == "pipeline":
            print(f"  (Subcommand `pipeline` is not yet implemented — see "
                  f"src/pipeline/README.md for direct script invocation.)",
                  file=sys.stderr)
        return 1

    if not hasattr(module, "main"):
        print(f"prismaquant-llama: {module_path} has no main(); cannot dispatch",
              file=sys.stderr)
        return 1

    return module.main(argv[1:])


if __name__ == "__main__":
    sys.exit(main())
