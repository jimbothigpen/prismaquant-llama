"""
prismaquant-llama — unified CLI.

Three subcommands:

    prismaquant-llama calibrate {system|model} INPUT [flags]
        Measure per-format size/PPL/throughput for either the system default
        perf table or a model-specific one.

    prismaquant-llama run INPUT [flags]
        Execute the full prismaquant pipeline (probe → imatrix → costs →
        bridge → allocate → quantize → eval). INPUT must be safetensors
        (HF id or on-disk directory).

    prismaquant-llama explore INPUT --budgets ... --priorities ... [flags]
        Run A–F (cached on subsequent calls), then sweep the cartesian
        (budgets × priorities) product through the allocator. Outputs a
        matrix of predicted size/ΔPPL/TG/PP per cell — no GGUF produced.
        Useful for picking a (budget, priority) before committing to a
        full `run`.

Defaults come from ~/.prismaquant-llama/config.toml — auto-installed from
the shipped default on first run. Edit by hand. CLI flags always win.
"""

from __future__ import annotations
import sys
from typing import Optional

from . import __version__


def _print_help() -> None:
    print(__doc__.strip())
    print()
    print("Subcommands:")
    print("  calibrate   measure per-format perf data (writes calibration/)")
    print("  run         run the full prismaquant pipeline (writes ggufs/)")
    print("  explore     sweep (budget × priority) without producing a GGUF")
    print()
    print("Run `prismaquant-llama <subcommand> --help` for per-subcommand options.")


def main(argv: Optional[list[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]

    # First-run bootstrap: drop the starter config at the standard path
    # so even `prismaquant-llama run --help` triggers the install. Skip if
    # the user is explicitly pointing at an alternative via --config.
    if "--config" not in argv:
        from .config import install_default_config, DEFAULT_CONFIG_PATH
        if install_default_config():
            print(f"[prismaquant-llama] wrote starter config → {DEFAULT_CONFIG_PATH}",
                  file=sys.stderr)
            print(f"[prismaquant-llama] edit it to set defaults; CLI flags "
                  f"override per run.", file=sys.stderr)

    if not argv or argv[0] in ("-h", "--help", "help"):
        _print_help()
        return 0
    if argv[0] in ("-V", "--version"):
        print(f"prismaquant-llama {__version__}")
        return 0

    cmd = argv[0]
    if cmd == "calibrate":
        from . import calibration
        return calibration.main(argv[1:])
    if cmd == "run":
        from . import pipeline_runner
        return pipeline_runner.main(argv[1:])
    if cmd == "explore":
        from . import explore
        return explore.main(argv[1:])

    print(f"prismaquant-llama: unknown subcommand: {cmd}", file=sys.stderr)
    print("Try `prismaquant-llama --help`.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
