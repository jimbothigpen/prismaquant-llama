"""
prismaquant-llama show-frontier — display Stage-K size/PPL sweep results.

Stage K writes ``summary-PQ{B}{suffix}.json`` per run, with all
swept-priority candidates and the winner. This subcommand renders it as
a sorted table with Pareto-frontier marks (`*`) and the winner mark
(`★`), so users can see the size/quality curve, not just the picked
recipe.

Usage:
    prismaquant-llama show-frontier INPUT
        Print every summary found for INPUT's most recent run.

    prismaquant-llama show-frontier INPUT --budget 25
        Narrow to one budget. Without ``--budget``, all summaries are shown.

    prismaquant-llama show-frontier INPUT --run Qwen3.5-4B-20260515-103000
        Pick a specific historical run instead of the latest.

    prismaquant-llama show-frontier INPUT --all-runs
        Print every run's summaries, not just the latest.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .config import load_config
from .input_resolver import sanitize_model_name


def _find_run_dirs(layout_base: Path, model_name: str,
                   run_label: Optional[str]) -> list[Path]:
    work = layout_base / "work"
    if not work.exists():
        return []
    if run_label is not None:
        d = work / run_label
        return [d] if d.is_dir() else []
    matches = sorted([d for d in work.glob(f"{model_name}-*") if d.is_dir()],
                     key=lambda p: p.stat().st_mtime)
    return matches


def _render_summary(summary_path: Path) -> str:
    data = json.loads(summary_path.read_text())
    candidates = data.get("candidates", [])
    if not candidates:
        return f"== {summary_path.name} ==  (empty)\n"

    # Sort by size for the curve view.
    rows = sorted(candidates, key=lambda r: r["size_gb"])
    winner_p = data.get("winner_priority")

    lines = []
    lines.append(f"== {summary_path.name} ==")
    lines.append(f"  path             : {summary_path}")
    lines.append(f"  budget_gb        : {data.get('budget_gb'):.2f}")
    lines.append(f"  user_priority    : {data.get('user_priority')}")
    lines.append(f"  winner_priority  : {winner_p}")
    lines.append(f"  winner_ppl       : {data.get('winner_ppl'):.4f}")
    lines.append(f"  winner_size_gb   : {data.get('winner_size_gb'):.2f}")
    n_pareto = sum(1 for r in rows if r.get("is_pareto"))
    lines.append(f"  pareto_frontier  : {n_pareto}/{len(rows)}")
    lines.append("")
    lines.append(f"  {'priority':<10} {'size_gb':>9} {'ppl':>10}  "
                 f"{'pareto':<7} {'winner':<6}")
    for r in rows:
        pareto = "*" if r.get("is_pareto") else " "
        winner = "★" if r["priority"] == winner_p else " "
        lines.append(f"  {r['priority']:<10} {r['size_gb']:>9.2f} "
                     f"{r['ppl']:>10.4f}  {pareto:<7} {winner:<6}")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="prismaquant-llama show-frontier",
        description="Display Stage-K Pareto frontier for an input.")
    p.add_argument("input", metavar="INPUT",
                   help="Anything resolvable to a model name: an HF id, a "
                        "safetensors dir, a BF16 GGUF path, or just the bare "
                        "sanitized model name (e.g. 'Qwen3.5-4B'). The input "
                        "does NOT need to still exist on disk — only the "
                        "historical work directory does.")
    p.add_argument("--config", type=Path, default=None,
                   help="path to config.toml (default: "
                        "~/.prismaquant-llama/config.toml)")
    p.add_argument("--base", type=Path, default=None,
                   help="override [prismaquant-llama].base for one invocation")
    p.add_argument("--budget", type=int, default=None,
                   help="restrict to one PQ budget (e.g. 25); default: all")
    p.add_argument("--run", default=None,
                   help="exact run label (e.g. Qwen3.5-4B-20260515-103000); "
                        "default: latest run for the model")
    p.add_argument("--all-runs", action="store_true",
                   help="print frontiers for every run, not just the latest")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    base = (args.base or cfg.base).expanduser().resolve()

    model_name = sanitize_model_name(args.input)
    runs = _find_run_dirs(base, model_name, args.run)
    if not runs:
        target = args.run or f"{model_name}-*"
        print(f"show-frontier: no run directory found at {base / 'work' / target}",
              file=sys.stderr)
        return 1

    if not args.all_runs and args.run is None:
        runs = runs[-1:]  # latest by mtime

    if args.budget is None:
        glob_pat = "summary-PQ*.json"
    else:
        glob_pat = f"summary-PQ{args.budget}*.json"

    n_found = 0
    for run_dir in runs:
        stage_k = run_dir / "stage-k"
        if not stage_k.exists():
            continue
        summaries = sorted(stage_k.glob(glob_pat))
        if not summaries:
            continue
        print(f"# run: {run_dir.name}")
        print()
        for s in summaries:
            print(_render_summary(s))
            n_found += 1

    if n_found == 0:
        print(f"show-frontier: no Stage-K summaries matched in "
              f"{[r.name for r in runs]} (glob={glob_pat})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
