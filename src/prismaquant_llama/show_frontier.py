"""
prismaquant-llama show-frontier — display Stage-K size/PPL sweep results.

Stage K writes ``summary-PQ{B}{suffix}.json`` per run, with all
swept-priority candidates and the winner. This subcommand renders it as
a sorted table with Pareto-frontier marks (`*`) and the winner mark
(`★`), so users can see the size/quality curve, not just the picked
recipe.

The default output is a human-readable text table on stdout. Pass any of
``--output-csv``, ``--output-json``, or ``--output-md`` (mirroring
`explore`'s flags) to additionally emit machine-readable forms. Multiple
output flags may be combined; stdout text is unchanged either way.

Usage:
    prismaquant-llama show-frontier INPUT
        Print every summary found for INPUT's most recent run.

    prismaquant-llama show-frontier INPUT --budget 25
        Narrow to one budget. Without ``--budget``, all summaries are shown.

    prismaquant-llama show-frontier INPUT --run Qwen3.5-4B-20260515-103000
        Pick a specific historical run instead of the latest.

    prismaquant-llama show-frontier INPUT --all-runs
        Print every run's summaries, not just the latest.

    prismaquant-llama show-frontier INPUT --output-csv frontier.csv \\
        --output-json frontier.json --output-md frontier.md
        Also emit machine-readable forms alongside the stdout table.
"""

from __future__ import annotations
import argparse
import csv
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


def _summary_record(run_dir: Path, summary_path: Path) -> dict:
    """Parse one summary-PQ*.json into the in-memory schema shared by all
    rendering paths (text/MD/CSV/JSON)."""
    data = json.loads(summary_path.read_text())
    candidates = data.get("candidates", [])
    rows = sorted(candidates, key=lambda r: r["size_gb"])
    winner_p = data.get("winner_priority")
    fisher = summary_path.stem.endswith("-fisher")
    return {
        "run": run_dir.name,
        "summary_file": summary_path.name,
        "summary_path": str(summary_path),
        "budget_gb": data.get("budget_gb"),
        "user_priority": data.get("user_priority"),
        "winner_priority": winner_p,
        "winner_ppl": data.get("winner_ppl"),
        "winner_size_gb": data.get("winner_size_gb"),
        "fisher": fisher,
        "candidates": [
            {
                "priority": r["priority"],
                "size_gb": r["size_gb"],
                "ppl": r["ppl"],
                "is_pareto": bool(r.get("is_pareto")),
                "is_winner": r["priority"] == winner_p,
                "recipe": r.get("recipe"),
                "candidate_gguf": r.get("candidate_gguf"),
            }
            for r in rows
        ],
    }


def _render_text(rec: dict) -> str:
    rows = rec["candidates"]
    if not rows:
        return f"== {rec['summary_file']} ==  (empty)\n"
    n_pareto = sum(1 for r in rows if r["is_pareto"])
    lines = [
        f"== {rec['summary_file']} ==",
        f"  path             : {rec['summary_path']}",
        f"  budget_gb        : {rec['budget_gb']:.2f}",
        f"  user_priority    : {rec['user_priority']}",
        f"  winner_priority  : {rec['winner_priority']}",
        f"  winner_ppl       : {rec['winner_ppl']:.4f}",
        f"  winner_size_gb   : {rec['winner_size_gb']:.2f}",
        f"  pareto_frontier  : {n_pareto}/{len(rows)}",
        "",
        f"  {'priority':<10} {'size_gb':>9} {'ppl':>10}  "
        f"{'pareto':<7} {'winner':<6}",
    ]
    for r in rows:
        pareto = "*" if r["is_pareto"] else " "
        winner = "★" if r["is_winner"] else " "
        lines.append(f"  {r['priority']:<10} {r['size_gb']:>9.2f} "
                     f"{r['ppl']:>10.4f}  {pareto:<7} {winner:<6}")
    lines.append("")
    return "\n".join(lines)


def _render_markdown(rec: dict) -> str:
    """Render one summary as a Markdown section + table."""
    rows = rec["candidates"]
    header_lines = [
        f"### {rec['summary_file']}",
        "",
        f"- path: `{rec['summary_path']}`",
        f"- budget_gb: {rec['budget_gb']:.2f}",
        f"- user_priority: `{rec['user_priority']}`",
        f"- winner_priority: `{rec['winner_priority']}`",
        f"- winner_ppl: {rec['winner_ppl']:.4f}",
        f"- winner_size_gb: {rec['winner_size_gb']:.2f}",
        f"- pareto_frontier: "
        f"{sum(1 for r in rows if r['is_pareto'])}/{len(rows)}",
        "",
    ]
    if not rows:
        header_lines.append("_(no candidates)_")
        return "\n".join(header_lines) + "\n"
    header_lines += [
        "| priority | size_gb | ppl | pareto | winner |",
        "|---|---:|---:|:---:|:---:|",
    ]
    for r in rows:
        pareto = "*" if r["is_pareto"] else ""
        winner = "★" if r["is_winner"] else ""
        header_lines.append(
            f"| `{r['priority']}` | {r['size_gb']:.2f} | {r['ppl']:.4f} "
            f"| {pareto} | {winner} |"
        )
    return "\n".join(header_lines) + "\n"


def _write_csv(records: list[dict], out_path: Path) -> None:
    """One row per candidate, joined with its parent summary's metadata."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run", "summary_file", "budget_gb", "fisher",
        "user_priority", "winner_priority",
        "priority", "size_gb", "ppl", "is_pareto", "is_winner",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rec in records:
            for cand in rec["candidates"]:
                w.writerow({
                    "run": rec["run"],
                    "summary_file": rec["summary_file"],
                    "budget_gb": rec["budget_gb"],
                    "fisher": rec["fisher"],
                    "user_priority": rec["user_priority"],
                    "winner_priority": rec["winner_priority"],
                    "priority": cand["priority"],
                    "size_gb": cand["size_gb"],
                    "ppl": cand["ppl"],
                    "is_pareto": cand["is_pareto"],
                    "is_winner": cand["is_winner"],
                })


def _write_json(records: list[dict], out_path: Path) -> None:
    """One aggregated JSON document covering every rendered summary."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"schema_version": 1, "frontiers": records}
    out_path.write_text(json.dumps(doc, indent=2) + "\n")


def _write_md(records: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sections = []
    last_run = None
    for rec in records:
        if rec["run"] != last_run:
            sections.append(f"## run: {rec['run']}\n")
            last_run = rec["run"]
        sections.append(_render_markdown(rec))
    out_path.write_text("\n".join(sections))


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
    p.add_argument("--output-csv", type=Path, default=None,
                   help="also write one-row-per-candidate CSV to this path")
    p.add_argument("--output-json", type=Path, default=None,
                   help="also write the aggregated frontier(s) as JSON")
    p.add_argument("--output-md", type=Path, default=None,
                   help="also write the frontier(s) as a Markdown document")
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

    records: list[dict] = []
    for run_dir in runs:
        stage_k = run_dir / "stage-k"
        if not stage_k.exists():
            continue
        summaries = sorted(stage_k.glob(glob_pat))
        for s in summaries:
            records.append(_summary_record(run_dir, s))

    if not records:
        print(f"show-frontier: no Stage-K summaries matched in "
              f"{[r.name for r in runs]} (glob={glob_pat})", file=sys.stderr)
        return 1

    # Stdout: always text, grouped by run (preserves prior behaviour).
    last_run = None
    for rec in records:
        if rec["run"] != last_run:
            print(f"# run: {rec['run']}")
            print()
            last_run = rec["run"]
        print(_render_text(rec))

    if args.output_csv:
        _write_csv(records, args.output_csv)
        print(f"[show-frontier] wrote CSV → {args.output_csv}")
    if args.output_json:
        _write_json(records, args.output_json)
        print(f"[show-frontier] wrote JSON → {args.output_json}")
    if args.output_md:
        _write_md(records, args.output_md)
        print(f"[show-frontier] wrote Markdown → {args.output_md}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
