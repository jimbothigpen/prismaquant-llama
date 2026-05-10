"""
prismaquant-llama explore — sweep (budget × priority) without producing GGUFs.

Runs the same A–F preparation as `run` (download → convert → probe → imatrix
→ costs → bridge), then for each (budget, priority) cell calls the allocator
once and reports predicted size, ΔPPL, TG, and PP. Skips stages H+I, so a
full sweep finishes in seconds once A–F is cached.

Predicted ΔPPL/TG/PP come from the same calibration JSON the allocator uses
for TPS (model-specific > system-default), via a size-weighted aggregation
over per-format `ppl_delta_vs_f16` / `tg` / `pp`. This is an approximation
(treats per-format effects as additive across mixed recipes) but uses real
backend-measured numbers — so backend-specific quality regressions like the
Vulkan IQ4_KSS divergence become visible in the matrix.

Usage:
    prismaquant-llama explore INPUT \\
        --budgets 22,25,28,32 \\
        --priorities 111,522,252,225,323 \\
        [--output-csv explore.csv] [--output-md explore.md]
"""

from __future__ import annotations
import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Optional

from .config import Config, load_config, subprocess_env, resolve_corpus
from .input_resolver import resolve as resolve_input
from .paths import Layout
from . import pipeline_runner as pr


def _bundled_script_path(name: str) -> Path:
    return Path(pr.__file__).parent / "scripts" / name


def _load_perf_data(perf_file: Optional[Path]) -> dict:
    """Load calibration JSON; strip leading-underscore metadata keys."""
    if perf_file is None or not perf_file.exists():
        return {}
    with perf_file.open() as f:
        d = json.load(f)
    return {k: v for k, v in d.items() if not k.startswith("_")}


def _predict_metrics(recipe: dict[str, str],
                     costs: dict,
                     perf: dict) -> tuple[float, float, float, int]:
    """Size-weighted aggregate of per-format calibration data over a recipe.

    Returns (predicted_ppl_delta, predicted_tg, predicted_pp, total_size).
    Tensors whose chosen format is missing from `perf` (or whose cost entry
    is missing) contribute 0 to ΔPPL and a neutral 1.0 throughput.
    """
    total_size = 0
    sum_ppl_delta = 0.0
    sum_inv_tg = 0.0
    sum_inv_pp = 0.0
    for t, f in recipe.items():
        if t not in costs or f not in costs[t]:
            continue
        sz = costs[t][f][1]
        total_size += sz
        e = perf.get(f, {})
        ppl_delta = float(e.get("ppl_delta_vs_f16") or 0.0)
        tg = float(e.get("tg") or 1.0)
        pp = float(e.get("pp") or 1.0)
        sum_ppl_delta += sz * ppl_delta
        sum_inv_tg += sz / max(tg, 1e-9)
        sum_inv_pp += sz / max(pp, 1e-9)
    if total_size == 0:
        return 0.0, 0.0, 0.0, 0
    return (sum_ppl_delta / total_size,
            total_size / sum_inv_tg,
            total_size / sum_inv_pp,
            total_size)


def _prep_costs_for_allocator(raw_costs: dict, gguf_meta: dict, allow: set[str],
                               floor_rules: dict[str, float]) -> dict:
    """Apply the same propagation + allow-types + floor-bpw filters that
    `stage_g_allocate` applies before calling `bisect_lambda`.

    Mutates `raw_costs` in place (caller should pass a fresh copy if needed).
    Returns the filtered costs dict."""
    # Import bundled allocator helpers
    sys.path.insert(0, str(_bundled_script_path("allocator.py").parent))
    from allocator import (detect_layer_types, propagate_costs,
                            auto_pick_exemplar_layers)

    gguf_names = set(gguf_meta.keys())
    layer_type = detect_layer_types(gguf_meta)
    exemplars = auto_pick_exemplar_layers(layer_type)
    costs, _ = propagate_costs(raw_costs, gguf_names, exemplars,
                                layer_type=layer_type)

    # allow-types
    for t in list(costs):
        costs[t] = {f: v for f, v in costs[t].items() if f in allow}
        if not costs[t]:
            del costs[t]

    # floor-bpw (keep only formats whose bpw >= applicable_min)
    if floor_rules:
        compiled = [(re.compile(p), float(m)) for p, m in floor_rules.items()]
        for t in list(costs):
            applicable = None
            for pat, m in compiled:
                if pat.search(t) and (applicable is None or m > applicable):
                    applicable = m
            if applicable is None:
                continue
            costs[t] = {f: v for f, v in costs[t].items() if v[3] >= applicable}
            if not costs[t]:
                raise SystemExit(
                    f"[explore] floor-bpw rule (min={applicable}) removed all "
                    f"candidates for tensor {t!r}.")
    return costs


def explore_sweep(cfg: Config, resolved, imatrix_override: Optional[str],
                  budgets_pct: list[int], priorities: list[str],
                  out_csv: Optional[Path], out_md: Optional[Path],
                  assume_yes: bool) -> int:
    """Run A–F (cached if present), then sweep (budgets × priorities).

    Output: writes CSV / MD if paths given. Always prints a Markdown table
    to stdout."""
    layout = Layout.for_run(base=cfg.base, model_name=resolved.model_name)
    layout.make()

    # Resolve corpora (download if URL — same behaviour as `run`)
    ppl_corpus, _ppl_dl = resolve_corpus(cfg, "ppl")
    imatrix_corpus, _im_dl = resolve_corpus(cfg, "imatrix")

    # A: safetensors
    if resolved.kind == "hf":
        safetensors_dir = pr.download_hf(cfg, layout, resolved.hf_id,
                                          resolved.model_name)
    elif resolved.kind == "safetensors_dir":
        safetensors_dir = resolved.safetensors_dir
        pr._log(layout, "A", f"A. using on-disk safetensors at {safetensors_dir}")
    else:
        raise SystemExit("explore requires safetensors input")

    # B: BF16
    bf16_path = pr.convert_to_bf16(cfg, layout, safetensors_dir,
                                    resolved.model_name)

    # C: probe; D: imatrix
    probe_path = pr.stage_c_probe(cfg, layout, safetensors_dir, imatrix_corpus,
                                   resolved.model_name)
    if imatrix_override:
        imatrix_path = pr._resolve_imatrix_override(cfg, layout, imatrix_override)
        pr._log(layout, "D", f"D. using --imatrix override: {imatrix_path}")
    else:
        imatrix_path = pr.stage_d_imatrix(cfg, layout, bf16_path, imatrix_corpus)

    # E: costs; F: bridge
    costs_path = pr.stage_e_costs(cfg, layout, bf16_path, imatrix_path)
    bridge_path, mtp_tensors_path = pr.stage_f_bridge(cfg, layout, probe_path,
                                                      safetensors_dir)

    # Load allocator inputs
    sys.path.insert(0, str(_bundled_script_path("allocator.py").parent))
    from allocator import (load_costs, parse_priority, compute_norms,
                            bisect_lambda, read_gguf_tensor_meta,
                            apply_mtp_format_override, _recompute_size_loss)

    mtp_names: set[str] = set()
    if mtp_tensors_path is not None and cfg.mtp_format:
        with open(mtp_tensors_path) as f:
            mtp_names = set(json.load(f))
        print(f"[explore] mtp override active: {len(mtp_names)} tensor(s) "
              f"will be pinned to {cfg.mtp_format} per cell", flush=True)

    raw_costs = load_costs(str(costs_path))
    with bridge_path.open() as f:
        bridge = json.load(f)
    fisher = bridge["h_trace"]
    gguf_meta = read_gguf_tensor_meta(str(bf16_path))

    perf_file = pr.find_perf_file(layout, resolved.model_name)
    perf = _load_perf_data(perf_file)
    if not perf:
        print("[explore] WARN: no calibration perf data found — predicted "
              "ΔPPL/TG/PP columns will be 0/1.0", file=sys.stderr)

    # Same pinned + floor-bpw + allow-types as stage_g_allocate
    pinned = {"output.weight": "Q6_K", "token_embd.weight": "Q8_0"}
    floor_rules = {r"^blk\..*\.attn_(q|k|v|qkv|gate|output)\.weight$": 4.0}
    allow = set(cfg.quants)

    bf16_gb = bf16_path.stat().st_size / 1024**3

    # Sweep
    rows = []
    for pri in priorities:
        weights = parse_priority(pri)
        for bpct in budgets_pct:
            budget_gb = round(bf16_gb * bpct / 100, 3)
            budget_bytes = int(budget_gb * 1024**3)

            costs = _prep_costs_for_allocator({k: dict(v) for k, v in raw_costs.items()},
                                               gguf_meta, allow, floor_rules)
            norms = compute_norms(costs, fisher, perf)
            lam, recipe, total_size, total_loss = bisect_lambda(
                fisher, costs, pinned, budget_bytes,
                weights=weights, tps=perf, norms=norms,
                band_bytes=int(0.25 * 1024**3))

            if mtp_names:
                recipe, _ = apply_mtp_format_override(recipe, mtp_names,
                                                      cfg.mtp_format)
                total_size, total_loss = _recompute_size_loss(recipe, costs, fisher)

            pred_dppl, pred_tg, pred_pp, _ = _predict_metrics(recipe, costs, perf)

            from collections import Counter
            fmt_counts = Counter(recipe.values())
            top_fmts = ", ".join(f"{f}={c}" for f, c in
                                 sorted(fmt_counts.items(), key=lambda kv: -kv[1])[:4])
            rows.append({
                "budget_pct": bpct,
                "priority": pri,
                "budget_GB": budget_gb,
                "actual_GB": total_size / 1024**3,
                "delta_GB": (total_size - budget_bytes) / 1024**3,
                "predicted_dppl": pred_dppl,
                "predicted_tg": pred_tg,
                "predicted_pp": pred_pp,
                "lambda": lam,
                "loss_surrogate": total_loss,
                "top_formats": top_fmts,
            })

    # Render Markdown table
    headers = ("budget", "priority", "actual_GB", "Δ_GB",
               "pred_ΔPPL", "pred_TG", "pred_PP", "top_formats")
    md_lines = ["| " + " | ".join(headers) + " |",
                "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        md_lines.append("| {} | {} | {:.2f} | {:+.2f} | {:+.3f} | {:.2f} | {:.2f} | {} |".format(
            f"{r['budget_pct']}%", r["priority"], r["actual_GB"], r["delta_GB"],
            r["predicted_dppl"], r["predicted_tg"], r["predicted_pp"], r["top_formats"]))
    md_table = "\n".join(md_lines)
    print()
    print(md_table)
    print()

    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[explore] wrote CSV → {out_csv}")

    if out_md:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(md_table + "\n")
        print(f"[explore] wrote Markdown → {out_md}")

    return 0


def add_explore_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("input", metavar="INPUT",
                   help="HuggingFace id or on-disk safetensors directory")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--libs", type=Path, default=None)
    p.add_argument("--base", type=Path, default=None)
    p.add_argument("--path", type=Path, default=None)
    p.add_argument("--quants", default=None,
                   help="comma-separated allowed-quants list (default: from config)")
    p.add_argument("--budgets", default="22,25,28,32",
                   help="comma-separated budget percentages to sweep "
                        "(default: 22,25,28,32)")
    p.add_argument("--priorities", default="111,522,252,225,323",
                   help="comma-separated priority specs to sweep "
                        "(default: 111,522,252,225,323)")
    p.add_argument("--ppl-corpus", default=None)
    p.add_argument("--imatrix-corpus", default=None)
    p.add_argument("--imatrix", default=None)
    p.add_argument("--ppl-chunks", type=int, default=None)
    p.add_argument("--imatrix-chunks", type=int, default=None)
    p.add_argument("--convert-script", type=Path, default=None)
    p.add_argument("--output-csv", type=Path, default=None,
                   help="write the sweep matrix as CSV to this path")
    p.add_argument("--output-md", type=Path, default=None,
                   help="write the sweep matrix as Markdown to this path")
    p.add_argument("--yes", "-y", action="store_true")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="prismaquant-llama explore",
                                description=__doc__.split("\n\n")[0])
    add_explore_args(p)
    args = p.parse_args(argv)

    cfg = pr.cfg_from_args(args)
    resolved = resolve_input(args.input, allow_gguf=False)

    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]
    priorities = [x.strip() for x in args.priorities.split(",") if x.strip()]
    if not budgets or not priorities:
        print("explore: --budgets and --priorities must each have ≥1 value",
              file=sys.stderr)
        return 2

    try:
        return explore_sweep(cfg, resolved, args.imatrix,
                              budgets, priorities,
                              args.output_csv, args.output_md,
                              assume_yes=args.yes)
    except (SystemExit, FileNotFoundError, ValueError) as e:
        print(f"\nFAIL: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
