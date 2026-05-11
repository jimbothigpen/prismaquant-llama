"""
Stage F+ — pre-condition BF16 weights of ≥4-bit recipe entries.

Slots between Stage G (allocator) and Stage H (quantize) in `pipeline_runner`.
Reads the recipe + costs CSV, decides per-tensor whether the chosen format is
at or above the bpw floor, records the decision in a manifest, and produces a
preconditioned BF16 GGUF that Stage H consumes.

P1 status (current): skeleton only. The pc-GGUF is a hardlink (or cross-fs
copy fallback) of the source BF16 — no tensor data is modified. The decision
walk and manifest format are wired up so P2-P5 can plug in AWQ / GPTQ /
scale-sweep / HALO without re-shaping the pipeline.


"""

from __future__ import annotations
import csv
import json
import os
import shutil
import time
from pathlib import Path
from typing import Optional

from .config import Config
from .paths import Layout


# Canonical bits-per-weight for GGUF formats the bundled allocator may pick.
# Used when the costs.csv lacks a row for (tensor, chosen_fmt) — e.g.,
# tensors assigned by shape-propagation from exemplar layers.
_BPW_FALLBACK = {
    "BF16": 16.0, "F16": 16.0, "F32": 32.0,
    "Q8_0": 8.5,
    "Q6_K": 6.5,
    "Q5_K_M": 5.7, "Q5_K_S": 5.5, "Q5_K": 5.5,
    "Q4_K_M": 4.8, "Q4_K_S": 4.5, "Q4_K": 4.5,
    "IQ4_NL": 4.5, "IQ4_XS": 4.25,
    "Q3_K_L": 3.7, "Q3_K_M": 3.5, "Q3_K_S": 3.4, "Q3_K": 3.4,
    "IQ3_M": 3.7, "IQ3_S": 3.5, "IQ3_XS": 3.3, "IQ3_XXS": 3.1,
}


def _log(layout: Layout, msg: str) -> None:
    """Stage-F+ logger; mirrors pipeline_runner._log's tee-to-file pattern."""
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    log_path = layout.logs_dir / "stage-F+.log"
    if log_path.parent.exists():
        with log_path.open("a") as f:
            f.write(line + "\n")


def _normalize_fmt(fmt) -> str:
    """Recipe entries may be strings or {'type': ...} / {'format': ...} dicts;
    matches `stage_h_quantize`'s defensive handling."""
    if isinstance(fmt, dict):
        fmt = fmt.get("type") or fmt.get("format") or ""
    return str(fmt).strip().upper()


def _load_costs_bpw(costs_path: Path) -> dict[tuple[str, str], float]:
    """Returns {(tensor_name, fmt_upper): bpw}."""
    out: dict[tuple[str, str], float] = {}
    with costs_path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                bpw = float(r["bpw"])
            except (KeyError, ValueError):
                continue
            out[(r["tensor_name"], r["fmt"].upper())] = bpw
    return out


def _resolve_bpw(tensor: str, fmt: str,
                 costs_bpw: dict[tuple[str, str], float]) -> Optional[float]:
    """Prefer measured bpw from costs.csv; fall back to the canonical table;
    return None when the format is unknown (caller treats as skip:unknown)."""
    bpw = costs_bpw.get((tensor, fmt))
    if bpw is not None:
        return bpw
    return _BPW_FALLBACK.get(fmt)


def _link_or_copy(src: Path, dst: Path) -> str:
    """Hardlink when on the same filesystem; copy2 across filesystems.
    Returns "hardlink" or "copy" for the manifest."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def stage_fp_precondition(
    cfg: Config,
    layout: Layout,
    bf16_path: Path,
    recipe_path: Path,
    costs_path: Path,
    model_name: str,
) -> tuple[Path, Optional[Path]]:
    """Stage F+. Returns (bf16_to_feed_stage_H, manifest_path).

    When `cfg.precondition_mode == "off"`, returns `(bf16_path, None)`
    immediately — Stage H reads the original BF16 unchanged. Zero-cost no-op
    so the default pipeline is unaffected.

    Otherwise (P1: "on"), walks the recipe, writes a manifest, hardlinks
    bf16 → `<model>-<REF>-pc.gguf`, and returns the new path. No weights
    are modified in P1.
    """
    if cfg.precondition_mode == "off":
        _log(layout, "F+. precondition mode = off; passthrough (no-op)")
        return bf16_path, None

    pc_path = layout.preconditioned_bf16_path(model_name, cfg.reference_format)
    manifest_path = layout.precondition_manifest_path()

    # Idempotent skip: both outputs already produced by a prior run.
    if pc_path.exists() and manifest_path.exists():
        _log(layout, f"F+. cached (pc-gguf + manifest exist; skip)")
        return pc_path, manifest_path

    _log(layout, f"F+. precondition mode = {cfg.precondition_mode}, "
                 f"bpw_floor = {cfg.precondition_bpw_floor}")

    recipe_data = json.loads(recipe_path.read_text())
    assignments = recipe_data.get("assignments") or recipe_data

    costs_bpw = _load_costs_bpw(costs_path)

    entries: list[dict] = []
    n_skip_lowbit = 0
    n_skip_unknown = 0
    n_would_apply = 0

    for tensor, raw_fmt in assignments.items():
        fmt = _normalize_fmt(raw_fmt)
        bpw = _resolve_bpw(tensor, fmt, costs_bpw)

        if bpw is None:
            reason = "skip:unknown_bpw"
            n_skip_unknown += 1
        elif bpw < cfg.precondition_bpw_floor:
            reason = "skip:lowbit"
            n_skip_lowbit += 1
        else:
            # P2-P5 will replace this with "applied:awq" etc.
            reason = "skip:p1_stub"
            n_would_apply += 1

        entries.append({
            "tensor": tensor,
            "format": fmt,
            "bpw": bpw,
            "reason": reason,
        })

    link_kind = _link_or_copy(bf16_path, pc_path)

    manifest = {
        "schema": 1,
        "phase": "P1",
        "precondition_mode": cfg.precondition_mode,
        "bpw_floor": cfg.precondition_bpw_floor,
        "source_bf16": str(bf16_path),
        "preconditioned_bf16": str(pc_path),
        "link_kind": link_kind,
        "recipe": str(recipe_path),
        "costs": str(costs_path),
        "summary": {
            "n_tensors": len(entries),
            "n_would_apply_in_P2": n_would_apply,
            "n_skip_lowbit": n_skip_lowbit,
            "n_skip_unknown_bpw": n_skip_unknown,
        },
        "entries": entries,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    _log(layout,
         f"F+. {len(entries)} tensors: "
         f"{n_would_apply} would-precondition (P2+), "
         f"{n_skip_lowbit} skip:lowbit, "
         f"{n_skip_unknown} skip:unknown_bpw")
    _log(layout, f"F+. pc-bf16: {pc_path} ({link_kind})")
    _log(layout, f"F+. manifest: {manifest_path}")

    return pc_path, manifest_path
