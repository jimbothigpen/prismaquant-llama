"""
Stage F+ — pre-condition BF16 weights of ≥4-bit recipe entries.

Slots between Stage G (allocator) and Stage H (quantize) in `pipeline_runner`.
Reads the recipe + costs CSV, decides per-tensor whether the chosen format is
at or above the bpw floor, records the decision in a manifest, and produces a
preconditioned BF16 GGUF that Stage H consumes.

P2 status: AWQ proper-fold pass on ≥4-bit Q/K/V (softmax-attn layers) and
gate/up (every layer), folding 1/s into the predecessor RMSNorm γ. Math
identity preserved per fold by applying s to every reader of γ.

Conservative interpretation of "disable below 4 bits": a fold is SKIPPED
for the layer if any reader of γ is sub-floor or is missing activation
cache. The companion-rescale of a sub-floor reader (per prismaquant's
`_awq_fold_layer_predecessors`) is intentionally not used here.

P3-P5 (GPTQ / scale-sweep / HALO) will compose on top of P2's pc-bf16.


"""

from __future__ import annotations
import csv
import json
import mmap
import os
import re
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .config import Config
from .paths import Layout
from .scripts.awq_apply import (
    awq_joint_channel_scale,
    awq_apply_to_weight,
    awq_fold_into_norm,
)
from .scripts.gguf_mmap_patch import (
    parse_gguf_header, GgufHeader,
    read_tensor_fp32, write_tensor_fp32,
)


# Canonical bits-per-weight for GGUF formats the bundled allocator may pick.
# Used when the costs.csv lacks a row for (tensor, chosen_fmt) — e.g.,
# tensors assigned by shape-propagation from exemplar layers.
_BPW_FALLBACK = {
    # Reference dtypes
    "BF16": 16.0, "F16": 16.0, "F32": 32.0,
    # Mainline k-quants
    "Q8_0": 8.5,
    "Q6_K": 6.5,
    "Q5_K_M": 5.7, "Q5_K_S": 5.5, "Q5_K": 5.5,
    "Q4_K_M": 4.8, "Q4_K_S": 4.5, "Q4_K": 4.5,
    "Q3_K_L": 3.7, "Q3_K_M": 3.5, "Q3_K_S": 3.4, "Q3_K": 3.4,
    # Mainline IQ family
    "IQ4_NL": 4.5, "IQ4_XS": 4.25,
    "IQ3_M": 3.7, "IQ3_S": 3.5, "IQ3_XS": 3.3, "IQ3_XXS": 3.1,
    # ik_llama IQ-K family (approx; precise bpw is shape-dependent)
    "IQ4_K": 4.5, "IQ4_KS": 4.25, "IQ4_KSS": 4.0,
    "IQ3_K": 3.4, "IQ3_KS": 3.0,
    # ik_llama ternary-3-bit family
    "TQ3_4S": 3.0625, "TQ3_1S": 3.0,
    # MoE-only FP4 (mainline)
    "MXFP4_MOE": 4.25,
}


# Reverse map: GGUF reader → HF leaf name (used to locate the per-Linear
# activation cache file). Only the AWQ-eligible readers for P2 are listed;
# everything else doesn't fold and doesn't need an HF name.
#
# Pattern -> (hf-leaf-template, leaf-name-for-predecessor-classification).
_AWQ_READER_PATTERNS = [
    (re.compile(r"^blk\.(\d+)\.attn_q\.weight$"),
     "model.layers.{N}.self_attn.q_proj", "q_proj"),
    (re.compile(r"^blk\.(\d+)\.attn_k\.weight$"),
     "model.layers.{N}.self_attn.k_proj", "k_proj"),
    (re.compile(r"^blk\.(\d+)\.attn_v\.weight$"),
     "model.layers.{N}.self_attn.v_proj", "v_proj"),
    (re.compile(r"^blk\.(\d+)\.ffn_gate\.weight$"),
     "model.layers.{N}.mlp.gate_proj", "gate_proj"),
    (re.compile(r"^blk\.(\d+)\.ffn_up\.weight$"),
     "model.layers.{N}.mlp.up_proj", "up_proj"),
]


# Per-layer fold groups: norm name candidates + reader names. The first
# matching norm is the fold partner. Reader names use {N} for layer index.
_FOLD_GROUPS = [
    {
        "kind": "attn",
        "norm_candidates": ["blk.{N}.attn_norm.weight"],
        "readers": ["blk.{N}.attn_q.weight",
                    "blk.{N}.attn_k.weight",
                    "blk.{N}.attn_v.weight"],
    },
    {
        "kind": "ffn",
        # Qwen3.5 hybrid uses `post_attention_norm`; standard llama.cpp
        # dense models use `ffn_norm`. Try both.
        "norm_candidates": ["blk.{N}.post_attention_norm.weight",
                             "blk.{N}.ffn_norm.weight"],
        "readers": ["blk.{N}.ffn_gate.weight",
                    "blk.{N}.ffn_up.weight"],
    },
]


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


def _gguf_reader_to_hf_name(gguf_name: str) -> Optional[str]:
    """Inverse of bridge_probe_to_gguf for the 5 AWQ reader patterns.
    Returns the HF tensor base name (without `.weight`) or None if the
    GGUF name doesn't match a P2-eligible reader pattern.
    """
    for pat, template, _leaf in _AWQ_READER_PATTERNS:
        m = pat.match(gguf_name)
        if m:
            return template.format(N=int(m.group(1)))
    return None


def _actcache_file(act_cache_dir: Path, hf_base_name: str) -> Path:
    """`model.layers.3.self_attn.q_proj` → `<dir>/model__layers__3__self_attn__q_proj.pt`."""
    return act_cache_dir / (hf_base_name.replace(".", "__") + ".pt")


def _load_acts(act_cache_dir: Path, hf_base_name: str
                ) -> Optional[torch.Tensor]:
    """Load `inputs` tensor from the per-Linear activation cache file.
    Returns None on miss (file absent or unexpected format)."""
    p = _actcache_file(act_cache_dir, hf_base_name)
    if not p.exists():
        return None
    try:
        blob = torch.load(p, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if isinstance(blob, dict) and "inputs" in blob:
        return blob["inputs"]
    if isinstance(blob, torch.Tensor):
        return blob
    return None


def _pick_norm(hdr: GgufHeader, candidates: list[str], N: int) -> Optional[str]:
    """First norm-candidate template that resolves to a real tensor."""
    for tmpl in candidates:
        name = tmpl.format(N=N)
        if name in hdr.tensors:
            return name
    return None


def _discover_layer_indices(hdr: GgufHeader) -> list[int]:
    """Enumerate all transformer block indices present in the GGUF."""
    layer_ids: set[int] = set()
    for name in hdr.tensors:
        m = re.match(r"^blk\.(\d+)\.", name)
        if m:
            layer_ids.add(int(m.group(1)))
    return sorted(layer_ids)


def _copy_for_inplace_mutation(src: Path, dst: Path,
                                 log_fn) -> tuple[str, float]:
    """Materialize a writable copy of `src` at `dst`. Returns (kind, seconds).

    Never hardlink — we will mutate `dst` in place via mmap. A hardlink
    would propagate mutations back into the source BF16, breaking the
    SHA-keyed imatrix/costs caches built against it.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    t0 = time.time()
    try:
        # cephfs supports reflinks via FICLONE; very fast COW copy.
        import subprocess
        rc = subprocess.run(["cp", "--reflink=auto", str(src), str(dst)],
                             check=False).returncode
        if rc != 0:
            raise OSError("reflink cp failed")
        kind = "reflink"
    except Exception:
        shutil.copy2(src, dst)
        kind = "copy"
    dt = time.time() - t0
    log_fn(f"F+. materialized pc-bf16 via {kind} in {dt:.1f}s "
           f"({dst.stat().st_size/1024**3:.2f} GB)")
    return kind, dt


def _build_fold_plan(hdr: GgufHeader, bpw_by_tensor: dict[str, Optional[float]],
                     bpw_floor: float, act_cache_dir: Path,
                     log_fn) -> list[dict]:
    """Returns a list of fold-plan entries:
        {layer: N, kind: 'attn'|'ffn', norm: str, readers: [str], hf_names: [str]}
    Only includes folds that pass all P2 eligibility gates:
      - norm tensor exists in the GGUF
      - every reader tensor exists AND is at/above bpw_floor
      - every reader has an activation cache file
      - all readers share the same in_features (== norm.dim)
    Each excluded fold candidate is logged at INFO with the gating reason.
    """
    plan: list[dict] = []
    for N in _discover_layer_indices(hdr):
        for grp in _FOLD_GROUPS:
            norm = _pick_norm(hdr, grp["norm_candidates"], N)
            if norm is None:
                continue
            readers = [tmpl.format(N=N) for tmpl in grp["readers"]]
            if not all(r in hdr.tensors for r in readers):
                continue  # readers absent — e.g. linear-attn layer has no attn_q
            # bpw floor gate
            below = [r for r in readers
                     if (b := bpw_by_tensor.get(r)) is None or b < bpw_floor]
            if below:
                log_fn(f"F+. skip {grp['kind']}@blk.{N}: "
                       f"sub-floor readers {below}")
                continue
            # activation cache gate
            hf_names = []
            missing_acts = []
            for r in readers:
                hf = _gguf_reader_to_hf_name(r)
                if hf is None or not _actcache_file(act_cache_dir, hf).exists():
                    missing_acts.append(r)
                else:
                    hf_names.append(hf)
            if missing_acts:
                log_fn(f"F+. skip {grp['kind']}@blk.{N}: "
                       f"no acts for {missing_acts}")
                continue
            # in_features agreement (ggml dims[0] = in)
            in_features = hdr.tensors[readers[0]].dims[0]
            mismatch = [r for r in readers[1:]
                        if hdr.tensors[r].dims[0] != in_features]
            if mismatch:
                log_fn(f"F+. skip {grp['kind']}@blk.{N}: "
                       f"in_features mismatch {mismatch}")
                continue
            if hdr.tensors[norm].dims[0] != in_features:
                log_fn(f"F+. skip {grp['kind']}@blk.{N}: "
                       f"norm dim {hdr.tensors[norm].dims} ≠ in={in_features}")
                continue
            plan.append({
                "layer": N, "kind": grp["kind"],
                "norm": norm, "readers": readers, "hf_names": hf_names,
                "in_features": in_features,
            })
    return plan


def _apply_fold(hdr: GgufHeader, mm, plan_entry: dict, act_cache_dir: Path
                 ) -> dict:
    """Apply one AWQ fold in-place via mmap. Returns per-fold stats for
    the manifest: scale min/max/geomean, applied tensors."""
    readers = plan_entry["readers"]
    hf_names = plan_entry["hf_names"]
    norm = plan_entry["norm"]
    # 1) Load activations, compute joint scale.
    acts = [_load_acts(act_cache_dir, hf) for hf in hf_names]
    acts = [a for a in acts if a is not None]
    if not acts:
        raise RuntimeError(
            f"fold {plan_entry['kind']}@blk.{plan_entry['layer']}: "
            f"all acts vanished between plan and apply")
    s = awq_joint_channel_scale(acts).cpu().numpy().astype(np.float32)
    s_max = float(s.max())
    s_min = float(s.min())
    s_geomean = float(np.exp(np.mean(np.log(np.clip(s, 1e-30, None)))))
    # 2) Apply s to every reader's weight (input-channel rescale).
    for r in readers:
        W = read_tensor_fp32(hdr, mm, r)                # [out, in]
        W_scaled = W * s[None, :]
        write_tensor_fp32(hdr, mm, r, W_scaled.astype(np.float32))
    # 3) Fold γ /= s on the predecessor norm.
    gamma = read_tensor_fp32(hdr, mm, norm)             # [in]
    s_safe = np.maximum(s, 1e-12)
    gamma_folded = (gamma.astype(np.float32) / s_safe).astype(np.float32)
    write_tensor_fp32(hdr, mm, norm, gamma_folded)
    return {
        "scale_min": s_min, "scale_max": s_max, "scale_geomean": s_geomean,
        "n_channels": int(s.shape[0]),
    }


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
    immediately — Stage H reads the original BF16 unchanged.

    Otherwise (P2: "on"): copies bf16 → pc-bf16, mmaps the copy PROT_WRITE,
    walks the fold plan, applies AWQ proper-fold per-layer in-place, and
    writes a manifest describing what ran.
    """
    if cfg.precondition_mode == "off":
        _log(layout, "F+. precondition mode = off; passthrough (no-op)")
        return bf16_path, None

    pc_path = layout.preconditioned_bf16_path(model_name, cfg.reference_format)
    manifest_path = layout.precondition_manifest_path()

    # Idempotent skip: both outputs already produced by a prior run.
    if pc_path.exists() and manifest_path.exists():
        _log(layout, "F+. cached (pc-gguf + manifest exist; skip)")
        return pc_path, manifest_path

    _log(layout, f"F+. precondition mode = {cfg.precondition_mode}, "
                 f"bpw_floor = {cfg.precondition_bpw_floor}")

    recipe_data = json.loads(recipe_path.read_text())
    # Bundled allocator wraps the per-tensor map under "recipe"; older
    # callers and the legacy explore-style JSON use "assignments"; fall back
    # to the top-level dict only when neither key is present.
    assignments = (recipe_data.get("recipe")
                   or recipe_data.get("assignments")
                   or recipe_data)

    costs_bpw = _load_costs_bpw(costs_path)

    # bpw_by_tensor + per-tensor entries: same recipe walk as P1.
    entries: list[dict] = []
    bpw_by_tensor: dict[str, Optional[float]] = {}
    fmt_by_tensor: dict[str, str] = {}
    n_skip_lowbit = 0
    n_skip_unknown = 0
    n_above_floor = 0

    for tensor, raw_fmt in assignments.items():
        fmt = _normalize_fmt(raw_fmt)
        fmt_by_tensor[tensor] = fmt
        bpw = _resolve_bpw(tensor, fmt, costs_bpw)
        bpw_by_tensor[tensor] = bpw

        if bpw is None:
            reason = "skip:unknown_bpw"
            n_skip_unknown += 1
        elif bpw < cfg.precondition_bpw_floor:
            reason = "skip:lowbit"
            n_skip_lowbit += 1
        else:
            reason = "pending"
            n_above_floor += 1

        entries.append({
            "tensor": tensor, "format": fmt, "bpw": bpw, "reason": reason,
        })
    entry_by_name = {e["tensor"]: e for e in entries}

    # Materialize pc-bf16 (full copy — must not hardlink, see helper docstring).
    log_fn = lambda m: _log(layout, m)
    link_kind, copy_seconds = _copy_for_inplace_mutation(
        bf16_path, pc_path, log_fn)

    # Parse the COPY's header (offsets are identical to the source, but
    # parsing the copy keeps us honest about which file we're mutating).
    hdr = parse_gguf_header(pc_path)
    act_cache_dir = layout.probe_dir / "act-cache"
    if not act_cache_dir.exists():
        raise SystemExit(
            f"F+ needs activation cache at {act_cache_dir} (stage C output) "
            f"but the directory is missing")

    # Build the per-layer fold plan from the recipe + GGUF tensor list.
    plan = _build_fold_plan(hdr, bpw_by_tensor, cfg.precondition_bpw_floor,
                             act_cache_dir, log_fn)
    n_planned_folds = len(plan)
    _log(layout, f"F+. fold plan: {n_planned_folds} folds across "
                 f"{len({p['layer'] for p in plan})} layers")

    # Apply folds in place via a single mmap session.
    fold_records: list[dict] = []
    applied_norms: set[str] = set()
    applied_readers: set[str] = set()
    if plan:
        t_apply = time.time()
        with open(pc_path, "r+b") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_WRITE)
            try:
                for p in plan:
                    stats = _apply_fold(hdr, mm, p, act_cache_dir)
                    rec = {
                        "layer": p["layer"], "kind": p["kind"],
                        "norm": p["norm"], "readers": p["readers"],
                        **stats,
                    }
                    fold_records.append(rec)
                    applied_norms.add(p["norm"])
                    applied_readers.update(p["readers"])
                mm.flush()
            finally:
                mm.close()
        _log(layout,
             f"F+. applied {len(fold_records)} folds in "
             f"{time.time()-t_apply:.1f}s "
             f"(touched {len(applied_readers)} reader weights + "
             f"{len(applied_norms)} norms)")

    # Update per-tensor entry reasons now that we know what fold-touched.
    n_applied_awq = 0
    n_skip_mixed = 0
    for e in entries:
        t = e["tensor"]
        if t in applied_readers:
            e["reason"] = "applied:awq"
            n_applied_awq += 1
        elif e["reason"] == "pending":
            # Above floor but didn't get folded (either no fold group covers
            # this tensor, a fold-mate was sub-floor, or activation cache miss).
            e["reason"] = "skip:no_fold"
            n_skip_mixed += 1
    # Norms that got γ-divided are also "applied" tensors — record so the
    # manifest's set-of-modified-tensors is complete.
    for norm_name in applied_norms:
        e = entry_by_name.get(norm_name)
        if e is None:
            entries.append({
                "tensor": norm_name, "format": "F32",
                "bpw": _BPW_FALLBACK["F32"],
                "reason": "applied:awq_norm_fold",
            })
        else:
            e["reason"] = "applied:awq_norm_fold"

    manifest = {
        "schema": 2,
        "phase": "P2",
        "precondition_mode": cfg.precondition_mode,
        "bpw_floor": cfg.precondition_bpw_floor,
        "source_bf16": str(bf16_path),
        "preconditioned_bf16": str(pc_path),
        "link_kind": link_kind,
        "copy_seconds": round(copy_seconds, 2),
        "recipe": str(recipe_path),
        "costs": str(costs_path),
        "summary": {
            "n_tensors": len(entries),
            "n_above_floor": n_above_floor,
            "n_applied_awq": n_applied_awq,
            "n_applied_awq_norms": len(applied_norms),
            "n_skip_lowbit": n_skip_lowbit,
            "n_skip_unknown_bpw": n_skip_unknown,
            "n_skip_no_fold": n_skip_mixed,
            "n_folds": len(fold_records),
        },
        "folds": fold_records,
        "entries": entries,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    _log(layout, f"F+. {len(entries)} tensors: "
                 f"{n_applied_awq} applied:awq, "
                 f"{len(applied_norms)} norms folded, "
                 f"{n_skip_mixed} above-floor but no-fold, "
                 f"{n_skip_lowbit} skip:lowbit, "
                 f"{n_skip_unknown} skip:unknown_bpw")
    _log(layout, f"F+. pc-bf16: {pc_path} ({link_kind})")
    _log(layout, f"F+. manifest: {manifest_path}")

    return pc_path, manifest_path
