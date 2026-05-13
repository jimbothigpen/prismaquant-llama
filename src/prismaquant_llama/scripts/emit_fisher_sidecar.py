#!/usr/bin/env python3
"""Emit per-Linear Fisher sidecar binaries for llama-quantize-cost.

Reads the probe's activation cache (one .pt per HF Linear with `inputs` and
optionally `row_indices`) and the probe's h-detail blobs (per-Linear `.pt`
carrying `g2_per_token`), then writes one C++-friendly binary sidecar per
GGUF tensor target. llama-quantize-cost's `--fisher-sidecar DIR` consumes
these.

Sidecar layout (matches llama-quantize-cost's reader):

    offset  size                       field
    0       4 bytes                    magic = "PQFS"
    4       4 bytes (uint32 LE)        version = 1
    8       4 bytes (uint32 LE)        N (sampled activation rows)
    12      4 bytes (uint32 LE)        in_features
    16      N * in_features * 4 bytes  X (float32, row-major [N, in_features])
    ...     N * 4 bytes                fisher_weights (float32 per row)

Filename: GGUF tensor name with `.` -> `__`, plus `.bin`.

A single HF Linear may map to multiple GGUF tensors (e.g. an HF packed
gate_up_proj splits into GGUF ffn_gate_exps + ffn_up_exps). We emit the
same X / fisher_weights to each target — the C++ tool reconstructs the
per-split output error from the GGUF weight subset and the shared activation.

Rows that survived the probe's per-Linear cap (default 256) keep their
original calibration token offsets via the `row_indices` field added by
the post-2026-05-12 upstream merge; we use those to look up the matching
g2_per_token entries. Pre-merge probes without row_indices fall back to
uniform fisher weights (1.0 per row) — the resulting fisher_output_mse
column is then the unweighted per-row output MSE rather than truly Fisher-
weighted, but it's still a usable signal across formats.
"""

from __future__ import annotations

import argparse
import pickle
import re
import struct
import sys
from pathlib import Path
from typing import Optional

import torch


# Import the bridge mapping directly (sibling script, same package).
sys.path.insert(0, str(Path(__file__).parent))
from bridge_probe_to_gguf import map_hf_to_gguf  # noqa: E402


SIDECAR_MAGIC = b"PQFS"
SIDECAR_VERSION = 1


def _hf_from_actcache_filename(fname: str) -> str:
    """`model__layers__3__self_attn__q_proj.pt` -> `model.layers.3.self_attn.q_proj`."""
    stem = fname[:-3] if fname.endswith(".pt") else fname
    return stem.replace("__", ".")


def _safe_gguf_name(gguf_name: str) -> str:
    """`blk.0.attn_q.weight` -> `blk__0__attn_q__weight` (sidecar basename)."""
    return gguf_name.replace(".", "__")


def _hdetail_path(h_detail_dir: Path, hf_name: str) -> Path:
    """Mirrors `prismaquant.measure_quant_cost.HDetailIndex._FNAME_SUB`:
    `[^A-Za-z0-9_-]` -> `__`, with `.pt` suffix."""
    fname = re.sub(r"[^A-Za-z0-9_-]", "__", hf_name) + ".pt"
    return h_detail_dir / fname


def _load_g2_per_token(h_detail_dir: Optional[Path],
                       hf_name: str) -> Optional[torch.Tensor]:
    """Return the per-token Fisher g² vector for this Linear, or None
    if h-detail wasn't produced for it (caller falls back to uniform)."""
    if h_detail_dir is None:
        return None
    p = _hdetail_path(h_detail_dir, hf_name)
    if not p.exists():
        return None
    try:
        blob = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"[fisher-sidecar] WARN: failed to load h-detail {p}: {e}",
              file=sys.stderr)
        return None
    if isinstance(blob, dict) and "g2_per_token" in blob:
        g2 = blob["g2_per_token"]
        if isinstance(g2, torch.Tensor):
            return g2
    return None


def _gather_fisher_weights(g2_per_token: Optional[torch.Tensor],
                           row_indices: Optional[torch.Tensor],
                           n_rows: int) -> torch.Tensor:
    """Resolve a length-N fp32 fisher_weights vector for sampled rows.

    Three cases:
      - h-detail g2 present + row_indices present -> gather g2[row_indices]
        and clamp negative / NaN entries to a small positive so the C++
        weighting stays sane.
      - g2 absent or row_indices absent -> uniform 1.0 / n_rows weights.
        This makes fisher_output_mse degenerate into plain per-row output
        MSE, which is still a usable signal for cross-format comparison.
    """
    if g2_per_token is not None and row_indices is not None:
        g2 = g2_per_token.detach().to(torch.float32).flatten()
        idx = row_indices.detach().to(torch.long).flatten()
        if idx.numel() == n_rows and int(idx.max()) < g2.numel():
            w = g2.index_select(0, idx).clone()
            w = torch.where(torch.isfinite(w) & (w > 0), w,
                            torch.full_like(w, 1e-12))
            return w
    return torch.full((n_rows,), 1.0, dtype=torch.float32)


def _write_sidecar(out_path: Path, X: torch.Tensor,
                   fisher_weights: torch.Tensor) -> None:
    """Write a single sidecar .bin in the format llama-quantize-cost expects.
    X is [N, in_features] fp32 contiguous; fisher_weights is [N] fp32."""
    X = X.detach().to(torch.float32).contiguous()
    fw = fisher_weights.detach().to(torch.float32).contiguous()
    assert X.ndim == 2 and fw.ndim == 1 and X.shape[0] == fw.shape[0], (
        f"sidecar shape mismatch: X={tuple(X.shape)} fw={tuple(fw.shape)}")
    n_rows, in_features = int(X.shape[0]), int(X.shape[1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("wb") as f:
        f.write(SIDECAR_MAGIC)
        f.write(struct.pack("<III", SIDECAR_VERSION, n_rows, in_features))
        f.write(X.numpy().tobytes(order="C"))
        f.write(fw.numpy().tobytes(order="C"))
    tmp.rename(out_path)


def emit_sidecars(*,
                  probe_pkl: Path,
                  out_dir: Path,
                  n_hidden_layers: Optional[int],
                  n_nextn_layers: int = 0,
                  act_cache_dir: Optional[Path] = None,
                  h_detail_dir: Optional[Path] = None,
                  gguf_path: Optional[Path] = None,
                  ) -> tuple[int, int, int, int]:
    """Walk act-cache, emit one sidecar per (HF -> GGUF) mapping target.
    Returns (n_written, n_unmapped, n_skipped, n_dim_mismatch).

    When `gguf_path` is given, validate each (HF -> GGUF) sidecar's
    `inputs.shape[1]` against the GGUF tensor's `ne[0]` (the inner / in-dim
    of the weight matrix). Sidecars whose dim doesn't match are silently
    SKIPPED rather than written — this dodges a known upstream-prismaquant
    probe issue where the act-cache `inputs` blob doesn't match the HF
    Linear's true in_features for many tensors (only ~13% of Qwen3.5-4B
    sidecars match). llama-quantize-cost would WARN+ignore mismatched
    sidecars anyway, but suppressing them at emit time keeps the output
    cleaner and makes the dim-mismatch count visible in pipeline logs."""
    with probe_pkl.open("rb") as f:
        probe = pickle.load(f)
    meta = probe.get("meta", {}) or {}
    if act_cache_dir is None:
        act_cache_dir_str = meta.get("activation_cache_dir")
        if not act_cache_dir_str:
            print(f"[fisher-sidecar] FAIL: probe.pkl has no activation_cache_dir",
                  file=sys.stderr)
            return (0, 0, 0)
        act_cache_dir = Path(act_cache_dir_str)
    if not act_cache_dir.exists():
        print(f"[fisher-sidecar] FAIL: act-cache dir not found: {act_cache_dir}",
              file=sys.stderr)
        return (0, 0, 0)

    # Optional GGUF dim-validation. We import allocator.read_gguf_tensor_meta
    # rather than duplicating the header parser; both files live in scripts/
    # and scripts/'s dir is already on sys.path via the bridge_probe_to_gguf
    # import above.
    gguf_meta: dict[str, tuple] = {}
    if gguf_path is not None:
        from allocator import read_gguf_tensor_meta  # noqa: E402
        gguf_meta = read_gguf_tensor_meta(str(gguf_path))
        print(f"[fisher-sidecar] gguf dim-validation: {len(gguf_meta)} tensors "
              f"loaded from {gguf_path}")

    n_written = 0
    n_unmapped = 0
    n_skipped = 0
    n_dim_mismatch = 0
    for pt_path in sorted(act_cache_dir.glob("*.pt")):
        hf_name = _hf_from_actcache_filename(pt_path.name)
        try:
            blob = torch.load(pt_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[fisher-sidecar] WARN: failed to load {pt_path}: {e}",
                  file=sys.stderr)
            n_skipped += 1
            continue
        if not isinstance(blob, dict) or "inputs" not in blob:
            n_skipped += 1
            continue
        X = blob["inputs"]
        if not isinstance(X, torch.Tensor) or X.ndim != 2:
            n_skipped += 1
            continue
        row_indices = blob.get("row_indices")
        g2 = _load_g2_per_token(h_detail_dir, hf_name)
        fw = _gather_fisher_weights(g2, row_indices, int(X.shape[0]))
        targets = map_hf_to_gguf(hf_name,
                                  n_hidden_layers=n_hidden_layers,
                                  n_nextn_layers=n_nextn_layers)
        if not targets:
            n_unmapped += 1
            continue
        sc_in_features = int(X.shape[1])
        for gguf_name, _frac in targets:
            if gguf_meta:
                meta = gguf_meta.get(gguf_name)
                if meta is None:
                    n_dim_mismatch += 1
                    continue
                if int(meta[0]) != sc_in_features:
                    n_dim_mismatch += 1
                    continue
            out_path = out_dir / f"{_safe_gguf_name(gguf_name)}.bin"
            _write_sidecar(out_path, X, fw)
            n_written += 1
    return n_written, n_unmapped, n_skipped, n_dim_mismatch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True,
                    help="path to probe.pkl from Stage C")
    ap.add_argument("--output-dir", required=True,
                    help="directory to write per-tensor sidecar .bin files")
    ap.add_argument("--n-hidden-layers", type=int, default=None,
                    help="num regular transformer layers (needed for MTP map)")
    ap.add_argument("--n-nextn-layers", type=int, default=0,
                    help="num MTP blocks the model declares (0 if none)")
    ap.add_argument("--act-cache-dir", default=None,
                    help="override; otherwise resolved from probe.pkl meta")
    ap.add_argument("--h-detail-dir", default=None,
                    help="path to h-detail dir produced by --h-detail-dir on "
                         "the probe; if absent, fisher_weights default to 1.0")
    ap.add_argument("--gguf", default=None,
                    help="optional path to the BF16 GGUF; enables per-target "
                         "dim validation (sidecar in_features must equal "
                         "GGUF ne[0]). Mismatched targets are silently skipped.")
    args = ap.parse_args()
    n_w, n_u, n_s, n_dm = emit_sidecars(
        probe_pkl=Path(args.probe),
        out_dir=Path(args.output_dir),
        n_hidden_layers=args.n_hidden_layers,
        n_nextn_layers=args.n_nextn_layers,
        act_cache_dir=Path(args.act_cache_dir) if args.act_cache_dir else None,
        h_detail_dir=Path(args.h_detail_dir) if args.h_detail_dir else None,
        gguf_path=Path(args.gguf) if args.gguf else None,
    )
    print(f"[fisher-sidecar] wrote {n_w} sidecars "
          f"({n_u} unmapped HF entries, {n_s} skipped, "
          f"{n_dm} skipped-on-gguf-dim-mismatch)")


if __name__ == "__main__":
    main()
