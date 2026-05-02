#!/usr/bin/env python3
"""Multi-choice knapsack allocator: pick one GGUF format per tensor under
a total-size budget, minimizing prismaquant's Δloss surrogate.

Inputs:
    --bridge   probe-bridged-to-gguf JSON (from bridge_probe_to_gguf.py)
    --costs    quantize-cost CSV (from llama-quantize-cost)
    --budget-gb FLOAT
    --pinned   JSON {gguf_name: format} for hard pins (lm_head, embed, etc.)

Output:
    --recipe-out  JSON {gguf_name: chosen_format}
    --pareto-csv  optional sweep across multiple budgets

Algorithm: Lagrangian relaxation. For each λ ≥ 0:
    pick fmt[t] = argmin_f ( 0.5 * fisher[t] * MSE[t,f] + λ * size[t,f] )
This is independent per-tensor → O(N · F) per λ. Sweep λ to find one whose
total size lands at-or-below budget. Bisection finds the budget-respecting
λ in ~50 iterations. Result is a near-optimal knapsack solution (LP-relaxation
optimal, knapsack-optimal up to one tensor's worth of slack).
"""

import argparse
import csv
import json
import math
import struct
import sys
from collections import defaultdict
from pathlib import Path


def load_costs(path: str) -> dict:
    """Returns {tensor_name: {format: (mse, size_bytes, n_elements, bpw)}}."""
    by_tensor = defaultdict(dict)
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            mse_str = r["mse"]
            if mse_str == "nan" or mse_str == "" or mse_str.lower() == "nan":
                continue  # type required imatrix not provided
            mse = float(mse_str)
            sz  = int(r["size_bytes"])
            n   = int(r["n_elements"])
            bpw = float(r["bpw"])
            by_tensor[r["tensor_name"]][r["fmt"]] = (mse, sz, n, bpw)
    return dict(by_tensor)


def read_gguf_tensor_names(path: str) -> set[str]:
    """Walks the GGUF header to collect tensor names. No data deps."""
    GGUF_TYPES = {
        4: ("u32", 4), 5: ("i32", 4), 6: ("f32", 4),
        10: ("u64", 8), 11: ("i64", 8), 12: ("f64", 8),
        7: ("bool", 1), 8: ("string", None), 9: ("array", None),
        0: ("u8", 1), 1: ("i8", 1), 2: ("u16", 2), 3: ("i16", 2),
    }
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"Not a GGUF: {path}")
        version, tensor_count, kv_count = struct.unpack("<IQQ", f.read(20))
        def read_string():
            n, = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8")
        def skip_value(t):
            name, size = GGUF_TYPES[t]
            if name == "string":
                read_string()
            elif name == "array":
                etype, n = struct.unpack("<IQ", f.read(12))
                for _ in range(n):
                    skip_value(etype)
            else:
                f.read(size)
        for _ in range(kv_count):
            read_string()
            t, = struct.unpack("<I", f.read(4))
            skip_value(t)
        names = set()
        for _ in range(tensor_count):
            tn = read_string()
            n_dims, = struct.unpack("<I", f.read(4))
            f.read(8 * n_dims)
            f.read(4 + 8)
            names.add(tn)
        return names


def read_gguf_tensor_meta(path: str) -> dict[str, tuple]:
    """Like read_gguf_tensor_names but also captures tensor SHAPES.

    Returns {tensor_name: (ne[0], ne[1], ...)} where dims follow GGUF
    convention: ne[0] is the inner / per-row width, ne[1] the outer / n_rows.
    For a Linear weight saved as [in, out], ne[0]=in, ne[1]=out.

    Shape data is what lets us distinguish iSWA layers (gemma-4) where
    `attn_q.weight` has different ne[1] for full_attention vs sliding_attention.
    """
    GGUF_TYPES = {
        4: ("u32", 4), 5: ("i32", 4), 6: ("f32", 4),
        10: ("u64", 8), 11: ("i64", 8), 12: ("f64", 8),
        7: ("bool", 1), 8: ("string", None), 9: ("array", None),
        0: ("u8", 1), 1: ("i8", 1), 2: ("u16", 2), 3: ("i16", 2),
    }
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"Not a GGUF: {path}")
        version, tensor_count, kv_count = struct.unpack("<IQQ", f.read(20))
        def read_string():
            n, = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8")
        def skip_value(t):
            name, size = GGUF_TYPES[t]
            if name == "string":
                read_string()
            elif name == "array":
                etype, n = struct.unpack("<IQ", f.read(12))
                for _ in range(n):
                    skip_value(etype)
            else:
                f.read(size)
        for _ in range(kv_count):
            read_string()
            t, = struct.unpack("<I", f.read(4))
            skip_value(t)
        meta = {}
        for _ in range(tensor_count):
            tn = read_string()
            n_dims, = struct.unpack("<I", f.read(4))
            dims = struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims))
            f.read(4 + 8)  # type code + data offset
            meta[tn] = dims
        return meta


def detect_layer_types(tensor_names_or_meta) -> dict[int, str]:
    """For each layer index, classify by attention "shape signature" so that
    iSWA architectures (gemma-3, gemma-4: full_attention vs sliding_attention
    with different head_dim) get distinct types and don't share exemplars
    across mismatched sizes.

    Accepts either:
      - set[str]: tensor name set (legacy — produces coarse 'linear'/'softmax' types)
      - dict[str, tuple]: tensor_name → shape (preferred — produces shape-aware
        types like 'softmax_q4096_k1024' that distinguish iSWA layer subtypes)
    """
    has_shapes = isinstance(tensor_names_or_meta, dict)
    sigs_by_layer: dict[int, dict[str, tuple]] = {}
    for tn in tensor_names_or_meta:
        if not tn.startswith("blk."):
            continue
        parts = tn.split(".", 2)
        if len(parts) < 3:
            continue
        try:
            L = int(parts[1])
        except ValueError:
            continue
        suffix = parts[2]
        shape = tensor_names_or_meta[tn] if has_shapes else None
        # Match ONLY .weight tensors — skip biases, norms (1D) which would
        # corrupt the shape signature with their wrong dimensionality.
        if suffix == "attn_qkv.weight":
            sigs_by_layer.setdefault(L, {})["attn_qkv"] = shape
        elif suffix == "attn_q.weight":
            sigs_by_layer.setdefault(L, {})["attn_q"] = shape
        elif suffix == "attn_k.weight":
            sigs_by_layer.setdefault(L, {})["attn_k"] = shape

    def _shape_tag(shape):
        """Return shape[1] (output dim) if available, else None.
        Defensive: handles None, 1D shapes, etc."""
        if shape is None or len(shape) < 2:
            return None
        return shape[1]

    out = {}
    for L, sigs in sigs_by_layer.items():
        if "attn_qkv" in sigs:
            # Linear-attention (Gated-Delta-Net / Mamba-2 / etc.); shape-tag if available
            tag = _shape_tag(sigs["attn_qkv"])
            out[L] = f"linear_{tag}" if tag is not None else "linear"
        elif "attn_q" in sigs:
            q_tag = _shape_tag(sigs["attn_q"])
            k_tag = _shape_tag(sigs.get("attn_k"))
            if q_tag is not None and k_tag is not None:
                out[L] = f"softmax_q{q_tag}_k{k_tag}"
            elif q_tag is not None:
                out[L] = f"softmax_q{q_tag}"
            else:
                out[L] = "softmax"
    return out


def auto_pick_exemplar_layers(layer_type: dict[int, str],
                              max_exemplars: int = 4) -> list[int]:
    """Pick the smallest layer index for each distinct layer_type so the
    exemplar set spans every shape signature in the model.

    For pure-Llama (1 type), returns [0]. For iSWA (2 types, e.g. sliding +
    full), returns [0, first_full_layer]. For Mamba+attention hybrids
    (multiple types), returns up to max_exemplars representative layers.
    """
    seen = {}
    for L in sorted(layer_type.keys()):
        t = layer_type[L]
        if t not in seen:
            seen[t] = L
            if len(seen) >= max_exemplars:
                break
    return sorted(seen.values())


def propagate_costs(costs: dict, gguf_names: set[str],
                    exemplar_layers: list[int],
                    layer_type: dict[int, str] | None = None) -> tuple[dict, int]:
    """Replicate cost data from exemplar tensors to peer-type tensors in
    other layers. Returns (propagated_costs, n_propagated).

    `layer_type` (optional, preferred): pre-computed layer→type mapping from
    detect_layer_types(tensor_meta) so iSWA layers stay distinct. If omitted,
    falls back to name-only classification (no shape distinguishing)."""
    if layer_type is None:
        layer_type = detect_layer_types(gguf_names)
    exemplar_for_type: dict[str, int] = {}
    for L in exemplar_layers:
        t = layer_type.get(L)
        if t and t not in exemplar_for_type:
            exemplar_for_type[t] = L
    out = {k: dict(v) for k, v in costs.items()}
    n_added = 0
    for tn in gguf_names:
        if tn in out:
            continue
        if not tn.startswith("blk."):
            continue  # non-blk tensors can't propagate
        parts = tn.split(".", 2)
        try:
            L = int(parts[1])
        except ValueError:
            continue
        suffix = parts[2]
        ltype = layer_type.get(L)
        if not ltype:
            continue
        ex_L = exemplar_for_type.get(ltype)
        if ex_L is None:
            continue
        exemplar_name = f"blk.{ex_L}.{suffix}"
        if exemplar_name in costs:
            out[tn] = dict(costs[exemplar_name])
            n_added += 1
    return out, n_added


def parse_priority(spec: str) -> tuple[float, float, float]:
    """Parse a 3-digit priority spec like '531' (PPL=5,TG=3,PP=1).
    Returns (w_ppl, w_tg, w_pp) normalized to sum=1.0 (so weights are
    dimensionless ratios). Sum of digits is conventionally 9 but any sum > 0
    is accepted. The string '900' means pure-PPL (the original allocator)."""
    if len(spec) != 3 or not spec.isdigit():
        raise ValueError(f"priority must be 3 digits like '531', got {spec!r}")
    p, t, q = int(spec[0]), int(spec[1]), int(spec[2])
    s = p + t + q
    if s == 0:
        raise ValueError("priority spec sums to 0 — at least one weight must be nonzero")
    return (p/s, t/s, q/s)


def compute_norms(costs: dict, fisher: dict, tps: dict) -> tuple[float, float, float]:
    """Population means of each cost component over all (t, f) pairs. Used to
    bring the three terms into comparable units before applying weights."""
    n = 0
    sum_ppl = 0.0
    sum_tg  = 0.0
    sum_pp  = 0.0
    for t, fmts in costs.items():
        h = fisher.get(t, 0.0)
        for f, (mse, sz, _, _) in fmts.items():
            sum_ppl += 0.5 * h * mse
            tps_f = tps.get(f, {})
            tg = tps_f.get("tg", 1.0)
            pp = tps_f.get("pp", 1.0)
            sum_tg += sz / tg
            sum_pp += sz / pp
            n += 1
    if n == 0:
        return 1.0, 1.0, 1.0
    # Use mean; guard against zero
    return (max(sum_ppl/n, 1e-30),
            max(sum_tg/n,  1e-30),
            max(sum_pp/n,  1e-30))


def solve_for_lambda(fisher: dict[str, float],
                     costs: dict,
                     pinned: dict[str, str],
                     lam: float,
                     weights: tuple[float, float, float] = (1.0, 0.0, 0.0),
                     tps: dict | None = None,
                     norms: tuple[float, float, float] = (1.0, 1.0, 1.0),
                     ) -> tuple[dict[str, str], int, float]:
    """Per-tensor argmin( w_PPL · ĉ_ppl + w_TG · ĉ_tg + w_PP · ĉ_pp + λ · size ),
    where ĉ_x are normalized cost components (per-tensor PPL surrogate, TG
    latency proxy, PP latency proxy). Returns (recipe, total_size, total_loss).
    Default weights=(1,0,0) reduces to the original PPL-only allocator.

    `tps[fmt]` should provide {'pp': float, 'tg': float}. Missing formats
    contribute 0 to the speed terms (treated as ideal speed)."""
    w_ppl, w_tg, w_pp = weights
    n_ppl, n_tg, n_pp = norms
    tps = tps or {}
    recipe = {}
    total_size = 0
    total_loss = 0.0
    for t, fmts in costs.items():
        if t in pinned:
            f = pinned[t]
            if f not in fmts:
                # Fall back to nearest available higher-bit format if pinned is missing
                f = max(fmts.keys(), key=lambda k: fmts[k][3])
            mse, sz, _, _ = fmts[f]
            recipe[t] = f
            total_size += sz
            total_loss += 0.5 * fisher.get(t, 0.0) * mse
            continue
        h = fisher.get(t, 0.0)
        best_f, best_score = None, math.inf
        for f, (mse, sz, _, _) in fmts.items():
            ppl_term = 0.5 * h * mse / n_ppl
            tps_f = tps.get(f, {})
            tg_tps = tps_f.get("tg", 1.0)
            pp_tps = tps_f.get("pp", 1.0)
            tg_term = (sz / tg_tps) / n_tg
            pp_term = (sz / pp_tps) / n_pp
            multi_obj = w_ppl * ppl_term + w_tg * tg_term + w_pp * pp_term
            score = multi_obj + lam * sz
            if score < best_score:
                best_score = score
                best_f = f
        if best_f is None:
            continue
        recipe[t] = best_f
        mse, sz, _, _ = fmts[best_f]
        total_size += sz
        total_loss += 0.5 * h * mse
    return recipe, total_size, total_loss


def bisect_lambda(fisher: dict, costs: dict, pinned: dict, budget_bytes: int,
                  weights: tuple[float, float, float] = (1.0, 0.0, 0.0),
                  tps: dict | None = None,
                  norms: tuple[float, float, float] = (1.0, 1.0, 1.0),
                  tol_bytes: int = 10_000_000,
                  band_bytes: int = 0,
                  ) -> tuple[float, dict, int, float]:
    """Find λ such that the recipe size is within tol_bytes of the budget.

    Sign convention:
      λ > 0  penalizes size (chooses smaller formats)
      λ = 0  natural optimum of the multi-objective cost
      λ < 0  rewards size (chooses LARGER formats — used when the natural
             optimum is already below budget; pushes recipe up to bind budget)

    For PPL-only allocator (default weights), λ ≥ 0 always works because PPL
    cost prefers larger formats. For TG/PP-weighted priorities, the multi-
    objective cost can naturally land below budget, requiring λ < 0 to spend
    the remaining budget on quality."""
    kw = dict(weights=weights, tps=tps, norms=norms)

    # First, check the natural optimum (λ=0) — does it land at, above, or below budget?
    recipe0, size0, loss0 = solve_for_lambda(fisher, costs, pinned, 0.0, **kw)
    if abs(size0 - budget_bytes) < tol_bytes:
        return 0.0, recipe0, size0, loss0

    if size0 > budget_bytes:
        # Natural opt overruns budget — search positive λ (size penalty)
        lo, hi = 0.0, 1e6
        for _ in range(60):
            recipe_hi, size_hi, _ = solve_for_lambda(fisher, costs, pinned, hi, **kw)
            if size_hi <= budget_bytes:
                break
            hi *= 2.0
        else:
            return hi, recipe_hi, size_hi, _
    else:
        # Natural opt under-shoots budget — search negative λ (size reward).
        # This forces the recipe to fill up to the budget, picking larger
        # formats wherever that doesn't sacrifice too much weighted cost.
        hi, lo = 0.0, -1.0
        for _ in range(60):
            recipe_lo, size_lo, _ = solve_for_lambda(fisher, costs, pinned, lo, **kw)
            if size_lo >= budget_bytes:
                break
            lo *= 2.0
        else:
            # Cannot reach budget even with extreme negative λ; return best
            return lo, recipe_lo, size_lo, _

    # Bisect [lo, hi] — track best-so-far closest to budget (in either direction).
    # Slight overshoot is preferred over a large undershoot because the format
    # space is discrete and there are gaps; landing closest is more useful than
    # strictly under-budget for benchmarking 3-sizes-per-priority.
    best: tuple[float, dict, int, float] | None = None
    def consider(lam_v, recipe_v, size_v, loss_v):
        nonlocal best
        d = abs(size_v - budget_bytes)
        if best is None or d < abs(best[2] - budget_bytes):
            best = (lam_v, recipe_v, size_v, loss_v)
    # Seed with the natural opt and current endpoints
    consider(0.0, recipe0, size0, loss0)
    if size0 > budget_bytes:
        consider(hi, recipe_hi, size_hi, _ if isinstance(_, (int, float)) else 0)
    else:
        consider(lo, recipe_lo, size_lo, _ if isinstance(_, (int, float)) else 0)

    # If `band_bytes` is set and we find any recipe inside the
    # [budget - band, budget + band] window, accept it immediately.
    in_band = lambda s: band_bytes > 0 and abs(s - budget_bytes) <= band_bytes
    if in_band(size0):
        return 0.0, recipe0, size0, loss0
    for _ in range(80):
        mid = (lo + hi) / 2 if abs(hi) < 1e10 else lo + 1.0
        recipe_m, size_m, loss_m = solve_for_lambda(fisher, costs, pinned, mid, **kw)
        consider(mid, recipe_m, size_m, loss_m)
        if in_band(size_m):
            return mid, recipe_m, size_m, loss_m
        if abs(size_m - budget_bytes) < tol_bytes:
            return mid, recipe_m, size_m, loss_m
        if size_m > budget_bytes:
            lo = mid
        else:
            hi = mid
        if abs(hi - lo) < 1e-12:
            break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge", required=True)
    ap.add_argument("--costs", required=True)
    ap.add_argument("--budget-gb", type=float, required=True)
    ap.add_argument("--pinned", default=None,
                    help="JSON {tensor_name: format} for hard-pinned tensors")
    ap.add_argument("--recipe-out", required=True)
    ap.add_argument("--pareto-csv", default=None)
    ap.add_argument("--pareto-budgets-gb", default=None,
                    help="Comma-separated list of budgets (GB) to sweep for Pareto curve")
    ap.add_argument("--allow-types", default=None,
                    help="Comma-separated whitelist of formats; empty=all in costs")
    ap.add_argument("--gguf", default=None,
                    help="Path to BF16 GGUF; required if --propagate-from-exemplars")
    ap.add_argument("--propagate-from-exemplars", action="store_true",
                    help="Propagate costs from measured exemplar tensors to peer-type peers")
    ap.add_argument("--exemplar-layers", default="0,3",
                    help="Comma-separated layer indices to treat as exemplars (default 0,3 — covers linear-attn + softmax-attn)")
    ap.add_argument("--priority", default="900",
                    help="3-digit weight spec PPL-TG-PP (sum=9 convention; e.g. "
                         "'900'=pure-PPL [default; same as old allocator], "
                         "'333'=balanced, '522'=PPL-primary, '531'=PPL>TG>PP)")
    ap.add_argument("--tps", default=None,
                    help="JSON file with per-format {pp, tg} TPS values for the "
                         "speed terms. Required when --priority weights TG or PP.")
    ap.add_argument("--budget-band-gb", type=float, default=0.25,
                    help="Acceptance band around budget. The bisection accepts "
                         "any recipe whose size falls in [budget-band, "
                         "budget+band] (default 0.25 GB). Useful because the "
                         "discrete format space has gaps that can otherwise "
                         "produce far-off-budget recipes. Set to 0 to require "
                         "the old strict closest-to-budget behavior.")
    args = ap.parse_args()

    with open(args.bridge) as f:
        bridge = json.load(f)
    fisher = bridge["h_trace"]
    print(f"[allocator] bridge: {len(fisher)} tensors with H_trace", flush=True)

    costs = load_costs(args.costs)
    print(f"[allocator] costs: {len(costs)} tensors", flush=True)

    if args.propagate_from_exemplars:
        if not args.gguf:
            ap.error("--propagate-from-exemplars requires --gguf")
        # Read tensor SHAPES (not just names) so detect_layer_types can
        # distinguish iSWA layers (gemma-3, gemma-4) by attn_q output dim.
        # Shape-aware tags ('softmax_q<n>_k<n>') prevent propagating sliding-
        # attention sizes to full-attention layers and vice versa.
        gguf_meta = read_gguf_tensor_meta(args.gguf)
        gguf_names = set(gguf_meta.keys())
        layer_type = detect_layer_types(gguf_meta)
        # Auto-pick exemplars to cover every layer type unless --exemplar-layers
        # was passed explicitly with non-default value.
        if args.exemplar_layers == "0,3":
            # Default value — replace with auto-detected coverage
            exemplars = auto_pick_exemplar_layers(layer_type)
            print(f"[allocator] auto-detected layer types: "
                  f"{sorted(set(layer_type.values()))}", flush=True)
            print(f"[allocator] auto-picked exemplar layers: {exemplars} "
                  f"(one per shape signature)", flush=True)
        else:
            exemplars = [int(x) for x in args.exemplar_layers.split(",")]
        costs, n_added = propagate_costs(costs, gguf_names, exemplars,
                                          layer_type=layer_type)
        print(f"[allocator] propagated costs to {n_added} additional tensors "
              f"(exemplar layers: {exemplars})", flush=True)
        print(f"[allocator] costs after propagation: {len(costs)} tensors", flush=True)

    # Filter formats if requested
    if args.allow_types:
        allow = set(t.strip() for t in args.allow_types.split(",") if t.strip())
        for t, fmts in list(costs.items()):
            costs[t] = {f: v for f, v in fmts.items() if f in allow}
            if not costs[t]:
                del costs[t]
        print(f"[allocator] filtered to formats: {sorted(allow)}", flush=True)
        print(f"[allocator] tensors after filter: {len(costs)}", flush=True)

    # Pinned
    pinned = {}
    if args.pinned:
        with open(args.pinned) as f:
            pinned = json.load(f)
        print(f"[allocator] pinned: {len(pinned)} tensors", flush=True)

    # Coverage check
    common = set(fisher) & set(costs)
    fisher_only = set(fisher) - set(costs)
    costs_only  = set(costs)  - set(fisher)
    print(f"[allocator] tensor sets: fisher={len(fisher)}  costs={len(costs)}  "
          f"common={len(common)}  fisher_only={len(fisher_only)}  costs_only={len(costs_only)}",
          flush=True)
    if fisher_only:
        print(f"[allocator]  fisher-only sample: {list(fisher_only)[:5]}", flush=True)
    if costs_only:
        print(f"[allocator]  costs-only sample:  {list(costs_only)[:5]}  "
              f"(no Fisher → treated as h_trace=0; format chosen by size only)",
              flush=True)

    # Multi-objective: parse priority, load per-format TPS, compute norms
    weights = parse_priority(args.priority)
    print(f"[allocator] priority={args.priority} → weights "
          f"(w_PPL={weights[0]:.3f}, w_TG={weights[1]:.3f}, w_PP={weights[2]:.3f})",
          flush=True)
    tps = {}
    if args.tps:
        with open(args.tps) as f:
            tps = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
        print(f"[allocator] loaded TPS data for {len(tps)} formats from {args.tps}",
              flush=True)
    elif weights[1] > 0 or weights[2] > 0:
        print(f"[allocator] WARN: priority requests TG/PP weight but no --tps file; "
              f"speed terms will use tps=1.0 (no-op)", flush=True)
    norms = compute_norms(costs, fisher, tps)
    print(f"[allocator] cost norms (per-tensor mean): "
          f"PPL={norms[0]:.3e}  TG={norms[1]:.3e}  PP={norms[2]:.3e}", flush=True)

    budget_bytes = int(args.budget_gb * (1024 ** 3))
    band_bytes = int(args.budget_band_gb * (1024 ** 3))
    lam, recipe, total_size, total_loss = bisect_lambda(
        fisher, costs, pinned, budget_bytes,
        weights=weights, tps=tps, norms=norms,
        band_bytes=band_bytes)
    delta_gb = (total_size - budget_bytes) / (1024 ** 3)
    in_band = abs(delta_gb) <= args.budget_band_gb
    band_marker = "✓ in-band" if in_band else "⚠ out-of-band"
    print(f"[allocator] solved: λ={lam:.6e}  recipe-size={total_size/(1024**3):.2f} GB  "
          f"budget={args.budget_gb:.2f}±{args.budget_band_gb:.2f} GB  "
          f"Δ={delta_gb:+.2f} GB ({band_marker})  loss-surrogate={total_loss:.6e}",
          flush=True)

    # Format distribution
    fmt_counts = defaultdict(int)
    for f in recipe.values(): fmt_counts[f] += 1
    print(f"[allocator] format distribution:", flush=True)
    for f, c in sorted(fmt_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {f:>12s}: {c}", flush=True)

    Path(args.recipe_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.recipe_out, "w") as f:
        json.dump({
            "recipe": recipe,
            "budget_gb": args.budget_gb,
            "actual_size_bytes": total_size,
            "actual_size_gb": total_size / (1024**3),
            "loss_surrogate": total_loss,
            "lambda": lam,
            "priority": args.priority,
            "weights": {"ppl": weights[0], "tg": weights[1], "pp": weights[2]},
            "format_counts": dict(fmt_counts),
        }, f, indent=2)
    print(f"[allocator] wrote recipe to {args.recipe_out}", flush=True)

    # Emit a text version that --tensor-type-file consumes directly.  Each
    # entry is `^name$=type` so the regex-search match is anchored — exact
    # tensor name only.  llama-quantize lowercases the pattern; GGUF tensor
    # names are already lowercase so this is a no-op.
    txt_path = str(args.recipe_out).replace(".json", ".txt")
    if txt_path == args.recipe_out:
        txt_path = args.recipe_out + ".txt"
    with open(txt_path, "w") as f:
        for tensor_name, fmt in sorted(recipe.items()):
            f.write(f"^{tensor_name}$={fmt}\n")
    print(f"[allocator] wrote --tensor-type-file format to {txt_path}", flush=True)

    # Pareto sweep
    if args.pareto_csv and args.pareto_budgets_gb:
        with open(args.pareto_csv, "w") as f:
            w = csv.writer(f)
            w.writerow(["budget_gb", "lambda", "actual_size_gb", "loss_surrogate", *sorted({f for fmts in costs.values() for f in fmts})])
            fmt_keys = sorted({f for fmts in costs.values() for f in fmts})
            for b_str in args.pareto_budgets_gb.split(","):
                b = float(b_str)
                bytes_b = int(b * (1024**3))
                ll, rr, ss, lo = bisect_lambda(
                    fisher, costs, pinned, bytes_b,
                    weights=weights, tps=tps, norms=norms)
                fc = defaultdict(int)
                for ff in rr.values(): fc[ff] += 1
                w.writerow([b, ll, ss/(1024**3), lo, *[fc[k] for k in fmt_keys]])
        print(f"[allocator] wrote Pareto sweep to {args.pareto_csv}", flush=True)


if __name__ == "__main__":
    main()
