#!/usr/bin/env python3
"""Bridge prismaquant probe.pkl HF tensor names to GGUF tensor names.

Tailored for qwen35moe (Qwen3.6-35B-A3B): hybrid Gated-Delta-Net + softmax MoE.
- 10 softmax-attn layers (every 4th): self_attn.{q,k,v,o}_proj -> attn_{q,k,v,output}
- 30 linear-attn layers: linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj}
  -> attn_qkv, attn_gate, ssm_alpha, ssm_beta, ssm_out
- All layers: mlp.experts.{gate_up_proj,down_proj} (HF pre-packed across 256 experts)
  -> ffn_{gate,up}_exps + ffn_down_exps  (gate_up split into 2 with h/2 each)
- All layers: mlp.shared_expert.{gate,up,down}_proj -> ffn_{gate,up,down}_shexp
- Top-level: lm_head -> output

Probe entries omit the trailing `.weight` (e.g. `model.layers.7.self_attn.q_proj`),
so patterns match without it. mtp.* entries (multi-token-prediction head) are
dropped: those tensors are not present in the deployed GGUF.

Output: same JSON shape as before — {h_trace: {gguf_name: float}, ...}.
"""

import argparse
import json
import pickle
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path

# Each entry is (regex, list of GGUF-name format strings, weight_split).
# weight_split distributes the source HF Fisher across N GGUF targets.
# A list of length N with sum=1.0 is required.
PATTERNS = [
    # Softmax-attn layers (separate q/k/v/output in both HF and GGUF; 1:1)
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.q_proj$"),
     ["blk.{0}.attn_q.weight"], [1.0]),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.k_proj$"),
     ["blk.{0}.attn_k.weight"], [1.0]),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.v_proj$"),
     ["blk.{0}.attn_v.weight"], [1.0]),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.o_proj$"),
     ["blk.{0}.attn_output.weight"], [1.0]),
    # Linear-attn (Gated-Delta-Net) layers
    (re.compile(r"^model\.layers\.(\d+)\.linear_attn\.in_proj_qkv$"),
     ["blk.{0}.attn_qkv.weight"], [1.0]),
    (re.compile(r"^model\.layers\.(\d+)\.linear_attn\.in_proj_z$"),
     ["blk.{0}.attn_gate.weight"], [1.0]),
    (re.compile(r"^model\.layers\.(\d+)\.linear_attn\.in_proj_a$"),
     ["blk.{0}.ssm_alpha.weight"], [1.0]),
    (re.compile(r"^model\.layers\.(\d+)\.linear_attn\.in_proj_b$"),
     ["blk.{0}.ssm_beta.weight"], [1.0]),
    (re.compile(r"^model\.layers\.(\d+)\.linear_attn\.out_proj$"),
     ["blk.{0}.ssm_out.weight"], [1.0]),
    # MoE experts (pre-packed in HF probe across 256 experts)
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.down_proj$"),
     ["blk.{0}.ffn_down_exps.weight"], [1.0]),
    # gate_up_proj is fused gate+up; GGUF splits into two tensors
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.gate_up_proj$"),
     ["blk.{0}.ffn_gate_exps.weight", "blk.{0}.ffn_up_exps.weight"], [0.5, 0.5]),
    # Shared expert (always-on MLP)
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.shared_expert\.gate_proj$"),
     ["blk.{0}.ffn_gate_shexp.weight"], [1.0]),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.shared_expert\.up_proj$"),
     ["blk.{0}.ffn_up_shexp.weight"], [1.0]),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.shared_expert\.down_proj$"),
     ["blk.{0}.ffn_down_shexp.weight"], [1.0]),
    # Router gate (rare in probe; kept for completeness)
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.gate$"),
     ["blk.{0}.ffn_gate_inp.weight"], [1.0]),
    # Dense FFN (qwen3_5 hybrid like Qwopus3.5-9B-v3.5: no experts, plain MLP).
    # NOTE: regex requires that mlp.{gate,up,down}_proj NOT be preceded by
    # `experts.` or `shared_expert.` — those have their own patterns above.
    # Anchored on the layer prefix so they only match the bare dense form.
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.gate_proj$"),
     ["blk.{0}.ffn_gate.weight"], [1.0]),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.up_proj$"),
     ["blk.{0}.ffn_up.weight"], [1.0]),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.down_proj$"),
     ["blk.{0}.ffn_down.weight"], [1.0]),
    # Top-level
    (re.compile(r"^lm_head$"), ["output.weight"], [1.0]),
]

# Drop these prefixes silently (not present in deployed GGUF):
DROP_PREFIXES = ("mtp.",)


def map_hf_to_gguf(hf_name: str) -> list[tuple[str, float]] | None:
    """Returns [(gguf_name, weight_fraction), ...] or None if unmapped/dropped."""
    if any(hf_name.startswith(p) for p in DROP_PREFIXES):
        return []  # drop signal (vs None = unmapped warning)
    # Tolerate trailing `.weight` if some probes include it.
    base = hf_name[:-len(".weight")] if hf_name.endswith(".weight") else hf_name
    for pat, fmts, weights in PATTERNS:
        m = pat.match(base)
        if m:
            return [(fmt.format(*m.groups()), w) for fmt, w in zip(fmts, weights)]
    return None


# Minimal GGUF tensor-name reader (no full gguf-py dep). Only walks the header
# to extract the tensor-info section; doesn't need tensor data.
def read_gguf_tensor_names(path: str) -> set[str]:
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
        # Skip KV section
        for _ in range(kv_count):
            read_string()
            t, = struct.unpack("<I", f.read(4))
            skip_value(t)
        # Tensor info section: name (string), n_dims (u32), dims (n_dims*u64),
        # type (u32), offset (u64)
        names = set()
        for _ in range(tensor_count):
            tn = read_string()
            n_dims, = struct.unpack("<I", f.read(4))
            f.read(8 * n_dims)  # dims
            f.read(4 + 8)        # type + offset
            names.add(tn)
        return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True, help="prismaquant probe.pkl")
    ap.add_argument("--output", required=True, help="JSON map (gguf_name -> h_trace)")
    ap.add_argument("--aggregate", choices=["sum", "max"], default="sum",
                    help="How to combine multiple HF sources mapping to one GGUF target")
    ap.add_argument("--unmapped-out", default=None,
                    help="Optional JSON listing HF names that didn't map (diagnostic)")
    ap.add_argument("--verify-gguf", default=None,
                    help="Optional GGUF path; if set, checks every emitted GGUF name "
                         "exists as a tensor in the file and warns on mismatches")
    args = ap.parse_args()

    with open(args.probe, "rb") as f:
        probe = pickle.load(f)

    stats = probe.get("stats", {})
    print(f"[bridge] probe has {len(stats)} HF tensor entries", flush=True)
    print(f"[bridge] meta: {probe.get('meta', {}).get('model', '<unknown>')}", flush=True)
    print(f"[bridge] expert_saliency entries: "
          f"{sum(len(v) for v in probe.get('expert_saliency', {}).values())}", flush=True)

    h_trace_per_gguf: dict[str, list[float]] = defaultdict(list)
    unmapped: list[str] = []
    dropped: list[str] = []
    n_split = 0

    for hf_name, blob in stats.items():
        if not isinstance(blob, dict):
            continue
        if "h_trace_raw" not in blob:
            continue
        targets = map_hf_to_gguf(hf_name)
        if targets is None:
            unmapped.append(hf_name)
            continue
        if not targets:  # explicit drop
            dropped.append(hf_name)
            continue
        h = float(blob["h_trace_raw"])
        if len(targets) > 1:
            n_split += 1
        for gguf_name, frac in targets:
            h_trace_per_gguf[gguf_name].append(h * frac)

    aggregated = {}
    for gguf_name, h_list in h_trace_per_gguf.items():
        if args.aggregate == "sum":
            aggregated[gguf_name] = float(sum(h_list))
        else:
            aggregated[gguf_name] = float(max(h_list))

    expert_packed = sum(1 for k in aggregated if "_exps." in k)
    layer_one_to_one = len(aggregated) - expert_packed

    print(f"[bridge] mapped to {len(aggregated)} GGUF tensors", flush=True)
    print(f"[bridge]   packed-expert tensors: {expert_packed}", flush=True)
    print(f"[bridge]   layer/single tensors:  {layer_one_to_one}", flush=True)
    print(f"[bridge]   split sources (gate_up): {n_split}", flush=True)
    print(f"[bridge]   dropped (mtp/etc):     {len(dropped)}", flush=True)
    print(f"[bridge]   unmapped HF entries:   {len(unmapped)}", flush=True)
    if unmapped[:10]:
        print(f"[bridge] first unmapped (sample): {unmapped[:10]}", flush=True)

    verify_warnings: list[str] = []
    if args.verify_gguf:
        gguf_names = read_gguf_tensor_names(args.verify_gguf)
        missing_in_gguf = [n for n in aggregated if n not in gguf_names]
        # Diagnostic: which large-Fisher tensors in GGUF have no probe coverage?
        print(f"[bridge] GGUF has {len(gguf_names)} tensors total", flush=True)
        print(f"[bridge] emitted names missing from GGUF: {len(missing_in_gguf)}", flush=True)
        if missing_in_gguf:
            print(f"[bridge]   sample: {missing_in_gguf[:10]}", flush=True)
            verify_warnings.extend(missing_in_gguf)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "h_trace": aggregated,
            "aggregate": args.aggregate,
            "n_hf_entries": len(stats),
            "n_gguf_tensors": len(aggregated),
            "n_unmapped": len(unmapped),
            "n_dropped": len(dropped),
            "verify_warnings": verify_warnings,
            "probe_meta": probe.get("meta", {}),
        }, f, indent=2, default=str)
    print(f"[bridge] wrote {args.output}", flush=True)

    if args.unmapped_out:
        with open(args.unmapped_out, "w") as f:
            json.dump({"unmapped": unmapped, "dropped": dropped}, f, indent=2)


if __name__ == "__main__":
    main()
