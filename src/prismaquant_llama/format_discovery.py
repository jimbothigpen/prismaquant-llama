"""
Auto-discover supported quantization formats from a llama-quantize binary
and overlay our curated metadata. Result is cached per binary hash so
re-runs are fast.

Why hybrid auto-discovery:
- `llama-quantize --help` gives the canonical list of supported types
  for THIS binary (handles fork-specific additions automatically).
- Pure auto-discovery surfaces internal-use types (F32, COPY) that
  aren't allocator candidates — need a denylist regardless.
- Metadata overlay gives us bpw, family, source, recommend flag for the
  wizard UX (sort by bpw, group by family, default to recommended-only).
- Cache hit: re-runs are instant. Cache miss: ~50ms to parse --help.

Public API:
    discover_formats(binary_path: Path, all: bool = False) -> dict[str, FormatInfo]
"""

from __future__ import annotations
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_THIS_DIR = Path(__file__).parent
METADATA_BASE_PATH = _THIS_DIR / "format_metadata_base.json"

# Metadata search paths (highest priority last — later layers override).
# Convention path for fork-specific extensions: <binary>/../../tools/prismaquant/
def _convention_metadata_paths(binary_path: Optional[Path]) -> list[Path]:
    paths = [METADATA_BASE_PATH]  # always loaded as floor
    if binary_path:
        # binary lives at <fork>/build/bin/llama-quantize → fork is parents[2]
        fork_root = binary_path.resolve().parents[2] if len(binary_path.resolve().parents) >= 2 else None
        if fork_root:
            fork_meta_dir = fork_root / "tools" / "prismaquant"
            if fork_meta_dir.exists():
                paths.extend(sorted(fork_meta_dir.glob("format_metadata_*.json")))
    # User overrides
    user_dir = Path.home() / ".config" / "prismaquant-wizard"
    if user_dir.exists():
        paths.extend(sorted(user_dir.glob("format_metadata_*.json")))
    return paths


CACHE_ROOT = Path.home() / ".cache" / "prismaquant-wizard" / "binary-types"

# Lines look like:  "  52  or  IQ4_K   :  4.50 bpw imatrix 4-bit (ik_llama)"
# or:               "  17  or  Q5_K    : alias for Q5_K_M"
# Some entries omit the leading number (e.g. "COPY    : only copy tensors").
_RE_TYPE_LINE = re.compile(
    r"""^\s*
        (?:(?P<id>\d+)\s+or\s+)?     # optional numeric ID
        (?P<name>[A-Z][A-Z0-9_]*)    # format name (uppercase + digits + _)
        \s*:\s*
        (?P<desc>.+?)\s*$            # rest = description
    """,
    re.VERBOSE,
)

# Hard denylist — types that show up in --help but aren't allocator candidates.
DENYLIST = frozenset({
    "F32",       # internal full-precision; not a quantization
    "COPY",      # not a quant; copy mode
})


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FormatInfo:
    name: str
    id: Optional[int] = None
    binary_desc: str = ""
    bpw: Optional[float] = None
    family: str = "?"          # k / i / ik / tq / fp / ?
    source: str = "?"          # mainline / ikllama / frankenturbo / ?
    recommend: bool = False
    note: str = ""
    in_metadata: bool = False  # whether we have curated info for this format


# ─────────────────────────────────────────────────────────────────────────────
# Discovery
# ─────────────────────────────────────────────────────────────────────────────

def _binary_hash(binary_path: Path) -> str:
    """SHA-256 of the binary file (first 16 hex chars used for cache key).
    Hashing the whole file is fast (~50MB binary in <100ms on NVMe)."""
    h = hashlib.sha256()
    with binary_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _parse_help_output(help_text: str) -> list[tuple[Optional[int], str, str]]:
    """Extract (id, name, desc) tuples from --help output. Tolerant of
    column reflow and missing IDs. Lines without a recognizable format
    name are skipped silently."""
    types: list[tuple[Optional[int], str, str]] = []
    in_types_block = False
    for line in help_text.splitlines():
        if "allowed quantization types" in line.lower():
            in_types_block = True
            continue
        if not in_types_block:
            # Some binaries dump types at top of --help; tolerate that too
            # by always trying to parse, but still recognize the section
            # marker if present.
            pass
        m = _RE_TYPE_LINE.match(line)
        if not m:
            continue
        id_str = m.group("id")
        type_id = int(id_str) if id_str else None
        name = m.group("name").upper()
        desc = m.group("desc")
        if name in DENYLIST:
            continue
        types.append((type_id, name, desc))
    return types


def _load_metadata(binary_path: Optional[Path] = None) -> dict[str, dict]:
    """Merge format_metadata_*.json files in search-path order. Later
    files override earlier ones — base < fork extension < user override."""
    merged: dict[str, dict] = {}
    for path in _convention_metadata_paths(binary_path):
        if not path.exists():
            continue
        try:
            raw = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for k, v in raw.items():
            if k.startswith("_"):
                continue
            merged[k] = v
    return merged


def _cache_path(binary_hash: str) -> Path:
    return CACHE_ROOT / f"{binary_hash}.json"


def _try_load_cache(binary_hash: str) -> Optional[dict[str, FormatInfo]]:
    p = _cache_path(binary_hash)
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text())
        return {name: FormatInfo(**info) for name, info in raw.items()}
    except Exception:
        return None


def _save_cache(binary_hash: str, formats: dict[str, FormatInfo]) -> None:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    p = _cache_path(binary_hash)
    serial = {name: asdict(info) for name, info in formats.items()}
    p.write_text(json.dumps(serial, indent=2))


# Hard-coded fallback if `--help` parsing returns zero hits (e.g., binary
# format changed upstream). Mirrors the prior wizard.py DEFAULT_FORMATS.
_FALLBACK_FORMATS = [
    "Q4_K", "Q5_K", "Q6_K", "Q8_0",
    "IQ4_XS", "IQ4_NL",
    "IQ4_K", "IQ4_KS", "IQ4_KSS",
    "IQ3_K", "IQ3_KS",
    "IQ2_K",
]


def discover_formats(
    binary_path: Path,
    all_formats: bool = False,
    use_cache: bool = True,
) -> dict[str, FormatInfo]:
    """
    Discover supported quantization formats from a llama-quantize binary.

    Args:
        binary_path: path to llama-quantize binary
        all_formats: if False (default), return only formats marked
            recommend=true in metadata. If True, return everything the
            binary supports (intersected with non-deny metadata).
        use_cache: whether to use ~/.cache/prismaquant/binary-types/

    Returns:
        dict[name, FormatInfo] — sorted by bpw ascending then by name.
    """
    binary_path = Path(binary_path).resolve()
    if not binary_path.exists():
        raise FileNotFoundError(f"binary not found: {binary_path}")

    bhash = _binary_hash(binary_path)
    if use_cache:
        cached = _try_load_cache(bhash)
        if cached is not None:
            return _filter(cached, all_formats=all_formats)

    # Run --help and parse
    try:
        proc = subprocess.run(
            [str(binary_path), "--help"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        help_text = proc.stdout + "\n" + proc.stderr
    except subprocess.TimeoutExpired:
        help_text = ""

    parsed = _parse_help_output(help_text)
    metadata = _load_metadata(binary_path)

    formats: dict[str, FormatInfo] = {}
    for type_id, name, desc in parsed:
        meta = metadata.get(name, {})
        if meta.get("exclude_from_wizard"):
            continue
        formats[name] = FormatInfo(
            name=name,
            id=type_id,
            binary_desc=desc,
            bpw=meta.get("bpw"),
            family=meta.get("family", "?"),
            source=meta.get("source", "?"),
            recommend=meta.get("recommend", False),
            note=meta.get("note", ""),
            in_metadata=name in metadata,
        )

    # Fallback if parsing yielded nothing
    if not formats:
        for name in _FALLBACK_FORMATS:
            meta = metadata.get(name, {})
            formats[name] = FormatInfo(
                name=name,
                bpw=meta.get("bpw"),
                family=meta.get("family", "?"),
                source=meta.get("source", "?"),
                recommend=meta.get("recommend", False),
                note=meta.get("note", "fallback (parse failed)"),
                in_metadata=name in metadata,
            )

    if use_cache:
        _save_cache(bhash, formats)

    return _filter(formats, all_formats=all_formats)


def _filter(formats: dict[str, FormatInfo], all_formats: bool) -> dict[str, FormatInfo]:
    """Apply recommend filter unless all_formats=True. Sort by bpw then name."""
    if not all_formats:
        formats = {n: f for n, f in formats.items() if f.recommend}
    return dict(sorted(formats.items(), key=lambda kv: (kv[1].bpw or 99, kv[0])))


# ─────────────────────────────────────────────────────────────────────────────
# CLI for ad-hoc inspection
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Inspect supported quant formats from a llama-quantize binary.")
    p.add_argument("binary", type=Path, help="path to llama-quantize")
    p.add_argument("--all", action="store_true", help="show all supported (not just recommended)")
    p.add_argument("--no-cache", action="store_true", help="skip cache, always parse --help")
    args = p.parse_args(argv)

    fmts = discover_formats(args.binary, all_formats=args.all, use_cache=not args.no_cache)
    print(f"  binary: {args.binary}")
    print(f"  found:  {len(fmts)} formats ({'all' if args.all else 'recommended only'})")
    print()
    print(f"  {'name':<12} {'bpw':>6}  {'family':<8} {'source':<14} {'note'}")
    print(f"  {'-'*12} {'-'*6}  {'-'*8} {'-'*14} {'-'*40}")
    for name, info in fmts.items():
        bpw = f"{info.bpw:.2f}" if info.bpw is not None else "?"
        note = info.note or ("(no curated metadata)" if not info.in_metadata else "")
        print(f"  {name:<12} {bpw:>6}  {info.family:<8} {info.source:<14} {note}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
