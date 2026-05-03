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
import os
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


# Centralized in paths.py.
try:
    from .paths import DEFAULT_CACHE_ROOT as CACHE_ROOT
except ImportError:
    from paths import DEFAULT_CACHE_ROOT as CACHE_ROOT  # type: ignore

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

# Hard denylist — uppercase tokens that match the parser regex but aren't
# actual quantization types. Includes section headers, warning labels, and
# meta-modes that some llama.cpp builds emit in their --help output.
DENYLIST = frozenset({
    "F32",         # internal full-precision; not a quantization
    "COPY",        # not a quant; copy mode
    "WARNING",     # warning-block prefix
    "NOTE",        # note-block prefix
    "IMPORTANT",   # general help-text label
    "ERROR",       # error-block prefix
    "USAGE",       # usage-line prefix in some builds
    "FORMATS",     # plural — some help builds use it as a section header
    "TYPES",
    "EXAMPLE",
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


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic classification — used when no curated metadata covers a format
# ─────────────────────────────────────────────────────────────────────────────

_RE_BPW   = re.compile(r"(\d+(?:\.\d+)?)\s*bpw", re.IGNORECASE)
_RE_GBYTE = re.compile(r"(\d+(?:\.\d+)?)\s*G\b")  # "4.34G" — size-at-Llama3-8B; bpw ≈ size

# Family pattern matching by NAME alone (no fork-attribution by name)
_RE_FAMILY = [
    (re.compile(r"^IQ\d+_K(S{1,2}|T)?$"),    "ik"),  # IQ4_K, IQ4_KS, IQ4_KSS, IQ4_KT, IQ3_K, IQ3_KS, IQ2_K
    (re.compile(r"^IQ\d+(_[A-Z]+)?$"),        "i"),  # IQ4_NL, IQ4_XS, IQ3_S, IQ3_M, IQ2_S, IQ1_M, etc.
    (re.compile(r"^Q\d+_K(_[SML])?$"),        "k"),  # Q2_K, Q4_K_S, Q5_K_M, Q6_K
    (re.compile(r"^Q\d+_[01]$"),              "k"),  # Q4_0, Q4_1, Q5_0, Q5_1, Q8_0
    (re.compile(r"^Q\d+_K_[SML]$"),           "k"),
    (re.compile(r"^Q1_0(_G\d+)?$"),           "k"),  # Q1_0, Q1_0_G128 (carlosfundora ternary)
    (re.compile(r"^TQ\d+_[01][A-Z]*$"),       "tq"), # TQ3_0, TQ3_1S, TQ3_4S, TQ4_1S, TQ3_1S_AP1
    (re.compile(r"^Q\d+_[01]_TQ$"),           "tq"), # Q4_0_TQ, Q4_1_TQ
    (re.compile(r"^MXFP\d+(_MOE)?$"),         "fp"),
    (re.compile(r"^NVFP\d+$"),                "fp"),
    (re.compile(r"^F(16|32)$|^BF16$"),        "fp"),
]

# Source attribution from --help description (fork-specific markers)
_RE_SOURCE_PATTERNS = [
    (re.compile(r"\(ik_llama\)", re.IGNORECASE),                                "ikllama"),
    (re.compile(r"WHT|Lloyd-Max|ternarization|ternary|four E3M5|four scales\+shifts|promoted superblock"),
                                                                                "frankenturbo"),
]

_RECOMMEND_DEFAULT_NAMES = frozenset({
    "Q4_K", "Q4_K_S", "Q4_K_M", "Q5_K", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0",
    "IQ4_XS",
    "IQ4_K", "IQ4_KS", "IQ4_KSS", "IQ3_K", "IQ3_KS", "IQ2_K",
})


def _heuristic_classify(name: str, desc: str) -> dict:
    """Best-effort classification from name + --help description.
    Used when the format isn't covered by base/extension metadata."""
    bpw: Optional[float] = None
    m = _RE_BPW.search(desc)
    if m:
        bpw = float(m.group(1))
    else:
        # Fallback: derive bpw from "X.XXG" reference-size mention. The
        # llama.cpp --help quotes sizes at Llama-3-8B (~8B params); for
        # 8B params, size_GB ≈ bpw within ~10%.
        gm = _RE_GBYTE.search(desc)
        if gm:
            bpw = float(gm.group(1)) * 1.07  # rough fudge factor

    family = "?"
    for rx, fam in _RE_FAMILY:
        if rx.match(name):
            family = fam
            break

    source = "mainline"  # default — overridden by description markers
    for rx, src in _RE_SOURCE_PATTERNS:
        if rx.search(desc):
            source = src
            break

    return {
        "bpw": round(bpw, 4) if bpw is not None else None,
        "family": family,
        "source": source,
        "recommend": name in _RECOMMEND_DEFAULT_NAMES,
        "note": "(heuristic-classified)" if family != "?" else "(unrecognized; verify before use)",
    }


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

    # Run --help and parse. Inject LD_LIBRARY_PATH for binaries that depend
    # on libllama.so installed alongside (e.g., /path/to/llama.cpp/build/bin + /opt/llama/lib).
    env = dict(os.environ)
    bin_dir = binary_path.parent
    candidate_lib_dirs = [bin_dir, bin_dir.parent / "lib", bin_dir]  # try sibling-bin, ../lib, then bin again
    extra_paths = [str(p) for p in candidate_lib_dirs if p.is_dir()]
    if extra_paths:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(extra_paths + [env.get("LD_LIBRARY_PATH", "")])
    try:
        proc = subprocess.run(
            [str(binary_path), "--help"],
            capture_output=True, text=True, timeout=30, check=False, env=env,
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
        # Heuristic-classify FIRST (always derives something from name + --help
        # description), then let curated metadata override field-by-field. This
        # gives unknown formats sensible defaults instead of "?" everywhere.
        # Note: we only use the heuristic's `note` if the format is NOT in
        # curated metadata at all — otherwise empty string (curated format
        # without an explicit note field is correctly noteless, not heuristic).
        heuristic = _heuristic_classify(name, desc)
        in_meta = name in metadata
        formats[name] = FormatInfo(
            name=name,
            id=type_id,
            binary_desc=desc,
            bpw=meta.get("bpw", heuristic["bpw"]),
            family=meta.get("family", heuristic["family"]),
            source=meta.get("source", heuristic["source"]),
            recommend=meta.get("recommend", heuristic["recommend"]),
            note=meta.get("note", "" if in_meta else heuristic["note"]),
            in_metadata=in_meta,
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

def generate_metadata_skeleton(binary_path: Path,
                               output_path: Path,
                               include_already_curated: bool = False) -> int:
    """
    Auto-generate a format_metadata_<forkname>.json skeleton from a
    binary's --help output. Heuristic-classified for bpw / family / source;
    user can edit before saving as a discovery extension.

    By default, formats already covered by the loaded base/extension
    metadata are EXCLUDED (so the user only edits formats the wizard
    doesn't already know about). Pass include_already_curated=True to
    dump everything.
    """
    fmts = discover_formats(binary_path, all_formats=True, use_cache=False)
    out: dict = {
        "_comment": (
            f"Auto-generated by `prismaquant-llama discover --generate-metadata` "
            f"from {binary_path.name} on {Path.cwd()}. "
            f"Edit the `recommend` flags + add `note` fields for the "
            f"formats you want to surface in the wizard's recommend list. "
            f"Drop this file at <binary>/../../tools/prismaquant/"
            f"format_metadata_<forkname>.json (fork-shipped) or "
            f"~/.config/prismaquant-llama/format_metadata_<name>.json "
            f"(user override) for auto-discovery on next run."
        )
    }
    n_emitted = 0
    for name, info in fmts.items():
        if not include_already_curated and info.in_metadata:
            continue
        entry = {
            "bpw": round(info.bpw, 4) if info.bpw is not None else None,
            "family": info.family,
            "source": info.source,
            "recommend": False,    # default to false; user sets true for ones they want
        }
        if info.note:
            entry["note"] = info.note
        elif not info.in_metadata:
            entry["note"] = "(auto-generated; review before enabling recommend=true)"
        out[name] = entry
        n_emitted += 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2))
    return n_emitted


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Inspect supported quant formats from a llama-quantize binary.")
    p.add_argument("binary", type=Path, help="path to llama-quantize")
    p.add_argument("--all", action="store_true", help="show all supported (not just recommended)")
    p.add_argument("--no-cache", action="store_true", help="skip cache, always parse --help")
    p.add_argument("--generate-metadata", type=Path, metavar="FILE",
                   help="emit a fork-extension metadata skeleton JSON to FILE — "
                        "covers only formats not already in base/extension metadata "
                        "(use --generate-all to dump everything)")
    p.add_argument("--generate-all", action="store_true",
                   help="with --generate-metadata, include formats already in "
                        "base/extension metadata (overrides instead of extending)")
    args = p.parse_args(argv)

    if args.generate_metadata:
        n = generate_metadata_skeleton(args.binary, args.generate_metadata,
                                       include_already_curated=args.generate_all)
        print(f"  ✓ wrote {n} format entries to {args.generate_metadata}")
        print(f"  → review + edit recommend flags, then drop the file at:")
        print(f"      <binary>/../../tools/prismaquant/format_metadata_<forkname>.json")
        print(f"    (fork-shipped extension, auto-discovered) OR")
        print(f"      ~/.config/prismaquant-llama/format_metadata_<name>.json")
        print(f"    (personal override, applies to all binaries)")
        return 0

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

    # Suggested --formats presets — computed from full availability (always
    # use all_formats=True so recommended-only table still shows wide preset).
    fmts_all = (fmts if args.all
                else discover_formats(args.binary, all_formats=True, use_cache=not args.no_cache))
    print()
    print("Suggested --formats presets (copy-paste into `prismaquant-llama pipeline run`):")
    print()

    # Conservative mainline (5 staples)
    conservative = ["Q4_K", "Q5_K", "Q6_K", "Q8_0", "IQ4_XS"]
    conservative_avail = [f for f in conservative if f in fmts_all]
    if len(conservative_avail) >= 4:
        print(f"  conservative (mainline staples — safe for any binary):")
        print(f"    --formats {','.join(conservative_avail)}")
        print()

    # Wide mainline — adds 2- and 3-bit formats
    wide_mainline = ["Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0",
                     "IQ2_S", "IQ3_XXS", "IQ3_S",
                     "IQ4_XS", "IQ4_NL"]
    wide_avail = [f for f in wide_mainline if f in fmts_all and fmts_all[f].source == "mainline"]
    if len(wide_avail) >= 7:
        print(f"  wide mainline (broad bpw coverage, allocator's default):")
        print(f"    --formats {','.join(wide_avail)}")
        print()

    # Mainline + IK-K (only if binary supports them)
    ikk_extensions = ["IQ4_K", "IQ4_KS", "IQ4_KSS", "IQ3_K", "IQ3_KS"]
    ikk_avail = [f for f in ikk_extensions if f in fmts_all]
    if ikk_avail:
        wide_plus_ikk = wide_avail + ikk_avail
        print(f"  wide + IK-K extensions (this binary supports {len(ikk_avail)} fork formats):")
        print(f"    --formats {','.join(wide_plus_ikk)}")
        print()

    # Note about formats not in any preset — known PPL cliffs / very lossy
    # only. Q2_K's "extreme low-bit" hedge is for non-imatrix workflows;
    # prismaquant always builds imatrix in Stage D, so Q2_K is fine to include.
    extreme_excluded = []
    cliff_types = {"IQ2_K"}    # documented +30 PPL cliff with imatrix
    for name, info in fmts_all.items():
        if name in cliff_types or (info.note and "very lossy" in info.note):
            extreme_excluded.append(name)
    if extreme_excluded:
        print(f"  (excluded from suggestions — known PPL cliffs / very lossy):")
        print(f"    {', '.join(extreme_excluded)}")
        print(f"  Add via --formats only if you specifically need extreme compression.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
