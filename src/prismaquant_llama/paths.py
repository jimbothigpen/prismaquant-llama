"""
Path management for prismaquant-wizard.

Handles two responsibilities:

  1. **Binary discovery** — find llama-quantize via --binary flag → $PATH →
     common build dirs → prompt. Same pattern for llama-perplexity / llama-bench.

  2. **Output directory tree** — creates and tracks the convention layout:

        <output>/                                   --output / -o
        ├── _shared/                                 reusable across runs
        │   ├── calibration/                         wikitext / c4 / the-pile
        │   ├── hf-cache/                            HF safetensors (or override)
        │   └── imatrix-cache/                       cached imatrix files (keyed by hash)
        ├── ggufs/                                   final deliverable artifacts
        └── work/<model>-<timestamp>/                per-run scratch (cleanable)
            ├── bf16/                                HF → GGUF intermediate
            ├── probe/                               prismaquant Hessian probe
            ├── imatrix/                             working copy (symlink to cache)
            ├── costs/                               Stage D MSE measurements
            ├── recipes/                             allocator JSON outputs
            └── logs/                                per-stage log files

  Imatrix caching is keyed by (model_sha256, corpus_sha256, n_chunks) so
  the most common power-user iteration loop ("try priority 522, 900, 333
  on the same model") reuses imatrix across runs.

Default output root: ~/prismaquant-builds (visible; ggufs/ is what users
actively want to find).
"""

from __future__ import annotations
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


# Source-format suffixes to strip when sanitizing HF model IDs into filenames.
# Case-insensitive, matches at end of name. e.g.
#   "unsloth/gpt-oss-20b-BF16"  → "gpt-oss-20b"
#   "Qwen/Qwen3-30B-A3B-Instruct" → "Qwen3-30B-A3B-Instruct" (no change)
_SOURCE_FORMAT_SUFFIX_RE = re.compile(
    r"-(?:BF16|FP16|bf16|fp16|F32|F16|FP32)$",
    re.IGNORECASE,
)


def sanitize_model_name(hf_model_id: str) -> str:
    """Convert an HF model ID into a clean filename component.

    Drops the org prefix (everything before the last `/`) and strips
    common source-format suffixes (`-BF16`, `-FP16`, etc.). The result
    is what shows up in run labels, BF16 GGUF cache filenames, and
    final PQ GGUF filenames.

    Examples:
        unsloth/gpt-oss-20b-BF16              → gpt-oss-20b
        google/gemma-4-E4B-it                 → gemma-4-E4B-it
        Qwen/Qwen3-30B-A3B-Instruct           → Qwen3-30B-A3B-Instruct
        Jackrong/Qwopus3.5-9B-v3.5            → Qwopus3.5-9B-v3.5
    """
    # Take the segment after the last `/` (drop org prefix).
    short = hf_model_id.rsplit("/", 1)[-1]
    # Strip source-format suffix if present.
    short = _SOURCE_FORMAT_SUFFIX_RE.sub("", short)
    # Filesystem-safe: replace any remaining unsafe chars (rare).
    short = short.replace(" ", "_")
    return short


# ─────────────────────────────────────────────────────────────────────────────
# Defaults — single root for everything
# ─────────────────────────────────────────────────────────────────────────────
#
# Layout under PRISMAQUANT_LLAMA_ROOT:
#   builds/                  ← pipeline outputs (--output default)
#       <model>-prismaquant/
#           _shared/, ggufs/, work/
#   cache/binary-types/      ← per-binary calibration JSONs (perf, calibrated)
#   config/                  ← system-wide preferences
#       system-default-format-perf.json
#   scratch/                 ← per-format temp ggufs (calibrate-deep)
#
# All overridable individually via flags, OR globally via the env var:
#   PRISMAQUANT_LLAMA_ROOT=/some/path  → relocates everything
#
# Legacy lookups (~/prismaquant-builds, ~/.cache/prismaquant-{llama,wizard},
# ~/.config/prismaquant-llama) remain supported for read in
# find_format_perf_file_for_binary so existing data is still discovered.
PRISMAQUANT_LLAMA_ROOT = Path(
    os.environ.get("PRISMAQUANT_LLAMA_ROOT") or (Path.home() / ".prismaquant-llama")
)

DEFAULT_OUTPUT_ROOT = PRISMAQUANT_LLAMA_ROOT / "builds"
DEFAULT_CACHE_ROOT = PRISMAQUANT_LLAMA_ROOT / "cache" / "binary-types"
DEFAULT_CONFIG_DIR = PRISMAQUANT_LLAMA_ROOT / "config"
DEFAULT_SCRATCH_ROOT = PRISMAQUANT_LLAMA_ROOT / "scratch"
DEFAULT_SYSTEM_PERF_PATH = DEFAULT_CONFIG_DIR / "system-default-format-perf.json"
DEFAULT_USER_FORMATS_PATH = DEFAULT_CONFIG_DIR / "default-formats.txt"
DEFAULT_USER_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.toml"


_user_config_cache: "Optional[dict]" = None


def load_user_config() -> dict:
    """Load user's config.toml from ~/.prismaquant-llama/config/config.toml.

    Returns a parsed dict. If the file is missing, malformed, or empty,
    returns an empty dict (callers should treat missing keys as "use
    built-in default"). Cached after first call.

    Schema sections (all optional):
        [paths]              — output_root, hf_cache, scratch, calibration, etc.
        [binaries]           — default_set + named binary sets
        [defaults]           — budget_auto_ratio, priority, chunks_*, ctx, no_mmap, ...
        [huggingface]        — default_revision, download_resume, ...
        [wizard]             — setup_complete, auto_suggest_perf, disk_warn_pct, ...

    Example file ships at examples/config.toml in the repo. Hand-editable.
    Empty / missing file = "use all built-in defaults" (backward compatible).
    """
    global _user_config_cache
    if _user_config_cache is not None:
        return _user_config_cache
    if not DEFAULT_USER_CONFIG_PATH.exists():
        _user_config_cache = {}
        return _user_config_cache
    try:
        import tomllib
        with DEFAULT_USER_CONFIG_PATH.open("rb") as f:
            _user_config_cache = tomllib.load(f)
    except Exception as e:
        # Don't crash the pipeline on a malformed user config; warn + ignore.
        import sys as _sys
        print(f"[paths] WARN: failed to parse {DEFAULT_USER_CONFIG_PATH}: {e}",
              file=_sys.stderr)
        _user_config_cache = {}
    return _user_config_cache


def get_user_config_value(section: str, key: str, default=None):
    """Look up `[section] key` from the user config, returning `default`
    if missing. Convenience wrapper around `load_user_config()`."""
    cfg = load_user_config()
    return cfg.get(section, {}).get(key, default)


def get_user_default_path(role: str, fallback: Path) -> Path:
    """Return user's [paths] entry for `role` if set, expanding ~ and
    relative paths; otherwise return `fallback`. Roles: output_root,
    hf_cache, binary_cache, scratch, calibration."""
    val = get_user_config_value("paths", role)
    if val is None:
        return fallback
    return Path(str(val)).expanduser()


def get_user_default_binary_set() -> "Optional[dict]":
    """Return the user's default binary set (`[binaries.<default_set>]`)
    as a dict of tool→path, or None if not configured."""
    cfg = load_user_config()
    binaries = cfg.get("binaries", {})
    default_name = binaries.get("default_set")
    if not default_name:
        return None
    return binaries.get(default_name)


def load_user_default_formats() -> "Optional[list[str]]":
    """Load user's default formats list from ~/.prismaquant-llama/config/default-formats.txt.

    Returns a list of format names (e.g., ["Q4_K", "Q5_K", ...]) or None
    if the file doesn't exist or has no usable entries. Lines starting
    with `#` are comments; blank lines are ignored.

    This sits between the CLI `--formats` flag (highest precedence) and
    the hardcoded PipelineConfig default (lowest). When set, runs that
    omit `--formats` will use this list instead of the built-in default.
    """
    if not DEFAULT_USER_FORMATS_PATH.exists():
        return None
    formats: list[str] = []
    try:
        for line in DEFAULT_USER_FORMATS_PATH.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            # Tolerate "Q4_K, Q5_K" or "Q4_K Q5_K" on a single line too.
            for fmt in line.replace(",", " ").split():
                fmt = fmt.strip()
                if fmt:
                    formats.append(fmt)
    except OSError:
        return None
    return formats or None

# Standard GGUF llama.cpp tools we look for via discovery.
TOOL_NAMES = {
    "quantize":   "llama-quantize",
    "perplexity": "llama-perplexity",
    "bench":      "llama-bench",
    "imatrix":    "llama-imatrix",
    "cli":        "llama-cli",
}


# ─────────────────────────────────────────────────────────────────────────────
# Binary discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_binary(tool: str = "quantize",
                user_path: Optional[Path] = None,
                hint_dirs: Optional[list[Path]] = None,
                must_exist: bool = True) -> Optional[Path]:
    """
    3-tier search for a llama.cpp tool binary.

    Order:
      1. Explicit user_path (if given and exists).
      2. $PATH lookup via shutil.which.
      3. Common build directories: cwd, parents, $PWD/build/bin,
         $PWD/../build/bin. Walks up to 3 parents.
      4. hint_dirs (if given) — for cases where caller knows where to look.

    Returns first hit; None if not found and must_exist=False; raises
    if must_exist=True and no hit.
    """
    name = TOOL_NAMES.get(tool, tool)

    # 1. Explicit path
    if user_path:
        user_path = Path(user_path).resolve()
        if user_path.exists():
            return user_path
        if must_exist:
            raise FileNotFoundError(f"{name} not found at {user_path}")
        return None

    # 2. PATH
    p = shutil.which(name)
    if p:
        return Path(p).resolve()

    # 3. Common build dirs (parent-walk up to 3 levels)
    candidates = []
    cwd = Path.cwd().resolve()
    for level in range(4):
        base = cwd
        for _ in range(level):
            base = base.parent
        candidates.extend([
            base / "build" / "bin" / name,
            base / "build-1150" / "bin" / name,
            base / "build-1102" / "bin" / name,
            base / "bin" / name,
        ])
    for c in candidates:
        if c.exists():
            return c.resolve()

    # 4. Hint dirs
    for d in (hint_dirs or []):
        c = d / name
        if c.exists():
            return c.resolve()

    if must_exist:
        raise FileNotFoundError(
            f"{name} not found. Pass --binary, add to $PATH, or run from "
            f"a build directory (looked in cwd + parents up to 3 levels)."
        )
    return None


def discover_companion_binaries(quantize_path: Path) -> dict[str, Optional[Path]]:
    """Given a quantize binary, find sibling tools (perplexity / bench /
    imatrix / cli) in the same directory. Returns dict keyed by tool name."""
    bin_dir = quantize_path.parent
    out = {"quantize": quantize_path}
    for tool_key, tool_name in TOOL_NAMES.items():
        if tool_key == "quantize":
            continue
        candidate = bin_dir / tool_name
        out[tool_key] = candidate.resolve() if candidate.exists() else None
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Output directory tree
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WorkPaths:
    """All paths for a single wizard invocation. Construct via for_run()."""
    root: Path
    shared: Path
    calibration_dir: Path
    hf_cache: Path
    imatrix_cache: Path
    ggufs: Path
    work: Path
    bf16_dir: Path
    probe_dir: Path
    imatrix_dir: Path
    costs_dir: Path
    recipes_dir: Path
    logs_dir: Path
    run_label: str = ""

    @classmethod
    def for_run(cls,
                root: Optional[Path] = None,
                model_name: str = "model",
                run_timestamp: Optional[str] = None,
                hf_cache_override: Optional[Path] = None,
                imatrix_cache_override: Optional[Path] = None,
                ggufs_override: Optional[Path] = None) -> "WorkPaths":
        root = (root or DEFAULT_OUTPUT_ROOT).expanduser().resolve()
        ts = run_timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
        # Sanitize model name → drop org prefix + source-format suffix.
        safe_model = sanitize_model_name(model_name)
        run_label = f"{safe_model}-{ts}"

        shared = root / "_shared"
        ggufs = (ggufs_override or (root / "ggufs")).expanduser().resolve()
        work = root / "work" / run_label

        return cls(
            root=root,
            shared=shared,
            calibration_dir=shared / "calibration",
            hf_cache=(hf_cache_override or (shared / "hf-cache")).expanduser().resolve(),
            imatrix_cache=(imatrix_cache_override or (shared / "imatrix-cache")).expanduser().resolve(),
            ggufs=ggufs,
            work=work,
            bf16_dir=shared / "bf16",
            probe_dir=shared / "probe",
            imatrix_dir=work / "imatrix",
            costs_dir=work / "costs",
            recipes_dir=work / "recipes",
            logs_dir=work / "logs",
            run_label=run_label,
        )

    def make(self) -> None:
        """Create every directory. Idempotent — existing dirs are left alone."""
        for d in [self.root, self.shared, self.calibration_dir,
                  self.hf_cache, self.imatrix_cache, self.ggufs,
                  self.work, self.bf16_dir, self.probe_dir,
                  self.imatrix_dir, self.costs_dir, self.recipes_dir,
                  self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def cleanup_work(self) -> None:
        """Remove the per-run work/<label>/ subtree. Idempotent."""
        if self.work.exists():
            shutil.rmtree(self.work)

    def cleanup_shared(self, model_key: str, *,
                       hf_cache: bool = True,
                       bf16: bool = True,
                       imatrix: bool = False,
                       probe: bool = False) -> dict:
        """Delete shared cache artifacts for one model. Returns {category: bytes_freed}.

        Default targets the heavy intermediates: HF safetensors snapshot in
        `_shared/hf-cache/<model_key>/` and the converted BF16 GGUF at
        `_shared/bf16/<model_key>-BF16.gguf`. Smaller caches (imatrix, probe)
        are kept by default since they're cheap to retain and useful for
        re-runs / cross-format experiments — opt in with imatrix=True / probe=True.

        Idempotent: silently skips paths that don't exist.
        """
        freed: dict = {}

        def _size(p: Path) -> int:
            if p.is_file():
                return p.stat().st_size
            if p.is_dir():
                return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            return 0

        if hf_cache:
            d = self.hf_cache / model_key
            if d.exists():
                freed["hf_cache"] = _size(d)
                shutil.rmtree(d)
        if bf16:
            f = self.bf16_dir / f"{model_key}-BF16.gguf"
            if f.exists():
                freed["bf16"] = f.stat().st_size
                f.unlink()
        if imatrix:
            f = self.imatrix_cache / f"{model_key}-BF16.imatrix.gguf"
            if f.exists():
                freed["imatrix"] = f.stat().st_size
                f.unlink()
        if probe:
            for p in self.probe_dir.glob(f"{model_key}-*"):
                try:
                    if p.is_file():
                        freed["probe"] = freed.get("probe", 0) + p.stat().st_size
                        p.unlink()
                    elif p.is_dir():
                        freed["probe"] = freed.get("probe", 0) + _size(p)
                        shutil.rmtree(p)
                except OSError:
                    pass
        return freed

    def imatrix_cache_path(self, model_sha: str, corpus_sha: str, chunks: int) -> Path:
        """Deterministic path to cached imatrix file. Returns the file path
        whether or not the file currently exists — caller checks exists()."""
        # Use 12-char hash prefixes (enough to avoid collisions across the
        # tens-of-models scale this tool targets; keeps filenames human-skimmable)
        key = f"{model_sha[:12]}__{corpus_sha[:12]}__c{chunks}.imatrix.gguf"
        return self.imatrix_cache / key

    def link_imatrix(self, cached_imatrix: Path) -> Path:
        """Symlink the cached imatrix into the working dir for this run.
        Returns the symlink path (in self.imatrix_dir)."""
        symlink_path = self.imatrix_dir / cached_imatrix.name
        if symlink_path.is_symlink() or symlink_path.exists():
            symlink_path.unlink()
        symlink_path.symlink_to(cached_imatrix)
        return symlink_path

    def gguf_output_path(self, base_name: str, budget_gb: float, priority: str) -> Path:
        """PQ<budget>-<XYZ> filename in ggufs/."""
        fname = f"{base_name}-PQ{budget_gb}-{priority}.gguf"
        return self.ggufs / fname

    def summary_lines(self) -> list[str]:
        """Human-readable layout summary for stdout."""
        return [
            f"  output root:      {self.root}",
            f"  shared cache:     {self.shared}",
            f"  ├─ calibration:   {self.calibration_dir}",
            f"  ├─ hf-cache:      {self.hf_cache}",
            f"  └─ imatrix-cache: {self.imatrix_cache}",
            f"  ggufs (output):   {self.ggufs}",
            f"  work scratch:     {self.work}",
        ]


# ─────────────────────────────────────────────────────────────────────────────
# argparse helpers (so wizard.py can defer to this module for path flags)
# ─────────────────────────────────────────────────────────────────────────────

def add_path_args(parser) -> None:
    """Register the standard path/binary flags on an argparse.ArgumentParser.
    The wizard's main() calls this before parse_args."""
    g = parser.add_argument_group("paths and binaries")
    g.add_argument("--binary", type=Path, default=None,
                   help=f"path to {TOOL_NAMES['quantize']} (default: $PATH search, "
                        f"then build/bin/, then prompt)")
    g.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT_ROOT,
                   help=f"output root directory (default: {DEFAULT_OUTPUT_ROOT})")
    g.add_argument("--keep-work", action="store_true",
                   help="don't auto-clean work/<label>/ after build "
                        "(keeps probe + costs + intermediate BF16)")
    g.add_argument("--output-ggufs", type=Path, default=None,
                   help="override just the ggufs/ subdir (e.g., put final "
                        "GGUFs in /mnt/cephfs/models/ instead of <output>/ggufs/)")
    g.add_argument("--hf-cache", type=Path, default=None,
                   help="override HF safetensors cache dir (default: <output>/_shared/hf-cache)")
    g.add_argument("--imatrix-cache", type=Path, default=None,
                   help="override imatrix cache dir (default: <output>/_shared/imatrix-cache)")


def resolve_paths_from_args(args, model_name: str = "model") -> WorkPaths:
    """Build a WorkPaths from parsed argparse args. Wizard calls this once
    after the user picks a model in screen 1 (so model_name is known)."""
    return WorkPaths.for_run(
        root=args.output,
        model_name=model_name,
        hf_cache_override=getattr(args, "hf_cache", None),
        imatrix_cache_override=getattr(args, "imatrix_cache", None),
        ggufs_override=getattr(args, "output_ggufs", None),
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI for ad-hoc inspection
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Inspect path layout / discover binaries")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_layout = sub.add_parser("layout", help="show what dirs would be created")
    add_path_args(p_layout)
    p_layout.add_argument("--model-name", default="example-model")
    p_layout.add_argument("--make", action="store_true", help="actually create the dirs")

    p_find = sub.add_parser("find-binaries", help="locate llama.cpp tools")
    p_find.add_argument("--binary", type=Path, default=None,
                        help="explicit llama-quantize path")

    args = p.parse_args(argv)

    if args.cmd == "layout":
        wp = resolve_paths_from_args(args, model_name=args.model_name)
        print("Path layout:")
        for line in wp.summary_lines():
            print(line)
        print(f"  per-run scratch tree:")
        for d in [wp.bf16_dir, wp.probe_dir, wp.imatrix_dir,
                  wp.costs_dir, wp.recipes_dir, wp.logs_dir]:
            print(f"    {d.relative_to(wp.work)}/")
        if args.make:
            wp.make()
            print(f"\n  ✓ all dirs created under {wp.root}")
    elif args.cmd == "find-binaries":
        try:
            qb = find_binary("quantize", user_path=args.binary)
            print(f"  llama-quantize: {qb}")
            companions = discover_companion_binaries(qb)
            for tool, path in companions.items():
                marker = "✓" if path else "✗"
                print(f"  {marker} {TOOL_NAMES[tool]:<18} {path or '(not found)'}")
        except FileNotFoundError as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
