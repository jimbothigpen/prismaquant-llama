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
# Defaults
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_OUTPUT_ROOT = Path.home() / "prismaquant-builds"

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
