"""
Empirical calibration of weight quants on a user's binary + hardware.

Three modes, ordered by cost:

  1. calibrate_quick(binary, ref_model, formats)
     Just quantizes ref_model with each format, measures output size,
     derives bpw. ~30 sec/format = ~25 min for 53 formats. Result
     supersedes heuristic-derived bpw permanently.

  2. calibrate_deep(binary, ref_model, calibration_corpus, formats)
     Adds llama-perplexity (PPL Δ vs f16) + llama-bench (PP/TG tps) per
     format. ~5-15 min/format = 6-12 hours for full sweep. Generates an
     empirical "this format costs X PPL and runs at Y tps on my hardware"
     table.

  3. ingest_prismaquant_cost_csv(cost_csv_path)
     Slurps per-(tensor, format) MSE data from a prismaquant Stage D
     output. Free, automatic — every pipeline run incrementally fills
     out the metadata cache.

All three modes write/update a JSON file at:
    ~/.cache/prismaquant-wizard/binary-types/<binary-sha256>-calibrated.json

Resume-safe: if the output file already has a format's data, that
format is skipped on rerun. Disk-safe: each format's quantized output is
deleted immediately after measurement (peak disk = 1× output, not Nx).

Status: SCAFFOLD. Subprocess invocations + parser logic implemented; not
yet tested end-to-end against a live binary because that consumes the
GPU which is currently running the comparison sweep on ai00.
"""

from __future__ import annotations
import csv
import hashlib
import json
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Iterable


# ─────────────────────────────────────────────────────────────────────────────
# Cache layout
# ─────────────────────────────────────────────────────────────────────────────

CACHE_ROOT = Path.home() / ".cache" / "prismaquant-wizard" / "binary-types"


def _binary_sha256(binary: Path) -> str:
    h = hashlib.sha256()
    with binary.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def calibration_cache_path(binary: Path) -> Path:
    """Full path to the calibrated-metadata JSON for this binary."""
    bsha = _binary_sha256(binary)
    return CACHE_ROOT / f"{bsha[:16]}-calibrated.json"


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CalibrationHeader:
    binary_path: str = ""
    binary_sha256: str = ""
    ref_model: str = ""
    ref_model_sha256: str = ""
    ref_model_params: int = 0
    calibration_corpus: str = ""
    wikitext_chunks: int = 0
    machine_id: str = ""
    calibrated_at: str = ""


@dataclass
class FormatMeasurement:
    """Per-format calibration result. All fields optional — different
    modes populate different subsets.

    The format-perf subset emitted via export_format_perf_subset() carries
    a stable schema documented below. The CalibrationFile (this class) is
    the richer superset.

    format-perf schema (consumed by allocator's --tps flag):
        Required: pp (≡ pp512_tps), tg (≡ tg128_tps)
        Reserved future keys (not yet emitted; reader should ignore unknown):
          - lat_p99_ms       — 99th-percentile decode latency (ms/token)
          - mem_gb_per_param — runtime memory footprint per param
          - ppl_delta_vs_f16 — quality penalty signal (could let allocator
                               hard-skip a format whose Δ exceeds a bound)
          - compatible       — explicit per-format compatibility flag
                               (e.g. False if architecture's tensor shape
                               can't be quantized to this format at all)
    """
    bpw: Optional[float] = None
    size_bytes: Optional[int] = None
    quantize_wallclock_sec: Optional[float] = None
    ppl: Optional[float] = None
    ppl_stderr: Optional[float] = None
    ppl_delta_vs_f16: Optional[float] = None
    pp512_tps: Optional[float] = None
    tg128_tps: Optional[float] = None
    bench_wallclock_sec: Optional[float] = None
    # Reserved for future schema extensions; not yet measured.
    lat_p99_ms: Optional[float] = None
    mem_gb_per_param: Optional[float] = None
    compatible: Optional[bool] = None
    error: Optional[str] = None


@dataclass
class CalibrationFile:
    header: CalibrationHeader = field(default_factory=CalibrationHeader)
    formats: dict[str, FormatMeasurement] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "_header": asdict(self.header),
            **{name: asdict(m) for name, m in self.formats.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CalibrationFile":
        header = CalibrationHeader(**d.get("_header", {}))
        formats = {
            name: FormatMeasurement(**v)
            for name, v in d.items()
            if not name.startswith("_")
        }
        return cls(header=header, formats=formats)


def load_calibration_file(path: Path) -> CalibrationFile:
    if not path.exists():
        return CalibrationFile()
    return CalibrationFile.from_dict(json.loads(path.read_text()))


def export_format_perf_subset(calib: CalibrationFile, output_path: Path,
                              absolute_only: bool = False,
                              ratios_only: bool = False) -> int:
    """Extract format-perf subset from a CalibrationFile.

    Output shape (consumed by the allocator's --tps flag and any future
    consumers of per-format quality signals):

        {
          "_schema_version": 3,
          "_reference_format": "BF16",
          "Q4_K": {
            "pp": 336.60,                    # PP512 tps (absolute, this binary+GPU)
            "tg": 18.74,                     # TG128 tps (absolute)
            "pp_ratio_vs_bf16": 0.91,        # hardware-portable: pp / pp(BF16)
            "tg_ratio_vs_bf16": 2.62,        # hardware-portable: tg / tg(BF16)
            "ppl": 6.92,
            "ppl_delta_vs_f16": 0.094,
            "bpw": 4.5106
          },
          ...
        }

    Schema versions:
      - v1: pp + tg only
      - v2: adds ppl, ppl_delta_vs_f16, bpw, size_bytes
      - v3 (this): adds pp_ratio_vs_bf16, tg_ratio_vs_bf16 alongside abs.

    Allocator reader prefers absolute values when present, falls back to
    ratios. Population-mean normalization in the cost function makes the
    two functionally equivalent (only relative magnitudes matter).

    Modes:
      - default: emit BOTH absolute + ratio (per-binary cache)
      - absolute_only=True: only abs (smaller schema, legacy)
      - ratios_only=True: emit ratio columns; abs set to None
        (used for hardware-agnostic shipped defaults — abs values from
        one machine are misleading on another)
    """
    out = {
        "_comment": "Per-format perf characteristics from prismaquant-llama "
                    "calibrate-deep. Schema v3: absolute pp/tg are this-binary-"
                    "specific; pp_ratio_vs_bf16 / tg_ratio_vs_bf16 transfer "
                    "across hardware (within ~20%). Allocator uses abs if "
                    "available, ratios otherwise.",
        "_reference_model": (calib.header.ref_model or "(unknown)"),
        "_reference_model_sha256": calib.header.ref_model_sha256,
        "_reference_model_params": calib.header.ref_model_params,
        "_binary_sha256": calib.header.binary_sha256,
        "_machine_id": calib.header.machine_id,
        "_calibrated_at": calib.header.calibrated_at,
        "_schema_version": 3,
        "_reference_format": "BF16",
    }

    # Find BF16 (preferred) or F16 baseline for ratio computation
    bf16_pp = bf16_tg = None
    for ref_name in ("BF16", "F16"):
        m = calib.formats.get(ref_name)
        if m and m.pp512_tps is not None and m.tg128_tps is not None:
            bf16_pp, bf16_tg = m.pp512_tps, m.tg128_tps
            out["_reference_format"] = ref_name
            break

    n_emitted = 0
    for fmt, m in calib.formats.items():
        if m.pp512_tps is None or m.tg128_tps is None:
            continue
        entry = {}
        # Absolute values (per-binary; null in ratios-only mode)
        if ratios_only:
            entry["pp"] = None
            entry["tg"] = None
        elif not absolute_only:
            entry["pp"] = round(m.pp512_tps, 2)
            entry["tg"] = round(m.tg128_tps, 2)
        else:
            entry["pp"] = round(m.pp512_tps, 2)
            entry["tg"] = round(m.tg128_tps, 2)
        # Ratios (hardware-portable; absent in absolute_only mode)
        if not absolute_only and bf16_pp and bf16_tg:
            entry["pp_ratio_vs_bf16"] = round(m.pp512_tps / bf16_pp, 4)
            entry["tg_ratio_vs_bf16"] = round(m.tg128_tps / bf16_tg, 4)
        # Quality + size (model-specific but useful)
        if m.ppl is not None:
            entry["ppl"] = round(m.ppl, 4)
        if m.ppl_delta_vs_f16 is not None:
            entry["ppl_delta_vs_f16"] = round(m.ppl_delta_vs_f16, 4)
        if m.bpw is not None:
            entry["bpw"] = round(m.bpw, 4)
        if m.size_bytes is not None:
            entry["size_bytes"] = int(m.size_bytes)
        out[fmt] = entry
        n_emitted += 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2))
    return n_emitted


SYSTEM_DEFAULT_PERF_PATH = (
    Path.home() / ".config" / "prismaquant-llama" / "system-default-format-perf.json"
)


def set_system_default_perf(source_path: Path) -> Path:
    """Copy a format-perf file into the system-default slot at
    ~/.config/prismaquant-llama/system-default-format-perf.json. Future
    pipeline runs auto-discover this as a tier-3 fallback (after the
    per-binary cache, before the package-shipped examples).

    Returns the destination path."""
    import shutil
    SYSTEM_DEFAULT_PERF_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, SYSTEM_DEFAULT_PERF_PATH)
    return SYSTEM_DEFAULT_PERF_PATH


def find_format_perf_file_for_binary(binary: Path,
                              examples_dir: Optional[Path] = None,
                              default_format_perf_override: Optional[Path] = None,
                              ) -> Optional[Path]:
    """Resolve the best-available format-perf file for this binary.

    Priority (highest → lowest):
        1. `~/.cache/.../binary-types/<sha-prefix>-perf.json` (or legacy
           `-tps.json`) — auto-generated from `calibrate deep` on this binary.
           Most specific; always wins when present.
        2. `default_format_perf_override` — user-supplied via
           `--default-format-perf` flag or PRISMAQUANT_DEFAULT_FORMAT_PERF env
           var. Per-run preference.
        3. `~/.config/prismaquant-llama/system-default-format-perf.json` — the
           user's "system default", written by `calibrate deep --set-as-system-default`.
           Persists across binary rebuilds; serves as cross-binary fallback.
        4. `<pkg>/examples/format-perf-default.json` — package-shipped
           hardware-agnostic baseline. Format-relative throughput ratios
           transfer roughly across GPUs (within ~20%) so this gives sensible
           defaults for any user. Run `calibrate deep --set-as-system-default`
           to get measured values for your specific hardware.

    Returns None if no candidate file exists.
    """
    import os
    sha = _binary_sha256(binary)
    sha_short = sha[:16]
    # Cache dirs: new (`prismaquant-llama`) + legacy (`prismaquant-wizard`).
    # Filenames: new (`<sha>-perf.json`) + legacy (`<sha>-tps.json`).
    # Each may use full sha or 16-char prefix. Search the cross-product.
    cache_dirs = [
        Path.home() / ".cache" / "prismaquant-llama" / "binary-types",
        Path.home() / ".cache" / "prismaquant-wizard" / "binary-types",
    ]
    for cd in cache_dirs:
        for name in (f"{sha_short}-perf.json", f"{sha}-perf.json",
                     f"{sha_short}-tps.json", f"{sha}-tps.json"):
            cand = cd / name
            if cand.exists():
                return cand
    # User local default (CLI flag or env var)
    if default_format_perf_override is None:
        env_override = os.environ.get("PRISMAQUANT_DEFAULT_FORMAT_PERF")
        if env_override:
            default_format_perf_override = Path(env_override)
    if default_format_perf_override is not None and default_format_perf_override.exists():
        return default_format_perf_override
    # System default (user wrote via `calibrate deep --set-as-system-default`)
    if SYSTEM_DEFAULT_PERF_PATH.exists():
        return SYSTEM_DEFAULT_PERF_PATH
    # Package-shipped baseline. The default file is intentionally hardware-
    # agnostic — format-relative throughput ratios transfer roughly across
    # GPUs (within ~20%) so a single calibrated reference serves as a sane
    # starting point for any user. For accurate values on your specific
    # hardware, run `prismaquant-llama calibrate deep --set-as-system-default`.
    # Fallback chain (new naming preferred; legacy arch-suffixed names kept
    # for backward compat with files emitted by earlier versions):
    if examples_dir is not None:
        for fname in ("format-perf-default.json",
                      "format-perf.json",
                      # legacy arch-suffixed (deprecated; matches by hostname)
                      *(f"format-perf-{a}.json"
                        for a in ("gfx1150", "gfx1102")),
                      *(f"format-tps-{a}.json"
                        for a in ("gfx1150", "gfx1102"))):
            cand = examples_dir / fname
            if cand.exists():
                return cand
    return None


def save_calibration_file(path: Path, calib: CalibrationFile) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calib.to_dict(), indent=2, default=str))


# ─────────────────────────────────────────────────────────────────────────────
# Param counting (read GGUF header)
# ─────────────────────────────────────────────────────────────────────────────

def count_gguf_params(gguf_path: Path) -> int:
    """Read the GGUF header to count total parameters across all tensors.
    Type-agnostic — works with any quant type since parameter count
    depends only on tensor shapes."""
    import struct
    total = 0
    with gguf_path.open("rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"not a GGUF file: {gguf_path}")
        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        # Skip metadata KV pairs
        def skip_value(vt: int, f) -> None:
            if vt in (0, 1, 7):  f.read(1)
            elif vt in (2, 3):   f.read(2)
            elif vt in (4, 5, 6):f.read(4)
            elif vt in (10, 11, 12): f.read(8)
            elif vt == 8:
                slen = struct.unpack("<Q", f.read(8))[0]
                f.read(slen)
            elif vt == 9:
                etype = struct.unpack("<I", f.read(4))[0]
                n = struct.unpack("<Q", f.read(8))[0]
                for _ in range(n):
                    skip_value(etype, f)

        for _ in range(n_kv):
            klen = struct.unpack("<Q", f.read(8))[0]; f.read(klen)
            skip_value(struct.unpack("<I", f.read(4))[0], f)

        for _ in range(n_tensors):
            nlen = struct.unpack("<Q", f.read(8))[0]; f.read(nlen)
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = struct.unpack("<" + "Q" * n_dims, f.read(8 * n_dims))
            f.read(4)   # type
            f.read(8)   # offset
            n_elem = 1
            for d in dims:
                n_elem *= d
            total += n_elem
    return total


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess wrappers
# ─────────────────────────────────────────────────────────────────────────────

def _run_quantize(binary: Path, src: Path, dst: Path, fmt: str,
                  nthreads: Optional[int] = None, timeout: float = 600) -> tuple[bool, str]:
    """Run llama-quantize. Returns (ok, log)."""
    cmd = [str(binary), str(src), str(dst), fmt]
    if nthreads:
        cmd.append(str(nthreads))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s"


def _run_perplexity(perp_binary: Path, model: Path, calibration: Path,
                    chunks: int = 4, c: int = 2048, env: Optional[dict] = None,
                    timeout: float = 1800) -> tuple[Optional[float], Optional[float], str]:
    """Run llama-perplexity. Returns (ppl, stderr, log)."""
    cmd = [str(perp_binary), "-m", str(model), "-f", str(calibration),
           "-c", str(c), "-ngl", "99", "-fa", "on",
           "--chunks", str(chunks), "--no-mmap"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, env=env)
        log = proc.stdout + "\n" + proc.stderr
    except subprocess.TimeoutExpired:
        return None, None, "timeout"
    m = re.search(r"Final estimate:\s*PPL\s*=\s*([\d.]+)\s*\+/-\s*([\d.]+)", log)
    if not m:
        return None, None, log
    return float(m.group(1)), float(m.group(2)), log


def _run_bench(bench_binary: Path, model: Path, p: int = 512, n: int = 128,
               threads: int = 12, timeout: float = 600) -> tuple[Optional[float], Optional[float], str]:
    """Run llama-bench. Returns (pp_tps, tg_tps, log)."""
    cmd = [str(bench_binary), "-m", str(model),
           "-p", str(p), "-n", str(n),
           "-t", str(threads), "-ngl", "99", "-fa", "1",
           "--output", "csv"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        log = proc.stdout + "\n" + proc.stderr
    except subprocess.TimeoutExpired:
        return None, None, "timeout"
    pp = tg = None
    for line in log.splitlines():
        if f'"{p}","0","0"' in line:
            parts = [p.strip('"') for p in line.split(",")]
            pp = float(parts[-2]) if len(parts) >= 2 else None
        elif f'"0","{n}","0"' in line:
            parts = [p.strip('"') for p in line.split(",")]
            tg = float(parts[-2]) if len(parts) >= 2 else None
    return pp, tg, log


# ─────────────────────────────────────────────────────────────────────────────
# Mode 1: quick (size only)
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_quick(binary: Path, ref_model: Path, formats: Iterable[str],
                    output_path: Optional[Path] = None,
                    machine_id: str = "",
                    log = print) -> CalibrationFile:
    """
    Mode 1 — Size-only calibration. Quantizes ref_model with each format,
    derives bpw from output size. ~30 sec/format. Resume-safe via output_path.

    Args:
        binary: path to llama-quantize
        ref_model: input GGUF (e.g., Llama-3.2-1B-BF16)
        formats: iterable of format names (e.g., ["Q4_K_M", "IQ4_K"])
        output_path: where to persist results (default: ~/.cache/.../<sha>.json)
        machine_id: optional human-readable machine tag for the header
        log: print function (default print; pass logger.info for proper logging)

    Returns:
        CalibrationFile with bpw + size_bytes populated per format.
    """
    binary = binary.resolve()
    ref_model = ref_model.resolve()
    output_path = output_path or calibration_cache_path(binary)

    calib = load_calibration_file(output_path)
    n_params = count_gguf_params(ref_model)

    # Update header (don't overwrite if existing entries are still valid)
    if not calib.header.binary_sha256:
        calib.header.binary_path = str(binary)
        calib.header.binary_sha256 = _binary_sha256(binary)
    calib.header.ref_model = ref_model.name
    if not calib.header.ref_model_sha256:
        calib.header.ref_model_sha256 = file_sha256(ref_model)
    calib.header.ref_model_params = n_params
    calib.header.machine_id = machine_id or calib.header.machine_id
    calib.header.calibrated_at = datetime.now(timezone.utc).isoformat()

    formats = list(formats)
    log(f"[calibrate-quick] {len(formats)} formats × ~30s each = ~{len(formats)//2} min")
    log(f"[calibrate-quick] cache: {output_path}")

    with tempfile.TemporaryDirectory(prefix="prismaquant-wizard-cal-") as tmp:
        tmpdir = Path(tmp)
        for i, fmt in enumerate(formats, 1):
            existing = calib.formats.get(fmt)
            if existing and existing.bpw is not None and existing.error is None:
                log(f"[{i:>3}/{len(formats)}] {fmt:<14} cached (bpw={existing.bpw:.4f})")
                continue
            out = tmpdir / f"ref-{fmt}.gguf"
            t0 = time.time()
            ok, _ = _run_quantize(binary, ref_model, out, fmt)
            dt = time.time() - t0
            if not ok or not out.exists():
                m = FormatMeasurement(error="quantize failed",
                                      quantize_wallclock_sec=round(dt, 1))
                calib.formats[fmt] = m
                log(f"[{i:>3}/{len(formats)}] {fmt:<14} FAIL ({dt:.1f}s)")
            else:
                size = out.stat().st_size
                bpw = round(size * 8 / n_params, 4)
                m = FormatMeasurement(
                    bpw=bpw, size_bytes=size,
                    quantize_wallclock_sec=round(dt, 1),
                )
                calib.formats[fmt] = m
                log(f"[{i:>3}/{len(formats)}] {fmt:<14} bpw={bpw:.4f} "
                    f"size={size/1024**3:.2f}GB ({dt:.1f}s)")
            # Persist after each format so a kill mid-sweep preserves progress
            save_calibration_file(output_path, calib)
            # Delete the quantized output to keep peak disk ~= 1× output
            try:
                out.unlink(missing_ok=True)
            except OSError:
                pass

    return calib


# ─────────────────────────────────────────────────────────────────────────────
# Mode 2: deep (size + PPL + bench)
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_deep(binary: Path, ref_model: Path, calibration_corpus: Path,
                   formats: Iterable[str],
                   perp_binary: Optional[Path] = None,
                   bench_binary: Optional[Path] = None,
                   output_path: Optional[Path] = None,
                   chunks: int = 4, ctx: int = 2048,
                   machine_id: str = "",
                   env: Optional[dict] = None,
                   log = print) -> CalibrationFile:
    """
    Mode 2 — Full calibration with PPL + bench. ~5-15 min per format.
    Resume-safe. Re-derives bpw too (so this mode supersedes quick).

    perp_binary / bench_binary: paths to llama-perplexity and llama-bench.
    Default: same dir as `binary`.
    """
    binary = binary.resolve()
    ref_model = ref_model.resolve()
    output_path = output_path or calibration_cache_path(binary)
    perp_binary = perp_binary or (binary.parent / "llama-perplexity")
    bench_binary = bench_binary or (binary.parent / "llama-bench")

    if not perp_binary.exists():
        raise FileNotFoundError(f"llama-perplexity not found: {perp_binary}")
    if not bench_binary.exists():
        raise FileNotFoundError(f"llama-bench not found: {bench_binary}")
    if not calibration_corpus.exists():
        raise FileNotFoundError(f"calibration corpus not found: {calibration_corpus}")

    calib = load_calibration_file(output_path)
    n_params = count_gguf_params(ref_model)

    # Header bookkeeping (same as quick)
    if not calib.header.binary_sha256:
        calib.header.binary_path = str(binary)
        calib.header.binary_sha256 = _binary_sha256(binary)
    calib.header.ref_model = ref_model.name
    if not calib.header.ref_model_sha256:
        calib.header.ref_model_sha256 = file_sha256(ref_model)
    calib.header.ref_model_params = n_params
    calib.header.calibration_corpus = str(calibration_corpus)
    calib.header.wikitext_chunks = chunks
    calib.header.machine_id = machine_id or calib.header.machine_id
    calib.header.calibrated_at = datetime.now(timezone.utc).isoformat()

    formats = list(formats)
    log(f"[calibrate-deep] {len(formats)} formats × ~10 min each = "
        f"~{len(formats) * 10 // 60} h estimated")
    log(f"[calibrate-deep] cache: {output_path}")

    # Get/establish f16 reference PPL (needed for ppl_delta_vs_f16). If F16
    # (or BF16 as fallback) is in the format list, move it to the front so
    # subsequent formats can compute Δ as they're measured. If F16 isn't in
    # the list at all and we don't have its ppl cached, prepend it.
    f16_ppl = calib.formats.get("F16", FormatMeasurement()).ppl
    if f16_ppl is None:
        if "F16" in formats:
            formats = ["F16"] + [f for f in formats if f != "F16"]
        elif "BF16" in formats and calib.formats.get("BF16", FormatMeasurement()).ppl is None:
            # No F16 in list; promote BF16 if present (16-bit reference)
            formats = ["BF16"] + [f for f in formats if f != "BF16"]
        else:
            formats = ["F16"] + formats   # neither cached nor in list — prepend F16

    with tempfile.TemporaryDirectory(prefix="prismaquant-wizard-cal-") as tmp:
        tmpdir = Path(tmp)
        for i, fmt in enumerate(formats, 1):
            m = calib.formats.get(fmt, FormatMeasurement())
            if (m.ppl is not None and m.pp512_tps is not None
                    and m.bpw is not None and m.error is None):
                log(f"[{i:>3}/{len(formats)}] {fmt:<14} cached (full)")
                continue

            out = tmpdir / f"ref-{fmt}.gguf"
            log(f"[{i:>3}/{len(formats)}] {fmt:<14} quantize...")
            t0 = time.time()
            ok, qlog = _run_quantize(binary, ref_model, out, fmt)
            qdt = time.time() - t0
            if not ok or not out.exists():
                m.error = "quantize failed"
                m.quantize_wallclock_sec = round(qdt, 1)
                calib.formats[fmt] = m
                save_calibration_file(output_path, calib)
                continue

            size = out.stat().st_size
            m.size_bytes = size
            m.bpw = round(size * 8 / n_params, 4)
            m.quantize_wallclock_sec = round(qdt, 1)

            log(f"             {fmt:<14} ppl... (chunks={chunks}, c={ctx})")
            t0 = time.time()
            ppl, ppl_err, plog = _run_perplexity(perp_binary, out, calibration_corpus,
                                                 chunks=chunks, c=ctx, env=env)
            pdt = time.time() - t0
            if ppl is None:
                m.error = (m.error or "") + " ppl failed;"
            else:
                m.ppl = ppl
                m.ppl_stderr = ppl_err
                # Set f16_ppl from F16 first; fall back to BF16 if no F16.
                if fmt == "F16" or (fmt == "BF16" and f16_ppl is None):
                    f16_ppl = ppl
                    # Backfill deltas for any prior formats that lacked an
                    # f16 reference at measurement time.
                    for prior_fmt, prior_m in calib.formats.items():
                        if (prior_m.ppl is not None
                                and prior_m.ppl_delta_vs_f16 is None
                                and prior_fmt not in ("F16", "BF16")):
                            prior_m.ppl_delta_vs_f16 = round(prior_m.ppl - f16_ppl, 4)
                if f16_ppl is not None:
                    m.ppl_delta_vs_f16 = round(ppl - f16_ppl, 4)

            log(f"             {fmt:<14} bench...")
            t0 = time.time()
            pp, tg, blog = _run_bench(bench_binary, out)
            bdt = time.time() - t0
            m.pp512_tps = pp
            m.tg128_tps = tg
            m.bench_wallclock_sec = round(bdt, 1)

            calib.formats[fmt] = m
            save_calibration_file(output_path, calib)
            log(f"             {fmt:<14} → bpw={m.bpw:.4f} "
                f"ppl={m.ppl} Δ={m.ppl_delta_vs_f16} "
                f"pp={pp} tg={tg}  ({qdt+pdt+bdt:.0f}s total)")

            try:
                out.unlink(missing_ok=True)
            except OSError:
                pass

    # After all formats measured, emit the format-perf subset so pipeline_runner
    # auto-discovers it via find_format_perf_file_for_binary().
    perf_path = output_path.parent / f"{output_path.stem.replace('-calibrated','')}-perf.json"
    n = export_format_perf_subset(calib, perf_path)
    log(f"[calibrate-deep] format-perf subset → {perf_path} ({n} formats)")

    return calib


# ─────────────────────────────────────────────────────────────────────────────
# Mode 3: ingest from prismaquant pipeline output
# ─────────────────────────────────────────────────────────────────────────────

def ingest_prismaquant_cost_csv(cost_csv: Path,
                                output_path: Optional[Path] = None,
                                binary_for_cache_key: Optional[Path] = None,
                                log = print) -> CalibrationFile:
    """
    Mode 3 — Slurp per-(tensor, format) MSE data from prismaquant Stage D.

    The prismaquant pipeline already measures MSE per format during cost
    measurement. This mode reads that CSV and folds the per-format
    aggregates into the wizard's metadata cache. Free, automatic.

    cost_csv: path to a Stage D output CSV produced by llama-quantize-cost.
    binary_for_cache_key: which binary's cache to update. Required if
        output_path isn't given.
    """
    if output_path is None:
        if binary_for_cache_key is None:
            raise ValueError("supply output_path or binary_for_cache_key")
        output_path = calibration_cache_path(binary_for_cache_key)

    calib = load_calibration_file(output_path)

    # Aggregate MSE per format (mean across tensors that have data)
    by_format: dict[str, list[float]] = {}
    with cost_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            fmt = row.get("format") or row.get("type")
            if not fmt:
                continue
            try:
                mse = float(row.get("mse") or row.get("MSE"))
            except (TypeError, ValueError):
                continue
            by_format.setdefault(fmt.upper(), []).append(mse)

    n_updated = 0
    for fmt, mses in by_format.items():
        m = calib.formats.get(fmt, FormatMeasurement())
        # MSE is supplemental — store it in a field we extend later, or
        # surface as a derived "quality_proxy". For now, we just log.
        log(f"  ingest {fmt}: n_tensors={len(mses)} mean_MSE={sum(mses)/len(mses):.4g}")
        # Future: persist by_format[fmt] into FormatMeasurement.mse_per_tensor
        n_updated += 1

    save_calibration_file(output_path, calib)
    log(f"[ingest-prismaquant-cost] processed {n_updated} formats from {cost_csv}")
    return calib


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    import sys
    p = argparse.ArgumentParser(description="Calibrate weight quants on this binary + hardware")
    sub = p.add_subparsers(dest="cmd", required=True)

    # quick
    pq = sub.add_parser("quick", help="size-only calibration (~25 min)")
    pq.add_argument("--binary", type=Path, required=True)
    pq.add_argument("--ref-model", type=Path, required=True,
                    help="GGUF to quantize (e.g., Llama-3.2-1B-BF16.gguf)")
    pq.add_argument("--formats", required=True,
                    help="comma-separated format names, or 'all' (use discover)")
    pq.add_argument("--output", type=Path)
    pq.add_argument("--machine-id", default="")

    # deep
    pd = sub.add_parser("deep", help="size + PPL + bench (~6-12 hours)")
    pd.add_argument("--binary", type=Path, required=True)
    pd.add_argument("--ref-model", type=Path, required=True)
    pd.add_argument("--calibration-corpus", type=Path, required=True)
    pd.add_argument("--formats", required=True)
    pd.add_argument("--output", type=Path)
    pd.add_argument("--chunks", type=int, default=4)
    pd.add_argument("--ctx", type=int, default=2048)
    pd.add_argument("--machine-id", default="")
    pd.add_argument("--perp-binary", type=Path)
    pd.add_argument("--bench-binary", type=Path)
    pd.add_argument("--set-as-system-default", action="store_true",
                    help="After calibration completes, copy the resulting "
                         "format-perf JSON to ~/.config/prismaquant-llama/"
                         "system-default-format-perf.json. Future pipeline "
                         "runs auto-discover this as a cross-binary default "
                         "(tier 3 in find_format_perf_file_for_binary). "
                         "Use once with your preferred reference model "
                         "(e.g., a 9B dense like Qwopus3.5-9B-v3.5).")

    # set-default (manual)
    psd = sub.add_parser("set-default-perf",
                         help="Set an existing format-perf JSON as the system default")
    psd.add_argument("--source", type=Path, required=True,
                     help="path to existing format-perf JSON to install")

    # ingest
    pi = sub.add_parser("ingest", help="absorb prismaquant Stage D cost.csv")
    pi.add_argument("--cost-csv", type=Path, required=True)
    pi.add_argument("--binary", type=Path, required=True,
                    help="binary whose cache to update (for cache key)")

    args = p.parse_args(argv)

    if args.cmd in ("quick", "deep"):
        if args.formats == "all":
            from format_discovery import discover_formats
            fmts = list(discover_formats(args.binary, all_formats=True).keys())
        else:
            fmts = [f.strip() for f in args.formats.split(",") if f.strip()]
        if args.cmd == "quick":
            calibrate_quick(args.binary, args.ref_model, fmts,
                            output_path=args.output, machine_id=args.machine_id)
        else:
            calib = calibrate_deep(args.binary, args.ref_model, args.calibration_corpus,
                           fmts, perp_binary=args.perp_binary,
                           bench_binary=args.bench_binary,
                           output_path=args.output, chunks=args.chunks, ctx=args.ctx,
                           machine_id=args.machine_id)
            if getattr(args, "set_as_system_default", False):
                # Locate the perf file calibrate_deep just emitted
                cache_path = args.output or calibration_cache_path(args.binary)
                perf_path = cache_path.parent / f"{cache_path.stem.replace('-calibrated','')}-perf.json"
                if perf_path.exists():
                    dst = set_system_default_perf(perf_path)
                    print(f"[calibrate-deep] system default → {dst}")
                else:
                    print(f"[calibrate-deep] WARN: --set-as-system-default specified but "
                          f"no perf file at {perf_path}", file=sys.stderr)
    elif args.cmd == "ingest":
        ingest_prismaquant_cost_csv(args.cost_csv, binary_for_cache_key=args.binary)
    elif args.cmd == "set-default-perf":
        if not args.source.exists():
            print(f"ERROR: source perf file not found: {args.source}", file=sys.stderr)
            return 1
        dst = set_system_default_perf(args.source)
        print(f"system default → {dst}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
