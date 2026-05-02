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
    modes populate different subsets."""
    bpw: Optional[float] = None
    size_bytes: Optional[int] = None
    quantize_wallclock_sec: Optional[float] = None
    ppl: Optional[float] = None
    ppl_stderr: Optional[float] = None
    ppl_delta_vs_f16: Optional[float] = None
    pp512_tps: Optional[float] = None
    tg128_tps: Optional[float] = None
    bench_wallclock_sec: Optional[float] = None
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

    # Get/establish f16 reference PPL (needed for ppl_delta_vs_f16)
    f16_ppl = calib.formats.get("F16", FormatMeasurement()).ppl
    if f16_ppl is None and "F16" not in formats:
        formats = ["F16"] + formats   # measure F16 first

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
                if fmt == "F16":
                    f16_ppl = ppl
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
            calibrate_deep(args.binary, args.ref_model, args.calibration_corpus,
                           fmts, perp_binary=args.perp_binary,
                           bench_binary=args.bench_binary,
                           output_path=args.output, chunks=args.chunks, ctx=args.ctx,
                           machine_id=args.machine_id)
    elif args.cmd == "ingest":
        ingest_prismaquant_cost_csv(args.cost_csv, binary_for_cache_key=args.binary)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
