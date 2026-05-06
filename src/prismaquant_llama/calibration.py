"""
Calibration: measure per-format size, PPL, and pp/tg throughput.

Two modes only:

    prismaquant-llama calibrate system <input>
        Writes {base}/calibration/system.json. Used as the system-default
        perf file for any model that doesn't have a model-specific calibration.

    prismaquant-llama calibrate model <input>
        Writes {base}/calibration/models/<model_name>.json. Used specifically
        for that model in subsequent `run` invocations.

Input forms (all four accepted by both modes — calibrate doesn't run the
Bayesian probe so it doesn't need safetensors):

    1. HuggingFace id              "unsloth/Qwen3.6-35B-A3B"
    2. on-disk safetensors dir     "/path/to/safetensors"
    3. on-disk f16/bf16 GGUF       "/path/to/model-BF16.gguf"
    4. URL(s) to f16/bf16 GGUF     "https://...gguf"  (or split, comma-separated)

Pipeline:
    1. Resolve input → BF16 GGUF on disk (download/convert as needed)
    2. For each format in cfg.quants (plus BF16/F16 reference):
         - quantize BF16 GGUF → format-specific GGUF
         - run llama-perplexity → ppl, ppl_delta_vs_f16
         - run llama-bench → pp, tg
         - compute ratios vs BF16
         - delete the format-specific GGUF
    3. Write the perf JSON
"""

from __future__ import annotations
import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .config import (Config, load_config, find_tool, subprocess_env,
                     resolve_corpus)
from .input_resolver import ResolvedInput, resolve as resolve_input
from .paths import Layout
from .pipeline_runner import (download_hf, convert_to_bf16, download_gguf_url,
                              stage_d_imatrix, _resolve_imatrix_override,
                              cfg_from_args,
                              estimate_calibrate, confirm_or_abort)


# ─────────────────────────────────────────────────────────────────────────────
# Per-format measurement
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FormatMeasurement:
    bpw: Optional[float] = None
    size_bytes: Optional[int] = None
    ppl: Optional[float] = None
    ppl_stderr: Optional[float] = None
    ppl_delta_vs_f16: Optional[float] = None
    pp: Optional[float] = None
    tg: Optional[float] = None
    pp_ratio_vs_bf16: Optional[float] = None
    tg_ratio_vs_bf16: Optional[float] = None
    error: Optional[str] = None


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(log_path: Path, msg: str) -> None:
    """Print to stdout + append to a rolling log file. Used for meta-progress
    messages so a single `tail -f calibrate.log` shows the whole calibration."""
    line = f"[{_ts()}] {msg}"
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(line + "\n")


def _run_cmd(cmd: list[str], env: dict, log_path: Path,
             timeout: float = 1800) -> tuple[int, str]:
    """Run subprocess with live tee to log_path (line-buffered) AND return the
    full captured output for parsing PPL/bench numbers downstream.

    Each invocation appends to log_path with a `=== <ts>: <cmd> ===` header,
    so callers can `tail -f` the same file across multiple invocations.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  $ {' '.join(str(c) for c in cmd)}", flush=True)
    captured: list[str] = []
    with log_path.open("a") as f:
        f.write(f"\n=== {_ts()}: {' '.join(str(c) for c in cmd)} ===\n")
        f.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                env=env, text=True, bufsize=1)
        try:
            for line in proc.stdout:  # type: ignore
                f.write(line)
                f.flush()
                captured.append(line)
        except KeyboardInterrupt:
            proc.terminate()
            raise
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            return 124, "".join(captured) + f"\ntimeout after {timeout}s\n"
    return proc.returncode, "".join(captured)


def _quantize_one(cfg: Config, src: Path, dst: Path, fmt: str,
                  log_path: Path,
                  imatrix: Optional[Path] = None) -> bool:
    bin_ = find_tool(cfg, "llama-quantize")
    cmd = [str(bin_)]
    if imatrix is not None:
        cmd += ["--imatrix", str(imatrix)]
    cmd += [str(src), str(dst), fmt]
    rc, _ = _run_cmd(cmd, subprocess_env(cfg), log_path)
    return rc == 0 and dst.exists()


def _measure_perplexity(cfg: Config, gguf: Path, corpus: Path,
                        log_path: Path
                        ) -> tuple[Optional[float], Optional[float]]:
    bin_ = find_tool(cfg, "llama-perplexity")
    rc, log = _run_cmd(
        [str(bin_), "-m", str(gguf), "-f", str(corpus),
         "-c", "2048", "-ngl", "99", "-fa", "on",
         "-ctk", "f16", "-ctv", "f16",
         "--chunks", str(cfg.ppl_chunks), "--no-mmap"],
        subprocess_env(cfg), log_path)
    m = re.search(r"Final estimate:\s*PPL\s*=\s*([\d.]+)\s*\+/-\s*([\d.]+)", log)
    if not m:
        print(f"    WARN: no Final estimate (rc={rc}); last 5 log lines:")
        for line in log.splitlines()[-5:]:
            print(f"      | {line}")
        return None, None
    return float(m.group(1)), float(m.group(2))


def _measure_bench(cfg: Config, gguf: Path, log_path: Path
                   ) -> tuple[Optional[float], Optional[float]]:
    bin_ = find_tool(cfg, "llama-bench")
    rc, log = _run_cmd(
        [str(bin_), "-m", str(gguf), "-p", "512", "-n", "128",
         "-t", "12", "-ngl", "99", "-fa", "1",
         "-ctk", "f16", "-ctv", "f16", "--output", "csv"],
        subprocess_env(cfg), log_path)
    pp = tg = None
    for line in log.splitlines():
        if '"512","0","0"' in line:
            parts = [p.strip('"') for p in line.split(",")]
            if len(parts) >= 2:
                try: pp = float(parts[-2])
                except ValueError: pass
        elif '"0","128","0"' in line:
            parts = [p.strip('"') for p in line.split(",")]
            if len(parts) >= 2:
                try: tg = float(parts[-2])
                except ValueError: pass
    return pp, tg


def _count_params(gguf: Path) -> int:
    """Count total parameters from GGUF header (type-agnostic)."""
    import struct
    total = 0
    with gguf.open("rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"not a GGUF: {gguf}")
        f.read(4)  # version
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
        def skip_value(vt: int) -> None:
            sizes = {0:1,1:1,7:1, 2:2,3:2, 4:4,5:4,6:4, 10:8,11:8,12:8}
            if vt in sizes: f.read(sizes[vt])
            elif vt == 8:
                slen = struct.unpack("<Q", f.read(8))[0]; f.read(slen)
            elif vt == 9:
                etype = struct.unpack("<I", f.read(4))[0]
                n = struct.unpack("<Q", f.read(8))[0]
                for _ in range(n): skip_value(etype)
        for _ in range(n_kv):
            klen = struct.unpack("<Q", f.read(8))[0]; f.read(klen)
            skip_value(struct.unpack("<I", f.read(4))[0])
        for _ in range(n_tensors):
            nlen = struct.unpack("<Q", f.read(8))[0]; f.read(nlen)
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = struct.unpack("<" + "Q" * n_dims, f.read(8 * n_dims))
            f.read(4); f.read(8)
            n_elem = 1
            for d in dims: n_elem *= d
            total += n_elem
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Calibration core
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_reference_gguf(cfg: Config, layout: Layout,
                            resolved: ResolvedInput) -> Path:
    """Get a BF16/F16 GGUF on disk for the reference model. Uses pipeline
    helpers so calibrate→run sequences share intermediates."""
    if resolved.kind == "hf":
        sf = download_hf(cfg, layout, resolved.hf_id, resolved.model_name)
        return convert_to_bf16(cfg, layout, sf, resolved.model_name)
    if resolved.kind == "safetensors_dir":
        return convert_to_bf16(cfg, layout, resolved.safetensors_dir,
                               resolved.model_name)
    if resolved.kind == "gguf_local":
        return resolved.gguf_path
    if resolved.kind == "gguf_url":
        return download_gguf_url(cfg, layout, resolved.gguf_urls,
                                  resolved.model_name)
    raise ValueError(f"unknown input kind: {resolved.kind}")


def _is_complete(m: FormatMeasurement) -> bool:
    """A measurement is 'complete' (and skippable on resume) when every
    field that calibrate normally fills is present, and there's no error."""
    return (m.error is None
            and m.bpw is not None
            and m.ppl is not None
            and m.pp is not None
            and m.tg is not None)


def _load_existing(output_path: Path) -> dict[str, FormatMeasurement]:
    """Load a prior partial perf JSON from disk so calibration can resume."""
    if not output_path.exists():
        return {}
    try:
        data = json.loads(output_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[str, FormatMeasurement] = {}
    for fmt, entry in data.items():
        if fmt.startswith("_") or not isinstance(entry, dict):
            continue
        out[fmt] = FormatMeasurement(
            bpw=entry.get("bpw"),
            size_bytes=entry.get("size_bytes"),
            ppl=entry.get("ppl"),
            ppl_delta_vs_f16=entry.get("ppl_delta_vs_f16"),
            pp=entry.get("pp"),
            tg=entry.get("tg"),
            pp_ratio_vs_bf16=entry.get("pp_ratio_vs_bf16"),
            tg_ratio_vs_bf16=entry.get("tg_ratio_vs_bf16"),
            error=entry.get("error"),
        )
    return out


def _calibrate_formats(cfg: Config, ref_gguf: Path, scratch_dir: Path,
                       ppl_corpus: Path, layout: Layout, output_path: Path,
                       model_name: str,
                       imatrix: Optional[Path] = None,
                       ) -> dict[str, FormatMeasurement]:
    """Run quantize → perplexity → bench for each format. Persists the perf
    JSON after every format so a kill mid-sweep preserves progress, and on
    re-entry skips formats already measured to completion.

    Live subprocess output streams to {logs_dir}/calibrate-<fmt>.log; meta
    progress lines also go to {logs_dir}/calibrate.log so a single tail
    shows the whole calibration."""
    n_params = _count_params(ref_gguf)
    meta_log = layout.logs_dir / "calibrate.log"

    # Always measure BF16 reference for ratio computation.
    formats = list(cfg.quants)
    if "BF16" not in formats:
        formats = ["BF16"] + formats

    # Resume from any prior partial perf JSON.
    results: dict[str, FormatMeasurement] = _load_existing(output_path)
    if results:
        n_complete = sum(1 for m in results.values() if _is_complete(m))
        _log(meta_log, f"[calibrate] resume: {n_complete}/{len(formats)} "
                       f"formats already complete in {output_path}")

    bf16_ppl: Optional[float] = None
    bf16_existing = results.get("BF16")
    if bf16_existing and bf16_existing.ppl is not None:
        bf16_ppl = bf16_existing.ppl

    _log(meta_log, f"[calibrate] {len(formats)} formats × ~5–15 min each")
    _log(meta_log, f"[calibrate] reference model: {ref_gguf} "
                   f"({n_params/1e9:.2f}B params)")
    _log(meta_log, f"[calibrate] live logs: {layout.logs_dir}/calibrate-*.log")

    for i, fmt in enumerate(formats, 1):
        existing = results.get(fmt)
        if existing and _is_complete(existing):
            _log(meta_log, f"[{i:>2}/{len(formats)}] {fmt} — cached, skip")
            continue

        _log(meta_log, f"[{i:>2}/{len(formats)}] {fmt}")
        fmt_log = layout.logs_dir / f"calibrate-{fmt}.log"
        m = FormatMeasurement()
        scratch = scratch_dir / f"ref-{fmt}.gguf"

        if fmt in ("BF16", "F16") and ref_gguf.stem.endswith(("BF16", "F16")):
            # Reference GGUF already IS this format — skip quantize, measure directly.
            target = ref_gguf
        else:
            _log(meta_log, f"   quantize → {scratch.name}  (live: {fmt_log.name})")
            t0 = time.time()
            if not _quantize_one(cfg, ref_gguf, scratch, fmt, fmt_log,
                                  imatrix=imatrix):
                m.error = "quantize failed"
                results[fmt] = m
                _write_perf_json(output_path, results, model_name, cfg.ppl_chunks)
                _log(meta_log, f"   FAIL ({time.time()-t0:.1f}s)")
                continue
            _log(meta_log, f"   quantize done ({time.time()-t0:.1f}s)")
            target = scratch

        m.size_bytes = target.stat().st_size
        m.bpw = round(m.size_bytes * 8 / n_params, 4)

        _log(meta_log, f"   ppl (chunks={cfg.ppl_chunks})  (live: {fmt_log.name})")
        ppl, ppl_err = _measure_perplexity(cfg, target, ppl_corpus, fmt_log)
        m.ppl, m.ppl_stderr = ppl, ppl_err
        if fmt == "BF16" and bf16_ppl is None:
            bf16_ppl = ppl
        elif bf16_ppl is not None and ppl is not None:
            m.ppl_delta_vs_f16 = round(ppl - bf16_ppl, 4)

        _log(meta_log, f"   bench  (live: {fmt_log.name})")
        m.pp, m.tg = _measure_bench(cfg, target, fmt_log)

        results[fmt] = m
        # Persist after every format so a crash preserves progress.
        _write_perf_json(output_path, results, model_name, cfg.ppl_chunks)

        # Delete scratch (not the original ref_gguf)
        if target is scratch:
            try: scratch.unlink(missing_ok=True)
            except OSError: pass

        _log(meta_log, f"   → bpw={m.bpw} ppl={m.ppl} "
                       f"Δ={m.ppl_delta_vs_f16} pp={m.pp} tg={m.tg}")

    # Backfill ratios after we have BF16 numbers (idempotent — re-runnable on resume).
    bf16 = results.get("BF16")
    if bf16 and bf16.pp and bf16.tg:
        for m in results.values():
            if m.pp is not None:
                m.pp_ratio_vs_bf16 = round(m.pp / bf16.pp, 4)
            if m.tg is not None:
                m.tg_ratio_vs_bf16 = round(m.tg / bf16.tg, 4)
    # Backfill ppl_delta_vs_f16 too, in case BF16 was measured later than other formats.
    if bf16 and bf16.ppl is not None:
        for fmt_name, m in results.items():
            if (m.ppl is not None and m.ppl_delta_vs_f16 is None
                    and fmt_name != "BF16"):
                m.ppl_delta_vs_f16 = round(m.ppl - bf16.ppl, 4)
    # Final write with all backfilled ratios in place.
    _write_perf_json(output_path, results, model_name, cfg.ppl_chunks)

    return results


def _write_perf_json(output: Path, results: dict[str, FormatMeasurement],
                     model_name: str, chunks: int) -> None:
    out = {
        "_schema_version": 4,
        "_reference_model": model_name,
        "_reference_format": "BF16",
        "_calibrated_at": datetime.now(timezone.utc).isoformat(),
        "_calibration_chunks": chunks,
    }
    for fmt, m in results.items():
        entry: dict = {}
        if m.bpw is not None:        entry["bpw"] = m.bpw
        if m.size_bytes is not None: entry["size_bytes"] = m.size_bytes
        if m.ppl is not None:        entry["ppl"] = round(m.ppl, 4)
        if m.ppl_delta_vs_f16 is not None:
            entry["ppl_delta_vs_f16"] = m.ppl_delta_vs_f16
        if m.pp is not None:         entry["pp"] = round(m.pp, 2)
        if m.tg is not None:         entry["tg"] = round(m.tg, 2)
        if m.pp_ratio_vs_bf16 is not None:
            entry["pp_ratio_vs_bf16"] = m.pp_ratio_vs_bf16
        if m.tg_ratio_vs_bf16 is not None:
            entry["tg_ratio_vs_bf16"] = m.tg_ratio_vs_bf16
        if m.error:
            entry["error"] = m.error
        out[fmt] = entry
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Public entrypoints
# ─────────────────────────────────────────────────────────────────────────────

def run_calibrate(cfg: Config, mode: str, resolved: ResolvedInput,
                  purge: str,
                  imatrix_override: Optional[str] = None,
                  assume_yes: bool = False) -> Optional[Path]:
    """Run calibration. mode ∈ {'system', 'model'}.

    Returns path to the written perf JSON, or None if the user aborted
    at the pre-flight prompt."""
    layout = Layout.for_run(base=cfg.base, model_name=resolved.model_name)

    # Pre-flight estimate + accept/abort.
    est = estimate_calibrate(cfg, mode, resolved, layout)
    if not confirm_or_abort(est, assume_yes=assume_yes):
        return None

    layout.make()

    print("=" * 70)
    print(f"prismaquant-llama calibrate {mode} — {resolved.model_name}")
    print("=" * 70)
    for line in layout.summary_lines():
        print(line)
    print()

    ppl_corpus, ppl_was_downloaded = resolve_corpus(cfg, "ppl")
    imatrix_corpus, imatrix_was_downloaded = resolve_corpus(cfg, "imatrix")
    print(f"[calibrate] ppl_corpus:     {ppl_corpus}")
    print(f"[calibrate] imatrix_corpus: {imatrix_corpus}")

    # Compute the output path BEFORE running so resume can find a prior
    # partial JSON if a previous calibration of this model+mode crashed.
    if mode == "system":
        output = layout.system_calibration_path()
    else:
        output = layout.model_calibration_path(resolved.model_name)

    ref_gguf = _prepare_reference_gguf(cfg, layout, resolved)

    # imatrix: explicit override > generate from imatrix_corpus.
    # i-quants and IK-family quants are designed around imatrix-weighted
    # quantization; calibrating without one would systematically penalize
    # those formats vs K-quants and bias the allocator's ranking.
    if imatrix_override:
        imatrix_path = _resolve_imatrix_override(cfg, layout, imatrix_override)
        print(f"[calibrate] using --imatrix override: {imatrix_path}")
    else:
        imatrix_path = stage_d_imatrix(cfg, layout, ref_gguf, imatrix_corpus)

    with tempfile.TemporaryDirectory(prefix="pq-cal-",
                                     dir=str(layout.work)) as tmp:
        _calibrate_formats(cfg, ref_gguf, Path(tmp), ppl_corpus,
                           layout, output, resolved.model_name,
                           imatrix=imatrix_path)

    print(f"\n[calibrate] perf JSON written → {output}")

    # Purge logic mirrors run_pipeline: delete what we downloaded/generated
    # if input wasn't on-disk. Corpora download cleanup also applies.
    user_owns_model = resolved.kind in ("safetensors_dir", "gguf_local")
    if purge == "yes":
        if not user_owns_model:
            if resolved.kind == "hf":
                target = layout.hf_cache / resolved.model_name
                if target.exists():
                    shutil.rmtree(target, ignore_errors=True)
                    print(f"  [purge] removed: hf-cache/{resolved.model_name}")
            if resolved.kind == "gguf_url":
                target = layout.gguf_cache / resolved.model_name
                if target.exists():
                    shutil.rmtree(target, ignore_errors=True)
                    print(f"  [purge] removed: gguf-cache/{resolved.model_name}")
            bf16 = layout.bf16_dir / f"{resolved.model_name}-BF16.gguf"
            if bf16.exists() and resolved.kind in ("hf", "safetensors_dir"):
                # Only delete if WE generated it (not for gguf_local/url where
                # the input GGUF is the file itself)
                if resolved.kind == "hf":
                    bf16.unlink()
                    print("  [purge] removed: bf16")
        if ppl_was_downloaded:
            ppl_corpus.unlink(missing_ok=True)
            print("  [purge] removed: ppl-corpus")

    print("=" * 70)
    return output


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def add_calibrate_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("mode", choices=("system", "model"),
                   help="'system' writes {base}/calibration/system.json; "
                        "'model' writes {base}/calibration/models/<name>.json")
    p.add_argument("input", metavar="INPUT",
                   help="HuggingFace id, on-disk safetensors dir, on-disk "
                        "f16/bf16 GGUF, or URL(s) to a GGUF "
                        "(comma-separate split files)")
    p.add_argument("--config", type=Path, default=None,
                   help="alternative config.toml")
    p.add_argument("--libs", type=Path, default=None,
                   help="extra dir prepended to LD_LIBRARY_PATH")
    p.add_argument("--base", type=Path, default=None)
    p.add_argument("--path", type=Path, default=None)
    p.add_argument("--quants", default=None,
                   help="comma-separated quants list (default: from config)")
    p.add_argument("--ppl-corpus", default=None)
    p.add_argument("--ppl-chunks", type=int, default=None)
    p.add_argument("--imatrix-corpus", default=None,
                   help="imatrix corpus path or URL (default: from config). "
                        "Used to generate the imatrix file consumed by "
                        "llama-quantize during calibration; without it, "
                        "i-quants and IK-family quants are penalized.")
    p.add_argument("--imatrix-chunks", type=int, default=None,
                   help="chunks for llama-imatrix (default: from config)")
    p.add_argument("--imatrix", default=None,
                   help="existing imatrix file path or URL (overrides "
                        "imatrix generation)")
    p.add_argument("--convert-script", type=Path, default=None,
                   help="path to convert_hf_to_gguf.py (default: from config "
                        "or auto-discover; only relevant when input is "
                        "safetensors and Stage B convert needs to run)")
    p.add_argument("--purge", choices=("yes", "no"), default="yes",
                   help="clean up downloaded/generated artifacts after "
                        "(default: yes; never deletes user-supplied on-disk inputs)")
    p.add_argument("--yes", "-y", action="store_true",
                   help="skip the pre-flight disk + time confirmation prompt "
                        "(required for non-interactive / scripted use)")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="prismaquant-llama calibrate",
                                description="Measure per-format perf data")
    add_calibrate_args(p)
    args = p.parse_args(argv)

    # Reuse run-style config resolution for the shared flags.
    # Fill missing run-only flags with defaults so cfg_from_args is happy.
    args.budget = None
    args.priority = None
    cfg = cfg_from_args(args)

    resolved = resolve_input(args.input, allow_gguf=True)
    try:
        result = run_calibrate(cfg, args.mode, resolved, args.purge,
                                imatrix_override=args.imatrix,
                                assume_yes=args.yes)
        return 0 if result is not None else 1
    except (SystemExit, FileNotFoundError, ValueError) as e:
        print(f"\nFAIL: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
