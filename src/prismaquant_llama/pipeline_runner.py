"""
End-to-end prismaquant pipeline.

Stages (safetensors-input only — Stage C requires safetensors for the
Hessian probe):

    A. Download HF safetensors                  (HF id only)
    B. Convert safetensors → BF16 GGUF          (HF id and on-disk safetensors)
    C. Hessian probe via prismaquant.incremental_probe
    D. imatrix generation (llama-imatrix)
    E. per-(tensor, format) MSE costs (llama-quantize-cost)
    F. Bridge HF→GGUF tensor names (bundled bridge_probe_to_gguf.py)
    G. Allocate formats per tensor (bundled allocator.py)
    H. Apply recipe (llama-quantize)
    I. Final PPL eval (llama-perplexity)

Each stage caches by file existence and is idempotent.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import (Config, load_config, find_tool, subprocess_env,
                     resolve_corpus, LLAMA_TOOLS)
from .input_resolver import ResolvedInput, resolve as resolve_input
from .paths import Layout


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers (subprocess wrapper, file hashing, convert-script finder)
# ─────────────────────────────────────────────────────────────────────────────

def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(layout: Layout, stage: str, msg: str) -> None:
    line = f"[{_ts()}] {msg}"
    print(line)
    log_path = layout.logs_dir / f"stage-{stage}.log"
    if layout.logs_dir.exists():
        with log_path.open("a") as f:
            f.write(line + "\n")


def _run(cmd: list[str], log_path: Path, env: Optional[dict] = None,
         timeout: Optional[float] = None) -> int:
    """Run subprocess, tee output to log_path, return rc."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    with log_path.open("a") as f:
        f.write(f"\n=== {_ts()}: {' '.join(str(c) for c in cmd)} ===\n")
        f.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                env=env, text=True, bufsize=1)
        try:
            for line in proc.stdout:
                f.write(line)
                f.flush()
        except KeyboardInterrupt:
            proc.terminate()
            raise
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            return 124
    return proc.returncode


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_convert_script(cfg: Config) -> Path:
    """Locate convert_hf_to_gguf.py from the llama.cpp source tree.

    Resolution order:
        1. cfg.convert_script (from --convert-script flag or config key)
        2. <fork-root>/convert_hf_to_gguf.py (binary.parents[2], for build-tree layouts)
        3. binary.parent/convert_hf_to_gguf.py (next to binary)
        4. shutil.which (on $PATH)
    """
    if cfg.convert_script is not None:
        if not cfg.convert_script.exists():
            raise FileNotFoundError(
                f"convert_hf_to_gguf.py not found at {cfg.convert_script} "
                f"(from config 'convert_script' or --convert-script).")
        return cfg.convert_script
    binary = find_tool(cfg, "llama-quantize")
    candidates = [binary.parent / "convert_hf_to_gguf.py"]
    if len(binary.parents) >= 2:
        candidates.insert(0, binary.parents[2] / "convert_hf_to_gguf.py")
    for cand in candidates:
        if cand.exists():
            return cand
    p = shutil.which("convert_hf_to_gguf.py")
    if p:
        return Path(p)
    raise FileNotFoundError(
        "convert_hf_to_gguf.py not found. The script lives at the ROOT of "
        "your llama.cpp source tree and is NOT installed by `cmake --install`. "
        "Either:\n"
        "  - keep your llama.cpp source tree at <fork>/build/bin/llama-quantize "
        "    (we walk up to <fork>/convert_hf_to_gguf.py),\n"
        "  - set 'convert_script = \"/path/to/convert_hf_to_gguf.py\"' in "
        "    ~/.prismaquant-llama/config.toml,\n"
        "  - or pass --convert-script /path/to/convert_hf_to_gguf.py per run.")


# ─────────────────────────────────────────────────────────────────────────────
# Input preparation (Stage A + Stage B; reused by calibration)
# ─────────────────────────────────────────────────────────────────────────────

def download_hf(cfg: Config, layout: Layout, hf_id: str,
                model_name: str) -> Path:
    """Stage A. Download HF safetensors snapshot. Idempotent.

    Returns the safetensors directory."""
    target = layout.hf_cache / model_name
    marker = target / ".download.complete"
    if marker.exists():
        _log(layout, "A", f"A. HF model cached at {target} (skip)")
        return target
    _log(layout, "A", f"A. downloading {hf_id} → {target}")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit(
            "ERROR: huggingface_hub not installed. Run:\n"
            "       pip install --user huggingface_hub")
    snapshot_download(repo_id=hf_id, local_dir=str(target))
    marker.touch()
    _log(layout, "A", f"A. download complete → {target}")
    return target


def convert_to_bf16(cfg: Config, layout: Layout,
                    safetensors_dir: Path, model_name: str) -> Path:
    """Stage B. Convert safetensors → BF16 GGUF. Idempotent."""
    out = layout.bf16_dir / f"{model_name}-BF16.gguf"
    if out.exists():
        _log(layout, "B", f"B. BF16 GGUF cached at {out} (skip)")
        return out
    convert_script = _find_convert_script(cfg)
    _log(layout, "B", f"B. converting {safetensors_dir} → {out}")
    rc = _run(["python3", str(convert_script), str(safetensors_dir),
               "--outtype", "bf16", "--outfile", str(out)],
              layout.logs_dir / "stage-B.log",
              env=subprocess_env(cfg))
    if rc != 0 or not out.exists():
        raise SystemExit(f"FAIL: B convert_hf_to_gguf.py exit={rc}")
    _log(layout, "B", f"B. BF16 GGUF: {out.stat().st_size/1024**3:.2f} GB")
    return out


def download_gguf_url(cfg: Config, layout: Layout, urls: list[str],
                      model_name: str) -> Path:
    """Download single or split-file GGUF from URL(s) into the gguf-cache.

    For a single URL → returns the local path.
    For split URLs (comma-separated) → downloads all parts; returns the
    first part. llama.cpp tools auto-detect splits via the standard naming."""
    target_dir = layout.gguf_cache / model_name
    target_dir.mkdir(parents=True, exist_ok=True)
    local_paths: list[Path] = []
    for url in urls:
        filename = url.rsplit("/", 1)[-1] or "model.gguf"
        local = target_dir / filename
        local_paths.append(local)
        if local.exists():
            print(f"  GGUF cached → {local} (skip)")
            continue
        print(f"  downloading {url} → {local}")
        import urllib.request
        with urllib.request.urlopen(url) as resp, local.open("wb") as f:
            shutil.copyfileobj(resp, f)
    return local_paths[0]


# ─────────────────────────────────────────────────────────────────────────────
# Stages C–I
# ─────────────────────────────────────────────────────────────────────────────

def stage_c_probe(cfg: Config, layout: Layout, safetensors_dir: Path,
                  imatrix_corpus: Path, model_name: str) -> Path:
    """Stage C — Hessian probe. Requires safetensors."""
    probe_path = layout.probe_dir / f"{model_name}-probe.pkl"
    if probe_path.exists():
        _log(layout, "C", f"C. probe cached at {probe_path} (skip)")
        return probe_path
    _log(layout, "C", f"C. running prismaquant.incremental_probe on {safetensors_dir}")
    rc = _run(["python3", "-m", "prismaquant.incremental_probe",
               "--model", str(safetensors_dir),
               "--dataset", str(imatrix_corpus),
               "--nsamples", "16", "--seqlen", "512",
               "--device", "cpu", "--dtype", "bf16",
               "--output", str(probe_path),
               "--activation-cache-dir", str(layout.probe_dir / "act-cache"),
               "--work-dir", str(layout.probe_dir / "work")],
              layout.logs_dir / "stage-C.log",
              env=subprocess_env(cfg))
    if rc != 0 or not probe_path.exists():
        raise SystemExit(
            f"FAIL: C prismaquant.incremental_probe exit={rc}\n"
            f"  Likely cause: prismaquant package not installed.\n"
            f"  Install our fork: "
            f"pip install git+https://github.com/jimbothigpen/prismaquant.git")
    _log(layout, "C", f"C. probe.pkl: {probe_path.stat().st_size/1024**2:.1f} MB")
    return probe_path


def stage_d_imatrix(cfg: Config, layout: Layout, bf16_path: Path,
                    imatrix_corpus: Path, ctx: int = 4096) -> Path:
    """Stage D — generate (or download) imatrix file."""
    model_sha = _file_sha256(bf16_path)
    corpus_sha = _file_sha256(imatrix_corpus)
    cache = layout.imatrix_cache_path(model_sha, corpus_sha, cfg.imatrix_chunks)
    if cache.exists():
        _log(layout, "D", f"D. imatrix cached at {cache} (skip)")
        return cache
    imatrix_bin = find_tool(cfg, "llama-imatrix")
    _log(layout, "D", f"D. generating imatrix → {cache}")
    rc = _run([str(imatrix_bin), "-m", str(bf16_path),
               "-f", str(imatrix_corpus), "-o", str(cache),
               "-c", str(ctx), "-ngl", "99", "--no-mmap",
               "--chunks", str(cfg.imatrix_chunks),
               "-ctk", "f16", "-ctv", "f16"],
              layout.logs_dir / "stage-D.log",
              env=subprocess_env(cfg))
    if rc != 0 or not cache.exists():
        raise SystemExit(f"FAIL: D llama-imatrix exit={rc}")
    _log(layout, "D", f"D. imatrix: {cache.stat().st_size/1024**2:.1f} MB")
    return cache


def stage_e_costs(cfg: Config, layout: Layout, bf16_path: Path,
                  imatrix_path: Path) -> Path:
    """Stage E — per-(tensor, format) MSE measurement."""
    costs = layout.costs_dir / "costs.csv"
    if costs.exists():
        _log(layout, "E", f"E. costs.csv cached at {costs} (skip)")
        return costs
    cost_bin = find_tool(cfg, "llama-quantize-cost")
    exemplars = _auto_pick_exemplars(bf16_path)
    top_level = _discover_top_level_weights(bf16_path)
    layer_alt = "|".join(str(L) for L in exemplars)
    top_alt = "|".join(top_level)
    include_regex = rf"^({top_alt}|blk\.({layer_alt}))\."
    _log(layout, "E", f"E. measuring per-(tensor, format) MSE → {costs}")
    _log(layout, "E", f"E. top-level weights: {top_level}")
    _log(layout, "E", f"E. exemplar layers: {exemplars}")
    rc = _run([str(cost_bin),
               "--model", str(bf16_path),
               "--types", ",".join(cfg.quants),
               "--imatrix", str(imatrix_path),
               "--include-regex", include_regex,
               "--output", str(costs)],
              layout.logs_dir / "stage-E.log",
              env=subprocess_env(cfg))
    if rc != 0 or not costs.exists():
        raise SystemExit(f"FAIL: E llama-quantize-cost exit={rc}")
    _log(layout, "E", f"E. costs.csv rows: {sum(1 for _ in costs.open())}")
    return costs


def stage_f_bridge(cfg: Config, layout: Layout, probe_path: Path) -> Path:
    """Stage F — bridge HF→GGUF tensor names."""
    bridge = layout.work / "bridge.json"
    if bridge.exists():
        _log(layout, "F", f"F. bridge.json cached (skip)")
        return bridge
    bridge_script = _bundled_script("bridge_probe_to_gguf.py")
    _log(layout, "F", f"F. bridging probe → {bridge}")
    rc = _run(["python3", str(bridge_script),
               "--probe", str(probe_path),
               "--output", str(bridge),
               "--aggregate", "sum",
               "--unmapped-out", str(layout.work / "bridge-unmapped.json")],
              layout.logs_dir / "stage-F.log",
              env=subprocess_env(cfg))
    if rc != 0 or not bridge.exists():
        raise SystemExit(f"FAIL: F bridge exit={rc}")
    return bridge


def stage_g_allocate(cfg: Config, layout: Layout,
                     bridge_path: Path, costs_path: Path,
                     bf16_path: Path, perf_file: Optional[Path],
                     budget_gb: float, model_name: str) -> Path:
    """Stage G — multi-choice knapsack allocation."""
    recipe = layout.recipes_dir / f"recipe-PQ{cfg.budget}-{cfg.priority}.json"
    if recipe.exists():
        _log(layout, "G", f"G. recipe cached (skip)")
        return recipe
    allocator = _bundled_script("allocator.py")
    pinned = layout.work / "pinned.json"
    pinned.write_text(json.dumps({
        "output.weight": "Q6_K",
        "token_embd.weight": "Q8_0",
    }, indent=2))
    _log(layout, "G", f"G. allocating @ {budget_gb:.2f} GB priority {cfg.priority}")
    cmd = ["python3", str(allocator),
           "--bridge", str(bridge_path),
           "--costs", str(costs_path),
           "--budget-gb", str(budget_gb),
           "--budget-band-gb", "0.25",
           "--pinned", str(pinned),
           "--priority", cfg.priority,
           "--recipe-out", str(recipe),
           "--allow-types", ",".join(cfg.quants),
           "--gguf", str(bf16_path),
           "--propagate-from-exemplars",
           "--exemplar-layers", "0,3"]
    if perf_file is not None and perf_file.exists():
        cmd += ["--tps", str(perf_file)]
        _log(layout, "G", f"G. perf data: {perf_file}")
    # Floor 4.0 bpw on attention weights to avoid 2/3-bit on QK^T-sensitive paths.
    floor = {r"^blk\..*\.attn_(q|k|v|qkv|gate|output)\.weight$": 4.0}
    cmd += ["--floor-bpw", json.dumps(floor)]
    rc = _run(cmd, layout.logs_dir / "stage-G.log", env=subprocess_env(cfg))
    if rc != 0 or not recipe.exists():
        raise SystemExit(f"FAIL: G allocator exit={rc}")
    return recipe


def stage_h_quantize(cfg: Config, layout: Layout, bf16_path: Path,
                     recipe_path: Path, imatrix_path: Path,
                     model_name: str) -> Path:
    """Stage H — apply allocation."""
    out = layout.gguf_output_path(model_name, cfg.budget, cfg.priority)
    if out.exists():
        _log(layout, "H", f"H. GGUF cached at {out} (skip)")
        return out
    recipe_txt = recipe_path.with_suffix(".txt")
    if not recipe_txt.exists():
        data = json.loads(recipe_path.read_text())
        assignments = data.get("assignments") or data
        with recipe_txt.open("w") as f:
            for tensor, fmt in assignments.items():
                if isinstance(fmt, dict):
                    fmt = fmt.get("type") or fmt.get("format")
                f.write(f"{tensor}={fmt}\n")
    quantize_bin = find_tool(cfg, "llama-quantize")
    _log(layout, "H", f"H. applying recipe → {out}")
    rc = _run([str(quantize_bin),
               "--imatrix", str(imatrix_path),
               "--tensor-type-file", str(recipe_txt),
               str(bf16_path), str(out), "Q4_K"],
              layout.logs_dir / "stage-H.log",
              env=subprocess_env(cfg))
    if rc != 0 or not out.exists():
        raise SystemExit(f"FAIL: H llama-quantize exit={rc}")
    _log(layout, "H", f"H. final GGUF: {out} ({out.stat().st_size/1024**3:.2f} GB)")
    return out


def stage_i_eval(cfg: Config, layout: Layout, gguf: Path,
                 ppl_corpus: Path, ctx: int = 4096) -> Optional[float]:
    """Stage I — final PPL evaluation."""
    perp_bin = find_tool(cfg, "llama-perplexity")
    log = layout.logs_dir / "stage-I.log"
    _log(layout, "I", f"I. PPL eval @ chunks={cfg.ppl_chunks}")
    rc = _run([str(perp_bin), "-m", str(gguf), "-f", str(ppl_corpus),
               "-c", str(ctx), "-b", "2048",
               "-ctk", "f16", "-ctv", "f16", "-fa", "on",
               "-ngl", "99", "--chunks", str(cfg.ppl_chunks), "--no-mmap"],
              log, env=subprocess_env(cfg))
    import re as _re
    text = log.read_text()
    m = _re.search(r"Final estimate:\s*PPL\s*=\s*([\d.]+)", text)
    if m:
        ppl = float(m.group(1))
        _log(layout, "I", f"I. PPL = {ppl:.4f}")
        return ppl
    _log(layout, "I", f"I. WARN: no Final estimate (rc={rc})")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Auto-detection helpers (invoked by Stage E)
# ─────────────────────────────────────────────────────────────────────────────

def _bundled_script(name: str) -> Path:
    """Resolve a bundled pipeline script. Lives inside the installed
    package, not the source tree, so it works post-`pip install`."""
    p = Path(__file__).parent / "scripts" / name
    if not p.exists():
        raise FileNotFoundError(f"bundled script missing: {p}")
    return p


def _auto_pick_exemplars(bf16_path: Path) -> list[int]:
    """Choose exemplar layers for Stage E. iSWA arches → one per layer-type;
    standard arches → [0, 3] safe pair."""
    try:
        sys.path.insert(0, str(_bundled_script("allocator.py").parent))
        from allocator import (read_gguf_tensor_meta, detect_layer_types,
                                auto_pick_exemplar_layers)
        meta = read_gguf_tensor_meta(str(bf16_path))
        layer_type = detect_layer_types(meta)
        if not layer_type or len(set(layer_type.values())) <= 1:
            return [0, 3]
        return auto_pick_exemplar_layers(layer_type)
    except Exception as e:
        print(f"  WARN: exemplar auto-detect failed ({e}); using [0, 3]")
        return [0, 3]


def _discover_top_level_weights(bf16_path: Path) -> list[str]:
    """Find all top-level weight tensors in the BF16 GGUF (not under blk.).
    Catches per_layer_token_embd.weight on gemma-4, etc."""
    try:
        sys.path.insert(0, str(_bundled_script("allocator.py").parent))
        from allocator import read_gguf_tensor_meta
        meta = read_gguf_tensor_meta(str(bf16_path))
        names = set()
        for tn, shape in meta.items():
            if tn.startswith("blk.") or not tn.endswith(".weight"):
                continue
            if len(shape) < 2:
                continue
            base = tn[:-len(".weight")]
            if "." in base:
                continue
            names.add(base)
        return sorted(names) or ["token_embd", "output"]
    except Exception as e:
        print(f"  WARN: top-level weight discovery failed ({e}); using default")
        return ["token_embd", "output"]


# ─────────────────────────────────────────────────────────────────────────────
# Perf-file resolution (model > system > shipped default)
# ─────────────────────────────────────────────────────────────────────────────

def find_perf_file(layout: Layout, model_name: str) -> Optional[Path]:
    """Two-tier lookup: model-specific > system > shipped."""
    model_specific = layout.model_calibration_path(model_name)
    if model_specific.exists():
        return model_specific
    system = layout.system_calibration_path()
    if system.exists():
        return system
    shipped = Path(__file__).parent / "data" / "system.json.default"
    if shipped.exists():
        return shipped
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PurgeContext:
    """Tracks artifacts to clean up at end of run if --purge yes."""
    user_owns_model: bool                      # True if input was on-disk
    ppl_was_downloaded: bool
    imatrix_was_downloaded: bool
    ppl_corpus_local: Path
    imatrix_corpus_local: Path


def cleanup(layout: Layout, resolved: ResolvedInput, ctx: PurgeContext,
            purge: str) -> None:
    """Delete what we downloaded/generated this run."""
    if purge != "yes":
        return
    freed: list[str] = []

    def _rm(p: Path, label: str) -> None:
        try:
            if p.is_file() or p.is_symlink():
                p.unlink()
                freed.append(label)
            elif p.is_dir():
                shutil.rmtree(p)
                freed.append(label)
        except OSError:
            pass

    if not ctx.user_owns_model:
        if resolved.kind == "hf":
            _rm(layout.hf_cache / resolved.model_name, "hf-cache")
        if resolved.kind == "gguf_url":
            _rm(layout.gguf_cache / resolved.model_name, "gguf-cache")
        _rm(layout.bf16_dir / f"{resolved.model_name}-BF16.gguf", "bf16")
        for p in layout.probe_dir.glob(f"{resolved.model_name}-*"):
            _rm(p, "probe")
        for p in layout.imatrix_cache.glob(f"{resolved.model_name}*"):
            _rm(p, "imatrix")

    if ctx.ppl_was_downloaded:
        _rm(ctx.ppl_corpus_local, "ppl-corpus")
    if ctx.imatrix_was_downloaded:
        _rm(ctx.imatrix_corpus_local, "imatrix-corpus")

    if freed:
        print(f"  [purge] removed: {', '.join(freed)}")


# ─────────────────────────────────────────────────────────────────────────────
# Run pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(cfg: Config, resolved: ResolvedInput,
                 imatrix_override: Optional[str], purge: str) -> Path:
    """Execute the full A→I pipeline. Returns path to the final GGUF."""
    layout = Layout.for_run(base=cfg.base, model_name=resolved.model_name)
    layout.make()

    print("=" * 70)
    print(f"prismaquant-llama run — {layout.run_label}")
    print("=" * 70)
    for line in layout.summary_lines():
        print(line)
    print()

    # Resolve corpora (may download if URL)
    ppl_corpus, ppl_was_downloaded = resolve_corpus(cfg, "ppl")
    imatrix_corpus, imatrix_was_downloaded = resolve_corpus(cfg, "imatrix")

    purge_ctx = PurgeContext(
        user_owns_model=resolved.kind in ("safetensors_dir", "gguf_local"),
        ppl_was_downloaded=ppl_was_downloaded,
        imatrix_was_downloaded=imatrix_was_downloaded,
        ppl_corpus_local=ppl_corpus,
        imatrix_corpus_local=imatrix_corpus,
    )

    # A: get safetensors directory
    if resolved.kind == "hf":
        safetensors_dir = download_hf(cfg, layout, resolved.hf_id, resolved.model_name)
    elif resolved.kind == "safetensors_dir":
        safetensors_dir = resolved.safetensors_dir
        _log(layout, "A", f"A. using on-disk safetensors at {safetensors_dir}")
    else:
        raise AssertionError("run requires safetensors input")

    # B: convert to BF16
    bf16_path = convert_to_bf16(cfg, layout, safetensors_dir, resolved.model_name)
    bf16_gb = bf16_path.stat().st_size / 1024**3
    budget_gb = round(bf16_gb * cfg.budget / 100, 3)
    _log(layout, "B", f"B. budget = {cfg.budget}% × {bf16_gb:.2f} GB = {budget_gb:.2f} GB")

    # C: probe
    probe_path = stage_c_probe(cfg, layout, safetensors_dir, imatrix_corpus,
                                resolved.model_name)

    # D: imatrix (or use --imatrix override)
    if imatrix_override:
        imatrix_path = _resolve_imatrix_override(cfg, layout, imatrix_override)
        _log(layout, "D", f"D. using --imatrix override: {imatrix_path}")
    else:
        imatrix_path = stage_d_imatrix(cfg, layout, bf16_path, imatrix_corpus)

    # E–H
    costs_path = stage_e_costs(cfg, layout, bf16_path, imatrix_path)
    bridge_path = stage_f_bridge(cfg, layout, probe_path)
    perf_file = find_perf_file(layout, resolved.model_name)
    recipe_path = stage_g_allocate(cfg, layout, bridge_path, costs_path,
                                    bf16_path, perf_file, budget_gb,
                                    resolved.model_name)
    final_gguf = stage_h_quantize(cfg, layout, bf16_path, recipe_path,
                                   imatrix_path, resolved.model_name)

    # I
    ppl = stage_i_eval(cfg, layout, final_gguf, ppl_corpus)

    # Cleanup
    cleanup(layout, resolved, purge_ctx, purge)

    print()
    print("=" * 70)
    print(f"  ✓ DONE: {final_gguf}")
    print(f"    size: {final_gguf.stat().st_size/1024**3:.2f} GB")
    if ppl is not None:
        print(f"    PPL:  {ppl:.4f}")
    print(f"    logs: {layout.logs_dir}")
    print("=" * 70)
    return final_gguf


def _resolve_imatrix_override(cfg: Config, layout: Layout,
                              spec: str) -> Path:
    """--imatrix flag accepts a path or URL. URLs download to imatrix-cache."""
    if spec.startswith(("http://", "https://")):
        target = layout.imatrix_cache / spec.rsplit("/", 1)[-1]
        if not target.exists():
            print(f"  downloading imatrix {spec} → {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            import urllib.request
            with urllib.request.urlopen(spec) as resp, target.open("wb") as f:
                shutil.copyfileobj(resp, f)
        return target
    p = Path(spec).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"--imatrix path not found: {p}")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("input", metavar="INPUT",
                   help="HuggingFace id (e.g., 'unsloth/Qwen3.6-35B-A3B') "
                        "or on-disk safetensors directory")
    p.add_argument("--config", type=Path, default=None,
                   help="alternative config.toml path "
                        "(default: ~/.prismaquant-llama/config.toml)")
    p.add_argument("--libs", type=Path, default=None,
                   help="extra directory prepended to LD_LIBRARY_PATH for "
                        "all subprocess calls (use when llama libs aren't "
                        "on the system loader path)")
    p.add_argument("--base", type=Path, default=None,
                   help="working directory (default: from config)")
    p.add_argument("--path", type=Path, default=None,
                   help="llama.cpp binary directory (default: from config)")
    p.add_argument("--quants", default=None,
                   help="comma-separated allowed-quants list (default: from config)")
    p.add_argument("--budget", type=int, default=None,
                   help="target size as %% of BF16 (default: from config)")
    p.add_argument("--priority", default=None,
                   help="3-digit XYZ ratio (default: from config)")
    p.add_argument("--ppl-corpus", default=None,
                   help="PPL corpus path or URL (default: from config)")
    p.add_argument("--imatrix-corpus", default=None,
                   help="imatrix corpus path or URL (default: from config)")
    p.add_argument("--imatrix", default=None,
                   help="existing imatrix file path or URL (overrides "
                        "imatrix generation)")
    p.add_argument("--ppl-chunks", type=int, default=None,
                   help="chunks for llama-perplexity (default: from config)")
    p.add_argument("--imatrix-chunks", type=int, default=None,
                   help="chunks for llama-imatrix (default: from config)")
    p.add_argument("--convert-script", type=Path, default=None,
                   help="path to convert_hf_to_gguf.py (default: from config "
                        "or auto-discover from llama.cpp source tree)")
    p.add_argument("--purge", choices=("yes", "no"), default="yes",
                   help="clean up downloaded/generated artifacts at end "
                        "(default: yes; never deletes user-supplied on-disk inputs)")


def cfg_from_args(args) -> Config:
    """Apply CLI overrides to the loaded Config and return it."""
    cfg = load_config(args.config, libs=args.libs)
    if args.base is not None:
        cfg.base = Path(args.base).expanduser().resolve()
    if args.path is not None:
        cfg.path = Path(args.path).expanduser().resolve()
    if args.quants is not None:
        cfg.quants = [q.strip().upper() for q in args.quants.split(",") if q.strip()]
    if args.budget is not None:
        cfg.budget = args.budget
    if args.priority is not None:
        cfg.priority = args.priority
    if args.ppl_chunks is not None:
        cfg.ppl_chunks = args.ppl_chunks
    if args.imatrix_chunks is not None:
        cfg.imatrix_chunks = args.imatrix_chunks
    if args.ppl_corpus is not None:
        cfg.ppl_corpus = args.ppl_corpus
    if args.imatrix_corpus is not None:
        cfg.imatrix_corpus = args.imatrix_corpus
    if getattr(args, "convert_script", None) is not None:
        cfg.convert_script = Path(args.convert_script).expanduser().resolve()
    return cfg


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="prismaquant-llama run",
                                description="Run the full prismaquant pipeline")
    add_run_args(p)
    args = p.parse_args(argv)
    cfg = cfg_from_args(args)
    resolved = resolve_input(args.input, allow_gguf=False)
    try:
        run_pipeline(cfg, resolved, args.imatrix, args.purge)
        return 0
    except (SystemExit, FileNotFoundError, ValueError) as e:
        print(f"\nFAIL: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
