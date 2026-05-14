"""
End-to-end prismaquant pipeline.

Stages (safetensors-input only — Stage C requires safetensors for the
Hessian probe):

    A.  Download HF safetensors                 (HF id only)
    B.  Convert safetensors → BF16 GGUF         (HF id and on-disk safetensors)
    C.  Hessian probe via prismaquant.incremental_probe
    D.  imatrix generation (llama-imatrix)
    E.  per-(tensor, format) MSE costs (llama-quantize-cost)
    F.  Bridge HF→GGUF tensor names (bundled bridge_probe_to_gguf.py)
    G.  Allocate formats per tensor (bundled allocator.py)
    F+. (optional) Pre-condition ≥4-bit BF16 tensors (see precondition.py)
    H.  Apply recipe (llama-quantize)
    I.  Final PPL eval (llama-perplexity)

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
from dataclasses import dataclass, replace as _dc_replace
from pathlib import Path
from typing import Optional

from .config import (Config, load_config, find_tool, subprocess_env,
                     resolve_corpus, LLAMA_TOOLS, VALID_PRECONDITION_MODES)
from .input_resolver import ResolvedInput, resolve as resolve_input
from .paths import Layout, wipe_model_artifacts


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
    """Stage B. Convert safetensors → reference GGUF (BF16 or F16, per
    cfg.reference_format). Idempotent. Function name retained for backward
    API compatibility; the on-disk filename and --outtype reflect the
    configured reference format."""
    ref_upper = cfg.reference_format.upper()  # "BF16" or "F16"
    out = layout.bf16_dir / f"{model_name}-{ref_upper}.gguf"
    if out.exists():
        _log(layout, "B", f"B. {ref_upper} GGUF cached at {out} (skip)")
        return out
    convert_script = _find_convert_script(cfg)
    _log(layout, "B", f"B. converting {safetensors_dir} → {out}")
    rc = _run([sys.executable, str(convert_script), str(safetensors_dir),
               "--outtype", cfg.reference_format, "--outfile", str(out)],
              layout.logs_dir / "stage-B.log",
              env=subprocess_env(cfg))
    if rc != 0 or not out.exists():
        raise SystemExit(f"FAIL: B convert_hf_to_gguf.py exit={rc}")
    _log(layout, "B", f"B. {ref_upper} GGUF: {out.stat().st_size/1024**3:.2f} GB")
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
    probe_cmd = [sys.executable, "-m", "prismaquant.incremental_probe",
                 "--model", str(safetensors_dir),
                 "--dataset", str(imatrix_corpus),
                 "--nsamples", "16", "--seqlen", "512",
                 "--device", "cpu", "--dtype", "bf16",
                 "--output", str(probe_path),
                 "--activation-cache-dir", str(layout.probe_dir / "act-cache"),
                 "--work-dir", str(layout.probe_dir / "work")]
    if cfg.fisher_output_mse:
        # h-detail blobs carry per-Linear g² / H_diag that the fisher sidecar
        # emitter needs. Adding --h-detail-dir changes the probe's content
        # cache key (h_detail_dir is in prismaquant's _CONTENT_META_KEYS), so
        # the first run after toggling fisher_output_mse will re-probe.
        h_detail_dir = layout.probe_dir / "h-detail"
        h_detail_dir.mkdir(parents=True, exist_ok=True)
        probe_cmd += ["--h-detail-dir", str(h_detail_dir)]
        _log(layout, "C", f"C. fisher_output_mse=true → h-detail-dir={h_detail_dir}")
    rc = _run(probe_cmd, layout.logs_dir / "stage-C.log",
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


def stage_e_pre_sidecar(cfg: Config, layout: Layout, probe_path: Path,
                         safetensors_dir: Path,
                         bf16_path: Optional[Path] = None) -> Optional[Path]:
    """Stage E-pre — emit per-tensor Fisher sidecars for llama-quantize-cost.

    Runs only when cfg.fisher_output_mse is set. Reads probe.pkl + act-cache
    + h-detail blobs (all produced by Stage C in fisher mode), maps HF
    Linear names to GGUF tensor names via the same bridge logic the bridge
    stage uses, and writes one `.bin` per GGUF target to `work/fisher-sidecar/`.

    Idempotent: re-running with an existing non-empty sidecar dir is fine
    (files are atomically renamed; absent ones get written). Returns the
    sidecar directory path, or None when fisher mode is off.
    """
    if not cfg.fisher_output_mse:
        return None
    sidecar_dir = layout.work / "fisher-sidecar"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    h_detail_dir = layout.probe_dir / "h-detail"
    n_hidden, n_nextn = _read_layer_counts(safetensors_dir)
    _log(layout, "E", f"E-pre. emitting fisher sidecars → {sidecar_dir}")
    emit_script = _bundled_script("emit_fisher_sidecar.py")
    cmd = [sys.executable, str(emit_script),
           "--probe", str(probe_path),
           "--output-dir", str(sidecar_dir)]
    if n_hidden is not None:
        cmd += ["--n-hidden-layers", str(n_hidden)]
    if n_nextn:
        cmd += ["--n-nextn-layers", str(n_nextn)]
    if h_detail_dir.exists():
        cmd += ["--h-detail-dir", str(h_detail_dir)]
    else:
        _log(layout, "E", "E-pre. WARN: no h-detail dir — fisher weights "
                          "fall back to uniform; column will reflect "
                          "unweighted per-row output MSE instead of true "
                          "Fisher weighting")
    if bf16_path is not None:
        # Validate sidecar in_features against GGUF ne[0]; suppress mismatched
        # targets at emit time so llama-quantize-cost doesn't WARN-spam later
        # and the dim-mismatch count is visible in stage-E-pre.log.
        cmd += ["--gguf", str(bf16_path)]
    rc = _run(cmd, layout.logs_dir / "stage-E-pre.log",
              env=subprocess_env(cfg))
    if rc != 0:
        raise SystemExit(f"FAIL: E-pre fisher sidecar emit exit={rc}")
    n_written = sum(1 for _ in sidecar_dir.glob("*.bin"))
    _log(layout, "E", f"E-pre. wrote {n_written} sidecars")
    return sidecar_dir


def _read_layer_counts(safetensors_dir: Path) -> tuple[Optional[int], int]:
    """Pluck (num_hidden_layers, num_nextn_predict_layers) from config.json.
    Mirrors stage_f_bridge's logic — kept local so emit-sidecar can run before
    bridge if a future stage rearranges things."""
    config_json = safetensors_dir / "config.json"
    if not config_json.exists():
        return (None, 0)
    try:
        with config_json.open() as f:
            hf_cfg = json.load(f)
    except Exception:
        return (None, 0)
    text_cfg = hf_cfg.get("text_config", hf_cfg)
    n_hidden = (text_cfg.get("num_hidden_layers")
                or hf_cfg.get("num_hidden_layers"))
    n_nextn = (text_cfg.get("num_nextn_predict_layers")
               or hf_cfg.get("num_nextn_predict_layers")
               or 0)
    return (int(n_hidden) if n_hidden is not None else None, int(n_nextn))


def stage_e_costs(cfg: Config, layout: Layout, bf16_path: Path,
                  imatrix_path: Path,
                  fisher_sidecar_dir: Optional[Path] = None) -> Path:
    """Stage E — per-(tensor, format) MSE measurement.

    Resume-safe: the output costs.csv is content-addressed under
    `_shared/costs-cache/` keyed by (bf16_sha, imatrix_sha, formats_hash).
    Existing cache hits skip the multi-hour measurement. Cache misses
    write to a `.tmp` path and atomically rename on rc=0, so a killed
    Stage E never leaves a partial CSV that a future run might mistake
    for a complete one.

    When `fisher_sidecar_dir` is provided, llama-quantize-cost emits a
    `fisher_output_mse` column. We fold that into the formats_hash so the
    cache namespace cleanly partitions fisher-on vs fisher-off runs and we
    never serve a fisher-off cache to a fisher-on caller.
    """
    bf16_sha = _file_sha256(bf16_path)
    imatrix_sha = _file_sha256(imatrix_path)
    import hashlib as _hashlib
    fmt_key = ",".join(cfg.quants)
    if fisher_sidecar_dir is not None:
        fmt_key = "fisher+" + fmt_key
    formats_hash = _hashlib.sha256(fmt_key.encode()).hexdigest()
    costs = layout.costs_cache_path(bf16_sha, imatrix_sha, formats_hash)

    if costs.exists():
        _log(layout, "E", f"E. costs.csv cached at {costs} (skip)")
        return costs

    cost_bin = find_tool(cfg, "llama-quantize-cost")
    exemplars = _auto_pick_exemplars(bf16_path)
    top_level = _discover_top_level_weights(bf16_path)
    mtp_blocks = _discover_mtp_blocks(bf16_path)
    # MTP blocks (any blk.<N>.nextn.* tensor present) carry MTP-unique
    # tensors like nextn.eh_proj that have no body sibling for shape-based
    # propagation. Include them as direct-measure targets so the allocator
    # has costs for the MTP recipe entries (and so the MTP override can
    # actually flip them to BF16 in the recipe).
    layer_idxs = sorted(set(exemplars) | set(mtp_blocks))
    layer_alt = "|".join(str(L) for L in layer_idxs)
    top_alt = "|".join(top_level)
    include_regex = rf"^({top_alt}|blk\.({layer_alt}))\."
    _log(layout, "E", f"E. measuring per-(tensor, format) MSE → {costs}")
    _log(layout, "E", f"E. top-level weights: {top_level}")
    _log(layout, "E", f"E. exemplar layers: {exemplars}")
    if mtp_blocks:
        _log(layout, "E", f"E. MTP blocks (direct-measure): {mtp_blocks}")

    # Atomic write: quantize-cost emits to .tmp, rename on success.
    costs.parent.mkdir(parents=True, exist_ok=True)
    tmp = costs.with_suffix(".csv.tmp")
    if tmp.exists():
        # Stale tmp from a prior killed run — unlink so we don't accidentally
        # treat partial output as cached.
        tmp.unlink()

    cmd = [str(cost_bin),
           "--model", str(bf16_path),
           "--types", ",".join(cfg.quants),
           "--imatrix", str(imatrix_path),
           "--include-regex", include_regex,
           "--output", str(tmp)]
    if fisher_sidecar_dir is not None:
        cmd += ["--fisher-sidecar", str(fisher_sidecar_dir)]
        _log(layout, "E", f"E. fisher sidecar dir: {fisher_sidecar_dir}")
    rc = _run(cmd, layout.logs_dir / "stage-E.log",
              env=subprocess_env(cfg))
    if rc != 0 or not tmp.exists():
        # Clean up partial tmp before failing so a re-run starts fresh.
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"FAIL: E llama-quantize-cost exit={rc}")
    tmp.rename(costs)
    _log(layout, "E", f"E. costs.csv rows: {sum(1 for _ in costs.open())}")
    return costs


def stage_f_bridge(cfg: Config, layout: Layout, probe_path: Path,
                   safetensors_dir: Path) -> tuple[Path, Optional[Path]]:
    """Stage F — bridge HF→GGUF tensor names. Returns (bridge_json, mtp_tensors_json|None).

    Reads `num_hidden_layers` and `num_nextn_predict_layers` from the model's
    HF config.json so the bridge can map mtp.* probe entries to their
    GGUF block indices. When the model declares zero MTP layers the
    bridge still runs without those flags and the mtp_tensors_json
    return value is None.
    """
    bridge = layout.work / "bridge.json"
    mtp_tensors = layout.work / "mtp-tensors.json"
    if bridge.exists():
        _log(layout, "F", f"F. bridge.json cached (skip)")
        return bridge, (mtp_tensors if mtp_tensors.exists() else None)
    bridge_script = _bundled_script("bridge_probe_to_gguf.py")
    _log(layout, "F", f"F. bridging probe → {bridge}")
    cmd = [sys.executable, str(bridge_script),
           "--probe", str(probe_path),
           "--output", str(bridge),
           "--aggregate", "sum",
           "--unmapped-out", str(layout.work / "bridge-unmapped.json")]
    config_json = safetensors_dir / "config.json"
    n_hidden: Optional[int] = None
    n_nextn = 0
    if config_json.exists():
        with open(config_json) as f:
            hf_cfg = json.load(f)
        text_cfg = hf_cfg.get("text_config", hf_cfg)
        n_hidden = text_cfg.get("num_hidden_layers") or hf_cfg.get("num_hidden_layers")
        # Tolerate `null` / missing — Qwen/Qwen3.5-4B's upstream config has
        # no num_nextn_predict_layers despite shipping mtp.* weights. The
        # bridge auto-detects n_nextn from probe contents in that case.
        n_nextn = (text_cfg.get("num_nextn_predict_layers")
                   or hf_cfg.get("num_nextn_predict_layers")
                   or 0)
    if n_hidden is not None:
        cmd += ["--n-hidden-layers", str(n_hidden),
                "--mtp-tensors-out", str(mtp_tensors)]
        if n_nextn:
            cmd += ["--n-nextn-layers", str(n_nextn)]
        _log(layout, "F",
             f"F. MTP-aware bridge: n_hidden_layers={n_hidden}, "
             f"n_nextn_layers={n_nextn or 'auto'} (bridge no-ops if probe "
             f"has no mtp.* entries)")
    rc = _run(cmd, layout.logs_dir / "stage-F.log", env=subprocess_env(cfg))
    if rc != 0 or not bridge.exists():
        raise SystemExit(f"FAIL: F bridge exit={rc}")
    # The sidecar may exist but be empty (e.g. text-only models); only return
    # it when it has at least one MTP tensor name.
    if mtp_tensors.exists():
        try:
            mtp_list = json.loads(mtp_tensors.read_text())
        except Exception:
            mtp_list = []
        if mtp_list:
            return bridge, mtp_tensors
    return bridge, None


def stage_g_allocate(cfg: Config, layout: Layout,
                     bridge_path: Path, costs_path: Path,
                     bf16_path: Path, perf_file: Optional[Path],
                     budget_gb: float, model_name: str,
                     mtp_tensors_path: Optional[Path] = None) -> Path:
    """Stage G — multi-choice knapsack allocation."""
    # Fisher-on recipes get a `-fisher` filename suffix so fisher and weight-MSE
    # runs cleanly partition the recipe cache (matches Stage E's costs-cache
    # `fisher+` prefix scheme from S1; PRISMAQUANT_FISHER_OUTPUT_MSE_ALLOCATOR
    # is set below to make the bundled allocator switch scoring modes).
    suffix = "-fisher" if cfg.fisher_output_mse else ""
    recipe = layout.recipes_dir / f"recipe-PQ{cfg.budget}-{cfg.priority}{suffix}.json"
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
    cmd = [sys.executable, str(allocator),
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
    if mtp_tensors_path is not None and cfg.mtp_format:
        cmd += ["--mtp-tensors", str(mtp_tensors_path),
                "--mtp-format", cfg.mtp_format]
        _log(layout, "G",
             f"G. mtp override: pin {mtp_tensors_path.name} → {cfg.mtp_format}")
    g_env = subprocess_env(cfg)
    if cfg.fisher_output_mse:
        # Mirror upstream prismaquant: env gate, not CLI flag. Keeps the bundled
        # allocator's CLI surface unchanged and matches the upstream toggle name.
        g_env["PRISMAQUANT_FISHER_OUTPUT_MSE_ALLOCATOR"] = "1"
    rc = _run(cmd, layout.logs_dir / "stage-G.log", env=g_env)
    if rc != 0 or not recipe.exists():
        raise SystemExit(f"FAIL: G allocator exit={rc}")
    return recipe


def _mark_pareto(results: list[dict]) -> None:
    """In-place: set ``is_pareto`` on each Stage-K candidate.

    Non-dominated in (size_gb, ppl): no other candidate has both
    ``size_gb <= self.size_gb`` and ``ppl <= self.ppl`` with strict
    inequality on at least one axis. Ties (identical size+ppl) all
    stay on the frontier.
    """
    for i, r in enumerate(results):
        dominated = False
        for j, q in enumerate(results):
            if i == j:
                continue
            if (q["size_gb"] <= r["size_gb"] and q["ppl"] <= r["ppl"]
                    and (q["size_gb"] < r["size_gb"] or q["ppl"] < r["ppl"])):
                dominated = True
                break
        r["is_pareto"] = not dominated


def stage_k_validate(cfg: Config, layout: Layout,
                     bridge_path: Path, costs_path: Path,
                     bf16_path: Path, imatrix_path: Path,
                     perf_file: Optional[Path],
                     budget_gb: float, ppl_corpus: Path,
                     baseline_recipe: Path,
                     model_name: str,
                     mtp_tensors_path: Optional[Path] = None) -> Path:
    """Stage K — KL/PPL-validated frontier picker.

    Sweeps the allocator over `cfg.kl_priorities` at `budget_gb`, quantizes
    each candidate, runs a short PPL pass (cfg.kl_ppl_chunks), then returns
    the lowest-PPL recipe. `baseline_recipe` is the Stage-G output for the
    user-requested priority; it gets included in the sweep so the user can
    always opt out by setting cfg.priority in kl_priorities.

    Designed as a real-quantize validator because upstream prismaquant's
    `validate_assignments_kl` only knows serving formats (NVFP4/MXFP8/etc.)
    and can't simulate K-quants; ground-truth quantize+PPL is the only
    correct measure for our pipeline.
    """
    suffix = "-fisher" if cfg.fisher_output_mse else ""
    work = layout.work / "stage-k"
    work.mkdir(parents=True, exist_ok=True)
    summary_path = work / f"summary-PQ{cfg.budget}{suffix}.json"

    # Build the sweep set: baseline priority + each priority in cfg.kl_priorities.
    # De-dup by priority string (preserve user order, baseline first).
    sweep_priorities: list[str] = []
    seen: set[str] = set()
    for p in [cfg.priority] + list(cfg.kl_priorities):
        if p not in seen:
            sweep_priorities.append(p)
            seen.add(p)

    perp_bin = find_tool(cfg, "llama-perplexity")
    quantize_bin = find_tool(cfg, "llama-quantize")
    results: list[dict] = []
    # Recipe-SHA → first result entry that produced it. When the allocator
    # resolves two priorities to byte-identical recipes (common on wide
    # priority sweeps when the budget pins the same tensor assignments),
    # the second occurrence reuses the first's quantize + PPL artifacts
    # and is recorded as a `duplicate_of` candidate in the summary.
    seen_recipes: dict[str, dict] = {}

    for p in sweep_priorities:
        recipe_path = (layout.recipes_dir
                       / f"recipe-PQ{cfg.budget}-{p}{suffix}.json")
        if p == cfg.priority and baseline_recipe.exists():
            recipe_path = baseline_recipe
        elif not recipe_path.exists():
            _log(layout, "K", f"K. allocator @ priority={p}")
            allocator = _bundled_script("allocator.py")
            pinned = layout.work / "pinned.json"
            if not pinned.exists():
                pinned.write_text(json.dumps({
                    "output.weight": "Q6_K",
                    "token_embd.weight": "Q8_0",
                }, indent=2))
            cmd = [sys.executable, str(allocator),
                   "--bridge", str(bridge_path),
                   "--costs", str(costs_path),
                   "--budget-gb", str(budget_gb),
                   "--budget-band-gb", "0.25",
                   "--pinned", str(pinned),
                   "--priority", p,
                   "--recipe-out", str(recipe_path),
                   "--allow-types", ",".join(cfg.quants),
                   "--gguf", str(bf16_path),
                   "--propagate-from-exemplars",
                   "--exemplar-layers", "0,3",
                   "--floor-bpw", json.dumps(
                       {r"^blk\..*\.attn_(q|k|v|qkv|gate|output)\.weight$":
                        4.0})]
            if perf_file is not None and perf_file.exists():
                cmd += ["--tps", str(perf_file)]
            if mtp_tensors_path is not None and cfg.mtp_format:
                cmd += ["--mtp-tensors", str(mtp_tensors_path),
                        "--mtp-format", cfg.mtp_format]
            g_env = subprocess_env(cfg)
            if cfg.fisher_output_mse:
                g_env["PRISMAQUANT_FISHER_OUTPUT_MSE_ALLOCATOR"] = "1"
            rc = _run(cmd, layout.logs_dir / f"stage-K-alloc-{p}.log",
                      env=g_env)
            if rc != 0 or not recipe_path.exists():
                raise SystemExit(f"FAIL: K allocator @ priority={p} exit={rc}")

        recipe_sha = hashlib.sha256(recipe_path.read_bytes()).hexdigest()
        prior = seen_recipes.get(recipe_sha)
        if prior is not None:
            _log(layout, "K",
                 f"K. priority={p} recipe matches priority={prior['priority']} "
                 f"(sha {recipe_sha[:12]}); reusing PPL={prior['ppl']:.4f} "
                 f"size={prior['size_gb']:.2f} GB (skip quantize + PPL)")
            results.append({
                "priority": p,
                "recipe": str(recipe_path),
                "candidate_gguf": prior["candidate_gguf"],
                "ppl": prior["ppl"],
                "size_gb": prior["size_gb"],
                "recipe_sha": recipe_sha,
                "duplicate_of": prior["priority"],
            })
            continue

        recipe_txt = _h_materialize_recipe_txt(recipe_path)
        cand_gguf = work / f"candidate-PQ{cfg.budget}-{p}{suffix}.gguf"

        if not cand_gguf.exists():
            _log(layout, "K", f"K. quantize candidate @ priority={p}")
            rc = _run([str(quantize_bin),
                       "--imatrix", str(imatrix_path),
                       "--tensor-type-file", str(recipe_txt),
                       str(bf16_path), str(cand_gguf), "Q4_K"],
                      layout.logs_dir / f"stage-K-quant-{p}.log",
                      env=subprocess_env(cfg))
            if rc != 0 or not cand_gguf.exists():
                raise SystemExit(f"FAIL: K quantize @ priority={p} exit={rc}")

        ppl_log = layout.logs_dir / f"stage-K-ppl-{p}.log"
        if ppl_log.exists():
            _log(layout, "K", f"K. ppl cached @ priority={p}")
        else:
            _log(layout, "K",
                 f"K. perplexity @ priority={p} chunks={cfg.kl_ppl_chunks}")
            rc = _run([str(perp_bin), "-m", str(cand_gguf), "-f",
                       str(ppl_corpus),
                       "-c", "4096", "-b", "2048",
                       "-ctk", "f16", "-ctv", "f16", "-fa", "on",
                       "-ngl", "99", "--chunks", str(cfg.kl_ppl_chunks),
                       "--no-mmap"],
                      ppl_log, env=subprocess_env(cfg))
            if rc != 0:
                _log(layout, "K",
                     f"K. WARN: perplexity rc={rc} for priority={p}")
        import re as _re
        m = _re.search(r"Final estimate:\s*PPL\s*=\s*([\d.]+)",
                       ppl_log.read_text())
        if not m:
            _log(layout, "K",
                 f"K. WARN: no Final estimate in {ppl_log.name}; "
                 f"excluding priority={p}")
            continue
        ppl = float(m.group(1))
        size_gb = cand_gguf.stat().st_size / 1024**3
        _log(layout, "K",
             f"K. priority={p}  PPL={ppl:.4f}  size={size_gb:.2f} GB")
        entry = {
            "priority": p,
            "recipe": str(recipe_path),
            "candidate_gguf": str(cand_gguf),
            "ppl": ppl,
            "size_gb": size_gb,
            "recipe_sha": recipe_sha,
        }
        results.append(entry)
        seen_recipes[recipe_sha] = entry

    if not results:
        raise SystemExit("FAIL: K no candidates produced a Final estimate; "
                         "cannot pick a frontier winner")

    _mark_pareto(results)
    winner = min(results, key=lambda r: r["ppl"])
    _log(layout, "K",
         f"K. winner: priority={winner['priority']}  "
         f"PPL={winner['ppl']:.4f}  size={winner['size_gb']:.2f} GB")
    frontier = [r["priority"] for r in results if r["is_pareto"]]
    _log(layout, "K",
         f"K. pareto frontier ({len(frontier)}/{len(results)}): "
         f"{', '.join(frontier)}")

    # Write validated-frontier recipe at a stable name so Stage H picks it up.
    validated = (layout.recipes_dir
                 / f"recipe-PQ{cfg.budget}-{cfg.priority}{suffix}"
                 f"-validated.json")
    src = Path(winner["recipe"])
    validated.write_text(src.read_text())
    src_txt = src.with_suffix(".txt")
    if src_txt.exists():
        validated.with_suffix(".txt").write_text(src_txt.read_text())

    summary_path.write_text(json.dumps({
        "schema_version": 2,
        "budget_gb": budget_gb,
        "user_priority": cfg.priority,
        "winner_priority": winner["priority"],
        "winner_ppl": winner["ppl"],
        "winner_size_gb": winner["size_gb"],
        "candidates": results,
    }, indent=2))
    _log(layout, "K", f"K. summary written → {summary_path}")
    return validated


_H_CACHEKEY_VERSION = 1


def _h_input_descriptor(path: Path) -> dict:
    """Stat-cheap + SHA-correct descriptor for a Stage-H input file.
    Stat sig lets us short-circuit re-hashing when the file is byte-identical
    on the same inode; SHA is the source of truth on stat-mismatch."""
    st = path.stat()
    return {
        "path": str(path),
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "sha256": _file_sha256(path),
    }


def _h_descriptor_matches(cached: dict, path: Path) -> bool:
    """Compare a previously-stored descriptor against current path state.
    Returns True iff content is provably equivalent (SHA match)."""
    try:
        st = path.stat()
    except FileNotFoundError:
        return False
    if cached.get("size") == st.st_size and cached.get("mtime_ns") == st.st_mtime_ns:
        return True  # stat-identity is strong enough to skip re-hash
    return cached.get("sha256") == _file_sha256(path)


def _h_materialize_recipe_txt(recipe_path: Path) -> Path:
    """Stage G normally emits both recipe.json and recipe.txt; this dead-code
    path only fires if a caller hands us a JSON without its txt sibling."""
    recipe_txt = recipe_path.with_suffix(".txt")
    if recipe_txt.exists():
        return recipe_txt
    data = json.loads(recipe_path.read_text())
    assignments = data.get("recipe") or data.get("assignments") or data
    with recipe_txt.open("w") as f:
        for tensor, fmt in assignments.items():
            if isinstance(fmt, dict):
                fmt = fmt.get("type") or fmt.get("format")
            f.write(f"{tensor}={fmt}\n")
    return recipe_txt


def stage_h_quantize(cfg: Config, layout: Layout, bf16_path: Path,
                     recipe_path: Path, imatrix_path: Path,
                     model_name: str) -> Path:
    """Stage H — apply allocation.

    Caches by input identity (bf16, imatrix, recipe-txt) via a sidecar
    cachekey.json, so a precondition rerun against the same output filename
    correctly invalidates a stale baseline GGUF written by an earlier run.
    """
    out = layout.gguf_output_path(model_name, cfg.budget, cfg.priority)
    recipe_txt = _h_materialize_recipe_txt(recipe_path)
    sidecar = out.parent / (out.name + ".cachekey.json")

    if out.exists() and sidecar.exists():
        try:
            stored = json.loads(sidecar.read_text())
        except (json.JSONDecodeError, OSError):
            stored = None
        if (stored
                and stored.get("version") == _H_CACHEKEY_VERSION
                and stored.get("out_default_quant") == "Q4_K"
                and _h_descriptor_matches(stored.get("bf16", {}), bf16_path)
                and _h_descriptor_matches(stored.get("imatrix", {}), imatrix_path)
                and _h_descriptor_matches(stored.get("recipe_txt", {}), recipe_txt)):
            _log(layout, "H", f"H. GGUF cached at {out} (skip, cachekey match)")
            return out
        _log(layout, "H",
             f"H. cachekey mismatch at {sidecar.name}; rebuilding {out.name}")
    elif out.exists():
        _log(layout, "H",
             f"H. no cachekey sidecar for existing {out.name}; rebuilding")

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

    sidecar.write_text(json.dumps({
        "version": _H_CACHEKEY_VERSION,
        "out_default_quant": "Q4_K",
        "bf16": _h_input_descriptor(bf16_path),
        "imatrix": _h_input_descriptor(imatrix_path),
        "recipe_txt": _h_input_descriptor(recipe_txt),
    }, indent=2))
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


def _discover_mtp_blocks(bf16_path: Path) -> list[int]:
    """Find block indices that contain MTP-unique tensors (`blk.<N>.nextn.*`)
    in the BF16 GGUF. The presence of nextn.eh_proj / nextn.shared_head_norm
    / etc. is the unambiguous MTP-block signature; the standard attn_/ffn_
    tensors at those indices share shape with body softmax-attn layers and
    would not be distinguished by `detect_layer_types` alone."""
    try:
        sys.path.insert(0, str(_bundled_script("allocator.py").parent))
        from allocator import read_gguf_tensor_meta
        meta = read_gguf_tensor_meta(str(bf16_path))
        mtp_idxs: set[int] = set()
        for tn in meta:
            if not tn.startswith("blk."):
                continue
            parts = tn.split(".", 3)
            if len(parts) < 4 or not parts[1].isdigit():
                continue
            if parts[2] == "nextn":
                mtp_idxs.add(int(parts[1]))
        return sorted(mtp_idxs)
    except Exception as e:
        print(f"  WARN: MTP-block discovery failed ({e}); skipping")
        return []


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

def _is_complete_for_quants(perf_json_path: Path, quants: list[str]) -> bool:
    """Check whether a perf JSON has complete measurements for every format
    in `quants`. Used by `run --calibrate` to decide whether the cached
    model.json is sufficient or the calibration step needs to run."""
    if not perf_json_path.exists():
        return False
    try:
        data = json.loads(perf_json_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    for q in quants:
        entry = data.get(q)
        if not isinstance(entry, dict):
            return False
        if entry.get("error"):
            return False
        # Same completeness check as calibration._is_complete:
        if not all(entry.get(k) is not None for k in ("bpw", "ppl", "pp", "tg")):
            return False
    return True


def _compose_run_with_calibrate(run_est: "Estimate",
                                cal_est: "Estimate") -> "Estimate":
    """Combine run + calibrate-model estimates into a single pre-flight
    block when --calibrate is given. Times sum (calibrate runs first,
    then pipeline); peak is the max (they don't overlap on disk)."""
    return Estimate(
        label="run --calibrate",
        model_name=run_est.model_name,
        n_formats=run_est.n_formats,
        source_size=run_est.source_size,
        bf16_size=run_est.bf16_size,
        final_size=run_est.final_size,
        peak=max(run_est.peak or 0, cal_est.peak or 0) or None,
        free_disk=run_est.free_disk,
        time_low_min=run_est.time_low_min + cal_est.time_low_min,
        time_high_min=run_est.time_high_min + cal_est.time_high_min,
        source_cached=run_est.source_cached,
        bf16_cached=run_est.bf16_cached,
        budget=run_est.budget,
        priority=run_est.priority,
        chunks_imatrix=run_est.chunks_imatrix,
        chunks_ppl=run_est.chunks_ppl,
    )


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

# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight estimate + accept/abort prompt
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Estimate:
    """Disk + time estimate, formatted for the accept/abort prompt."""
    label: str                              # "run" or "calibrate system" etc.
    model_name: str
    n_formats: int
    source_size: Optional[int] = None       # bytes; None if unknown
    bf16_size: Optional[int] = None
    final_size: Optional[int] = None
    peak: Optional[int] = None
    free_disk: int = 0
    time_low_min: int = 0
    time_high_min: int = 0
    source_cached: bool = False
    bf16_cached: bool = False
    budget: Optional[int] = None
    priority: Optional[str] = None
    chunks_imatrix: int = 0
    chunks_ppl: int = 0


def _free_disk_at(path: Path) -> int:
    p = Path(path).expanduser().resolve()
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free


def _hf_weight_size(hf_id: str) -> Optional[int]:
    """Query HF for the total weight bytes. Returns None on any failure."""
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo_id=hf_id, files_metadata=True)
    except Exception:
        return None
    weight_exts = (".safetensors", ".bin", ".gguf", ".pt", ".pth", ".npz", ".h5")
    return sum((s.size or 0) for s in (info.siblings or [])
               if s.rfilename.endswith(weight_exts) and (s.size or 0) > 0)


def _source_size_and_cached(resolved: "ResolvedInput",
                             layout: Layout) -> tuple[Optional[int], bool]:
    if resolved.kind == "hf":
        cache_dir = layout.hf_cache / resolved.model_name
        if (cache_dir / ".download.complete").exists():
            size = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
            return size, True
        return _hf_weight_size(resolved.hf_id), False
    if resolved.kind == "safetensors_dir":
        size = sum(f.stat().st_size
                   for f in resolved.safetensors_dir.glob("*.safetensors"))
        return size, True
    if resolved.kind == "gguf_local":
        return resolved.gguf_path.stat().st_size, True
    if resolved.kind == "gguf_url":
        return None, False
    return None, False


def _estimate_run_minutes(bf16_gb: float) -> tuple[int, int]:
    """Rough wall-time range for the full A→I pipeline. Empirical scaling
    from observed runs: 4B → ~15-35 min, 9B → ~30-60 min, 35B → ~90-180 min."""
    if bf16_gb <= 0:
        return 30, 240
    return max(15, int(8 + bf16_gb * 1.2)), max(45, int(20 + bf16_gb * 4))


def _estimate_calibrate_minutes(bf16_gb: float, n_formats: int) -> tuple[int, int]:
    """Rough range. Per format: 5-15 min on a 4B model, scales with size."""
    if bf16_gb <= 0:
        return max(60, n_formats * 5), max(180, n_formats * 15)
    low = 5 + n_formats * max(3, int(3 + bf16_gb * 0.3))
    high = 10 + n_formats * max(8, int(6 + bf16_gb * 0.8))
    return low, high


def estimate_run(cfg: Config, resolved: "ResolvedInput",
                 layout: Layout) -> Estimate:
    src_size, src_cached = _source_size_and_cached(resolved, layout)
    bf16_path = layout.bf16_dir / f"{resolved.model_name}-BF16.gguf"
    bf16_cached = bf16_path.exists()
    bf16_size = (bf16_path.stat().st_size if bf16_cached
                 else int((src_size or 0) * 1.05) or None)
    final_size = (int(bf16_size * cfg.budget / 100) if bf16_size else None)
    peak = ((0 if src_cached else (src_size or 0))
            + (0 if bf16_cached else (bf16_size or 0))
            + (final_size or 0) + 50 * 1024**2)  # +50MB for probe/imatrix/costs/recipe
    free = _free_disk_at(cfg.base)
    bf16_gb = (bf16_size / 1024**3) if bf16_size else 0
    low, high = _estimate_run_minutes(bf16_gb)
    return Estimate(
        label="run", model_name=resolved.model_name, n_formats=len(cfg.quants),
        source_size=src_size, bf16_size=bf16_size, final_size=final_size,
        peak=peak if peak > 0 else None, free_disk=free,
        time_low_min=low, time_high_min=high,
        source_cached=src_cached, bf16_cached=bf16_cached,
        budget=cfg.budget, priority=cfg.priority,
        chunks_imatrix=cfg.imatrix_chunks, chunks_ppl=cfg.ppl_chunks,
    )


def estimate_calibrate(cfg: Config, mode: str, resolved: "ResolvedInput",
                       layout: Layout) -> Estimate:
    src_size, src_cached = _source_size_and_cached(resolved, layout)
    bf16_path = layout.bf16_dir / f"{resolved.model_name}-BF16.gguf"
    bf16_cached = bf16_path.exists()
    if resolved.kind == "gguf_local":
        bf16_size = src_size  # input GGUF IS the reference
    elif resolved.kind == "gguf_url":
        bf16_size = src_size  # whatever the URL points at
    else:
        bf16_size = (bf16_path.stat().st_size if bf16_cached
                     else int((src_size or 0) * 1.05) or None)
    # Calibration scratch: only ONE per-format GGUF on disk at a time.
    # Q8_0 (~8.5 bpw) is the largest non-16bpw format; ~bf16/1.9.
    biggest_scratch = (bf16_size // 2) if bf16_size else 0
    needs_convert = resolved.kind in ("hf", "safetensors_dir")
    peak = ((0 if src_cached else (src_size or 0))
            + ((0 if bf16_cached else (bf16_size or 0)) if needs_convert else 0)
            + biggest_scratch + 50 * 1024**2)
    free = _free_disk_at(cfg.base)
    bf16_gb = (bf16_size / 1024**3) if bf16_size else 0
    low, high = _estimate_calibrate_minutes(bf16_gb, len(cfg.quants))
    return Estimate(
        label=f"calibrate {mode}", model_name=resolved.model_name,
        n_formats=len(cfg.quants),
        source_size=src_size, bf16_size=bf16_size, final_size=None,
        peak=peak if peak > 0 else None, free_disk=free,
        time_low_min=low, time_high_min=high,
        source_cached=src_cached, bf16_cached=bf16_cached,
        chunks_imatrix=cfg.imatrix_chunks, chunks_ppl=cfg.ppl_chunks,
    )


def _fmt_bytes(b: Optional[int]) -> str:
    if b is None:
        return "(unknown — HF API unreachable?)"
    if b == 0:
        return "negligible"
    if b >= 1024**3:
        return f"~{b/1024**3:.1f} GB"
    if b >= 1024**2:
        return f"~{b/1024**2:.0f} MB"
    if b >= 1024:
        return f"~{b/1024:.0f} KB"
    return f"~{b} B"


def _fmt_time_range(low_m: int, high_m: int) -> str:
    def fmt(m: int) -> str:
        if m < 60:
            return f"{m}m"
        h = m / 60
        return f"{h:.1f}h" if h < 10 else f"{int(h)}h"
    return f"{fmt(low_m)}–{fmt(high_m)}"


def confirm_or_abort(est: Estimate, assume_yes: bool) -> bool:
    """Print the estimate + prompt y/N. Returns True to proceed."""
    print()
    print("┌─ prismaquant-llama " + est.label
          + " ─ " + est.model_name + " " + "─" * 4)
    if est.budget is not None:
        print(f"│ Budget:    {est.budget}% of BF16    Priority: {est.priority}")
    print(f"│ Formats:   {est.n_formats}    "
          f"imatrix_chunks: {est.chunks_imatrix}    "
          f"ppl_chunks: {est.chunks_ppl}")
    print(f"│")
    print(f"│ Estimated disk:")
    src_tag = " (cached)" if est.source_cached else ""
    bf16_tag = " (cached)" if est.bf16_cached else ""
    print(f"│   source / HF download:   {_fmt_bytes(est.source_size)}{src_tag}")
    print(f"│   BF16 GGUF:              {_fmt_bytes(est.bf16_size)}{bf16_tag}")
    if est.final_size is not None:
        print(f"│   final PQ GGUF:          {_fmt_bytes(est.final_size)}")
    if est.peak is not None:
        print(f"│   peak during pipeline:   {_fmt_bytes(est.peak)}")
    print(f"│")
    print(f"│ Estimated wall time:    {_fmt_time_range(est.time_low_min, est.time_high_min)}  (rough)")
    print(f"│")
    print(f"│ Free disk:              {_fmt_bytes(est.free_disk)}")
    if (est.peak is not None and est.free_disk
            and est.peak > est.free_disk):
        short = est.peak - est.free_disk
        print(f"│   ⚠ INSUFFICIENT — short by {_fmt_bytes(short)} at peak")
    print("└" + "─" * 60)

    if assume_yes:
        print("[--yes] proceeding without prompt")
        return True
    if not sys.stdin.isatty():
        print("ERROR: non-interactive shell; pass --yes to proceed.",
              file=sys.stderr)
        return False
    try:
        response = input("Proceed? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    if response in ("y", "yes"):
        return True
    print("aborted.")
    return False


@dataclass
class PurgeContext:
    """Tracks artifacts to clean up at end of run if --purge yes."""
    user_owns_model: bool                      # True if input was on-disk
    ppl_was_downloaded: bool
    imatrix_was_downloaded: bool
    ppl_corpus_local: Path
    imatrix_corpus_local: Path
    costs_path: Optional[Path] = None          # tracked output of Stage E


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
        if ctx.costs_path is not None:
            _rm(ctx.costs_path, "costs")

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
                 imatrix_override: Optional[str], purge: str,
                 assume_yes: bool = False,
                 do_calibrate: bool = False,
                 calibrate_chunks: Optional[int] = None,
                 force: bool = False) -> Optional[Path]:
    """Execute the full A→I pipeline. Returns path to the final GGUF, or
    None if the user aborted at the pre-flight prompt.

    If `do_calibrate=True`, runs `calibrate model` against this input first
    (writing calibration/models/<name>.json). Skipped automatically if a
    complete model calibration already exists for the configured `quants`.
    """
    layout = Layout.for_run(base=cfg.base, model_name=resolved.model_name)

    # --force: nuke prior artifacts for this model BEFORE the pre-flight
    # estimate + the calibration-needed check, so they reflect a real
    # from-scratch run. Always wipe the model.json since `run` consults it
    # via find_perf_file even without --calibrate.
    if force:
        deleted = wipe_model_artifacts(layout, resolved.model_name,
                                        cfg.reference_format,
                                        calibration_mode="model")
        if deleted:
            print(f"[force] removed {len(deleted)} prior artifact"
                  f"{'s' if len(deleted) != 1 else ''}:")
            for d in deleted:
                print(f"  - {d}")
        else:
            print("[force] no prior artifacts to remove")

    # Decide whether the calibration step is actually needed.
    cal_cfg: Optional[Config] = None
    needs_calibration = False
    if do_calibrate:
        cal_path = layout.model_calibration_path(resolved.model_name)
        if _is_complete_for_quants(cal_path, cfg.quants):
            print(f"[run] --calibrate: model.json already complete at "
                  f"{cal_path}; skipping calibration step")
        else:
            needs_calibration = True
            cal_cfg = _dc_replace(cfg)
            if calibrate_chunks is not None:
                cal_cfg.ppl_chunks = calibrate_chunks

    # Pre-flight estimate + accept/abort. Layout is constructed but not
    # mkdir'd yet, so an aborted invocation leaves no traces.
    est = estimate_run(cfg, resolved, layout)
    if needs_calibration and cal_cfg is not None:
        cal_est = estimate_calibrate(cal_cfg, "model", resolved, layout)
        est = _compose_run_with_calibrate(est, cal_est)
    if not confirm_or_abort(est, assume_yes=assume_yes):
        return None

    layout.make()

    # Run model calibration first if requested + needed. Force purge="no"
    # because the BF16 GGUF + imatrix that calibrate produces are
    # immediately consumed by the pipeline that follows; purging would
    # delete them and force re-generation. The pipeline's own --purge
    # setting handles end-of-run cleanup correctly.
    if needs_calibration and cal_cfg is not None:
        from .calibration import run_calibrate
        cal_result = run_calibrate(cal_cfg, "model", resolved, purge="no",
                                    imatrix_override=imatrix_override,
                                    assume_yes=True)
        if cal_result is None:
            print("[run] calibration step failed/aborted; halting pipeline.",
                  file=sys.stderr)
            return None
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

    # E-pre: fisher sidecar emission (no-op unless cfg.fisher_output_mse).
    fisher_sidecar_dir = stage_e_pre_sidecar(cfg, layout, probe_path,
                                              safetensors_dir,
                                              bf16_path=bf16_path)

    # E–H
    costs_path = stage_e_costs(cfg, layout, bf16_path, imatrix_path,
                                fisher_sidecar_dir=fisher_sidecar_dir)
    purge_ctx.costs_path = costs_path     # for --purge yes cleanup of shared cache
    bridge_path, mtp_tensors_path = stage_f_bridge(cfg, layout, probe_path,
                                                    safetensors_dir)
    perf_file = find_perf_file(layout, resolved.model_name)
    recipe_path = stage_g_allocate(cfg, layout, bridge_path, costs_path,
                                    bf16_path, perf_file, budget_gb,
                                    resolved.model_name,
                                    mtp_tensors_path=mtp_tensors_path)

    # K: optional KL/PPL-validated frontier picker. Sweeps allocator priorities,
    # quantizes each candidate, runs short PPL, picks lowest-loss recipe.
    if cfg.kl_validate:
        recipe_path = stage_k_validate(
            cfg, layout, bridge_path, costs_path,
            bf16_path, imatrix_path, perf_file,
            budget_gb, ppl_corpus, recipe_path,
            resolved.model_name,
            mtp_tensors_path=mtp_tensors_path)

    # F+: pre-condition BF16 weights of ≥4-bit recipe entries. Lazy import to
    # avoid the precondition module's `from .pipeline_runner import _log`-
    # style coupling concerns; the module is self-contained otherwise.
    from .precondition import stage_fp_precondition
    bf16_for_quantize, pc_manifest = stage_fp_precondition(
        cfg, layout, bf16_path, recipe_path, costs_path, resolved.model_name)

    # D': if F+ actually mutated weights, re-run imatrix on pc-bf16 so the
    # k-quants / IQ-quants in Stage H see fresh activation second-moments
    # against the rescaled weights. stage_d_imatrix's SHA-keyed cache auto-
    # keys off pc-bf16's content, so no special invalidation logic needed.
    # Skipped when the user provided an explicit --imatrix (they've opted out
    # of pipeline-managed imatrix); also skipped when no folds ran.
    if (bf16_for_quantize != bf16_path
            and pc_manifest is not None
            and not imatrix_override):
        summary = json.loads(Path(pc_manifest).read_text()).get("summary", {})
        if summary.get("n_folds", 0) > 0:
            _log(layout, "D'",
                 f"D'. F+ applied {summary['n_folds']} fold(s); "
                 f"re-running imatrix on pc-bf16")
            imatrix_path = stage_d_imatrix(cfg, layout, bf16_for_quantize,
                                             imatrix_corpus)

    final_gguf = stage_h_quantize(cfg, layout, bf16_for_quantize, recipe_path,
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
    p.add_argument("--yes", "-y", action="store_true",
                   help="skip the pre-flight disk + time confirmation prompt "
                        "(required for non-interactive / scripted use)")
    p.add_argument("--calibrate", action="store_true",
                   help="run `calibrate model` against this input before the "
                        "pipeline, writing calibration/models/<name>.json. "
                        "The allocator then prefers it over system.json. "
                        "Skipped automatically when a complete model.json "
                        "already exists for the configured quants list.")
    p.add_argument("--calibrate-chunks", type=int, default=None,
                   help="ppl_chunks override for the --calibrate step only "
                        "(does NOT affect Stage I's final eval, which uses "
                        "--ppl-chunks). Useful for high-fidelity calibration "
                        "while keeping the run's eval fast.")
    p.add_argument("--force", action="store_true",
                   help="nuclear: before running, delete every artifact "
                        "associated with this model under {base} (BF16 GGUF, "
                        "HF cache, probe, SHA-keyed imatrix-cache and "
                        "costs-cache entries for this model, final GGUFs, and "
                        "the model calibration JSON) so the entire pipeline "
                        "is recomputed from scratch. Use after a llama.cpp "
                        "bugfix that invalidates prior outputs.")
    p.add_argument("--precondition",
                   choices=tuple(sorted(VALID_PRECONDITION_MODES)),
                   default=None,
                   help="Stage F+ pre-conditioning method stack. "
                        "'off' skips F+; 'awq' runs AWQ proper-fold (P2). "
                        "Default: from [precondition].mode in config "
                        "(typically 'off').")
    p.add_argument("--precondition-bpw-floor", type=float, default=None,
                   help="Skip F+ for any tensor whose chosen-format bpw is "
                        "below this threshold — the literal 'disable below 4 "
                        "bits' cutoff. Default: from [precondition].bpw_floor "
                        "in config (typically 4.0).")


def cfg_from_args(args) -> Config:
    """Apply CLI overrides to the loaded Config and return it."""
    cfg = load_config(args.config, libs=args.libs)
    if args.base is not None:
        cfg.base = Path(args.base).expanduser().resolve()
    if args.path is not None:
        cfg.path = Path(args.path).expanduser().resolve()
    if args.quants is not None:
        cfg.quants = [q.strip().upper() for q in args.quants.split(",") if q.strip()]
    if getattr(args, "budget", None) is not None:
        cfg.budget = args.budget
    if getattr(args, "priority", None) is not None:
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
    if getattr(args, "precondition", None) is not None:
        cfg.precondition_mode = args.precondition
    if getattr(args, "precondition_bpw_floor", None) is not None:
        cfg.precondition_bpw_floor = args.precondition_bpw_floor
    return cfg


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="prismaquant-llama run",
                                description="Run the full prismaquant pipeline")
    add_run_args(p)
    args = p.parse_args(argv)
    cfg = cfg_from_args(args)
    resolved = resolve_input(args.input, allow_gguf=False)
    try:
        result = run_pipeline(cfg, resolved, args.imatrix, args.purge,
                               assume_yes=args.yes,
                               do_calibrate=args.calibrate,
                               calibrate_chunks=args.calibrate_chunks,
                               force=args.force)
        return 0 if result is not None else 1
    except (SystemExit, FileNotFoundError, ValueError) as e:
        print(f"\nFAIL: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
